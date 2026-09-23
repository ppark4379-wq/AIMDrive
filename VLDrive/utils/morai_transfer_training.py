"""Separate MORAI multitask fine-tuning of the full LMDrive visual checkpoint.

Does not change the ImageNet/fresh paper-recipe trainer. Batch 1, all admitted
frames, constant LR, v3_lav masks and losses. No connector/LLM training.
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from models.vision_encoder import create_vision_encoder
from datasets.morai_sensor_record_dataset import (
    PathResolver, load_publications, SensorRecordDataset, digest)
from datasets.morai_encoder_contract import sensor_record_encoder_contract
from inference.morai_route import navigation_recipe
from .morai_transfer_objective import multitask_loss, DETECTION_LOSS_POLICIES
from .morai_transfer_metrics import evaluate, to_device

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('rgb_backbone.', 'lidar_backbone.', 'encoder.', 'decoder.',
          'traffic_pred_head.', 'traffic_light_pred_head.', 'waypoints_generator.')
DEFAULT_CHECKPOINT_HASH = '348c2885113c69f9c41813f6c1c02edc4c69568c6241aa4a4d4bae830cdd9023'


def add_arguments(parser):
    parser.add_argument('--config', type=Path, help='PC-local JSON: manifest_index, checkpoint, checkpoint_sha256, output_root, path_mappings')
    parser.add_argument('--manifest-index', type=Path, action='append', help='Repeat to combine published campaigns without changing their splits')
    parser.add_argument('--manifest-campaign', type=Path, action='append', default=[],
                        help='Append the latest sealed training publication below this campaign')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--velocity-source', choices=('gps','zero'), default='gps')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--preflight', action='store_true', help='Validate data and one real forward/backward; no optimizer step')
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--diagnostic-limit', type=int, default=0, help='Explicit bounded diagnostic only; 0 uses all frames')
    parser.add_argument('--initialization', choices=('lmdrive_base','morai_checkpoint'), default='lmdrive_base')
    parser.add_argument('--trainable-scope', choices=('full','object_head'), default='full')
    parser.add_argument('--frame-sampling', choices=('all','object_balanced'), default='all')
    parser.add_argument('--object-negative-ratio', type=float, default=1.0,
                        help='Verified empty object-valid frames per object-positive frame')
    parser.add_argument('--detection-loss-policy', choices=DETECTION_LOSS_POLICIES,
                        default='lmdrive_lav_v1')
    parser.add_argument('--skip-baseline', action='store_true',
                        help='Skip unchanged starting-checkpoint validation for a quick refinement run')
    return parser


def resolve_args(args):
    path = args.config or ROOT/'configs/morai_transfer.local.json'
    if args.config and not path.is_file():
        raise FileNotFoundError('Missing configuration: '+str(path))
    cfg = json.loads(path.read_text()) if path.exists() else {}
    if args.manifest_index is None:
        args.manifest_index = list(cfg.get('manifest_indices', [])) or ([cfg['manifest_index']] if cfg.get('manifest_index') else [])
        campaigns = list(cfg.get('manifest_campaigns', []))
    else:
        campaigns = []
    campaigns += list(args.manifest_campaign)
    for campaign in campaigns:
        campaign = Path(campaign).expanduser()
        candidates = sorted(campaign.glob('validation_*/training_dataset*/manifest_index.json'),
                            key=lambda p: (p.stat().st_mtime_ns, str(p)))
        if not candidates:
            raise ValueError('Additional campaign is not ready; finish 03 validation first: '+str(campaign))
        # Never silently omit the requested new campaign. Full seal and
        # source checks below reject an incomplete or failed publication.
        args.manifest_index.append(candidates[-1])
    args.checkpoint = args.checkpoint or (Path(cfg['checkpoint']) if cfg.get('checkpoint') else None)
    args.checkpoint_sha256 = args.checkpoint_sha256 or cfg.get('checkpoint_sha256', DEFAULT_CHECKPOINT_HASH)
    if not args.manifest_index or args.checkpoint is None:
        raise ValueError('Set --manifest-index and --checkpoint or copy configs/morai_transfer.example.json to morai_transfer.local.json')
    args.manifest_index = [Path(p).expanduser().resolve() for p in args.manifest_index]
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if args.output is None:
        output_root = Path(cfg.get('output_root', ROOT/'outputs/checkpoints/vision_encoder_morai'))
        args.output = output_root / ('morai_transfer_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000))
    args.output = args.output.expanduser().resolve()
    if (min(args.epochs,args.threads,args.log_interval)<1 or args.workers<0 or args.diagnostic_limit<0
            or not math.isfinite(args.learning_rate) or args.learning_rate<=0
            or not math.isfinite(args.object_negative_ratio) or args.object_negative_ratio<=0):
        raise ValueError('Invalid epoch/thread/worker/LR/limit settings')
    return PathResolver(cfg.get('path_mappings', []))


def make_model(checkpoint, expected_sha256, device, initialization='lmdrive_base'):
    if digest(checkpoint) != expected_sha256:
        raise ValueError('Initial checkpoint SHA256 mismatch')
    # Only trusted tensor data; no old timm pickle imports needed.
    doc = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
    if not isinstance(doc,dict) or 'state_dict' not in doc:
        raise ValueError('Expected LMDrive state_dict checkpoint')
    if initialization == 'lmdrive_base' and ('pretraining' in doc or 'morai_initialization' in doc):
        raise ValueError('This path starts from the base LMDrive checkpoint, not an earlier MORAI run')
    model = create_vision_encoder({'name':'memfuser_baseline_e1d3'}, pretrained_backbone=False)
    model.load_state_dict(doc['state_dict'], strict=True)
    model.requires_grad_(True)
    model.stop_sign_head.requires_grad_(False)
    return model.to(device)


def configure_trainable_scope(model, scope):
    if scope == 'full':
        model.requires_grad_(True)
        model.stop_sign_head.requires_grad_(False)
        return tuple(GROUPS)
    model.requires_grad_(False)
    model.traffic_pred_head.requires_grad_(True)
    return ('traffic_pred_head.',)


def selected_training_indices(dataset, args):
    indices = list(range(len(dataset)))
    selection = {'policy': args.frame_sampling, 'available_frames': len(indices)}
    if args.frame_sampling == 'object_balanced':
        positives = [i for i, row in enumerate(dataset.rows)
                     if row[1]['object_valid'] and row[1]['object_positive']]
        negatives = [i for i, row in enumerate(dataset.rows)
                     if row[1]['object_valid'] and not row[1]['object_positive']]
        if not positives or not negatives:
            raise ValueError('Object-balanced sampling requires positive and verified empty frames')
        rng = random.Random(args.seed + 7301)
        rng.shuffle(negatives)
        negative_count = min(len(negatives), max(1, int(round(len(positives) * args.object_negative_ratio))))
        indices = positives + negatives[:negative_count]
        rng.shuffle(indices)
        selection.update(object_positive_frames=len(positives),
                         selected_verified_empty_frames=negative_count,
                         object_negative_ratio=args.object_negative_ratio)
    if args.diagnostic_limit:
        indices = indices[:min(len(indices), args.diagnostic_limit)]
    selection['selected_frames'] = len(indices)
    return indices, selection


def frozen_state_digest(model, excluded_prefixes=()):
    value=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        if any(name.startswith(prefix) for prefix in excluded_prefixes):
            continue
        array=tensor.detach().cpu().contiguous().numpy()
        value.update(name.encode());value.update(str(array.dtype).encode())
        value.update(str(array.shape).encode());value.update(array.tobytes())
    return value.hexdigest()


def train_mode(model, scope='full'):
    if scope == 'object_head':
        model.eval()
        model.traffic_pred_head.train()
        return
    model.train()
    # v3_lav physical batch 1: keep pretrained BN running stats, train affine.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def gradients(model):
    result={prefix:False for prefix in GROUPS}
    for name,p in model.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                raise ValueError('Nonfinite gradient: '+name)
            for prefix in GROUPS:
                if name.startswith(prefix) and p.grad.count_nonzero():
                    result[prefix]=True
    return result


def write_json(path, data):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False,default=str)+'\n')
    temp.replace(path)


def save(path, payload):
    temp=path.with_suffix(path.suffix+'.partial')
    torch.save(payload,str(temp))
    temp.replace(path)


def object_selection_score(metrics):
    """Rank validation checkpoints by occupancy AP, then best achievable F1."""
    values = []
    for key in ('detection_cell_average_precision', 'detection_cell_best_f1'):
        value = metrics.get(key)
        values.append(float(value) if value is not None and math.isfinite(float(value)) else -math.inf)
    return tuple(values)


def run(args):
    resolver=resolve_args(args)
    torch.set_num_threads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this terminal; --device cpu --preflight permits checks without training')
    print('[CHECK] validating publication, evidence seals and GPS history',flush=True)
    index, manifests=load_publications(args.manifest_index,resolver)
    datasets={key:SensorRecordDataset(rows,resolver,args.velocity_source) for key,rows in manifests.items()}
    encoder_contract = sensor_record_encoder_contract(
        (row[2].config for dataset in datasets.values() for row in dataset.rows), args.velocity_source)
    encoder_contract['route_source_sha256s'] = sorted({
        doc['navigation_contract']['global_path_sha256']
        for rows in manifests.values() for _, doc in rows})
    encoder_contract['navigation_recipes'] = []
    for rows in manifests.values():
        for _, doc in rows:
            recipe = navigation_recipe(doc['navigation_contract'])
            if recipe not in encoder_contract['navigation_recipes']:
                encoder_contract['navigation_recipes'].append(recipe)
    coverage={key:{'usable':dict(ds.coverage),'excluded':dict(ds.excluded)} for key,ds in datasets.items()}
    print(json.dumps(coverage),flush=True)
    for split,ds in datasets.items():
        if not all(ds.coverage[k]>0 for k in ('object_positive','traffic_light_0','traffic_light_1')):
            raise ValueError('Insufficient multitask coverage in '+split)
    model=make_model(args.checkpoint,args.checkpoint_sha256,args.device,args.initialization)
    expected_gradient_groups=configure_trainable_scope(model,args.trainable_scope)
    frozen_before=(frozen_state_digest(model,('traffic_pred_head.',))
                   if args.trainable_scope=='object_head' else None)
    train_indices,sampling=selected_training_indices(datasets['train'],args)
    args.output.mkdir(parents=True,exist_ok=False)
    implementation={str(p.relative_to(ROOT)):digest(p) for p in
                    [Path(__file__),ROOT/'models/vision_encoder.py',ROOT/'models/pointpillars.py',
                     ROOT/'models/resnet.py',ROOT/'models/transformer.py']+
                    sorted((ROOT/'datasets').glob('morai_transfer*.py'))+
                    [ROOT/'datasets/morai_sensor_record_dataset.py',ROOT/'datasets/morai_gps_speed.py',
                     ROOT/'datasets/morai_encoder_contract.py',
                     ROOT/'inference/morai_route.py',
                     ROOT/'utils/morai_transfer_objective.py',ROOT/'utils/morai_transfer_metrics.py']}
    provenance=dict(schema='morai_visual_transfer_v1',args=vars(args),coverage=coverage,
        publications=index['publications'],
        initial_checkpoint_sha256=args.checkpoint_sha256,
        implementation_sha256=implementation,
        initialization=args.initialization,
        encoder_contract=encoder_contract,
        trainable_scope=args.trainable_scope,
        loss=('0.5*detection+0.5*waypoint_L1+0.05*object_speed_L1+0.1*light_CE; '
              'detection_policy='+args.detection_loss_policy),
        batch_size=1,frame_sampling=sampling,augmentation='none',
        velocity_source=args.velocity_source,velocity_contract='causal consecutive GPS fixes, dt<=0.25s; no Ego fallback' if args.velocity_source=='gps' else 'legacy constant zero',
        waypoint_source='sealed future Ego positions at 0.5/1/1.5/2/2.5s',
        target_point='official global path 25m lookahead; prediction head only',
        batchnorm='running statistics frozen; affine parameters trainable',
        stop_sign='frozen_without_verified_GT',physical_simultaneity_verified=False,
        validation_scope=index.get('validation_scope'),diagnostic=bool(args.preflight or args.diagnostic_limit))
    provenance=json.loads(json.dumps(provenance,default=str))
    write_json(args.output/'config.json',provenance)
    if args.preflight:
        # Cover every input episode, including newly added publications.
        indices=[]
        episodes={}
        for i,row in enumerate(datasets['train'].rows):
            episodes.setdefault(id(row[2]),[]).append(i)
        for members in episodes.values():
            indices.append(members[0])
            for test in (lambda e:e['object_positive'],lambda e:e['traffic_light']==0,lambda e:e['traffic_light']==1):
                found=next((i for i in members if test(datasets['train'].rows[i][1])),None)
                if found is not None:indices.append(found)
        seen={k:False for k in GROUPS}
        for i in dict.fromkeys(indices):
            print('[PREFLIGHT] forward/backward train sample '+str(i),flush=True)
            model.zero_grad(set_to_none=True);train_mode(model,args.trainable_scope)
            inputs,targets=to_device(next(iter(DataLoader(Subset(datasets['train'],[i]),batch_size=1))),args.device)
            loss,_=multitask_loss(model(inputs),targets,detection_loss_policy=args.detection_loss_policy)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            seen={k:seen[k] or v for k,v in gradients(model).items()}
        if not all(seen[k] for k in expected_gradient_groups):
            raise ValueError('Disconnected trainable groups: '+str(seen))
        checked_val=set()
        model.eval()
        for i,row in enumerate(datasets['val'].rows):
            if id(row[2]) in checked_val:continue
            checked_val.add(id(row[2]))
            val_inputs,_=to_device(next(iter(DataLoader(Subset(datasets['val'],[i]),batch_size=1))),args.device)
            with torch.no_grad():predictions=model(val_inputs)
            if not all(torch.isfinite(t).all() for t in predictions):raise ValueError('Invalid validation episode output')
        model.eval();model.return_feature=True
        with torch.no_grad():features=model(inputs)
        if tuple(features.shape)!=(1,106,256) or not torch.isfinite(features).all():raise ValueError('Invalid feature output')
        report=dict(preflight_passed=True,optimizer_updates=0,gradient_groups=seen,feature_shape=list(features.shape),coverage=coverage,
                    checked_train_samples=len(set(indices)),checked_validation_episodes=len(checked_val))
        write_json(args.output/'preflight.json',report);print(json.dumps(report),flush=True)
        return report
    val_indices=list(range(min(len(datasets['val']),args.diagnostic_limit))) if args.diagnostic_limit else list(range(len(datasets['val'])))
    loaders={'train':DataLoader(Subset(datasets['train'],train_indices),batch_size=1,shuffle=True,num_workers=args.workers),
             'val':DataLoader(Subset(datasets['val'],val_indices),batch_size=1,shuffle=False,num_workers=args.workers)}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=0.01)
    best_checkpoint=args.output/'checkpoint-best-object.pth.tar'
    best_object=None

    def consider_object_checkpoint(metrics, epoch, steps):
        nonlocal best_object
        score=object_selection_score(metrics)
        if best_object is not None and score <= tuple(best_object['selection_score']):
            return False
        payload=dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,steps=steps,
                     morai_encoder_contract=encoder_contract,
                     pretraining=provenance,metrics=metrics,token_shape=[106,256],
                     feature_model_name='memfuser_baseline_e1d3_return_feature')
        save(best_checkpoint,payload)
        best_object=dict(epoch=epoch,steps=steps,selection_score=list(score),
                         detection_cell_average_precision=metrics.get('detection_cell_average_precision'),
                         detection_cell_best_f1=metrics.get('detection_cell_best_f1'),
                         detection_cell_best_threshold=metrics.get('detection_cell_best_threshold'),
                         checkpoint=str(best_checkpoint))
        write_json(args.output/'best_object.json',best_object)
        return True

    before=None
    if not args.skip_baseline:
        before=evaluate(model,loaders['val'],args.device,progress_phase='baseline',detection_loss_policy=args.detection_loss_policy)
        write_json(args.output/'baseline.json',before)
        consider_object_checkpoint(before,0,0)
    observed={k:False for k in GROUPS};step=0
    with (args.output/'train.jsonl').open('x') as log:
        for epoch in range(args.epochs):
            train_mode(model,args.trainable_scope)
            for batch in loaders['train']:
                inputs,targets=to_device(batch,args.device);optimizer.zero_grad(set_to_none=True)
                loss,tasks=multitask_loss(model(inputs),targets,detection_loss_policy=args.detection_loss_policy)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward()
                if not all(observed[k] for k in expected_gradient_groups):
                    observed={k:observed[k] or v for k,v in gradients(model).items()}
                norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();step+=1
                row=dict(epoch=epoch+1,step=step,loss=float(loss.detach()),grad_norm=float(norm),**{k:float(v.detach()) for k,v in tasks.items()})
                log.write(json.dumps(row)+'\n')
                if step%args.log_interval==0 or step==1:log.flush();print(json.dumps(row),flush=True)
            after=evaluate(model,loaders['val'],args.device,progress_phase='validation',detection_loss_policy=args.detection_loss_policy)
            payload=dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,steps=step,
                         morai_encoder_contract=encoder_contract,
                         pretraining=provenance,metrics=after,token_shape=[106,256],
                         feature_model_name='memfuser_baseline_e1d3_return_feature')
            save(args.output/'checkpoint-last.pth.tar',payload)
            write_json(args.output/('epoch_%03d.json'%(epoch+1)),after)
            promoted=consider_object_checkpoint(after,epoch+1,step)
            print(json.dumps(dict(event='object_checkpoint_selection',epoch=epoch+1,
                                  promoted=promoted,best=best_object)),flush=True)
    # Final selection reproduces v3_lav; expose all task metrics, not only waypoint loss.
    model.eval();model.return_feature=True
    with torch.no_grad():features=model(inputs)
    if tuple(features.shape)!=(1,106,256) or not torch.isfinite(features).all():raise ValueError('Invalid feature output')
    checkpoint=args.output/'checkpoint-last.pth.tar'
    if best_object is None:
        raise RuntimeError('No finite object validation metric was available for checkpoint selection')
    frozen_unchanged=(frozen_state_digest(model,('traffic_pred_head.',))==frozen_before
                      if frozen_before is not None else None)
    if frozen_unchanged is False:
        raise RuntimeError('Frozen shared encoder or waypoint/light parameters changed')
    report=dict(before=before,after=after,feature_shape=list(features.shape),nonzero_finite_gradient_groups=observed,
                checkpoint_sha256=digest(best_checkpoint),checkpoint_selection='best_validation_object_ap_then_f1',
                selected_checkpoint=str(best_checkpoint),best_object=best_object,
                last_checkpoint=str(checkpoint),last_checkpoint_sha256=digest(checkpoint),optimizer_updates=step,
                frame_sampling=sampling,trainable_scope=args.trainable_scope,
                detection_loss_policy=args.detection_loss_policy,
                frozen_non_object_state_unchanged=frozen_unchanged,
                velocity_source=args.velocity_source,validation_scope=index.get('validation_scope'),
                downstream_connected=False)
    write_json(args.output/'report.json',report)
    print(json.dumps(report,indent=2),flush=True)
    print('완료: '+str(args.output/'report.json'),flush=True)
    return report
