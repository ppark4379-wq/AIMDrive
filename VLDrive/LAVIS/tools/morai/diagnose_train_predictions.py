#!/usr/bin/env python3
"""Bounded, train-only counterfactual inference; never updates model weights."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from offline_train_replay import ROOT, read, write, bootstrap, sha256, np, torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError('Use a new diagnostic output inside LAVIS')
    output.mkdir(parents=True)
    settings, provenance = read(run/'resolved_config.json'), read(run/'provenance.json')
    backend = bootstrap(ROOT.parent)
    from lavis.datasets.datasets.morai.expert_vldrive_dataset import MoraiExpertVLDriveDataset
    from lavis.runners.vldrive_trainer import seed_all, source_hashes, restore_model_checkpoint
    from lavis.runners.morai_trainer import batch, autocast
    from lavis.models.vldrive.training_model import VLDriveTrainingConfig, VLDriveTrainingModel
    torch.set_num_threads(settings.get('cpu_threads', 4))
    seed_all(settings['seed'])
    dataset = MoraiExpertVLDriveDataset(settings['dataset'], cache_root=settings['feature_cache'],
        split_file=ROOT/settings['split_file'], instructions=settings['instructions'], split='train',
        sequence_length=settings['sequence_length'])
    if (backend != provenance['backend'] or source_hashes(ROOT) != provenance['source']
            or dataset.describe() != provenance['datasets']['train']):
        raise ValueError('Provenance mismatch')
    rows = read(args.replay/'recorded_pose_comparison.json')
    by_frame = {r['frame_id']: r for r in rows}
    window_by_frame = {int(dataset.entries[end-1]['frame_id']): i
                       for i, (begin, end) in enumerate(dataset.windows)}
    model = VLDriveTrainingModel(VLDriveTrainingConfig(**settings['model']), device='cuda')
    state = restore_model_checkpoint(run/'model-last.pt', model, provenance=provenance)
    assert state['updates'] == 9540
    del state
    model.eval()
    results = []

    def language_prediction(connected, instruction):
        with autocast(model):
            pred = model.language_model([instruction], connected['language_tokens'],
                connected['selected_valid_mask'], predict_per_frame=True)
        return pred.predicted_waypoints[0, -1].float().reshape(5, 2).cpu().numpy()

    with torch.inference_mode():
        for frame in (280, 990, 991, 995, 1000, 1180, 2865, 3320):
            item = batch(dataset[window_by_frame[frame]], 'cuda')
            feature = item['model_inputs']['visual_feature']
            correct = item['model_inputs']['instruction'][0]
            gt = item['training_targets']['future_trajectory'][0, -1].cpu().numpy()
            with autocast(model):
                connected = model.connector(feature.float(), compute_reconstruction=(frame == 2865))
            predictions = {}
            instructions = {'correct': correct}
            if frame in (280, 1180, 2865, 3320):
                instructions.update(generic='Proceed along this route.',
                    right='Proceed ahead and make a right turn.',
                    straight='Please proceed straight at the next intersection.',
                    left='Proceed ahead and make a left turn.')
            for name, instruction in instructions.items():
                value = language_prediction(connected, instruction)
                predictions[name] = dict(instruction=instruction, prediction=value.tolist(),
                    ade_m=float(np.linalg.norm(value-gt, axis=1).mean()),
                    fde_m=float(np.linalg.norm(value[-1]-gt[-1])))
            if frame in (280, 1180, 2865, 3320):
                for name, modified in (
                    ('repeat_current_history', feature[:, -1:].expand_as(feature).contiguous()),
                    ('reverse_past_history', torch.cat([feature[:, :-1].flip(1), feature[:, -1:]], 1)),
                ):
                    with autocast(model):
                        changed = model.connector(modified.float(), compute_reconstruction=False)
                    value = language_prediction(changed, correct)
                    predictions[name] = dict(prediction=value.tolist(),
                        ade_m=float(np.linalg.norm(value-gt, axis=1).mean()),
                        fde_m=float(np.linalg.norm(value[-1]-gt[-1])))
            row = dict(frame_id=frame, actual_future=gt.tolist(), probes=predictions)
            if frame in by_frame:
                row['saved_prediction_max_diff_m'] = float(np.max(np.abs(
                    np.asarray(predictions['correct']['prediction'])-np.asarray(by_frame[frame]['prediction']))))
            if frame == 2865:
                normalized = connected['normalized_features'].reshape(-1, 106, 256)
                error = (normalized.float()-connected['reconstructed_features'].float()).square().sum(-1)
                row['reconstruction_unmasked_error_per_token'] = dict(
                    min=float(error.min()), median=float(error.median()), max=float(error.max()),
                    direct_pruning_gradient_per_token=14/106,
                    observed_reconstruction_loss=float(connected['reconstruction_loss']),
                    selected_count=connected['selected_count_per_frame'][0].cpu().tolist())
            results.append(row)
            write(output/'counterfactuals.json', results)
            print(json.dumps(row), flush=True)

    # The exact pretrained GRU head receives the unpooled final five cached tokens.
    # No raw pixels/labels are replaced. target_point is a pretraining-head input,
    # not an input to the trained downstream model above.
    from tools.morai.cache_expert_vision_tokens import make_dataset_and_loader
    from timm.models.memfuser import GRUWaypointsPredictor
    raw_args = SimpleNamespace(dataset_root=Path(settings['dataset']),
        split_file=ROOT/settings['split_file'], batch_size=1, workers=0)
    raw_dataset, _ = make_dataset_and_loader(raw_args, 'train')
    lookup = {key: i for i, key in enumerate(raw_dataset.route_frames)}
    checkpoint_path = Path(settings['model']['visual_checkpoint'])
    if sha256(checkpoint_path) != dataset.checkpoint_sha256:
        raise ValueError('Vision checkpoint mismatch')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    head = GRUWaypointsPredictor(256).cuda().eval()
    prefix = 'waypoints_generator.'
    head.load_state_dict({k[len(prefix):]: v for k, v in checkpoint['state_dict'].items()
                         if k.startswith(prefix)}, strict=True)
    del checkpoint
    head_results = []
    with torch.inference_mode():
        for row in results:
            frame = row['frame_id']
            begin, end = dataset.windows[window_by_frame[frame]]
            entry = dataset.entries[end-1]
            sensors, _ = raw_dataset[lookup[(entry['route_id'], frame)]]
            target = sensors['target_point'][None].cuda()
            feature = dataset._array(entry['visual_feature'], (106, 256), 'visual')[None, 101:106].cuda()
            gt = np.asarray(row['actual_future'])
            probes = {}
            for name, value in (('recorded_target_point', target), ('zero_target_point', torch.zeros_like(target)),
                                ('opposite_target_point', -target)):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = head(feature, value)[0].float().cpu().numpy()
                probes[name] = dict(prediction=pred.tolist(), ade_m=float(np.linalg.norm(pred-gt, axis=1).mean()),
                    fde_m=float(np.linalg.norm(pred[-1]-gt[-1])))
            result = dict(frame_id=frame, target_point=target[0].cpu().tolist(), probes=probes)
            head_results.append(result)
            write(output/'vision_head_probes.json', head_results)
            print(json.dumps(dict(vision_head=result)), flush=True)
    write(output/'metadata.json', dict(split='train_only', updates=9540,
        checkpoint=str(run/'model-last.pt'), checkpoint_sha256=sha256(run/'model-last.pt'),
        note='Counterfactuals are diagnostic input interventions, not valid new driving GT; no optimizer updates.'))
    print('DIAGNOSIS_COMPLETE', flush=True)


def probe_lateral_target(run, output):
    """Isolate head route-lateral dependence without changing forward distance."""
    run, output = Path(run).resolve(), Path(output).resolve()
    if ROOT not in output.parents:
        raise ValueError('Output must stay inside LAVIS')
    settings = read(run/'resolved_config.json')
    bootstrap(ROOT.parent)
    from timm.models.memfuser import GRUWaypointsPredictor
    torch.set_num_threads(4)
    checkpoint = torch.load(settings['model']['visual_checkpoint'], map_location='cpu', weights_only=False)
    head = GRUWaypointsPredictor(256).cuda().eval()
    prefix = 'waypoints_generator.'
    head.load_state_dict({k[len(prefix):]: v for k, v in checkpoint['state_dict'].items()
                         if k.startswith(prefix)}, strict=True)
    del checkpoint
    cache = Path(settings['feature_cache'])
    original = read(output/'vision_head_probes.json')
    rows = []
    with torch.inference_mode():
        for row in original:
            frame = row['frame_id']
            matches = list((cache/'features').glob('*/%04d.npy' % frame))
            if len(matches) != 1:
                raise ValueError('Ambiguous cached frame')
            feature = torch.from_numpy(np.load(matches[0]).astype(np.float32))[None, 101:106].cuda()
            target = torch.tensor([row['target_point']], device='cuda')
            probes = {}
            for name, factor in [('recorded', [1, 1]), ('zero_lateral', [0, 1]), ('mirror_lateral', [-1, 1])]:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = head(feature, target*torch.tensor([factor], device='cuda'))[0].float()
                probes[name] = pred.cpu().tolist()
            rows.append(dict(frame_id=frame, target_point=row['target_point'], probes=probes))
    write(output/'vision_lateral_target_probes.json', rows)
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
