"""Opt-in route-permission head on an unchanged, frozen MORAI visual encoder.

Example (from VLDrive):
  python -m training.vision_encoder.recipes.route_permission --config CONFIG \
      --output NEW_DIRECTORY --mode preflight --device cpu

Training requires explicit --mode train. A separate head checkpoint is tied to
the exact base encoder SHA256; this never replaces the red/not-red head, a live
Expert input, or the concurrently running perception training recipe.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import Subset

from data.src.loaders.morai_dataloader import make_morai_loader
from data.src.loaders.morai_sensor_record_dataset import (
    PathResolver, SensorRecordDataset, digest, load_publications)
from data.src.loaders.morai_route_permission import RoutePermissionDataset
from models.ve.route_permission import RoutePermissionEncoder
from training.vision_encoder.losses.route_permission import route_permission_loss
from training.vision_encoder.utils.morai_route_permission_metrics import (
    RoutePermissionAccumulator, selection_decision)
from training.vision_encoder.utils.morai_runtime_state import nested_state_digest
from training.vision_encoder.utils.morai_transfer_metrics import to_device
from utils.common.runtime import deterministic_pointpillar_reductions
from utils.evaluation.morai_transfer_evaluation import _load_model


def write(path, data):
    pending = path.with_suffix(path.suffix + '.partial')
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                  allow_nan=False, default=str) + '\n')
    pending.replace(path)


def save(path, payload):
    pending = path.with_suffix(path.suffix + '.partial')
    torch.save(payload, str(pending))
    pending.replace(path)


def loader(dataset, indices, args, shuffle=False, seed=0):
    return make_morai_loader(Subset(dataset, indices), workers=args.workers,
                            shuffle=shuffle, generator=torch.Generator().manual_seed(seed),
                            persistent_workers=False)


def valid_indices(dataset):
    return [i for i, row in enumerate(dataset.metadata) if row['valid']]


def preflight_indices(dataset):
    """First admitted example of every movement/state/target, never rewrite GT."""
    seen = set()
    indices = []
    for index, row in enumerate(dataset.metadata):
        if not row['valid']:
            continue
        key = (row['movement'], row['raw_signal_bits'], row['target'])
        if key not in seen:
            seen.add(key)
            indices.append(index)
    return indices


def forward(model, batch, device):
    inputs, targets = to_device(batch, device)
    movement = inputs.pop('route_movement')
    result = model(inputs, movement)
    loss, terms = route_permission_loss(
        result['route_permission'], targets['route_permission'],
        targets['route_permission_valid'], movement)
    return result, loss, terms, inputs, targets, movement


def evaluate(model, dataset, indices, args, phase):
    model.eval()
    meter = RoutePermissionAccumulator()
    metadata = dataset.metadata
    loss_sum = 0.
    with torch.no_grad(), deterministic_pointpillar_reductions():
        for position, batch in enumerate(loader(dataset, indices, args)):
            result, loss, _, _, targets, _ = forward(model, batch, args.device)
            meter.update(result['route_permission'], targets['route_permission'],
                         [metadata[indices[position]]])
            loss_sum += float(loss)
            if (position + 1) % 200 == 0:
                print(json.dumps(dict(phase=phase, frames=position+1)), flush=True)
    metrics = meter.compute()
    metrics['loss'] = loss_sum / len(indices) if indices else None
    metrics['source_admitted_frames'] = len(dataset)
    metrics['source_excluded_permission_frames'] = sum(not row['valid'] for row in metadata)
    metrics['evaluated_permission_frames'] = len(indices)
    return metrics


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('preflight', 'train', 'evaluate'), default='preflight')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--head-checkpoint', type=Path)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    return parser


def run(args):
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    if (args.epochs < 1 or args.threads < 1 or args.workers < 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError('Invalid epoch/worker/thread/LR settings')
    if args.mode == 'evaluate' and args.head_checkpoint is None:
        raise ValueError('Evaluation requires a trained --head-checkpoint')
    if args.mode == 'train' and args.head_checkpoint is not None:
        raise ValueError('This bounded pilot starts a fresh head; no implicit optimizer resume')
    config = json.loads(args.config.read_text())
    checkpoint = args.checkpoint or Path(config['checkpoint'])
    expected_sha = args.checkpoint_sha256 or (None if args.checkpoint else config.get('checkpoint_sha256'))
    if not expected_sha or len(expected_sha) != 64:
        raise ValueError('An explicit exact base checkpoint SHA256 is required')
    if config.get('velocity_source', 'ego_topic') != 'ego_topic':
        raise ValueError('This pilot preserves the existing ego_topic input contract')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for preflight')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    resolver = PathResolver(config.get('path_mappings', []))
    index, manifests = load_publications(config['manifest_indices'], resolver)
    datasets = {name: RoutePermissionDataset(SensorRecordDataset(
        rows, resolver, velocity_source='ego_topic'), index['manifest_scenarios'])
        for name, rows in manifests.items()}
    selected = {name: valid_indices(dataset) for name, dataset in datasets.items()}
    coverage = {}
    for name, dataset in datasets.items():
        metadata = dataset.metadata
        states = Counter(str(metadata[i]['raw_signal_bits']) for i in selected[name])
        labels = Counter(str(metadata[i]['target']) for i in selected[name])
        if set(labels) != {'0', '1'}:
            raise ValueError('Both deny and allow supervision required: ' + name)
        coverage[name] = dict(admitted_frames=len(dataset), valid_permission=len(selected[name]),
                              excluded_permission=len(dataset)-len(selected[name]),
                              target_counts=dict(labels), raw_signal_counts=dict(states))
    print(json.dumps(dict(phase='permission_admission', coverage=coverage)), flush=True)
    base, payload, actual_sha = _load_model(resolver(checkpoint), expected_sha, args.device)
    contract = payload.get('morai_encoder_contract', {})
    if (contract.get('velocity_input') != 'causal_ego_topic_speed_mps_50ms'
            or any(dataset.base_dataset.map_basis != contract.get('map_basis')
                   for dataset in datasets.values())):
        raise ValueError('Base encoder and permission data input/map contract differ')
    del payload
    model = RoutePermissionEncoder(base, freeze_encoder=True).to(args.device)
    from models.ve.route_permission import build_route_permission_checkpoint, load_route_permission_checkpoint
    if args.head_checkpoint:
        model.permission_head = load_route_permission_checkpoint(
            torch.load(str(args.head_checkpoint), map_location='cpu', weights_only=True),
            actual_sha).to(args.device)
    initial_base_digest = nested_state_digest(base.state_dict())
    args.output.mkdir(parents=True)
    provenance = dict(schema='morai_route_permission_pilot_v1', args=vars(args),
        base_checkpoint=str(checkpoint), base_checkpoint_sha256=actual_sha,
        base_encoder_frozen=True, legacy_heads_preserved=True,
        new_head_inputs=['native_traffic_light_visual_token', 'planned_route_movement'],
        forbidden_model_inputs=['GT permission', 'raw_signal_bits', 'GT stopline distance', 'light ID'],
        publications=index['publications'], coverage=coverage,
        scope='Original held-out laps; same-map/route pilot. Check state/distance support; not deployment certification.',
        training_started=args.mode=='train', live_expert_connected=False)
    provenance = json.loads(json.dumps(provenance, default=str))
    write(args.output/'config.json', provenance)
    start = time.monotonic()
    if args.mode == 'preflight':
        checked = []
        model.train()
        with deterministic_pointpillar_reductions():
            for split, dataset in datasets.items():
                probes = preflight_indices(dataset)
                metadata_rows = dataset.metadata
                for position, batch in enumerate(loader(dataset, probes, args)):
                    model.zero_grad(set_to_none=True)
                    result, loss, _, inputs, _, _ = forward(model, batch, args.device)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite permission preflight loss')
                    loss.backward()
                    gradients = [p.grad for p in model.permission_head.parameters()]
                    if (not all(g is not None and torch.isfinite(g).all() for g in gradients)
                            or not any(torch.count_nonzero(g) for g in gradients)
                            or any(p.grad is not None for p in base.parameters())):
                        raise ValueError('Head/base gradient contract failed')
                    with torch.no_grad():
                        native = base(inputs)
                    errors = [float((a-b).abs().max()) for a,b in zip(native,result['base_outputs'])]
                    if len(native) != len(result['base_outputs']) or any(error != 0 for error in errors):
                        raise ValueError('Wrapper changed legacy native outputs')
                    metadata = metadata_rows[probes[position]]
                    checked.append(dict(split=split, frame_index=probes[position],
                        raw_signal_bits=metadata['raw_signal_bits'], target=metadata['target'],
                        loss=float(loss), native_output_max_abs_differences=errors))
                    print(json.dumps(dict(phase='preflight', **checked[-1])), flush=True)
        metrics = evaluate(model, datasets['val'], preflight_indices(datasets['val']), args, 'preflight')
        report = dict(status='COMPLETE', preflight_passed=True, training_updates=0,
                      base_encoder_updates=0, probes=checked, coverage=coverage,
                      diagnostic_metrics=metrics, trained_performance_measured=False,
                      live_expert_connected=False)
    elif args.mode == 'evaluate':
        metrics = evaluate(model, datasets['val'], selected['val'], args, 'evaluation')
        report = dict(status='COMPLETE', training_updates=0, metrics=metrics,
                      decision=selection_decision(metrics), base_encoder_updates=0,
                      live_expert_connected=False)
    else:
        optimizer = torch.optim.AdamW(model.permission_head.parameters(), lr=args.learning_rate)
        baseline = evaluate(model, datasets['val'], selected['val'], args, 'untrained_baseline')
        write(args.output/'untrained_baseline.json', baseline)
        best = None
        updates = 0
        with (args.output/'train.jsonl').open('x', buffering=1) as log:
            for epoch in range(1,args.epochs+1):
                model.train()
                for batch in loader(datasets['train'], selected['train'], args, True, args.seed+epoch):
                    optimizer.zero_grad(set_to_none=True)
                    _, loss, _, _, _, _ = forward(model, batch, args.device)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite permission loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.permission_head.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                    if updates % 100 == 0:
                        row = dict(epoch=epoch, updates=updates, loss=float(loss),
                                   elapsed_seconds=time.monotonic()-start)
                        log.write(json.dumps(row)+'\n')
                        print(json.dumps(row), flush=True)
                metrics = evaluate(model,datasets['val'],selected['val'],args,'epoch_%03d'%epoch)
                decision = selection_decision(metrics)
                head_payload = build_route_permission_checkpoint(model.permission_head, actual_sha)
                head_payload['metadata'] = dict(epoch=epoch,training_updates=updates,
                    provenance=provenance,metrics=metrics,selection=decision)
                save(args.output/('head-epoch-%03d.pt'%epoch),head_payload)
                write(args.output/('evaluation_%03d.json'%epoch),dict(metrics=metrics,decision=decision))
                if decision['eligible'] and (best is None or decision['score'] > best['score']):
                    save(args.output/'head-best.pt',head_payload)
                    best = dict(epoch=epoch,score=decision['score'],path=str(args.output/'head-best.pt'))
        report = dict(status='COMPLETE',training_updates=updates,base_encoder_updates=0,
                      completed_epochs=args.epochs,best=best,after=metrics,
                      qualified_candidate_available=best is not None,live_expert_connected=False)
    unchanged = initial_base_digest == nested_state_digest(base.state_dict())
    if not unchanged:
        raise ValueError('Frozen base encoder changed')
    report.update(base_encoder_unchanged=unchanged,base_checkpoint_sha256=actual_sha,
                  elapsed_seconds=time.monotonic()-start,automatic_deployment=False)
    write(args.output/'report.json', report)
    print(json.dumps(dict(status='COMPLETE',report=str(args.output/'report.json'),
                         training_updates=report['training_updates'])),flush=True)
    return report


def main():
    run(argument_parser().parse_args())


if __name__ == '__main__':
    main()
