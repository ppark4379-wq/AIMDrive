#!/usr/bin/env python3
"""Evaluate sealed protected-left signal laps without training or a simulator."""
import argparse
import datetime
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def preflight(plan_path):
    from data.src.loaders.morai_sensor_record_dataset import (
        PathResolver, SensorRecordDataset, digest, load_publication)
    from utils.evaluation.morai_signal_waypoint_metrics import load_arrow_transition

    plan = json.loads(plan_path.read_text())
    if plan.get('schema') != 'morai_signal_ve_baseline_plan_v1':
        raise ValueError('Unsupported signal baseline plan')
    if plan['split'] != 'val' or plan['expected_frames'] != 416:
        raise ValueError('This pilot evaluates the original complete held-out lap')
    if plan['velocity_source'] != 'ego_topic':
        raise ValueError('Preserve the checkpoint Ego-topic velocity contract')
    for record in plan.get('evidence_files', []):
        if digest(Path(record['path'])) != record['sha256']:
            raise ValueError('Prepared signal evidence changed: ' + record['path'])
    for name, expected in plan['implementation_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Prepared evaluation source changed: ' + name)
    resolver = PathResolver(plan.get('path_mappings', []))
    publication_path = resolver(plan['manifest_index'])
    if digest(publication_path) != plan['manifest_index_sha256']:
        raise ValueError('Prepared signal publication changed')
    publication, splits = load_publication(publication_path, resolver)
    if publication['counts'] != {'train': 831, 'val': 416}:
        raise ValueError('Signal pilot frame membership changed')
    if publication.get('scenario_name') != 'full_loop_traffic_light':
        raise ValueError('Unexpected signal scenario')
    if len(splits['val']) != 1 or publication['val_laps'] != [3]:
        raise ValueError('Hold out the original entire third lap')
    manifest_path, document = splits['val'][0]
    dataset = SensorRecordDataset(splits['val'], resolver, velocity_source='ego_topic')
    if len(dataset) != plan['expected_frames'] or dataset.excluded:
        raise ValueError('Official dataset admission changed the held-out frames')
    if dataset.map_basis != plan['map_basis']:
        raise ValueError('Signal targets and selected checkpoint bases differ')
    episode = resolver(document['episode'], manifest_path.parent)
    transition = load_arrow_transition(episode)
    from collections import Counter
    from utils.evaluation.morai_signal_waypoint_metrics import annotate_entry
    phase_counts = dict(Counter(annotate_entry(entry, transition)['group']
                                for entry in document['entries']))
    if phase_counts != plan['expected_phase_counts']:
        raise ValueError('Signal phase membership changed')
    for checkpoint in plan['checkpoints']:
        path = resolver(checkpoint['path'])
        if not path.is_file() or digest(path) != checkpoint['sha256']:
            raise ValueError('Selected checkpoint hash differs: ' + checkpoint['label'])
    summary = {'status': 'PASS', 'frames': len(dataset),
               'manifest_index': str(publication_path),
               'map_basis': dataset.map_basis, 'velocity_source': 'ego_topic',
               'checkpoints': [row['label'] for row in plan['checkpoints']],
               'model_loaded': False, 'training_updates': 0,
               'maximum_sensor_forwards': len(dataset) * len(plan['checkpoints']),
               'phase_counts': phase_counts,
               'arrow_transition': transition}
    return plan, dataset, summary


def run(args):
    import torch
    from utils.evaluation.morai_transfer_evaluation import (
        _load_model, make_evaluation_loader, ordered_object_entries)
    from utils.evaluation.morai_transfer_metrics import evaluate
    from utils.evaluation.morai_signal_waypoint_metrics import SignalWaypointMetricsAccumulator
    from data.src.loaders.morai_sensor_record_dataset import digest

    torch.set_num_threads(args.threads)
    started = time.monotonic()
    print(json.dumps({'phase': 'verify_publication_and_checkpoint_hashes',
                      'model_loaded': False, 'training_updates': 0}), flush=True)
    plan, dataset, summary = preflight(args.plan)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.check:
        return summary
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not replace the full evaluation with CPU silently')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = args.output or args.plan.parent / 'runs' / ('run_' + stamp + '_' + str(time.time_ns() % 1000000))
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    entries = ordered_object_entries(dataset)
    write_json(output / 'preflight.json', summary)
    write_json(output / 'plan.json', plan)
    results = []

    class RecordingModel(torch.nn.Module):
        def __init__(self, model, accumulator):
            super().__init__()
            self.model = model
            self.accumulator = accumulator
            self.pending_targets = None

        def forward(self, inputs):
            if self.pending_targets is None:
                raise ValueError('Missing exact target batch for signal recording')
            predictions = self.model(inputs)
            self.accumulator.update(predictions, self.pending_targets, inputs)
            self.pending_targets = None
            return predictions

    class RecordingLoader:
        def __init__(self, loader, recorder):
            self.loader, self.recorder = loader, recorder

        def __iter__(self):
            for inputs, targets in self.loader:
                if self.recorder.pending_targets is not None:
                    raise ValueError('Previous evaluation batch did not consume its targets')
                self.recorder.pending_targets = targets
                yield inputs, targets

    try:
        for selected in plan['checkpoints']:
            label = selected['label']
            print(json.dumps({'phase': 'load_model', 'checkpoint': label}), flush=True)
            model, payload, checkpoint_hash = _load_model(
                Path(selected['path']), selected['sha256'], args.device)
            contract = payload.get('morai_encoder_contract', {})
            if contract.get('map_basis') != dataset.map_basis:
                raise ValueError('Loaded model/map basis mismatch; bypass is disabled')
            if contract.get('velocity_input') != 'causal_ego_topic_speed_mps_50ms':
                raise ValueError('Loaded model velocity contract mismatch')
            checkpoint_metadata = {'epoch': payload.get('epoch'), 'steps': payload.get('steps')}
            del payload
            accumulator = SignalWaypointMetricsAccumulator(entries, summary['arrow_transition'])
            recorder = RecordingModel(model, accumulator)
            loader = RecordingLoader(make_evaluation_loader(dataset, args.workers), recorder)
            checkpoint_started = time.monotonic()
            metrics = evaluate(recorder, loader, args.device,
                               progress_phase='signal_evaluation:' + label, progress_every=25,
                               detection_loss_policy='lmdrive_lav_v1', object_entries=entries)
            phase_metrics = accumulator.compute()
            predictions = output / (label + '_predictions.jsonl')
            with predictions.open('x', encoding='utf-8') as stream:
                for row in accumulator.rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            row = {'label': label, 'selection_role': selected['role'],
                   'checkpoint': selected['path'], 'checkpoint_sha256': checkpoint_hash,
                   'checkpoint_metadata': checkpoint_metadata, 'frames': len(dataset),
                   'metrics': metrics, 'signal_waypoint_metrics': phase_metrics,
                   'predictions': str(predictions), 'predictions_sha256': digest(predictions),
                   'elapsed_seconds': time.monotonic() - checkpoint_started,
                   'training_updates': 0, 'model_forwards': len(accumulator.rows)}
            write_json(output / (label + '_report.json'), row)
            results.append(row)
            print(json.dumps({'completed': label, 'frames': len(accumulator.rows),
                              'ade_m': metrics['ade_m'], 'fde_m': metrics['fde_m'],
                              'elapsed_seconds': row['elapsed_seconds']}, ensure_ascii=False), flush=True)
            del recorder, loader, model, accumulator
            gc.collect()
            if args.device == 'cuda':
                torch.cuda.empty_cache()
        report = {'schema': 'morai_signal_ve_baseline_result_v1', 'status': 'COMPLETE',
                  'plan': str(args.plan), 'plan_sha256': digest(args.plan),
                  'manifest_index': summary['manifest_index'], 'evaluated_split': 'val',
                  'frames_per_checkpoint': len(dataset), 'results': results,
                  'training_updates': 0, 'weights_updated': False,
                  'automatic_training_or_deployment': False,
                  'execution': {'device': args.device, 'workers': args.workers,
                                'main_torch_threads': args.threads, 'batch_size': 1},
                  'elapsed_seconds': time.monotonic() - started,
                  'scope': 'Held-out third lap on the same map and route; recorded-input evaluation.',
                  'note': 'Color logits do not directly gate VE waypoints. Arrow color supervision remains masked.'}
        write_json(output / 'report.json', report)
        print(json.dumps({'status': 'COMPLETE', 'report': str(output / 'report.json'),
                          'training_updates': 0, 'elapsed_seconds': report['elapsed_seconds']},
                         ensure_ascii=False), flush=True)
        return report
    except BaseException as error:
        write_json(output / 'failure.json', {'status': 'FAILED', 'error': repr(error),
                                            'completed_checkpoints': [row['label'] for row in results],
                                            'training_updates': 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true', help='Official metadata admission only; no model construction')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 0 or args.threads < 1:
        parser.error('Invalid worker/thread count')
    args.plan = args.plan.expanduser().resolve()
    run(args)


if __name__ == '__main__':
    main()
