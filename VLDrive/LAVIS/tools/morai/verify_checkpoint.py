#!/usr/bin/env python3
"""Fresh-process restore, prediction identity and exact next optimizer update."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
import os
sys.path.insert(0, str(ROOT))
os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
from lavis.common.morai_runtime import configure_runtime, add_path_arguments, resolve_settings
configure_runtime()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    output = args.output.resolve()
    if ROOT not in experiment.parents or ROOT not in output.parents or output.exists():
        raise ValueError('Use an existing experiment and new verification output in this checkout')
    output.mkdir(parents=True)
    from lavis.common.morai_runtime import bootstrap
    backend = bootstrap(ROOT.parent)
    import torch
    import numpy as np
    from lavis.models.vldrive.morai_model import MoraiDrivingModel, MoraiDrivingConfig
    from lavis.datasets.datasets.morai.training_dataset import MoraiTrainingDataset
    from lavis.runners.morai_trainer import (seed_all, source_hashes, make_optimizer, restore_checkpoint,
        evaluate, train, parameter_hash)
    settings = json.loads((experiment / 'resolved_config.json').read_text())
    indices = json.loads((experiment / 'window_indices.json').read_text())
    expected_report = json.loads((experiment / 'report.json').read_text())
    config = MoraiDrivingConfig(**settings['model'])
    for field in ('visual_checkpoint', 'language_checkpoint'):
        setattr(config, field, str((ROOT / getattr(config, field)).resolve()))
    torch.set_num_threads(settings.get('cpu_threads', 4))
    seed_all(settings['seed'])
    datasets = {split: MoraiTrainingDataset(ROOT / settings['dataset'],
        split_file=ROOT / settings['dataset'] / settings['split_file'], split=split,
        sequence_length=settings['sequence_length'], input_mode=config.input_mode,
        instructions=ROOT / settings['instructions'] if settings.get('instructions') else None)
        for split in ('train', 'val')}
    provenance = dict(backend=backend, source=source_hashes(ROOT), datasets={k: v.describe() for k, v in datasets.items()})
    if not config.visual_sha256:
        config.visual_sha256 = datasets['train'].base.checkpoint_sha256
    model = MoraiDrivingModel(config, device=settings['device'])
    optimizer = make_optimizer(model, **settings.get('optimizer', {}))
    saved = restore_checkpoint(experiment / 'checkpoint.pt', model, optimizer, provenance=provenance)
    report = dict(restored_trainable_hash=parameter_hash(model, trainable=True),
                  restored_frozen_hash=parameter_hash(model, trainable=False), saved_updates=saved['updates'])
    report['validation'] = evaluate(model, datasets['val'], indices['val'], settings['device'],
                                    save_predictions=output / 'val_predictions.json')
    expected = json.loads((experiment / 'val_predictions.json').read_text())
    actual = json.loads((output / 'val_predictions.json').read_text())
    if [r['sample_id'] for r in expected] != [r['sample_id'] for r in actual]:
        raise ValueError('Restored evaluation sample identities changed')
    deltas = {k: max(float(np.max(np.abs(np.array(a[k]) - np.array(b[k]))))
              for a, b in zip(expected, actual)) for k in ('trajectory', 'speed')}
    report['restore_prediction_max_delta'] = deltas
    report['parameter_hash_matches'] = report['restored_trainable_hash'] == expected_report['trainable_hash']
    # Exercise the actual full-graph optimizer twice from identical checkpoint/RNG state.
    hashes = []
    for attempt in range(2):
        restore_checkpoint(experiment / 'checkpoint.pt', model, optimizer, provenance=provenance)
        train(model, optimizer, datasets['train'], indices['train'], start_update=saved['updates'],
            updates=saved['updates'] + 1, seed=settings['seed'], device=settings['device'],
            log_path=output / ('resume_attempt_{}.jsonl'.format(attempt)))
        hashes.append(parameter_hash(model, trainable=True))
    report.update(next_step_hashes=hashes, exact_next_update=hashes[0] == hashes[1],
                  probe_optimizer_updates=2, source_checkpoint_modified=False)
    report['passed'] = report['parameter_hash_matches'] and report['exact_next_update'] and max(deltas.values()) <= 1e-5
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit('Checkpoint verification failed')


if __name__ == '__main__':
    main()
