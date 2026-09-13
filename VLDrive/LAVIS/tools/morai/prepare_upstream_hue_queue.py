#!/usr/bin/env python3
"""Fingerprint an explicit smoke/full diagnostic queue without starting training."""
import argparse
import json
from pathlib import Path

from run_bounded_diagnostics import ROOT, WORKSPACE, digest, validate_plan

BASE = ROOT / 'outputs/diagnosis_upstream_hue_20260911'
VISION = ROOT.parent / 'vision_encoder'
ROTATION = ROOT / 'tools/morai/train_matched_upstream_probe.py'
SIGNAL = VISION / 'tools/paired_hue_finetune.py'
OLD = ROOT / 'outputs/morai_paper_last_encoder_20260910T055054Z'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('rotation_smoke', 'signal_smoke', 'full'), required=True)
    parser.add_argument('--tag', default='v1')
    args = parser.parse_args()
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.tag):
        parser.error('Use a safe unique tag')
    plan_path = BASE / (args.phase + '_' + args.tag + '_queue.json')
    if plan_path.exists():
        raise FileExistsError(plan_path)
    jobs = []
    if args.phase in ('rotation_smoke', 'full'):
        prefix = 'smoke_rotation' if args.phase == 'rotation_smoke' else 'rotation'
        a_output = BASE / (prefix + '_A_' + args.tag)
        for arm in ('A', 'B'):
            output = BASE / (prefix + '_' + arm + '_' + args.tag)
            if output.exists():
                raise FileExistsError(output)
            command = ['--arm', arm, '--output', str(output), '--steps',
                       '2' if args.phase == 'rotation_smoke' else '1280']
            if args.phase == 'rotation_smoke':
                command += ['--eval-limit', '2']
            if arm == 'B':
                command += ['--paired-reference', str(a_output)]
            jobs.append(dict(name=prefix.lower() + '_' + arm.lower(), script=str(ROTATION),
                             cwd=str(ROOT), args=command,
                             timeout_seconds=1800 if args.phase == 'rotation_smoke' else 14400))
    if args.phase in ('signal_smoke', 'full'):
        prefix = 'smoke_signal' if args.phase == 'signal_smoke' else 'signal'
        output = BASE / (prefix + '_' + args.tag)
        if output.exists():
            raise FileExistsError(output)
        command = ['--action', 'run', '--output', str(output)]
        if args.phase == 'signal_smoke':
            command += ['--smoke', '--max-train-frames', '8']
        prepare_command = list(command)
        prepare_command[1] = 'prepare'
        jobs.append(dict(name=prefix + '_prepare', script=str(SIGNAL), cwd=str(VISION),
                         args=prepare_command, timeout_seconds=1800))
        jobs.append(dict(name=prefix, script=str(SIGNAL), cwd=str(VISION), args=command,
                         timeout_seconds=1800 if args.phase == 'signal_smoke' else 14400))
    paths = [BASE / 'EXPERIMENT_PLAN.md', Path(__file__),
             ROOT / 'tools/morai/run_bounded_diagnostics.py',
             OLD / 'model-last.pt', OLD / 'model-last.pt.sha256.json',
             OLD / 'resolved_config.json', OLD / 'window_indices.json',
             OLD / 'provenance.json',
             ROOT / '.runtime/morai_vldrive/full_loop_instructions.jsonl',
             ROOT / '.runtime/morai_vldrive/full_loop_tokens_epoch25/cache_meta.json',
             ROOT / '.runtime/morai_vldrive/full_loop_tokens_epoch25/manifest.jsonl',
             ROOT / 'assets/pretrained/vision_encoder/morai-full-loop-epoch25.pth.tar',
             VISION / 'output_morai/full_loop_expert/checkpoint-last.pth.tar',
             VISION / 'configs/morai_full_loop_80_20.json',
             VISION / '.runtime/traffic_labels_v2/full_loop_20260911.json',
             VISION / 'train_morai_pretrain.py', VISION / 'morai_data.py',
             VISION / 'morai_traffic_labels.py',
             WORKSPACE / 'MoraiDrive/dataset_local/competition_train/data/full_loop_basic_drive_2026_09_09_01_44_15/measurements_all.json']
    for folder, pattern in ((ROOT / 'lavis/models/vldrive', '*.py'),
                            (ROOT / 'lavis/runners', '*morai*.py'),
                            (ROOT / 'lavis/datasets/datasets/morai', '*.py'),
                            (VISION / 'timm', '**/*.py')):
        paths.extend(folder.glob(pattern))
    paths.extend([ROOT / 'lavis/runners/vldrive_trainer.py', ROOT / 'lavis/common/morai_runtime.py'])
    paths.extend(Path(job['script']) for job in jobs)
    for filename in ('model_readout_train.py', 'audit_selection.json'):
        paths.append(ROOT / 'outputs/diagnosis_controlled_experiments_20260911' / filename)
    hashes = {}
    for path in sorted(set(p.resolve() for p in paths)):
        hashes[str(path)] = digest(path)
    plan = dict(schema='bounded-diagnostics-v1', phase=args.phase, tag=args.tag,
                automatic_full_training=False, promote_models=False, min_free_gib=100,
                jobs=jobs, preserved_sha256=hashes,
                scope='Only preregistered training-split diagnostic pilots, not a production model')
    validate_plan(plan)
    with plan_path.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(plan=str(plan_path), plan_sha256=digest(plan_path),
                         jobs=[j['name'] for j in jobs], preserved_files=len(hashes)), indent=2))


if __name__ == '__main__':
    main()
