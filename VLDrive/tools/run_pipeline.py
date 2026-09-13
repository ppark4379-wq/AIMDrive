#!/usr/bin/env python3
"""Sequential full-route production pipeline: VE fine-tune, cache, downstream training.

Unlike the bounded diagnostic runner this produces promotable assets and allows
long stages, so every transition is gated on an explicit artifact check. Any
failed gate stops the pipeline instead of spending later stages on bad inputs.
Run it as a persistent foreground WSL process; it holds the shared GPU lock for
the whole pipeline and writes a heartbeat.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
MIN_FREE_GIB = 120


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def gate_vision_encoder(run):
    """The trainer verifies the connector contract itself; confirm it recorded one."""
    contract = run / 'feature_contract.json'
    if not contract.is_file():
        raise RuntimeError('Vision encoder produced no feature_contract.json')
    spec = json.loads(contract.read_text(encoding='utf-8'))
    shape = list(spec.get('connector_token_shape') or spec.get('token_shape') or [])
    if shape[-2:] != [106, 256]:
        raise RuntimeError('Vision encoder token shape is not [106,256]: {}'.format(shape))
    if not (run / 'checkpoint-last.pth.tar').is_file():
        raise RuntimeError('Vision encoder produced no checkpoint-last.pth.tar')
    return dict(feature_contract=spec, checkpoint_sha256=digest(run / 'checkpoint-last.pth.tar'))


def gate_cache(cache, split_file, encoder_checkpoint):
    meta_path = cache / 'cache_meta.json'
    manifest = cache / 'manifest.jsonl'
    if not meta_path.is_file() or not manifest.is_file():
        raise RuntimeError('Token cache is missing cache_meta.json or manifest.jsonl')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('split_sha256') != digest(split_file):
        raise RuntimeError('Token cache was built against a different split manifest')
    if meta.get('checkpoint_sha256') != digest(encoder_checkpoint):
        raise RuntimeError('Token cache was built against a different encoder checkpoint')
    rows = [json.loads(line) for line in manifest.open(encoding='utf-8') if line.strip()]
    if not rows:
        raise RuntimeError('Token cache manifest is empty')
    return dict(frames=len(rows), split_sha256=meta['split_sha256'],
                checkpoint_sha256=meta['checkpoint_sha256'])


def gate_preflight(run):
    report = run / 'report.json'
    if not report.is_file():
        raise RuntimeError('Preflight produced no report.json')
    return json.loads(report.read_text(encoding='utf-8'))


def build_stages(args, stamp):
    ve_run = ROOT / 'outputs/checkpoints/vision_encoder' / args.encoder_name
    cache = ROOT / 'data/processed/vision_tokens' / args.cache_name
    preflight_out = ROOT / 'outputs/logs' / ('fullroute_preflight_' + stamp)
    train_out = ROOT / 'outputs/logs' / ('fullroute_train_' + stamp)
    encoder_ckpt = ve_run / 'checkpoint-last.pth.tar'
    split_file = ROOT / 'configs/splits' / args.split_name
    config = ROOT / 'configs/vldrive_train_fullroute.json'

    stages = [
        dict(name='vision_encoder', target='vision_encoder_run', cwd=ROOT, command=[
            sys.executable, '-u', str(ROOT / 'train.py'), 'vision-encoder',
            str(WORKSPACE / 'MoraiDrive/dataset_local/competition_train'),
            '--split-file', str(split_file),
            '--output', str(ve_run),
            '--warm-start', str(args.warm_start),
            '--warm-start-sha256', args.warm_start_sha256,
            '--warm-start-source-split-sha256', args.warm_start_source_split_sha256,
            '--traffic-label-sidecar', str(args.sidecar),
            '--epochs', str(args.encoder_epochs),
            '--warmup-epochs', str(args.encoder_warmup),
            '--reference-lr', str(args.encoder_lr),
            '--reference-rgb-lr', str(args.encoder_rgb_lr),
            '--rgb-hue-jitter', '0',
            '--feature-checkpoint', 'last',
        ], gate=lambda: gate_vision_encoder(ve_run)),
        dict(name='token_cache', target='token_cache', cwd=ROOT, command=[
            sys.executable, '-u', str(ROOT / 'tools/cache_vision_tokens.py'),
            '--dataset-root', str(WORKSPACE / 'MoraiDrive/dataset_local/competition_train'),
            '--split-file', str(split_file),
            '--checkpoint', str(encoder_ckpt),
            '--output', str(cache),
            '--batch-size', '1', '--workers', '2',
        ], gate=lambda: gate_cache(cache, split_file, encoder_ckpt)),
        dict(name='preflight', target='preflight', cwd=ROOT, command=[
            sys.executable, '-u', str(ROOT / 'train.py'), 'vldrive',
            '--config', str(config), '--mode', 'preflight',
            '--output', str(preflight_out),
        ], gate=lambda: gate_preflight(preflight_out)),
        dict(name='downstream_training', target='training', cwd=ROOT, command=[
            sys.executable, '-u', str(ROOT / 'train.py'), 'vldrive',
            '--config', str(config), '--output', str(train_out),
        ], gate=lambda: dict(output=str(train_out))),
    ]
    return stages, dict(vision_encoder_run=str(ve_run), token_cache=str(cache),
                        preflight=str(preflight_out), training=str(train_out))


def run(args):
    import fcntl
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = ROOT / '.runtime/training_runs' / ('full_route_pipeline_' + stamp)
    if run_dir.exists():
        raise RuntimeError('Pipeline run directory already exists')
    stages, targets = build_stages(args, stamp)
    names = [stage['name'] for stage in stages]
    if args.start_stage not in names:
        raise ValueError('Unknown --start-stage; expected one of ' + ', '.join(names))
    skipped = names[:names.index(args.start_stage)]
    stages = stages[names.index(args.start_stage):]
    for stage in stages:
        if Path(targets[stage['target']]).exists():
            raise RuntimeError('Refusing to overwrite existing output: ' + targets[stage['target']])
    for name in skipped:
        gate = next(s for s in build_stages(args, stamp)[0] if s['name'] == name)['gate']
        gate()

    lock_path = ROOT / '.runtime/training_runs/last_encoder.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_dir.mkdir(parents=True)
        state = dict(pipeline='full_route_v1', worker_pid=os.getpid(), targets=targets,
                     stages_total=len(stages), skipped_stages=skipped, completed=[],
                     started_utc=datetime.now(timezone.utc).isoformat())

        def status(phase, **extra):
            state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
            write_json(run_dir / 'status.json', state)

        child = None
        try:
            env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            for index, stage in enumerate(stages):
                if shutil.disk_usage(ROOT).free < MIN_FREE_GIB * 1024 ** 3:
                    raise RuntimeError('Insufficient free disk before ' + stage['name'])
                log_path = run_dir / (stage['name'] + '.log')
                started = time.monotonic()
                status('running', stage=stage['name'], stage_index=index, child_pid=None,
                       log=str(log_path), command=stage['command'])
                with log_path.open('x') as log:
                    child = subprocess.Popen(stage['command'], cwd=str(stage['cwd']), env=env,
                                             stdin=subprocess.DEVNULL, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    status('running', stage=stage['name'], stage_index=index,
                           child_pid=child.pid, log=str(log_path))
                    while True:
                        try:
                            code = child.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            status('running', stage=stage['name'], stage_index=index,
                                   child_pid=child.pid, log=str(log_path),
                                   elapsed_stage_seconds=time.monotonic() - started)
                if code:
                    raise RuntimeError('{} failed with exit code {}; see {}'.format(
                        stage['name'], code, log_path))
                child = None
                status('gating', stage=stage['name'], stage_index=index, child_pid=None)
                evidence = stage['gate']()
                state['completed'].append(dict(name=stage['name'],
                                               seconds=time.monotonic() - started,
                                               evidence=evidence))
            status('complete_needs_scientific_review', child_pid=None)
        except BaseException as error:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
            status('failed', child_pid=None, error=repr(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-stage', default='vision_encoder',
                        help='Resume from this stage; earlier stages must already satisfy their gate')
    parser.add_argument('--split-name', default='morai_full_loop_95_05.json')
    parser.add_argument('--encoder-name', default='full_route_hue_safe')
    parser.add_argument('--cache-name', default='full_loop_tokens_fullroute')
    parser.add_argument('--warm-start', type=Path,
                        default=ROOT / 'outputs/checkpoints/vision_encoder/morai-full-loop-epoch25.pth.tar')
    parser.add_argument('--warm-start-sha256',
                        default='5bd84eecae64fff810a5db4d7cae9f55db1ef5844aad2edf0c1bc25e3fcbdbb4')
    parser.add_argument('--warm-start-source-split-sha256',
                        default='8b1750410c3da6217be30dd54b0f03a2838dfd45a7c88fdf20c2fd8f7e412db6',
                        help='SHA-256 of the 80/20 split the epoch25 encoder was trained on')
    parser.add_argument('--sidecar', type=Path,
                        default=ROOT / 'data/processed/traffic_labels/full_route_95_05.json')
    parser.add_argument('--encoder-epochs', type=int, default=12)
    parser.add_argument('--encoder-warmup', type=int, default=2)
    parser.add_argument('--encoder-lr', type=float, default=3e-4)
    parser.add_argument('--encoder-rgb-lr', type=float, default=1.2e-4)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Pipeline received signal {}'.format(signum))
    signal.signal(signal.SIGTERM, interrupted)
    run(args)


if __name__ == '__main__':
    main()
