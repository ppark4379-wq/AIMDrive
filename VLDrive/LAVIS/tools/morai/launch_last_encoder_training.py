#!/usr/bin/env python3
"""Launch the authorized epoch-25 cache -> VLDrive training run in WSL.

The worker survives the invoking terminal, logs each phase, and never starts
training if cache generation/validation fails. Existing best artifacts stay intact.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'lavis/projects/vldrive/morai_paper_last_encoder.json'
RUNS = ROOT / '.runtime/training_runs'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    pending.replace(path)


def worker(run):
    # One owner for this last-encoder cache and training pipeline at a time.
    with (RUNS / 'last_encoder.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('The last-encoder pipeline is already running')
        config = json.loads(CONFIG.read_text(encoding='utf-8'))
        output = ROOT / 'outputs' / run.name
        state = dict(worker_pid=os.getpid(), started_utc=now(), config=str(CONFIG),
                     output=str(output), log=str(run / 'pipeline.log'))
        def status(phase, **fields):
            state.update(phase=phase, updated_utc=now(), **fields)
            write_json(run / 'status.json', state)
            print(json.dumps(state), flush=True)
        def execute(phase, command):
            with subprocess.Popen(command, cwd=ROOT) as process:
                status(phase, child_pid=process.pid, command=command)
                code = process.wait()
            if code:
                raise RuntimeError('{} exited with code {}'.format(phase, code))
        try:
            status('checking')
            if output.exists():
                raise FileExistsError(output)
            # Two rotating + one final snapshot, temporary save and model files.
            if shutil.disk_usage(ROOT).free < 100 * 1024**3:
                raise RuntimeError('At least 100 GiB of free disk is required for this run')
            checkpoint = (ROOT / config['model']['visual_checkpoint']).resolve()
            with checkpoint.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest() if sys.version_info >= (3, 11) else None
                if digest is None:
                    h = hashlib.sha256()
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                        h.update(block)
                    digest = h.hexdigest()
            if digest != config['model']['visual_sha256']:
                raise ValueError('Epoch-25 encoder hash mismatch')
            cache = (ROOT / config['feature_cache']).resolve()
            if not (cache / 'cache_meta.json').is_file():
                execute('caching', [sys.executable, '-u', 'tools/morai/cache_expert_vision_tokens.py',
                    '--dataset-root', config['dataset'], '--split-file', config['split_file'],
                    '--checkpoint', str(checkpoint), '--output', str(cache),
                    '--batch-size', '4', '--workers', '4', '--log-interval', '100'])
            status('validating_cache', child_pid=None)
            os.environ['LAVIS_IMPORT_PROFILE'] = 'morai'
            sys.path.insert(0, str(ROOT))
            from lavis.datasets.datasets.morai.expert_vldrive_dataset import MoraiExpertVLDriveDataset
            frames = {}
            for split in ('train', 'val'):
                data = MoraiExpertVLDriveDataset(ROOT / config['dataset'], cache_root=cache,
                    split_file=ROOT / config['split_file'], instructions=ROOT / config['instructions'],
                    split=split, sequence_length=config['sequence_length'])
                if data.checkpoint_sha256 != digest:
                    raise ValueError('Cache does not belong to the last encoder')
                for row in data.entries:
                    data._array(row['visual_feature'], (106, 256), 'visual feature')
                    data._array(row['future_trajectory'], (5, 2), 'future trajectory')
                frames[split] = len(data.entries)
            status('cache_verified', verified_frames=frames)
            execute('training', [sys.executable, '-u', 'train_vldrive.py',
                '--config', str(CONFIG), '--output', str(output)])
            status('complete', child_pid=None, completed_utc=now())
        except Exception as error:
            status('failed', child_pid=None, error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run = args.worker.resolve()
        if run.parent != RUNS.resolve() or not run.is_dir():
            raise ValueError('Invalid worker run directory')
        worker(run)
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    name = 'morai_paper_last_encoder_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = RUNS / name
    run.mkdir()
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               LAVIS_IMPORT_PROFILE='morai', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    with (run / 'pipeline.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(run)],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    launch = dict(worker_pid=process.pid, run_dir=str(run), output=str(ROOT / 'outputs' / name),
                  status=str(run / 'status.json'), log=str(run / 'pipeline.log'), launched_utc=now())
    write_json(run / 'launch.json', launch)
    print(json.dumps(launch, indent=2), flush=True)


if __name__ == '__main__':
    main()
