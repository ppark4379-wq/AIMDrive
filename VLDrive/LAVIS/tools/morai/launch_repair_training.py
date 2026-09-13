#!/usr/bin/env python3
"""Launch authorized frozen-encoder repair pilot, then full training only on a pass."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT/'.runtime/training_runs'


def write(path, value):
    pending = path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    pending.replace(path)


def worker(run, config, pilot_only):
    with (RUNS/'last_encoder.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another frozen-encoder pipeline already owns the training lock')
        output = ROOT/'outputs'/run.name
        state = dict(worker_pid=os.getpid(), output=str(output), config=str(config),
                     progress=str(output/'progress.json'), log=str(run/'pipeline.log'))
        def status(phase, **extra):
            state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
            write(run/'status.json', state)
            print(json.dumps(state), flush=True)
        try:
            status('checking')
            free_gib = shutil.disk_usage(ROOT).free/1024**3
            if free_gib < 100:
                raise RuntimeError('Need at least100GiB free for preserved prior models plus rotating repair checkpoints; available%.1fGiB' % free_gib)
            command = [sys.executable, '-u', 'train_morai_repair.py', '--config', str(config), '--output', str(output)]
            if pilot_only:
                command.append('--pilot-only')
            with subprocess.Popen(command, cwd=ROOT) as child:
                status('running_gated_repair', child_pid=child.pid, command=command, free_gib=free_gib)
                code = child.wait()
            if code:
                raise RuntimeError('Repair child exited with code%d' % code)
            progress = json.loads((output/'progress.json').read_text())
            status(progress['phase'], child_pid=None)
        except Exception as error:
            status('failed', error=str(error), child_pid=None)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'lavis/projects/vldrive/morai_repair_epoch25.json')
    parser.add_argument('--pilot-only', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = args.config.resolve()
    if ROOT not in config.parents or not config.is_file():
        raise ValueError('Use a checked-in repair config under LAVIS')
    if args.worker:
        run = args.worker.resolve()
        if run.parent != RUNS.resolve() or not run.is_dir():
            raise ValueError('Invalid worker run directory')
        worker(run, config, args.pilot_only)
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    name = 'morai_repair_epoch25_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = RUNS/name
    run.mkdir()
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               LAVIS_IMPORT_PROFILE='morai', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(run), '--config', str(config)]
    if args.pilot_only:
        command.append('--pilot-only')
    with (run/'pipeline.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch = dict(worker_pid=process.pid, run_dir=str(run), output=str(ROOT/'outputs'/name),
                  progress=str(ROOT/'outputs'/name/'progress.json'), log=str(run/'pipeline.log'))
    write(run/'launch.json', launch)
    print(json.dumps(launch, indent=2), flush=True)


if __name__ == '__main__':
    main()
