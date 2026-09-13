#!/usr/bin/env python3
"""Run the matched A/B pilot sequentially; never silently launch full training."""
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
sys.path.insert(0, str(ROOT))
from tools.morai.root_pilot_checks import compare
RUNS = ROOT / '.runtime/training_runs'


def write(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def worker(run, config):
    with (RUNS / 'last_encoder.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(worker_pid=os.getpid(), run_dir=str(run), config=str(config),
                     scope='bounded train-only matched pilot', automatic_full_training=False)
        def status(phase, **extra):
            state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
            write(run / 'status.json', state)
            print(json.dumps(state), flush=True)
        try:
            if shutil.disk_usage(ROOT).free < 100 * 1024**3:
                raise RuntimeError('Require 100 GiB free for preserved checkpoints and two pilot arms')
            reports = []
            for arm in ('legacy', 'retained_output_st'):
                output = ROOT / 'outputs' / (run.name + '_' + arm)
                command = [sys.executable, '-u', str(ROOT / 'train_morai_root_pilot.py'),
                           '--config', str(config), '--output', str(output), '--arm', arm,
                           '--lock-fd', str(lock.fileno())]
                with (run / (arm + '.log')).open('x') as log:
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                             pass_fds=(lock.fileno(),))
                    status('running_' + arm, child_pid=child.pid, output=str(output),
                           progress=str(output / 'progress.json'), log=str(run / (arm + '.log')))
                    code = child.wait()
                if code:
                    raise RuntimeError('%s arm exited with code %d; see its log' % (arm, code))
                reports.append(json.loads((output / 'report.json').read_text()))
            result = compare(*reports)
            write(run / 'comparison.json', result)
            status('comparison_passed_review_encoder_stage' if result['passed'] else 'comparison_gate_failed',
                   child_pid=None, comparison=result,
                   next_step='Review report before corrected-label encoder fine-tune/cache regeneration/full downstream')
        except Exception as error:
            status('failed', child_pid=None, error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'lavis/projects/vldrive/morai_root_cause_pilot.json')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = args.config.resolve()
    if ROOT not in config.parents or not config.is_file():
        raise ValueError('Expected an explicit config inside LAVIS')
    if args.worker:
        run = args.worker.resolve()
        if run.parent != RUNS.resolve() or not run.is_dir():
            raise ValueError('Invalid worker output')
        worker(run, config)
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    # Check BEFORE spawning as well; the worker holds the lock for both arms.
    with (RUNS / 'last_encoder.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = RUNS / ('morai_root_pilot_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    run.mkdir()
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               LAVIS_IMPORT_PROFILE='morai', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(run), '--config', str(config)]
    with (run / 'worker.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launch = dict(worker_pid=process.pid, run_dir=str(run), status=str(run / 'status.json'))
    write(run / 'launch.json', launch)
    print(json.dumps(launch, indent=2), flush=True)


if __name__ == '__main__':
    main()
