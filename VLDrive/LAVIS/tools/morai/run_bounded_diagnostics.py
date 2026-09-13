#!/usr/bin/env python3
"""Run an explicitly approved, finite diagnostic queue; never promote models.

Run this foreground worker through a persistent WSL process. It holds the shared
GPU training lock across all children, writes a heartbeat, and fails closed.
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

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parents[1]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def validate_plan(plan, workspace=WORKSPACE):
    if plan.get('automatic_full_training') is not False or plan.get('promote_models') is not False:
        raise ValueError('Automatic full training and model promotion must be disabled')
    jobs = plan.get('jobs', [])
    if not jobs or len(jobs) > 12:
        raise ValueError('Expected one to twelve explicitly bounded jobs')
    names = set()
    for job in jobs:
        name = job['name']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name) or name in names:
            raise ValueError('Job names must be unique safe identifiers')
        names.add(name)
        if not inside(job['script'], workspace) or not Path(job['script']).is_file():
            raise ValueError('Job script must exist inside the workspace')
        if not inside(job['cwd'], workspace) or not Path(job['cwd']).is_dir():
            raise ValueError('Job cwd must exist inside the workspace')
        if not 1 <= job['timeout_seconds'] <= 14400:
            raise ValueError('Every job needs a finite timeout <= four hours')
        if not isinstance(job['args'], list) or not all(isinstance(x, str) for x in job['args']):
            raise ValueError('Arguments must be a literal string list, never a shell command')
        if '--lock-fd' in job['args']:
            raise ValueError('Only the worker supplies the lock descriptor')
    if not plan.get('preserved_sha256'):
        raise ValueError('A preserved input/source fingerprint is required')
    for path, expected in plan['preserved_sha256'].items():
        if not inside(path, workspace) or len(expected) != 64:
            raise ValueError('Invalid preserved fingerprint')
    return plan


def check_preserved(plan):
    for path, expected in plan['preserved_sha256'].items():
        if digest(path) != expected:
            raise RuntimeError('Preserved source/input changed: ' + path)


def write_json(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def stop_owned_child(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=15)


def worker(plan_path, run):
    import fcntl
    plan_bytes = plan_path.read_bytes()
    plan = validate_plan(json.loads(plan_bytes))
    if not inside(run, ROOT / 'outputs') or run == ROOT / 'outputs' or run.exists():
        raise ValueError('Use a new, dedicated run directory inside LAVIS/outputs')
    lock_path = ROOT / '.runtime/training_runs/last_encoder.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run.mkdir(parents=True, exist_ok=False)
        (run / 'approved_plan.json').write_bytes(plan_bytes)
        state = dict(worker_pid=os.getpid(), plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                     jobs_total=len(plan['jobs']), completed_jobs=[], automatic_full_training=False,
                     promote_models=False, started_utc=datetime.now(timezone.utc).isoformat())

        def status(phase, **extra):
            state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
            write_json(run / 'status.json', state)

        child = None
        try:
            status('checking_preserved_inputs', child_pid=None)
            check_preserved(plan)
            env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                       LAVIS_IMPORT_PROFILE='morai', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            for index, job in enumerate(plan['jobs']):
                if shutil.disk_usage(ROOT).free < plan.get('min_free_gib', 100) * 1024**3:
                    raise RuntimeError('Insufficient free disk before ' + job['name'])
                if plan_path.read_bytes() != plan_bytes:
                    raise RuntimeError('Approved plan changed while running')
                # Full fingerprints include immutable checkpoints and source. Checking
                # at every boundary prevents combining runs from changed code.
                check_preserved(plan)
                command = [sys.executable, '-u', job['script'], *job['args'],
                           '--lock-fd', str(lock.fileno())]
                log_path = run / (job['name'] + '.log')
                started = time.monotonic()
                with log_path.open('x') as log:
                    child = subprocess.Popen(command, cwd=job['cwd'], env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        pass_fds=(lock.fileno(),))
                    while True:
                        elapsed = time.monotonic() - started
                        remaining = job['timeout_seconds'] - elapsed
                        if remaining <= 0:
                            raise TimeoutError('Bounded job exceeded its deadline: ' + job['name'])
                        status('running', job=job['name'], job_index=index, child_pid=child.pid,
                               elapsed_job_seconds=elapsed, log=str(log_path), command=command)
                        try:
                            code = child.wait(timeout=min(10, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            if time.monotonic() - started >= job['timeout_seconds']:
                                raise TimeoutError('Bounded job exceeded its deadline: ' + job['name'])
                if code:
                    raise RuntimeError('{} failed with exit code {}; see {}'.format(job['name'], code, log_path))
                state['completed_jobs'].append(dict(name=job['name'], seconds=time.monotonic() - started,
                                                    log=str(log_path), exit_code=code))
                child = None
            check_preserved(plan)
            if plan_path.read_bytes() != plan_bytes:
                raise RuntimeError('Approved plan changed while running')
            status('complete_needs_scientific_review', child_pid=None, preserved_inputs_verified=True)
        except BaseException as error:
            if child is not None:
                stop_owned_child(child)
            status('failed', child_pid=None, error=repr(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    # Explicit worker termination should reap only its own process-group child.
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Worker received signal {}'.format(signum))
    signal.signal(signal.SIGTERM, interrupted)
    worker(args.plan.resolve(), args.run_dir.resolve())


if __name__ == '__main__':
    main()
