"""Supervise official data preparation, full GPU training and validation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model/output"
PYTHON = ROOT / ".venv-gpu/bin/python"
ENV = dict(os.environ, TF_USE_LEGACY_KERAS="1", OMP_NUM_THREADS="4",
           TF_NUM_INTRAOP_THREADS="4", TF_NUM_INTEROP_THREADS="2",
           CAM2BEV_GPU_MEMORY_MB="6144", TF_CUDNN_WORKSPACE_LIMIT_IN_MB="512")
cuda_paths = sorted(str(path) for path in
                    (ROOT / ".venv-gpu/lib").glob("python*/site-packages/nvidia/*/lib"))
ENV["LD_LIBRARY_PATH"] = ":".join(cuda_paths + ["/usr/lib/wsl/lib", str(ROOT / ".venv-gpu/lib"),
                                               ENV.get("LD_LIBRARY_PATH", "")])
STATUS = OUT / "full-training-status.json"


def state(stage, **fields):
    document = dict(stage=stage, pid=os.getpid(), updated_at=time.strftime("%Y-%m-%d %H:%M:%S"), **fields)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2))
    temporary.replace(STATUS)
    print(document, flush=True)


def run(command, log, cwd=ROOT):
    with (OUT / log).open("a") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
        print("Child PID:", process.pid, "Log:", log, flush=True)
        code = process.wait()
    if code:
        raise RuntimeError("Command failed (exit " + str(code) + "); see " + log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-download-pid", type=int)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    lock = (OUT / "full-training.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A full training supervisor is already running")
    try:
        state("downloading")
        if args.wait_download_pid:
            command = Path("/proc") / str(args.wait_download_pid) / "cmdline"
            while command.exists():
                try:
                    if b"scripts/download_1_frlr.py" not in command.read_bytes():
                        break
                except FileNotFoundError:
                    break
                time.sleep(10)
        run([sys.executable, "-u", str(ROOT / "scripts/download_1_frlr.py")], "full-data-download.log")
        state("validating_gpu")
        run([str(PYTHON), "-u", str(ROOT / "scripts/check_training_gpu.py")], "gpu-preflight.log")
        state("training")
        run([str(PYTHON), "-u", "train.py", "-c", "config.1_FRLR.full.yml"],
            "full-training.log", ROOT / "model")
        runs = sorted((OUT / "full").iterdir())
        latest = runs[-1]
        weights = latest / "Checkpoints/best_road_weights.hdf5"
        if not weights.is_file():
            raise RuntimeError("Missing road checkpoint after training")
        state("evaluating", run_directory=str(latest))
        run([str(PYTHON), "-u", "evaluate.py", "-c", "config.1_FRLR.full.yml",
             "--model-weights", str(weights)], "full-evaluation.log", ROOT / "model")
        state("completed", run_directory=str(latest), weights=str(weights),
              limitation="Official dataset validation only; MORAI validation is still required.")
    except BaseException as exc:
        state("failed", error=str(exc))
        traceback.print_exc()
        raise
