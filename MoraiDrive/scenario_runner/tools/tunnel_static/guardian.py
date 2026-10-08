#!/usr/bin/env python3
"""Foreground supervisor for one frozen tunnel search CLI; no simulator calls.

Command file schema:
  {"command": ["/usr/bin/python3", "-m", "scenario_runner.tools.run_tunnel_static_search", ...],
   "campaign_root": "/absolute/output", "source_pins_path": "/absolute/pins.json"}

Run from a persistent unified exec session. Search CLI owns actor recovery and
restoration. A hard-killed or uncertain worker is never reported as cleaned up.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.runtime import (
    ExecutionLock, atomic_json, digest, process_identity, read_json, require_sources,
    signal_owned_group, bounded_run,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-file", type=Path, required=True)
    parser.add_argument("--deadline-sec", type=float, default=43200.)
    parser.add_argument("--cleanup-reserve-sec", type=float, default=240.)
    args = parser.parse_args()
    if not 60 < args.deadline_sec <= 43200 or not 30 <= args.cleanup_reserve_sec < args.deadline_sec:
        parser.error("Guardian deadline is at most twelve hours and reserves at least thirty seconds for cleanup")
    specification = read_json(args.command_file)
    command = specification.get("command")
    if not isinstance(command, list) or not command or any(not isinstance(part, str) for part in command):
        parser.error("Command file requires a complete argv array")
    root = Path(specification["campaign_root"]).resolve()
    pins_path = Path(specification["source_pins_path"]).resolve()
    expected_module = "scenario_runner.tools.run_tunnel_static_search"
    root.mkdir(parents=True, exist_ok=True)
    state = {"schema": "aimdrive_tunnel_search_guardian_v1", "status": "starting",
             "guardian_pid": os.getpid(), "started_unix": time.time(),
             "deadline_sec": args.deadline_sec, "cleanup_reserve_sec": args.cleanup_reserve_sec,
             "command_file": str(args.command_file.resolve()), "command_sha256": digest(args.command_file),
             "command": command, "campaign_root": str(root), "worker_identity": None,
             "program_makes_model_api_calls": False, "program_makes_simulator_calls": False,
             "cleanup_verified": False}
    worker = None
    identity = None
    stopping = False

    def received(signum, _frame):
        nonlocal stopping
        stopping = True
        state["received_signal"] = int(signum)

    def save():
        state["heartbeat_unix"] = time.time()
        atomic_json(root / "supervisor.json", state)

    def cleanup_is_verified():
        manifest = read_json(root / "search_manifest.json", {})
        state["worker_status"] = manifest.get("status")
        return (manifest.get("scene_restored") is True and manifest.get("inflight") is None
                and manifest.get("cleanup_verified") is True)

    def fallback_cleanup():
        # Root CLI re-proves/stops its durable owned inflight child while holding
        # the fixed global execution lock, before any simulator recovery call.
        if worker is not None and worker.poll() is None:
            raise RuntimeError("Fallback recovery cannot race a still-running worker")
        remaining = deadline - time.monotonic()
        if remaining < 65.:
            state["fallback_skipped"] = "Insufficient remaining hard-deadline budget"
            return False
        frozen = read_json(pins_path)
        if isinstance(frozen, dict) and "source_snapshot" in frozen:
            frozen = frozen["source_snapshot"]
        require_sources(frozen)
        recovery_command = [part for part in command if part not in ("--live", "--resume", "--recover-only")]
        recovery_command += ["--resume", "--recover-only"]
        def launched(fields):
            state["fallback_inflight"] = fields
            save()
        result = bounded_run(recovery_command, root / "guardian_cleanup.log",
            min(args.cleanup_reserve_sec, remaining - 35.), campaign_root=root,
            stage="guardian_recovery", expected_module=expected_module, on_start=launched, grace=30.)
        state["fallback_process"] = result
        state["fallback_inflight"] = None
        return cleanup_is_verified()

    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    for signum in handlers:
        signal.signal(signum, received)
    started = time.monotonic()
    deadline = started + args.deadline_sec
    interrupt_at = deadline - args.cleanup_reserve_sec
    interrupted = False
    forced = False
    result_code = 2
    guardian_lock = ExecutionLock(root / "guardian.lock")
    acquired = False
    try:
        guardian_lock.__enter__()
        acquired = True
        save()
        pins = read_json(pins_path)
        if isinstance(pins, dict) and "source_snapshot" in pins:
            pins = pins["source_snapshot"]
        require_sources(pins)
        state["source_pins_path"] = str(pins_path)
        state["source_pins_sha256"] = digest(pins_path)
        # Entry point/root constraints apply before launching any worker.
        module_launch = any(command[index:index + 2] == ["-m", expected_module] for index in range(len(command) - 1))
        wrapper = PROJECT / "scenario_runner/tools/tunnel_static/run_tunnel_static_search.sh"
        wrapper_launch = len(command) > 1 and Path(command[0]).name == "bash" and Path(command[1]).resolve() == wrapper
        if not (module_launch or wrapper_launch):
            raise RuntimeError("Guardian command must invoke the exact search CLI Python module")
        output_flags = ("--output", "--campaign-root")
        outputs = [command[index + 1] for index, value in enumerate(command[:-1]) if value in output_flags]
        if len(outputs) != 1 or Path(outputs[0]).resolve() != root:
            raise RuntimeError("Guardian command output must equal its campaign root")
        environment = dict(os.environ, AIMDRIVE_TUNNEL_BATCH_ROOT=str(root), PYTHONDONTWRITEBYTECODE="1",
                           PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1")
        with (root / "worker.log").open("ab") as log:
            worker = subprocess.Popen(command, cwd=str(PROJECT), env=environment, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            identity = process_identity(worker.pid, root, expected_module, launch_command=command)
            state.update(status="running", worker_pid=worker.pid, worker_identity=identity)
            save()
            previous_progress = None
            last_save = time.monotonic()
            while worker.poll() is None:
                now = time.monotonic()
                if not interrupted and (stopping or now >= interrupt_at):
                    if identity is None:
                        raise RuntimeError("Worker launch identity missing; no signal is authorized")
                    signal_owned_group(identity, signal.SIGINT)
                    interrupted = True
                    if stopping:
                        deadline = min(deadline, now + args.cleanup_reserve_sec)
                    state.update(status="stopping_for_cleanup", interrupted_unix=time.time())
                    save()
                if now >= deadline:
                    signal_owned_group(identity, signal.SIGKILL)
                    forced = True
                    state["status"] = "hard_deadline_reached"
                    break
                progress_path = root / "progress.json"
                progress = read_json(progress_path, {})
                displayed = {key: progress.get(key) for key in ("status", "actual_drives", "actual_passes", "actual_failures", "preflight_rejects", "generation_count", "nominal_passes")}
                displayed["stage"] = (progress.get("inflight") or {}).get("stage")
                key = json.dumps(displayed, sort_keys=True)
                if key != previous_progress:
                    print(json.dumps({"guardian_at": time.time(), "progress": displayed}), flush=True)
                    previous_progress = key
                if now - last_save >= 5.:
                    save()
                    last_save = now
                time.sleep(1.)
            state["worker_returncode"] = worker.wait(timeout=5.)
        state.update(forced_kill=forced, interrupted=interrupted)
        # Only the search coordinator's durable verified cleanup is evidence.
        manifest_path = root / "search_manifest.json"
        cleanup_verified = cleanup_is_verified()
        if not cleanup_verified:
            cleanup_verified = fallback_cleanup()
        state["cleanup_verified"] = cleanup_verified
        state["worker_manifest_path"] = str(manifest_path)
        if state["worker_returncode"] == 0 and cleanup_verified and not forced and state["worker_status"] == "completed":
            state["status"] = "completed"
            result_code = 0
        else:
            state["status"] = "stopped_cleanup_verified" if cleanup_verified else "stopped_cleanup_unverified"
            result_code = 1
    except BaseException as error:
        state.update(status="failed", failure_reason=str(error) or type(error).__name__)
        if worker is not None and worker.poll() is None and identity is not None:
            try:
                signal_owned_group(identity, signal.SIGINT)
                cleanup_until = min(deadline, time.monotonic() + args.cleanup_reserve_sec)
                while worker.poll() is None and time.monotonic() < cleanup_until:
                    time.sleep(.2)
                if worker.poll() is None:
                    signal_owned_group(identity, signal.SIGKILL)
                    state["forced_kill"] = True
                state["worker_returncode"] = worker.wait(timeout=5.)
            except BaseException as shutdown_error:
                state["shutdown_failure"] = str(shutdown_error)
        if acquired and worker is not None and worker.poll() is not None:
            try:
                state["cleanup_verified"] = cleanup_is_verified() or fallback_cleanup()
            except BaseException as recovery_error:
                state["fallback_failure"] = str(recovery_error)
    finally:
        state["finished_unix"] = time.time()
        state["elapsed_sec"] = time.monotonic() - started
        if acquired:
            save()
            guardian_lock.__exit__(None, None, None)
        for signum, previous in handlers.items():
            signal.signal(signum, previous)
        print(json.dumps(state, indent=2), flush=True)
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
