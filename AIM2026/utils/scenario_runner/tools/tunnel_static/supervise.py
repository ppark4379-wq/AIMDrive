"""Resume a frozen tunnel search after narrowly audited clean preflight rejects.

The original producer sources, failure proofs, approval rules and counters stay
unchanged. Only the separate review tool can exclude a proven unstarted road
margin case. All other failures stop this supervisor. No model API is used.
"""

# Resolve this relocated package before importing its shared dependencies.
import sys as _runner_sys
from pathlib import Path as _RunnerPath
for _runner_ancestor in _RunnerPath(__file__).resolve().parents:
    if (_runner_ancestor / "project_paths.py").is_file():
        _runner_sys.path.insert(0, str(_runner_ancestor.parent))
        break
from scenario_runner.project_paths import AIM_ROOT, activate_project_paths
activate_project_paths()


import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ...scenarios.full_loop_parts.tunnel_static_parts import runtime
from .review import review_campaign


PROJECT = AIM_ROOT
GUARDIAN_MODULE = "scenario_runner.tools.tunnel_static_search_guardian"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--max-reviewed-exclusions", type=int, default=8)
    parser.add_argument("--supervisor-pin-file", type=Path,
                        help="New revision pin file within the campaign; existing pins are never refreshed")
    args = parser.parse_args()
    if not 1 <= args.max_reviewed_exclusions <= 8:
        parser.error("Reviewed exclusions are bounded to eight")
    root = args.campaign_root.resolve()
    state_path = root / "review_supervisor.json"
    manifest = runtime.read_json(root / "search_manifest.json")
    if not isinstance(manifest, dict):
        parser.error("An existing frozen search journal is required")
    specification = root / "launch_spec.json"
    if not specification.is_file():
        parser.error("The original exact launch specification is required")
    # Include time spent stopped/reviewed in the original twelve-hour budget.
    hard_deadline = float(manifest["created_unix"]) + 43200.
    own_paths = [Path(__file__), Path(__file__).with_name("guardian.py"),
                 Path(__file__).with_name("review.py"), specification]
    pin_path = (args.supervisor_pin_file.resolve() if args.supervisor_pin_file
                else root / "review_supervisor_source_pins.json")
    if pin_path.parent != root:
        parser.error("Supervisor pin revisions must stay within the campaign directory")
    own_pins = {str(path.resolve()): runtime.digest(path) for path in own_paths}
    prior_pins = runtime.read_json(pin_path)
    if prior_pins is None:
        runtime.atomic_json(pin_path, own_pins)
    elif prior_pins != own_pins:
        parser.error("Reviewed supervisor sources changed; refuse to refresh pins")
    state = dict(schema="aimdrive_tunnel_review_supervisor_v1", status="starting",
                 started_unix=time.time(), hard_deadline_unix=hard_deadline,
                 max_reviewed_exclusions=args.max_reviewed_exclusions,
                 source_pins_path=str(pin_path), source_pins_sha256=runtime.digest(pin_path),
                 inflight=None, cycles=[], program_makes_model_api_calls=False)
    stopping = False
    child = None
    identity = None

    def save():
        state["heartbeat_unix"] = time.time()
        runtime.atomic_json(state_path, state)

    def receive(signum, _frame):
        nonlocal stopping
        stopping = True
        state["received_signal"] = int(signum)

    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    for signum in handlers:
        signal.signal(signum, receive)
    exit_code = 1
    try:
        with runtime.ExecutionLock(root / "review_supervisor.lock"):
            while not stopping:
                if own_pins != {str(path.resolve()): runtime.digest(path) for path in own_paths}:
                    raise RuntimeError("Frozen reviewed supervisor or launch specification changed")
                manifest = runtime.read_json(root / "search_manifest.json")
                if manifest.get("status") == "completed":
                    if not (manifest.get("actual_drives") == 100 and manifest.get("cleanup_verified") is True
                            and manifest.get("scene_restored") is True and manifest.get("inflight") is None):
                        raise RuntimeError("Completed journal lacks the exact 100-run goal or verified cleanup")
                    state["status"] = "completed"
                    exit_code = 0
                    break
                if len(state["cycles"]) >= args.max_reviewed_exclusions:
                    raise RuntimeError("Bounded guarded restart budget exhausted")
                review = review_campaign(root, apply=True, max_reviews=args.max_reviewed_exclusions)
                if state["cycles"] and (review.get("status") != "applied" or any(
                        item["review"].get("candidate_id") == review.get("candidate_id")
                        for item in state["cycles"])):
                    raise RuntimeError("Guardian failure lacks a newly reviewed terminal preflight; refuse to retry")
                state["cycles"].append(dict(review=review, reviewed_unix=time.time()))
                remaining = hard_deadline - time.time()
                if remaining <= 480.:
                    raise RuntimeError("Insufficient original deadline budget for a guarded restart")
                command = [sys.executable, "-u", "-m", GUARDIAN_MODULE,
                           "--command-file", str(specification), "--deadline-sec", str(min(43200., remaining)),
                           "--cleanup-reserve-sec", "240"]
                environment = dict(os.environ, AIMDRIVE_TUNNEL_BATCH_ROOT=str(root),
                                   PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
                log_path = root / ("reviewed_guardian_%02d.log" % len(state["cycles"]))
                with log_path.open("ab") as log:
                    child = subprocess.Popen(command, cwd=str(PROJECT), env=environment,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    identity = runtime.process_identity(child.pid, root, GUARDIAN_MODULE, launch_command=command)
                    if identity is None:
                        raise RuntimeError("Guardian exited before its launch identity was captured")
                    state.update(status="running", inflight=identity)
                    save()
                    last_save = time.monotonic()
                    interrupted = False
                    while child.poll() is None:
                        if stopping and not interrupted:
                            runtime.signal_owned_group(identity, signal.SIGINT)
                            interrupted = True
                        if time.monotonic() - last_save >= 5.:
                            save()
                            last_save = time.monotonic()
                        time.sleep(.2)
                    code = child.wait()
                state["cycles"][-1].update(guardian_returncode=code, finished_unix=time.time())
                state["inflight"] = None
                child, identity = None, None
                save()
                # A failed guardian is examined afresh by the strict reviewer
                # next iteration; it never enables a general retry policy.
            if stopping:
                state["status"] = "stopped_by_request"
    except BaseException as error:
        state.update(status="blocked", failure_reason=str(error) or type(error).__name__)
        if child is not None and child.poll() is None and identity is not None:
            runtime.signal_owned_group(identity, signal.SIGINT)
            # The inner guardian owns its bounded recovery and restoration.
            # Do not kill it early while a separate Expert process is stopping.
            child.wait()
    finally:
        state["finished_unix"] = time.time()
        save()
        for signum, prior in handlers.items():
            signal.signal(signum, prior)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
