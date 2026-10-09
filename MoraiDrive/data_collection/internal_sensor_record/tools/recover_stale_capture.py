#!/usr/bin/env python3
"""Recover only known capture processes from one interrupted/failed local run."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import datetime
import signal
import subprocess
import sys
import time



def recovery_reference(executable, integration):
    """Only recognize these capture entrypoints in this exact checkout."""
    kinds = {
        "run_async_scenario_review.py": "scenario_runner",
        "run_proven_three_lap_scenario.py": "scenario_runner",
        "collect_async_raw.py": "raw_recorder",
        "collect_async_raw_proven.py": "raw_recorder",
    }
    executable = Path(executable).resolve()
    kind = kinds.get(executable.name)
    if kind is None:
        return None, None
    reference = (Path(integration) / "tools" / executable.name).resolve()
    if executable != reference or not reference.is_file():
        return None, None
    return reference, kind


def process_identity(pid):
    """Start time and command line prevent signaling a reused PID."""
    proc = Path("/proc") / str(pid)
    try:
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "start_ticks": fields[19],
            "cmdline_sha256": hashlib.sha256((proc / "cmdline").read_bytes()).hexdigest(),
        }
    except (OSError, IndexError):
        return None



def process_snapshot(proc_root=Path("/proc")):
    """Read process identity and parentage; ignore exited/zombie processes."""
    result = {}
    for proc in Path(proc_root).glob("[0-9]*"):
        try:
            raw = (proc / "cmdline").read_bytes()
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            status = (proc / "status").read_text().splitlines()
            if not raw or fields[0] == "Z":
                continue
            uid = int(next(line for line in status if line.startswith("Uid:")).split()[1])
            result[int(proc.name)] = dict(
                ppid=int(fields[1]), uid=uid,
                args=[value.decode(errors="surrogateescape") for value in raw.split(b"\0") if value],
                identity=dict(start_ticks=fields[19],
                              cmdline_sha256=hashlib.sha256(raw).hexdigest()))
        except (OSError, ValueError, IndexError, StopIteration):
            continue
    return result


def argument_value(args, name):
    for index, value in enumerate(args):
        if value == name:
            return args[index + 1] if index + 1 < len(args) else None
        if value.startswith(name + "="):
            return value[len(name) + 1:]
    return None


def recovery_run_location(run, base):
    """Accept legacy runs or a direct staging run owned by its campaign plan."""
    base, run = Path(base).resolve(), Path(run).resolve()
    if run.parent == base:
        return run
    if base.name != "sensor_record_auto_v1":
        return None
    staging = base.parent / "dataset" / ".collecting"
    try:
        parts = run.relative_to(staging).parts
        if len(parts) != 3 or run.name == "_campaign":
            return None
        plan = json.loads((run.parent / "_campaign/campaign_plan.json").read_text())
        if (plan.get("schema") != "morai_sensor_record_campaign_v1"
                or plan.get("campaign_id") != parts[1]
                or plan.get("dataset_scenario_folder") != parts[0]):
            return None
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return run


def campaign_ledgers(base):
    base = Path(base).resolve()
    for ledger in (base / "campaigns").glob("*/runs.tsv"):
        yield ledger, base
    if base.name == "sensor_record_auto_v1":
        for ledger in (base.parent / "dataset/.collecting").glob("*/*/_campaign/runs.tsv"):
            yield ledger, ledger.parent.parent


def orphaned_stop_evidence(row, base, processes):
    """A started ledger alone never authorizes stopping a recorder."""
    if (row.get("status") != "started" or row.get("kind") != "raw_recorder"
            or not row.get("safe_code_identity") or not row.get("run")):
        return None
    run = recovery_run_location(row.get("run_root", Path(base) / row["run"]), base)
    if run is None or run.name != row["run"]:
        return None
    child = processes.get(row["pid"])
    if not child or child["identity"] != row.get("process_identity"):
        return None
    parent = processes.get(child["ppid"])
    # Linux PID 1 or WSL's root-owned /init has adopted this child.
    adopted = bool(parent and parent["uid"] == 0
                   and (child["ppid"] == 1 or parent["args"] == ["/init"]))
    if not adopted:
        return None
    # A launcher may still be cleaning up after its runner exited.
    for pid, process in processes.items():
        if pid == row["pid"]:
            continue
        output = argument_value(process["args"], "--output")
        if (argument_value(process["args"], "--name") == run.name
                or (output and Path(output).resolve() == run)):
            return None
    request = run / "episode_000/sensor_record_coordination/stop.request"
    try:
        if request.is_symlink() or request.stat().st_size > 16384:
            return None
        payload = request.read_bytes()
        record = json.loads(payload)
        when = int(record["request_wall_ns"])
        if record.get("reason") != "campaign_interrupted" or not 0 < when <= time.time_ns():
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return dict(request=str(request), request_sha256=hashlib.sha256(payload).hexdigest(),
                reason=record["reason"], request_wall_ns=when,
                adopted_by_pid=child["ppid"], parent_identity=parent["identity"])


def append_recovered_interruption(ledger, run_name):
    """Append an observed recovery outcome; preserve the original started row."""
    with Path(ledger).open("r+", newline="", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames
        latest = None
        for row in reader:
            if row.get("run_name") == run_name:
                latest = row
        if latest is None or latest.get("status") != "started":
            return False
        latest["status"] = "interrupted"
        latest["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        stream.seek(0, os.SEEK_END)
        csv.DictWriter(stream, fields, delimiter="\t", lineterminator="\n").writerow(latest)
        stream.flush()
        os.fsync(stream.fileno())
    return True


def partition_candidates(candidates):
    stale = [
        row for row in candidates
        if row.get("safe_code_identity") and row.get("process_identity")
        and (row.get("status") == "interrupted"
             or str(row.get("status")).startswith("failed_")
             or (row.get("status") == "started" and row.get("orphaned_stop_evidence")))
    ]
    return stale, [row for row in candidates if row not in stale]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--simulator-pid", type=int, required=True)
    parser.add_argument("--check", action="store_true", help="Inspect recovery eligibility only")
    args = parser.parse_args()
    integration = Path(__file__).resolve().parents[1]
    base = args.base.resolve()
    simulator_pid = args.simulator_pid
    check_only = args.check
    lock_path = base / ".locks" / "collector_capture.lock"
    campaign_lock_path = base / ".locks" / "collector_campaign.lock"

    def lock_is_free(path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return False
                return True
        except OSError as error:
            print("[FAIL] capture lock 저장소를 사용할 수 없습니다: %s: %s" %
                  (path, error), file=sys.stderr)
            raise SystemExit(2)

    def locks_are_free():
        return lock_is_free(lock_path) and lock_is_free(campaign_lock_path)

    if locks_are_free():
        raise SystemExit(0)

    terminal = {}
    ledger_paths = {}
    for ledger, run_parent in campaign_ledgers(base):
        try:
            with ledger.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    run = recovery_run_location(run_parent / row["run_name"], base)
                    if run is not None:
                        terminal[str(run)] = row["status"]
                        ledger_paths[str(run)] = ledger
        except (OSError, KeyError):
            continue

    lock_identities = {}
    for path in (lock_path, campaign_lock_path):
        stat = path.stat()
        lock_identities[(stat.st_dev, stat.st_ino)] = path.name

    candidates = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
            args = [value.decode(errors="surrogateescape") for value in args if value]
        except (OSError, PermissionError):
            continue
        held_locks = []
        try:
            for descriptor in (proc / "fd").iterdir():
                try:
                    stat = descriptor.stat()
                except (OSError, PermissionError):
                    continue
                name = lock_identities.get((stat.st_dev, stat.st_ino))
                if name:
                    held_locks.append(name)
        except (OSError, PermissionError):
            continue
        if not held_locks:
            continue
        script_arg = next((value for value in args[1:] if value.endswith(".py")), None)
        if script_arg is None:
            candidates.append({"pid": int(proc.name), "kind": "unknown",
                               "held_locks": sorted(held_locks), "args": args[:8],
                               "safe_code_identity": False})
            continue
        try:
            executable = Path(script_arg).resolve()
        except OSError:
            executable = Path(script_arg)
        reference, known_kind = recovery_reference(executable, integration)
        same_code = reference is not None
        if reference is None or "--output" not in args:
            candidates.append({"pid": int(proc.name), "kind": "unknown",
                               "held_locks": sorted(held_locks), "args": args[:8],
                               "script": str(executable),
                               "safe_code_identity": same_code})
            continue
        try:
            output = Path(args[args.index("--output") + 1]).resolve()
        except (IndexError, OSError):
            candidates.append({"pid": int(proc.name), "kind": "unknown",
                               "held_locks": sorted(held_locks), "args": args[:8],
                               "script": str(executable),
                               "safe_code_identity": same_code})
            continue
        if known_kind == "scenario_runner":
            run = output
            kind = "scenario_runner"
        else:
            # collect_async_raw output is <base>/<run>/episode_000/raw.
            if output.name != "raw" or output.parent.name != "episode_000":
                candidates.append({"pid": int(proc.name), "kind": "unknown",
                                   "held_locks": sorted(held_locks),
                                   "safe_code_identity": False})
                continue
            run = output.parent.parent
            kind = "raw_recorder"
        if recovery_run_location(run, base) is None:
            candidates.append({"pid": int(proc.name), "kind": kind,
                               "held_locks": sorted(held_locks), "args": args[:8],
                               "script": str(executable), "run": run.name,
                               "output": str(output), "status": "different_storage_root",
                               "safe_code_identity": same_code})
            continue
        status = terminal.get(str(run))
        candidates.append({"pid": int(proc.name), "kind": kind,
                           "run": run.name, "output": str(output), "status": status})
        candidates[-1].update(held_locks=sorted(held_locks), run_root=str(run),
                              script=str(executable), safe_code_identity=same_code)

    for row in candidates:
        row["process_identity"] = process_identity(row["pid"])
    processes = process_snapshot()
    for row in candidates:
        proof = orphaned_stop_evidence(row, base, processes)
        if proof:
            row["orphaned_stop_evidence"] = proof
    stale, active = partition_candidates(candidates)
    if active or not stale or len({row["run_root"] for row in stale}) != 1:
        print("[FAIL] 다른 수집 프로세스가 capture lock을 사용 중입니다.", file=sys.stderr)
        for row in candidates:
            print("       PID=%s kind=%s run=%s status=%s script=%s" % (
                row.get("pid"), row.get("kind"), row.get("run", "?"),
                row.get("status", "?"), row.get("script", "?")), file=sys.stderr)
        print("       실행 중인 02번 터미널이 있으면 그 실행을 먼저 정상 종료하세요.",
              file=sys.stderr)
        print(json.dumps({"error": "capture_lock_owner_not_safely_recoverable",
                          "candidates": candidates}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)

    if check_only:
        print(json.dumps({"recoverable": True, "candidates": stale}, ensure_ascii=False))
        return

    run_name = stale[0]["run"]
    recovery_name = "stale_capture_recovery_" + str(time.time_ns())
    processes = process_snapshot()
    for row in stale:
        if row.get("orphaned_stop_evidence"):
            if orphaned_stop_evidence(row, base, processes) != row["orphaned_stop_evidence"]:
                raise RuntimeError("Orphaned capture evidence changed; recovery cancelled")
    stop = subprocess.run(
        [sys.executable, str(integration / "tools" / "recover_sensor_record_stop.py"),
         "--pid", str(simulator_pid), "--name", recovery_name],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=45,
    )
    if stop.returncode:
        print(stop.stderr, file=sys.stderr)
        raise SystemExit(2)

    stop_result = json.loads(stop.stdout)
    if stop_result.get("status") != "off_verified":
        raise RuntimeError("Sensor Record OFF was not verified; no process signal sent")
    # The UI stop can take seconds. Recheck identity and orphan evidence.
    processes = process_snapshot()
    for row in stale:
        current = process_identity(row["pid"])
        if current is None:
            continue
        if current != row["process_identity"]:
            raise RuntimeError("Capture PID identity changed; no signal sent to that process")
        if row.get("orphaned_stop_evidence"):
            if orphaned_stop_evidence(row, base, processes) != row["orphaned_stop_evidence"]:
                raise RuntimeError("Capture is no longer an abandoned recorder; no signal sent")
        try:
            os.kill(row["pid"], signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 90
    print("[RECOVERY] Sensor Record OFF 확인; 중단 수집기의 종료와 잠금 해제를 기다립니다.", flush=True)
    while time.monotonic() < deadline and not locks_are_free():
        time.sleep(0.2)
    if not locks_are_free():
        print(json.dumps({"error": "stale_capture_did_not_release_locks",
                          "candidates": stale}),
              file=sys.stderr)
        raise SystemExit(2)

    ledger_updated = False
    if any(row.get("orphaned_stop_evidence") for row in stale):
        ledger_updated = append_recovered_interruption(ledger_paths[stale[0]["run_root"]], run_name)
    report = {"schema": "morai_stale_capture_recovery_v1", "run": run_name,
              "processes": stale, "interruption_appended_to_ledger": ledger_updated,
              "sensor_record_stop": stop_result,
              "signal": "SIGINT", "lock_released": True,
              "recovered_wall_ns": time.time_ns(), "training_visible": False}
    report_path = base / (recovery_name + ".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")
    print("[RECOVERED] 이전 중단 run의 capture/campaign lock을 정상 해제했습니다: " + run_name)


if __name__ == "__main__":
    main()
