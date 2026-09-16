"""Build disk-backed export inventories and launch the Windows NAS worker."""
import datetime
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import sqlite3
import subprocess
import time

ROOT = Path(__file__).resolve().parent
SENSORS = ("CAMERA_1", "CAMERA_2", "CAMERA_3", "CAMERA_4",
           "LIDAR_5", "GPS_6", "IMU_7")
RESERVED = {".NAS_TRANSFER_ID.json", ".NAS_TRANSFER_COMPLETE.json"}


def component(value):
    if (not isinstance(value, str) or not value or value in (".", "..")
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', value)
            or value.endswith((" ", "."))):
        raise ValueError("Unsafe export name: " + repr(value))
    if value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL"} | {
            p + str(i) for p in ("COM", "LPT") for i in range(1, 10)}:
        raise ValueError("Reserved Windows name: " + value)
    return value


def campaign_folder(plan):
    name = component(plan["campaign_id"])
    match = re.search(r"_(\d{8})_(\d{6})(?:_|$)", name)
    if match:
        stamp = match.group(2)
    else:
        stamp = datetime.datetime.fromtimestamp(
            int(plan["created_wall_ns"]) / 1e9).strftime("%H%M%S")
    return "full loop_" + stamp


def validate_target(value):
    target = PureWindowsPath(value)
    if (not target.is_absolute() or str(target) == target.anchor
            or "'" in str(target) or any(c in str(target) for c in "\r\n")):
        raise ValueError("NAS root must be an absolute Windows folder or UNC share subfolder")
    for part in target.parts[1:]:
        component(part)
    return str(target)


def roots_for_run(base, name):
    component(name)
    base = Path(base).resolve()
    run = base / name
    if run.is_symlink() or not run.is_dir():
        raise ValueError("Missing real run directory: " + str(run))
    setup = json.loads((run / "native_setup.json").read_text(encoding="utf-8"))
    native = Path(setup["native_root"])
    expected = base / "native_sessions" / name
    if native.is_symlink() or native.resolve() != expected.resolve() or not native.is_dir():
        raise ValueError("Native root does not match the recorded lap")
    for sensor in SENSORS:
        if not (native / sensor).is_dir() or (native / sensor).is_symlink():
            raise ValueError("Missing/linked native sensor directory: " + sensor)
    if not (native / "SESSION.json").is_file():
        raise ValueError("Native SESSION.json is missing")
    # Session setup evidence is generated beside the run, not inside it.
    # Select only this lap's exact filenames; never walk the shared base.
    session_files = tuple(base / (name + "_native_session_" + kind + ".json")
                          for kind in ("apply", "plan"))
    for source in session_files:
        if source.is_symlink() or not source.is_file():
            raise ValueError("Missing/linked native session evidence: " + str(source))
    return (run, native) + session_files


def inventory(roots, path, translate):
    """Stream directory entries; disk-backed sorting keeps RAM independent of file count."""
    path = Path(path)
    count = total = 0

    def walk(root):
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ValueError("Export refuses symbolic links: " + entry.path)
                yield Path(entry.path), entry.is_dir(follow_symlinks=False)
                if entry.is_dir(follow_symlinks=False):
                    yield from walk(entry.path)

    with sqlite3.connect(str(path.with_suffix(".sqlite"))) as db:
        db.execute("DROP TABLE IF EXISTS names")
        db.execute("CREATE TABLE names (name TEXT PRIMARY KEY, payload TEXT)")
        db.execute("PRAGMA temp_store=FILE")
        for root in roots:
            root = Path(root)
            if any(p.is_symlink() for p in (root, *root.parents)):
                raise ValueError("Linked source ancestor")
            single_file = root.is_file()
            if not single_file and not root.is_dir():
                raise ValueError("Missing/unsupported export source: " + str(root))
            entries = ((root, False),) if single_file else walk(root)
            for source, is_dir in entries:
                component(source.name)
                relative = source.name if single_file else source.relative_to(root).as_posix()
                first = relative.split("/")[0].upper()
                if first.startswith((".NAS-", ".NAS_TRANSFER_")) or first in {x.upper() for x in RESERVED}:
                    raise ValueError("Source uses reserved publication metadata")
                stat = source.stat()
                if not is_dir and not source.is_file():
                    raise ValueError("Only regular files/directories can be exported")
                payload = json.dumps(dict(source=translate(source), relative=relative,
                    kind="directory" if is_dir else "file",
                    size=0 if is_dir else stat.st_size, mtime_ns=stat.st_mtime_ns),
                    ensure_ascii=False, sort_keys=True)
                try:
                    db.execute("INSERT INTO names VALUES (?, ?)", (relative.casefold(), payload))
                except sqlite3.IntegrityError:
                    raise ValueError("Colliding export paths: " + relative)
                if not is_dir:
                    count += 1
                    total += stat.st_size
        with path.open("w", encoding="utf-8") as out:
            for (payload,) in db.execute("SELECT payload FROM names ORDER BY name"):
                out.write(payload + "\n")
    return count, total


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_job(base, run_name, plan, target_root, state_root, translate):
    roots = roots_for_run(base, run_name)
    state = Path(state_root) / component(run_name)
    state.mkdir(parents=True, exist_ok=True)
    draft = state / "inventory.new.jsonl"
    count, size = inventory(roots, draft, translate)
    fingerprint = sha256(draft)
    manifest = state / ("inventory-" + fingerprint + ".jsonl")
    os.replace(draft, manifest)
    destination = str(PureWindowsPath(validate_target(target_root),
                                      campaign_folder(plan), run_name))
    identity = hashlib.sha256((destination.casefold() + "\n" + fingerprint).encode()).hexdigest()
    job = dict(schema="morai_nas_transfer_v1", identity=identity,
               destination=destination, sources=[translate(p) for p in roots],
               inventory=translate(manifest), inventory_sha256=fingerprint,
               state=translate(state / identity), files=count, bytes=size,
               campaign_id=plan["campaign_id"], run_name=run_name)
    job_path = state / ("job-" + identity + ".json")
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    return job_path


def publish_validated(report_path, target_root=None, dry_run=False):
    from runtime_paths import LOCAL_MACHINE, POWERSHELL_EXE, windows_path_from_wsl
    from ssd_storage import SSD_ROOT
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    seal = json.loads((report_path.parent / "COMPLETE.json").read_text(encoding="utf-8"))
    if (report.get("valid") is not True or seal.get("status") != "passed"
            or seal.get("files", {}).get("report.json") != sha256(report_path)):
        raise ValueError("NAS publication requires a passed, sealed campaign report")
    campaign = Path(report["campaign"]).resolve()
    plan_path = campaign / "campaign_plan.json"
    if report.get("campaign_plan_sha256") != sha256(plan_path):
        raise ValueError("Campaign plan changed after validation")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if report.get("runs_ledger_sha256") != sha256(campaign / "runs.tsv"):
        raise ValueError("Run ledger changed after validation")
    if report.get("collection_completion_sha256") != sha256(campaign / "COLLECTION_COMPLETE.json"):
        raise ValueError("Collection completion changed after validation")
    if len(report["laps"]) != int(plan["requested_laps"]) or not report["laps"]:
        raise ValueError("Validation does not cover the complete campaign")
    target_root = (target_root or os.environ.get("MORAI_NAS_ROOT")
                   or LOCAL_MACHINE.get("nas_root"))
    if not target_root:
        raise ValueError("Set --nas-root, MORAI_NAS_ROOT or local_machine.json nas_root; "
                         "use --no-nas-copy for local validation only")
    validate_target(target_root)
    state_root = SSD_ROOT / "sensor_record_auto_v1" / ".nas_transfers"
    state_root.mkdir(parents=True, exist_ok=True)
    worker_source = ROOT / "tools" / "publish_sensor_record_nas.ps1"
    worker = state_root / ("worker-" + sha256(worker_source)[:16] + ".ps1")
    if not worker.exists():
        shutil.copyfile(worker_source, worker)
    results = []
    for lap in report["laps"]:
        if lap.get("valid") is not True:
            raise ValueError("Invalid lap cannot be published")
        job_path = make_job(SSD_ROOT / "sensor_record_auto_v1", lap["run_name"],
                            plan, target_root, state_root, windows_path_from_wsl)
        job = json.loads(job_path.read_text(encoding="utf-8"))
        print("[NAS] %s: %d files, %.2f GiB -> %s%s" % (
            lap["run_name"], job["files"], job["bytes"] / 2**30,
            job["destination"], " (plan only; destination untouched)" if dry_run else ""),
            flush=True)
        if dry_run:
            results.append(dict(run_name=lap["run_name"], planned=True, job=str(job_path)))
            continue
        state = job_path.parent / job["identity"]
        state.mkdir(exist_ok=True)
        lease = state / "parent.lease"
        lease.write_text("active", encoding="ascii")
        worker_started = time.time()
        process = subprocess.Popen([
            str(POWERSHELL_EXE), "-NoProfile", "-NonInteractive",
            "-ExecutionPolicy", "Bypass", "-File", windows_path_from_wsl(worker),
            "-JobFile", windows_path_from_wsl(job_path),
        ])
        try:
            while process.poll() is None:
                lease.touch()
                time.sleep(1)
            code = process.returncode
        except BaseException:
            lease.unlink(missing_ok=True)
            # Keep the capture locks until the Windows worker acknowledges exit.
            # Its native watchdog also terminates it if Windows I/O is stuck.
            process.wait()
            raise
        finally:
            lease.unlink(missing_ok=True)
        status_path = state / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        watchdog = {}
        failure_path = state / "watchdog_failure.json"
        if failure_path.exists() and failure_path.stat().st_mtime >= worker_started:
            try:
                watchdog = json.loads(failure_path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                pass
        exit_record = dict(
            started_utc=datetime.datetime.fromtimestamp(worker_started, datetime.timezone.utc).isoformat(),
            ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            exit_code=code, last_worker_status=status, watchdog=watchdog,
        )
        exit_path = state / "worker_exit.json"
        temp_exit = state / "worker_exit.json.tmp"
        temp_exit.write_text(json.dumps(exit_record, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_exit, exit_path)
        if code or status.get("status") != "copy_and_readback_verified":
            reason = watchdog.get("reason") or status.get("reason") or "worker exited without a completion marker"
            detail = ""
            if watchdog:
                if watchdog.get("operation") == "idle" and watchdog.get("last_completed_operation"):
                    detail = " last_completed_operation=%s path=%s duration=%.1fs; no I/O active at exit;" % (
                        watchdog["last_completed_operation"], watchdog.get("last_completed_path", "?"),
                        watchdog.get("last_completed_elapsed_ms", 0) / 1000.0,
                    )
                else:
                    detail = " operation=%s path=%s waited=%.1fs;" % (
                        watchdog.get("operation", "?"), watchdog.get("path", "?"),
                        watchdog.get("operation_elapsed_ms", 0) / 1000.0,
                    )
            raise RuntimeError("NAS transfer stopped (exit %d): %s;%s local data retained. "
                               "Details: %s. Rerun 03 or use --nas-only with this sealed report."
                               % (code, reason, detail, exit_path))
        results.append(dict(run_name=lap["run_name"], verified_readback=True,
                            destination=job["destination"], remote_durability_verified=False))
    return results
