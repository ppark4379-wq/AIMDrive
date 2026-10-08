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
LOCAL_MARKER = ".LOCAL_DATASET_COMPLETE.json"
SCENARIO_FOLDERS = {
    "full_loop": "full_route",
    "full_loop_normal": "basic_driving",
    "full_loop_dynamic": "dynamic_obstacle",
    "full_loop_static": "static_obstacle",
    "full_loop_tunnel_static": "tunnel_static_obstacle",
    "full_loop_roundabout": "roundabout",
    "full_loop_roundabout_nonuniform": "roundabout_nonuniform",
    "full_loop_roundabout_speed_random": "roundabout_speed_random",
    "full_loop_roundabout_brake": "roundabout_brake",
    "full_loop_highway": "highway",
    "full_loop_tunnel_dynamic": "tunnel_dynamic",
    "full_loop_tunnel_vehicle": "tunnel_vehicle",
    # Reserved for the dedicated mode when the Scenario Runner adds it.
    "full_loop_traffic_light": "traffic_light",
}


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


def scenario_folder(plan):
    """Return the stable local/NAS category for one Scenario Runner mode."""
    scenario = plan.get("scenario_name", "full_loop")
    if scenario not in SCENARIO_FOLDERS:
        raise ValueError("Unsupported dataset scenario: " + repr(scenario))
    return component(SCENARIO_FOLDERS[scenario])


def dataset_campaign_id(plan):
    """Public dataset name; raw campaign identity remains immutable evidence."""
    return component(plan.get("dataset_campaign_id", plan["campaign_id"]))


def dataset_run_name(plan, lap):
    """Stable public run name derived from the public campaign and lap number."""
    source_name = component(lap["run_name"])
    number = lap.get("lap")
    if number is None:
        match = re.search(r"_lap_(\d+)$", source_name)
        if match:
            number = int(match.group(1))
    if number is None:
        return source_name
    number = int(number)
    if number < 1:
        raise ValueError("Invalid lap number: " + repr(number))
    return "%s_lap_%03d" % (dataset_campaign_id(plan), number)


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


def inventory_mapped(specs, path, translate, excluded_names=()):
    """Stream directory entries; disk-backed sorting keeps RAM independent of file count."""
    path = Path(path)
    count = total = 0
    excluded_names = set(excluded_names)

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
        for root, prefix in specs:
            root = Path(root)
            prefix = str(prefix).strip("/")
            if prefix:
                for part in prefix.split("/"):
                    component(part)
            if any(p.is_symlink() for p in (root, *root.parents)):
                raise ValueError("Linked source ancestor")
            single_file = root.is_file()
            if not single_file and not root.is_dir():
                raise ValueError("Missing/unsupported export source: " + str(root))
            entries = ((root, False),) if single_file else walk(root)
            for source, is_dir in entries:
                component(source.name)
                if source.name in excluded_names:
                    continue
                local_relative = source.name if single_file else source.relative_to(root).as_posix()
                relative = "/".join(part for part in (prefix, local_relative) if part)
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


def inventory(roots, path, translate):
    return inventory_mapped(((root, "") for root in roots), path, translate)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _verified_report(report_path):
    """Verify the immutable campaign identity used by local and NAS publication."""
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    seal_path = report_path.parent / "COMPLETE.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (report.get("valid") is not True or seal.get("status") != "passed"
            or seal.get("files", {}).get("report.json") != sha256(report_path)):
        raise ValueError("Dataset publication requires a passed, sealed campaign report")
    raw_campaign = Path(report["campaign"])
    campaign = ((report_path.parent / raw_campaign)
                if not raw_campaign.is_absolute() else raw_campaign).resolve()
    plan_path = campaign / "campaign_plan.json"
    ledger_path = campaign / "runs.tsv"
    completion_path = campaign / "COLLECTION_COMPLETE.json"
    if report.get("campaign_plan_sha256") != sha256(plan_path):
        raise ValueError("Campaign plan changed after validation")
    if report.get("runs_ledger_sha256") != sha256(ledger_path):
        raise ValueError("Run ledger changed after validation")
    if report.get("collection_completion_sha256") != sha256(completion_path):
        raise ValueError("Collection completion changed after validation")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    declared_folder = plan.get("dataset_scenario_folder")
    if declared_folder is not None and declared_folder != scenario_folder(plan):
        raise ValueError("Campaign dataset scenario folder differs from scenario_name")
    schema = report.get("schema")
    if schema == "morai_sensor_record_campaign_validation_v1":
        complete_selection = len(report["laps"]) == int(plan["requested_laps"])
    elif schema == "morai_sensor_record_campaign_validation_subset_v1":
        complete_selection = (
            len(report["laps"]) == int(report.get("selected_laps", -1))
            and len(report["laps"]) >= 2
            and bool(report.get("excluded_laps"))
            and report.get("subset_coverage_preserved") is True
            and report.get("subset_policy")
                == "whole_lap_fail_closed_exclusion_with_full_variation_preservation_v1"
        )
    else:
        complete_selection = False
    if not report.get("laps") or not complete_selection:
        raise ValueError(
            "Validation does not cover the complete campaign or a valid sealed subset")
    campaign_identity = campaign.parent.name if campaign.name == "_campaign" else campaign.name
    if component(plan["campaign_id"]) != campaign_identity:
        raise ValueError("Campaign directory and campaign_id differ")
    return report_path, report, campaign, plan


def _materialize_inventory(manifest, destination, identity, metadata):
    """Create/update one same-volume, no-duplicate hardlink view.

    Existing files must be the exact same inode as their sealed source. An
    interrupted first attempt can therefore resume, while unrelated or stale
    files are never overwritten.
    """
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("Canonical dataset destination cannot be a link")
    destination.mkdir(parents=True, exist_ok=True)
    expected = {LOCAL_MARKER}
    with Path(manifest).open(encoding="utf-8") as rows:
        for line in rows:
            row = json.loads(line)
            relative = row["relative"]
            expected.add(relative)
            target = destination.joinpath(*relative.split("/"))
            if row["kind"] == "directory":
                if target.exists() and (target.is_symlink() or not target.is_dir()):
                    raise ValueError("Canonical directory collision: " + str(target))
                target.mkdir(parents=True, exist_ok=True)
                continue
            source = Path(row["source"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.is_symlink() or not target.is_file() or not os.path.samefile(source, target):
                    raise ValueError("Canonical file collision: " + str(target))
            else:
                try:
                    os.link(str(source), str(target), follow_symlinks=False)
                except OSError as error:
                    raise OSError(
                        "Local dataset hardlink failed; source and dataset must be on "
                        "the same Windows local volume: %s -> %s: %s" %
                        (source, target, error)) from error
    for current, directories, files in os.walk(destination):
        current = Path(current)
        for name in directories + files:
            relative = (current / name).relative_to(destination).as_posix()
            if relative not in expected and not any(
                    item.startswith(relative + "/") for item in expected):
                raise ValueError("Unexpected file in canonical dataset: " + relative)
    marker = destination / LOCAL_MARKER
    payload = dict(schema="morai_local_dataset_materialization_v1",
                   identity=identity, hardlinked=True,
                   source_timestamps_modified=False, **metadata)
    temporary = marker.with_name(marker.name + ".tmp." + str(os.getpid()))
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(marker))
    return destination


def _materialize_specs(specs, destination, state, metadata, excluded_names=()):
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    draft = state / "inventory.new.jsonl"
    count, size = inventory_mapped(specs, draft, str, excluded_names=excluded_names)
    fingerprint = sha256(draft)
    manifest = state / ("inventory-" + fingerprint + ".jsonl")
    os.replace(str(draft), str(manifest))
    identity = hashlib.sha256(
        (str(Path(destination).resolve()).casefold() + "\n" + fingerprint).encode()
    ).hexdigest()
    _materialize_inventory(manifest, destination, identity, dict(
        inventory_sha256=fingerprint, files=count, bytes=size, **metadata))
    return dict(path=str(Path(destination).resolve()), identity=identity,
                inventory=str(manifest), files=count, bytes=size)


def materialize_validated(report_path):
    """Build the canonical scenario/campaign/run tree on the local drive."""
    from ssd_storage import SSD_ROOT
    report_path, report, campaign, plan = _verified_report(report_path)
    legacy_base = Path(SSD_ROOT) / "sensor_record_auto_v1"
    category = scenario_folder(plan)
    public_campaign_id = dataset_campaign_id(plan)
    canonical_campaign = (Path(SSD_ROOT) / "dataset" / category
                          / public_campaign_id)
    state_root = (legacy_base / ".local_publications" / category
                  / public_campaign_id)
    runs = []
    for lap in report["laps"]:
        # A completed campaign may retain failed attempts in its sealed report.
        # Their payload has already been removed by the fail-closed cleanup, so
        # only explicitly valid laps belong in the canonical training tree.
        if lap.get("valid") is not True:
            continue
        source_name = component(lap["run_name"])
        public_name = dataset_run_name(plan, lap)
        result = _materialize_specs(
            ((root, "") for root in roots_for_run(legacy_base, source_name)),
            canonical_campaign / public_name, state_root / public_name,
            dict(kind="run", scenario=plan.get("scenario_name", "full_loop"),
                 scenario_folder=category, campaign_id=plan["campaign_id"],
                 dataset_campaign_id=public_campaign_id,
                 source_run_name=source_name, run_name=public_name,
                 validation_report=str(report_path)),
        )
        result["source_run_name"] = source_name
        result["run_name"] = public_name
        result["relative"] = public_name
        runs.append(result)
    metadata = _materialize_specs(
        ((campaign / name, "") for name in
         ("campaign_plan.json", "runs.tsv", "COLLECTION_COMPLETE.json")),
        canonical_campaign / "_campaign" / "metadata",
        state_root / "_campaign_metadata",
        dict(kind="campaign_metadata", scenario_folder=category,
             campaign_id=plan["campaign_id"],
             dataset_campaign_id=public_campaign_id),
    )
    validation = _materialize_specs(
        ((report_path.parent, ""),),
        canonical_campaign / "_campaign" / "validations" / report_path.parent.name,
        state_root / ("validation_" + report_path.parent.name),
        dict(kind="campaign_validation", scenario_folder=category,
             campaign_id=plan["campaign_id"],
             dataset_campaign_id=public_campaign_id,
             validation=report_path.parent.name),
        excluded_names={"nas_publication.json", "local_dataset_publication.json"},
    )
    metadata["relative"] = "_campaign/metadata"
    validation["relative"] = "_campaign/validations/" + report_path.parent.name
    return dict(schema="morai_local_dataset_campaign_v1", scenario_folder=category,
                campaign_id=plan["campaign_id"],
                dataset_campaign_id=public_campaign_id,
                campaign=str(canonical_campaign),
                runs=runs, evidence=[metadata, validation], hardlinked=True,
                source_timestamps_modified=False)


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


def make_tree_job(source, relative_destination, plan, target_root, state_root, translate):
    """Publish one already canonical local tree without changing its layout."""
    source = Path(source).resolve()
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Missing/linked canonical dataset source: " + str(source))
    parts = tuple(component(part) for part in str(relative_destination).split("/") if part)
    if not parts:
        raise ValueError("Canonical destination must have at least one component")
    state_name = "__".join(parts)
    state = Path(state_root) / state_name
    state.mkdir(parents=True, exist_ok=True)
    draft = state / "inventory.new.jsonl"
    count, size = inventory((source,), draft, translate)
    fingerprint = sha256(draft)
    manifest = state / ("inventory-" + fingerprint + ".jsonl")
    os.replace(str(draft), str(manifest))
    destination = str(PureWindowsPath(
        validate_target(target_root), scenario_folder(plan),
        dataset_campaign_id(plan), *parts))
    identity = hashlib.sha256(
        (destination.casefold() + "\n" + fingerprint).encode()).hexdigest()
    job = dict(schema="morai_nas_transfer_v1", identity=identity,
               destination=destination, sources=[translate(source)],
               inventory=translate(manifest), inventory_sha256=fingerprint,
               state=translate(state / identity), files=count, bytes=size,
               campaign_id=plan["campaign_id"],
               dataset_campaign_id=dataset_campaign_id(plan),
               publication_relative="/".join(parts))
    job_path = state / ("job-" + identity + ".json")
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    return job_path


def retained_invalid_runs(report):
    """Run folders of invalid laps whose payload the validator kept for review."""
    return sorted({
        Path(str(lap["run"])).name
        for lap in report.get("laps", []) + report.get("excluded_laps", [])
        if lap.get("valid") is not True and lap.get("run")
        and (lap.get("payload_cleanup") or {}).get("status") == "retained"
    })


def publish_validated(report_path, target_root=None, dry_run=False,
                      local_publication=None):
    from runtime_paths import LOCAL_MACHINE, POWERSHELL_EXE, windows_path_from_wsl
    from ssd_storage import SSD_ROOT
    report_path, report, campaign, plan = _verified_report(report_path)
    direct_campaign = campaign.parent if campaign.name == "_campaign" else None
    local = local_publication or (
        dict(schema="morai_local_dataset_campaign_v2", scenario_folder=scenario_folder(plan),
             campaign_id=plan["campaign_id"],
             dataset_campaign_id=dataset_campaign_id(plan),
             campaign=str(direct_campaign), direct_storage=True)
        if direct_campaign is not None else materialize_validated(report_path))
    if (local.get("schema") not in {"morai_local_dataset_campaign_v1",
                                    "morai_local_dataset_campaign_v2"}
            or local.get("campaign_id") != plan["campaign_id"]
            or local.get("dataset_campaign_id", local.get("campaign_id"))
                != dataset_campaign_id(plan)
            or local.get("scenario_folder") != scenario_folder(plan)):
        raise ValueError("Canonical local dataset identity differs from validation")
    target_root = (target_root or os.environ.get("MORAI_NAS_ROOT")
                   or LOCAL_MACHINE.get("nas_root"))
    if not target_root:
        raise ValueError("Set --nas-root, MORAI_NAS_ROOT or local_machine.json nas_root; "
                         "use --no-nas-copy for local validation only")
    validate_target(target_root)
    state_root = SSD_ROOT / "dataset" / ".nas_transfers"
    state_root.mkdir(parents=True, exist_ok=True)
    worker_source = ROOT / "tools" / "publish_sensor_record_nas.ps1"
    worker = state_root / ("worker-" + sha256(worker_source)[:16] + ".ps1")
    if not worker.exists():
        shutil.copyfile(worker_source, worker)
    results = []
    if local.get("schema") == "morai_local_dataset_campaign_v2":
        direct_root = Path(local["campaign"]).resolve()
        # Invalid laps kept locally are review material, not shared data.
        skipped = retained_invalid_runs(report)
        for name in skipped:
            print("[NAS] %s: skipped, invalid lap payload retained locally" % name, flush=True)
            results.append(dict(publication=name, skipped="invalid_lap_payload_retained_locally"))
        publications = [
            {"path": str(child), "relative": child.name, "run_name": child.name}
            for child in sorted(direct_root.iterdir(), key=lambda item: item.name)
            if child.is_dir() and not child.is_symlink() and child.name not in skipped
        ]
    else:
        publications = local["runs"] + local["evidence"]
    for publication in publications:
        label = publication.get("run_name", publication["relative"])
        job_path = make_tree_job(
            publication["path"], publication["relative"], plan,
            target_root, state_root, windows_path_from_wsl)
        job = json.loads(job_path.read_text(encoding="utf-8"))
        print("[NAS] %s: %d files, %.2f GiB -> %s%s" % (
            label, job["files"], job["bytes"] / 2**30,
            job["destination"], " (plan only; destination untouched)" if dry_run else ""),
            flush=True)
        if dry_run:
            results.append(dict(publication=label, planned=True, job=str(job_path),
                                destination=job["destination"]))
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
        results.append(dict(publication=label, verified_readback=True,
                            destination=job["destination"], remote_durability_verified=False))
    return results
