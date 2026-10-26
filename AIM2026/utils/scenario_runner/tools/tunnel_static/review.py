"""Review one clean, known road-margin rejection without changing frozen code.

No simulator/ROS calls, process signals, approvals, counters or trial outcomes
are changed. The default only inspects. --apply excludes the selected layout
after archiving the exact stopped journal and its original failure evidence.
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from ...scenarios.full_loop_parts.tunnel_static_parts import runtime


REASON = "Observed expanded footprint lacks the shared road margin"
BLOCKED_REASON = "Trial stopped for transport, source, identity, publisher, cleanup or unclassified failure: " + REASON
SCHEMA = "aimdrive_tunnel_known_preflight_review_v1"
COUNTERS = ("actual_drives", "actual_passes", "actual_failures", "preflight_rejects")


def _check(condition, reason):
    if not condition:
        raise RuntimeError("Preflight review refused: " + reason)


def _within(path, root):
    path = Path(path).resolve()
    _check(path.is_relative_to(root) if hasattr(path, "is_relative_to") else root in path.parents,
           "evidence path lies outside the campaign")
    return path


def _bytes(document):
    return (json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _archive(path, payload):
    """Never overwrite an existing archive, including a retry after interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _check(path.read_bytes() == payload, "immutable archive differs: " + str(path))
        return
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _active_processes(root, proc_root=Path("/proc")):
    """Observe owned workers/Experts; never signal any process."""
    found = []
    modules = {b"scenario_runner.tools.run_tunnel_static_search",
               b"scenario_runner.tools.run_tunnel_static_trial",
               b"scenario_runner.tools.tunnel_static_batch_safety",
               b"scenario_runner.tools.tunnel_static_search_guardian"}
    for node in Path(proc_root).iterdir():
        if not node.name.isdecimal():
            continue
        try:
            if node.stat().st_uid != os.geteuid():
                continue
            fields = (node / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                continue
            argv = [value for value in (node / "cmdline").read_bytes().split(b"\0") if value]
            # Inspect the executable/entrypoint before environ: unrelated host
            # processes may be nondumpable even when they share this UID.
            if not argv or re.fullmatch(rb"python(?:\d+(?:\.\d+)*)?", os.fsencode(Path(os.fsdecode(argv[0])).name)) is None:
                continue
            index = 1
            while index < len(argv) and argv[index] in (b"-u", b"-B", b"-E", b"-s", b"-S", b"-I", b"-O", b"-OO", b"-q", b"-b", b"-bb"):
                index += 1
            module = argv[index + 1] if index + 1 < len(argv) and argv[index] == b"-m" else None
            script = argv[index] if index < len(argv) and not argv[index].startswith(b"-") else None
            expert = (module == b"scenario_runner.adapters.expert_process"
                      or script == os.fsencode(runtime.PACKAGE_ROOT / "adapters/expert_process.py"))
            guardian_entry = (module == b"scenario_runner.tools.tunnel_static_search_guardian"
                              or (script is not None and Path(os.fsdecode(script)).name == "aimdrive_tunnel_mass_guardian_20261008.py"))
            if module not in modules and not expert and not guardian_entry:
                continue
            env = dict(item.split(b"=", 1) for item in (node / "environ").read_bytes().split(b"\0") if b"=" in item)
            owned = env.get(b"AIMDRIVE_TUNNEL_BATCH_ROOT") == os.fsencode(root)
            private = env.get(b"GT_BEV_CONFIG_PATH", b"").startswith(os.fsencode(str(root) + "/"))
            command_files = [argv[position + 1] for position, value in enumerate(argv[:-1]) if value == b"--command-file"]
            guardian = guardian_entry and any(root in Path(os.fsdecode(value)).resolve().parents for value in command_files)
            if ((owned and module in modules) or (expert and (owned or private)) or guardian):
                found.append(int(node.name))
        except FileNotFoundError:
            continue
        except (OSError, ValueError, IndexError) as error:
            raise RuntimeError("Cannot prove campaign process absence") from error
    return found


def _parking(report):
    return (report.get("parked") is True and report.get("external_input_disabled") is True
            and report.get("gear_mode") == 1 and report.get("control_mode") == 1
            and isinstance(report.get("speed_norm_raw"), (int, float))
            and math.isfinite(report["speed_norm_raw"]) and 0 <= report["speed_norm_raw"] <= .03)


def _validate(root, state, supervisor, *, source_check, process_check):
    _check(state.get("schema") == "aimdrive_tunnel_static_adaptive_search_v1", "unsupported campaign schema")
    reference = state.get("source_snapshot", {})
    pins = _within(reference.get("path", ""), root)
    _check(pins == root / "source_pins.json" and runtime.digest(pins) == reference.get("sha256"), "source pin reference changed")
    source_check(runtime.read_json(pins))
    _check(state.get("status") == "blocked" and state.get("failure_reason") == BLOCKED_REASON,
           "failure is not the exact known road-margin preflight")
    _check(state.get("inflight") is None and state.get("cleanup_verified") is True and state.get("scene_restored") is True,
           "campaign is not recovered and restored")
    _check(supervisor.get("status") == "stopped_cleanup_verified" and supervisor.get("cleanup_verified") is True
           and supervisor.get("finished_unix") is not None and supervisor.get("worker_returncode") == 2
           and supervisor.get("forced_kill") is False and supervisor.get("interrupted") is False,
           "supervisor is not a clean terminal stop")
    _check(supervisor.get("campaign_root") == str(root), "supervisor belongs to another campaign")
    _check(process_check(root) == [], "a campaign worker, helper, guardian or Expert remains")
    for key in COUNTERS:
        _check(isinstance(state.get(key), int) and not isinstance(state[key], bool) and state[key] >= 0, "invalid counter " + key)
    _check(state["actual_drives"] == state["actual_passes"] + state["actual_failures"], "actual counters do not sum")
    attempts = state.get("attempts", [])
    _check(bool(attempts) and all(row.get("committed") is True for row in attempts), "uncommitted trial attempt")
    _check(sum(row.get("outcome", {}).get("actual_driving") is True for row in attempts) == state["actual_drives"],
           "actual count is not backed by the committed journal")
    _check(sum(row.get("outcome", {}).get("actual_driving") is True and row["outcome"].get("strict_pass") is True
               for row in attempts) == state["actual_passes"], "actual pass count is not backed by the committed journal")
    _check(sum(row.get("outcome", {}).get("actual_driving") is True and row["outcome"].get("strict_pass") is False
               for row in attempts) == state["actual_failures"], "actual failure count is not backed by the committed journal")
    _check(sum(row.get("outcome", {}).get("actual_driving") is False for row in attempts) == state["preflight_rejects"],
           "preflight count is not backed by the committed journal")
    entry = attempts[-1]
    result = entry.get("outcome", {})
    _check(entry.get("disposition") == "fatal" and entry.get("recovery_verified") is True,
           "last attempt is not the recovered terminal fatal preflight")
    _check(all(result.get(key) is False for key in ("actual_driving", "expert_entered", "physical_motion_observed", "strict_pass")),
           "actual driving, motion or Expert evidence exists")
    _check(result.get("planner_calls") == 0 and result.get("trajectory_checks") == 0
           and result.get("failure_reason") == REASON and result.get("fixture_cleanup_verified") is True
           and result.get("native_metadata_failure") is None and not result.get("completion_conditions"),
           "outcome is not the narrow clean geometry rejection")
    process = entry.get("process", {})
    _check(process.get("returncode") == 1 and all(process.get(key) is False for key in ("timed_out", "interrupted", "forced_kill")),
           "trial transport/termination was not clean")
    matches = [row for row in state.get("candidates", []) if row.get("id") == entry.get("candidate_id")]
    _check(len(matches) == 1 and matches[0].get("status") == "selected", "last failed candidate is not uniquely selected")
    row = matches[0]
    candidate_path = _within(row["candidate_path"], root)
    _check(runtime.digest(candidate_path) == row.get("candidate_sha256") == entry.get("candidate_sha256"), "candidate SHA changed")
    trial = _within(entry["trial_dir"], root)
    candidate = runtime.read_json(candidate_path)
    _check(runtime.digest(trial / "candidate.json") == row["candidate_sha256"], "trial candidate SHA differs")
    preview, manifest = runtime.read_json(trial / "preview.json"), runtime.read_json(trial / "manifest.json")
    _check(manifest.get("status") == "failed" and manifest.get("expert_verified") is False
           and manifest.get("live_avoidance_verified") is False and manifest.get("failure_reason") == REASON
           and manifest.get("fixture_result") == preview, "trial manifest does not bind the exact failed preview")
    _check(preview.get("status") == "failed" and preview.get("failure_reason") == REASON
           and preview.get("expert_verified") is False and preview.get("live_avoidance_verified") is False
           and preview.get("planned") == candidate and preview.get("route_screened") is False
           and preview.get("stationary_samples") == 0 and preview.get("measured_route_screen") is None
           and preview.get("native_metadata_failure") is None and preview.get("cleanup", {}).get("verified") is True,
           "preview reached another stage or contains another failure")
    _check(not any((trial / name).exists() for name in ("drive_trial.json", "planner_diagnostics.json", "ego_trace.csv", "expert_config.json")),
           "drive/planner/trace/Expert artifacts exist")
    observed = preview.get("observed", [])
    _check(len(observed) == candidate.get("count") == len(candidate.get("obstacles", [])) and bool(observed), "incomplete native observation")
    ids = [item.get("raw_actor_id") for item in observed]
    _check(len(set(ids)) == len(ids) and all(isinstance(identity, str) and identity.isdecimal()
           and identity not in ("2", "3", "4", "5") for identity in ids), "observed actor identities are not owned fixtures")
    destroyed = preview["cleanup"].get("actors", [])
    _check({item.get("raw_actor_id") for item in destroyed} == set(ids)
           and len(destroyed) == len(ids) and all(item.get("destroy_called") is True for item in destroyed),
           "owned native observations lack exact verified destruction")
    for planned, item in zip(candidate["obstacles"], observed):
        _check(item.get("model") == planned.get("model") and item.get("actor_kind") == planned.get("actor_kind")
               and item.get("spawn_rpc_error") is None, "native model/identity/RPC mismatch")
        _check(all(isinstance(item.get(key), (int, float)) and math.isfinite(item[key]) and item[key] > 0
                   for key in ("width_m", "length_m")), "invalid observed native dimensions")
        _check(len(item.get("center_xyz", [])) == 3 and all(math.isfinite(float(value)) for value in item["center_xyz"])
               and math.isfinite(float(item.get("heading_deg", float("nan")))), "invalid observed native box")
    recovery = runtime.read_json(trial.parent / "recovery/recover_result.json")
    final = runtime.read_json(root / "recovery_final/recover_result.json")
    restore = runtime.read_json(root / "scene_backup/restore_result.json")
    for report in (recovery, final):
        _check(report.get("status") == "recovered" and report.get("controller_publishers") == []
               and report.get("preserved_foreign_actor_ids") == ["2", "3", "4", "5"]
               and _parking(report.get("final_parking", {})), "recovery/parking/control publisher proof missing")
    _check(restore.get("status") == "restored" and _parking(restore.get("parking", {}))
           and sorted(item.get("id") for item in restore.get("actors", [])) == ["2", "4", "5"], "foreign restoration proof missing")
    return entry, row, trial


def _geometry_origin(preview, candidate):
    """Reproduce the same pinned local geometry gate, without live I/O."""
    from ...utils.map_loader import MGeoMapLoader
    from ...scenarios.full_loop_parts.tunnel_static_parts.layout import tunnel_geometry, expanded_road_boundary
    from ...scenarios.full_loop_parts.tunnel_static_parts.live import _geometry_checks, _stationary
    loader = MGeoMapLoader(str(runtime.PACKAGE_ROOT / "data/R_KR_PR_K-city_2025"))
    geometry = tunnel_geometry(loader)
    _check(candidate.get("source_geometry", {}).get("sha256") == geometry["source_sha256"], "mapped source geometry changed")
    _check(candidate.get("experimental_placement_domain") == "two_lane_physical_road", "candidate is not expanded geometry")
    geometry.update(placement_domain="two_lane_physical_road", road_boundary=expanded_road_boundary(loader))
    for item in preview["observed"]:
        _stationary(item, None, "Reviewed candidate actor")
    _stationary(preview["ego_observed"], None, "Reviewed parked Ego")
    try:
        _geometry_checks(preview["observed"], preview["ego_observed"]["raw_bbox_normalized"], geometry)
    except RuntimeError as error:
        _check(str(error) == REASON, "pinned geometry reproduction produced another failure")
        return
    raise RuntimeError("Preflight review refused: pinned observed geometry does not reproduce the road-margin failure")


def _finish_transaction(root, directory, *, apply, source_check, process_check, geometry_check):
    transaction = runtime.read_json(directory / "transaction.json")
    docket_path = directory / "docket.json"
    docket = runtime.read_json(docket_path)
    _check(runtime.digest(docket_path) == transaction["docket_sha256"], "pending docket changed")
    for path, expected in docket["archive_sha256"].items():
        _check(runtime.digest(_within(path, root)) == expected, "immutable original archive changed")
    original = runtime.read_json(directory / "original_search_manifest.json")
    entry, _row, trial = _validate(root, original, runtime.read_json(root / "supervisor.json"),
                                  source_check=source_check, process_check=process_check)
    geometry_check(runtime.read_json(trial / "preview.json"), runtime.read_json(_row["candidate_path"]))
    for path, expected in docket["proof_sha256"].items():
        if Path(path).name not in ("search_manifest.json", "batch_manifest.json", "progress.json"):
            _check(runtime.digest(path) == expected, "pending original proof changed")
    documents = []
    for item in transaction["targets"]:
        target = _within(item["path"], root)
        _check(target.name in ("search_manifest.json", "batch_manifest.json", "progress.json") and target.parent == root,
               "invalid transaction target")
        after_path = _within(item["after_path"], root)
        _check(runtime.digest(after_path) == item["after_sha256"], "immutable proposed checkpoint changed")
        _check(runtime.digest(target) in (item["before_sha256"], item["after_sha256"]), "checkpoint changed outside reviewed transition")
        before = runtime.read_json(directory / ("original_" + item["name"] + ".json"))
        expected = copy.deepcopy(before)
        expected["status"], expected["failure_reason"] = "prepared", ""
        if "candidates" in expected:
            next(row for row in expected["candidates"] if row["id"] == entry["candidate_id"])["status"] = "preflight_excluded"
            expected.setdefault("operator_review_geometry_exclusions", []).append(transaction["note"])
        after = runtime.read_json(after_path)
        _check(after == expected, "proposed transition changes more than the reviewed exclusion")
        documents.append((target, after))
    if not apply:
        return {"schema": SCHEMA, "status": "eligible_pending", "candidate_id": entry["candidate_id"], "docket_path": str(docket_path)}
    for target, document in documents:
        runtime.atomic_json(target, document)
    _archive(directory / "committed.json", _bytes({"schema": SCHEMA, "review_id": docket["review_id"],
        "after_sha256": {str(target): runtime.digest(target) for target, _document in documents}}))
    return {"schema": SCHEMA, "status": "applied", "candidate_id": entry["candidate_id"], "docket_path": str(docket_path)}


def review_campaign(campaign_root, *, apply=False, max_reviews=8,
                    source_check=runtime.require_sources, process_check=_active_processes,
                    geometry_check=_geometry_origin, global_lock_path=runtime.GLOBAL_EXECUTION_LOCK):
    root = Path(campaign_root).resolve()
    _check(1 <= max_reviews <= 8, "review budget must be 1..8")
    with runtime.ExecutionLock(root / "campaign.lock"), runtime.ExecutionLock(global_lock_path):
        pending = [path for path in (root / "operator_preflight_reviews").glob("*/transaction.json")
                   if not path.with_name("committed.json").exists()]
        _check(len(pending) <= 1, "multiple interrupted operator transitions")
        if pending:
            return _finish_transaction(root, pending[0].parent, apply=apply,
                source_check=source_check, process_check=process_check, geometry_check=geometry_check)
        state_path, batch_path, progress_path = (root / name for name in ("search_manifest.json", "batch_manifest.json", "progress.json"))
        state = runtime.read_json(state_path)
        notes = state.get("operator_review_geometry_exclusions", [])
        if notes and state.get("status") == "prepared" and not state.get("failure_reason"):
            note = notes[-1]
            docket_path = _within(note["docket_path"], root)
            _check(runtime.digest(docket_path) == note["docket_sha256"], "existing review docket changed")
            docket = runtime.read_json(docket_path)
            transaction = runtime.read_json(docket_path.with_name("transaction.json"))
            committed = runtime.read_json(docket_path.with_name("committed.json"))
            _check(committed is not None, "review commit marker is missing")
            for item in transaction["targets"]:
                _check(runtime.digest(_within(item["path"], root)) == item["after_sha256"]
                       == committed["after_sha256"].get(item["path"]), "committed reviewed journal changed")
            _check(docket.get("original_attempt") == state["attempts"][-1], "reviewed original attempt/outcome changed")
            for path, expected in docket.get("archive_sha256", {}).items():
                _check(runtime.digest(_within(path, root)) == expected, "immutable review archive changed")
            _check(state["attempts"][-1]["candidate_id"] == note["candidate_id"], "existing review does not match terminal attempt")
            _check(next(row for row in state["candidates"] if row["id"] == note["candidate_id"])["status"] == "preflight_excluded",
                   "reviewed candidate status changed")
            _check(runtime.digest(root / "source_pins.json") == state["source_snapshot"]["sha256"], "source pin reference changed")
            source_check(runtime.read_json(root / "source_pins.json"))
            _check(process_check(root) == [], "a campaign process remains")
            return {"schema": SCHEMA, "status": "already_applied", "candidate_id": note["candidate_id"], "docket_path": str(docket_path)}
        _check(len(notes) < max_reviews, "bounded review budget exhausted")
        supervisor = runtime.read_json(root / "supervisor.json")
        entry, row, trial = _validate(root, state, supervisor, source_check=source_check, process_check=process_check)
        geometry_check(runtime.read_json(trial / "preview.json"), runtime.read_json(row["candidate_path"]))
        batch, progress = runtime.read_json(batch_path), runtime.read_json(progress_path)
        _check(all(batch.get(key) == state.get(key) and progress.get(key) == state.get(key) for key in COUNTERS), "journal counter mirrors differ")
        _check(batch.get("candidates") == state["candidates"] and batch.get("attempts", [])[-len(state["attempts"]):] == state["attempts"],
               "batch journal does not preserve exact campaign attempts")
        proof_paths = [state_path, batch_path, progress_path, root / "supervisor.json", root / "source_pins.json",
                       Path(row["candidate_path"]), trial / "candidate.json", trial / "preview.json", trial / "manifest.json",
                       trial.parent / "recovery/recover_result.json", root / "recovery_final/recover_result.json", root / "scene_backup/restore_result.json"]
        proof_pins = {str(path): runtime.digest(path) for path in proof_paths}
        report = {"schema": SCHEMA, "status": "eligible", "candidate_id": row["id"], "failure_reason_original": REASON,
                  "counters_preserved": {key: state[key] for key in COUNTERS}, "proof_sha256": proof_pins,
                  "source_pins_unchanged": True, "geometric_impossibility_proven": False, "actual_approval": False}
        if not apply:
            return report
        review_id = runtime.fingerprint({"candidate_id": row["id"], "trial_dir": str(trial), "preview_sha256": proof_pins[str(trial / "preview.json")]})
        directory = root / "operator_preflight_reviews" / review_id
        originals = {"search_manifest": state_path.read_bytes(), "batch_manifest": batch_path.read_bytes(), "progress": progress_path.read_bytes()}
        for name, payload in originals.items():
            _archive(directory / ("original_" + name + ".json"), payload)
        for index, path in enumerate(proof_paths[3:]):
            _archive(directory / ("proof_%02d_%s" % (index, path.name)), path.read_bytes())
        docket_path = directory / "docket.json"
        docket = dict(report, review_id=review_id, status="operator_reviewed_geometry_exclusion", reviewed_unix=time.time(),
                      tool_sha256=runtime.digest(__file__), original_attempt=entry,
                      archive_sha256={str(path): runtime.digest(path) for path in directory.iterdir() if path.is_file()})
        if docket_path.exists():
            docket = runtime.read_json(docket_path)
            _check(docket.get("proof_sha256") == proof_pins, "pending docket proof changed")
        else:
            _archive(docket_path, _bytes(docket))
        note = {"review_id": review_id, "candidate_id": row["id"], "trial_dir": str(trial),
                "docket_path": str(docket_path), "docket_sha256": runtime.digest(docket_path),
                "failure_reason_original": BLOCKED_REASON, "actual_approval": False, "geometric_impossibility_proven": False}
        proposed = []
        for document in (batch, progress, state):
            after = copy.deepcopy(document)
            after["status"], after["failure_reason"] = "prepared", ""
            if "candidates" in after:
                next(item for item in after["candidates"] if item["id"] == row["id"])["status"] = "preflight_excluded"
                after.setdefault("operator_review_geometry_exclusions", []).append(note)
            proposed.append(after)
        targets = []
        for name, path, after in zip(("batch_manifest", "progress", "search_manifest"), (batch_path, progress_path, state_path), proposed):
            after_path = directory / ("proposed_" + name + ".json")
            _archive(after_path, _bytes(after))
            targets.append({"name": name, "path": str(path), "before_sha256": proof_pins[str(path)],
                            "after_path": str(after_path), "after_sha256": runtime.digest(after_path)})
        transaction = {"schema": SCHEMA, "docket_sha256": runtime.digest(docket_path), "note": note, "targets": targets}
        _archive(directory / "transaction.json", _bytes(transaction))
        source_check(runtime.read_json(root / "source_pins.json"))
        _check(process_check(root) == [], "campaign process appeared during review")
        _check(all(runtime.digest(path) == expected for path, expected in proof_pins.items()), "reviewed evidence changed before commit")
        # Primary journal is last; the immutable transaction resumes any partial
        # write after a crash without modifying counters or original outcomes.
        return _finish_transaction(root, directory, apply=True, source_check=source_check,
                                   process_check=process_check, geometry_check=geometry_check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--max-reviews", type=int, default=8)
    args = parser.parse_args()
    try:
        result = review_campaign(args.campaign_root, apply=args.apply, max_reviews=args.max_reviews)
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
