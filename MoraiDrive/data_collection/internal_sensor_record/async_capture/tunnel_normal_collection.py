"""Capture-owned evidence for empty tunnel demonstrations.

The existing Scenario Runner baseline owns the live inventory and drive checks.
This adapter seals its final result after MissionScenario's parking finally
block; validation reads captured bytes and never runs archived source code.
Sensor/teacher/actor association and collision admission remain separate.
"""
import hashlib
import json
import math
import os
from pathlib import Path


MODE = "full_loop_tunnel_normal"
LEGACY_CONTRACT_SCHEMA = "morai_tunnel_normal_collection_contract_v1"
CONTRACT_SCHEMA = "morai_tunnel_normal_collection_contract_v2"
STOP_POLICY = "common_auto_d_ros_brake"
PROOF_SCHEMA = "morai_tunnel_normal_collection_complete_v1"
PROOF_RELATIVE = Path("tunnel_normal/COLLECTION_TUNNEL_NORMAL_COMPLETE.json")
PROJECT = Path(__file__).resolve().parents[3]
LEGACY_SOURCES = (
    "scenario_runner/scenarios/segment_modes/tunnel_baseline.py",
    "scenario_runner/scenarios/segment_modes/scenario.py",
    "scenario_runner/scenarios/segment_modes/modes.py",
    "scenario_runner/scenarios/segment_modes/route.py",
    "scenario_runner/scenarios/segment_modes/events.py",
    "data_collection/internal_sensor_record/scenario_collection.py",
    "data_collection/internal_sensor_record/tools/run_proven_three_lap_scenario.py",
    "data_collection/internal_sensor_record/async_capture/tunnel_normal_collection.py",
)
SOURCES = LEGACY_SOURCES + (
    "scenario_runner/scenarios/segment_modes/tunnel_static_startup.py",
    "scenario_runner/scenarios/full_loop_parts/tunnel_static_parts/live.py",
    "scenario_runner/scenarios/full_loop_parts/tunnel_static_parts/connection.py",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, document):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def normal_collection_contract():
    return dict(
        schema=CONTRACT_SCHEMA, scenario=MODE,
        source_sha256={relative: _sha(PROJECT / relative) for relative in SOURCES},
        guard="Scenario Runner global actor inventory, mapped traversal and observed common brake stop",
        final_stop_policy=STOP_POLICY,
        manual_all_frame_camera_certification=False,
    )


def _verified_common_stop(report):
    """Validate actual brake-held D observations; never describe them as P."""
    record = report.get("final_parking", {})
    if (report.get("stop_policy") != STOP_POLICY or record.get("stop_policy") != STOP_POLICY
            or report.get("final_stop_verified") is not True
            or report.get("ego_stopped_and_parked") is not False
            or record.get("status") != "verified"
            or record.get("velocity_reset_used") is not False
            or record.get("external_input_disabled") is not False
            or record.get("gear_p_verified") is not False
            or not all(record.get(key) is True for key in (
                "physical_stop_verified", "owned_expert_stopped", "controller_publishers_absent",
                "publisher_released_after_verified", "post_release_verified", "final_verified"))):
        return False
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    def vector(value):
        return isinstance(value, (list, tuple)) and len(value) == 3 and all(finite(x) for x in value)
    stops = [record.get(key, {}) for key in ("actual_stop", "actual_final")]
    for observed in stops:
        if (not isinstance(observed, dict) or observed.get("control_mode") != 3
                or observed.get("gear_mode") != 4 or observed.get("brake_hold_verified") is not True
                or not finite(observed.get("brake")) or not .99 <= observed["brake"] <= 1.
                or not finite(observed.get("throttle")) or abs(observed["throttle"]) > .01
                or not finite(observed.get("speed_norm_raw")) or not 0. <= observed["speed_norm_raw"] <= .05
                or not vector(observed.get("position_xyz"))
                or not vector(observed.get("global_velocity_raw"))
                or not vector(observed.get("angular_velocity_raw"))
                or math.sqrt(sum(x*x for x in observed["global_velocity_raw"])) > .05
                or math.sqrt(sum(x*x for x in observed["angular_velocity_raw"])) > .05):
            return False
    return math.dist(stops[0]["position_xyz"], stops[1]["position_xyz"]) <= .02


def _baseline_checks(report, contract=None):
    contract = contract or {}
    policy = report.get("stop_policy")
    final_stop = (_verified_common_stop(report) and contract.get("schema") == CONTRACT_SCHEMA if policy == STOP_POLICY
                  else policy is None and contract.get("schema") == LEGACY_CONTRACT_SCHEMA
                  and report.get("ego_stopped_and_parked") is True)
    if contract.get("schema") == CONTRACT_SCHEMA:
        final_stop = final_stop and contract.get("final_stop_policy") == STOP_POLICY and policy == STOP_POLICY
    route = report.get("test_route", {})
    geometry = report.get("mapped_tunnel", {})
    distances = [route.get(key) for key in (
        "length_m", "mapped_entry_route_s_m", "mapped_exit_route_s_m",
        "approach_length_m", "after_exit_length_m",
    )]
    finite = all(isinstance(x, (int, float)) and not isinstance(x, bool)
                 and math.isfinite(x) and x > 0 for x in distances)
    return dict(
        baseline_schema=report.get("schema") == "aimdrive_tunnel_baseline_v1"
            and report.get("scenario") == MODE,
        completed_drive=report.get("status") == "passed"
            and report.get("baseline_drive_passed") is True and not report.get("failure_reason"),
        actual_traversal=all(report.get(key) is True for key in (
            "started_before_entry", "mapped_entry_observed", "mapped_exit_observed"))
            and isinstance(report.get("inside_observations"), int)
            and not isinstance(report.get("inside_observations"), bool)
            and report["inside_observations"] > 0,
        empty_road_inventory=report.get("inventory_scope") == "all clients, vehicle/pedestrian/obstacle"
            and isinstance(report.get("inventory_checks"), int) and report["inventory_checks"] >= 2
            and not isinstance(report.get("inventory_checks"), bool)
            and report.get("last_blocking_actors") == [],
        mapped_route=finite and distances[1] < distances[2] < distances[0]
            and geometry.get("ego_link") == "A2256W000126"
            and geometry.get("opposing_link") == "A2256W000131",
        final_stop=final_stop,
    )


def finalize_normal_collection(context, outcome):
    """Called only after scenario.run returns, after its final parking check."""
    if context.cfg.get("mission_mode") != MODE:
        return
    episode = Path(context.output).resolve()
    contract = context.cfg.get("_tunnel_normal_collection", {})
    if contract.get("schema") != CONTRACT_SCHEMA or set(contract.get("source_sha256", {})) != set(SOURCES):
        raise ValueError("Empty tunnel collection lacks its captured source contract")
    report = getattr(context, "tunnel_baseline_report", None)
    if not isinstance(report, dict):
        raise ValueError("Empty tunnel collection lacks the live baseline report")
    directory = episode / "tunnel_normal"
    directory.mkdir(exist_ok=False)
    baseline = directory / "baseline.json"
    _write(baseline, report)
    from scenario_collection import tunnel_collection_evidence
    outcome["tunnel_mission"] = tunnel_collection_evidence(context)
    _write(episode / "outcome.json", outcome)
    paths = dict(episode=episode / "episode.json", outcome=episode / "outcome.json",
                 raw_seal=episode / "raw/RAW_COMPLETE.json", baseline=baseline)
    for relative, expected in contract["source_sha256"].items():
        source = PROJECT / relative
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("Empty tunnel source changed during capture: " + relative)
        target = directory / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        paths["source:" + relative] = target
    checks = _baseline_checks(report, contract)
    checks["collection_completed"] = outcome.get("status") == "mission_completed" \
        and outcome.get("full_mission_completed") is True
    checks["raw_capture_closed"] = _json(paths["raw_seal"]).get("status") == "completed"
    proof = dict(
        schema=PROOF_SCHEMA, status="completed" if all(checks.values()) else "failed",
        scenario=MODE, seed=getattr(context, "run_seed", None), checks=checks,
        source_sha256=contract["source_sha256"],
        files={role: dict(path=path.relative_to(episode).as_posix(), sha256=_sha(path))
               for role, path in paths.items()},
        physical_semantic_validation_completed=False,
        manual_all_frame_camera_certification=False,
        native_assets_modified=False,
    )
    _write(episode / PROOF_RELATIVE, proof)


def validate_normal_collection(episode, scenario_name):
    """Require capture-local, hash-verified drive evidence for negative examples."""
    if scenario_name != MODE:
        return dict(required=False, valid=True)
    episode = Path(episode).resolve()
    proof_path = episode / PROOF_RELATIVE
    checks, errors = {}, []
    try:
        from .sensor_record_admission import _safe_local_file
        if not _safe_local_file(proof_path, episode):
            raise ValueError("Empty tunnel final proof missing or unsafe")
        proof = _json(proof_path)
        checks["final_proof"] = proof.get("schema") == PROOF_SCHEMA \
            and proof.get("status") == "completed" and proof.get("scenario") == MODE
        documents = {}
        files = proof.get("files", {})
        for role in ("episode", "outcome", "raw_seal", "baseline"):
            entry = files.get(role, {})
            relative = Path(entry.get("path", ""))
            path = episode / relative
            if relative.is_absolute() or not _safe_local_file(path, episode) or _sha(path) != entry.get("sha256"):
                raise ValueError("Missing, changed or unsafe empty tunnel evidence: " + role)
            documents[role] = _json(path)
        checks["sealed_episode_files"] = True
        metadata, outcome = documents["episode"], documents["outcome"]
        contract = metadata.get("scenario_config", {}).get("_tunnel_normal_collection", {})
        source_pins = contract.get("source_sha256", {})
        # Earlier captures sealed the original one-shot park implementation.
        # Verify their exact captured bytes without requiring new runtime files.
        if (contract.get("schema") not in (CONTRACT_SCHEMA, LEGACY_CONTRACT_SCHEMA)
                or set(source_pins) not in (set(SOURCES), set(LEGACY_SOURCES))
                or (contract.get("schema") == CONTRACT_SCHEMA and set(source_pins) != set(SOURCES))
                or source_pins != proof.get("source_sha256")):
            raise ValueError("Empty tunnel captured source identity is inconsistent")
        for relative, expected in source_pins.items():
            entry = files.get("source:" + relative, {})
            path = episode / entry.get("path", "")
            if (Path(entry.get("path", "")).is_absolute() or not _safe_local_file(path, episode)
                    or entry.get("sha256") != expected or _sha(path) != expected):
                raise ValueError("Empty tunnel captured source missing or changed: " + relative)
        checks["captured_source_snapshot"] = True
        checks["episode_identity"] = metadata.get("scenario_name") == MODE \
            and metadata.get("seed") == proof.get("seed")
        checks.update(_baseline_checks(documents["baseline"], contract))
        checks["collection_completed"] = outcome.get("status") == "mission_completed" \
            and outcome.get("full_mission_completed") is True
        checks["raw_capture_closed"] = documents["raw_seal"].get("status") == "completed"
        checks["outcome_baseline_identity"] = outcome.get("tunnel_mission", {}).get("baseline") == documents["baseline"]
        # Current source may have changed after capture. Only sealed snapshots
        # are inspected; archived code is neither imported nor executed.
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        errors.append(str(error))
    return dict(required=True, valid=bool(checks) and all(checks.values()) and not errors,
                checks=checks, errors=errors, proof_path=str(proof_path),
                historical_source_evidence_only=True,
                physical_semantic_validation_completed=False,
                manual_all_frame_camera_certification=False)
