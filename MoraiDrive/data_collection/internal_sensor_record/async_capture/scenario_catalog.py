"""Collection entrypoints, including the separately guarded tunnel mission.

The Scenario Runner's original primary-mode list stays unchanged. Collection
adds the tunnel mode only through its owned-fixture adapter and final evidence
gate, rather than treating it as the ordinary static-obstacle zone.
"""
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path

from scenario_runner.scenarios.segment_modes.modes import (
    PRIMARY_SCENARIOS, get_scenario_zones,
)


TUNNEL_STATIC_SCENARIO = "full_loop_tunnel_static"
COLLECTION_SCENARIOS = tuple(PRIMARY_SCENARIOS) + (TUNNEL_STATIC_SCENARIO,)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Exact registry-only change which records the three reviewed stop-source pins.
# The capture's original pins and all physical evidence stay sealed.
_GOAL_STOP_REGISTRY_PAIR = (
    "28145880c36f24175aa0e87005e6f44a78d6057e3f49203652e973a2e690b129",
    "5f0eb6c56acd0835125d5ffdcf51cc65aa80f777fcf77be0045034c57dcd3f9c",
)


def scenario_zone_contract(scenario_name):
    if scenario_name not in COLLECTION_SCENARIOS:
        raise ValueError("Unsupported collection scenario: " + str(scenario_name))
    zones = asdict(get_scenario_zones(scenario_name))
    if scenario_name != TUNNEL_STATIC_SCENARIO:
        zones.pop("tunnel_static", None)
    return zones


def _reviewed_goal_stop_sources(episode, metadata, proof, drive, raw_seal_path):
    """Retain exactly reviewed producer evidence after the normal-stop repair.

    This permits only the three known runtime hashes and their hash registry.
    It does not change capture, motion, shape, collision or label admission.
    """
    from scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.source_compatibility import (
        PROJECT_ROOT, PRE_MAIN_RUNTIME_SOURCE_SHA256, archived_source_path,
        reviewed_goal_stop_runtime_transition,
    )
    from .journal import digest, verify_run
    from .sensor_record_admission import _safe_local_file
    runtime = proof.get("runtime_source_sha256", {})
    try:
        producer_root = Path(metadata["expert_root"]).parent
        if not producer_root.is_absolute():
            return None
        relative = {path: Path(path).relative_to(producer_root) for path in runtime}
        normalized = {str((PROJECT_ROOT/path).resolve()): runtime[original] for original,path in relative.items()}
        if len(normalized) != len(runtime):
            return None
        # Only the already reviewed pre-main stop repair may supplement a
        # missing original blob. Never compare old evidence with new Expert code.
        transition = reviewed_goal_stop_runtime_transition(normalized, PRE_MAIN_RUNTIME_SOURCE_SHA256)
        if transition is None:
            return None
        current = {original: PRE_MAIN_RUNTIME_SOURCE_SHA256[str((PROJECT_ROOT/path).resolve())]
                   for original,path in relative.items()}
        if any(archived_source_path(path, sha) is None
               for path,sha in PRE_MAIN_RUNTIME_SOURCE_SHA256.items()):
            # Keep the original pre-main validation path when those exact
            # bytes are still current and no main archive is configured.
            if any(not Path(path).is_file() or digest(path) != sha for path,sha in current.items()):
                return None
        expected_changes = {str((producer_root / path).resolve()) for path in (
            "scenario_runner/core/full_loop_drive_base.py",
            "scenario_runner/scenarios/segment_modes/scenario.py",
            "scenario_runner/scenarios/segment_modes/tunnel_static_trial.py",
        )}
        changed = {path for path in runtime if runtime[path] != current[path]}
        if changed != expected_changes:
            return None
        raw_root = Path(raw_seal_path).parent
        for name in ("raw_manifest.json", "session.json"):
            if not _safe_local_file(raw_root / name, episode):
                return None
        manifest, session = verify_run(raw_root, payloads=False)
        producer = session.get("episode", {})
        if (manifest.get("status") != "completed" or producer != metadata
                or producer.get("scenario_name") != TUNNEL_STATIC_SCENARIO
                or producer.get("tunnel_static_trial", {}).get("runtime_source_sha256") != runtime
                or drive.get("runtime_source_sha256") != runtime
                or producer.get("tunnel_static_trial", {}).get("selected_candidate_sha256")
                    != proof.get("selected_candidate_sha256")):
            return None
        sources = proof.get("source_files", {})
        if any(sources.get(path) != expected for path, expected in runtime.items()):
            return None
        registry = str((producer_root /
            "scenario_runner/scenarios/full_loop_parts/tunnel_static_parts/source_compatibility.py").resolve())
        registry_archive = archived_source_path(registry, _GOAL_STOP_REGISTRY_PAIR[1])
        registry_current = Path(registry).is_file() and digest(registry) == _GOAL_STOP_REGISTRY_PAIR[1]
        if sources.get(registry) != _GOAL_STOP_REGISTRY_PAIR[0] or not (registry_archive is not None or registry_current):
            return None
        pairs = {path: (runtime[path], current[path]) for path in changed}
        pairs[registry] = _GOAL_STOP_REGISTRY_PAIR
        return {"scope": transition["scope"], "source_pairs": pairs,
                "producer_source_pins_verified": True,
                "historical_sources_rewritten": False}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _source_pins_match(sources, reviewed=None, archived=None):
    if not isinstance(sources, dict) or not sources:
        return False
    pairs = (reviewed or {}).get("source_pairs", {})
    from scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.source_compatibility import archived_source_path
    for path, expected in sources.items():
        try:
            actual = hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else None
        except (OSError, TypeError):
            return False
        if actual == expected or actual is not None and pairs.get(path) == (expected, actual):
            continue
        snapshot = archived_source_path(path, expected)
        if snapshot is None:
            pair = pairs.get(path)
            if pair and pair[0] == expected:
                snapshot = archived_source_path(path, pair[1])
        if snapshot is None:
            return False
        if archived is not None:
            archived[path] = {"captured_sha256": expected, "resolved_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                              "archive_path": str(snapshot), "historical_source_evidence_only": True}
    return True


def validate_tunnel_static_collection(episode, scenario_name=None, *, map_root=None):
    """Require final capture, physical-drive, identity and restoration evidence.

    Guard artifacts must belong to this episode. Exact recorded digests remain
    authoritative after staging-directory rename; an episode-relative copy may
    resolve the original absolute path, without rewriting a stored hash.
    Native asset and GT-label admission remain the existing separate gates.
    """
    episode = Path(episode).resolve()
    if scenario_name is not None and scenario_name != TUNNEL_STATIC_SCENARIO:
        return {"required": False, "valid": True, "checks": {}, "errors": []}
    try:
        metadata = json.loads((episode / "episode.json").read_text())
    except (OSError,ValueError) as error:
        return {"required": scenario_name == TUNNEL_STATIC_SCENARIO, "valid": False, "checks": {}, "errors": [str(error)]}
    scenario_name = scenario_name or metadata.get("scenario_name")
    if scenario_name != TUNNEL_STATIC_SCENARIO:
        return {"required": False, "valid": True, "checks": {}, "errors": []}
    from scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.diversity import STRICT_FLAGS, STRICT_CONDITIONS
    checks, errors, paths, documents = {}, [], {}, {}
    reviewed_sources = None
    archived_sources = {}
    proof_path = episode / "tunnel_static/COLLECTION_TUNNEL_COMPLETE.json"
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def identity(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),allow_nan=False).encode()).hexdigest()
    try:
        proof = json.loads(proof_path.read_text())
        checks["final_schema"] = proof.get("schema") == "aimdrive_tunnel_static_collection_complete_v1"
        checks["final_passed"] = proof.get("status") == "passed" and proof.get("scenario") == TUNNEL_STATIC_SCENARIO
        for key in ("fixture_cleanup_verified", "scene_restored", "raw_record_complete"):
            checks[key] = proof.get(key) is True
        roles = ("candidate", "preview_report", "drive_report", "planner_diagnostics", "ego_trace",
                 "episode", "outcome", "raw_seal", "scene_restore")
        for role in roles:
            item = proof.get("files", {}).get(role, {})
            recorded = Path(item.get("path", ""))
            if recorded.is_absolute():
                if episode in recorded.resolve().parents:
                    path = recorded
                else:
                    anchors = [i for i,part in enumerate(recorded.parts) if part == episode.name]
                    if not anchors:
                        raise ValueError("Guard proof is outside its capture episode: " + role)
                    path = episode.joinpath(*recorded.parts[anchors[-1]+1:])
            else:
                path = episode / recorded
            from .sensor_record_admission import _safe_local_file
            if not _safe_local_file(path, episode):
                raise ValueError("Missing, changed or unsafe tunnel guard proof: " + role)
            path = path.resolve()
            if episode not in path.parents or not item.get("sha256") or sha(path) != item["sha256"]:
                raise ValueError("Missing, changed or unsafe tunnel guard proof: " + role)
            paths[role] = path
            if role != "ego_trace":
                documents[role] = json.loads(path.read_text())
        checks["sealed_episode_files"] = True
        sources = proof.get("source_files", {})
        candidate, preview, drive, planner = (documents[key] for key in ("candidate", "preview_report", "drive_report", "planner_diagnostics"))
        reviewed_sources = _reviewed_goal_stop_sources(
            episode, metadata, proof, drive, paths["raw_seal"])
        checks["source_pins"] = _source_pins_match(sources, reviewed_sources, archived_sources)
        outcome, raw, restoration = (documents[key] for key in ("outcome", "raw_seal", "scene_restore"))
        screen = preview.get("measured_route_screen", {})
        conditions = drive.get("completion_conditions", {})
        checks["actual_strict_trial"] = (drive.get("status") == "passed" and all(drive.get(key) is True for key in STRICT_FLAGS)
            and all(conditions.get(key) is True for key in STRICT_CONDITIONS) and all(value is True for value in conditions.values())
            and drive.get("collision_objects") == [] and not drive.get("failed_completion_conditions"))
        runtime = proof.get("runtime_source_sha256", {})
        checks["runtime_source_pins"] = (drive.get("runtime_source_sha256") == runtime
            and _source_pins_match(runtime, reviewed_sources, archived_sources))
        checks["measured_path"] = (preview.get("status") == "valid_driving_trial" and preview.get("cleanup", {}).get("verified") is True
            and screen.get("status") == "nominal_path_found" and screen.get("path_found") is True and screen.get("geometry_checks", {}).get("passed") is True)
        checks["planner_live_validation"] = planner.get("validated_path_seen") is True and planner.get("calls", 0) > 0 and bool(planner.get("selected_routes"))
        references = drive.get("owned_actor_geometry", [])
        ids = {str(row.get("raw_actor_id")) for row in references}
        observed_ids = {str(row.get("raw_actor_id")) for row in preview.get("observed", [])}
        checks["owned_actor_identity"] = (candidate.get("count") in (1,2,3) and len(candidate.get("obstacles", [])) == candidate["count"]
            and len(references) == candidate["count"] == len(ids) and "None" not in ids and ids == observed_ids
            and set(drive.get("observed_object_ids", [])) == ids)
        checks["scene_geometry"] = bool(references) and identity(references) == planner.get("scene_geometry_sha256")
        boundary = screen.get("road_boundary_policy")
        boundary_sha = screen.get("road_boundary_policy_sha256")
        checks["road_body_policy"] = bool(boundary) and identity(boundary) == boundary_sha and all(
            obj.get("road_boundary_policy") == boundary and obj.get("road_boundary_policy_sha256") == boundary_sha for obj in (drive,planner))
        trajectory = drive.get("trajectory", [])
        checks["actual_motion"] = (len(trajectory) >= 3 and all(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in row["position_xyz"]) for row in trajectory)
            and max(math.dist(trajectory[0]["position_xyz"], row["position_xyz"]) for row in trajectory) >= .5)
        with paths["ego_trace"].open(newline="") as stream:
            trace = [(float(row["x"]),float(row["y"]),float(row.get("speed",0))) for row in csv.DictReader(stream)]
        checks["finite_csv_motion"] = len(trace) >= 3 and all(all(math.isfinite(v) for v in row) for row in trace) and max(math.dist(trace[0][:2],row[:2]) for row in trace) >= .5
        from scenario_runner.scenarios.full_loop_parts.tunnel_static_parts import diversity
        from scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.layout import tunnel_geometry
        from scenario_runner.utils.map_loader import MGeoMapLoader
        geometry_root = map_root or metadata.get("global_config", {}).get("paths", {}).get("mgeo_root")
        geometry = tunnel_geometry(MGeoMapLoader(str(geometry_root)))
        actual = diversity.path_features([row[:2] for row in trace], geometry, evidence_kind="completed_actual_trace")
        guard_trace = diversity.path_features([row["position_xyz"] for row in trajectory],geometry,evidence_kind="completed_actual_trace")
        checks["csv_matches_guard_trajectory"] = diversity.compare_paths(actual,guard_trace)["duplicate"] is True
        checks["collection_completed"] = outcome.get("status") == "mission_completed" and outcome.get("full_mission_completed") is True
        checks["raw_capture_closed"] = raw.get("status") == "completed"
        parking = restoration.get("parking", {})
        verified_empty_restore = (restoration.get("status") == "nothing_to_restore"
            and restoration.get("actors") == [] and restoration.get("scene_was_moved") is False
            and parking.get("parked") is True and (parking.get("external_input_disabled") is True or (
                parking.get("brake_hold_verified") is True and parking.get("control_publishers_absent") is True
                and parking.get("external_input_disabled") is False
                and parking.get("gear_mode") == 4 and parking.get("control_mode") == 3
                and all(math.isfinite(parking.get(key, float('inf'))) for key in ('brake', 'throttle', 'speed_norm_raw'))
                and parking.get("brake", 0.) >= .99 and abs(parking.get("throttle", 1.)) <= .01
                and 0. <= parking.get("speed_norm_raw", float('inf')) <= .05)))
        checks["foreign_scene_restored"] = restoration.get("status") == "restored" or verified_empty_restore
        checks["selected_layout"] = sha(paths["candidate"]) == proof.get("selected_candidate_sha256") and bool(proof.get("library_sha256"))
    except (OSError,ValueError,TypeError,KeyError,IndexError,AttributeError) as error:
        errors.append(str(error))
    result = {"required": True, "valid": bool(checks) and all(checks.values()) and not errors,
            "checks": checks, "errors": errors, "proof_path": str(proof_path),
            "proof_sha256": sha(proof_path) if proof_path.is_file() else None}
    if reviewed_sources is not None:
        result["reviewed_goal_stop_source_transition"] = reviewed_sources
    if archived_sources:
        result["captured_source_archives"] = archived_sources
        result["current_expert_approval_transferred"] = False
    return result
