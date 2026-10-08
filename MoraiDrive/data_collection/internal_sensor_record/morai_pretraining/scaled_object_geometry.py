"""Conservative scale evidence for measured MORAI object geometry.

Catalog dimensions were measured at the catalog's own scale, not necessarily
at [1, 1, 1]. Only proportional changes relative to that scale are calibrated:
this does not guess how Unity scale axes map to MORAI bbox W/L/H axes.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

from async_capture.alignment import PayloadReader
from async_capture.journal import digest, verify_run
from async_capture.portable_paths import recorded_episode_directory


def positive_vector(value):
    try:
        return (isinstance(value, list) and len(value) == 3
                and all(math.isfinite(float(x)) and float(x) > 0 for x in value))
    except (ValueError, TypeError, OverflowError):
        return False


def scaled_reference(case, scale):
    """Return reference W/L/H and evidence, without reading observed dimensions."""
    catalog_scale = case.get("scale")
    views = case.get("views", [])
    if not positive_vector(scale) or not positive_vector(catalog_scale):
        return None, {"reason": "invalid_or_missing_catalog_or_instance_scale"}
    if len(views) != 4 or any(
        not positive_vector(view.get("dimensions_lwh_m"))
        or view.get("scale") != catalog_scale for view in views
    ):
        return None, {"reason": "catalog_scale_not_consistent_across_four_views"}
    ratios = [float(a) / float(b) for a, b in zip(scale, catalog_scale)]
    multiplier = statistics.median(ratios)
    evidence = {"catalog_scale_xyz": catalog_scale, "instance_scale_xyz": scale,
                "relative_scale_xyz": ratios,
                "relative_scale_multiplier": multiplier}
    # The generator writes scale vectors rounded to six decimal places.
    if any(abs(ratio - multiplier) > max(2e-6, abs(multiplier) * 2e-6)
           for ratio in ratios):
        return None, dict(evidence, reason="anisotropic_relative_scale_uncalibrated")
    reference = [statistics.median(view["dimensions_lwh_m"][axis] for view in views)
                 * multiplier for axis in (1, 0, 2)]
    return reference, dict(evidence, reason=None,
        scale_policy="proportional_to_measured_catalog_scale_v1")


def _position(value):
    try:
        result = [float(value[axis]) for axis in ("x", "y", "z")]
        return result if all(math.isfinite(v) for v in result) else None
    except (TypeError, KeyError, ValueError):
        return None


def match_static_instances(obstacles, observations, *, position_tolerance_m=.03):
    """Join preset to raw actor incarnations using model and actual pivot pose.

    Position is the actor pivot, not bbox center (offset models must not be
    mis-matched). Every snapshot must agree. Reused IDs, moved actors, unknown
    identities and ambiguous same-model placements remain unresolved.
    """
    matches = {}
    used = defaultdict(list)
    for key, rows in observations.items():
        candidates = []
        for index, obstacle in enumerate(obstacles):
            expected = _position(obstacle.get("position"))
            if expected is None or not positive_vector(obstacle.get("scale")):
                continue
            if rows and all(
                row.get("identity_valid") is True
                and row.get("model") == obstacle.get("model")
                and row.get("position") is not None
                and math.dist(row["position"], expected) <= position_tolerance_m
                for row in rows
            ):
                candidates.append(index)
        row = {"instance_key": key, "observations": len(rows),
               "record_ids": sorted({row["record_id"] for row in rows}),
               "valid": len(candidates) == 1,
               "reason": None if len(candidates) == 1 else "missing_or_ambiguous_preset_instance_match"}
        if len(candidates) == 1:
            index = candidates[0]
            row.update(preset_obstacle_index=index,
                       model_name=obstacles[index]["model"],
                       scale_xyz=obstacles[index]["scale"],
                       actor_id=rows[0]["actor_id"])
            used[index].append(key)
        matches[key] = row
    for keys in used.values():
        if len(keys) > 1:
            for key in keys:
                matches[key].update(valid=False, reason="preset_matches_multiple_actor_incarnations")
    return matches


def resolve_pinned_preset(record, episode, repository_root):
    """Resolve exact bytes locally or at the recorded path; never basename-only."""
    import yaml
    if record is None:
        return None, None
    if not isinstance(record, dict):
        raise ValueError("Pinned static preset record must be an object or null")
    expected = record.get("sha256")
    recorded = Path(record.get("path", ""))
    if not expected or not record.get("path"):
        return None, None
    candidates = [episode / "static_obstacle_preset.yaml",
                  episode.parent / "static_obstacle_preset.yaml", recorded]
    parts = recorded.parts
    if "scenario_runner" in parts:
        relative = Path(*parts[parts.index("scenario_runner"):])
        candidates.extend([repository_root / relative,
                           episode.parent / "software_snapshot/project" / relative])
    for path in candidates:
        if path.is_file() and digest(path) == expected:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("schema") != "morai_static_obstacle_preset_v1":
                raise ValueError("Unsupported pinned static preset schema")
            return value, path
    return None, None


def _tunnel_proof_file(episode, item, relative):
    """Resolve one exact local role, including a staged-directory rename."""
    target = episode / relative
    recorded = Path(item.get("path", ""))
    expected = item.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("Tunnel scale proof lacks an exact file digest")
    suffix = (episode.parent.name, episode.name) + Path(relative).parts
    if (not item.get("path") or ".." in recorded.parts
            or (recorded.is_absolute() and recorded.parts[-len(suffix):] != suffix)
            or (not recorded.is_absolute() and recorded.parts != Path(relative).parts)):
        raise ValueError("Tunnel scale proof role does not belong to this episode")
    for path in (target,) + tuple(target.parents):
        if path.is_symlink():
            raise ValueError("Tunnel scale proof must not traverse a symlink")
        if path == episode:
            break
    if not target.is_file() or digest(target) != expected:
        raise ValueError("Tunnel scale proof file is missing or changed")
    return target


def _json_identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _finite_triple(values):
    return (isinstance(values, list) and len(values) == 3
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values))


def _load_tunnel_scale_proof(episode, raw, manifest, session, expected_raw_hash):
    """Authorize scale only from finalized capture-era fixture evidence.

    Captured runtime pins are compared with the sealed raw session, never with
    today's source tree. Full capture/label admission remains a separate gate.
    """
    from scenario_runner.scenarios.full_loop_parts.tunnel_static_parts.diversity import STRICT_FLAGS, STRICT_CONDITIONS
    metadata = session.get("episode") or {}
    recorded = metadata.get("tunnel_static_trial") or {}
    proof_path = episode / "tunnel_static/COLLECTION_TUNNEL_COMPLETE.json"
    if proof_path.is_symlink():
        raise ValueError("Tunnel scale proof must not be a symlink")
    proof = json.loads(proof_path.read_text())
    if (proof.get("schema") != "aimdrive_tunnel_static_collection_complete_v1"
            or proof.get("scenario") != "full_loop_tunnel_static" or proof.get("status") != "passed"
            or any(proof.get(key) is not True for key in (
                "fixture_cleanup_verified", "scene_restored", "raw_record_complete"))):
        raise ValueError("Tunnel scale requires a finalized successful collection proof")
    roles = {"candidate": "tunnel_static/candidate.json", "preview_report": "tunnel_static/preview.json",
        "drive_report": "tunnel_static/drive_trial.json", "planner_diagnostics": "tunnel_static/planner_diagnostics.json",
        "ego_trace": "tunnel_static/ego_trace.csv", "episode": "episode.json", "outcome": "outcome.json",
        "raw_seal": "raw/RAW_COMPLETE.json", "scene_restore": "tunnel_static/scene_restore.json"}
    paths = {role: _tunnel_proof_file(episode, proof.get("files", {}).get(role, {}), relative)
             for role, relative in roles.items()}
    documents = {role: json.loads(path.read_text()) for role, path in paths.items() if role != "ego_trace"}
    candidate, preview, drive, planner = (documents[key] for key in (
        "candidate", "preview_report", "drive_report", "planner_diagnostics"))
    if (documents["episode"] != metadata or paths["raw_seal"].parent.resolve() != raw.resolve()
            or documents["raw_seal"].get("status") != "completed"
            or documents["raw_seal"].get("manifest_sha256") != expected_raw_hash
            or manifest.get("session_sha256") != digest(raw / "session.json")
            or digest(paths["candidate"]) != recorded.get("selected_candidate_sha256")
            or proof.get("selected_candidate_sha256") != recorded.get("selected_candidate_sha256")
            or not proof.get("library_sha256")):
        raise ValueError("Tunnel scale proof differs from its sealed raw session or selected candidate")
    runtime = recorded.get("runtime_source_sha256")
    sources = proof.get("source_files")
    if (not isinstance(runtime, dict) or len(runtime) != 8
            or runtime != proof.get("runtime_source_sha256") or runtime != drive.get("runtime_source_sha256")
            or not isinstance(sources, dict) or not sources
            or any(not isinstance(path, str) or not isinstance(sha, str) or len(sha) != 64
                   or any(c not in "0123456789abcdef" for c in sha) for path, sha in list(runtime.items())+list(sources.items()))
            or not set(runtime).issubset(sources)
            or any(sources[path] != sha for path, sha in runtime.items())):
        raise ValueError("Tunnel scale proof differs from captured runtime/source identities")
    conditions = drive.get("completion_conditions", {})
    if (drive.get("status") != "passed" or any(drive.get(key) is not True for key in STRICT_FLAGS)
            or any(conditions.get(key) is not True for key in STRICT_CONDITIONS)
            or any(value is not True for value in conditions.values())
            or drive.get("collision_objects") != [] or drive.get("failed_completion_conditions")
            or documents["outcome"].get("status") != "mission_completed"
            or documents["outcome"].get("full_mission_completed") is not True):
        raise ValueError("Tunnel scale requires successful actual-drive and stationary-fixture evidence")
    screen = preview.get("measured_route_screen", {})
    boundary = screen.get("road_boundary_policy")
    boundary_sha = recorded.get("road_boundary_policy_sha256")
    if (preview.get("status") != "valid_driving_trial" or preview.get("planned") != candidate
            or preview.get("cleanup", {}).get("verified") is not True
            or any(not isinstance(preview.get(key), (int, float)) or isinstance(preview[key], bool)
                   or not math.isfinite(preview[key]) or preview[key] < 5.
                   for key in ("hold_seconds", "stationary_observed_seconds"))
            or screen.get("path_found") is not True or screen.get("geometry_checks", {}).get("passed") is not True
            or not boundary or _json_identity(boundary) != boundary_sha
            or any(row.get("road_boundary_policy") != boundary or row.get("road_boundary_policy_sha256") != boundary_sha
                   for row in (screen, drive, planner))
            or planner.get("validated_path_seen") is not True or not planner.get("selected_routes")):
        raise ValueError("Tunnel scale proof lacks matching measured scene and planner guards")
    restoration = documents["scene_restore"]
    parking = restoration.get("parking", {})
    empty_restore = (restoration.get("status") == "nothing_to_restore" and restoration.get("actors") == []
        and restoration.get("scene_was_moved") is False and parking.get("parked") is True
        and (parking.get("external_input_disabled") is True or (
            parking.get("brake_hold_verified") is True and parking.get("control_publishers_absent") is True
            and parking.get("external_input_disabled") is False and parking.get("gear_mode") == 4
            and parking.get("control_mode") == 3
            and all(isinstance(parking.get(key), (int, float)) and not isinstance(parking[key], bool)
                    and math.isfinite(parking[key]) for key in ("brake", "throttle", "speed_norm_raw"))
            and parking["brake"] >= .99 and abs(parking["throttle"]) <= .01 and 0. <= parking["speed_norm_raw"] <= .05)))
    if restoration.get("status") != "restored" and not empty_restore:
        raise ValueError("Tunnel scale proof lacks verified foreign-scene restoration")
    placements, observed, references = candidate.get("obstacles", []), preview.get("observed", []), drive.get("owned_actor_geometry", [])
    count = candidate.get("count")
    if (isinstance(count, bool) or count not in (1, 2, 3) or len(placements) != count
            or len(observed) != count or len(references) != count
            or _json_identity(references) != planner.get("scene_geometry_sha256")):
        raise ValueError("Tunnel scale proof has an incomplete or changed owned scene")
    ids = [str(row.get("raw_actor_id", "")) for row in observed]
    if (len(set(ids)) != count or any(identity in ("", "None", "Ego") for identity in ids)
            or {str(row.get("raw_actor_id")) for row in references} != set(ids)
            or set(drive.get("observed_object_ids", [])) != set(ids)):
        raise ValueError("Tunnel scale proof changed owned actor identity")
    cleanup = preview["cleanup"]
    owner = cleanup.get("fixture_client_key")
    bindings = {str(row.get("raw_actor_id")): row for row in cleanup.get("owned_request_bindings", [])}
    if (not isinstance(owner, str) or not owner or set(bindings) != set(ids)
            or len(cleanup.get("owned_request_bindings", [])) != count):
        raise ValueError("Tunnel scale proof lacks exact fixture ownership")
    expected, static_placements = {}, []
    for placement, row in zip(placements, observed):
        identity = str(row["raw_actor_id"])
        reference = next(item for item in references if str(item["raw_actor_id"]) == identity)
        binding = bindings[identity]
        if (row.get("actor_kind") != placement.get("actor_kind") or row.get("model") != placement.get("model")
                or reference.get("actor_kind") != row.get("actor_kind") or reference.get("model") != row.get("model")
                or any(reference.get(key) != row.get(key) for key in (
                    "position_xyz", "rotation_xyz_deg", "world_bbox", "raw_bbox_normalized", "request_id", "handle_id"))
                or binding.get("verified") is not True or binding.get("source_client_key") != owner
                or binding.get("request_id") != row.get("request_id") or not row.get("request_id")
                or row.get("request_id") != row.get("handle_id")):
            raise ValueError("Tunnel scale proof changed fixture model, body or creating owner")
        if placement.get("actor_kind") != "obstacle":
            continue
        if (not positive_vector(placement.get("scale")) or not _finite_triple(placement.get("scale"))
                or placement["scale"] != [1., 1., 1.]
                or not _finite_triple(row.get("position_xyz"))
                or _position(placement.get("position")) is None
                or math.dist(row["position_xyz"], _position(placement["position"])) > .03
                or not _finite_triple(row.get("global_velocity_raw"))
                or math.sqrt(sum(v*v for v in row["global_velocity_raw"])) > .05
                or not _finite_triple(row.get("angular_velocity_raw"))
                or math.sqrt(sum(v*v for v in row["angular_velocity_raw"])) > .05):
            raise ValueError("Tunnel scale requires explicit unit spawn scale and the exact stationary pivot")
        lock = row.get("stationary_controls_accepted", {}).get("static_transform_lock", [])
        if (len(lock) < 2 or any(item.get("state") != "stable" for item in lock[-2:])
                or not any(item.get("repair", {}).get("physics_disabled") is True
                           and item.get("repair", {}).get("transform_reapplied") is True
                           for item in lock if isinstance(item.get("repair"), dict))):
            raise ValueError("Tunnel scale lacks verified static spawn/freeze evidence")
        expected[identity] = deepcopy(row)
        expected[identity]["fixture_owner"] = owner
        static_placements.append(deepcopy(placement))
    return {"schema": "morai_static_obstacle_preset_v1", "obstacles": static_placements}, expected, {
        "path": str(proof_path), "sha256": digest(proof_path), "file_sha256": {
            role: proof["files"][role]["sha256"] for role in roles},
        "captured_runtime_source_sha256": runtime, "captured_source_sha256": sources,
        "authorization_scope": "exact_fixture_scale_only; capture_and_label_admission_remain_separate"}


def _tunnel_inventory_agrees(actor, expected):
    """Compare recorded native pivot/body/motion to its sealed spawn readback."""
    try:
        xyz = lambda value: [value[axis] for axis in "xyz"]
        actual = [xyz(actor["transform"]["location"]), xyz(actor["transform"]["rotation"]),
                  xyz(actor["bounding_box"]["location"]), xyz(actor["bounding_box"]["extent"]),
                  xyz(actor["bounding_box"]["rotation"]), xyz(actor["global_velocity"]), xyz(actor["angular_velocity"])]
        if not all(_finite_triple(values) for values in actual):
            return False
        position, rotation, center, extent, box_rotation, velocity, angular = actual
        box = expected["world_bbox"]
        angle = lambda a,b: abs((a-b+180.) % 360.-180.)
        return (actor.get("actor_info", {}).get("client_key", "") in ("", expected["fixture_owner"])
            and math.dist(position, expected["position_xyz"]) <= .03
            and math.dist(center, box["center_xyz"]) <= .03 and math.dist(extent, box["extent_xyz"]) <= .01
            and all(angle(a,b) <= .5 for a,b in zip(rotation, expected["rotation_xyz_deg"]))
            and all(angle(a,b) <= .5 for a,b in zip(box_rotation, box["rotation_xyz_deg"]))
            and min(extent) > 0 and math.sqrt(sum(v*v for v in velocity)) <= .05
            and math.sqrt(sum(v*v for v in angular)) <= .05)
    except (KeyError,TypeError,ValueError):
        return False


def load_scale_evidence(episode, active, expected_raw_hash, ssd_root, repository_root):
    """Read only hash-verified raw metadata/payloads, with no simulator calls."""
    episode = Path(episode)
    index = json.loads((episode / "quality/INDEX_COMPLETE.json").read_text())
    raw = recorded_episode_directory(index["raw_path"], episode, "raw", ssd_root)
    manifest, session = verify_run(raw, payloads=False)
    if digest(raw / "raw_manifest.json") != expected_raw_hash:
        raise ValueError("Scale evidence belongs to a different raw recording")
    record = (session.get("episode") or {}).get("static_obstacle_preset", {})
    preset, path = resolve_pinned_preset(record, episode, repository_root)
    evidence = {"schema": "morai_static_scale_evidence_v1",
        "raw_manifest_sha256": expected_raw_hash,
        "raw_session_sha256": manifest["session_sha256"],
        "preset_record": record, "preset_resolved": preset is not None,
        "source": "raw_session_pinned_preset_and_hash_verified_inventory_pivot",
        "instance_position_tolerance_m": .03, "instances": {}}
    tunnel_expected = None
    if preset is None and record is None and (session.get("episode") or {}).get("scenario_name") == "full_loop_tunnel_static":
        preset, tunnel_expected, proof = _load_tunnel_scale_proof(episode, raw, manifest, session, expected_raw_hash)
        evidence.update(preset_resolved=True, source="finalized_tunnel_fixture_scale_bridge_v1", tunnel_proof=proof,
                        preset=preset, preset_sha256=_json_identity(preset))
    elif preset is None:
        evidence["reason"] = "pinned_preset_unavailable_no_scale_recovery"
        return evidence
    else:
        evidence.update(preset=preset, preset_resolved_path=str(path), preset_sha256=digest(path))
    observations = defaultdict(list)
    reader = PayloadReader(raw)
    try:
        for meta in active:
            if meta["role"] != "actor_inventory":
                continue
            payload = json.loads(reader.read(meta))
            for actor in (payload.get("response") or {}).get("states", []):
                info = actor.get("actor_info", {})
                if info.get("object_type") != "OBJECT_TYPE_OBSTACLE":
                    continue
                actor_id = str(info.get("id", {}).get("value", ""))
                identity = payload.get("actor_instances", {}).get(actor_id, {})
                key = identity.get("instance_key")
                if not key:
                    continue
                observations[key].append({"actor_id": actor_id,
                    "identity_valid": identity.get("model_identity_valid") is True,
                    "model": identity.get("model_name"), "record_id": meta["record_id"],
                    "position": _position(actor.get("transform", {}).get("location")),
                    **({"tunnel_body_matches": actor_id in tunnel_expected and _tunnel_inventory_agrees(actor, tunnel_expected[actor_id])}
                       if tunnel_expected is not None else {})})
    finally:
        reader.close()
    evidence["instances"] = match_static_instances(preset.get("obstacles", []), observations)
    if tunnel_expected is not None:
        for key, match in evidence["instances"].items():
            rows = observations[key]
            expected = tunnel_expected.get(match.get("actor_id"))
            if match.get("valid") is True and (expected is None or expected.get("model") != match.get("model_name")
                    or not all(row.get("tunnel_body_matches") is True for row in rows)):
                match.update(valid=False, reason="recorded_instance_differs_from_sealed_tunnel_spawn")
    return evidence


def check_scaled_object_geometry(item, catalog_cases, catalog_hash,
                                 absolute_tolerance_m, relative_tolerance,
                                 evidence, source_record_ids, base_checker):
    """V2 gates plus independent proportional-scale evidence for static actors."""
    result = base_checker(item, catalog_cases, catalog_hash,
                          absolute_tolerance_m, relative_tolerance)
    if item.get("kind") != "obstacle":
        return result
    match = evidence.get("instances", {}).get(item.get("instance_key"), {})
    reasons = [reason for reason in result["reasons"]
               if reason != "observed_dimensions_differ_from_catalog"]
    if not (match.get("valid") is True
            and match.get("actor_id") == str(item.get("id"))
            and match.get("model_name") == item.get("model_name")
            and source_record_ids
            and set(source_record_ids).issubset(match.get("record_ids", []))):
        reasons.append(match.get("reason") or "scale_instance_evidence_missing")
    reference = None
    if not reasons:
        reference, scale = scaled_reference(catalog_cases[item["model_name"]], match["scale_xyz"])
        result["scale_evidence"] = dict(scale, preset_obstacle_index=match["preset_obstacle_index"])
        if scale["reason"]:
            reasons.append(scale["reason"])
    if reference is not None and not reasons:
        tolerances = [max(absolute_tolerance_m, relative_tolerance * value) for value in reference]
        errors = [abs(float(a) - b) for a, b in zip(item["size_wlh_m"], reference)]
        result.update(reference_size_wlh_m=reference, absolute_error_m=errors, tolerance_m=tolerances)
        if any(error > tolerance for error, tolerance in zip(errors, tolerances)):
            reasons.append("observed_dimensions_differ_from_scaled_catalog")
    result.update(reasons=reasons, api_box_geometry_candidate=not reasons)
    return result
