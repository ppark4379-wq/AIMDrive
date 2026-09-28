"""Conservative scale evidence for measured MORAI object geometry.

Catalog dimensions were measured at the catalog's own scale, not necessarily
at [1, 1, 1]. Only proportional changes relative to that scale are calibrated:
this does not guess how Unity scale axes map to MORAI bbox W/L/H axes.
"""
from collections import defaultdict
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
    if preset is None:
        evidence["reason"] = "pinned_preset_unavailable_no_scale_recovery"
        return evidence
    evidence.update(preset= preset, preset_resolved_path=str(path),
                    preset_sha256=digest(path))
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
                    "position": _position(actor.get("transform", {}).get("location"))})
    finally:
        reader.close()
    evidence["instances"] = match_static_instances(preset.get("obstacles", []), observations)
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
