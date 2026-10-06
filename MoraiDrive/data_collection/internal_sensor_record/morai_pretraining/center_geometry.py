"""Pivot-to-box-centre and visible-return evidence for MORAI object boxes.

MORAI does not report a stable box centre.  In the static36 campaign the
Scooter_02 ``bounding_box.location`` is bit-identical to its actor pivot, while
the measured catalog capture of the same model and scale reported the box
centre at pivot + [0.545, -0.011] m.  V3 only compared W/L/H, so that 0.54 m
BEV error passed.  V4 keeps every V3 gate and adds three independent checks:

* The catalog's own visible LiDAR returns must support the box in all views.
* The observed pivot-to-centre offset must equal the catalog offset.  The one
  known MORAI failure mode, a box reported exactly at the pivot, is corrected
  to pivot + catalog offset; any other disagreement stays masked.
* Catalog scale is calibrated with the Unity Y-up convention when both
  horizontal axes scale equally, so W/L need no axis guess.

Only the horizontal centre is corrected: the BEV target uses X/Y, heading,
length and width.  Sealed GT timelines are never modified.
"""
import json
import math
from pathlib import Path
import statistics

from async_capture.alignment import PayloadReader
from async_capture.journal import digest, verify_run
from async_capture.portable_paths import recorded_episode_directory

from .scaled_object_geometry import positive_vector


DEFAULT_CENTER_POLICY = {
    "center_tolerance_m": .05,
    "pivot_report_tolerance_m": .01,
    "max_center_correction_m": 1.0,
    "min_catalog_returns_per_view": 10,
    "max_catalog_outside_fraction": .15,
    "max_catalog_center_range_m": .02,
    "max_observed_offset_range_m": .01,
    "max_box_pivot_yaw_difference_deg": .5,
}


def pivot_local_offset(actor):
    """Box centre relative to the actor pivot, as in the measured catalog."""
    transform, box = actor["transform"], actor["bounding_box"]
    x, y, z = [float(box["location"][k]) - float(transform["location"][k]) for k in "xyz"]
    roll, pitch, yaw = [math.radians(float(transform["rotation"][k])) for k in "xyz"]
    # Invert Rz(yaw) Ry(pitch) Rx(roll) explicitly, in reverse order.
    x, y = math.cos(yaw) * x + math.sin(yaw) * y, -math.sin(yaw) * x + math.cos(yaw) * y
    x, z = math.cos(pitch) * x - math.sin(pitch) * z, math.sin(pitch) * x + math.cos(pitch) * z
    y, z = math.cos(roll) * y + math.sin(roll) * z, -math.sin(roll) * y + math.cos(roll) * z
    return [x, y, z]


def yaw_difference_deg(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def pedestrian_velocity_parameter(actor):
    """Native pedestrian velocity parameter (km/h), not differenced world motion."""
    if actor.get("actor_info", {}).get("object_type") != "OBJECT_TYPE_PEDESTRIAN":
        return None
    try:
        values = [float(actor["velocity"][axis]) for axis in "xyz"]
        return math.sqrt(sum(v*v for v in values)) if all(math.isfinite(v) for v in values) else None
    except (KeyError, TypeError, ValueError):
        return None


def summarize_pivot_offsets(observations, policy=DEFAULT_CENTER_POLICY):
    """Require one rigid pivot-to-centre offset across every snapshot."""
    result = {}
    for key, rows in observations.items():
        offsets = [row["offset_xyz_m"] for row in rows]
        ranges = [max(o[i] for o in offsets) - min(o[i] for o in offsets) for i in range(3)]
        yaw = max(row["box_pivot_yaw_difference_deg"] for row in rows)
        reason = None
        if max(ranges[:2]) > policy["max_observed_offset_range_m"]:
            reason = "observed_pivot_center_offset_unstable"
        elif yaw > policy["max_box_pivot_yaw_difference_deg"]:
            reason = "box_and_pivot_yaw_differ"
        result[key] = {
            "observations": len(rows),
            "record_ids": sorted({row["record_id"] for row in rows}),
            "offset_xyz_m": [statistics.median(o[i] for o in offsets) for i in range(3)],
            "offset_range_xyz_m": ranges,
            "max_box_pivot_yaw_difference_deg": yaw,
            "max_actor_roll_pitch_deg": max((row.get("actor_roll_pitch_deg", 180.0) for row in rows)),
            "pedestrian_velocity_parameters_kmh": (
                sorted({round(row["pedestrian_velocity_parameter_kmh"], 5) for row in rows})
                if all(row.get("pedestrian_velocity_parameter_kmh") is not None for row in rows)
                else None),
            "valid": reason is None,
            "reason": reason,
        }
    return result


def load_pivot_evidence(episode, active, expected_raw_hash, ssd_root):
    """Read hash-verified raw actor states; no simulator calls."""
    episode = Path(episode)
    index = json.loads((episode / "quality/INDEX_COMPLETE.json").read_text())
    raw = recorded_episode_directory(index["raw_path"], episode, "raw", ssd_root)
    manifest, _ = verify_run(raw, payloads=False)
    if digest(raw / "raw_manifest.json") != expected_raw_hash:
        raise ValueError("Pivot evidence belongs to a different raw recording")
    observations = {}
    reader = PayloadReader(raw)
    try:
        for meta in active:
            if meta["role"] != "actor_inventory":
                continue
            payload = json.loads(reader.read(meta))
            for actor in (payload.get("response") or {}).get("states", []):
                actor_id = str(actor.get("actor_info", {}).get("id", {}).get("value", ""))
                key = payload.get("actor_instances", {}).get(actor_id, {}).get("instance_key")
                if not key or actor_id == "Ego":
                    continue
                try:
                    offset = pivot_local_offset(actor)
                    yaw = yaw_difference_deg(actor["bounding_box"]["rotation"]["z"],
                                             actor["transform"]["rotation"]["z"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not all(math.isfinite(v) for v in offset + [yaw]):
                    continue
                observations.setdefault(key, []).append({
                    "record_id": meta["record_id"], "actor_id": actor_id,
                    "offset_xyz_m": offset, "box_pivot_yaw_difference_deg": yaw,
                    "pedestrian_velocity_parameter_kmh": pedestrian_velocity_parameter(actor),
                    "actor_roll_pitch_deg": max(abs((float(actor["transform"]["rotation"][axis]) + 180) % 360 - 180) for axis in "xy")})
    finally:
        reader.close()
    return {
        "schema": "morai_pivot_center_evidence_v1",
        "raw_manifest_sha256": expected_raw_hash,
        "raw_session_sha256": manifest["session_sha256"],
        "source": "hash_verified_raw_actor_inventory_transform_and_bounding_box",
        "policy": dict(DEFAULT_CENTER_POLICY),
        "instances": summarize_pivot_offsets(observations),
    }


def horizontal_scale_reference(case, scale):
    """Scale catalog W/L/H with Unity Y up; horizontal scaling must be equal."""
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
    evidence = {"catalog_scale_xyz": catalog_scale, "instance_scale_xyz": scale,
                "relative_scale_xyz": ratios}
    horizontal, vertical = ratios[0], ratios[1]
    # The generator writes scale vectors rounded to six decimal places.
    if abs(ratios[2] - horizontal) > max(2e-6, abs(horizontal) * 2e-6):
        return None, dict(evidence, reason="horizontal_relative_scale_unequal")
    lwh = [statistics.median(view["dimensions_lwh_m"][axis] for view in views)
           for axis in range(3)]
    reference = [lwh[1] * horizontal, lwh[0] * horizontal, lwh[2] * vertical]
    return reference, dict(evidence, reason=None,
        horizontal_multiplier=horizontal, vertical_multiplier=vertical,
        scale_policy="unity_y_up_equal_horizontal_scale_v1")


def catalog_return_support(case, policy=DEFAULT_CENTER_POLICY):
    """Every catalog view must have enough returns and few outside the box."""
    views = case.get("views", [])
    rows = [{"view": view.get("view"),
             "foreground_returns": view.get("foreground_returns"),
             "outside_box_fraction": view.get("outside_box_fraction")} for view in views]
    reason = None
    if len(views) != 4:
        reason = "catalog_lacks_four_views"
    elif any(type(row["foreground_returns"]) is not int
             or row["foreground_returns"] < policy["min_catalog_returns_per_view"]
             for row in rows):
        reason = "catalog_visible_returns_insufficient"
    elif any(type(row["outside_box_fraction"]) not in (float, int)
             or not math.isfinite(row["outside_box_fraction"])
             or row["outside_box_fraction"] > policy["max_catalog_outside_fraction"]
             for row in rows):
        reason = "catalog_visible_returns_outside_box"
    return {"views": rows, "reason": reason}


def catalog_center_offset(case, horizontal_multiplier, policy=DEFAULT_CENTER_POLICY):
    views = case.get("views", [])
    offsets = [view.get("local_center_offset_m") for view in views]
    if (len(offsets) != 4 or any(not isinstance(o, list) or len(o) != 3 for o in offsets)
            or any(type(v) not in (float, int) or not math.isfinite(v)
                   for o in offsets for v in o)):
        return None, "catalog_center_offset_missing"
    ranges = [max(o[i] for o in offsets) - min(o[i] for o in offsets) for i in range(2)]
    if max(ranges) > policy["max_catalog_center_range_m"]:
        return None, "catalog_center_offset_unstable"
    return [statistics.median(o[i] for o in offsets) * horizontal_multiplier
            for i in range(2)], None


def resolve_training_center(item, catalog_xy, observed_xy, policy=DEFAULT_CENTER_POLICY):
    """Return the world X/Y used as the BEV box centre, or a masking reason."""
    delta = [catalog_xy[0] - observed_xy[0], catalog_xy[1] - observed_xy[1]]
    error = math.hypot(*delta)
    evidence = {"catalog_center_offset_xy_m": catalog_xy,
                "observed_center_offset_xy_m": observed_xy,
                "center_offset_error_m": error}
    if error <= policy["center_tolerance_m"]:
        source, correction = "api_box_center_consistent_with_catalog", [0.0, 0.0]
    elif math.hypot(*observed_xy) <= policy["pivot_report_tolerance_m"]:
        if error > policy["max_center_correction_m"]:
            return None, dict(evidence, reason="center_correction_exceeds_bev_margin")
        source, correction = "catalog_offset_applied_to_pivot_reported_box", delta
    else:
        return None, dict(evidence, reason="observed_center_offset_differs_from_catalog")
    heading = math.radians(float(item["heading_deg"]))
    c, s = math.cos(heading), math.sin(heading)
    position = item["position"]
    center = [float(position[0]) + c * correction[0] - s * correction[1],
              float(position[1]) + s * correction[0] + c * correction[1]]
    if not all(math.isfinite(v) for v in center):
        return None, dict(evidence, reason="nonfinite_training_center")
    return center, dict(evidence, reason=None, center_source=source,
                        center_correction_local_xy_m=correction)


def check_center_object_geometry(item, catalog_cases, catalog_hash,
                                 absolute_tolerance_m, relative_tolerance,
                                 scale_evidence, pivot_evidence, source_record_ids,
                                 base_checker, policy=DEFAULT_CENTER_POLICY):
    """V2 gates plus scale, visible-return and pivot-centre evidence."""
    result = base_checker(item, catalog_cases, catalog_hash,
                          absolute_tolerance_m, relative_tolerance)
    reasons = list(result["reasons"])
    case = catalog_cases.get(str(item.get("model_name", "")).strip())
    horizontal = 1.0
    if item.get("kind") == "obstacle" and case is not None:
        reasons = [reason for reason in reasons
                   if reason != "observed_dimensions_differ_from_catalog"]
        match = scale_evidence.get("instances", {}).get(item.get("instance_key"), {})
        if not (match.get("valid") is True
                and match.get("actor_id") == str(item.get("id"))
                and match.get("model_name") == item.get("model_name")
                and source_record_ids
                and set(source_record_ids).issubset(match.get("record_ids", []))):
            reasons.append(match.get("reason") or "scale_instance_evidence_missing")
        elif not reasons:
            reference, scale = horizontal_scale_reference(case, match["scale_xyz"])
            result["scale_evidence"] = dict(scale, preset_obstacle_index=match["preset_obstacle_index"])
            if scale["reason"]:
                reasons.append(scale["reason"])
            else:
                horizontal = scale["horizontal_multiplier"]
                tolerances = [max(absolute_tolerance_m, relative_tolerance * value)
                              for value in reference]
                errors = [abs(float(a) - b) for a, b in zip(item["size_wlh_m"], reference)]
                result.update(reference_size_wlh_m=reference, absolute_error_m=errors,
                              tolerance_m=tolerances)
                if any(error > tolerance for error, tolerance in zip(errors, tolerances)):
                    reasons.append("observed_dimensions_differ_from_scaled_catalog")
    if case is not None and not reasons:
        support = catalog_return_support(case, policy)
        result["catalog_return_support"] = support
        if support["reason"]:
            reasons.append(support["reason"])
    if case is not None and not reasons:
        catalog_xy, reason = catalog_center_offset(case, horizontal, policy)
        pivot = pivot_evidence.get("instances", {}).get(item.get("instance_key"), {})
        if reason:
            reasons.append(reason)
        elif not (pivot.get("valid") is True and source_record_ids
                  and set(source_record_ids).issubset(pivot.get("record_ids", []))):
            reasons.append(pivot.get("reason") or "pivot_center_evidence_missing")
        else:
            center, evidence = resolve_training_center(
                item, catalog_xy, pivot["offset_xyz_m"][:2], policy)
            result["center_evidence"] = evidence
            if center is None:
                reasons.append(evidence["reason"])
            else:
                result["training_center_world_xy"] = center
    result.update(reasons=reasons, api_box_geometry_candidate=not reasons)
    return result
