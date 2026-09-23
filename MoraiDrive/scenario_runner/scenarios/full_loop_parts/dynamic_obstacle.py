"""Validated, time-aware pedestrian crossings for the full-loop scenario.

The route catalogue is deliberately fail-closed: geometry is checked for every
entry, but only entries with MORAI closed-loop approval participate in normal
runs. A run builds one reproducible, route-ordered schedule and spawns only
the pedestrian that Ego is currently approaching.
"""

import itertools
import hashlib
import json
import math
import os
import random
import time

from ...utils import safe_print as print
from . import traffic_light_override


def _dynamic_cfg(self):
    return dict(self.cfg.get("dynamic_obstacle", {}))


def _signal_guard_cfg(self):
    cfg = dict(_dynamic_cfg(self).get("pre_signal_guard", {}))
    if not bool(cfg.get("enabled", False)):
        return cfg
    positive = (
        "maximum_conflict_to_stopline_m",
        "activation_distance_m",
        "query_period_wall_sec",
        "retry_period_wall_sec",
        "hold_refresh_wall_sec",
        "settle_wall_sec",
        "unknown_grace_wall_sec",
        "red_grace_wall_sec",
        "maximum_command_attempts",
    )
    for key in positive:
        value = float(cfg.get(key, 0.0))
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(
                f"dynamic_obstacle.pre_signal_guard.{key} must be positive"
            )
    target = int(cfg.get("target_color", traffic_light_override.STRAIGHT_GREEN_BIT))
    if not traffic_light_override.target_is_satisfied(
        target, traffic_light_override.STRAIGHT_GREEN_BIT
    ):
        raise RuntimeError(
            "dynamic_obstacle.pre_signal_guard.target_color must permit straight movement"
        )
    cfg["target_color"] = target
    attempts = float(cfg["maximum_command_attempts"])
    if not attempts.is_integer():
        raise RuntimeError(
            "dynamic_obstacle.pre_signal_guard.maximum_command_attempts must be an integer"
        )
    cfg["maximum_command_attempts"] = int(attempts)
    if float(cfg["activation_distance_m"]) < float(
        _dynamic_cfg(self).get("maximum_trigger_distance_m", 60.0)
    ):
        raise RuntimeError(
            "dynamic_obstacle.pre_signal_guard.activation_distance_m must cover "
            "maximum_trigger_distance_m"
        )
    return cfg


def _strict_assessment(self):
    # Collection keeps its existing acceptance contract; explicit review must
    # never silently approve an unobserved response or a corridor warning.
    policy = _dynamic_cfg(self).get("assessment_policy", "strict")
    if policy not in ("report", "strict"):
        raise RuntimeError("dynamic_obstacle.assessment_policy must be report or strict")
    return (policy == "strict" or bool(self.cfg.get("dynamic_obstacle_review"))
            or callable(getattr(self, "observe_collision", None)))


def _assessment_issue(self, code, event=None, **details):
    metrics = self.dynamic_validation
    issues = metrics.setdefault("assessment_issues", [])
    issue = {"code": code, "sequence": event.get("sequence") if event else None}
    if not any(item.get("code") == code and item.get("sequence") == issue["sequence"]
               for item in issues):
        issue.update(details)
        issues.append(issue)
        if event is not None:
            event.setdefault("assessment_issues", []).append(dict(issue))
        print(f"[DYNAMIC] assessment_warning {issue}")
    metrics["safety_validation_passed"] = False


def _read_event_state(self, event):
    """Preserve decorated collector handles, then fall back by exact actor ID.

    A transient miss is deferred for a bounded number of ticks, never replaced
    with a stale position or another pedestrian's state.
    """
    # The collection decorator already retries and verifies object type and
    # uniqueness. Do not bypass its stronger contract after an exception.
    if {"read_states", "attempts"}.issubset(getattr(event["actor"], "__dict__", {})):
        return event["actor"].get_actor_state()
    errors = []
    try:
        state = event["actor"].get_actor_state()
        if state is not None:
            event["state_read_failures"] = 0
            return state
    except Exception as error:
        errors.append(str(error))
    actor_id = event.get("actor_id") or _actor_id(event["actor"])
    try:
        matches = [state for state in self.grpc.get_all_pedestrian_states(strict=True)
                   if str(state.actor_info.id.value).strip() == str(actor_id)
                   and getattr(state.actor_info, "object_type", 2) == 2]
        if len(matches) == 1:
            event["state_read_failures"] = 0
            return matches[0]
        errors.append(f"expected one exact pedestrian, found {len(matches)}")
    except Exception as error:
        errors.append(str(error))
    count = int(event.get("state_read_failures", 0)) + 1
    event["state_read_failures"] = count
    event["state_read_error"] = "; ".join(errors) or "exact actor ID absent"
    if count >= 3:
        _fail_event(self, event, "Failed to read dynamic obstacle state after 3 checks: "
                    + event["state_read_error"])
    return None


def _model_selection_policy(self):
    policy = _dynamic_cfg(self).get("model_selection_policy", "approved_pairs")
    if policy not in ("approved_pairs", "yaml_pool"):
        raise RuntimeError("dynamic_obstacle.model_selection_policy must be approved_pairs or yaml_pool")
    # Explicit review must always test exactly the requested model and speed,
    # including when normal runs sample the wider YAML pool.
    return "review_exact" if self.cfg.get("dynamic_obstacle_review") else policy


def _excluded_model_path_pairs(self):
    """Return exact path/model pairs disabled for ordinary random scenes."""
    raw = _dynamic_cfg(self).get("excluded_model_path_pairs", [])
    if not isinstance(raw, list):
        raise RuntimeError(
            "dynamic_obstacle.excluded_model_path_pairs must be a list"
        )
    excluded = set()
    for item in raw:
        if not isinstance(item, dict):
            raise RuntimeError(
                "dynamic_obstacle.excluded_model_path_pairs entries must be mappings"
            )
        path_id = str(item.get("path_id", "")).strip()
        model = str(item.get("model", "")).strip()
        if not path_id or not model:
            raise RuntimeError(
                "dynamic_obstacle.excluded_model_path_pairs entries require path_id and model"
            )
        pair = (path_id, model)
        if pair in excluded:
            raise RuntimeError(
                "dynamic_obstacle.excluded_model_path_pairs contains a duplicate: "
                f"{path_id}/{model}"
            )
        excluded.add(pair)
    return excluded


def _single_event_probability(cfg):
    value = float(cfg.get("single_event_probability", 0.5))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError(
            "dynamic_obstacle.single_event_probability must be between 0 and 1"
        )
    return value


def _select_random_schedule(scene_random, schedules, cfg):
    """Choose a one- or two-event schedule with an explicit mode ratio."""
    available_counts = [count for count in (1, 2) if schedules.get(count)]
    if not available_counts:
        raise RuntimeError("No one- or two-event dynamic schedule is available")
    if len(available_counts) == 2:
        count = 1 if scene_random.random() < _single_event_probability(cfg) else 2
    else:
        count = available_counts[0]
    return list(scene_random.choice(schedules[count]))


def _ego_speed_mps(ego_state):
    # The gRPC adapter supplies an explicit SI field. The ROS fallback and
    # injected collection observations already provide metres/second.
    return max(0.0, float(ego_state.get("speed_mps", ego_state.get("speed", 0.0))))


def _full_route_points(self):
    window = getattr(self, "mission_route", None)
    if window is not None:
        return list(window.full_points)
    if hasattr(self, "_full_route_points"):
        return list(self._full_route_points)
    return list(self.route_points)


def _point(spec, key):
    value = spec[key]
    return {
        "x": float(value["x"]),
        "y": float(value["y"]),
        "z": float(value.get("z", 0.0)),
    }


def _distance_to_segment(point_x, point_y, start, end):
    dx = end["x"] - start["x"]
    dy = end["y"] - start["y"]
    length_squared = dx * dx + dy * dy
    if length_squared < 1e-9:
        raise RuntimeError("Dynamic obstacle path has zero length")
    ratio = (
        (float(point_x) - start["x"]) * dx
        + (float(point_y) - start["y"]) * dy
    ) / length_squared
    ratio = max(0.0, min(1.0, ratio))
    closest_x = start["x"] + ratio * dx
    closest_y = start["y"] + ratio * dy
    return math.hypot(float(point_x) - closest_x, float(point_y) - closest_y), (
        closest_x,
        closest_y,
        start["z"] + ratio * (end["z"] - start["z"]),
    ), ratio


def _legacy_catalog_entry(spec):
    """Keep old route YAMLs usable while treating the proven path as approved."""
    speed = float(spec.get("speed_mps", 2.5))
    if "speed_kph" in spec:
        converted = float(spec["speed_kph"]) / 3.6
        if "speed_mps" in spec and not math.isclose(speed, converted, abs_tol=1e-9):
            raise RuntimeError("Legacy dynamic speed_mps and speed_kph must agree")
        speed = converted
    return {
        "id": "legacy_right_to_left",
        "route_link": spec.get("route_link"),
        "route_path_index": int(spec["route_path_index"]),
        "start": dict(spec["start"]),
        "end": dict(spec["end"]),
        "allowed_speeds_mps": [speed],
        "trigger_distance_m": float(spec.get("trigger_distance_m", 25.0)),
        "arrival_tolerance_m": float(spec.get("arrival_tolerance_m", 0.8)),
        "max_crossing_sec": float(spec.get("max_crossing_sec", 12.0)),
        "status": "approved",
        "verification": {
            "morai_closed_loop": True,
            "collision_free": True,
            "source": "legacy_proven_full_loop_dynamic",
        },
    }


def _catalogue_entries(spec):
    """Do not silently shadow old single-crossing settings with a catalogue."""
    if "crossing_catalog" not in spec:
        return [_legacy_catalog_entry(spec)]
    entries = spec["crossing_catalog"]
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("dynamic_obstacle.crossing_catalog must be a non-empty list")
    legacy_keys = {
        "route_link", "route_path_index", "start", "end", "conflict_point",
        "speed_mps", "speed_kph", "allowed_speeds_mps", "move_distance_m",
        "trigger_distance_m", "arrival_tolerance_m", "max_crossing_sec",
    }
    misplaced = sorted(legacy_keys.intersection(spec))
    if misplaced:
        raise RuntimeError(
            "Mixed dynamic obstacle settings: " + ", ".join(misplaced)
            + ". Edit the matching crossing_catalog entry instead of the zone root; "
            "use allowed_speeds_mps for speed. Legacy YAML without crossing_catalog "
            "is still supported."
        )
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("Each dynamic crossing_catalog entry must be a mapping")
        aliases = sorted({"speed_mps", "speed_kph", "move_distance_m"}.intersection(entry))
        if aliases or "allowed_speeds_mps" not in entry:
            raise RuntimeError(
                f"Dynamic crossing {entry.get('id', '?')}: use allowed_speeds_mps "
                "only; speed_mps/speed_kph/move_distance_m are not catalogue settings. "
                "Movement distance is calculated from start/end."
            )
    return list(entries)


def _route_intersections(points, start, end, first, last):
    """Exact finite-segment intersections; deduplicate shared route vertices."""
    px, py = end["x"] - start["x"], end["y"] - start["y"]
    hits = []
    for index in range(first, last):
        a, b = points[index], points[index + 1]
        rx, ry = b[0] - a[0], b[1] - a[1]
        denom = px * ry - py * rx
        if abs(denom) < 1e-9:
            continue
        qx, qy = a[0] - start["x"], a[1] - start["y"]
        t = (qx * ry - qy * rx) / denom
        u = (qx * py - qy * px) / denom
        if -1e-8 <= t <= 1 + 1e-8 and -1e-8 <= u <= 1 + 1e-8:
            hit = (start["x"] + t * px, start["y"] + t * py, index, u)
            if not any(math.hypot(hit[0] - old[0], hit[1] - old[1]) < 1e-5 for old in hits):
                hits.append(hit)
    return hits


def _geometry_signature(entry):
    geometry = {key: _point(entry, key) for key in ("start", "end")}
    geometry["route_path_index"] = int(entry["route_path_index"])
    return hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_placement_bounds(entry, zone_spec):
    """Keep the complete straight crossing inside the separate NPC region.

    The driving window also contains approach/recovery road: it is NOT the
    NPC placement area. In a convex rectangle, checking both endpoints with
    an inset also contains every point on the segment. This is a placement
    check, not a surveyed curb/footprint or live terrain certification.
    Legacy external route files without this optional bound remain usable.
    """
    bounds = zone_spec.get("placement_bounds")
    if bounds is None:
        return
    try:
        values = [float(bounds[key]) for key in ("min_x", "max_x", "min_y", "max_y")]
        margin = float(bounds.get("endpoint_margin_m", 0.0))
        if not all(math.isfinite(value) for value in values + [margin]) or margin < 0:
            raise ValueError("non-finite or negative margin")
        xmin, xmax, ymin, ymax = values
        if xmin + margin >= xmax - margin or ymin + margin >= ymax - margin:
            raise ValueError("empty inset")
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise RuntimeError("Dynamic placement_bounds must define finite ordered bounds and a non-negative inset") from error
    for key in ("start", "end"):
        point = _point(entry, key)
        if not (xmin + margin <= point["x"] <= xmax - margin
                and ymin + margin <= point["y"] <= ymax - margin):
            raise RuntimeError(
                f"Dynamic crossing {entry['id']} {key} is outside placement_bounds inset"
            )


def _expand_catalogue(self, raw_entries, spec):
    """Generate finite opposite-roadside endpoint combinations in memory.

    Anchor areas require a site survey. Offsets never inherit approval: each
    changed direction/angle is a pending candidate until individually reviewed.
    No preset files are generated.
    """
    result = list(raw_entries)
    points = _full_route_points(self)
    first, last = int(spec.get("start_path_index", 753)), int(spec.get("end_path_index", 1081))
    for anchor in raw_entries:
        offsets = anchor.get("candidate_endpoint_offsets_m", [])
        overrides = anchor.get("candidate_geometry_overrides", {})
        if not isinstance(overrides, dict) or (overrides and not offsets):
            raise RuntimeError("candidate_geometry_overrides requires a mapping of generated variant IDs")
        if not offsets:
            continue
        if len(offsets) > 5 or not all(math.isfinite(float(value)) and abs(float(value)) <= 5 for value in offsets):
            raise RuntimeError("Dynamic endpoint candidate offsets must be finite, <=5m and at most five samples")
        variant_keys = {
            f"{ai}_{bi}_{'reverse' if reverse else 'forward'}"
            for ai, ao in enumerate(offsets) for bi, bo in enumerate(offsets)
            for reverse in (False, True)
            if reverse or float(ao) != 0 or float(bo) != 0
        }
        for key, geometry in overrides.items():
            if key not in variant_keys:
                raise RuntimeError(f"Unknown candidate_geometry_overrides variant: {key!r}")
            try:
                endpoints = [_point(geometry, name) for name in ("start", "end")]
                if not all(math.isfinite(value) for point in endpoints for value in point.values()):
                    raise ValueError("non-finite endpoint")
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                raise RuntimeError("candidate_geometry_overrides requires finite start/end coordinates") from error
        reverse_geometry = anchor.get("reverse_candidate_geometry")
        if reverse_geometry is not None:
            try:
                reverse_points = [_point(reverse_geometry, key) for key in ("start", "end")]
                if not all(math.isfinite(value) for point in reverse_points for value in point.values()):
                    raise ValueError("non-finite endpoint")
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("reverse_candidate_geometry requires finite start/end coordinates") from error
        index = int(anchor["route_path_index"])
        dx = points[index + 1][0] - points[index - 1][0]
        dy = points[index + 1][1] - points[index - 1][1]
        length = math.hypot(dx, dy)
        for ai, ao in enumerate(offsets):
            for bi, bo in enumerate(offsets):
                for reverse in (False, True):
                    if float(ao) == 0 and float(bo) == 0 and not reverse:
                        continue
                    a, b = _point(anchor, "start"), _point(anchor, "end")
                    # A reverse traversal need not retrace a forward diagonal
                    # into a stopped Ego. Surveyed reverse endpoints may use
                    # a shallower crossing, without changing approved forward
                    # geometry. Each offset variant still requires its own
                    # geometry hash and live approval below.
                    if reverse and reverse_geometry is not None:
                        a = _point(reverse_geometry, "end")
                        b = _point(reverse_geometry, "start")
                    for endpoint, offset in ((a, ao), (b, bo)):
                        endpoint["x"] += dx / length * float(offset)
                        endpoint["y"] += dy / length * float(offset)
                    start, end = (b, a) if reverse else (a, b)
                    variant_key = f"{ai}_{bi}_{'reverse' if reverse else 'forward'}"
                    # A site-specific repair must not move every sibling path.
                    # Absolute endpoints still require the usual geometry hash
                    # and a new live review before this variant is approved.
                    if variant_key in overrides:
                        start = _point(overrides[variant_key], "start")
                        end = _point(overrides[variant_key], "end")
                    hits = _route_intersections(points, start, end, first, last)
                    if len(hits) != 1:
                        if variant_key in overrides:
                            raise RuntimeError(
                                f"candidate_geometry_overrides {variant_key} must cross the dynamic route exactly once"
                            )
                        continue
                    x, y, segment, ratio = hits[0]
                    candidate = dict(anchor)
                    candidate.update(
                        id=f"{anchor['id']}__{ai}_{bi}_{'reverse' if reverse else 'forward'}",
                        start=start, end=end, route_path_index=segment + int(ratio >= 0.5),
                        conflict_point={"x": x, "y": y}, status="pending",
                        verification={"morai_closed_loop": False, "collision_free": False,
                                      "source": "generated_endpoint_candidate_not_approved"},
                        approved_combinations=[],
                    )
                    approval = anchor.get("approved_endpoint_variants", {}).get(variant_key)
                    # Keep prior live evidence in YAML when a near-duplicate
                    # geometry is retired. Disabled variants are still
                    # expanded as pending and can be restored without losing
                    # their review record.
                    if approval:
                        candidate.update(
                            disabled=bool(approval.get("disabled", False)),
                            verification=dict(approval["verification"]),
                            approved_combinations=list(approval["approved_combinations"]),
                        )
                        if not bool(approval.get("disabled", False)):
                            candidate["status"] = "approved"
                    result.append(candidate)
    return result


def _validate_catalog_entry(self, raw, zone_spec, cfg, known_ids):
    entry = dict(raw)
    crossing_id = str(entry.get("id", "")).strip()
    if not crossing_id or crossing_id in known_ids:
        raise RuntimeError(
            "Dynamic crossing IDs must be non-empty and unique: "
            f"{crossing_id!r}"
        )
    known_ids.add(crossing_id)

    status = str(entry.get("status", "pending")).strip().lower()
    if status not in {"pending", "approved"}:
        raise RuntimeError(
            f"Dynamic crossing {crossing_id} has invalid status={status!r}"
        )

    route_index = int(entry["route_path_index"])
    zone_start = int(zone_spec.get("start_path_index", 753))
    zone_end = int(zone_spec.get("end_path_index", 1081))
    points = _full_route_points(self)
    if not zone_start <= route_index <= zone_end or route_index >= len(points):
        raise RuntimeError(
            f"Dynamic crossing {crossing_id} path index {route_index} "
            f"is outside {zone_start}..{zone_end}"
        )
    if route_index <= 0 or route_index >= len(points) - 1:
        raise RuntimeError(
            f"Dynamic crossing {crossing_id} cannot compute a route tangent"
        )

    start = _point(entry, "start")
    end = _point(entry, "end")
    path_dx = end["x"] - start["x"]
    path_dy = end["y"] - start["y"]
    path_length = math.hypot(path_dx, path_dy)
    minimum_path = float(cfg.get("minimum_crossing_path_m", 6.0))
    if path_length < minimum_path:
        raise RuntimeError(
            f"Dynamic crossing {crossing_id} is too short: {path_length:.2f}m"
        )
    if not all(math.isfinite(value) for point in (start, end) for value in point.values()):
        raise RuntimeError(f"Dynamic crossing {crossing_id} contains non-finite coordinates")

    previous = points[route_index - 1]
    following = points[route_index + 1]
    tangent_x = float(following[0]) - float(previous[0])
    tangent_y = float(following[1]) - float(previous[1])
    tangent_length = math.hypot(tangent_x, tangent_y)
    if tangent_length < 1e-6:
        raise RuntimeError(
            f"Dynamic crossing {crossing_id} has an undefined route tangent"
        )
    tangent_x /= tangent_length
    tangent_y /= tangent_length
    route_point = points[route_index]
    start_d = (
        tangent_x * (start["y"] - float(route_point[1]))
        - tangent_y * (start["x"] - float(route_point[0]))
    )
    end_d = (
        tangent_x * (end["y"] - float(route_point[1]))
        - tangent_y * (end["x"] - float(route_point[0]))
    )
    side_clearance = float(cfg.get("minimum_side_clearance_m", 2.5))
    if start_d * end_d >= 0.0 or min(abs(start_d), abs(end_d)) < side_clearance:
        raise RuntimeError(
            f"Dynamic crossing {crossing_id} does not join opposite safe sides: "
            f"start_d={start_d:.2f}, end_d={end_d:.2f}"
        )

    crossing_unit_x = path_dx / path_length
    crossing_unit_y = path_dy / path_length
    crossing_sine = abs(
        tangent_x * crossing_unit_y - tangent_y * crossing_unit_x
    )
    if crossing_sine < float(cfg.get("minimum_crossing_sine", 0.35)):
        raise RuntimeError(
            f"Dynamic crossing {crossing_id} is too parallel to Ego route: "
            f"sine={crossing_sine:.3f}"
        )

    route_error, conflict_point, conflict_ratio = _distance_to_segment(
        route_point[0], route_point[1], start, end,
    )
    if len(_route_intersections(points, start, end, zone_start, zone_end)) != 1:
        raise RuntimeError(f"Dynamic crossing {crossing_id} must intersect Ego route exactly once")
    maximum_route_error = float(cfg.get("maximum_conflict_route_error_m", 1.0))
    if route_error > maximum_route_error or not 0.0 < conflict_ratio < 1.0:
        raise RuntimeError(
            f"Dynamic crossing {crossing_id} misses Ego route: "
            f"error={route_error:.2f}m"
        )
    configured_conflict = entry.get("conflict_point")
    if configured_conflict is not None:
        configured_error = math.hypot(
            float(configured_conflict["x"]) - conflict_point[0],
            float(configured_conflict["y"]) - conflict_point[1],
        )
        if configured_error > maximum_route_error:
            raise RuntimeError(
                f"Dynamic crossing {crossing_id} conflict point differs from "
                f"route projection by {configured_error:.2f}m"
            )

    _validate_placement_bounds(entry, zone_spec)
    verification = dict(entry.get("verification", {}))
    if status == "approved" and (
        crossing_id != "legacy_right_to_left" or "geometry_sha256" in verification
    ) and verification.get("geometry_sha256") != _geometry_signature(entry):
        raise RuntimeError(f"Dynamic crossing {crossing_id} geometry changed since approval")
    if status == "approved" and crossing_id != "legacy_right_to_left":
        if not entry.get("approved_combinations"):
            raise RuntimeError(f"Dynamic crossing {crossing_id} has no reviewed model/speed pair")
    if status == "approved" and not (
        verification.get("morai_closed_loop") is True
        and verification.get("collision_free") is True
    ):
        raise RuntimeError(
            f"Approved dynamic crossing {crossing_id} lacks closed-loop evidence"
        )

    speeds = entry.get(
        "allowed_speeds_mps",
        [entry.get("speed_mps", zone_spec.get("speed_mps", 2.5))],
    )
    speeds = sorted({float(value) for value in speeds
                     if math.isfinite(float(value)) and float(value) > 0.0})
    if not speeds:
        raise RuntimeError(
            f"Dynamic crossing {crossing_id} has no positive approved speed"
        )

    allowed_models = [
        str(value) for value in entry.get("allowed_models", []) if str(value)
    ]
    signal_guard_mode = str(entry.get("signal_guard_mode", "auto")).strip().lower()
    if signal_guard_mode not in {"auto", "through_conflict"}:
        raise RuntimeError(
            f"Dynamic crossing {crossing_id} has invalid "
            f"signal_guard_mode={signal_guard_mode!r}"
        )
    yaw_deg = math.degrees(math.atan2(path_dy, path_dx))
    route_s = float(self.path_s_at_index(route_index))
    return {
        "id": crossing_id,
        "route_link": entry.get("route_link", zone_spec.get("route_link")),
        "route_path_index": route_index,
        "route_s": route_s,
        "start": start,
        "end": end,
        "conflict_point": {
            "x": conflict_point[0],
            "y": conflict_point[1],
            "z": conflict_point[2],
        },
        "start_d_m": start_d,
        "end_d_m": end_d,
        "path_length_m": path_length,
        "start_to_conflict_m": path_length * conflict_ratio,
        "conflict_to_end_m": path_length * (1.0 - conflict_ratio),
        "yaw_deg": yaw_deg,
        "allowed_models": allowed_models,
        "signal_guard_mode": signal_guard_mode,
        "allowed_speeds_mps": speeds,
        "trigger_distance_m": float(
            entry.get(
                "trigger_distance_m",
                zone_spec.get("trigger_distance_m", 25.0),
            )
        ),
        "arrival_tolerance_m": float(
            entry.get(
                "arrival_tolerance_m",
                zone_spec.get("arrival_tolerance_m", 0.8),
            )
        ),
        "max_crossing_sec": float(
            entry.get(
                "max_crossing_sec",
                zone_spec.get("max_crossing_sec", 12.0),
            )
        ),
        "status": status,
        "disabled": bool(entry.get("disabled", False)),
        "verification": verification,
        "approved_combinations": list(entry.get("approved_combinations", [])),
    }


def _schedule_candidates(entries, maximum_count, minimum_spacing_m):
    approved = sorted(
        (entry for entry in entries if entry["status"] == "approved"),
        key=lambda item: (item["route_s"], item["id"]),
    )
    schedules = {}
    for count in range(1, min(int(maximum_count), len(approved)) + 1):
        valid = []
        for combination in itertools.combinations(approved, count):
            if all(
                following["route_s"] - current["route_s"]
                >= float(minimum_spacing_m)
                for current, following in zip(combination[:-1], combination[1:])
            ):
                valid.append(combination)
        if valid:
            schedules[count] = valid
    return schedules


def _choose_schedule(self, catalogue, cfg):
    _single_event_probability(cfg)
    excluded = _excluded_model_path_pairs(self)
    known_paths = {entry["id"] for entry in catalogue}
    unknown_paths = sorted({path_id for path_id, _ in excluded} - known_paths)
    unknown_models = sorted(
        {model for _, model in excluded}
        - set(self.cfg.get("dynamic_obstacle_models", []))
    )
    # A one-entry catalogue is the supported legacy fallback; exclusions for
    # modern catalogue IDs are simply irrelevant there. Multi-entry production
    # catalogues stay fail-closed against misspelled path IDs.
    if (unknown_paths and len(catalogue) > 1) or unknown_models:
        raise RuntimeError(
            "dynamic_obstacle.excluded_model_path_pairs references unknown "
            f"paths={unknown_paths}, models={unknown_models}"
        )
    review = self.cfg.get("dynamic_obstacle_review")
    if review:
        if self.cfg.get("mission_mode") != "full_loop_dynamic" or self.cfg.get("sync_runtime", {}).get("enabled", True):
            raise RuntimeError("Dynamic review requires the asynchronous dynamic-only scenario")
        by_id = {entry["id"]: entry for entry in catalogue}
        selected = []
        for selection in review:
            if selection["id"] not in by_id:
                raise RuntimeError("Unknown dynamic review crossing: " + selection["id"])
            entry = dict(by_id[selection["id"]])
            if selection["model"] not in self.dynamic_obstacle_models:
                raise RuntimeError("Dynamic review model is not an active available YAML model")
            if selection["speed_mps"] not in entry["allowed_speeds_mps"]:
                raise RuntimeError("Dynamic review speed is outside candidate speed steps")
            entry["allowed_models"] = [selection["model"]]
            entry["allowed_speeds_mps"] = [selection["speed_mps"]]
            entry["approved_combinations"] = []  # Explicit trial, not approval.
            selected.append(entry)
        selected.sort(key=lambda entry: entry["route_s"])
        if not 1 <= len(selected) <= 3 or any(
            b["route_s"] - a["route_s"] < float(cfg.get("minimum_event_spacing_m", 35.0))
            for a, b in zip(selected, selected[1:])
        ):
            raise RuntimeError("Dynamic review schedule has unsafe count or spacing")
    else:
        selected = None
    # Filter unavailable model/speed pairs before selecting an event count.
    # A catalogue with an unavailable model must not poison an otherwise valid
    # one- or two-event schedule.
    eligible = []
    for entry in catalogue:
        if _model_speed_pairs(self, entry):
            eligible.append(entry)
    # Production scenes deliberately use at most two sequential interactions.
    maximum_count = max(1, min(2, int(cfg.get("max_count", 2))))
    schedules = _schedule_candidates(
        eligible,
        maximum_count,
        float(cfg.get("minimum_event_spacing_m", 35.0)),
    )
    zone = self.route_definition["zones"]["dynamic_obstacle"]
    zone_start_s = self.path_s_at_index(int(zone.get("start_path_index", 753)))
    schedules = {
        count: [schedule for schedule in choices
                if _schedule_has_timing_room(self, schedule, cfg, zone_start_s)]
        for count, choices in schedules.items()
    }
    schedules = {count: choices for count, choices in schedules.items() if choices}
    if not schedules and selected is None:
        raise RuntimeError(
            "No approved dynamic crossing schedule is available; pending "
            "catalogue entries cannot run before MORAI closed-loop approval"
        )
    # Isolate scene randomization from weather/profile history and unrelated
    # zone setup. The same run_seed therefore reproduces the same complete
    # pedestrian scene even when the previous run's profile file differs.
    scene_random = random.Random(int(self.run_seed) ^ 0x44594E414D4943)
    if selected is None:
        selected = _select_random_schedule(scene_random, schedules, cfg)
    events = []
    for sequence, entry in enumerate(selected, 1):
        pairs = _model_speed_pairs(self, entry)
        previous_s = selected[sequence - 2]["route_s"] if sequence > 1 else zone_start_s
        available = entry["route_s"] - previous_s
        if sequence > 1:
            available -= float(cfg.get("retire_after_conflict_m", 3.0))
        pairs = [pair for pair in pairs if _timing_room(self, entry, pair[1], available, cfg)]
        if not pairs:
            raise RuntimeError(
                f"Dynamic crossing {entry['id']} has no eligible model/speed pair"
            )
        model, speed = scene_random.choice(pairs)
        event = dict(entry)
        event.update(
            sequence=sequence,
            phase="armed" if sequence == 1 else "pending",
            model=model,
            speed_mps=speed,
            model_selection_policy=_model_selection_policy(self),
            actor=None,
            actor_id=None,
            trigger=None,
            response_observed=False,
            response_kind=None,
            resumed_after_clear=False,
            min_ego_distance_m=float("inf"),
            min_ego_speed_mps=float("inf"),
            max_lateral_error_m=0.0,
            failure_reason=None,
            completion_reason=None,
        )
        events.append(event)
    return events


def _model_speed_pairs(self, entry):
    policy = _model_selection_policy(self)
    models = sorted(set(self.dynamic_obstacle_models))
    if policy != "yaml_pool" and entry["allowed_models"]:
        models = [model for model in models if model in entry["allowed_models"]]
    pairs = [(model, speed) for model in models for speed in entry["allowed_speeds_mps"]]
    approved = entry.get("approved_combinations", [])
    if approved:
        permitted = {(str(item["model"]), float(item["speed_mps"])) for item in approved}
        if policy == "yaml_pool":
            # Only broaden models, never promote untested speed steps. Keep
            # approved_combinations unchanged as pair-specific live evidence.
            reviewed_speeds = {speed for _, speed in permitted}
            pairs = [pair for pair in pairs if pair[1] in reviewed_speeds]
        else:
            pairs = [pair for pair in pairs if pair in permitted]
    # Explicit review remains capable of reproducing a blocked pair. Ordinary
    # random selection removes it before drawing, so no redraw can skew other
    # models on the same path.
    if not self.cfg.get("dynamic_obstacle_review"):
        excluded = _excluded_model_path_pairs(self)
        pairs = [pair for pair in pairs if (entry["id"], pair[0]) not in excluded]
    return pairs


def _timing_room(self, entry, npc_speed, available, cfg):
    # Static feasibility over a bounded approach-speed range; the real-time
    # gate still uses measured speed. Include recovery room after retirement.
    available -= float(cfg.get("event_recovery_distance_m", 2.0))
    low = float(cfg.get("minimum_ego_speed_mps", 2.0))
    high = max(low, float(cfg.get("planning_ego_speed_mps", 8.333333)))
    for step in range(9):
        speed = low + (high - low) * step / 8.0
        event = dict(entry, speed_mps=npc_speed)
        timing = _timing_snapshot(self, event, {"speed_mps": speed}, entry["route_s"])
        distance = timing["trigger_distance_m"]
        delta = distance / speed - timing["npc_eta_sec"]
        if (distance <= available and distance >= timing["safe_stop_distance_m"]
                and float(cfg.get("minimum_eta_delta_sec", 0.0)) <= delta
                <= float(cfg.get("maximum_eta_delta_sec", 5.0))):
            return True
    return False


def _schedule_has_timing_room(self, entries, cfg, zone_start_s):
    previous_s = zone_start_s
    for index, entry in enumerate(entries):
        available = entry["route_s"] - previous_s
        if index:
            available -= float(cfg.get("retire_after_conflict_m", 3.0))
        if not any(_timing_room(self, entry, speed, available, cfg)
                   for _, speed in _model_speed_pairs(self, entry)):
            return False
        previous_s = entry["route_s"]
    return True


def _current_event(self):
    events = getattr(self, "dynamic_obstacle_events", [])
    index = int(getattr(self, "dynamic_obstacle_event_index", 0))
    if 0 <= index < len(events):
        return events[index]
    return None


def _apply_event_compatibility_state(self, event):
    self.dynamic_obstacle_start = dict(event["start"])
    self.dynamic_obstacle_end = dict(event["end"])
    self.dynamic_obstacle_path_length_m = float(event["path_length_m"])
    self.dynamic_obstacle_yaw_deg = float(event["yaw_deg"])
    self.dynamic_obstacle_speed_mps = float(event["speed_mps"])
    self.dynamic_obstacle_api_speed_kph = self.dynamic_obstacle_speed_mps * 3.6
    self.dynamic_obstacle_trigger_distance_m = float(
        event["trigger_distance_m"]
    )
    self.dynamic_obstacle_arrival_tolerance_m = float(
        event["arrival_tolerance_m"]
    )
    self.dynamic_obstacle_max_crossing_sec = float(event["max_crossing_sec"])
    self.dynamic_obstacle_route_s = float(event["route_s"])
    self.dynamic_obstacle_model = str(event["model"])
    self.dynamic_obstacle = event.get("actor")
    self.dynamic_obstacle_direction_checked = bool(
        event.get("direction_checked", False)
    )
    self.dynamic_obstacle_direction_respawned = False
    self.dynamic_obstacle_last_print_sec = int(event.get("last_print_sec", -1))
    self.dynamic_obstacle_min_ego_distance_m = float(
        event.get("min_ego_distance_m", float("inf"))
    )


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _event_report(event):
    guard = event.get("signal_guard") or {}
    return {
        "sequence": int(event["sequence"]),
        "crossing_id": event["id"],
        "phase": event["phase"],
        "actor_id": event.get("actor_id"),
        "model": event["model"],
        "planned_model": event.get("planned_model", event["model"]),
        "model_selection_policy": event.get("model_selection_policy", "approved_pairs"),
        "has_pair_review_record": any(
            str(item["model"]) == event["model"]
            and float(item["speed_mps"]) == float(event["speed_mps"])
            for item in event.get("approved_combinations", [])
        ),
        "spawn_attempts": [dict(attempt) for attempt in event.get("spawn_attempts", [])],
        "start": dict(event["start"]),
        "end": dict(event["end"]),
        "conflict_point": dict(event["conflict_point"]),
        "route_path_index": int(event["route_path_index"]),
        "route_s": float(event["route_s"]),
        "yaw_deg": float(event["yaw_deg"]),
        "speed_mps": float(event["speed_mps"]),
        "path_length_m": float(event["path_length_m"]),
        "trigger": event.get("trigger"),
        "response_observed": bool(event.get("response_observed", False)),
        "response_kind": event.get("response_kind"),
        "resumed_after_clear": bool(event.get("resumed_after_clear", False)),
        "min_ego_distance_m": _finite_or_none(
            event.get("min_ego_distance_m", float("inf"))
        ),
        "min_ego_speed_mps": _finite_or_none(
            event.get("min_ego_speed_mps", float("inf"))
        ),
        "max_lateral_error_m": float(event.get("max_lateral_error_m", 0.0)),
        "completion_reason": event.get("completion_reason"),
        "failure_reason": event.get("failure_reason"),
        "assessment_issues": list(event.get("assessment_issues", [])),
        "response_signal_limited": bool(event.get("response_signal_limited")),
        "signal_guard": {
            "required": bool(guard.get("required")),
            "mode": guard.get("mode"),
            "light_id": guard.get("light_id"),
            "stopline_route_s": guard.get("stopline_route_s"),
            "release_route_s": guard.get("release_route_s"),
            "verified": bool(guard.get("verified")),
            "intervention": bool(guard.get("intervention")),
            "command_accepted": bool(guard.get("command_accepted")),
            "command_attempts": int(guard.get("command_attempts", 0)),
            "query_failures": int(guard.get("query_failures", 0)),
            "red_observations": int(guard.get("red_observations", 0)),
            "confounded": bool(guard.get("confounded")),
            "phase": guard.get("phase"),
        },
        "resume_wait_reason": event.get("resume_wait_reason"),
        "state_read_error": event.get("state_read_error"),
        "removed": event.get("actor_id") is not None and event.get("actor") is None,
    }


def dynamic_obstacle_reports(self):
    return [
        _event_report(event)
        for event in getattr(self, "dynamic_obstacle_events", [])
    ]


def _persist_progress(self, during_cleanup=False):
    profile_path_getter = getattr(self, "profile_state_path", None)
    if not callable(profile_path_getter):
        return
    path = profile_path_getter()
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as source:
            document = json.load(source)
        if not isinstance(document, dict):
            return
        if document.get("run_seed") != getattr(self, "run_seed", None):
            return  # Setup failed before this run's profile was installed.
        document["dynamic_obstacle_phase"] = getattr(
            self, "dynamic_obstacle_phase", "unknown"
        )
        document["dynamic_obstacle_events"] = dynamic_obstacle_reports(self)
        document["dynamic_obstacle_model"] = self.dynamic_obstacle_model
        document["dynamic_validation"] = dict(getattr(self, "dynamic_validation", {}))
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as output:
            json.dump(document, output, indent=2)
            output.write("\n")
        os.replace(temporary, path)
    except Exception as error:
        metrics = getattr(self, "dynamic_validation", {})
        message = f"Failed to persist dynamic obstacle progress: {error}"
        if message not in metrics.setdefault("recording_errors", []):
            metrics["recording_errors"].append(message)
            print(f"[DYNAMIC] {message}")
        metrics["safety_validation_passed"] = False
        # Preserve the primary failure, especially while unwinding cleanup.
        if not during_cleanup and _strict_assessment(self):
            raise RuntimeError(message) from error


def _emit_event(self, name, event, **fields):
    payload = {
        "crossing_id": event["id"],
        # ``RawScenario.event`` owns the generic ``sequence`` field for the
        # collection-wide event order.  Keep the pedestrian's order in its
        # own namespace so multi-event evidence cannot collide with it.
        "dynamic_sequence": int(event["sequence"]),
        "model": event["model"],
        "route_s": float(event["route_s"]),
    }
    payload.update(fields)
    print(f"[DYNAMIC] {name} {payload}")
    writer = getattr(self, "event", None)
    if callable(writer):
        writer(name, training_visible=not bool(self.cfg.get("dynamic_obstacle_review")), **payload)


def spawn_dynamic_obstacle(self):
    _strict_assessment(self)  # Reject invalid policy before creating actors.
    zone_spec = self.route_definition["zones"]["dynamic_obstacle"]
    cfg = _dynamic_cfg(self)
    catalogue_raw = _catalogue_entries(zone_spec)
    catalogue_raw = _expand_catalogue(self, catalogue_raw, zone_spec)
    known_ids = set()
    catalogue = [
        _validate_catalog_entry(self, raw, zone_spec, cfg, known_ids)
        for raw in catalogue_raw
    ]
    self.dynamic_obstacle_catalog = catalogue
    self.dynamic_obstacle_events = _choose_schedule(self, catalogue, cfg)
    if bool(_signal_guard_cfg(self).get("enabled", False)) and not getattr(
        self, "traffic_light_stops", []
    ):
        raise RuntimeError(
            "Dynamic pre-signal guard is enabled but the active route has no "
            "verified traffic-light stop"
        )
    for event in self.dynamic_obstacle_events:
        _initialize_signal_guard(self, event)
    self.dynamic_obstacle_event_index = 0
    self.dynamic_obstacle_owned_ids = set()
    self.dynamic_validation = {
        "review_only": bool(self.cfg.get("dynamic_obstacle_review")),
        "goal_reached": False,
        "cleanup_verified": False,
        "max_ego_center_error_m": 0.0,
        "corridor_observations": 0,
        "max_ego_body_lateral_extent_m": 0.0,
        "collision_objects": [],
        "safety_validation_passed": False,
        "failure_reason": None,
    }
    self.dynamic_obstacle_phase = "waiting"
    self.dynamic_obstacle_started_at = None
    self.dynamic_obstacle_started_scenario_time = None
    self.dynamic_obstacle = None
    first = self.dynamic_obstacle_events[0]
    _apply_event_compatibility_state(self, first)
    schedule_text = ", ".join(
        f"{event['sequence']}:{event['id']}@{event['route_s']:.1f}m/"
        f"{event['model']}/{event['speed_mps']:.2f}mps"
        for event in self.dynamic_obstacle_events
    )
    print(
        f"[FullLoop] dynamic schedule seed={self.run_seed}, "
        f"model_policy={_model_selection_policy(self)}, "
        f"events={len(self.dynamic_obstacle_events)} [{schedule_text}]"
    )


def _actor_id(actor):
    getter = getattr(actor, "get_id", None)
    if not callable(getter):
        return None
    value = str(getter() or "").strip()
    return value or None


def _spawn_active_dynamic_obstacle(self, yaw_offset_deg=0.0):
    event = _current_event(self)
    if event is None or event["phase"] != "armed":
        raise RuntimeError("No armed dynamic obstacle event is available")
    start = event["start"]
    stale_ids = self.grpc.destroy_pedestrians_near(
        start["x"], start["y"], max_dist_m=3.0,
    )
    if stale_ids:
        time.sleep(0.2)
        remaining = self.grpc.find_spawned_pedestrian_near(
            start["x"], start["y"], max_dist_m=3.0,
        )
        if remaining is not None:
            raise RuntimeError(
                "Failed to clear stale pedestrian at dynamic obstacle start: "
                f"actor_id={remaining.get_id()}"
            )

    # Keep the seeded choice first, then try other models eligible under the
    # same selection policy and speed. Stable ordering must not depend on the
    # previous run or the order returned by MORAI's model inventory.
    planned_model = event.setdefault("planned_model", event["model"])
    attempts = event.setdefault("spawn_attempts", [])
    tried = {attempt["model"] for attempt in attempts}
    alternatives = sorted({
        model for model, speed in _model_speed_pairs(self, event)
        if speed == float(event["speed_mps"]) and model != planned_model
    })
    models = [model for model in [planned_model] + alternatives if model not in tried]
    if not models:
        _fail_event(self, event, "Failed to spawn every eligible dynamic obstacle model")
    selected_model = models[0]
    attempt = {
        "model": selected_model,
        "scenario_time": self._scenario_now(),
        "trigger": dict(event.get("trigger") or {}),
        "result": "requested",
    }
    attempts.append(attempt)
    try:
        actor = self.grpc.spawn_pedestrian(
            transform=self.grpc.make_transform(
                start["x"], start["y"], start["z"],
                event["yaw_deg"] + float(yaw_offset_deg),
            ),
            model_name=selected_model,
            label=(
                f"full_loop_dynamic_{self.profile_index}_"
                f"{event['sequence']}_{event['id']}"
            ),
            velocity=float(event["speed_mps"]) * 3.6,
            active_dist=0.0,
            move_dist=float(event["path_length_m"]) + 2.0,
            start_action=True,
        )
    except Exception as error:
        # An RPC exception is not proof that no actor was created. Preserve
        # the existing stop-on-exception behaviour; do not create duplicates.
        attempt.update(result="error", error=str(error))
        _fail_event(self, event, f"Dynamic obstacle spawn raised: {error}")
    if actor is None:
        attempt["result"] = "failed"
        if len(models) == 1:
            _fail_event(
                self, event,
                "Failed to spawn every eligible dynamic obstacle model: "
                f"{[item['model'] for item in attempts]}",
            )
        _emit_event(
            self, "dynamic_obstacle_spawn_retry_pending", event,
            failed_model=selected_model, next_model=models[1],
        )
        _persist_progress(self)
        # Stay armed. Do not loop with the old Ego state: the next timeline
        # tick must pass the signal, speed, ETA and stopping-distance gates
        # again before a fallback model may be spawned.
        return

    attempt["result"] = (
        "recovered" if getattr(actor, "_scenario_runner_ambiguous_recovery", False)
        is True else "spawned"
    )
    event["actor"] = actor
    event["actor_id"] = _actor_id(actor)
    event["model"] = selected_model
    event["phase"] = "crossing"
    event["started_scenario_time"] = self._scenario_now()
    event["started_at"] = event["started_scenario_time"]
    event["spawn_yaw_offset_deg"] = float(yaw_offset_deg)
    if event["actor_id"]:
        self.dynamic_obstacle_owned_ids.add(event["actor_id"])
    self.scenario_actors.append(actor)
    if not event["actor_id"]:
        raise RuntimeError("Spawned dynamic pedestrian has no verifiable actor ID")
    self.dynamic_obstacle_phase = "crossing"
    self.dynamic_obstacle_started_scenario_time = event["started_scenario_time"]
    self.dynamic_obstacle_started_at = event["started_at"]
    _apply_event_compatibility_state(self, event)
    _emit_event(
        self,
        "dynamic_obstacle_started",
        event,
        actor_id=event["actor_id"],
        yaw_deg=float(event["yaw_deg"]),
        speed_mps=float(event["speed_mps"]),
        trigger=event.get("trigger"),
    )
    _persist_progress(self)


def _owned_actor_is_present(self, actor_id):
    if not actor_id or not hasattr(self.grpc, "get_all_pedestrian_states"):
        raise RuntimeError("Cannot verify pedestrian removal without actor ID and inventory")
    try:
        states = self.grpc.get_all_pedestrian_states(strict=True)
    except TypeError:
        # Narrow compatibility for mocked/older adapters used by offline tests.
        states = self.grpc.get_all_pedestrian_states()
    for state in states:
        state_id = str(state.actor_info.id.value).strip()
        if state_id == str(actor_id):
            return True
    return False


def _destroy_event_actor(self, event, strict=True):
    actor = event.get("actor")
    if actor is None:
        return
    actor_id = event.get("actor_id") or _actor_id(actor)
    result, present, error_text = None, True, None
    absent_checks = 0
    for attempt in range(3):
        # Retry deletion only when still present/unknown. An ambiguous ACK is
        # accepted only after two independent exact-ID absence observations.
        if absent_checks == 0:
            try:
                result = actor.destroy()
            except Exception as error:
                error_text = str(error)
                result = False
        try:
            present = _owned_actor_is_present(self, actor_id)
            absent_checks = absent_checks + 1 if not present else 0
            if not present and (result is not False or absent_checks >= 2):
                break
        except Exception as error:
            error_text = str(error)
            present = True
            absent_checks = 0
        if attempt < 2:
            time.sleep(0.1)
    if present or (result is False and absent_checks < 2):
        message = (
            f"Failed to remove dynamic pedestrian actor_id={actor_id}: "
            f"destroy_result={result}, still_present={present}, error={error_text}"
        )
        if strict:
            raise RuntimeError(message)
        print(f"[FullLoop] {message}")
        return
    if actor in self.scenario_actors:
        self.scenario_actors.remove(actor)
    event["actor"] = None
    if self.dynamic_obstacle is actor:
        self.dynamic_obstacle = None


def _destroy_dynamic_obstacle_actor(self):
    event = _current_event(self)
    if event is not None:
        _destroy_event_actor(self, event, strict=True)


def _fail_event(self, event, reason):
    event["phase"] = "failed"
    event["failure_reason"] = str(reason)
    self.dynamic_obstacle_phase = "failed"
    cleanup_error = None
    try:
        _destroy_event_actor(self, event, strict=True)
    except Exception as error:
        cleanup_error = error
    try:
        _emit_event(self, "dynamic_obstacle_failed", event, reason=str(reason))
    except Exception as error:
        self.dynamic_validation.setdefault("recording_errors", []).append(str(error))
    _persist_progress(self, during_cleanup=True)
    if cleanup_error is not None:
        raise RuntimeError(f"{reason}; cleanup also failed: {cleanup_error}")
    raise RuntimeError(str(reason))


def _finish_dynamic_obstacle(self, reason):
    event = _current_event(self)
    actor = event.get("actor") if event is not None else self.dynamic_obstacle
    if actor is not None:
        start = event["start"] if event is not None else self.dynamic_obstacle_start
        end = event["end"] if event is not None else self.dynamic_obstacle_end
        dx = float(end["x"]) - float(start["x"])
        dy = float(end["y"]) - float(start["y"])
        length = math.hypot(dx, dy)
        if length < 1e-3:
            raise RuntimeError("Dynamic obstacle stop direction is undefined")
        stopped = self.grpc.control_pedestrian(
            actor, dx / length, dy / length, 0.0, verbose=True,
        )
        if not stopped:
            if event is not None:
                _fail_event(
                    self,
                    event,
                    "Failed to stop dynamic obstacle at endpoint",
                )
            raise RuntimeError("Failed to stop dynamic obstacle at endpoint")

    if event is None:
        self.dynamic_obstacle_phase = "cleared"
        return
    event["phase"] = "cleared"
    event["completion_reason"] = str(reason)
    event["cleared_scenario_time"] = self._scenario_now()
    self.dynamic_obstacle_phase = "cleared"
    elapsed = event["cleared_scenario_time"] - float(
        event.get("started_scenario_time", event["cleared_scenario_time"])
    )
    min_ego = event["min_ego_distance_m"]
    min_ego_text = "unknown" if not math.isfinite(min_ego) else f"{min_ego:.2f}m"
    _emit_event(
        self,
        "dynamic_obstacle_cleared",
        event,
        reason=str(reason),
        elapsed_sec=float(elapsed),
        min_ego_distance_m=_finite_or_none(min_ego),
        response_observed=bool(event["response_observed"]),
    )
    print(
        f"[FullLoop] dynamic obstacle DONE id={event['id']} reason={reason}, "
        f"elapsed={elapsed:.1f}s, min_ego_distance={min_ego_text}"
    )
    cfg = _dynamic_cfg(self)
    guard = event.get("signal_guard") or {}
    signal_confounded = bool(
        event.get("response_signal_limited") or guard.get("confounded")
    )
    response_valid = bool(event["response_observed"]) and not signal_confounded
    guard_valid = not guard.get("required") or bool(guard.get("verified"))
    if bool(cfg.get("require_ego_response", True)) and not (
        response_valid and guard_valid
    ):
        reason = (
            "response_unverified_signal"
            if signal_confounded or not guard_valid
            else "response_not_observed"
        )
        _assessment_issue(self, reason, event)
        if _strict_assessment(self):
            _fail_event(self, event,
                        f"Dynamic crossing {event['id']} completed without meaningful Ego response ({reason})")
    _persist_progress(self)


def _initialize_signal_guard(self, event):
    cfg = _signal_guard_cfg(self)
    stop = None
    mode = str(event.get("signal_guard_mode", "auto"))
    if bool(cfg.get("enabled", False)):
        if mode == "through_conflict":
            maximum_gap = float(cfg.get("maximum_stopline_to_conflict_m", 5.0))
            stop = next(
                (
                    item
                    for item in getattr(self, "traffic_light_stops", [])
                    if 0.0
                    <= float(event["route_s"]) - float(item["route_s"])
                    <= maximum_gap
                ),
                None,
            )
            if stop is None:
                raise RuntimeError(
                    f"Dynamic crossing {event['id']} requires a signal guard "
                    "through its conflict point, but no nearby preceding stop line exists"
                )
        else:
            maximum_gap = float(cfg["maximum_conflict_to_stopline_m"])
            stop = next(
                (
                    item
                    for item in getattr(self, "traffic_light_stops", [])
                    if 0.0
                    <= float(item["route_s"]) - float(event["route_s"])
                    <= maximum_gap
                ),
                None,
            )
    release_route_s = None
    if stop is not None:
        release_route_s = float(stop["route_s"])
        if mode == "through_conflict":
            release_route_s = float(event["route_s"]) + float(
                _dynamic_cfg(self).get("retire_after_conflict_m", 3.0)
            )
    event["signal_guard"] = {
        "required": stop is not None,
        "mode": mode if stop is not None else "not_required",
        "light_id": str(stop["light_id"]) if stop is not None else None,
        "stopline_route_s": float(stop["route_s"]) if stop is not None else None,
        "release_route_s": release_route_s,
        "verified": False,
        "intervention": False,
        "command_accepted": False,
        "command_attempts": 0,
        "query_failures": 0,
        "confounded": False,
        "phase": "pending" if stop is not None else "not_required",
        "last_query_wall": float("-inf"),
        "last_command_wall": float("-inf"),
        "last_verified_wall": None,
        "verified_since_wall": None,
        "unknown_since_wall": None,
        "red_since_wall": None,
        "red_observations": 0,
    }


def _cache_guard_signal(self, light_id, blocked):
    self._dynamic_signal_cache = {
        "light_id": str(light_id),
        "checked_at": self._scenario_now(),
        "blocked": blocked,
    }


def _emit_signal_guard(self, name, event, **fields):
    guard = event["signal_guard"]
    payload = {
        "crossing_id": event["id"],
        "dynamic_sequence": int(event["sequence"]),
        "light_id": guard["light_id"],
        "stopline_route_s": float(guard["stopline_route_s"]),
        "release_route_s": float(guard["release_route_s"]),
        "guard_mode": guard["mode"],
        "intervention": True,
        "mission_distribution_changed": True,
    }
    payload.update(fields)
    traffic_light_override._emit_collection_event(self, name, **payload)


def _command_signal_guard(self, event, now):
    cfg = _signal_guard_cfg(self)
    guard = event["signal_guard"]
    maximum_attempts = int(cfg["maximum_command_attempts"])
    if guard["command_attempts"] >= maximum_attempts:
        return False
    if now - float(guard["last_command_wall"]) < float(
        cfg["retry_period_wall_sec"]
    ):
        return False
    guard["command_attempts"] += 1
    guard["last_command_wall"] = now
    guard["intervention"] = True
    result = self.grpc.set_traffic_light_color_verified(
        guard["light_id"],
        int(cfg["target_color"]),
        impulse=bool(cfg.get("impulse", True)),
        set_sibling=bool(cfg.get("set_sibling", True)),
    )
    after = result.get("after") if isinstance(result, dict) else None
    accepted = bool(isinstance(result, dict) and result.get("accepted"))
    guard["command_accepted"] = bool(
        guard.get("command_accepted") or accepted
    )
    verified = bool(
        accepted
        and isinstance(after, dict)
        and str(after.get("id")) == guard["light_id"]
        and traffic_light_override.target_is_satisfied(
            after.get("color"), cfg["target_color"]
        )
    )
    _emit_signal_guard(
        self,
        "dynamic_signal_guard_command",
        event,
        command_attempt=int(guard["command_attempts"]),
        result=result,
        verified=verified,
    )
    if verified:
        if guard.get("verified_since_wall") is None:
            guard["verified_since_wall"] = now
        guard["verified"] = True
        guard["phase"] = (
            "confounded_recovered" if guard.get("confounded") else "green"
        )
        guard["last_verified_wall"] = now
        guard["unknown_since_wall"] = None
        guard["red_since_wall"] = None
        _cache_guard_signal(self, guard["light_id"], False)
    return verified


def _update_signal_guard(self, event, current_s, crossing):
    """Verify and hold green for one explicitly eligible crossing.

    Before spawn, failure to prove green keeps the actor out of the world.
    During crossing, a short query outage is tolerated, but an observed red or
    prolonged unknown marks the interaction as signal-confounded.
    """
    guard = event.get("signal_guard") or {}
    if not guard.get("required"):
        return True
    cfg = _signal_guard_cfg(self)
    # Legacy pre-stopline paths release at the stop line. Explicit front-
    # crosswalk paths keep green until Ego has passed the NPC conflict point
    # and its retirement margin. No unrelated route or scenario is affected.
    if crossing and float(current_s) > float(guard["release_route_s"]):
        release_name = (
            "conflict" if guard.get("mode") == "through_conflict" else "stopline"
        )
        guard["phase"] = (
            f"confounded_after_{release_name}" if guard.get("confounded")
            else f"released_after_{release_name}"
        )
        return True
    remaining = float(event["route_s"]) - float(current_s)
    if not crossing and remaining > float(cfg["activation_distance_m"]):
        return False
    now = time.monotonic()
    def ready():
        if not guard.get("verified"):
            return False
        if crossing:
            return True
        since = guard.get("verified_since_wall")
        return since is not None and now - float(since) >= float(
            cfg["settle_wall_sec"]
        )

    query_due = now - float(guard["last_query_wall"]) >= float(
        cfg["query_period_wall_sec"]
    )
    if not query_due:
        return ready()
    guard["last_query_wall"] = now
    state = None
    try:
        state = self.grpc.get_traffic_light_state_by_id(guard["light_id"])
    except Exception as error:
        print(f"[DYNAMIC] signal guard query unavailable: {error}")
    known = isinstance(state, dict) and str(state.get("id")) == guard["light_id"]
    satisfied = False
    if known:
        try:
            satisfied = traffic_light_override.target_is_satisfied(
                state.get("color"), cfg["target_color"]
            )
        except (TypeError, ValueError, OverflowError):
            known = False
    if known and satisfied and guard.get("command_accepted"):
        first_verification = not guard.get("verified")
        if guard.get("verified_since_wall") is None:
            guard["verified_since_wall"] = now
        guard["verified"] = True
        guard["phase"] = (
            "confounded_recovered" if guard.get("confounded") else "green"
        )
        guard["last_verified_wall"] = now
        guard["unknown_since_wall"] = None
        guard["red_since_wall"] = None
        _cache_guard_signal(self, guard["light_id"], False)
        if first_verification:
            _emit_signal_guard(
                self,
                "dynamic_signal_guard_verified",
                event,
                observed_color=int(state["color"]),
                command_attempts=int(guard["command_attempts"]),
            )
        # MORAI resumes its normal signal cycle after an impulse command.
        # Refresh only this guarded signal while Ego is still before its stop
        # line, so long diagonal crossings cannot drift back to red.
        if crossing and now - float(guard["last_command_wall"]) >= float(
            cfg["hold_refresh_wall_sec"]
        ):
            _command_signal_guard(self, event, now)
        return ready()
    if not known:
        guard["query_failures"] += 1
        if guard["unknown_since_wall"] is None:
            guard["unknown_since_wall"] = now
        if crossing and now - guard["unknown_since_wall"] >= float(
            cfg["unknown_grace_wall_sec"]
        ):
            guard["confounded"] = True
            event["response_signal_limited"] = True
            event["response_speed_samples"] = []
            guard["phase"] = "confounded_unknown"
        return False if not crossing else ready()
    guard["unknown_since_wall"] = None
    blocked = bool(
        known
        and traffic_light_override.straight_movement_is_blocked(
            state.get("color")
        )
    )
    if crossing and blocked:
        guard["red_observations"] += 1
        if guard.get("red_since_wall") is None:
            guard["red_since_wall"] = now
        guard["phase"] = "correcting_red"
        if now - float(guard["red_since_wall"]) >= float(
            cfg["red_grace_wall_sec"]
        ):
            guard["confounded"] = True
            event["response_signal_limited"] = True
            event["response_speed_samples"] = []
            guard["phase"] = "confounded_red"
    elif known:
        guard["red_since_wall"] = None
    # Command once even when naturally green so MORAI holds straight movement
    # through the pedestrian crossing. A non-target but non-red transition is
    # proactively corrected without falsely crediting it as a red-light stop.
    if not guard.get("command_accepted") or not satisfied:
        _command_signal_guard(self, event, now)
        return ready()
    return ready()


def _signal_wait_active(self, ego_state):
    threshold = float(_dynamic_cfg(self).get("stopped_speed_mps", 0.5))
    return (
        _ego_speed_mps(ego_state) <= threshold
        and getattr(self, "traffic_wait_state", None) is not None
    )


def _red_route_signal_ahead(self, current_s):
    """True=red/waiting, False=clear/no signal, None=temporarily unknown.

    Like the legacy traffic-light helper, an unavailable query skips this
    tick instead of aborting the run. Unknown must not count as green.
    """
    if getattr(self, "traffic_wait_state", None) is not None:
        return True
    stops = list(getattr(self, "traffic_light_stops", []))
    if not stops or not hasattr(self.grpc, "get_traffic_light_state_by_id"):
        return False
    maximum_distance = float(
        self.cfg.get("traffic_light_wait_override", {}).get(
            "maximum_stopline_distance_m", 25.0
        )
    )
    upcoming = traffic_light_override.upcoming_stop(self, current_s, maximum_distance)
    if upcoming is None:
        return False

    now = self._scenario_now()
    cache = getattr(self, "_dynamic_signal_cache", None)
    if (
        isinstance(cache, dict)
        and cache.get("light_id") == str(upcoming["light_id"])
        and now - float(cache.get("checked_at", float("-inf"))) < 0.25
    ):
        return cache.get("blocked")
    try:
        state = self.grpc.get_traffic_light_state_by_id(upcoming["light_id"])
    except Exception as error:
        # The standard adapter already converts RPC failures to None. Keep
        # the same non-fatal policy for other compatible query adapters.
        print(f"[DYNAMIC] signal query unavailable: {error}")
        state = None
    blocked = None
    if isinstance(state, dict) and str(state.get("id")) == str(
        upcoming["light_id"]
    ):
        try:
            color = int(state["color"])
            blocked = bool(color & 1) and not bool(color & 16)
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
    if blocked is None and now - getattr(self, "_dynamic_signal_warning_at", float("-inf")) >= 5.0:
        print(
            f"[DYNAMIC] signal_query_deferred light_id={upcoming['light_id']}: "
            "state unavailable; skip this signal-dependent check and retry"
        )
        self._dynamic_signal_warning_at = now
    self._dynamic_signal_cache = {
        "light_id": str(upcoming["light_id"]),
        "checked_at": now,
        "blocked": blocked,
    }
    return blocked


def _timing_snapshot(self, event, ego_state, current_s):
    cfg = _dynamic_cfg(self)
    speed = _ego_speed_mps(ego_state)
    remaining = float(event["route_s"]) - float(current_s)
    npc_eta = float(event["start_to_conflict_m"]) / float(event["speed_mps"])
    interaction_lead = float(cfg.get("interaction_lead_sec", 1.5))
    minimum_trigger = max(
        float(cfg.get("minimum_trigger_distance_m", 12.0)),
        float(event.get("trigger_distance_m", 0.0)),
    )
    maximum_trigger = float(cfg.get("maximum_trigger_distance_m", 60.0))
    reaction = float(cfg.get("reaction_time_sec", 1.0))
    deceleration = max(
        0.1, float(cfg.get("comfortable_deceleration_mps2", 3.5))
    )
    safety_margin = float(cfg.get("stopping_safety_margin_m", 3.0))
    stop_distance = speed * reaction + speed * speed / (2.0 * deceleration)
    stop_distance += safety_margin
    trigger_distance = speed * (npc_eta + interaction_lead)
    trigger_distance = max(
        minimum_trigger, trigger_distance, stop_distance + 1.0
    )
    trigger_distance = min(maximum_trigger, trigger_distance)
    ego_eta = remaining / max(speed, 0.1)
    return {
        "remaining_m": remaining,
        "ego_speed_mps": speed,
        "ego_eta_sec": ego_eta,
        "npc_eta_sec": npc_eta,
        "eta_delta_sec": ego_eta - npc_eta,
        "trigger_distance_m": trigger_distance,
        "safe_stop_distance_m": stop_distance,
    }


def _observe_ego_response(self, event, ego_state, current_s):
    cfg = _dynamic_cfg(self)
    guard = event.get("signal_guard") or {}
    if guard.get("required"):
        signal_limited = bool(
            guard.get("confounded")
            or not guard.get("verified")
            or _signal_wait_active(self, ego_state)
        )
    else:
        signal_limited = (
            _signal_wait_active(self, ego_state)
            or _red_route_signal_ahead(self, current_s) is not False
        )
    if signal_limited:
        # A red/unknown signal blocks new response evidence, but must not
        # retroactively erase a slowdown/stop already observed while the
        # guarded light was verified green. Sustained guard failure is marked
        # separately by guard.confounded in _update_signal_guard().
        event["response_speed_samples"] = []
        return
    remaining = float(event["route_s"]) - float(current_s)
    if abs(remaining) > float(cfg.get("response_observation_distance_m", 30.0)):
        return
    speed = _ego_speed_mps(ego_state)
    event["min_ego_speed_mps"] = min(event["min_ego_speed_mps"], speed)
    trigger = event.get("trigger") or {}
    now = self._scenario_now()
    samples = [(stamp, value) for stamp, value in event.get("response_speed_samples", [])
               if 0 <= now - stamp <= 4.0]
    # Initial speed remains usable only until a signal interrupts observation.
    if not event.get("response_samples_initialized") and not event.get("response_signal_limited"):
        samples.append((now, float(trigger.get("ego_speed_mps", speed))))
    event["response_samples_initialized"] = True
    samples.append((now, speed))
    event["response_speed_samples"] = samples
    baseline = max(value for _, value in samples)
    # Do not credit slowdown for an NPC still stuck at its spawn point.
    if float(event.get("observed_along_m", 0.0)) < float(cfg.get("minimum_movement_m", 0.4)):
        return
    stopped_threshold = float(cfg.get("stopped_speed_mps", 0.5))
    slowdown = baseline - speed
    response_kind = None
    if speed <= stopped_threshold and baseline >= float(
        cfg.get("minimum_ego_speed_mps", 1.0)
    ):
        response_kind = "stop"
    elif slowdown >= float(cfg.get("meaningful_slowdown_mps", 1.0)):
        response_kind = "slowdown"
    if response_kind is not None and (
        not event["response_observed"]
        or (response_kind == "stop" and event["response_kind"] != "stop")
    ):
        event["response_observed"] = True
        event["response_kind"] = response_kind
        _emit_event(
            self,
            "dynamic_obstacle_ego_response",
            event,
            response_kind=response_kind,
            baseline_speed_mps=baseline,
            observed_speed_mps=speed,
            route_remaining_m=remaining,
        )
        _persist_progress(self)


def _advance_after_clear(self, event, ego_state, current_s):
    cfg = _dynamic_cfg(self)
    now = self._scenario_now()
    speed = _ego_speed_mps(ego_state)
    if (event.get("signal_guard") or {}).get("required"):
        _update_signal_guard(self, event, current_s, crossing=True)
    actor = event.get("actor")
    if actor is not None:
        state = _read_event_state(self, event)
        if state is None:
            return
        distance = math.hypot(float(state.transform.location.x) - event["end"]["x"],
                              float(state.transform.location.y) - event["end"]["y"])
        if distance > max(1.0, event["arrival_tolerance_m"] + 0.5):
            _fail_event(self, event, "Pedestrian did not remain stopped at approved endpoint")
    retire_after = float(cfg.get("retire_after_conflict_m", 3.0))
    passed = float(current_s) >= float(event["route_s"]) + retire_after
    signal = _red_route_signal_ahead(self, current_s)
    previous_check = float(event.get("resume_checked_at", event.get("cleared_scenario_time", now)))
    delta = max(0.0, now - previous_check)
    event["resume_checked_at"] = now
    blocked = _signal_wait_active(self, ego_state) or signal is not False
    # Once Ego has actually passed this NPC, a subsequent junction's red
    # signal cannot keep the completed NPC/event alive.
    if passed and (event.get("resumed_after_clear") or speed >= float(cfg.get("resume_speed_mps", 1.0))):
        blocked = False
        event["resumed_after_clear"] = True
    previous_blocked = event.get("resume_was_blocked", blocked)
    event["resume_was_blocked"] = blocked
    if blocked:
        event["resume_wait_reason"] = "signal_unknown" if signal is None else "signal_red"
        event["resume_signal_wait_sec"] = float(event.get("resume_signal_wait_sec", 0.0)) + delta
        if event["resume_signal_wait_sec"] >= float(cfg.get("resume_signal_wait_limit_sec", 60.0)):
            _fail_event(self, event, "Dynamic resume signal wait exhausted: " + event["resume_wait_reason"])
        return
    event["resume_wait_reason"] = "awaiting_ego_progress"
    active_delta = 0.0 if previous_blocked else delta
    event["resume_unblocked_sec"] = float(event.get("resume_unblocked_sec", 0.0)) + active_delta
    last_progress = float(event.get("resume_progress_s", current_s))
    progressed = float(current_s) >= last_progress + 0.5
    if progressed:
        event["resume_progress_s"] = float(current_s)
        event["resume_stall_sec"] = 0.0
        event["resumed_after_clear"] = True
    else:
        event.setdefault("resume_progress_s", float(current_s))
        event["resume_stall_sec"] = float(event.get("resume_stall_sec", 0.0)) + active_delta
    if (
        not _signal_wait_active(self, ego_state)
        and speed >= float(cfg.get("resume_speed_mps", 1.0))
    ):
        event["resumed_after_clear"] = True
    resume_timeout = float(cfg.get("resume_timeout_sec", 20.0))
    if (
        not passed and event["resume_stall_sec"] >= resume_timeout
    ):
        _assessment_issue(self, "resume_delayed", event)
        if _strict_assessment(self) or event["resume_stall_sec"] >= float(cfg.get("resume_failure_timeout_sec", 40.0)):
            _fail_event(self, event, f"Ego did not resume after dynamic crossing {event['id']}")

    if not event["resumed_after_clear"] or not passed:
        return
    _destroy_event_actor(self, event, strict=True)
    _emit_event(
        self,
        "dynamic_obstacle_removed",
        event,
        ego_route_s=float(current_s),
    )
    self.dynamic_obstacle_event_index += 1
    following = _current_event(self)
    if following is None:
        self.dynamic_obstacle_phase = "cleared"
        _persist_progress(self)
        return
    following["phase"] = "armed"
    self.dynamic_obstacle_phase = "waiting"
    _apply_event_compatibility_state(self, following)
    _emit_event(self, "dynamic_obstacle_armed", following)
    _persist_progress(self)


def _collection_records_collisions(self):
    # Only the existing collection supervisor owns the record/abort policy.
    # A normal/review runner must never bypass collision rejection just by
    # gaining a similarly named setting.
    return (
        callable(getattr(self, "observe_collision", None))
        and getattr(self, "collision_policy", None) == "record"
        and not self.cfg.get("dynamic_obstacle_review")
    )


def _observe_dynamic_zone(self, ego_state, current_s):
    """Monitor a conservative body corridor without changing Expert control."""
    if not getattr(self, "dynamic_obstacle_events", []):
        return
    has_pose = all(key in ego_state for key in ("x", "y", "yaw_deg"))
    collisions = [str(item) for item in ego_state.get("collision_objects", []) if str(item)]
    if not has_pose and not collisions:
        return
    spec = self.route_definition["zones"]["dynamic_obstacle"]
    first, last = int(spec.get("start_path_index", 753)), int(spec.get("end_path_index", 1081))
    if not self.path_s_at_index(first) <= current_s <= self.path_s_at_index(last):
        return
    metrics = self.dynamic_validation
    if collisions:
        previous = metrics["collision_objects"]
        metrics["collision_objects"] = sorted(set(previous + collisions))
        metrics["safety_validation_passed"] = False
        if metrics["collision_objects"] != previous:
            _persist_progress(self)
        if not _collection_records_collisions(self):
            raise RuntimeError("Collision in dynamic zone: " + str(collisions))
    if not has_pose:
        return
    points = _full_route_points(self)
    best = None
    for index in range(first, last):
        a, b = points[index], points[index + 1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-9:
            continue
        ratio = max(0.0, min(1.0, ((ego_state["x"] - a[0]) * dx
                                  + (ego_state["y"] - a[1]) * dy) / length_sq))
        error = math.hypot(ego_state["x"] - a[0] - ratio * dx,
                           ego_state["y"] - a[1] - ratio * dy)
        if best is None or error < best[0]:
            best = (error, math.atan2(dy, dx))
    if best is None:
        raise RuntimeError("Cannot verify Ego corridor in dynamic zone")
    cfg = _dynamic_cfg(self)
    error, tangent = best
    relative_yaw = math.radians(float(ego_state["yaw_deg"])) - tangent
    extent = (error + float(cfg.get("ego_body_width_m", 2.0)) / 2 * abs(math.cos(relative_yaw))
              + float(cfg.get("ego_body_length_m", 4.635)) / 2 * abs(math.sin(relative_yaw)))
    metrics["corridor_observations"] += 1
    metrics["max_ego_center_error_m"] = max(metrics["max_ego_center_error_m"], error)
    metrics["max_ego_body_lateral_extent_m"] = max(metrics["max_ego_body_lateral_extent_m"], extent)
    limit = float(cfg.get("ego_lane_half_width_m", 1.75)) - float(cfg.get("ego_lane_margin_m", 0.15))
    if extent > limit:
        _assessment_issue(self, "conservative_corridor_exceeded", extent_m=extent, limit_m=limit)
        if _strict_assessment(self):
            raise RuntimeError(f"Ego left conservative dynamic lane corridor: {extent:.3f}m > {limit:.3f}m")


def update_dynamic_obstacle(self, ego_state, current_s):
    _observe_dynamic_zone(self, ego_state, current_s)
    event = _current_event(self)
    if event is None:
        return
    phase = event["phase"]
    if phase == "pending":
        return
    if phase == "cleared":
        _advance_after_clear(self, event, ego_state, current_s)
        return
    if phase == "failed":
        raise RuntimeError(
            f"Dynamic crossing {event['id']} is already failed: "
            f"{event['failure_reason']}"
        )

    if phase == "armed":
        timing = _timing_snapshot(self, event, ego_state, current_s)
        passed_tolerance = float(
            _dynamic_cfg(self).get("missed_trigger_tolerance_m", 1.0)
        )
        if timing["remaining_m"] < -passed_tolerance:
            _fail_event(
                self,
                event,
                f"missed_trigger: Ego passed dynamic crossing {event['id']}"
                + (" (signal unavailable during trigger window)" if event.get("trigger_signal_unknown") else ""),
            )
        guard = event.get("signal_guard") or {}
        if guard.get("required"):
            if not _update_signal_guard(self, event, current_s, crossing=False):
                if timing["remaining_m"] < timing["safe_stop_distance_m"]:
                    _fail_event(
                        self,
                        event,
                        "signal_guard_unavailable_before_safe_trigger: "
                        f"remaining={timing['remaining_m']:.2f}m "
                        f"safe_stop={timing['safe_stop_distance_m']:.2f}m",
                    )
                return
            signal = False
        else:
            if _signal_wait_active(self, ego_state):
                return
            signal = _red_route_signal_ahead(self, current_s)
            if signal is True:
                return
            if signal is None:
                event["trigger_signal_unknown"] = True
                if _strict_assessment(self):
                    return
                # Unknown is NOT green. Normal execution can still use the same
                # speed/ETA/braking gates; retain uncertainty, never approve it.
        minimum_speed = float(
            _dynamic_cfg(self).get("minimum_ego_speed_mps", 1.0)
        )
        if timing["ego_speed_mps"] < minimum_speed:
            return
        if timing["remaining_m"] > timing["trigger_distance_m"]:
            return
        if timing["remaining_m"] < timing["safe_stop_distance_m"]:
            _fail_event(
                self,
                event,
                "unsafe_trigger_window: "
                f"remaining={timing['remaining_m']:.2f}m "
                f"safe_stop={timing['safe_stop_distance_m']:.2f}m",
            )
        cfg = _dynamic_cfg(self)
        if timing["eta_delta_sec"] > float(cfg.get("maximum_eta_delta_sec", 5.0)):
            return  # Too early: keep the actor unspawned and reassess next tick.
        if timing["eta_delta_sec"] < float(cfg.get("minimum_eta_delta_sec", 0.0)):
            _fail_event(self, event, "unsafe_trigger_window: NPC would arrive after Ego")
        if signal is None:
            _assessment_issue(self, "trigger_signal_unverified", event)
        timing["trigger_scenario_time"] = self._scenario_now()
        timing["ego_route_s"] = float(current_s)
        event["trigger"] = timing
        print(
            f"[FullLoop] dynamic obstacle TRIGGER id={event['id']} "
            f"route_remaining={timing['remaining_m']:.2f}m, "
            f"ego_eta={timing['ego_eta_sec']:.2f}s, "
            f"npc_eta={timing['npc_eta_sec']:.2f}s, "
            f"safe_stop={timing['safe_stop_distance_m']:.2f}m"
        )
        # Keep subclass hooks: Sensor Record collection decorates the actor
        # with its existing exact-ID inventory reader after the base spawn.
        self._spawn_active_dynamic_obstacle()
        actor = event.get("actor")
        if actor is not None and self.dynamic_obstacle is not actor:
            observed = self.dynamic_obstacle
            if observed is None or _actor_id(observed) != event["actor_id"]:
                raise RuntimeError("Dynamic spawn hook changed the pedestrian actor identity")
            # Crossing, endpoint checks and retirement must all use the same
            # decorated handle. The collector may already have updated the
            # cleanup list; preserve that replacement without duplicating it.
            event["actor"] = observed
            self.scenario_actors = [
                observed if owned is actor else owned
                for owned in self.scenario_actors
            ]
        return

    if phase != "crossing" or event.get("actor") is None:
        return
    if (event.get("signal_guard") or {}).get("required"):
        _update_signal_guard(self, event, current_s, crossing=True)
    actor = event["actor"]
    state = _read_event_state(self, event)
    if state is None:
        return
    x = float(state.transform.location.x)
    y = float(state.transform.location.y)
    start = event["start"]
    end = event["end"]
    path_dx = end["x"] - start["x"]
    path_dy = end["y"] - start["y"]
    path_length = event["path_length_m"]
    along = (
        (x - start["x"]) * path_dx + (y - start["y"]) * path_dy
    ) / path_length
    lateral_error = abs(
        (x - start["x"]) * path_dy - (y - start["y"]) * path_dx
    ) / path_length
    distance_to_end = math.hypot(x - end["x"], y - end["y"])
    ego_distance = math.hypot(
        x - float(ego_state["x"]), y - float(ego_state["y"])
    )
    event["min_ego_distance_m"] = min(
        event["min_ego_distance_m"], ego_distance
    )
    event["max_lateral_error_m"] = max(
        event["max_lateral_error_m"], lateral_error
    )
    event["observed_along_m"] = along
    _observe_ego_response(self, event, ego_state, current_s)

    now = self._scenario_now()
    elapsed = now - float(event.get("started_scenario_time", now))
    print_second = int(elapsed)
    if print_second != int(event.get("last_print_sec", -1)):
        event["last_print_sec"] = print_second
        self.dynamic_obstacle_last_print_sec = print_second
        print(
            f"[FullLoop] dynamic crossing id={event['id']} t={elapsed:.1f}s "
            f"pos=({x:.2f},{y:.2f}) along={along:.2f}m "
            f"lateral={lateral_error:.2f}m to_end={distance_to_end:.2f}m "
            f"ego_distance={ego_distance:.2f}m"
        )

    displacement = math.hypot(x - start["x"], y - start["y"])
    movement_check_sec = float(
        _dynamic_cfg(self).get("movement_check_sec", 2.0)
    )
    movement_check_m = float(
        _dynamic_cfg(self).get("minimum_movement_m", 0.4)
    )
    if elapsed >= movement_check_sec and displacement < movement_check_m:
        _fail_event(
            self,
            event,
            f"Dynamic obstacle did not leave spawn: id={event['id']}, "
            f"model={event['model']}, displacement={displacement:.2f}m, "
            f"elapsed={elapsed:.1f}s",
        )

    if displacement >= movement_check_m:
        event["direction_checked"] = True
        self.dynamic_obstacle_direction_checked = True
        max_lateral = float(
            _dynamic_cfg(self).get("maximum_path_deviation_m", 1.0)
        )
        if along < -0.2 or lateral_error > max_lateral:
            _fail_event(
                self,
                event,
                f"Dynamic obstacle left approved crossing: id={event['id']}, "
                f"along={along:.2f}m, lateral={lateral_error:.2f}m",
            )

    self.dynamic_obstacle_min_ego_distance_m = event["min_ego_distance_m"]
    if distance_to_end <= event["arrival_tolerance_m"]:
        _finish_dynamic_obstacle(self, "reached_dest")
        return
    if along >= path_length:
        _finish_dynamic_obstacle(self, "passed_dest")
        return
    if elapsed >= event["max_crossing_sec"]:
        _fail_event(
            self,
            event,
            f"Dynamic obstacle crossing timeout: id={event['id']}, "
            f"model={event['model']}, along={along:.2f}m, "
            f"lateral={lateral_error:.2f}m, to_end={distance_to_end:.2f}m",
        )


def verify_dynamic_obstacles_removed(self):
    owned = set(getattr(self, "dynamic_obstacle_owned_ids", set()))
    if not owned or not hasattr(self.grpc, "get_all_pedestrian_states"):
        return
    try:
        states = self.grpc.get_all_pedestrian_states(strict=True)
    except TypeError:
        states = self.grpc.get_all_pedestrian_states()
    present = {
        str(state.actor_info.id.value).strip()
        for state in states
    }
    remaining = sorted(owned & present)
    if remaining:
        raise RuntimeError(
            f"Runner-owned dynamic pedestrians remain after cleanup: {remaining}"
        )
