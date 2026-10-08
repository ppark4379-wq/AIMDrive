"""Compare production paths and completed traces in measured tunnel coordinates.

This module has no ROS, SDK, NumPy or Shapely dependency.  A path is projected
onto the mapped own lane and, when supplied, its connected mapped departure.
Absolute stations are retained: translating a maneuver along the road changes
its signature.  Model names, requested patterns and recorded route names are
not path features.  The selector keeps their *coverage* as a separate concern.

The defaults suppress centimetre tracking noise with a 3 m running median and
1 m resampling.  A duplicate requires <= 0.10 m mean and <= 0.40 m maximum
lateral discrepancy, compatible excursion/turn sequences and return stations
within 6 m.  These are study thresholds, not driving safety tolerances.  A
fingerprint is a reproducible, quantized audit key; comparison, rather than
hash equality, determines behavioral duplicates.  Missing, nonfinite, reversed
or insufficient coverage is unknown and never evidence of novelty.
"""

import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .layout import polyline_stations, project_polyline, route_pose


SCHEMA = "aimdrive_tunnel_path_diversity_v1"
DEFAULT_POLICY = {
    "sample_step_m": 1.0,
    "smoothing_window_m": 3.0,
    "minimum_coverage_fraction": 0.80,
    "minimum_samples": 10,
    "maximum_backward_step_m": 0.35,
    "maximum_source_gap_m": 4.0,
    "departure_window_m": 20.0,
    "mean_duplicate_m": 0.10,
    "max_duplicate_m": 0.40,
    "width_duplicate_m": 0.40,
    "transition_duplicate_m": 6.0,
    "return_duplicate_m": 6.0,
    "lane_hysteresis_m": 0.25,
    "minimum_run_m": 2.0,
    "center_excursion_m": 0.65,
    "center_return_m": 0.60,
    "turn_slope_min": 0.04,
    "turn_amplitude_min_m": 0.30,
    "fingerprint_lateral_quantum_m": 0.10,
}
STRICT_FLAGS = ("completed", "collision_free", "returned_to_original_lane",
                "fixture_stationary", "expert_verified", "live_avoidance_verified",
                "goal_reached", "ego_lane_return_verified", "ego_stopped_and_parked")
STRICT_CONDITIONS = (
    "mapped_entry_observed", "mapped_exit_observed", "minimum_route_progress",
    "center_rejoined_original_route", "whole_body_in_original_lane", "heading_aligned",
    "all_owned_ros_objects_verified", "selected_path_observed", "collision_free",
    "final_fixture_and_road_checks_passed", "actual_jamming_recorded_path_validated",
    "scene_geometry_matches", "planner_boundary_policy_matches",
    "planner_recording_and_policy_sources_current", "runtime_handoff_sources_current",
)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _policy(policy):
    result = dict(DEFAULT_POLICY)
    if policy:
        unknown = set(policy)-set(result)
        if unknown:
            raise ValueError("Unknown diversity policy: " + ", ".join(sorted(unknown)))
        result.update(policy)
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Diversity policy must be finite: " + key)
        if value <= 0 and key != "departure_window_m":
            raise ValueError("Diversity policy must be positive: " + key)
    if not 0 < result["minimum_coverage_fraction"] <= 1 or result["departure_window_m"] < 0:
        raise ValueError("Invalid diversity coverage policy")
    return result


def _reference(geometry, policy):
    own = [list(point[:3]) for point in geometry["own_points"]]
    if len(own) < 2 or any(len(point) < 2 or not all(math.isfinite(float(v)) for v in point) for point in own):
        raise ValueError("Invalid mapped own lane")
    own_length = polyline_stations(own)[-1]
    if own_length <= 0:
        raise ValueError("Mapped own lane has no length")
    opposing = geometry["opposing_points"]
    if (len(opposing) < 2 or any(len(point) < 2 or not all(math.isfinite(float(v)) for v in point) for point in opposing)
            or polyline_stations(opposing)[-1] <= 0):
        raise ValueError("Invalid mapped opposing lane")
    lane_width = geometry.get("own_lane_width_m", 3.5)
    if isinstance(lane_width, bool) or not math.isfinite(float(lane_width)) or float(lane_width) <= 0:
        raise ValueError("Invalid mapped own lane width")
    points = list(own)
    departure = geometry.get("departure_points", [])
    if departure and policy["departure_window_m"]:
        departure = [list(point[:3]) for point in departure]
        if len(departure) < 2 or math.dist(own[-1][:2], departure[0][:2]) > 0.05:
            raise ValueError("Mapped departure must connect to the own lane")
        if any(not all(math.isfinite(float(v)) for v in point) for point in departure):
            raise ValueError("Nonfinite mapped departure")
        available = polyline_stations(departure)[-1]
        extension = min(policy["departure_window_m"], available)
        stations = polyline_stations(departure)
        points.extend(point for point, station in zip(departure[1:], stations[1:]) if station < extension)
        end, _ = route_pose(departure, extension)
        points.append(list(end))
    expected = polyline_stations(points)[-1]
    return points, own_length, expected, _hash(points)


def _unknown(reason, policy, evidence_kind, **facts):
    return dict(schema=SCHEMA, status="unknown", valid=False, reason=reason,
                evidence_kind=evidence_kind, policy=policy, **facts)


def _interpolate(stations, values, station):
    index = bisect.bisect_right(stations, station)-1
    if index < 0 or index >= len(stations)-1:
        return values[max(0, min(index, len(values)-1))]
    fraction = (station-stations[index])/(stations[index+1]-stations[index])
    return values[index]+fraction*(values[index+1]-values[index])


def _runs(stations, values, minimum_run):
    """Commit stable states only; uncertain hysteresis samples cannot switch."""
    transitions = []
    pending = pending_station = None
    for station, value in zip(stations, values):
        if value is None:
            pending = pending_station = None
            continue
        if value != pending:
            pending, pending_station = value, station
        if station-pending_station >= minimum_run and (not transitions or transitions[-1]["state"] != value):
            transitions.append({"state": value, "station_m": station})
    return transitions


def _shape(stations, lateral, geometry, own_length, policy):
    lanes = []
    directions = []
    own, opposite = geometry["own_points"], geometry.get("opposing_points", [])
    for station, value in zip(stations, lateral):
        if station < own_length and opposite:
            center, _ = route_pose(own, min(station, own_length))
            opposing_station = project_polyline(opposite, *center[:2])[0]
            opposing_center, _ = route_pose(opposite, opposing_station)
            separation = project_polyline(own, *opposing_center[:2])[1]
            coordinate = value if separation >= 0 else -value
            divider = abs(separation)*0.5
            lanes.append("opposing" if coordinate > divider+policy["lane_hysteresis_m"] else
                         "own" if coordinate < divider-policy["lane_hysteresis_m"] else None)
        else:
            lanes.append("own" if abs(value) < float(geometry.get("own_lane_width_m", 3.5))*0.5-policy["lane_hysteresis_m"] else None)
        directions.append("left" if value > policy["center_excursion_m"] else
                          "right" if value < -policy["center_excursion_m"] else
                          "center" if abs(value) < policy["center_return_m"] else None)
    transitions = _runs(stations, lanes, policy["minimum_run_m"])
    center_transitions = _runs(stations, directions, policy["minimum_run_m"])
    sequence = [row["state"] for row in transitions]
    center_sequence = [row["state"] for row in center_transitions]
    excursions = sum(sequence[i:i+3] == ["own", "opposing", "own"] for i in range(len(sequence)-2))
    signed_excursions = [center_sequence[i+1] for i in range(len(center_sequence)-2)
                         if center_sequence[i] == center_sequence[i+2] == "center"
                         and center_sequence[i+1] in ("left", "right")]
    returns = [row["station_m"] for i, row in enumerate(center_transitions)
               if i and row["state"] == "center" and center_transitions[i-1]["state"] != "center"]
    # Turn order uses net >=30 cm movements. Tiny derivative oscillations are ignored.
    turns = []
    start = 0
    current = None
    for i in range(1, len(stations)):
        slope = (lateral[i]-lateral[i-1])/(stations[i]-stations[i-1])
        direction = "left" if slope > policy["turn_slope_min"] else "right" if slope < -policy["turn_slope_min"] else None
        if direction != current:
            if current and abs(lateral[i-1]-lateral[start]) >= policy["turn_amplitude_min_m"]:
                if not turns or turns[-1] != current:
                    turns.append(current)
            start, current = i-1, direction
    if current and abs(lateral[-1]-lateral[start]) >= policy["turn_amplitude_min_m"]:
        if not turns or turns[-1] != current:
            turns.append(current)
    pattern = ("alternating_lane_detours" if excursions >= 2 else
               "single_opposing_lane_excursion" if excursions == 1 else
               "opposing_lane_without_verified_return" if "opposing" in sequence else
               "within_original_lane")
    return {"lane_sequence": sequence, "lane_transitions": transitions,
            "lane_transition_count": max(0, len(sequence)-1),
            "completed_opposing_excursions": excursions,
            "center_sequence": center_sequence, "center_transitions": center_transitions,
            "excursion_direction_sequence": signed_excursions, "turn_sequence": turns,
            "maximum_width_m": max(lateral)-min(lateral),
            "maximum_abs_lateral_m": max(abs(value) for value in lateral),
            "minimum_lateral_m": min(lateral), "maximum_lateral_m": max(lateral),
            "return_station_m": returns[-1] if returns else None,
            "return_verified": bool(returns) and bool(center_sequence) and center_sequence[-1] == "center",
            "realized_pattern": pattern,
            "s_maneuver_observed": excursions >= 2}


def path_features(points, geometry, *, evidence_kind="nominal_screen", policy=None):
    """Return one station/lateral representation for screen or actual XY points.

    ``geometry`` is ``layout.tunnel_geometry(loader)`` optionally enriched with
    the connected ``departure_points`` from MGeo A2256W000128.  No reference
    extension is invented when this geometry is missing.  Unknown records are
    intentionally JSON-safe and may be exported alongside valid features.
    """
    policy = _policy(policy)
    try:
        reference, own_length, expected, geometry_hash = _reference(geometry, policy)
        rows = list(points)
        projected = []
        for row in rows:
            if isinstance(row, dict):
                point = row.get("position_xyz", (row.get("x"), row.get("y")))
            else:
                point = row
            if len(point) < 2 or any(isinstance(v, bool) or not math.isfinite(float(v)) for v in point[:2]):
                return _unknown("nonfinite_or_invalid_point", policy, evidence_kind)
            station, lateral, _ = project_polyline(reference, float(point[0]), float(point[1]))
            # Approach/goal points project to clamped endpoints and do not
            # establish genuine tunnel/departure coverage.
            if 0.05 < station < expected-0.05:
                projected.append((station, lateral))
        if len(projected) < policy["minimum_samples"]:
            return _unknown("insufficient_samples", policy, evidence_kind, source_point_count=len(rows))
        high_water, backward = projected[0][0], 0.
        for station, _ in projected:
            backward = max(backward, high_water-station)
            high_water = max(high_water, station)
        if backward > policy["maximum_backward_step_m"]:
            return _unknown("nonmonotonic_station_coverage", policy, evidence_kind, maximum_backward_step_m=backward)
        monotonic = []
        for station, value in projected:
            if not monotonic or station > monotonic[-1][0]+1e-6:
                monotonic.append((station, value))
        if len(monotonic) < policy["minimum_samples"]:
            return _unknown("insufficient_monotonic_samples", policy, evidence_kind)
        source_stations, source_lateral = map(list, zip(*monotonic))
        largest_gap = max(b-a for a, b in zip(source_stations, source_stations[1:]))
        coverage = (source_stations[-1]-source_stations[0])/expected
        if coverage < policy["minimum_coverage_fraction"]:
            return _unknown("insufficient_station_coverage", policy, evidence_kind,
                            coverage_fraction=coverage, expected_length_m=expected)
        if largest_gap > policy["maximum_source_gap_m"]:
            return _unknown("missing_station_coverage", policy, evidence_kind, maximum_source_gap_m=largest_gap)
        step = policy["sample_step_m"]
        stations = [round(i*step, 6) for i in range(math.ceil(source_stations[0]/step),
                                                  math.floor(source_stations[-1]/step)+1)]
        lateral = [_interpolate(source_stations, source_lateral, station) for station in stations]
        radius = max(1, int(policy["smoothing_window_m"]/(2*step)))
        lateral = [statistics.median(lateral[max(0, i-radius):i+radius+1]) for i in range(len(lateral))]
        lateral = [round(value, 6) for value in lateral]
        shape = _shape(stations, lateral, geometry, own_length, policy)
        # Only measured/completed actual traces can assert an observed actual S.
        shape["predicted_s_path_shape"] = shape["s_maneuver_observed"]
        shape["s_maneuver_observed"] = evidence_kind == "completed_actual_trace" and shape["predicted_s_path_shape"]
        shape["actual_s_maneuver_observed"] = shape["s_maneuver_observed"]
        canonical = {"reference_geometry_sha256": geometry_hash,
                     "policy_sha256": _hash(policy), "stations_m": stations,
                     "lateral_quantized": [int(round(v/policy["fingerprint_lateral_quantum_m"])) for v in lateral],
                     "lane_sequence": shape["lane_sequence"], "turn_sequence": shape["turn_sequence"]}
        return dict(schema=SCHEMA, status="valid", valid=True, evidence_kind=evidence_kind,
                    geometry_basis="mapped MGeo own lane and connected departure" if expected > own_length else "mapped MGeo own tunnel lane",
                    reference_geometry_sha256=geometry_hash, tunnel_geometry_sha256=geometry.get("source_sha256"),
                    tunnel_length_m=own_length, expected_length_m=expected,
                    departure_return_coverage=expected > own_length,
                    station_range_m=[stations[0], stations[-1]], coverage_fraction=coverage,
                    source_point_count=len(rows), maximum_source_gap_m=largest_gap,
                    stations_m=stations, lateral_m=lateral,
                    normalized_station=[round(s/own_length, 8) for s in stations],
                    normalized_lateral=[round(v/float(geometry.get("own_lane_width_m", 3.5)), 8) for v in lateral],
                    fingerprint_sha256=_hash(canonical), policy=policy, **shape)
    except (TypeError, ValueError, KeyError, IndexError, OverflowError) as error:
        return _unknown("invalid_path_or_geometry", policy, evidence_kind, detail=str(error))


def compare_paths(first, second, *, policy=None):
    """Compare valid features over their common genuine station coverage."""
    policy = _policy(policy)
    unknown = dict(schema=SCHEMA, status="unknown", duplicate=None, distance=None)
    if first.get("valid") is not True or second.get("valid") is not True:
        return dict(unknown, reasons=["invalid_or_unknown_path_features"])
    if first.get("reference_geometry_sha256") != second.get("reference_geometry_sha256"):
        return dict(unknown, reasons=["reference_geometry_mismatch"])
    try:
        for feature in (first, second):
            stations, lateral = feature["stations_m"], feature["lateral_m"]
            if (len(stations) != len(lateral) or len(stations) < policy["minimum_samples"]
                    or any(isinstance(value, bool) or not math.isfinite(float(value)) for value in stations+lateral)
                    or any(b <= a or b-a > policy["maximum_source_gap_m"] for a,b in zip(stations,stations[1:]))
                    or not math.isfinite(feature["expected_length_m"]) or feature["expected_length_m"] <= 0
                    or not math.isfinite(feature["maximum_width_m"])):
                return dict(unknown, reasons=["malformed_path_features"])
            if feature["return_station_m"] is not None and not math.isfinite(feature["return_station_m"]):
                return dict(unknown, reasons=["malformed_path_features"])
        first_stations, second_stations = first["stations_m"], second["stations_m"]
        low, high = max(first_stations[0], second_stations[0]), min(first_stations[-1], second_stations[-1])
        expected = max(first["expected_length_m"], second["expected_length_m"])
        fraction = (high-low)/expected
        if fraction < policy["minimum_coverage_fraction"]:
            return dict(unknown, reasons=["insufficient_common_station_coverage"], common_coverage_fraction=fraction)
        step = policy["sample_step_m"]
        stations = [i*step for i in range(math.ceil(low/step), math.floor(high/step)+1)]
        discrepancies = [abs(_interpolate(first_stations, first["lateral_m"], s)-
                             _interpolate(second_stations, second["lateral_m"], s)) for s in stations]
        mean, maximum = statistics.mean(discrepancies), max(discrepancies)
        width = abs(first["maximum_width_m"]-second["maximum_width_m"])
        a_return, b_return = first["return_station_m"], second["return_station_m"]
        return_difference = abs(a_return-b_return) if a_return is not None and b_return is not None else None
        transition_difference = None
        a_transitions, b_transitions = first["lane_transitions"], second["lane_transitions"]
        if len(a_transitions) == len(b_transitions):
            transition_difference = max((abs(a["station_m"]-b["station_m"]) for a,b in zip(a_transitions,b_transitions)), default=0.)
        reasons = []
        for name in ("lane_sequence", "completed_opposing_excursions", "excursion_direction_sequence", "turn_sequence"):
            if first.get(name) != second.get(name):
                reasons.append(name + "_differs")
        if mean > policy["mean_duplicate_m"]:
            reasons.append("mean_lateral_discrepancy")
        if maximum > policy["max_duplicate_m"]:
            reasons.append("maximum_lateral_discrepancy")
        if width > policy["width_duplicate_m"]:
            reasons.append("maximum_width_differs")
        if (a_return is None) != (b_return is None) or (return_difference is not None and return_difference > policy["return_duplicate_m"]):
            reasons.append("return_station_differs")
        if transition_difference is not None and transition_difference > policy["transition_duplicate_m"]:
            reasons.append("transition_stations_differ")
        categorical = sum(reason.endswith("_differs") for reason in reasons)
        distance = (mean/policy["mean_duplicate_m"] + maximum/policy["max_duplicate_m"] +
                    width/policy["width_duplicate_m"] + categorical*2.0 +
                    (return_difference or 0)/policy["return_duplicate_m"])
        return dict(schema=SCHEMA, status="distinct" if reasons else "duplicate", duplicate=not reasons,
                    distance=round(distance, 6), reasons=reasons,
                    mean_lateral_discrepancy_m=round(mean, 6), max_lateral_discrepancy_m=round(maximum, 6),
                    width_difference_m=round(width, 6), return_station_difference_m=return_difference,
                    transition_station_difference_m=transition_difference,
                    common_station_range_m=[low, high], common_sample_count=len(stations),
                    common_coverage_fraction=round(fraction, 6))
    except (TypeError, ValueError, KeyError, IndexError, OverflowError, statistics.StatisticsError):
        return dict(unknown, reasons=["malformed_path_features"])


def layout_fingerprint(candidate):
    """Exact world-layout key, independent of RNG seed and requested role."""
    obstacles = []
    for row in candidate.get("obstacles", []):
        obstacles.append({key: row.get(key) for key in ("actor_kind", "model", "position", "yaw_deg", "scale", "planned_bbox")})
    return _hash(sorted(obstacles, key=lambda row: json.dumps(row, sort_keys=True)))


def _features_from_existing(rows):
    result = []
    for row in rows:
        feature = row if "stations_m" in row or row.get("status") == "unknown" else row.get("actual_features", row.get("features", {}))
        if feature.get("valid") is True:
            result.append(feature)
    return result


def _coverage(candidate):
    result = {"count:"+str(candidate.get("count", len(candidate.get("obstacles", []))))} | {
        "model:"+str(row.get("actor_kind"))+":"+str(row.get("model")) for row in candidate.get("obstacles", [])} | {
        "role:"+str(candidate.get("requested_pattern", candidate.get("requested_role", "unspecified")))}
    stations = []
    for row in candidate.get("obstacles", []):
        station, lateral = row.get("station_m"), row.get("lateral_m")
        if isinstance(station, (int,float)) and math.isfinite(station):
            stations.append(station)
            result.add("position:station_10m:"+str(math.floor(station/10.)))
        if isinstance(lateral, (int,float)) and math.isfinite(lateral):
            result.add("position:lateral_0.5m:"+str(math.floor(lateral/.5)))
    stations.sort()
    for first, last in zip(stations, stations[1:]):
        result.add("spacing:gap_10m:"+str(math.floor((last-first)/10.)))
    return result


def select_diverse_candidates(candidates, geometry, *, existing_paths=(), limit=100,
                              policy=None, count_quotas=None):
    """Deterministically choose farthest paths plus count/model/role coverage.

    Rows have ``id``, ``candidate`` and a production ``screen``.  Exact layouts
    are rejected even when metadata differs.  Similar paths may be retained to
    meet semantic coverage or balanced count quotas; rationale explicitly says
    they add no path novelty.  No selected unrun candidate is actual-approved.
    ``limit`` is capped at the user's 100-drive target.  Quotas may be supplied
    for an adaptive remaining batch (e.g. {1: 7, 2: 12, 3: 8}).
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 100:
        raise ValueError("limit must be an integer from 0 through 100")
    policy = _policy(policy)
    baseline = _features_from_existing(existing_paths)
    coverage = set()
    layouts = set()
    for row in existing_paths:
        if "candidate" in row:
            coverage.update(_coverage(row["candidate"]))
            layouts.add(layout_fingerprint(row["candidate"]))
    quotas = dict(count_quotas) if count_quotas is not None else {
        1: int(round(limit*.30)), 2: int(round(limit*.40)), 3: 0}
    if count_quotas is None:
        quotas[3] = limit-quotas[1]-quotas[2]
    if any(isinstance(k, bool) or k not in (1,2,3) or isinstance(v, bool) or not isinstance(v, int) or v < 0 for k,v in quotas.items()):
        raise ValueError("count_quotas require nonnegative integer targets for counts 1–3")
    feature_records, rejected, prepared = [], [], []
    for index, row in enumerate(candidates):
        identity = str(row.get("id", index))
        candidate, screen = row.get("candidate", {}), row.get("screen", row.get("route_screen", {}))
        if screen.get("status") != "nominal_path_found" or screen.get("path_found") is not True:
            feature = _unknown("screen_not_successful", policy, "nominal_screen")
        else:
            feature = path_features(screen.get("path", []), geometry, policy=policy)
        feature_records.append({"id": identity, "features": feature})
        if not feature["valid"]:
            rejected.append({"id": identity, "reason": feature["reason"]})
            continue
        try:
            layout = layout_fingerprint(candidate)
        except (ValueError, TypeError):
            rejected.append({"id": identity, "reason": "invalid_layout"})
            continue
        count = candidate.get("count", len(candidate.get("obstacles", [])))
        if count not in (1,2,3):
            rejected.append({"id": identity, "reason": "unsupported_obstacle_count"})
            continue
        if layout in layouts:
            rejected.append({"id": identity, "reason": "exact_layout_duplicate"})
            continue
        prepared.append({"id": identity, "candidate": candidate, "features": feature,
                         "layout_fingerprint_sha256": layout, "coverage": _coverage(candidate), "count": count})
    # Canonical order prevents input iteration order from choosing representatives.
    prepared.sort(key=lambda row: (row["layout_fingerprint_sha256"], row["id"]))
    selected, selected_features, counts = [], [], {1:0, 2:0, 3:0}
    while prepared and len(selected) < limit:
        ranking = []
        exhausted = []
        unmet = {count for count in (1,2,3) if counts[count] < quotas.get(count, 0)}
        available_unmet = unmet & {row["count"] for row in prepared}
        for row in prepared:
            if available_unmet and row["count"] not in available_unmet:
                continue
            comparisons = [compare_paths(row["features"], other, policy=policy) for other in baseline+selected_features]
            valid_comparisons = [item for item in comparisons if item["distance"] is not None]
            nearest = min(valid_comparisons, key=lambda item:item["distance"]) if valid_comparisons else None
            unknown_comparison = len(valid_comparisons) != len(comparisons)
            duplicate = any(item["duplicate"] is True for item in valid_comparisons)
            new_coverage = sorted(row["coverage"]-coverage)
            model_gain = sum(item.startswith("model:") for item in new_coverage)
            role_gain = sum(item.startswith("role:") for item in new_coverage)
            deficit = max(0, quotas.get(row["count"], 0)-counts[row["count"]])
            # Unknown comparisons cannot be credited as novel.
            novelty = (not duplicate and not unknown_comparison)
            if not novelty and not new_coverage and deficit == 0:
                exhausted.append(row)
                continue
            distance = nearest["distance"] if nearest else (10. if not comparisons else 0.)
            score = model_gain*8. + role_gain*2. + (4. if novelty else 0.) + min(distance, 10.) + deficit/max(1, quotas.get(row["count"],0))
            ranking.append((score, row["layout_fingerprint_sha256"], row["id"], row,
                            nearest, duplicate, novelty, new_coverage, unknown_comparison))
        for row in exhausted:
            rejected.append({"id":row["id"], "reason":"similar_path_without_coverage_deficit"})
            prepared.remove(row)
        if not ranking:
            break
        winner = min(ranking, key=lambda item:(-item[0], item[1], item[2]))
        score, _, _, row, nearest, duplicate, novelty, new_coverage, unknown_comparison = winner
        selected.append({"id": row["id"], "selection_index": len(selected),
                         "layout_fingerprint_sha256": row["layout_fingerprint_sha256"],
                         "path_fingerprint_sha256": row["features"]["fingerprint_sha256"],
                         "actual_approved": False,
                         "rationale": {"score": round(score,6), "new_coverage": new_coverage,
                             "path_novelty_predicted": novelty, "path_duplicate_to_comparison_set": duplicate,
                             "path_comparison_unknown": unknown_comparison,
                             "nearest_path_comparison": nearest,
                             "selection_basis": "nominal path diversity" if novelty else "count/model/role representative; no new path novelty established",
                             "count_quota": quotas.get(row["count"], 0),
                             "count_quota_deficit_before_selection":max(0, quotas.get(row["count"],0)-counts[row["count"]])}})
        selected_features.append(row["features"])
        coverage.update(row["coverage"])
        counts[row["count"]] += 1
        layouts.add(row["layout_fingerprint_sha256"])
        remaining = []
        for item in prepared:
            if item is row:
                continue
            if item["layout_fingerprint_sha256"] in layouts:
                rejected.append({"id":item["id"], "reason":"exact_layout_duplicate"})
            else:
                remaining.append(item)
        prepared = remaining
    return {"schema": SCHEMA, "selected": selected, "selected_ids": [row["id"] for row in selected],
            "feature_records": feature_records, "rejected": rejected, "policy": policy,
            "count_quotas": quotas, "selected_counts": counts,
            "quota_deficits": {count:max(0, target-counts.get(count,0)) for count,target in quotas.items()},
            "selected_count":len(selected), "unselected_valid_count":len(prepared),
            "selection_scope":"unrun nominal-screen candidates for guarded actual trials; not live approval"}


def cluster_paths(paths, *, policy=None):
    """Greedy deterministic clusters; unknown inputs are reported separately.

    Cluster identity is a representative comparison, not a transitive claim:
    pairwise threshold similarity is not mathematically transitive.
    """
    policy = _policy(policy)
    paths = list(paths)
    features = _features_from_existing(paths)
    features.sort(key=lambda feature:feature["fingerprint_sha256"])
    clusters = []
    for feature in features:
        matching = [(i, compare_paths(feature, group["representative"], policy=policy)) for i,group in enumerate(clusters)]
        matching = [(i, result) for i,result in matching if result["duplicate"] is True]
        if matching:
            i, _ = min(matching, key=lambda item:(item[1]["distance"], item[0]))
            clusters[i]["members"].append(feature["fingerprint_sha256"])
        else:
            clusters.append({"representative":feature, "members":[feature["fingerprint_sha256"]]})
    for group in clusters:
        group["representative_fingerprint_sha256"] = group["representative"]["fingerprint_sha256"]
        group["member_count"] = len(group["members"])
    return {"schema":SCHEMA, "cluster_count":len(clusters), "valid_path_count":len(features),
            "unknown_path_count":len(paths)-len(features),
            "clusters":clusters, "basis":"representative mean/max lateral and verified trajectory-shape comparisons"}


def _read_hashed(path, expected_hash=None):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None and digest != expected_hash:
        raise ValueError("Evidence SHA256 mismatch: " + str(path))
    return json.loads(raw), digest


def _planning_compatibility(screen, current):
    # Preparation-only code changes do not erase immutable historical drives.
    # Every Expert source, route, policy and recording must still agree.
    fields = ("recorded_library_sha256", "original_route_sha256", "screening_policy_sha256", "screening_policy")
    if any(screen.get(key) != current.get(key) for key in fields):
        return False
    old_sources, current_sources = screen.get("planner_source_sha256", {}), current.get("planner_source_sha256", {})
    keys = {key for key in current_sources if key.startswith("expert/") or key.endswith(("segment_modes/modes.py", "segment_modes/route.py"))}
    return bool(keys) and all(old_sources.get(key) == current_sources[key] for key in keys)


def load_approved_baseline(approved_path, geometry, *, current_provenance=None,
                           current_runtime_sources=None, policy=None, batch_manifest_path=None):
    """Load strictly successful historical traces for deduplication, never replay approval.

    Candidate, preview and drive hashes are verified against the immutable
    approvals.  Every loaded file gets a SHA256 in the result.  Older approval
    documents did not seal CSV/manifest hashes; those hashes are honestly marked
    computed at load, and the CSV shape must match its sealed drive trajectory.
    Source changes are exposed separately from compatible planning inputs.
    """
    policy = _policy(policy)
    approved_path = Path(approved_path)
    document, approved_hash = _read_hashed(approved_path)
    if current_provenance is None:
        from .screening import screening_provenance
        current_provenance = screening_provenance()
    manifest_path = Path(batch_manifest_path) if batch_manifest_path else approved_path.with_name("batch_manifest.json")
    manifest, manifest_hash = _read_hashed(manifest_path)
    attempts = {str(Path(row.get("trial_dir", "")).resolve()): row for row in manifest.get("attempts", [])}
    records, rejected = [], []
    for index, approval in enumerate(document.get("candidates", [])):
        identity = str(approval.get("candidate_sha256", index))
        try:
            candidate, candidate_hash = _read_hashed(approval["candidate_path"], approval["candidate_sha256"])
            preview, preview_hash = _read_hashed(approval["preview_report_path"], approval["preview_report_sha256"])
            drive, drive_hash = _read_hashed(approval["drive_report_path"], approval["drive_report_sha256"])
            directory = Path(approval["drive_report_path"]).parent
            trial_manifest, trial_manifest_hash = _read_hashed(directory/"manifest.json", approval.get("trial_manifest_sha256"))
            planner, planner_hash = _read_hashed(directory/"planner_diagnostics.json", approval.get("planner_diagnostics_sha256"))
            conditions = drive.get("completion_conditions", {})
            strict = (drive.get("status") == "passed" and all(drive.get(key) is True for key in STRICT_FLAGS)
                      and all(conditions.get(key) is True for key in STRICT_CONDITIONS)
                      and all(value is True for value in conditions.values())
                      and not drive.get("collision_objects") and not drive.get("failed_completion_conditions")
                      and trial_manifest.get("status") == "passed"
                      and trial_manifest.get("expert_verified") is True
                      and trial_manifest.get("live_avoidance_verified") is True
                      and planner.get("validated_path_seen") is True
                      and preview.get("status") == "valid_driving_trial"
                      and preview.get("cleanup", {}).get("verified") is True
                      and approval.get("fixture_cleanup_verified") is True
                      and approval.get("recovery_verified") is True)
            attempt = attempts.get(str(directory.resolve()), {})
            if not strict or attempt.get("outcome", {}).get("strict_pass") is not True or attempt.get("committed") is not True:
                raise ValueError("strict_successful_trial_proof_missing")
            screen = preview.get("measured_route_screen", {})
            if screen.get("status") != "nominal_path_found" or screen.get("path_found") is not True:
                raise ValueError("successful_measured_screen_missing")
            if candidate.get("source_geometry", {}).get("sha256") != geometry.get("source_sha256"):
                raise ValueError("mapped_tunnel_geometry_changed")
            if not _planning_compatibility(screen, current_provenance):
                raise ValueError("current_planning_inputs_incompatible")
            if (screen.get("road_boundary_policy_sha256") != drive.get("road_boundary_policy_sha256")
                    or screen.get("road_boundary_policy") != drive.get("road_boundary_policy")
                    or screen.get("road_boundary_policy_sha256") != planner.get("road_boundary_policy_sha256")):
                raise ValueError("trial_road_or_body_policy_mismatch")
            trace_path = directory/"ego_trace.csv"
            trace_raw = trace_path.read_bytes()
            trace_hash = hashlib.sha256(trace_raw).hexdigest()
            expected_trace = approval.get("ego_trace_sha256", attempt.get("ego_trace_sha256"))
            if expected_trace is not None and trace_hash != expected_trace:
                raise ValueError("ego_trace_sha256_mismatch")
            with trace_path.open(encoding="utf-8", newline="") as stream:
                trace = [[float(row["x"]), float(row["y"])] for row in csv.DictReader(stream)]
            actual = path_features(trace, geometry, evidence_kind="completed_actual_trace", policy=policy)
            report_actual = path_features([row["position_xyz"] for row in drive.get("trajectory", [])],
                                         geometry, evidence_kind="completed_actual_trace", policy=policy)
            alignment = compare_paths(actual, report_actual, policy=policy)
            if alignment["duplicate"] is not True:
                raise ValueError("csv_trace_does_not_match_sealed_drive_trajectory")
            nominal = path_features(screen["path"], geometry, evidence_kind="historical_measured_screen", policy=policy)
            saved_runtime = drive.get("runtime_source_sha256", {})
            if current_runtime_sources is None:
                runtime_current = bool(saved_runtime) and all(Path(path).is_file() and hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                                                             for path,digest in saved_runtime.items())
            else:
                runtime_current = bool(saved_runtime) and all(current_runtime_sources.get(path) == digest for path,digest in saved_runtime.items())
            all_sources_current = all(screen.get(key) == value for key,value in current_provenance.items())
            records.append({"id":identity, "candidate":candidate, "actual_features":actual,
                "nominal_features":nominal, "evidence_status":"historical_strict_success_planning_compatible",
                "dedup_usable":True, "actual_approved":False,
                "runtime_sources_current":runtime_current, "all_screen_sources_current":all_sources_current,
                "current_planning_inputs_compatible":True,
                "geometry_basis":actual["geometry_basis"],
                "file_sha256":{"candidate":candidate_hash, "preview":preview_hash, "drive":drive_hash,
                    "ego_trace":trace_hash, "trial_manifest":trial_manifest_hash, "planner_diagnostics":planner_hash},
                "csv_sha256_seal_status":"verified_against_original_proof" if expected_trace else "computed_at_load; trajectory_matches_original_hashed_drive",
                "source_paths":{"candidate":str(approval["candidate_path"]), "drive":str(approval["drive_report_path"]), "ego_trace":str(trace_path)},
                "trace_to_sealed_trajectory_comparison":alignment,
                "requested_pattern":approval.get("requested_pattern"), "realized_pattern":actual["realized_pattern"]})
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            rejected.append({"id":identity, "reason":str(error)})
    return {"schema":SCHEMA, "records":records, "rejected":rejected,
            "approved_source_sha256":approved_hash, "batch_manifest_sha256":manifest_hash,
            "loaded_count":len(records), "rejected_count":len(rejected),
            "scope":"historical completed actual trajectories for deduplication only; fresh measured guarded replay required",
            "clusters":cluster_paths(records, policy=policy)}
