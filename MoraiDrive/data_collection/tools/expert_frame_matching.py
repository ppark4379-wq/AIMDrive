"""Independent receipt-time and XY candidates, not exact collision exposure labels."""

import json
import math


def valid_ns(value):
    return type(value) is int and value >= 0


def valid_xy(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in value))


def load_frame_sources(episode):
    sources = {"receipt": {}, "position": {}, "warnings": []}
    for kind, relative in (
        ("receipt", "native_associations/associations.json"),
        ("position", "gt_timeline_v1/frames.json"),
    ):
        path = episode / relative
        try:
            with path.open(encoding="utf-8-sig") as stream:
                rows = json.load(stream)
            if not isinstance(rows, list) or not rows:
                raise ValueError("expected a nonempty JSON array")
            indexed = {}
            for row in rows:
                if not isinstance(row, dict) or not valid_ns(row.get("index")):
                    raise ValueError("invalid bundle index")
                index = row["index"]
                if index in indexed:
                    raise ValueError("duplicate bundle index {}".format(index))
                if kind == "receipt":
                    gps = row.get("gps")
                    if not isinstance(gps, dict) or not valid_ns(gps.get("receive_monotonic_ns")):
                        raise ValueError("invalid GPS receipt time at index {}".format(index))
                    if not valid_ns(gps.get("source_timestamp_ns")):
                        raise ValueError("invalid GPS source time at index {}".format(index))
                    indexed[index] = {"receive_ns": gps["receive_monotonic_ns"],
                                      "source_ns": gps["source_timestamp_ns"]}
                else:
                    if not valid_ns(row.get("lidar_source_ns")):
                        raise ValueError("invalid GT source time at index {}".format(index))
                    xy = row.get("ego_xy")
                    # Unbracketed GT legitimately has no ego position.
                    if xy is not None and not valid_xy(xy):
                        raise ValueError("invalid ego_xy at index {}".format(index))
                    indexed[index] = {"xy": xy, "source_ns": row["lidar_source_ns"]}
            sources[kind] = indexed
        except (OSError, ValueError) as exc:
            sources["warnings"].append("{}: {}".format(relative, exc))
    return sources


def nearest_candidates(scores):
    if not scores:
        return [], None
    minimum = min(scores.values())
    # Exact ties remain ambiguous; no arbitrary millisecond/metre match cutoff.
    return sorted(index for index, score in scores.items() if score == minimum), minimum


def interpolate_position(index, xy, positions):
    """Project onto contiguous adjacent bundles; never clamp or extrapolate."""
    candidates = []
    for left in (index - 1, index):
        a, b = positions.get(left), positions.get(left + 1)
        if not a or not b or not valid_xy(a["xy"]) or not valid_xy(b["xy"]):
            continue
        duration = b["source_ns"] - a["source_ns"]
        if duration <= 0:
            continue
        dx, dy = b["xy"][0] - a["xy"][0], b["xy"][1] - a["xy"][1]
        length_squared = dx * dx + dy * dy
        if length_squared == 0:
            continue
        fraction = ((xy[0] - a["xy"][0]) * dx + (xy[1] - a["xy"][1]) * dy) / length_squared
        if not 0 <= fraction <= 1:
            continue
        residual = math.hypot(xy[0] - a["xy"][0] - fraction * dx,
                              xy[1] - a["xy"][1] - fraction * dy)
        candidates.append((residual, left, fraction, a["source_ns"], duration))
    candidates.sort()
    if not candidates or (len(candidates) > 1 and candidates[0][0] == candidates[1][0]):
        return {}
    residual, left, fraction, source_ns, duration = candidates[0]
    estimate = source_ns + round(fraction * duration)
    return {
        "collision_between_frames": [left, left + 1],
        "estimated_fraction_after_frame": fraction,
        "estimated_collision_source_time_ns": estimate,
        "estimated_collision_source_time_seconds": estimate / 1e9,
        "interpolation_position_error_m": residual,
        "interpolation_method": "ego_xy_projection_on_adjacent_bundles",
    }


def match_collision(collision, sources):
    receipts, positions = sources["receipt"], sources["position"]
    result = {
        "sensor_bundle_index": None,
        "sensor_bundle_number": None,
        "sensor_bundle_index_base": 0,
        "match_status": "UNAVAILABLE",
        "candidate_index_distance": None,
        "candidate_source_time_gap_ms": None,
        "possible_collision_between_bundle_indices": None,
        "match_interpretation": "Insufficient evidence to select a sensor bundle.",
        "nearest_sensor_bundle_index": None,
        "nearest_sensor_frame_number": None,
        "receive_time_candidate_index": None,
        "position_candidate_index": None,
        "receive_time_candidate_indices": [],
        "position_candidate_indices": [],
        "receive_time_gap_ms": None,
        "position_error_m": None,
        "nearest_position_error_m": None,
        "match_confidence": "UNAVAILABLE",
        "match_reasons": list(sources["warnings"]),
        "neighbor_position_errors": [],
        "collision_between_frames": None,
        "estimated_fraction_after_frame": None,
        "estimated_collision_source_time_ns": None,
        "estimated_collision_source_time_seconds": None,
        "interpolation_position_error_m": None,
        "interpolation_method": None,
        "matching_method": "independent_gps_receive_monotonic_and_gt_ego_xy_nearest",
        "physical_collision_time_verified": False,
    }
    receive_ns, xy = collision.get("receive_monotonic_ns"), collision.get("ego_xy")
    time_scores = ({i: abs(row["receive_ns"] - receive_ns) for i, row in receipts.items()}
                   if valid_ns(receive_ns) else {})
    position_scores = ({i: math.hypot(row["xy"][0] - xy[0], row["xy"][1] - xy[1])
                        for i, row in positions.items() if valid_xy(row["xy"])}
                       if valid_xy(xy) else {})
    time_ids, gap = nearest_candidates(time_scores)
    position_ids, error = nearest_candidates(position_scores)
    result["receive_time_candidate_indices"] = time_ids
    result["position_candidate_indices"] = position_ids
    result["receive_time_gap_ms"] = gap / 1e6 if gap is not None else None
    result["nearest_position_error_m"] = error
    time_index = time_ids[0] if len(time_ids) == 1 else None
    position_index = position_ids[0] if len(position_ids) == 1 else None
    result["receive_time_candidate_index"] = time_index
    result["position_candidate_index"] = position_index
    # A time candidate remains visible even if the spatial check disagrees.
    result["nearest_sensor_bundle_index"] = time_index
    result["nearest_sensor_frame_number"] = time_index + 1 if time_index is not None else None
    result["position_error_m"] = position_scores.get(time_index)
    reasons = result["match_reasons"]
    if not time_ids or not position_ids:
        reasons.append("receipt time or ego position candidates unavailable")
        return result
    if len(time_ids) != 1 or len(position_ids) != 1:
        result["match_status"] = "AMBIGUOUS"
        result["match_interpretation"] = "Multiple equally near candidates; no bundle selected."
        result["match_confidence"] = "AMBIGUOUS"
        reasons.append("nearest candidate is tied")
        return result
    result["match_confidence"] = "LOW"
    result["neighbor_position_errors"] = [
        {"index": i, "position_error_m": position_scores[i]}
        for i in (time_index - 1, time_index, time_index + 1) if i in position_scores
    ]
    distance = abs(time_index - position_index)
    result["candidate_index_distance"] = distance
    result["match_status"] = "INCONSISTENT_EVIDENCE"
    result["match_interpretation"] = "Timing evidence is inconsistent; no bundle selected."
    if not min(row["receive_ns"] for row in receipts.values()) <= receive_ns <= max(
            row["receive_ns"] for row in receipts.values()):
        reasons.append("collision receipt is outside the sensor receipt range")
        return result
    for index in {time_index, position_index}:
        if (index not in receipts or index not in positions
                or receipts[index]["source_ns"] != positions[index]["source_ns"]):
            reasons.append("association and GT source timestamps missing or inconsistent")
            return result
    left, right = sorted((time_index, position_index))
    source_gap = positions[right]["source_ns"] - positions[left]["source_ns"]
    result["candidate_source_time_gap_ms"] = source_gap / 1e6
    if distance and source_gap <= 0:
        reasons.append("candidate source times do not increase with bundle index")
        return result
    if distance:
        reasons.append("receipt-time and position candidates disagree")
        if distance == 1:
            result["match_status"] = "BETWEEN_FRAMES"
            result["possible_collision_between_bundle_indices"] = [left, right]
            result["match_interpretation"] = (
                "Collision may lie between sensor_bundle_index {} and {} (0-based); "
                "adjacent candidates only, no exact bundle or collision time confirmed."
            ).format(left, right)
        else:
            result["match_status"] = "CANDIDATE_MISMATCH"
            result["match_interpretation"] = (
                "Candidates differ by {} bundle indices; no bundle selected."
            ).format(distance)
        return result
    result["match_confidence"] = "HIGH"
    result["match_status"] = "CANDIDATES_AGREE"
    result["sensor_bundle_index"] = time_index
    result["sensor_bundle_number"] = time_index + 1
    result["match_interpretation"] = (
        "Independent nearest candidates agree; selected proximity bundle, "
        "not a verified physical collision frame."
    )
    reasons.append("unique independent nearest candidates agree; proximity estimate only")
    result.update(interpolate_position(time_index, xy, positions))
    return result
