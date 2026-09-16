"""Conservative spatial relevance checks for the native 50x50 BEV target.

The object API reports a world-space bounding-box centre and W/L/H values.
This module only uses those observations to prove that an actor is entirely
outside the supervised BEV.  It deliberately uses a circumscribed circle, so
an uncertain boundary case stays relevant and is fail-closed by the measured
geometry audit.
"""
import math


DEFAULT_TRAINING_BEV = {
    "forward_min_m": -20.0,
    "forward_max_m": 30.0,
    "left_min_m": -25.0,
    "left_max_m": 25.0,
    "boundary_margin_m": 1.0,
}


def conservative_training_bev_relation(item, ego, *, forward_min_m=-20.0,
                                       forward_max_m=30.0,
                                       left_min_m=-25.0,
                                       left_max_m=25.0,
                                       boundary_margin_m=1.0):
    """Classify whether an observed actor can affect a supervised BEV cell.

    ``training_bev_relevant=False`` is returned only when a circle enclosing
    the complete observed horizontal box, plus a safety margin, is disjoint
    from the BEV rectangle.  Malformed or ambiguous observations remain
    relevant so that downstream geometry validation masks them rather than
    silently treating them as background.
    """
    bounds = [forward_min_m, forward_max_m, left_min_m, left_max_m]
    if (not all(math.isfinite(float(value)) for value in bounds)
            or forward_min_m >= forward_max_m or left_min_m >= left_max_m
            or not math.isfinite(float(boundary_margin_m))
            or boundary_margin_m < 0):
        raise ValueError("Invalid training BEV bounds")

    position = item.get("position")
    dimensions = item.get("size_wlh_m")
    yaw = ego.get("yaw")
    ego_x, ego_y = ego.get("x"), ego.get("y")
    values = []
    if isinstance(position, list):
        values.extend(position[:2])
    if isinstance(dimensions, list):
        values.extend(dimensions[:2])
    values.extend((ego_x, ego_y, yaw))
    valid = (
        isinstance(position, list) and len(position) >= 2
        and isinstance(dimensions, list) and len(dimensions) >= 2
        and len(values) == 7
        and all(value is not None and math.isfinite(float(value)) for value in values)
        and float(dimensions[0]) > 0 and float(dimensions[1]) > 0
        and item.get("position_semantics") == "grpc_bounding_box_world_center"
    )
    if not valid:
        return {
            "training_bev_relevant": True,
            "bev_relation": "indeterminate_fail_closed",
            "bev_exclusion_proven": False,
            "reason": "invalid_or_ambiguous_observed_box",
        }

    dx = float(position[0]) - float(ego_x)
    dy = float(position[1]) - float(ego_y)
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    forward = c * dx + s * dy
    left = -s * dx + c * dy
    separation_forward = max(float(forward_min_m) - forward, 0.0,
                             forward - float(forward_max_m))
    separation_left = max(float(left_min_m) - left, 0.0,
                          left - float(left_max_m))
    separation = math.hypot(separation_forward, separation_left)
    # This radius encloses the horizontal box for every possible heading.
    enclosing_radius = 0.5 * math.hypot(float(dimensions[0]),
                                        float(dimensions[1]))
    exclusion_threshold = enclosing_radius + float(boundary_margin_m)
    outside = separation > exclusion_threshold
    return {
        "training_bev_relevant": not outside,
        "bev_relation": ("definitely_outside" if outside
                         else "inside_or_boundary_guard"),
        "bev_exclusion_proven": outside,
        "center_forward_m": forward,
        "center_left_m": left,
        "distance_to_bev_m": separation,
        "enclosing_radius_m": enclosing_radius,
        "boundary_margin_m": float(boundary_margin_m),
        "reason": None,
    }
