"""World XY (metres) projected onto the official route's directed XY polyline.

Positive path_d_m and v_d_mps point left of the local route tangent. Velocity
from the GT timeline is world XYZ in km/h; output velocity is world XY in m/s.
Ego forward/left coordinates are copied from the geometry audit, not projected.
"""

import math


class RouteProjector:
    """Precompute segment lengths and route distance once per official path."""

    def __init__(self, waypoints):
        self.segments = []
        distance = 0.0
        for index, (a, b) in enumerate(zip(waypoints, waypoints[1:])):
            ax, ay = float(a[0]), float(a[1])
            dx, dy = float(b[0]) - ax, float(b[1]) - ay
            length = math.hypot(dx, dy)
            if not all(math.isfinite(v) for v in (ax, ay, dx, dy)):
                raise ValueError("nonfinite official route point")
            if length <= 0:
                continue
            tx, ty = dx / length, dy / length
            self.segments.append((index, ax, ay, tx, ty, length, distance))
            distance += length
        if not self.segments:
            raise ValueError("official route has no nonzero XY segments")

    def project(self, x, y):
        """Return (s, signed d, source segment index, unit tangent) in metres."""
        x, y = float(x), float(y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("nonfinite object world position")
        best = None
        for index, ax, ay, tx, ty, length, start_s in self.segments:
            along = min(length, max(0.0, (x - ax) * tx + (y - ay) * ty))
            px, py = ax + along * tx, ay + along * ty
            distance_sq = (x - px) ** 2 + (y - py) ** 2
            if best is None or distance_sq < best[0]:
                best = (distance_sq, start_s + along,
                        -(x - px) * ty + (y - py) * tx, index, tx, ty)
        return best[1:]


def numeric_features(gt_object, geometry_object, projector):
    """Compute independent ego and route coordinates for one matched actor."""
    result = {}
    for target, source in (("forward_m", "center_forward_m"),
                           ("left_m", "center_left_m")):
        value = geometry_object.get(source)
        if value is None:
            raise ValueError("missing_" + source)
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("nonfinite_" + source)
        result[target] = value
    # The sealed GT and geometry audit name this axis order width, length,
    # height (size_wlh_m); keep that convention without inferring a new box.
    size = gt_object.get("size_wlh_m")
    if not isinstance(size, list) or len(size) != 3:
        raise ValueError("missing_size_wlh_m")
    width, length = float(size[0]), float(size[1])
    if not all(math.isfinite(value) and value > 0 for value in (width, length)):
        raise ValueError("invalid_size_wlh_m")
    result["actor_width_m"] = width
    result["actor_length_m"] = length
    position = gt_object.get("position")
    if not isinstance(position, list) or len(position) < 2:
        raise ValueError("missing_world_position")
    velocity = gt_object.get("velocity_kmh")
    if not isinstance(velocity, list) or len(velocity) < 2:
        raise ValueError("missing_velocity_kmh")
    vx, vy = float(velocity[0]) / 3.6, float(velocity[1]) / 3.6
    if not math.isfinite(vx) or not math.isfinite(vy):
        raise ValueError("nonfinite_velocity_kmh")
    x, y = float(position[0]), float(position[1])
    s, d, index, tx, ty = projector.project(x, y)
    result.update(speed_mps=math.hypot(vx, vy), vx_mps=vx, vy_mps=vy,
                  world_x_m=x, world_y_m=y,
                  path_s_m=s, path_d_m=d, path_segment_index=index,
                  path_tangent_x=tx, path_tangent_y=ty,
                  v_s_mps=vx * tx + vy * ty,
                  v_d_mps=-vx * ty + vy * tx)
    return result
