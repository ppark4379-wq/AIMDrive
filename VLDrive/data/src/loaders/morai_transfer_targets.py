"""MORAI sealed geometry -> legacy LMDrive 50x50 targets (v3_lav convention)."""
import math
import numpy as np
import cv2


OBJECT_TARGET_POLICIES = ("footprint_v1", "small_object_support_v1")
# These are FULL lengths in metres, not half extents and not new GT dimensions.
# This bounded MORAI diagnostic variant is not the original LMDrive recipe.
SMALL_OBJECT_SUPPORT_MIN_SIZE_M = 1.5
SMALL_OBJECT_SUPPORT_PROVENANCE = (
    "MORAI diagnostic: oriented classification support with each full dimension "
    "at least 1.5m; physical-footprint ownership has priority; physical box "
    "regression is unchanged; only physical boxes with positive-area BEV "
    "intersection receive support. Not an exact LMDrive paper recipe."
)


def local_flu(xy, origin):
    d = np.asarray(xy, dtype=np.float64) - [origin["x"], origin["y"]]
    c, s = math.cos(origin["yaw"]), math.sin(origin["yaw"])
    return d @ np.array([[c, -s], [s, c]])



def _physical_box_intersects_bev(right, forward, ori, length, width):
    """Strict positive-area rectangle intersection, before any pixel rounding.

    The BEV is right [-25,25], forward [-20,30], as in object evaluation.
    Separating axes of the two rectangles exclude tangent-only contact and
    rotated boxes whose axis-aligned bounds overlap only at a BEV corner.
    A genuine arbitrarily thin overlap remains eligible for added support.
    """
    center = np.array([right, forward - 5.], dtype=np.float64)
    heading = np.array([ori[0], -ori[1]], dtype=np.float64)
    transverse = np.array([-heading[1], heading[0]], dtype=np.float64)
    for axis in (np.array([1., 0.]), np.array([0., 1.]), heading, transverse):
        object_radius = (length * abs(float(heading @ axis))
                         + width * abs(float(transverse @ axis))) / 2.
        bev_radius = 25. * (abs(axis[0]) + abs(axis[1]))
        if abs(float(center @ axis)) >= object_radius + bev_radius:
            return False
    return True


def render_object_target(frame, objects, grpc_bounds=True, exclusion_policy=None, *,
                         target_policy="footprint_v1", return_instance_ids=False):
    """Geometry renderer shared after each transport validates its own timing.

    Native frame acquisition must not fabricate ROS receipts just to reuse GT.
    ``small_object_support_v1`` broadens only classification/ownership support,
    retaining the original physical centre, dimensions, direction and speed.
    Physical box coverage always wins over another object's expanded support;
    equal-distance ownership ties retain the first object in input order.

    The legacy return is ``(target, valid, reason)``. With ``return_instance_ids``
    it additionally includes int64 IDs of shape (2500,), indexing the converted
    object list (after the legacy ROS self-entry filter), with -1 on inactive
    cells and invalid/empty observations.
    """
    if target_policy not in OBJECT_TARGET_POLICIES:
        raise ValueError("Unknown object target policy: %s" % target_policy)

    empty = np.zeros((2500, 8), np.float32)

    def finish(target, valid, reason, instance_ids=None):
        if return_instance_ids:
            if instance_ids is None:
                instance_ids = np.full(2500, -1, dtype=np.int64)
            return target, valid, reason, instance_ids
        return target, valid, reason

    cap = frame.get("capture", {})
    origin = frame["ego_raw"]
    converted = []
    for o in objects:
        if (grpc_bounds or cap.get("object_box_contract") == "per_model_actor_bounds_v2") and not o.get("geometry_valid"):
            return finish(empty, False, "unverified_object_model")
        vals = [o[k] for k in ("x", "y", "heading_deg", "width", "length", "height", "vel_x", "vel_y")]
        if not np.isfinite(vals).all() or min(o["width"], o["length"], o["height"]) <= 0:
            return finish(empty, False, "invalid_object_geometry")
        flu = local_flu([[o["x"], o["y"]]], origin)[0]
        if not grpc_bounds and np.linalg.norm(flu) < 1.:  # Legacy ROS self-entry heuristic.
            continue
        right, forward = -flu[1], flu[0]
        heading = math.radians(o["heading_deg"]) - origin["yaw"]
        ori = np.array([-math.sin(heading), -math.cos(heading)])
        converted.append((right, forward, ori, o))
    if not converted:
        return finish(empty, True, None)  # Fresh, complete and verified empty observation.
    heatmap = np.zeros((250, 250), np.uint8)
    owners = np.full((50, 50), -1, dtype=np.int32)
    owner_distance = np.full((50, 50), np.inf)
    rows, cols = np.indices((50, 50))
    support_eligible = np.ones(len(converted), dtype=bool)
    for index, (right, forward, ori, o) in enumerate(converted):
        if target_policy == "small_object_support_v1":
            support_eligible[index] = _physical_box_intersects_bev(
                right, forward, ori, o["length"], o["width"])
            if not support_eligible[index]:
                continue
        center = np.array([right, -forward])
        along = ori * o["length"] / 2
        across = np.array([-ori[1], ori[0]]) * o["width"] / 2
        corners = np.array([center + along + across, center + along - across,
                            center - along - across, center - along + across])
        pixels = np.rint(corners * 5 + [125, 150]).astype(np.int32)
        footprint = np.zeros_like(heatmap)
        cv2.fillConvexPoly(footprint, pixels, 255)
        heatmap = np.maximum(heatmap, footprint)
        touched = footprint.reshape(50, 5, 50, 5).max(axis=(1, 3)) > 0
        distance = (right - (cols - 24.5))**2 + (forward - (29.5 - rows))**2
        replace = touched & (distance < owner_distance)
        owners[replace] = index
        owner_distance[replace] = distance[replace]

    if target_policy == "small_object_support_v1":
        # Finish all physical ownership first. Expanded support may only fill
        # cells touched by no physical box, regardless of object input order.
        physical_touched = owners >= 0
        for index, (right, forward, ori, o) in enumerate(converted):
            if not support_eligible[index]:
                continue
            support_length = max(o["length"], SMALL_OBJECT_SUPPORT_MIN_SIZE_M)
            support_width = max(o["width"], SMALL_OBJECT_SUPPORT_MIN_SIZE_M)
            if support_length == o["length"] and support_width == o["width"]:
                continue
            center = np.array([right, -forward])
            along = ori * support_length / 2
            across = np.array([-ori[1], ori[0]]) * support_width / 2
            corners = np.array([center + along + across, center + along - across,
                                center - along - across, center - along + across])
            pixels = np.rint(corners * 5 + [125, 150]).astype(np.int32)
            support = np.zeros_like(heatmap)
            cv2.fillConvexPoly(support, pixels, 255)
            heatmap = np.maximum(heatmap, support)
            touched = support.reshape(50, 5, 50, 5).max(axis=(1, 3)) > 0
            distance = (right - (cols - 24.5))**2 + (forward - (29.5 - rows))**2
            replace = touched & ~physical_touched & (distance < owner_distance)
            owners[replace] = index
            owner_distance[replace] = distance[replace]

    occupied = heatmap.reshape(50, 5, 50, 5).mean(axis=(1, 3)) >= .05 * 255
    result = empty.reshape(50, 50, 8)
    for row, col in np.argwhere(occupied):
        cx, cy = col - 24.5, 29.5 - row
        # A nearby object's centre may lie outside this occupied footprint.
        # The legacy policy assigns only among actual box touches. Expanded
        # support retains that ownership wherever a physical box touches.
        obj = converted[owners[row, col]]
        right, forward, ori, o = obj
        dx, dy = right - cx, forward - cy
        prob = (.5 / max(.5, math.hypot(dx, dy) - math.sqrt(.5)))**.5
        result[row, col] = [prob, dx / 3.5, dy / 3.5, *ori,
                            o["length"] / 7, o["width"] / 4,
                            math.hypot(o["vel_x"], o["vel_y"]) / 8]
    instance_ids = None
    if return_instance_ids:
        instance_ids = np.where(occupied, owners, -1).astype(np.int64).reshape(2500)
    return finish(result.reshape(2500, 8), True, None, instance_ids)
