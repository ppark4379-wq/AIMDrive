"""MORAI sealed geometry -> legacy LMDrive 50x50 targets (v3_lav convention)."""
import math
import numpy as np
import cv2

def local_flu(xy, origin):
    d = np.asarray(xy, dtype=np.float64) - [origin["x"], origin["y"]]
    c, s = math.cos(origin["yaw"]), math.sin(origin["yaw"])
    return d @ np.array([[c, -s], [s, c]])


def render_object_target(frame, objects, grpc_bounds=True, exclusion_policy=None):
    """Geometry renderer shared after each transport validates its own timing.

    Native frame acquisition must not fabricate ROS receipts just to reuse GT.
    """
    empty = np.zeros((2500, 8), np.float32)
    cap = frame.get("capture", {})
    origin = frame["ego_raw"]
    converted = []
    for o in objects:
        if (grpc_bounds or cap.get("object_box_contract") == "per_model_actor_bounds_v2") and not o.get("geometry_valid"):
            return empty, False, "unverified_object_model"
        vals = [o[k] for k in ("x", "y", "heading_deg", "width", "length", "height", "vel_x", "vel_y")]
        if not np.isfinite(vals).all() or min(o["width"], o["length"], o["height"]) <= 0:
            return empty, False, "invalid_object_geometry"
        flu = local_flu([[o["x"], o["y"]]], origin)[0]
        if not grpc_bounds and np.linalg.norm(flu) < 1.:  # Legacy ROS self-entry heuristic.
            continue
        right, forward = -flu[1], flu[0]
        heading = math.radians(o["heading_deg"]) - origin["yaw"]
        ori = np.array([-math.sin(heading), -math.cos(heading)])
        converted.append((right, forward, ori, o))
    if not converted:
        return empty, True, None  # Fresh, complete and verified empty observation.
    heatmap = np.zeros((250, 250), np.uint8)
    owners = np.full((50, 50), -1, dtype=np.int32)
    owner_distance = np.full((50, 50), np.inf)
    rows, cols = np.indices((50, 50))
    for index, (right, forward, ori, o) in enumerate(converted):
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
    occupied = heatmap.reshape(50, 5, 50, 5).mean(axis=(1, 3)) >= .05 * 255
    result = empty.reshape(50, 50, 8)
    for row, col in np.argwhere(occupied):
        cx, cy = col - 24.5, 29.5 - row
        # A nearby object's centre may lie outside this occupied footprint.
        # Assign only among objects whose actual box touches this cell.
        obj = converted[owners[row, col]]
        right, forward, ori, o = obj
        dx, dy = right - cx, forward - cy
        prob = (.5 / max(.5, math.hypot(dx, dy) - math.sqrt(.5)))**.5
        result[row, col] = [prob, dx / 3.5, dy / 3.5, *ori,
                            o["length"] / 7, o["width"] / 4,
                            math.hypot(o["vel_x"], o["vel_y"]) / 8]
    return result.reshape(2500, 8), True, None
