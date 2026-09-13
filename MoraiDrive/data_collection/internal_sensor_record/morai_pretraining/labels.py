"""Training-only targets, with explicit source-clock and geometry contracts."""
import math
import numpy as np
import cv2
from .object_exclusion import definitely_outside_bev

HORIZONS = np.array([.5, 1., 1.5, 2., 2.5])


def gps_ego_audit(frames, interval_s=2.):
    """Compare motion over receipt-aligned intervals, not cross-sensor headers.

    This is a gross consistency gate for the earlier low-speed Ego failure,
    not an absolute simulator latency or GPS accuracy measurement.
    """
    bad, comparisons, unavailable = set(), [], 0
    start = 0
    for end in range(len(frames)):
        while start + 1 < end and (frames[end]["timestamp_ns"] - frames[start + 1]["timestamp_ns"]) / 1e9 >= interval_s:
            start += 1
        if (frames[end]["timestamp_ns"] - frames[start]["timestamp_ns"]) / 1e9 < interval_s:
            continue
        a, b = frames[start], frames[end]
        ga, gb = a.get("gnss_raw", {}), b.get("gnss_raw", {})
        if not ga or not gb or any(k not in g for g in (ga, gb) for k in ("latitude", "longitude")):
            unavailable += 1
            continue
        values = [ga["latitude"], gb["latitude"], ga["longitude"], gb["longitude"]]
        if not np.isfinite(values).all() or any(abs(g["latitude"]) < 1e-6 and abs(g["longitude"]) < 1e-6 for g in (ga, gb)):
            unavailable += 1
            continue
        lat = math.radians(.5 * (ga["latitude"] + gb["latitude"]))
        east = math.radians(gb["longitude"] - ga["longitude"]) * math.cos(lat) * 6371000
        north = math.radians(gb["latitude"] - ga["latitude"]) * 6371000
        ea, eb = a.get("ego_raw", {}), b.get("ego_raw", {})
        if not all(k in e for e in (ea, eb) for k in ("x", "y")):
            unavailable += 1
            continue
        gps_dist, ego_dist = math.hypot(east, north), math.hypot(eb["x"] - ea["x"], eb["y"] - ea["y"])
        mismatch = abs(gps_dist - ego_dist) > max(5., .4 * max(gps_dist, ego_dist))
        comparisons.append({"start_index": start, "end_index": end,
                            "gps_displacement_m": gps_dist, "ego_displacement_m": ego_dist,
                            "mismatch": mismatch})
        if mismatch:
            bad.update(range(start, end + 1))
    moving = sum(max(x["gps_displacement_m"], x["ego_displacement_m"]) > 5 for x in comparisons)
    return bad, {"moving_comparisons": moving, "unavailable_comparisons": unavailable,
                 "mismatch_comparisons": sum(x["mismatch"] for x in comparisons),
                 "status": "mismatch" if bad else "consistent_motion" if moving else "motion_not_validated",
                 "comparisons": comparisons}


def local_flu(xy, origin):
    d = np.asarray(xy, dtype=np.float64) - [origin["x"], origin["y"]]
    c, s = math.cos(origin["yaw"]), math.sin(origin["yaw"])
    return d @ np.array([[c, -s], [s, c]])


def waypoint_target(frames, index, max_gap_s=.25, allow_identical_source_repeat=False, motion_quality=None):
    """0.5..2.5 seconds in the Ego source clock, never sensor-clock subtraction.

    The origin is the causal Ego sample associated with the LiDAR receipt.
    Full horizon support is mandatory; resets, gaps and implausible motion fail.
    """
    if motion_quality is not None:
        if motion_quality.source_frames is not frames:raise ValueError('Motion audit belongs to another timeline')
        if index in motion_quality.bad_indices:return None, 'dense_ego_motion_inconsistent'
    origin = frames[index].get("ego_raw") or {}
    ts = origin.get("source_timestamp_ns")
    if not ts or not all(k in origin for k in ("x", "y", "yaw", "speed")):
        return None, "missing_ego_source"
    times, positions = [0.], [[origin["x"], origin["y"]]]
    prev = origin
    for future_index, f in enumerate(frames[index + 1:],index+1):
        if motion_quality is not None and future_index in motion_quality.bad_indices:
            return None, 'dense_ego_motion_inconsistent'
        e = f.get("ego_raw") or {}
        st = e.get("source_timestamp_ns")
        if not st or not all(k in e for k in ("x", "y", "yaw", "speed")):
            return None, "missing_future_ego"
        dt = (st - prev["source_timestamp_ns"]) / 1e9
        dr = (e["timestamp_ns"] - prev["timestamp_ns"]) / 1e9
        vals = [e["x"], e["y"], e["yaw"], e["speed"], *positions[-1]]
        if not np.isfinite(vals).all():
            return None, "nonfinite_ego"
        if dt < 0 or dr <= 0:
            return None, "ego_clock_reset_or_duplicate"
        if dt > max_gap_s or dr > max_gap_s:
            return None, "ego_gap"
        if dt == 0:
            if allow_identical_source_repeat and all(e.get(k)==prev.get(k) for k in ('x','y','z','yaw','speed','steer')):
                # Preserve all raw observations. Identical repeats supply no
                # extra simulation time; a long stalled source still fails dr.
                continue
            return None, "ego_clock_reset_or_duplicate"
        distance = math.hypot(e["x"] - prev["x"], e["y"] - prev["y"])
        expected = .5 * (e["speed"] + prev["speed"])
        if motion_quality is None and abs(distance / dt - expected) > max(5., .5 * expected):
            return None, "ego_position_velocity_inconsistent"
        times.append((st - ts) / 1e9)
        positions.append([e["x"], e["y"]])
        prev = e
        if times[-1] >= HORIZONS[-1]:
            xyz = np.asarray(positions)
            future = np.stack([np.interp(HORIZONS, times, xyz[:, j]) for j in (0, 1)], axis=-1)
            flu = local_flu(future, origin)
            # Native LMDrive waypoint head: right, negative-forward. Raw stays FLU.
            result = np.stack([-flu[:, 1], -flu[:, 0]], axis=-1).astype(np.float32)
            if not np.isfinite(result).all():
                return None, "nonfinite_waypoint"
            return result, None
    return None, "incomplete_future_horizon"


def capture_valid(frame):
    cap = frame.get("capture", {})
    if cap.get("schema") != "morai_visual_capture_v1" or cap.get("sync_clock") != "receive_monotonic_ns":
        return False, "missing_causal_capture_metadata"
    receipts = cap.get("receipts", {})
    required = ("cam_front", "cam_left", "cam_right", "cam_rear", "lidar", "gnss", "imu", "ego")
    if not all(k in receipts for k in required):
        return False, "missing_receipts"
    anchor = receipts["lidar"]["receive_monotonic_ns"]
    if any(not 0 <= anchor - receipts[k]["receive_monotonic_ns"] <= cap["max_sensor_age_ns"]
           for k in required):
        return False, "noncausal_or_stale_input"
    if anchor != frame.get("timestamp_ns"):
        return False, "anchor_mismatch"
    if (frame.get("ego_raw") or {}).get("source_timestamp_ns") != receipts["ego"].get("source_timestamp_ns"):
        return False, "ego_source_metadata_mismatch"
    return True, None


def light_target(frame, excluded_light_ids=()):
    nav = frame.get("navigation", {})
    binding = nav.get("signal_binding")
    key = binding.get("light_id") if binding else nav.get("affected_light_id_raw")
    if str(key) in excluded_light_ids:
        return -1
    valid_binding = binding.get("valid") if binding else nav.get("route_valid")
    if not key or valid_binding is not True:
        return -1
    cap = frame.get("capture", {})
    rec = cap.get("traffic_light_receipts", {}).get(str(key))
    if not rec or not 0 <= frame["timestamp_ns"] - rec["receive_monotonic_ns"] <= 10**9:
        return -1
    states = {str(x["tl_id"]): int(x["state"]) for x in frame.get("traffic_lights", [])}
    raw = states.get(str(key))
    # This task is pure red/green, not right of way. Mixed, unknown, arrow,
    # yellow and red+green bitfields must not silently become pure labels.
    return {1: 0, 16: 1}.get(raw, -1)


def object_target(frame, geometry_verified, exclusion_policy=None):
    """Native 50x50 detection contract; partial top-20 lists cannot label all background.

    Cells: row=29.5-forward, col=24.5+right, 1 metre/cell. Channels are
    occupancy, dx/3.5, dy/3.5, heading(right,-forward), half-length/3.5,
    half-width/2, speed/8. Shapes and losses match the native MVTL1 head.
    """
    empty = np.zeros((2500, 8), np.float32)
    cap = frame.get("capture", {})
    rec = cap.get("receipts", {}).get("objects")
    objects = frame.get("gt_objects_raw", [])
    grpc_bounds = cap.get("object_box_contract") == "grpc_actor_bounds_v1"
    if grpc_bounds:
        inventory = cap.get("actor_ground_truth", {})
        if not inventory.get("valid") or not inventory.get("normalization_verified"):
            return empty, False, "unverified_or_inconsistent_actor_inventory"
        rec = cap.get("receipts", {}).get("actor_inventory")
    if not geometry_verified or (not grpc_bounds and not cap.get("object_messages_present")) or not rec:
        return empty, False, "missing_or_unverified_objects"
    if not 0 <= frame["timestamp_ns"] - rec["receive_monotonic_ns"] <= cap["max_sensor_age_ns"]:
        return empty, False, "stale_objects"
    if not grpc_bounds and len(objects) >= cap.get("object_list_limit", 20):
        return empty, False, "possibly_truncated_object_list"
    return render_object_target(frame, objects, grpc_bounds, exclusion_policy)


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
            if definitely_outside_bev(o, frame, exclusion_policy):
                continue
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
