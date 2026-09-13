"""Measured ROS extrinsics and explicit projection hypotheses, without ROS imports.

TF stores child-to-parent transforms. Camera TF is a body frame here, not an
optical frame. Projection assumes +x forward, +y left, +z up and square pixels.
Neither a saved MORAI preset nor a successful TF check certifies active optics.
"""
import math
import numpy as np


def quaternion_matrix(rotation):
    q = np.array([rotation[k] for k in ("x", "y", "z", "w")], dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or not .99 <= norm <= 1.01:
        raise ValueError("Invalid TF quaternion")
    x, y, z, w = q / norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def tf_matrix(transform):
    result = np.eye(4)
    result[:3, :3] = quaternion_matrix(transform["rotation"])
    result[:3, 3] = [transform["translation"][k] for k in "xyz"]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite TF translation")
    return result


def zero_roll_preset_matrix(pos, rot):
    """Ry(pitch) @ Rz(yaw), verified against this MORAI session's zero-roll TF.

    Deliberately reject other rolls: general MORAI Euler order is unverified.
    Never reconstruct production extrinsics from these UI angles; use TF.
    """
    roll, pitch, yaw = [float(rot[k]) for k in ("roll", "pitch", "yaw")]
    if not all(math.isfinite(v) for v in (roll, pitch, yaw)) or abs(roll) > 1e-6:
        raise ValueError("Only measured zero-roll preset convention is supported")
    p, y = np.deg2rad([pitch, yaw])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    result = np.eye(4)
    result[:3, :3] = ry @ rz
    result[:3, 3] = [float(pos[k]) for k in "xyz"]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite preset position")
    return result


def transform_points(points, target_from_source):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected Nx3 points")
    return points @ target_from_source[:3, :3].T + target_from_source[:3, 3]


def project_body(points, width, height, fov_deg, fov_axis):
    """Project a *candidate* pinhole camera model; keep behind-camera points out."""
    if (width <= 0 or height <= 0 or not math.isfinite(fov_deg)
            or not 0 < fov_deg < 180 or fov_axis not in ("horizontal", "vertical")):
        raise ValueError("Invalid projection hypothesis")
    extent = width if fov_axis == "horizontal" else height
    focal = extent / (2 * math.tan(math.radians(fov_deg) / 2))
    points = np.asarray(points, dtype=float)
    depth = points[:, 0]
    valid = np.isfinite(points).all(axis=1)
    valid[valid] &= depth[valid] > .05
    uv = np.full((len(points), 2), np.nan)
    uv[valid, 0] = width / 2 - focal * points[valid, 1] / depth[valid]
    uv[valid, 1] = height / 2 - focal * points[valid, 2] / depth[valid]
    indices = np.flatnonzero(valid)
    visible = uv[indices]
    valid[indices] = ((visible[:, 0] >= 0) & (visible[:, 0] < width)
                      & (visible[:, 1] >= 0) & (visible[:, 1] < height))
    return uv, valid


def audit_snapshot(snapshot, preset, bridge_url="ws://172.18.40.28:9090", confirmed_fovs=None):
    from aimdrive.competition.sensor_spec import CURRENT_SENSOR_SPEC
    errors, cameras, sensors = [], {}, {}
    fixed = dict(zip(("Camera-1", "Camera-2", "Camera-3"), CURRENT_SENSOR_SPEC.fixed_cameras))
    transforms = {key.lstrip("/"): value for key, value in snapshot["transforms"].items()}
    groups = (("cameraList", "cc"), ("Lidar3DList", "lc"), ("GPSList", "gc"), ("IMUList", "ic"))
    expected_counts = (4, 1, 1, 1)
    for (group, config_key), expected_count in zip(groups, expected_counts):
        rows = preset.get(group, [])
        if len(rows) != expected_count:
            errors.append("%s count %s != %s" % (group, len(rows), expected_count))
        for row in rows:
            # GPS/IMU serialized config names vary by simulator build.
            config = row.get(config_key)
            if config is None:
                candidates = [v for v in row.values() if isinstance(v, dict) and "rosConfig" in v]
                config = candidates[0] if len(candidates) == 1 else {}
            ros = config.get("rosConfig", {})
            frame = ros.get("frameID", "")
            if not frame or frame in sensors:
                errors.append("Missing/duplicate sensor frame in " + group)
                continue
            sensor = sensors[frame] = dict(preset_position=row.get("pos"), preset_rpy=row.get("rot"))
            if config.get("commType") != 3 or ros.get("RosBridgeServerUrl") != bridge_url:
                errors.append(frame + ": saved network does not match ROS bridge")
            live = transforms.get("base_link->" + frame)
            if live is None:
                errors.append(frame + ": missing base_link TF")
                continue
            try:
                actual = tf_matrix(live["transform"]["transform"])
                expected = zero_roll_preset_matrix(row["pos"], row["rot"])
                position_error = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
                rotation_error = float(np.rad2deg(np.arccos(np.clip((np.trace(actual[:3, :3].T @ expected[:3, :3])-1)/2, -1, 1))))
                sensor.update(measured_child_to_base=actual.tolist(), position_error_m=position_error,
                              rotation_error_deg=rotation_error, receive_monotonic_ns=live["receive_monotonic_ns"])
                if position_error > .005 or rotation_error > .05:
                    errors.append(frame + ": saved extrinsics differ from live TF")
            except (KeyError, ValueError) as exc:
                errors.append(frame + ": " + str(exc))
            topic = ros.get("Topic", "")
            rate = snapshot.get("rates", {}).get(topic + "|" + frame, {})
            if rate.get("samples", 0) < 2 or rate.get("age_ms", float("inf")) > 500:
                errors.append(frame + ": absent/stale sensor callbacks")
            if group == "cameraList":
                size = [config.get("cameraResWidth"), config.get("cameraResHeight")]
                if frame not in ("Camera-1", "Camera-2", "Camera-3", "Camera-4"):
                    errors.append("Unexpected camera frame " + frame)
                if snapshot.get("images", {}).get(frame, {}).get("size") != size:
                    errors.append(frame + ": saved resolution differs from JPEG")
                if topic != "/image_jpeg/compressed":
                    errors.append(frame + ": unexpected image topic")
                fov = float(config["cameraFOV"])
                if not math.isfinite(fov) or not 0 < fov < 180:
                    errors.append(frame + ": invalid FOV")
                cameras[frame] = dict(size=size, saved_fov_deg=fov,
                    active_fov_confirmation="unconfirmed", fov_axis="unconfirmed",
                    model_type=config.get("modelType"), param_type=config.get("paramType"),
                    saved_lens_distortion=config.get("lensDistortion"), intrinsics_K=None)
                if frame in fixed:
                    spec = fixed[frame]
                    position = [float(row["pos"][k]) for k in "xyz"]
                    rpy = [float(row["rot"][k]) for k in ("roll", "pitch", "yaw")]
                    angle_error = (np.array(rpy) - spec.rpy_deg + 180) % 360 - 180
                    if (not np.allclose(position, spec.position_m, atol=.001)
                            or np.max(np.abs(angle_error)) > .01 or abs(fov-spec.fov_deg) > .01
                            or any(a > b or a <= 0 for a, b in zip(size, spec.max_resolution))):
                        errors.append(frame + ": preset differs from repository sensor specification")
                if confirmed_fovs is not None:
                    confirmed = confirmed_fovs.get(frame)
                    if confirmed != fov:
                        errors.append(frame + ": UI-confirmed FOV differs from preset")
                    else:
                        cameras[frame]["active_fov_confirmation"] = "operator_confirmed"
    errors.extend("probe: " + str(error) for error in snapshot.get("errors", []))
    return dict(schema_version="morai-sensor-calibration-audit-v1", errors=errors,
                measured_setup_matches_preset=not errors, calibration_complete=False,
                sensors=sensors, cameras=cameras,
                limits=["Saved file is not proof of the active camera optics",
                        "FOV axis, optical convention and distortion require projection verification",
                        "TF checked at snapshot time; not continuous mounting-drift monitoring",
                        "Cross-sensor source/header clocks are not assumed shared"])
