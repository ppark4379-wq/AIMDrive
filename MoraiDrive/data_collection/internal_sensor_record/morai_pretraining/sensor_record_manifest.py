"""Build and read visual-pretraining labels from sealed Sensor Record runs.

The native PNG/BIN bytes stay on the configured local spool. A manifest references those
bytes by their audited relative path and SHA256; it does not duplicate them.
Teacher state is serialized only as a target and never returned as an encoder
input.
"""
from async_capture.route_sources import resolve_audited_map_root
from bisect import bisect_left
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .labels import HORIZONS, render_object_target


LEGACY_SCHEMA = "morai_sensor_record_visual_manifest_v1"
SCHEMA = "morai_sensor_record_visual_manifest_v2"
REQUIRED_ROLES = ("camera_1", "camera_2", "camera_3", "camera_4", "lidar", "gps", "imu")


def _safe_child(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe Sensor Record asset path: " + str(relative))
    path = (Path(root) / relative).resolve()
    root = Path(root).resolve()
    if root not in path.parents:
        raise ValueError("Sensor Record asset escapes source root: " + str(relative))
    return path


def dense_ego_timeline(active_signals, max_gap_ns=50_000_000):
    """Collapse exact duplicate Ego callbacks into one simulation-time series."""
    unique = {}
    for row in active_signals:
        if row.get("role") != "ego" or row.get("source_timestamp_ns") is None:
            continue
        signal = row.get("signal") or {}
        position = signal.get("position") or []
        if len(position) < 2 or signal.get("yaw") is None or signal.get("speed") is None:
            raise ValueError("Incomplete Ego signal in sealed active timeline")
        value = (float(position[0]), float(position[1]),
                 float(signal["yaw"]), float(signal["speed"]))
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite Ego signal")
        stamp = int(row["source_timestamp_ns"])
        previous = unique.get(stamp)
        if previous is not None and previous != value:
            raise ValueError("Conflicting Ego values at one source timestamp")
        unique[stamp] = value
    if len(unique) < 2:
        raise ValueError("Insufficient Ego source timeline")
    times = np.asarray(sorted(unique), dtype=np.int64)
    values = np.asarray([unique[int(t)] for t in times], dtype=np.float64)
    gaps = np.diff(times)
    if np.any(gaps <= 0) or int(gaps.max()) > int(max_gap_ns):
        raise ValueError("Ego source timeline has a gap or reversal")
    values[:, 2] = np.unwrap(values[:, 2])
    return times, values


def interpolate_ego(times, values, stamp_ns, max_bracket_ns=50_000_000):
    stamp_ns = int(stamp_ns)
    pos = bisect_left(times, stamp_ns)
    if pos < len(times) and int(times[pos]) == stamp_ns:
        value = values[pos]
    else:
        if pos == 0 or pos == len(times):
            raise ValueError("Sensor time is outside Ego source timeline")
        if int(times[pos] - times[pos - 1]) > int(max_bracket_ns):
            raise ValueError("Sensor time lacks a bounded Ego source bracket")
        alpha = (stamp_ns - int(times[pos - 1])) / float(times[pos] - times[pos - 1])
        value = values[pos - 1] * (1.0 - alpha) + values[pos] * alpha
    return {"x": float(value[0]), "y": float(value[1]),
            "yaw": float(value[2]), "speed": float(value[3])}


def waypoint_at_sensor_time(times, values, stamp_ns):
    """LMDrive waypoint axes at 0.5..2.5 s from the sensor simulation time."""
    origin = interpolate_ego(times, values, stamp_ns)
    future_ns = int(stamp_ns) + np.rint(HORIZONS * 1_000_000_000).astype(np.int64)
    if int(future_ns[-1]) > int(times[-1]):
        raise ValueError("Incomplete 2.5 second future Ego horizon")
    future = np.column_stack([
        np.interp(future_ns, times, values[:, 0]),
        np.interp(future_ns, times, values[:, 1]),
    ])
    delta = future - [origin["x"], origin["y"]]
    c, s = math.cos(origin["yaw"]), math.sin(origin["yaw"])
    forward = c * delta[:, 0] + s * delta[:, 1]
    left = -s * delta[:, 0] + c * delta[:, 1]
    result = np.column_stack([-left, -forward]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite waypoint target")
    return origin, result


def coverage_failed_splits(index):
    """Splits whose only publication failure is a missing required example.

    Such a failed publication keeps sealed frames and run splits, so a
    geometry-only revision may use it as a source.  Any other state returns an
    empty list and remains unusable.
    """
    required = index.get("required_coverage") or []
    coverage = index.get("coverage") or {}
    if index.get("training_ready") is not False or not required or not coverage:
        return []
    return sorted(split for split, counts in coverage.items()
                  if any(int(counts.get(key, 0)) <= 0 for key in required))


def objects_for_entry(gt, geometry):
    geometry_rows = {str(row.get("instance_key")): row for row in geometry.get("objects", [])}
    result = []
    for obj in gt.get("object_gt", {}).get("objects", []):
        size = obj.get("size_wlh_m") or []
        position = obj.get("position") or []
        velocity = obj.get("velocity_kmh") or []
        if min(len(size), len(position), len(velocity)) < 3:
            raise ValueError("Incomplete object geometry in sealed GT timeline")
        key = str(obj.get("instance_key"))
        audit = geometry_rows.get(key, {})
        # V2 retains the complete MORAI inventory as evidence, but actors whose
        # conservative full box is proven disjoint from the supervised BEV do
        # not have a target cell and must not invalidate or enter the renderer.
        if audit.get("training_bev_relevant") is False:
            continue
        # V4 re-centres a box that MORAI reported at the actor pivot; the
        # sealed GT keeps the original API position as evidence.
        center = audit.get("training_center_world_xy") or position[:2]
        # V5 trains on the registry-selected MORAI size (e.g. /Object_topic).
        width, length = audit.get("training_size_wl_m") or size[:2]
        center_xyz = audit.get("training_center_world_xyz") or [center[0], center[1], position[2]]
        height = (audit.get("training_size_wlh_m") or size)[2]
        rotation = audit.get("training_rotation_rpy_deg") or obj.get("rotation_rpy_deg")
        result.append({
            "id": str(obj.get("id")),
            "instance_key": key,
            "model_name": str(obj.get("model_name") or obj.get("name") or ""),
            "x": float(center_xyz[0]), "y": float(center_xyz[1]), "z": float(center_xyz[2]),
            "heading_deg": float(rotation[2] if rotation is not None else obj["heading_deg"]),
            **({"rotation_rpy_deg": [float(v) for v in rotation]} if rotation is not None else {}),
            "width": float(width), "length": float(length), "height": float(height),
            "vel_x": float(velocity[0]) / 3.6, "vel_y": float(velocity[1]) / 3.6,
            "geometry_valid": bool(audit.get("api_box_geometry_candidate")),
        })
    return result


def detection_for_entry(entry):
    if not entry.get("object_valid"):
        return np.zeros((2500, 8), np.float32), False, "masked_by_sensor_record_audit"
    frame = {"ego_raw": entry["ego_pose"], "capture": {}}
    return render_object_target(frame, entry.get("objects", []), grpc_bounds=True)


def _verify_seal(folder, seal_name="COMPLETE.json"):
    from async_capture.sensor_record_admission import verify_sealed_directory
    return verify_sealed_directory(folder, seal_name)


def recovery_record_path(episode, recovery_name="ADMISSION_RECOVERY.json"):
    """Resolve an immutable recovery selected by a sealed manifest."""
    name = Path(str(recovery_name))
    if name.name != str(recovery_name) or name.suffix != ".json":
        raise ValueError("Admission recovery must be a run-local JSON filename")
    run = Path(episode).resolve().parent
    path = (run / name).resolve()
    if path.parent != run:
        raise ValueError("Admission recovery path escapes its run")
    return path


def effective_admission_directory(episode, recovery_name="ADMISSION_RECOVERY.json"):
    """Select an additive corrected admission while preserving admission_v1."""
    from async_capture.journal import digest
    episode = Path(episode).resolve()
    run = episode.parent
    original_dir = episode / "admission_v1"
    _verify_seal(original_dir)
    recovery_path = recovery_record_path(episode, recovery_name)
    if not recovery_path.is_file():
        if recovery_name != "ADMISSION_RECOVERY.json":
            raise ValueError("Selected admission recovery record is missing")
        return original_dir, {}

    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    recovery = load(recovery_path)
    original_path = original_dir / "report.json"
    if not (
        recovery.get("schema") == "morai_sensor_record_admission_recovery_v1"
        and recovery.get("status") == "completed"
        and recovery.get("recovery_is_additive") is True
        and recovery.get("source_timestamps_modified") is False
        and recovery.get("training_visible") is False
        and recovery.get("original_admission", {}).get("sha256") == digest(original_path)
    ):
        raise ValueError("Invalid additive admission recovery record")
    corrected_relative = recovery.get("corrected_admission", {}).get("path")
    corrected_path = (run / str(corrected_relative)).resolve()
    if (run.resolve() not in corrected_path.parents
            or corrected_path.name != "report.json"):
        raise ValueError("Admission recovery path escapes its run")
    corrected_dir = corrected_path.parent
    _verify_seal(corrected_dir)
    corrected = load(corrected_path)
    original = load(original_path)
    if not (
        recovery["corrected_admission"].get("sha256") == digest(corrected_path)
        and recovery["corrected_admission"].get("seal_sha256")
            == digest(corrected_dir / "COMPLETE.json")
        and corrected.get("raw_manifest_sha256") == original.get("raw_manifest_sha256")
        and corrected.get("capture_contract_valid") is True
        and corrected.get("runtime_streams_valid") is True
        and corrected.get("sensor_capture_status") == "admitted"
        and corrected.get("teacher_transport_status") == "admitted"
        and corrected.get("source_timestamps_modified") is False
        and corrected.get("training_visible") is False
    ):
        raise ValueError("Corrected admission evidence mismatch")
    seals = {
        "admission_recovery": digest(recovery_path),
        "effective_admission": digest(corrected_dir / "COMPLETE.json"),
    }
    geometry_record = recovery.get("corrected_object_geometry")
    if geometry_record is not None:
        geometry_relative = geometry_record.get("path")
        geometry_path = (run / str(geometry_relative)).resolve()
        if (run.resolve() not in geometry_path.parents
                or geometry_path.name != "report.json"):
            raise ValueError("Object geometry recovery path escapes its run")
        geometry_dir = geometry_path.parent
        _verify_seal(geometry_dir)
        geometry = load(geometry_path)
        if not (
            geometry_record.get("sha256") == digest(geometry_path)
            and geometry_record.get("seal_sha256")
                == digest(geometry_dir / "COMPLETE.json")
            and geometry.get("schema") in (
                "morai_sensor_record_object_geometry_audit_v2",
                "morai_sensor_record_object_geometry_audit_v3",
                "morai_sensor_record_object_geometry_audit_v4",
                "morai_sensor_record_object_geometry_audit_v5")
            and geometry.get("raw_manifest_sha256") == original.get("raw_manifest_sha256")
            and geometry.get("structural_valid") is True
            and corrected.get("input_seals", {}).get("object_geometry_audit")
                == digest(geometry_dir / "COMPLETE.json")
        ):
            raise ValueError("Corrected object geometry evidence mismatch")
        seals["effective_object_geometry"] = digest(geometry_dir / "COMPLETE.json")
    return corrected_dir, seals


def effective_object_geometry_directory(episode, recovery_name="ADMISSION_RECOVERY.json"):
    """Select the geometry ledger linked by the additive admission recovery."""
    episode = Path(episode).resolve()
    recovery_path = recovery_record_path(episode, recovery_name)
    if not recovery_path.is_file():
        if recovery_name != "ADMISSION_RECOVERY.json":
            raise ValueError("Selected admission recovery record is missing")
        directory = episode / "object_geometry_audit_v1"
        _verify_seal(directory)
        return directory
    # This validates the full recovery chain before returning its geometry.
    effective_admission_directory(episode, recovery_name)
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    if recovery.get("corrected_object_geometry") is None:
        directory = episode / "object_geometry_audit_v1"
        _verify_seal(directory)
        return directory
    path = (episode.parent / recovery["corrected_object_geometry"]["path"]).resolve()
    return path.parent


def verify_episode_evidence(episode, expected, recovery_name="ADMISSION_RECOVERY.json"):
    episode = Path(episode).resolve()
    stage_dirs = {
        "quality": ("quality", "QUALITY_COMPLETE.json"),
        "native_audit": ("native_audit", "COMPLETE.json"),
        "gt_timeline": ("gt_timeline_v1", "COMPLETE.json"),
        "object_geometry": ("object_geometry_audit_v1", "COMPLETE.json"),
        "route_signal": ("route_signal_audit_v1", "COMPLETE.json"),
        "motion_time": ("motion_time_v1", "COMPLETE.json"),
        "admission": ("admission_v1", "COMPLETE.json"),
    }
    from async_capture.journal import digest
    for key, (relative, seal_name) in stage_dirs.items():
        folder = episode / relative
        _verify_seal(folder, seal_name)
        if expected.get(key) != digest(folder / seal_name):
            raise ValueError("Sensor Record evidence seal changed: " + key)
    recovery_expected = (
        "admission_recovery" in expected or "effective_admission" in expected
    )
    if recovery_expected:
        _, recovery_seals = effective_admission_directory(episode, recovery_name)
        keys = ["admission_recovery", "effective_admission"]
        if "effective_object_geometry" in expected:
            keys.append("effective_object_geometry")
        for key in keys:
            if expected.get(key) != recovery_seals.get(key):
                raise ValueError("Sensor Record recovery seal changed: " + key)



def official_route_tracker(contract, route_report, episode):
    """Rebuild the exact audited competition route used for navigation input.

    The target is a point on the official global path 25 m ahead.  It is not
    derived from future Ego motion.  Both source files are checked against the
    already sealed route-signal audit before any target is published.
    """
    from async_capture.journal import digest
    from data_collection.mgeo_navigation import MGeoMap, MGeoRouteTracker, load_official_route

    from async_capture.route_sources import resolve_route_sources, verify_route_report
    sources = resolve_route_sources(episode, contract)
    verify_route_report(sources, route_report)
    map_root = sources["map_root"]
    route_definition = sources["route_definition"]
    global_path = sources["global_path"]
    route_key = sources["route_key"]
    world = MGeoMap(str(map_root / "link_set.json"),
                    node_set_path=str(map_root / "node_set.json"))
    route = load_official_route(str(route_definition), str(global_path), route_key)
    tracker = MGeoRouteTracker(world, route)
    provenance = {
        "source": "sealed_official_global_path_lookahead_v1",
        "route_key": route_key,
        "target_distance_m": float(tracker.target_distance_m),
        "route_topology": "closed_loop" if tracker.closed_route else "open",
        "route_point_count": int(tracker.route_point_count),
        "audited_map_root": str(map_root),
        "capture_copy_lf_crlf_equivalence_verified": sources["line_endings_equivalence_verified"],
        "audited_map_resolution": sources["resolution"],
        "route_source_resolution": sources["provenance"],
        "route_definition_sha256": route_report["route_definition_sha256"],
        "global_path_sha256": route_report["global_path_sha256"],
    }
    return tracker, provenance


def build_episode_manifest(episode, *, campaign_validation_sha256, split,
                           expected_evidence=None,
                           recovery_name="ADMISSION_RECOVERY.json",
                           published_native_root=None):
    """Build one lightweight manifest from an already admitted episode."""
    from async_capture.journal import digest
    episode = Path(episode).resolve()
    run = episode.parent
    stages = {
        "quality": (episode / "quality", "QUALITY_COMPLETE.json"),
        "native_audit": (episode / "native_audit", "COMPLETE.json"),
        "gt_timeline": (episode / "gt_timeline_v1", "COMPLETE.json"),
        "object_geometry": (episode / "object_geometry_audit_v1", "COMPLETE.json"),
        "route_signal": (episode / "route_signal_audit_v1", "COMPLETE.json"),
        "motion_time": (episode / "motion_time_v1", "COMPLETE.json"),
        "admission": (episode / "admission_v1", "COMPLETE.json"),
    }
    evidence = {}
    for key, (folder, seal_name) in stages.items():
        _verify_seal(folder, seal_name)
        evidence[key] = digest(folder / seal_name)

    if expected_evidence is not None:
        verify_episode_evidence(episode, expected_evidence)
    # The source campaign still pins its original recovery. Verify it above,
    # then bind the new publication explicitly to the requested additive one.
    admission_dir, recovery_seals = effective_admission_directory(episode, recovery_name)
    geometry_dir = effective_object_geometry_directory(episode, recovery_name)
    evidence.update(recovery_seals)

    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    admission = load(admission_dir / "frames.json")
    gt_frames = load(episode / "gt_timeline_v1/frames.json")
    geometry = load(geometry_dir / "frames.json")
    signals = load(episode / "route_signal_audit_v1/frames.json")
    route_report = load(episode / "route_signal_audit_v1/report.json")
    active = load(episode / "quality/active_signals.json")
    native = load(episode / "native_audit/report.json")
    contract_path = run / "sensor_contract.json"
    contract = load(contract_path)
    if not (len(admission) == len(gt_frames) == len(geometry) == len(signals)):
        raise ValueError("Sensor Record target timelines have different lengths")
    times, ego_values = dense_ego_timeline(active)
    route_tracker, navigation_contract = official_route_tracker(contract, route_report, episode)
    # An already sealed portable publication can pin a promoted native root;
    # the original audit deliberately retains its capture-time staging path.
    source_root = Path(published_native_root or native["source_root"]).resolve()
    if not source_root.is_dir():
        raise ValueError("Native Sensor Record source root is unavailable")

    entries, counts = [], Counter(raw_frames=len(admission))
    for index, (admit, gt, geom, signal) in enumerate(zip(admission, gt_frames, geometry, signals)):
        if any(int(row.get("index", -1)) != index for row in (admit, gt, geom, signal)):
            raise ValueError("Sensor Record frame index mismatch")
        if not admit.get("dataset_build_candidate"):
            counts["excluded_frames"] += 1
            continue
        stamp = int(admit["lidar_source_time_ns"])
        if stamp != int(gt["lidar_source_ns"]):
            raise ValueError("Admission and GT LiDAR source time differ")
        ego, waypoints = waypoint_at_sensor_time(times, ego_values, stamp)
        route_progress = route_tracker.update(ego["x"], ego["y"], ego["yaw"])
        audited_route_index = signal.get("route_index")
        if (audited_route_index is not None
                and int(audited_route_index) != int(route_progress.route_index)):
            raise ValueError("Rebuilt official route progress differs from sealed audit")
        navigation = {
            "route_target_valid": True,
            "route_index": int(route_progress.route_index),
            "target_x": float(route_progress.target_x),
            "target_y": float(route_progress.target_y),
            "target_route_index": int(route_progress.target_route_index),
            "target_source": "official_route_lookahead",
            "distance_to_route_m": float(route_progress.distance_to_route_m),
        }
        objects = objects_for_entry(gt, geom)
        object_valid = bool(
            admit.get("object_gt_transport_timing_candidate")
            and admit.get("full_actor_inventory_transport_candidate")
            and admit.get("object_model_identity_candidate")
            and admit.get("object_api_box_geometry_candidate")
            and geom.get("api_box_geometry_candidate")
        )
        temporary = {"object_valid": object_valid, "ego_pose": ego, "objects": objects}
        detection, rendered_valid, reason = detection_for_entry(temporary)
        if object_valid and not rendered_valid:
            raise ValueError("Object audit and renderer disagree: " + str(reason))
        object_positive = bool(object_valid and np.any(detection[:, 0] > 0))
        light = int(signal.get("training_label", -1))
        if not (admit.get("traffic_light_gt_transport_timing_candidate")
                and admit.get("route_signal_binding_candidate")
                and signal.get("valid") is True and light in (0, 1)):
            light = -1

        roles, assets, sizes = {}, {}, {}
        if set(admit.get("assets", {})) != set(REQUIRED_ROLES):
            raise ValueError("Training candidate lacks a required native asset")
        for role in REQUIRED_ROLES:
            meta = admit["assets"][role]
            relative = str(meta["relative_path"])
            # native_audit/COMPLETE.json already seals the full byte/size audit.
            # Re-statting 7 files for every frame across WSL/9p takes minutes
            # and adds no independent evidence here.  The loader checks size
            # and SHA256 on the first actual read of each selected asset.
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.parts:
                raise ValueError("Unsafe Sensor Record asset path: " + relative)
            roles[role] = relative
            assets[relative] = str(meta["sha256"])
            sizes[relative] = int(meta["size_bytes"])
        entries.append({
            "frame_id": index,
            "sensor_time_ns": stamp,
            "native_filename_time_ns": int(admit["native_filename_time_ns"]),
            "waypoints": waypoints.tolist(),
            "ego_pose": ego,
            "navigation": navigation,
            "objects": objects,
            "object_valid": object_valid,
            "object_positive": object_positive,
            "traffic_light": light,
            "asset_roles": roles,
            "assets": assets,
            "asset_sizes": sizes,
        })
        counts["waypoint_frames"] += 1
        counts["navigation_target_frames"] += 1
        counts["object_frames"] += int(object_valid)
        counts["object_positive_frames"] += int(object_positive)
        counts["traffic_light_" + str(light)] += 1
    if not entries:
        raise ValueError("No admitted Sensor Record training frames")
    return {
        "schema": SCHEMA,
        "source_clock": "sensor_record_simulation_time",
        "route": str(episode),
        "episode": str(episode),
        "native_root": str(source_root),
        "split": split,
        "navigation_contract": navigation_contract,
        "contract": contract,
        "contract_sha256": digest(contract_path),
        "evidence_seal_sha256": evidence,
        "admission_recovery_path": recovery_name,
        "campaign_validation_sha256": campaign_validation_sha256,
        "object_exclusion_policy": None,
        "excluded_light_ids": [],
        "report": {
            "counts": dict(counts),
            "all_tasks_have_training_examples": all(counts[k] > 0 for k in (
                "waypoint_frames", "object_positive_frames", "traffic_light_0", "traffic_light_1")),
            "teacher_motion": {
                "status": "consistent_motion",
                "method": "interpolated_full_rate_ego_source_timeline_v1",
                "clock": "sensor_record_simulation_time",
            },
            "task_mask_policy": "waypoint=admission; object=transport+inventory+identity+geometry; light=transport+route-binding+pure-color",
            "physical_simultaneity_verified": False,
            "official_camera_exposure_timestamp_available": False,
            "validation_scope": "Sensor Record internal synchronization and empirical audits",
        },
        "entries": entries,
        "training_visible": True,
    }


def read_native_lidar(path):
    raw = np.fromfile(path, dtype="<f4")
    if not len(raw) or len(raw) % 4:
        raise ValueError("Invalid native XYZI float32 payload: " + str(path))
    points = raw.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite native LiDAR payload: " + str(path))
    return points
