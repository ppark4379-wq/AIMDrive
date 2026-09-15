"""Fail-closed admission for internally synchronized MORAI Sensor Record runs.

This module deliberately keeps four decisions separate:

* whether the seven native sensor files form an admitted scene bundle;
* whether the recorded Ego stream can be associated with that scene;
* whether enough later Ego samples exist for the 2.5 second waypoint target;
* whether object/light messages are merely transport-timing candidates.

The result is an input to the offline dataset builder.  It never publishes a
training manifest and never upgrades unverified object geometry or signal
route binding into labels.
"""
from collections import Counter
import json
from pathlib import Path

from .internal_sensor_record_contract import CAPTURE_MODE, validate_preflight_snapshot
from .journal import digest


REQUIRED_ASSET_ROLES = (
    "camera_1", "camera_2", "camera_3", "camera_4",
    "lidar", "gps", "imu",
)
SENSOR_SCENE_STATE_CONTRACT = (
    "pinned internal Sensor Record producer; one unchanged simulation state per filename bundle"
)


def validate_runtime_mode(mode):
    checks = {
        "time_management": mode.get("observed_type") == CAPTURE_MODE["time_mode"],
        "external_tick_disabled": mode.get("external_tick_period") == 0,
        "collector_does_not_own_tick": mode.get("native_tick_owner") is False,
        "requested_simulation_free_run": mode.get("requested_time_mode") == "simulation_free_run",
    }
    return {"valid": all(checks.values()), "checks": checks}


def _is_within(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    return path == root or root in path.parents


def _safe_local_file(path, root):
    """Return True only for a regular, non-symlink file contained by root."""
    path = Path(path)
    root = Path(root)
    if not path.is_file() or path.is_symlink() or root.is_symlink():
        return False
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return False
    return _is_within(path, root)


def _verify_pinned_file(item, *, local_copy=None, local_root=None):
    """Resolve one immutable artifact without weakening its recorded digest."""
    recorded_text = item.get("path", "") if isinstance(item, dict) else ""
    expected = item.get("sha256") if isinstance(item, dict) else None
    recorded = Path(recorded_text) if recorded_text else None
    if (recorded is not None and expected and recorded.is_file()
            and not recorded.is_symlink() and digest(recorded) == expected):
        return {"valid": True, "path": str(recorded), "resolution": "recorded_path"}
    local_copy = Path(local_copy) if local_copy is not None else None
    if not (
        recorded is not None
        and expected
        and local_copy is not None
        and local_root is not None
        and recorded.name == local_copy.name
        and _safe_local_file(local_copy, local_root)
        and digest(local_copy) == expected
    ):
        return {"valid": False, "path": None, "resolution": "unresolved"}
    return {"valid": True, "path": str(local_copy.resolve()),
            "resolution": "run_local_exact_sha256_copy"}


def validate_native_session_snapshot(snapshot, *, transferred_run=None):
    transferred_run = (Path(transferred_run).resolve()
                       if transferred_run is not None else None)
    portable_identity = bool(
        transferred_run is not None
        and snapshot.get("session_name") == transferred_run.name
    )
    checks = {
        "prepared_before_capture": snapshot.get("prepared_before_capture") is True,
        "session_name": bool(snapshot.get("session_name")),
        "native_root": bool(snapshot.get("native_root")),
        "portable_run_identity": portable_identity if transferred_run is not None else True,
    }
    resolutions = {}
    for key in ("session_plan", "session_apply", "session_record"):
        item = snapshot.get(key, {})
        recorded_name = Path(item.get("path", "")).name
        local_copy = (transferred_run / recorded_name
                      if portable_identity and recorded_name else None)
        result = _verify_pinned_file(
            item, local_copy=local_copy, local_root=transferred_run)
        checks[key] = result["valid"]
        resolutions[key] = result
    return {"valid": all(checks.values()), "checks": checks,
            "resolutions": resolutions}


def _verify_producer_snapshot(snapshot, producer, *, transferred_run=None):
    if not snapshot or not producer or snapshot.get("files", {}) != producer:
        return {"valid": False, "root": None, "resolution": "unresolved"}
    roots = [(Path(snapshot.get("root", "")), "recorded_path")]
    if transferred_run is not None:
        roots.append((Path(transferred_run) / "software_snapshot",
                      "run_local_exact_sha256_copy"))
    for root, resolution in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        valid = True
        for name, expected in producer.items():
            relative = Path(name)
            path = root / relative
            if (relative.is_absolute() or ".." in relative.parts
                    or not _safe_local_file(path, root)
                    or digest(path) != expected):
                valid = False
                break
        if valid:
            return {"valid": True, "root": str(root.resolve()),
                    "resolution": resolution}
    return {"valid": False, "root": None, "resolution": "unresolved"}


def build_capture_contract(preflight, plan, *, plan_sha256, producer_software,
                           native_session, acquisition_contract,
                           producer_software_snapshot=None):
    """Build the record written immediately before Sensor Record is enabled."""
    runtime = validate_runtime_mode(plan.get("mode", {}))
    session = validate_native_session_snapshot(native_session)
    failures = []
    if not preflight.get("ready"):
        failures.append("immutable_preflight_failed")
    if not runtime["valid"]:
        failures.append("runtime_mode_mismatch")
    if not session["valid"]:
        failures.append("native_session_not_pinned")
    acquisition_path=Path(acquisition_contract.get('path',''))
    acquisition_valid=bool(acquisition_path.is_file() and
        acquisition_contract.get('sha256')==digest(acquisition_path))
    if not acquisition_valid:failures.append('acquisition_contract_not_pinned')
    transport_policy = plan.get('arguments', {}).get(
        'raw_camera_transport', 'required'
    )
    if transport_policy not in ('required', 'unrecorded', 'bridge_dropped', 'absent'):
        failures.append('invalid_ros_camera_transport_policy')
    if acquisition_valid:
        acquisition = json.loads(acquisition_path.read_text())
        recorded_policy = acquisition.get('capture', {}).get(
            'ros_camera_transport_policy', 'required'
        )
        recorded_cameras = acquisition.get('capture', {}).get(
            'record_ros_cameras', True
        )
        if not (
            recorded_policy == transport_policy
            and bool(recorded_cameras) == (transport_policy == 'required')
        ):
            failures.append('ros_camera_transport_contract_mismatch')
    return {
        "schema": "morai_internal_sensor_record_run_contract_v2",
        "ready": not failures,
        "failures": failures,
        "written_before_sensor_record": True,
        "static_preflight": preflight,
        "runtime_mode": plan.get("mode", {}),
        "runtime_mode_validation": runtime,
        "native_session": native_session,
        "native_session_validation": session,
        "acquisition_contract": acquisition_contract,
        "acquisition_contract_valid": acquisition_valid,
        "ros_camera_transport_policy": transport_policy,
        "run_plan_sha256": plan_sha256,
        "producer_software": producer_software,
        "producer_software_snapshot": producer_software_snapshot,
        "expected_native_assets": list(REQUIRED_ASSET_ROLES),
        "expected_capture_cadence_hz": CAPTURE_MODE["internal_sensor_synchronization_hz"],
        "sensor_scene_state_contract": SENSOR_SCENE_STATE_CONTRACT,
        "physical_exposure_millisecond_bound_verified": False,
        "source_timestamps_modified": False,
        "training_visible": False,
    }


def verify_capture_contract(contract, plan, plan_path):
    transferred_run = Path(plan_path).resolve().parent
    native_snapshot = contract.get("native_session", {})
    portable_identity = native_snapshot.get("session_name") == transferred_run.name
    runtime = validate_runtime_mode(plan.get("mode", {}))
    static = validate_preflight_snapshot(contract.get("static_preflight", {}))
    session = validate_native_session_snapshot(
        native_snapshot, transferred_run=transferred_run)
    producer=contract.get("producer_software",{})
    snapshot=contract.get('producer_software_snapshot')
    producer_resolution = None
    if snapshot:
        producer_resolution = _verify_producer_snapshot(
            snapshot, producer,
            transferred_run=transferred_run if portable_identity else None)
        producer_valid = producer_resolution["valid"]
    else:
        # Legacy contracts did not archive their source. Keep their original
        # fail-closed current-worktree check instead of silently upgrading them.
        source_root=Path(__file__).resolve().parents[1]
        for name,expected in producer.items():
            path=source_root/name
            producer_valid &= bool(not Path(name).is_absolute() and '..' not in Path(name).parts
                                   and path.is_file() and digest(path)==expected)
    acquisition=contract.get('acquisition_contract',{})
    acquisition_name = Path(acquisition.get('path', '')).name
    acquisition_resolution = _verify_pinned_file(
        acquisition,
        local_copy=(transferred_run / acquisition_name
                    if portable_identity and acquisition_name else None),
        local_root=transferred_run,
    )
    acquisition_valid = acquisition_resolution["valid"]
    acquisition_path = (Path(acquisition_resolution["path"])
                        if acquisition_valid else None)
    expected_transport = plan.get('arguments', {}).get(
        'raw_camera_transport', 'required'
    )
    contract_transport = contract.get(
        'ros_camera_transport_policy', 'required'
    )
    acquisition_transport = None
    acquisition_records_cameras = None
    if acquisition_valid:
        acquisition_data = json.loads(acquisition_path.read_text())
        acquisition_transport = acquisition_data.get('capture', {}).get(
            'ros_camera_transport_policy', 'required'
        )
        acquisition_records_cameras = acquisition_data.get('capture', {}).get(
            'record_ros_cameras', True
        )
    checks = {
        "schema": contract.get("schema") == "morai_internal_sensor_record_run_contract_v2",
        "ready_before_capture": contract.get("ready") is True,
        "written_before_sensor_record": contract.get("written_before_sensor_record") is True,
        "static_preflight": static["valid"],
        "runtime_mode": runtime["valid"] and contract.get("runtime_mode") == plan.get("mode"),
        "run_plan_hash": contract.get("run_plan_sha256") == digest(plan_path),
        "native_session": session["valid"],
        "producer_software": producer_valid,
        "acquisition_contract": acquisition_valid,
        "ros_camera_transport": bool(
            expected_transport in ('required', 'unrecorded', 'bridge_dropped', 'absent')
            and contract_transport == expected_transport
            and acquisition_transport == expected_transport
            and bool(acquisition_records_cameras)
                == (expected_transport == 'required')
        ),
        "asset_contract": tuple(contract.get("expected_native_assets", ())) == REQUIRED_ASSET_ROLES,
        "cadence_contract": contract.get("expected_capture_cadence_hz") == 10,
        "timestamps_preserved": contract.get("source_timestamps_modified") is False,
        "scene_state_contract": contract.get("sensor_scene_state_contract") == SENSOR_SCENE_STATE_CONTRACT,
        "no_false_millisecond_claim": contract.get("physical_exposure_millisecond_bound_verified") is False,
    }
    return {"valid": all(checks.values()), "checks": checks,
            "static_preflight_checks": static["checks"],
            "native_session_checks": session["checks"],
            "artifact_resolution": {
                "native_session": session["resolutions"],
                "producer_software_snapshot": producer_resolution,
                "acquisition_contract": acquisition_resolution,
            }}


def verify_sealed_directory(folder, seal_name="COMPLETE.json"):
    folder = Path(folder)
    seal_path = folder / seal_name
    seal = json.loads(seal_path.read_text())
    for name, expected in seal.get("files", {}).items():
        if Path(name).name != name or digest(folder / name) != expected:
            raise ValueError("Changed or unsafe sealed evidence: " + str(folder / name))
    return seal


def summarize_live_streams(active_signals, native_start_ns, native_end_ns, *,
                           max_ego_gap_ns=50_000_000,
                           sync_frame_period_ns=20_000_000,
                           max_sync_clock_span_error_ns=50_000_000):
    """Validate live streams without equating independent clock epochs.

    ``SyncModeInfo.frame`` is a MORAI process-lifetime 20 ms frame counter in
    the pinned 26.R1.H3 runtime.  This is distinct from the 10 Hz internal
    Sensor Record capture cadence.  It can keep
    increasing across a scenario reset while ``/clock`` starts from the new
    scenario time.  The recorder stores the last causally preceding /clock for
    every message, so use that clock to bracket Sensor Record frames and use
    only the *change* in SyncModeInfo.frame to check its rate.  Multiplying the
    absolute frame by 10 ms and treating it as /clock time is invalid.
    """
    sync_frames = []
    sync_clock_times = []
    sync_clock_offsets = []
    sync_inactive = 0
    ego_times = []
    last_ego_time = None
    ego_reversals = 0
    for row in active_signals:
        role = row.get("role")
        if role == "sync_mode":
            signal = row.get("signal") or {}
            if signal.get("status") is not True:
                sync_inactive += 1
            if signal.get("frame") is not None:
                frame = int(signal["frame"])
                sync_frames.append(frame)
                preceding_clock = row.get("preceding_received_clock_ns")
                if preceding_clock is not None:
                    preceding_clock = int(preceding_clock)
                    sync_clock_times.append(preceding_clock)
                    sync_clock_offsets.append(
                        preceding_clock - frame * int(sync_frame_period_ns))
        elif role == "ego" and row.get("source_timestamp_ns") is not None:
            stamp = int(row["source_timestamp_ns"])
            if last_ego_time is not None and stamp < last_ego_time:
                ego_reversals += 1
            if last_ego_time is None or stamp > last_ego_time:
                ego_times.append(stamp)
                last_ego_time = stamp
    sync_reversals = sum(b < a for a, b in zip(sync_frames, sync_frames[1:]))
    sync_clock_reversals = sum(
        b < a for a, b in zip(sync_clock_times, sync_clock_times[1:]))
    ego_gaps = [b - a for a, b in zip(ego_times, ego_times[1:])]
    sync_clock_start_ns = min(sync_clock_times) if sync_clock_times else None
    sync_clock_end_ns = max(sync_clock_times) if sync_clock_times else None
    offset_min_ns = min(sync_clock_offsets) if sync_clock_offsets else None
    offset_max_ns = max(sync_clock_offsets) if sync_clock_offsets else None
    offset_span_ns = (offset_max_ns - offset_min_ns
                      if sync_clock_offsets else None)
    sync_frame_span_ns = (
        (max(sync_frames) - min(sync_frames)) * int(sync_frame_period_ns)
        if sync_frames else None)
    sync_clock_span_ns = (
        sync_clock_end_ns - sync_clock_start_ns
        if sync_clock_times else None)
    sync_clock_span_error_ns = (
        abs(sync_frame_span_ns - sync_clock_span_ns)
        if sync_frame_span_ns is not None and sync_clock_span_ns is not None
        else None)
    # SyncModeInfo and /clock arrive through independent ROS callbacks.  The
    # causally preceding clock can therefore stay unchanged for one callback
    # and catch up on the next, making the per-message offset move by more than
    # one period even when both sources advance at exactly the same rate.  Use
    # their end-to-end deltas for the rate contract.  Keep offset_span_ns only
    # as a receive-order diagnostic; it must not reject a capture.
    sync_rate_consistent = bool(
        sync_frames and sync_clock_times
        and sync_clock_span_error_ns <= int(max_sync_clock_span_error_ns))
    sync_brackets_native = bool(
        sync_clock_times
        and sync_clock_start_ns <= native_start_ns
        and sync_clock_end_ns >= native_end_ns)
    return {
        "sync_samples": len(sync_frames),
        "sync_unique_frames": len(set(sync_frames)),
        "sync_inactive_samples": sync_inactive,
        "sync_frame_reversals": sync_reversals,
        "sync_frame_range": [min(sync_frames), max(sync_frames)] if sync_frames else [None, None],
        "sync_frame_period_ns": int(sync_frame_period_ns),
        "sync_frame_absolute_epoch_assumed_equal_to_clock": False,
        "sync_causal_clock_samples": len(sync_clock_times),
        "sync_causal_clock_reversals": sync_clock_reversals,
        "sync_causal_clock_range_ns": [sync_clock_start_ns, sync_clock_end_ns],
        "sync_frame_clock_offset_range_ns": [offset_min_ns, offset_max_ns],
        "sync_frame_clock_offset_span_ns": offset_span_ns,
        "sync_frame_clock_offset_is_receive_order_diagnostic": True,
        "sync_frame_span_ns": sync_frame_span_ns,
        "sync_causal_clock_span_ns": sync_clock_span_ns,
        "sync_frame_clock_span_error_ns": sync_clock_span_error_ns,
        "sync_frame_clock_span_error_limit_ns": int(max_sync_clock_span_error_ns),
        "sync_frame_rate_consistent_with_causal_clock": sync_rate_consistent,
        "sync_brackets_native_capture": sync_brackets_native,
        "ego_unique_source_samples": len(ego_times),
        "ego_source_reversals": ego_reversals,
        "ego_source_range_ns": [ego_times[0], ego_times[-1]] if ego_times else [None, None],
        "ego_max_source_gap_ns": max(ego_gaps, default=None),
        "ego_brackets_native_capture": bool(ego_times and ego_times[0] <= native_start_ns
                                             and ego_times[-1] >= native_end_ns),
        "cross_stream_absolute_clock_equality_assumed": False,
        "valid": bool(sync_frames and sync_clock_times and ego_times
                      and not sync_inactive and not sync_reversals
                      and not sync_clock_reversals and sync_rate_consistent
                      and not ego_reversals and max(ego_gaps, default=max_ego_gap_ns) <= max_ego_gap_ns
                      and sync_brackets_native
                      and ego_times[0] <= native_start_ns and ego_times[-1] >= native_end_ns),
    }


def build_frame_admission(*, contract_valid, native_report, review_frames, gt_frames,
                          motion_report, live_streams, runtime_streams_valid=True,
                          future_horizon_ns=2_500_000_000):
    """Return conservative frame records and aggregate counters."""
    if len(native_report["bundles"]) != len(review_frames) or len(review_frames) != len(gt_frames):
        raise ValueError("Native, review, and GT frame counts differ")
    asset_by_path = {row["relative_path"]: row for row in native_report["assets"]}
    exact_lidar = {
        row["relative_path"]: row for row in native_report["comparison"]
        if row.get("role") == "lidar" and row.get("exact_xyzi_array_equal") is True
    }
    healthy_teacher_stream = (
        live_streams["valid"]
        and motion_report.get("full_ego_stream", {}).get("source_gap_count") == 0
        and motion_report.get("full_ego_stream", {}).get("moving_position_holds") == 0
        and (motion_report.get("full_ego_stream", {}).get("relative_difference") is not None)
        and motion_report["full_ego_stream"]["relative_difference"] <= 0.05
    )
    ego_end_ns = live_streams["ego_source_range_ns"][1]
    frames = []
    for index, (bundle, review, gt) in enumerate(zip(native_report["bundles"], review_frames, gt_frames)):
        if review.get("index") != index or gt.get("index") != index:
            raise ValueError("Frame index sequence differs")
        paths = bundle.get("assets", {})
        complete_roles = not bundle.get("missing") and set(paths) == set(REQUIRED_ASSET_ROLES)
        assets = {}
        asset_hashes_present = True
        for role in REQUIRED_ASSET_ROLES:
            meta = asset_by_path.get(paths.get(role))
            if meta is None:
                asset_hashes_present = False
                continue
            assets[role] = {
                "relative_path": meta["relative_path"],
                "size_bytes": meta["size_bytes"],
                "sha256": meta["sha256"],
            }
        lidar = exact_lidar.get(paths.get("lidar"))
        source_ns = int(gt["lidar_source_ns"])
        observed_sensor_bundle_integrity_valid = bool(
            native_report.get("files_valid") is True
            and native_report.get("complete_cadence") is True
            and not native_report.get("errors") and complete_roles and asset_hashes_present
            and lidar is not None and review.get("complete_native_inputs") is True
            and review.get("exact_ros_native_lidar") is True
        )
        capture_environment_valid = bool(contract_valid and runtime_streams_valid)
        sensor_bundle_valid = bool(capture_environment_valid
                                   and observed_sensor_bundle_integrity_valid)
        observed_ego_teacher_time_valid = bool(
            observed_sensor_bundle_integrity_valid and healthy_teacher_stream
            and review.get("same_source_gps_teacher_associated") is True
            and review.get("adjacent_teacher_intervals_consistent") is True
        )
        ego_teacher_time_valid = bool(contract_valid and observed_ego_teacher_time_valid)
        observed_future_waypoint_source_valid = bool(
            observed_ego_teacher_time_valid and ego_end_ns is not None
            and source_ns + future_horizon_ns <= ego_end_ns
        )
        future_waypoint_source_valid = bool(contract_valid and observed_future_waypoint_source_valid)
        observed_object_candidate = bool(
            observed_sensor_bundle_integrity_valid
            and gt["object_gt"].get("transport_timing_candidate") is True
        )
        observed_light_candidate = bool(
            observed_sensor_bundle_integrity_valid
            and gt["traffic_light_gt"].get("transport_timing_candidate") is True
        )
        object_candidate = bool(contract_valid and observed_object_candidate)
        light_candidate = bool(contract_valid and observed_light_candidate)
        reasons = []
        if not contract_valid:
            reasons.append("capture_environment_not_pinned_before_recording")
        elif not runtime_streams_valid:
            reasons.append("runtime_stream_contract_failed")
        if not sensor_bundle_valid and contract_valid:
            reasons.append("sensor_bundle_failed")
        if sensor_bundle_valid and not ego_teacher_time_valid:
            reasons.append("ego_teacher_time_failed")
        if ego_teacher_time_valid and not future_waypoint_source_valid:
            reasons.append("future_2p5s_ego_not_available")
        frames.append({
            "index": index,
            "native_filename_time_ns": int(bundle["filename_simulation_time_hypothesis_ns"]),
            "lidar_source_time_ns": source_ns,
            "assets": assets,
            "observed_sensor_bundle_integrity_valid": observed_sensor_bundle_integrity_valid,
            "sensor_bundle_valid": sensor_bundle_valid,
            "camera_lidar_same_simulation_state_valid": sensor_bundle_valid,
            "observed_ego_teacher_time_valid": observed_ego_teacher_time_valid,
            "ego_teacher_time_valid": ego_teacher_time_valid,
            "observed_future_waypoint_2p5s_source_valid": observed_future_waypoint_source_valid,
            "future_waypoint_2p5s_source_valid": future_waypoint_source_valid,
            "dataset_build_candidate": future_waypoint_source_valid,
            "observed_object_gt_transport_timing_candidate": observed_object_candidate,
            "object_gt_transport_timing_candidate": object_candidate,
            "observed_traffic_light_gt_transport_timing_candidate": observed_light_candidate,
            "traffic_light_gt_transport_timing_candidate": light_candidate,
            "object_training_label_valid": False,
            "traffic_light_training_label_valid": False,
            "reasons": reasons,
            "source_timestamps_modified": False,
            "physical_exposure_millisecond_bound_verified": False,
            "training_visible": False,
        })
    fields = (
        "observed_sensor_bundle_integrity_valid", "observed_ego_teacher_time_valid",
        "observed_future_waypoint_2p5s_source_valid",
        "observed_object_gt_transport_timing_candidate",
        "observed_traffic_light_gt_transport_timing_candidate",
        "sensor_bundle_valid", "ego_teacher_time_valid", "future_waypoint_2p5s_source_valid",
        "dataset_build_candidate", "object_gt_transport_timing_candidate",
        "traffic_light_gt_transport_timing_candidate",
    )
    counts = {field: sum(bool(row[field]) for row in frames) for field in fields}
    counts["excluded_from_dataset_build"] = len(frames) - counts["dataset_build_candidate"]
    return frames, counts


def reason_counts(frames):
    return dict(Counter(reason for row in frames for reason in row.get("reasons", ())))
