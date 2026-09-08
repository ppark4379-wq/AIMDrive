"""Passive MORAI native-sync file consumer wired to the existing writers."""

import hashlib
import json
import math
import os
import time

import numpy as np

from mgeo_navigation import (
    MGeoMap, MGeoRouteTracker, load_official_route, stable_mgeo_id,
)
from schema import ActorData, EgoState, build_measurement_dict
from writer.async_frame_writer import AsyncFrameWriter
from writer.dataset_writer import DatasetWriter
from writer.image_writer import ImageWriter
from writer.lidar_writer import LidarWriter

from .morai_sync_file_source import (
    IncompleteNativeFrame,
    MoraiSyncFileSource,
    discover_split_sync_session_roots,
    discover_sync_session_root,
)
from .sync_capture_manifest import SyncCaptureManifest


CAMERA_VIEW_NAMES = ("front", "left", "right", "rear")
_OBJ_TYPE_TO_TPE = {"vehicle": 0, "pedestrian": 1}


def _actor_data(obj):
    heading = math.radians(float(obj.heading_deg))
    return ActorData(
        loc=[obj.x, obj.y, obj.z],
        ori=[math.cos(heading), math.sin(heading)],
        box=[obj.length / 2.0, obj.width / 2.0],
        vel=[obj.vel_x, obj.vel_y, 0.0],
        tpe=_OBJ_TYPE_TO_TPE[obj.obj_type],
    )


def _morai_light_to_lmdrive_state(status):
    status = int(status)
    if status & 1:
        return 0
    if status & 4:
        return 1
    if status & (16 | 32):
        return 2
    if status == 0:
        return 3
    return 4


def _angle_to_target_deg(x, y, theta, target_x, target_y):
    dx, dy = target_x - x, target_y - y
    local_x = math.cos(theta) * dx + math.sin(theta) * dy
    local_y = -math.sin(theta) * dx + math.cos(theta) * dy
    return math.degrees(math.atan2(local_y, local_x))


def _load_navigation(args):
    link_path = os.path.join(args.map_dir, "link_set.json") if args.map_dir else ""
    node_path = os.path.join(args.map_dir, "node_set.json") if args.map_dir else ""
    mgeo_map = MGeoMap(
        link_path,
        node_set_path=node_path if os.path.isfile(node_path) else None,
    ) if os.path.isfile(link_path) else None
    route_metadata = None
    if bool(args.route_definition or args.global_path):
        if not (args.route_definition and args.global_path):
            raise RuntimeError("--route-definition and --global-path must be specified together")
        if mgeo_map is None:
            raise RuntimeError("Official route collection requires link_set.json")
        route_metadata = load_official_route(
            args.route_definition, args.global_path, args.route_definition_key
        )
        missing = [x for x in route_metadata["nav_link_ids"] if x not in mgeo_map.links]
        if missing:
            raise RuntimeError("Route links missing from MGeo: {}".format(missing))
        route_metadata["nav_node_ids"] = [
            str(mgeo_map.links[route_metadata["nav_link_ids"][0]].get("from_node_idx"))
        ] + [
            str(mgeo_map.links[x].get("to_node_idx"))
            for x in route_metadata["nav_link_ids"]
        ]
    elif args.route_spec:
        if mgeo_map is None:
            raise RuntimeError("--route-spec requires link_set.json")
        with open(args.route_spec, encoding="utf-8") as source:
            route_metadata = json.load(source)
        if not (route_metadata.get("nav_link_ids") and route_metadata.get("nav_waypoints")):
            route_metadata.update(mgeo_map.build_checkpoint_route(
                route_metadata["checkpoints"], bool(route_metadata.get("close_loop"))
            ))
    tracker = None
    if route_metadata is not None:
        tracker = MGeoRouteTracker(
            mgeo_map, route_metadata, initial_route_index=args.route_start_index
        )
        route_metadata["nav_road_options"] = list(tracker.point_road_options)
    return mgeo_map, route_metadata, tracker


def _append_jsonl(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()


def _manifest_session_id(path):
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as source:
        for line in source:
            if line.strip():
                return str(json.loads(line).get("session_id", ""))
    return ""


def _checkpoint_path(dataset_root, job_name, manifest_path):
    manifest_path = os.path.abspath(manifest_path)
    session_name = os.path.basename(os.path.dirname(manifest_path)) or "session"
    safe_name = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in session_name
    )
    digest = hashlib.sha256(manifest_path.encode("utf-8")).hexdigest()[:12]
    return os.path.join(
        os.path.abspath(dataset_root), job_name, ".collector_state",
        "{}-{}.json".format(safe_name, digest),
    )


def _read_last_provenance(path):
    last = None
    try:
        with open(path, encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    last = json.loads(line)
    except (OSError, ValueError):
        return None
    return last


def _recover_resume_state(dataset_root, job_name, route_name, manifest_path):
    """Load a checkpoint, or migrate the newest matching provenance journal."""
    manifest_path = os.path.abspath(manifest_path)
    runtime_session_id = _manifest_session_id(manifest_path)
    checkpoint_path = _checkpoint_path(
        dataset_root, job_name, manifest_path
    )
    if os.path.isfile(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as source:
            state = json.load(source)
        if os.path.abspath(state.get("manifest_path", "")) != manifest_path:
            raise RuntimeError("Collector checkpoint manifest path mismatch")
        if not os.path.isdir(state.get("output_dir", "")):
            raise RuntimeError("Collector checkpoint output directory is missing")
        return state, checkpoint_path

    data_dir = os.path.join(os.path.abspath(dataset_root), job_name, "data")
    candidates = []
    if os.path.isdir(data_dir):
        for entry in os.scandir(data_dir):
            if not entry.is_dir() or not entry.name.startswith(route_name):
                continue
            provenance_path = os.path.join(entry.path, "capture_provenance.jsonl")
            last = _read_last_provenance(provenance_path)
            if last is None:
                continue
            if runtime_session_id and str(last.get("session_id", "")) != runtime_session_id:
                continue
            candidates.append((os.path.getmtime(provenance_path), entry.path, last))
    if not candidates:
        return None, checkpoint_path

    _, output_dir, last = max(candidates, key=lambda item: item[0])
    state = {
        "schema_version": 1,
        "manifest_path": manifest_path,
        "runtime_session_id": str(last.get("session_id", runtime_session_id)),
        "output_dir": os.path.abspath(output_dir),
        "last_consumed_native_frame": int(last["native_sync_frame"]),
        "next_sample_index": int(last["sample_index"]) + 1,
    }
    return state, checkpoint_path


def _write_checkpoint(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _measurement_for_record(record, route_tracker, args, label_snapshot=None):
    ego_raw = record["ego_state_for_frame"]
    position = ego_raw["position"]
    velocity = ego_raw["velocity"]
    heading = math.radians(float(ego_raw["heading"]))
    speed = math.sqrt(sum(float(velocity[axis]) ** 2 for axis in ("x", "y", "z")))
    progress = route_tracker.update(
        float(position["x"]), float(position["y"]), heading
    ) if route_tracker else None
    if (
        args.require_route_valid
        and progress is not None
        and progress.distance_to_route_m > args.route_valid_max_error
    ):
        return None, None, None

    control = record["applied_control"]
    steer = float(np.clip(
        args.morai_to_lmdrive_steer_sign
        * float(control.get("front_steer", 0.0))
        / math.radians(args.control_max_steer_deg),
        -1.0,
        1.0,
    ))
    target_speed = (
        float(control.get("velocity", 0.0)) / 3.6
        if int(control.get("longlCmdType", 1)) == 2 else 0.0
    )
    ego = EgoState(
        x=float(position["x"]),
        y=float(position["y"]),
        theta=heading,
        speed=speed,
        command=progress.command if progress else 4,
        gt_command=progress.gt_command if progress else 4,
        x_command=progress.target_x if progress else float(position["x"]),
        y_command=progress.target_y if progress else float(position["y"]),
        near_node_x=progress.near_x if progress else float(position["x"]),
        near_node_y=progress.near_y if progress else float(position["y"]),
        far_node_x=progress.target_x if progress else float(position["x"]),
        far_node_y=progress.target_y if progress else float(position["y"]),
        gps_x=float(position["x"]),
        gps_y=float(position["y"]),
        steer=steer,
        throttle=float(np.clip(control.get("accel", 0.0), 0.0, 1.0)),
        brake=float(np.clip(control.get("brake", 0.0), 0.0, 1.0)),
        target_speed=target_speed,
    )
    objects = list(label_snapshot.objects) if label_snapshot else []
    actors = {
        str(obj.obj_id): _actor_data(obj) for obj in objects
        if obj.obj_type in _OBJ_TYPE_TO_TPE
    }
    observed_lights = {
        light.tl_id: light.state
        for light in (label_snapshot.traffic_lights if label_snapshot else [])
    }
    affected_light_raw = progress.affected_light_id if progress else None
    affected_light_id = stable_mgeo_id(affected_light_raw)
    affected_state_raw = (
        observed_lights.get(affected_light_raw) if affected_light_raw else None
    )
    affected_state = (
        _morai_light_to_lmdrive_state(affected_state_raw)
        if affected_state_raw is not None else (4 if affected_light_raw else None)
    )
    hazards = (
        route_tracker.hazard_actor_ids(
            objects, ego.x, ego.y, ego.theta,
            progress.route_index, progress.is_junction,
        )
        if route_tracker and progress else
        {"vehicle": [], "lane_vehicle": [], "junction_vehicle": [],
         "pedestrian": [], "bike": []}
    )
    should_slow = 0
    if progress:
        near_angle = _angle_to_target_deg(
            ego.x, ego.y, ego.theta, progress.near_x, progress.near_y
        )
        far_angle = _angle_to_target_deg(
            ego.x, ego.y, ego.theta, progress.target_x, progress.target_y
        )
        should_slow = int(abs(far_angle) > 45.0 or abs(near_angle) > 5.0)
    is_red = affected_state == 0
    should_brake = int(
        any(hazards[key] for key in hazards) or is_red
    )
    measurement = build_measurement_dict(
        ego,
        actors=actors,
        is_junction=progress.is_junction if progress else False,
        is_red_light_present=[affected_light_id] if is_red else [],
        affected_light_id=affected_light_id,
        future_waypoints=progress.future_waypoints if progress else None,
        is_vehicle_present=hazards["vehicle"],
        is_bike_present=hazards["bike"],
        is_lane_vehicle_present=hazards["lane_vehicle"],
        is_junction_vehicle_present=hazards["junction_vehicle"],
        is_pedestrian_present=hazards["pedestrian"],
        is_stop_sign_present=[],
        should_slow=should_slow,
        should_brake=should_brake,
    )
    navigation = {
        "current_link_id": progress.current_link_id if progress else None,
        "distance_to_link_m": progress.distance_to_route_m if progress else None,
        "route_index": progress.route_index if progress else None,
        "route_link_index": progress.route_link_index if progress else None,
        "route_valid": (
            progress.distance_to_route_m <= args.route_valid_max_error
            if progress else None
        ),
        "affected_light_id_raw": affected_light_raw,
        "affected_light_state_raw": affected_state_raw,
        "affected_light_state_lmdrive": affected_state,
    }
    labels = {
        "objects": objects,
        "traffic_lights": (
            list(label_snapshot.traffic_lights) if label_snapshot else []
        ),
        "affected_light_state": affected_state,
    }
    return measurement, navigation, labels


def run_sync_native_files(args, dataset_root, dataset_config, ros_config,
                          grpc_config=None, state_history=None):
    """Consume manifest + native files without importing/starting ROS sync code."""
    if args.morai_camera_sync_base or args.morai_lidar_sync_base:
        if not (args.morai_camera_sync_base and args.morai_lidar_sync_base):
            raise RuntimeError(
                "Specify both --morai-camera-sync-base and --morai-lidar-sync-base"
            )
        camera_session_root, lidar_session_root, morai_session_id = (
            discover_split_sync_session_roots(
                args.morai_camera_sync_base, args.morai_lidar_sync_base
            )
        )
    else:
        camera_session_root = discover_sync_session_root(
            args.morai_sync_root, args.morai_sync_base
        )
        lidar_session_root = camera_session_root
        morai_session_id = os.path.basename(camera_session_root)
    mgeo_map, route_metadata, route_tracker = _load_navigation(args)
    if mgeo_map is not None:
        print("[collect/native] MGeo links={}".format(len(mgeo_map.links)))
    if route_metadata is not None:
        print(
            "[collect/native] route={} links={} waypoints={}".format(
                route_metadata.get("route_id", args.route),
                len(route_metadata["nav_link_ids"]),
                len(route_metadata["nav_waypoints"]),
            )
        )

    resume_state, checkpoint_path = _recover_resume_state(
        dataset_root, args.job, args.route, args.capture_manifest
    )
    writer = DatasetWriter(
        dataset_root,
        args.job,
        args.route,
        map_id=args.map_id,
        map_source_dir=args.map_dir,
        append_timestamp=dataset_config.append_timestamp and not args.no_route_timestamp,
        route_metadata=route_metadata,
        extra_map_files={"global_info.json": args.global_info} if args.global_info else None,
        resume_route_dir=resume_state["output_dir"] if resume_state else "",
        drive_tag=getattr(dataset_config, "drive_tag", ""),
        run_note=getattr(dataset_config, "run_note", ""),
    )
    async_writer = AsyncFrameWriter(
        writer,
        ImageWriter(
            writer.output_dir,
            view_names=CAMERA_VIEW_NAMES,
            ext=dataset_config.image_ext,
            quality=dataset_config.image_quality,
        ),
        LidarWriter(writer.output_dir),
        queue_size=args.writer_queue_size,
        # This collector is a passive consumer. Backpressure here only slows
        # manifest consumption; Sync Runtime and simulation ticks continue.
        block_when_full=True,
    )
    source = MoraiSyncFileSource(
        camera_session_root,
        lidar_session_root=lidar_session_root,
        timeout_sec=args.native_file_timeout,
        poll_interval_sec=args.native_poll_interval,
        stable_checks=args.native_stable_checks,
    )
    manifest = SyncCaptureManifest(
        args.capture_manifest, poll_interval_sec=args.manifest_poll_interval
    )
    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    sample_index = int(resume_state["next_sample_index"]) if resume_state else 0
    samples_this_run = 0
    last_consumed_native_frame = (
        int(resume_state["last_consumed_native_frame"]) if resume_state else -1
    )
    if writer.frame_count != sample_index:
        raise RuntimeError(
            "Resume state/sample files disagree: checkpoint next_sample_index={}, "
            "dataset frame_count={}".format(sample_index, writer.frame_count)
        )
    incomplete_count = 0
    collection_success = False
    incomplete_path = os.path.join(writer.output_dir, "incomplete_frames.jsonl")
    provenance_path = os.path.join(writer.output_dir, "capture_provenance.jsonl")
    print("[collect/native] manifest={}".format(os.path.abspath(args.capture_manifest)))
    print("[collect/native] MORAI session id={}".format(morai_session_id))
    print("[collect/native] camera session root={}".format(camera_session_root))
    print("[collect/native] lidar session root={}".format(lidar_session_root))
    print("[collect/native] output={}".format(writer.output_dir))
    print("[collect/native] checkpoint={}".format(checkpoint_path))
    if resume_state:
        print(
            "[collect/native] resume last_native_frame={} next_sample_index={}".format(
                last_consumed_native_frame, sample_index
            )
        )
    try:
        if grpc_config is not None and state_history is None:
            from grpc_ctrl.morai_state_reader import (
                GrpcStateHistory, MoraiGrpcStateReader,
            )
            state_history = GrpcStateHistory(
                MoraiGrpcStateReader(grpc_config),
                poll_interval_sec=grpc_config.get(
                    "state_poll_interval_sec", 0.005
                ),
                max_frames=grpc_config.get("state_history_frames", 2000),
                ego_match_distance_m=grpc_config.get(
                    "state_ego_match_distance_m", 0.5
                ),
            )
        if state_history is not None:
            state_history.start()
            print("[collect/native] read-only gRPC label history started")
        with open(incomplete_path, "a", encoding="utf-8", buffering=1) as incomplete, \
                open(provenance_path, "a", encoding="utf-8", buffering=1) as provenance:
            while deadline is None or time.monotonic() < deadline:
                records = manifest.wait_for_records(deadline)
                if not records:
                    continue
                if state_history is not None and not state_history.is_aligned:
                    aligned = False
                    for candidate in reversed(records):
                        if state_history.wait_for_record(candidate, 0.01) is not None:
                            aligned = True
                            print(
                                "[collect/native] exact gRPC/native tick "
                                "alignment verified"
                            )
                            break
                    if not aligned:
                        for record in records:
                            incomplete_count += 1
                            _append_jsonl(incomplete, {
                                "native_sync_frame": int(record["native_sync_frame"]),
                                "missing_files": [],
                                "timeout_reason": "gRPC label history not yet aligned",
                            })
                        continue
                for record in records:
                    native_frame = int(record["native_sync_frame"])
                    if native_frame <= last_consumed_native_frame:
                        continue
                    if deadline is not None and time.monotonic() >= deadline:
                        collection_success = True
                        return writer.output_dir, sample_index
                    if args.max_samples > 0 and samples_this_run >= args.max_samples:
                        collection_success = True
                        return writer.output_dir, sample_index
                    label_snapshot = None
                    if state_history is not None:
                        label_snapshot = state_history.wait_for_record(
                            record,
                            (grpc_config or {}).get(
                                "state_required_timeout_sec", 2.0
                            ),
                        )
                        if label_snapshot is None:
                            incomplete_count += 1
                            _append_jsonl(incomplete, {
                                "native_sync_frame": int(record["native_sync_frame"]),
                                "missing_files": [],
                                "timeout_reason": "matching gRPC ActorStates unavailable",
                            })
                            print(
                                "[collect/native] drop frame={} without exact "
                                "gRPC labels".format(record["native_sync_frame"])
                            )
                            continue
                    try:
                        payload = source.load(record)
                    except IncompleteNativeFrame as error:
                        incomplete_count += 1
                        _append_jsonl(incomplete, {
                            "native_sync_frame": error.native_sync_frame,
                            "missing_files": error.missing_files,
                            "timeout_reason": error.timeout_reason,
                        })
                        print(
                            "[collect/native] drop incomplete frame={} missing={}".format(
                                error.native_sync_frame, error.missing_files
                            )
                        )
                        continue

                    measurement, navigation, labels = _measurement_for_record(
                        record, route_tracker, args, label_snapshot
                    )
                    if measurement is None:
                        print("[collect/native] drop off-route frame={}".format(
                            record["native_sync_frame"]
                        ))
                        continue
                    full_measurement = dict(measurement)
                    full_measurement.update({
                        "frame_id": sample_index,
                        "sample_index": "{:04d}".format(sample_index),
                        "native_sync_frame": int(record["native_sync_frame"]),
                        "runtime_session_id": record["session_id"],
                        "scenario_id": record["scenario_id"],
                        "ego_state_source_frame": int(record["ego_state_source_frame"]),
                        "next_ego_state_source_frame": int(
                            record["next_ego_state_source_frame"]
                        ),
                        "ego_raw": record["ego_state_for_frame"],
                        "next_ego_raw": record["next_ego_state"],
                        "expert_control_raw": record["applied_control"],
                        "control_source_sequence": int(
                            record["control_source_sequence"]
                        ),
                        "control_reused": bool(record["control_reused"]),
                        "control_age_frames": int(record["control_age_frames"]),
                        "control_mode": record["control_mode"],
                        "capture_manifest_record": record,
                        "morai_sync_session_id": morai_session_id,
                        "morai_camera_sync_session_root": camera_session_root,
                        "morai_lidar_sync_session_root": lidar_session_root,
                        "object_available": label_snapshot is not None,
                        "gt_objects_available": label_snapshot is not None,
                        "traffic_scope": (
                            "ego_route_relevant" if label_snapshot else "unavailable"
                        ),
                        "simulator_labels_source": (
                            "grpc" if label_snapshot else "unavailable"
                        ),
                        "grpc_label_tick_raw": (
                            label_snapshot.tick if label_snapshot else None
                        ),
                        "grpc_label_timestamp_raw": (
                            label_snapshot.simulator_timestamp_raw
                            if label_snapshot else None
                        ),
                        "gt_objects_raw": [
                            vars(obj) for obj in labels["objects"]
                        ],
                        "traffic_lights": [
                            vars(light) for light in labels["traffic_lights"]
                        ],
                        "navigation": navigation,
                    })
                    async_writer.submit(
                        sample_index,
                        measurement,
                        full_measurement,
                        actors=measurement["actors_data"],
                        affordances={
                            "traffic_light": labels["affected_light_state"],
                            "stop_sign": False,
                        },
                        cameras=payload.cameras,
                        lidar_points=payload.lidar_points,
                    )
                    _append_jsonl(provenance, {
                        "sample_index": "{:04d}".format(sample_index),
                        "native_sync_frame": int(record["native_sync_frame"]),
                        "session_id": record["session_id"],
                    })
                    last_consumed_native_frame = native_frame
                    sample_index += 1
                    samples_this_run += 1
                    if sample_index == 1 or sample_index % 10 == 0:
                        print(
                            "[collect/native] queued samples={} native_frame={} "
                            "writer_pending={}".format(
                                sample_index, record["native_sync_frame"],
                                async_writer.pending,
                            )
                        )
        collection_success = True
    except KeyboardInterrupt:
        collection_success = sample_index > 0
        print("[collect/native] interrupted; Sync Runtime is not affected")
    finally:
        manifest.close()
        if state_history is not None:
            state_history.close()
        writer_error = None
        try:
            async_writer.close()
        except BaseException as error:
            writer_error = error
            collection_success = False
        if writer_error is None and last_consumed_native_frame >= 0:
            _write_checkpoint(checkpoint_path, {
                "schema_version": 1,
                "manifest_path": os.path.abspath(args.capture_manifest),
                "runtime_session_id": _manifest_session_id(args.capture_manifest),
                "output_dir": writer.output_dir,
                "last_consumed_native_frame": last_consumed_native_frame,
                "next_sample_index": sample_index,
            })
            print(
                "[collect/native] checkpoint saved last_native_frame={} "
                "next_sample_index={}".format(
                    last_consumed_native_frame, sample_index
                )
            )
        writer.finalize(success=collection_success)
        print(
            "[collect/native] done saved={} incomplete={} output={}".format(
                sample_index, incomplete_count, writer.output_dir
            )
        )
        if writer_error is not None:
            raise writer_error
    return writer.output_dir, sample_index
