"""
Entrypoint: connects to MORAI via gRPC (동기 모드 + tick 트리거)와 ROS1
(message_filters로 동기화된 센서 토픽 구독)으로 프레임을 수집해, 기존과 동일한
CARLA/LMDrive 스타일 route 폴더로 저장한다 — 출력 포맷은 그대로, 전송 계층만
UDP에서 gRPC+ROS1로 교체됨 (legacy_udp/에 이전 UDP 버전 보존).

프레임 트리거는 gRPC world.tick()이다: tick 1번 = 시뮬레이터가 tick_period(ms)만큼
진행 = 프레임 1개 저장. ROS 토픽 자체는 tick과 무관하게 계속 퍼블리시되고,
ros_manager.py가 message_filters로 미리 동기화해둔 최신 묶음을 매 tick마다 꺼내온다.

Run: python collect.py --route route_moraiA_scenario1 --duration 120
"""
import argparse
from dataclasses import asdict
from collections import Counter
import json
import math
import os
import time

from config import ROS, GRPC, DATASET
from ros.ros_manager import ROSManager
from ros.sensor_types import EgoTelemetry
from grpc_ctrl.morai_grpc_client import MoraiGrpcClient
from geo_reference import MGeoReference
from mgeo_navigation import (
    MGeoMap, MGeoRouteTracker, GNSSMotionEstimator, stable_mgeo_id,
    load_official_route,
)
from writer.dataset_writer import DatasetWriter
from writer.image_writer import ImageWriter
from writer.lidar_writer import LidarWriter
from schema import EgoState, ActorData, build_measurement_dict

CAMERA_VIEW_NAMES = ["front", "left", "right", "rear"]
_OBJ_TYPE_TO_TPE = {"vehicle": 0, "pedestrian": 1}


def _valid_wgs84(latitude: float, longitude: float) -> bool:
    """Reject MORAI startup/default GNSS samples before projecting to MGeo.

    MORAI can briefly publish latitude=0, longitude=0 while the simulator or
    scenario is initializing.  Projecting that default WGS84 coordinate creates
    huge fake MGeo positions, so those samples must fall back to Ego_topic
    local-map position instead.
    """
    latitude = float(latitude)
    longitude = float(longitude)
    if abs(latitude) < 1e-9 and abs(longitude) < 1e-9:
        return False
    return -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0


def _prepare_dataset_root(path: str) -> str:
    """Resolve the output root and fail before ROS startup if the drive is unavailable."""
    dataset_root = os.path.abspath(os.path.expanduser(path))
    if dataset_root == "/mnt/z" or dataset_root.startswith("/mnt/z/"):
        if not os.path.ismount("/mnt/z"):
            raise RuntimeError(
                "Windows Z: 드라이브가 WSL /mnt/z에 마운트되지 않았습니다. "
                "`sudo mount -t drvfs Z: /mnt/z` 실행 후 다시 시도하세요."
            )
    try:
        os.makedirs(dataset_root, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"데이터셋 저장 경로를 만들 수 없습니다: {dataset_root} ({exc})"
        ) from exc
    if not os.access(dataset_root, os.W_OK):
        raise RuntimeError(f"데이터셋 저장 경로에 쓰기 권한이 없습니다: {dataset_root}")
    return dataset_root


def _actor_data(obj) -> ActorData:
    heading_rad = math.radians(obj.heading_deg)
    return ActorData(
        loc=[obj.x, obj.y, obj.z],
        ori=[math.cos(heading_rad), math.sin(heading_rad)],
        box=[obj.length / 2.0, obj.width / 2.0],
        vel=[obj.vel_x, obj.vel_y, 0.0],
        tpe=_OBJ_TYPE_TO_TPE[obj.obj_type],
    )


def _angle_to_target_deg(x: float, y: float, theta_rad: float,
                         target_x: float, target_y: float) -> float:
    dx, dy = target_x - x, target_y - y
    local_x = math.cos(theta_rad) * dx + math.sin(theta_rad) * dy
    local_y = -math.sin(theta_rad) * dx + math.cos(theta_rad) * dy
    if abs(local_x) < 1e-9 and abs(local_y) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(local_y, local_x))


def _morai_light_to_lmdrive_state(status: int) -> int:
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


def _maybe_print_rate_status(skip_reasons: Counter, frame_id: int,
                             last_status_time: float, last_status_frame: int):
    now = time.monotonic()
    if now - last_status_time < 5.0:
        return last_status_time, last_status_frame
    interval = now - last_status_time
    frames_delta = frame_id - last_status_frame
    print(
        f"[collect] rate={frames_delta / interval:.2f}Hz "
        f"saved={frame_id} skipped={dict(skip_reasons)}"
    )
    skip_reasons.clear()
    return now, frame_id


def _update_grpc_ego(ros_mgr: ROSManager, grpc_client: MoraiGrpcClient) -> bool:
    state = grpc_client.get_ego_state()
    if state is None:
        return False
    ros_mgr.update_ego(EgoTelemetry(
        timestamp_ns=time.time_ns(),
        x=state["x"], y=state["y"], z=state["z"],
        yaw=math.radians(state["heading_deg"]),
        speed=math.sqrt(state["vel_x"] ** 2 + state["vel_y"] ** 2),
        steer=state["steer"],
    ))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("realtime", "grpc_legacy"), default="realtime",
                    help="realtime=손 주행 ROS 수집, grpc_legacy=이전 gRPC 검증 경로")
    ap.add_argument("--route", default=DATASET.route_name)
    ap.add_argument("--job", default=DATASET.job_name)
    ap.add_argument("--dataset-root", default=DATASET.dataset_root,
                    help="dataset output root (default: Z:\\DCM\\data_collector via /mnt/z)")
    ap.add_argument("--map-id", default=DATASET.map_id)
    ap.add_argument("--map-dir", default=DATASET.map_source_dir,
                    help="directory containing MORAI MGeo JSON files")
    ap.add_argument("--global-info", default="",
                    help="optional global_info.json path when --map-dir does not contain it")
    ap.add_argument("--route-spec", default="",
                    help="fallback checkpoint route JSON (not for the official competition route)")
    ap.add_argument("--route-definition", default="",
                    help="Scenario Runner route definition YAML (official route source)")
    ap.add_argument("--global-path", default="",
                    help="Scenario Runner official dense global-path text file")
    ap.add_argument("--route-definition-key", default="full_loop")
    ap.add_argument("--route-start-index", type=int, default=None,
                    help="repeated route 구간에서 최초 탐색을 시작할 waypoint index")
    ap.add_argument("--require-route-valid", action="store_true",
                    help="route_tracker 기준 planned route 근처에 있는 frame만 저장")
    ap.add_argument("--route-valid-max-error", type=float, default=3.0,
                    help="--require-route-valid 사용 시 허용할 최대 route 거리 오차 [m]")
    ap.add_argument("--no-route-timestamp", action="store_true")
    ap.add_argument("--duration", type=float, default=120.0, help="seconds")
    args = ap.parse_args()
    dataset_root = _prepare_dataset_root(args.dataset_root)
    print(f"[collect] 데이터셋 저장 루트: {dataset_root}")

    ros_mgr = ROSManager({
        "topics": ROS.topics,
        "sync_slop_sec": ROS.sync_slop_sec,
        "sync_queue_size": ROS.sync_queue_size,
        "ego_buffer_size": ROS.ego_buffer_size,
        "object_buffer_size": ROS.object_buffer_size,
        "traffic_light_buffer_size": ROS.traffic_light_buffer_size,
        "traffic_light_max_age_sec": ROS.traffic_light_max_age_sec,
        "control_buffer_size": ROS.control_buffer_size,
        "target_velocity_buffer_size": ROS.target_velocity_buffer_size,
        "control_max_steer_deg": ROS.control_max_steer_deg,
        "morai_to_lmdrive_steer_sign": ROS.morai_to_lmdrive_steer_sign,
        # realtime 센서는 ROS 도착시각으로 동기화하므로 Ego도 동일 clock을 사용한다.
        "normalize_ego_to_arrival": args.mode == "realtime",
    })
    grpc_client = None
    if args.mode == "grpc_legacy":
        grpc_client = MoraiGrpcClient({
            "host": GRPC.host, "port": GRPC.port, "client_key": GRPC.client_key,
            "map_name": GRPC.map_name, "ego_vehicle_model": GRPC.ego_vehicle_model,
            "tick_period_ms": GRPC.tick_period_ms,
        })
    frame_id = 0
    ds_writer = None
    collection_completed = False
    geo_reference = MGeoReference.from_optional_path(args.global_info, args.map_dir)
    if geo_reference is None:
        print("[collect] WARNING: global_info.json 없음 — gps_x/gps_y는 Ego 좌표로 대체")
    else:
        print("[collect] GNSS를 MGeo local 좌표로 변환하여 gps_x/gps_y에 저장")
    link_set_path = (os.path.join(args.map_dir, "link_set.json") if args.map_dir else "")
    node_set_path = os.path.join(args.map_dir, "node_set.json") if args.map_dir else ""
    mgeo_map = MGeoMap(
        link_set_path,
        node_set_path=node_set_path if os.path.isfile(node_set_path) else None,
    ) if os.path.isfile(link_set_path) else None
    if mgeo_map is None:
        print("[collect] WARNING: link_set.json 없음 — 현재 링크 매칭 비활성화")
    else:
        print(f"[collect] MGeo 링크 매칭 활성화 ({len(mgeo_map.links)} links)")
    route_metadata = None
    route_tracker = None
    motion_estimator = GNSSMotionEstimator()
    has_official_route = bool(args.route_definition or args.global_path)
    if has_official_route and not (args.route_definition and args.global_path):
        raise RuntimeError("--route-definition과 --global-path는 함께 지정해야 합니다")
    if has_official_route and args.route_spec:
        raise RuntimeError("공식 route 입력과 --route-spec을 동시에 사용할 수 없습니다")
    if has_official_route:
        if mgeo_map is None:
            raise RuntimeError("공식 route 사용에는 link_set.json이 필요합니다")
        route_metadata = load_official_route(
            args.route_definition, args.global_path, args.route_definition_key
        )
        missing_links = [
            link_id for link_id in route_metadata["nav_link_ids"]
            if link_id not in mgeo_map.links
        ]
        if missing_links:
            raise RuntimeError(
                f"Scenario Runner 공식 route link가 현재 MGeo에 없습니다: {missing_links}"
            )
        route_metadata["nav_node_ids"] = [
            str(mgeo_map.links[route_metadata["nav_link_ids"][0]].get("from_node_idx"))
        ] + [
            str(mgeo_map.links[link_id].get("to_node_idx"))
            for link_id in route_metadata["nav_link_ids"]
        ]
    elif args.route_spec:
        if mgeo_map is None:
            raise RuntimeError("--route-spec 사용에는 link_set.json이 필요합니다")
        with open(args.route_spec, encoding="utf-8") as f:
            route_spec = json.load(f)
        if route_spec.get("nav_link_ids") and route_spec.get("nav_waypoints"):
            route_metadata = route_spec
        else:
            route_metadata = dict(route_spec)
            route_metadata.update(mgeo_map.build_checkpoint_route(
                route_spec["checkpoints"], bool(route_spec.get("close_loop"))
            ))
    if route_metadata:
        print(
            f"[collect] 경로 로드: {route_metadata.get('route_id', args.route)} "
            f"({len(route_metadata['nav_link_ids'])} links, "
            f"{len(route_metadata['nav_waypoints'])} waypoints, "
            f"source={route_metadata.get('route_source', 'checkpoint_fallback')})"
        )
        route_tracker = MGeoRouteTracker(
            mgeo_map, route_metadata, initial_route_index=args.route_start_index
        )
        route_metadata["nav_road_options"] = list(route_tracker.point_road_options)

    try:
        ros_mgr.start()
        if grpc_client is not None:
            grpc_client.connect()
            grpc_client.start_sync_mode()
            print("[collect] gRPC tick을 진행하며 첫 센서 데이터 대기 중...")
            print("[collect] gRPC Start 후 MORAI ROS 연결이 풀렸다면 지금 Disconnect → Connect 하세요.")
        else:
            print("[collect] realtime 손 주행 모드 — ROS 센서+Ego 데이터 대기 중...")
        sensor_deadline = time.time() + 30.0
        next_status_time = time.time() + 2.0
        sensor_ready = False
        while time.time() < sensor_deadline:
            if ros_mgr.wait_for_data(timeout_sec=0.05):
                sensor_ready = True
                break
            if grpc_client is not None:
                if not grpc_client.tick(1):
                    raise RuntimeError("초기 센서 대기 중 gRPC tick 실패")
                _update_grpc_ego(ros_mgr, grpc_client)
            if time.time() >= next_status_time:
                print(f"[collect] 센서 대기 상태: {ros_mgr.readiness()}")
                next_status_time = time.time() + 2.0
        if not sensor_ready:
            raise RuntimeError(
                f"30초 안에 동기화 센서+Ego 데이터가 들어오지 않음: "
                f"{ros_mgr.readiness()}"
            )

        ds_writer = DatasetWriter(
            dataset_root, args.job, args.route,
            map_id=args.map_id, map_source_dir=args.map_dir,
            append_timestamp=DATASET.append_timestamp and not args.no_route_timestamp,
            route_metadata=route_metadata,
            extra_map_files=(
                {"global_info.json": args.global_info} if args.global_info else None
            ),
        )
        img_writer = ImageWriter(ds_writer.output_dir, view_names=CAMERA_VIEW_NAMES,
                                  ext=DATASET.image_ext, quality=DATASET.image_quality)
        lidar_writer = LidarWriter(ds_writer.output_dir)

        t_end = time.time() + args.duration
        last_grpc_frame_count_raw = None
        duplicate_tick_observed = 0
        no_new_sensor_count = 0
        skip_reasons = Counter()
        last_rate_status_time = time.monotonic()
        last_rate_status_frame = 0
        tick_interval_sec = GRPC.tick_period_ms / 1000.0
        next_tick_wall_time = time.monotonic()
        while time.time() < t_end:
            if grpc_client is None:
                snap = ros_mgr.get_snapshot(frame_id)
                if snap is None:
                    no_new_sensor_count += 1
                    skip_reasons["no_new_sensor"] += 1
                    last_rate_status_time, last_rate_status_frame = _maybe_print_rate_status(
                        skip_reasons, frame_id, last_rate_status_time,
                        last_rate_status_frame,
                    )
                    time.sleep(0.005)
                    continue
            else:
                # Legacy 검증 경로. 최종 expert 수집에서는 ROS SyncMode 서비스로 교체한다.
                sleep_sec = next_tick_wall_time - time.monotonic()
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                next_tick_wall_time += tick_interval_sec
                if next_tick_wall_time < time.monotonic():
                    next_tick_wall_time = time.monotonic() + tick_interval_sec

                sync_ts = grpc_client.tick(1)
                if not sync_ts:
                    continue
                if not _update_grpc_ego(ros_mgr, grpc_client):
                    print("frame skipped: gRPC Ego Actor State unavailable")
                    continue
                snap = ros_mgr.get_snapshot(
                    frame_id,
                    grpc_frame_count_raw=sync_ts.frame_count,
                    grpc_elapsed_time_raw=sync_ts.elapsed_time,
                )
            if snap is None or snap.ego is None:
                no_new_sensor_count += 1
                skip_reasons["missing_ego_or_snapshot"] += 1
                last_rate_status_time, last_rate_status_frame = _maybe_print_rate_status(
                    skip_reasons, frame_id, last_rate_status_time,
                    last_rate_status_frame,
                )
                continue
            if args.mode == "realtime" and (
                snap.control is None or snap.target_velocity is None
            ):
                no_new_sensor_count += 1
                skip_reasons["missing_control_or_target"] += 1
                last_rate_status_time, last_rate_status_frame = _maybe_print_rate_status(
                    skip_reasons, frame_id, last_rate_status_time,
                    last_rate_status_frame,
                )
                continue
            if (grpc_client is not None and last_grpc_frame_count_raw is not None and
                    sync_ts.frame_count <= last_grpc_frame_count_raw):
                duplicate_tick_observed += 1

            gps_local = None
            if (
                geo_reference is not None and snap.gnss is not None
                and _valid_wgs84(snap.gnss.latitude, snap.gnss.longitude)
            ):
                gps_local = geo_reference.to_local(
                    snap.gnss.latitude, snap.gnss.longitude, snap.gnss.altitude
                )
            if gps_local is not None:
                localization = motion_estimator.update(
                    gps_local[0], gps_local[1], snap.gnss.timestamp_ns,
                    snap.ego.yaw, snap.ego.speed,
                )
            else:
                localization = None
            route_progress = route_tracker.update(
                localization.x if localization else snap.ego.x,
                localization.y if localization else snap.ego.y,
                localization.heading_rad if localization else snap.ego.yaw,
            ) if route_tracker else None
            if (
                args.require_route_valid
                and route_progress is not None
                and route_progress.distance_to_route_m > args.route_valid_max_error
            ):
                no_new_sensor_count += 1
                skip_reasons["route_not_valid"] += 1
                last_rate_status_time, last_rate_status_frame = _maybe_print_rate_status(
                    skip_reasons, frame_id, last_rate_status_time,
                    last_rate_status_frame,
                )
                continue
            ego = EgoState(
                x=localization.x if localization else snap.ego.x,
                y=localization.y if localization else snap.ego.y,
                theta=localization.heading_rad if localization else snap.ego.yaw,
                speed=localization.speed_mps if localization else snap.ego.speed,
                command=route_progress.command if route_progress else 4,
                gt_command=route_progress.gt_command if route_progress else 4,
                x_command=route_progress.target_x if route_progress else 0.0,
                y_command=route_progress.target_y if route_progress else 0.0,
                near_node_x=route_progress.near_x if route_progress else 0.0,
                near_node_y=route_progress.near_y if route_progress else 0.0,
                far_node_x=route_progress.target_x if route_progress else 0.0,
                far_node_y=route_progress.target_y if route_progress else 0.0,
                gps_x=gps_local[0] if gps_local else None,
                gps_y=gps_local[1] if gps_local else None,
                steer=snap.control.steer if snap.control else 0.0,
                throttle=snap.control.throttle if snap.control else 0.0,
                brake=snap.control.brake if snap.control else 0.0,
                target_speed=(
                    snap.target_velocity.speed_mps if snap.target_velocity else 0.0
                ),
            )
            actors = {
                str(o.obj_id): _actor_data(o) for o in snap.gt_objects
                if o.obj_type in _OBJ_TYPE_TO_TPE
            } if snap.gt_objects else None

            affected_light_raw = (
                route_progress.affected_light_id if route_progress else None
            )
            affected_light_id = stable_mgeo_id(affected_light_raw)
            observed_lights = {light.tl_id: light.state for light in snap.tl_states}
            affected_state_raw = observed_lights.get(affected_light_raw)
            affected_state = (
                _morai_light_to_lmdrive_state(affected_state_raw)
                if affected_state_raw is not None else
                (4 if affected_light_raw else None)
            )
            is_red = affected_state == 0
            hazard_ids = (
                route_tracker.hazard_actor_ids(
                    snap.gt_objects, ego.x, ego.y, ego.theta,
                    route_progress.route_index, route_progress.is_junction,
                )
                if route_tracker and route_progress else
                {"vehicle": [], "lane_vehicle": [], "junction_vehicle": [],
                 "pedestrian": [], "bike": []}
            )
            should_slow = 0
            if route_progress:
                near_angle = _angle_to_target_deg(
                    ego.x, ego.y, ego.theta, route_progress.near_x, route_progress.near_y
                )
                far_angle = _angle_to_target_deg(
                    ego.x, ego.y, ego.theta, route_progress.target_x, route_progress.target_y
                )
                should_slow = int(abs(far_angle) > 45.0 or abs(near_angle) > 5.0)
            should_brake = int(
                bool(hazard_ids["vehicle"]) or
                bool(hazard_ids["lane_vehicle"]) or
                bool(hazard_ids["junction_vehicle"]) or
                bool(hazard_ids["pedestrian"]) or
                bool(hazard_ids["bike"]) or
                is_red
            )
            measurement = build_measurement_dict(
                ego, actors,
                is_junction=route_progress.is_junction if route_progress else False,
                is_red_light_present=[affected_light_id] if is_red else [],
                affected_light_id=affected_light_id,
                future_waypoints=(
                    route_progress.future_waypoints if route_progress else None
                ),
                is_vehicle_present=hazard_ids["vehicle"],
                is_bike_present=hazard_ids["bike"],
                is_lane_vehicle_present=hazard_ids["lane_vehicle"],
                is_junction_vehicle_present=hazard_ids["junction_vehicle"],
                is_pedestrian_present=hazard_ids["pedestrian"],
                is_stop_sign_present=[],
                should_slow=should_slow,
                should_brake=should_brake,
            )
            link_match = mgeo_map.match(snap.ego.x, snap.ego.y) if mgeo_map else None
            sensor_timestamps = {
                "cam_" + name: cam.timestamp_ns for name, cam in snap.cameras.items()
            }
            sensor_timestamps.update({
                "gnss": snap.gnss.timestamp_ns if snap.gnss else None,
                "imu": snap.imu.timestamp_ns if snap.imu else None,
                "lidar": snap.lidar.timestamp_ns if snap.lidar else None,
                "ego": snap.ego.timestamp_ns,
                "gt_objects": snap.gt_objects_timestamp_ns,
                "traffic_light": snap.traffic_light_timestamp_ns,
                "control": snap.control.timestamp_ns if snap.control else None,
                "target_velocity": (
                    snap.target_velocity.timestamp_ns if snap.target_velocity else None
                ),
            })
            full_measurement = dict(measurement)
            full_measurement.update({
                "frame_id": frame_id, "timestamp_ns": snap.timestamp_ns,
                "grpc_frame_count_raw": snap.grpc_frame_count_raw,
                "grpc_elapsed_time_raw": snap.grpc_elapsed_time_raw,
                "is_longtail": False,
                "sensor_timestamps_ns": sensor_timestamps,
                "ego_raw": asdict(snap.ego),
                "expert_control_raw": asdict(snap.control) if snap.control else None,
                "expert_target_velocity_raw": (
                    asdict(snap.target_velocity) if snap.target_velocity else None
                ),
                "localization": {
                    "source": "gnss_mgeo" if localization else "ego_status",
                    "heading_source": localization.heading_source if localization else "ego_status",
                    "x": localization.x if localization else snap.ego.x,
                    "y": localization.y if localization else snap.ego.y,
                    "heading_rad": localization.heading_rad if localization else snap.ego.yaw,
                    "speed_mps": localization.speed_mps if localization else snap.ego.speed,
                },
                "gnss_raw": asdict(snap.gnss) if snap.gnss else None,
                "gnss_mgeo_local": {
                    "x": gps_local[0], "y": gps_local[1], "z": gps_local[2]
                } if gps_local else None,
                "imu_raw": asdict(snap.imu) if snap.imu else None,
                "gt_objects_raw": [asdict(obj) for obj in snap.gt_objects],
                "traffic_lights": [asdict(light) for light in snap.tl_states],
                "navigation": {
                    "current_link_id": (
                        route_progress.current_link_id if route_progress else
                        (link_match.link_id if link_match else None)
                    ),
                    "distance_to_link_m": (
                        route_progress.distance_to_route_m if route_progress else
                        (link_match.distance_m if link_match else None)
                    ),
                    "link_heading_rad": link_match.heading_rad if link_match else None,
                    "projected_x": link_match.projected_x if link_match else None,
                    "projected_y": link_match.projected_y if link_match else None,
                    "route_index": route_progress.route_index if route_progress else None,
                    "route_link_index": (
                        route_progress.route_link_index if route_progress else None
                    ),
                    "near_route_index": (
                        route_progress.near_route_index if route_progress else None
                    ),
                    "target_route_index": (
                        route_progress.target_route_index if route_progress else None
                    ),
                    "command_source": "route_link_annotation" if route_progress else None,
                    "route_valid": (
                        route_progress.distance_to_route_m <= 3.0
                        if route_progress else None
                    ),
                    "distance_to_stopline": (
                        route_progress.distance_to_stopline_m if route_progress else None
                    ),
                    "affected_light_id_raw": affected_light_raw,
                    "affected_light_state_raw": affected_state_raw,
                    "affected_light_state_lmdrive": affected_state,
                },
            })
            affordances = {
                "traffic_light": affected_state,
                "stop_sign": False,
            }
            ds_writer.write_frame(
                frame_id, measurement, full_measurement,
                gnss=snap.gnss, imu=snap.imu, actors=measurement["actors_data"],
                affordances=affordances,
            )

            images = {name: cam.image for name, cam in snap.cameras.items()}
            img_writer.write(frame_id, images)
            if snap.lidar is not None:
                lidar_writer.write(frame_id, snap.lidar.points)

            frame_id += 1
            last_rate_status_time, last_rate_status_frame = _maybe_print_rate_status(
                skip_reasons, frame_id, last_rate_status_time, last_rate_status_frame,
            )
            if grpc_client is not None:
                last_grpc_frame_count_raw = sync_ts.frame_count
        print(
            f"[collect] {args.mode} summary: saved={frame_id}, "
            f"duplicate_tick_observed={duplicate_tick_observed}, "
            f"no_new_sensor={no_new_sensor_count}"
        )
        collection_completed = True
    finally:
        if ds_writer is not None:
            ds_writer.finalize(success=collection_completed)
        ros_mgr.stop()
        if grpc_client is not None:
            grpc_client.stop()
        if ds_writer is not None:
            print(f"done. {frame_id} frames written to {ds_writer.output_dir}")


if __name__ == "__main__":
    main()
