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
import math
import time

from config import ROS, GRPC, DATASET
from ros.ros_manager import ROSManager
from ros.sensor_types import EgoTelemetry
from grpc_ctrl.morai_grpc_client import MoraiGrpcClient
from writer.dataset_writer import DatasetWriter
from writer.image_writer import ImageWriter
from writer.lidar_writer import LidarWriter
from schema import EgoState, ActorData, build_measurement_dict

CAMERA_VIEW_NAMES = ["front", "left", "right", "rear"]
_OBJ_TYPE_TO_TPE = {"vehicle": 0, "pedestrian": 1}


def _actor_data(obj) -> ActorData:
    heading_rad = math.radians(obj.heading_deg)
    return ActorData(
        loc=[obj.x, obj.y, obj.z],
        ori=[math.cos(heading_rad), math.sin(heading_rad)],
        box=[2.0, 1.0],   # TODO: MORAI GT가 실제 바운딩박스를 주면 교체
        vel=[obj.vel_x, obj.vel_y, 0.0],
        tpe=_OBJ_TYPE_TO_TPE.get(obj.obj_type, 0),
    )


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
    ap.add_argument("--route", default=DATASET.route_name)
    ap.add_argument("--job", default=DATASET.job_name)
    ap.add_argument("--map-id", default=DATASET.map_id)
    ap.add_argument("--map-dir", default=DATASET.map_source_dir,
                    help="directory containing MORAI MGeo JSON files")
    ap.add_argument("--no-route-timestamp", action="store_true")
    ap.add_argument("--duration", type=float, default=120.0, help="seconds")
    args = ap.parse_args()

    ros_mgr = ROSManager({
        "topics": ROS.topics,
        "sync_slop_sec": ROS.sync_slop_sec,
        "sync_queue_size": ROS.sync_queue_size,
        "ego_buffer_size": ROS.ego_buffer_size,
    })
    grpc_client = MoraiGrpcClient({
        "host": GRPC.host, "port": GRPC.port, "client_key": GRPC.client_key,
        "map_name": GRPC.map_name, "ego_vehicle_model": GRPC.ego_vehicle_model,
        "tick_period_ms": GRPC.tick_period_ms,
    })
    frame_id = 0
    ds_writer = None
    collection_completed = False

    try:
        ros_mgr.start()
        grpc_client.connect()
        grpc_client.start_sync_mode()

        # 동기 모드에서는 tick 전까지 새 센서가 발행되지 않을 수 있으므로,
        # 센서를 수동 대기하지 않고 tick을 진행시키며 첫 동기화 묶음을 기다린다.
        print("[collect] gRPC tick을 진행하며 첫 센서 데이터 대기 중...")
        print("[collect] gRPC Start 후 MORAI ROS 연결이 풀렸다면 지금 Disconnect → Connect 하세요.")
        sensor_deadline = time.time() + 30.0
        next_status_time = time.time() + 2.0
        sensor_ready = False
        while time.time() < sensor_deadline:
            if ros_mgr.wait_for_data(timeout_sec=0.05):
                sensor_ready = True
                break
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
            DATASET.dataset_root, args.job, args.route,
            map_id=args.map_id, map_source_dir=args.map_dir,
            append_timestamp=DATASET.append_timestamp and not args.no_route_timestamp,
        )
        img_writer = ImageWriter(ds_writer.output_dir, view_names=CAMERA_VIEW_NAMES,
                                  ext=DATASET.image_ext, quality=DATASET.image_quality)
        lidar_writer = LidarWriter(ds_writer.output_dir)

        t_end = time.time() + args.duration
        while time.time() < t_end:
            sync_ts = grpc_client.tick(1)
            if not sync_ts:
                continue
            if not _update_grpc_ego(ros_mgr, grpc_client):
                print("frame skipped: gRPC Ego Actor State unavailable")
                continue

            snap = ros_mgr.get_snapshot(
                frame_id,
                tick_count=sync_ts.frame_count,
                sim_elapsed_ns=sync_ts.elapsed_time,
            )
            if snap is None or snap.ego is None:
                print("frame skipped: waiting for a new synchronized sensor frame")
                continue

            ego = EgoState(
                x=snap.ego.x, y=snap.ego.y,
                theta=snap.ego.yaw, speed=snap.ego.speed,
            )
            actors = {
                str(o.obj_id): _actor_data(o) for o in snap.gt_objects
            } if snap.gt_objects else None

            measurement = build_measurement_dict(ego, actors)
            sensor_timestamps = {
                "cam_" + name: cam.timestamp_ns for name, cam in snap.cameras.items()
            }
            sensor_timestamps.update({
                "gnss": snap.gnss.timestamp_ns if snap.gnss else None,
                "imu": snap.imu.timestamp_ns if snap.imu else None,
                "lidar": snap.lidar.timestamp_ns if snap.lidar else None,
                "ego": snap.ego.timestamp_ns,
            })
            full_measurement = dict(measurement)
            full_measurement.update({
                "frame_id": frame_id, "timestamp_ns": snap.timestamp_ns,
                "tick_count": snap.tick_count, "sim_elapsed_ns": snap.sim_elapsed_ns,
                "is_longtail": False,
                "sensor_timestamps_ns": sensor_timestamps,
                "ego_raw": asdict(snap.ego),
                "gnss_raw": asdict(snap.gnss) if snap.gnss else None,
                "imu_raw": asdict(snap.imu) if snap.imu else None,
                "traffic_lights": [asdict(light) for light in snap.tl_states],
                "navigation": {"current_link_id": None, "route_index": None,
                               "distance_to_stopline": None},
            })
            traffic_light_state = snap.tl_states[0].state if snap.tl_states else None
            ds_writer.write_frame(
                frame_id, measurement, full_measurement,
                gnss=snap.gnss, imu=snap.imu, actors=measurement["actors_data"],
                traffic_light_state=traffic_light_state,
            )

            images = {name: cam.image for name, cam in snap.cameras.items()}
            img_writer.write(frame_id, images)
            if snap.lidar is not None:
                lidar_writer.write(frame_id, snap.lidar.points)

            frame_id += 1
        collection_completed = True
    finally:
        if ds_writer is not None:
            ds_writer.finalize(success=collection_completed)
        ros_mgr.stop()
        grpc_client.stop()
        if ds_writer is not None:
            print(f"done. {frame_id} frames written to {ds_writer.output_dir}")


if __name__ == "__main__":
    main()
