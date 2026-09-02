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
import math
import time

from config import ROS, GRPC, DATASET
from ros.ros_manager import ROSManager
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default=DATASET.route_name)
    ap.add_argument("--duration", type=float, default=120.0, help="seconds")
    args = ap.parse_args()

    ros_mgr = ROSManager({
        "topics": ROS.topics,
        "sync_slop_sec": ROS.sync_slop_sec,
        "sync_queue_size": ROS.sync_queue_size,
    })
    ros_mgr.start()

    grpc_client = MoraiGrpcClient({
        "host": GRPC.host, "port": GRPC.port, "client_key": GRPC.client_key,
        "map_name": GRPC.map_name, "ego_vehicle_model": GRPC.ego_vehicle_model,
        "tick_period_ms": GRPC.tick_period_ms,
    })
    grpc_client.connect()
    grpc_client.start_sync_mode()

    ds_writer = DatasetWriter(DATASET.output_root, args.route)
    img_writer = ImageWriter(ds_writer.output_dir, view_names=CAMERA_VIEW_NAMES,
                              ext=DATASET.image_ext, quality=DATASET.image_quality)
    lidar_writer = LidarWriter(ds_writer.output_dir)

    print("[collect] 센서 데이터 대기 중...")
    if not ros_mgr.wait_for_data(timeout_sec=10.0):
        print("[collect] 센서 데이터 타임아웃 — MORAI ROS 브릿지/토픽 설정 확인 필요")
        ros_mgr.stop()
        grpc_client.stop()
        return

    frame_id = 0
    t_end = time.time() + args.duration

    try:
        while time.time() < t_end:
            sync_ts = grpc_client.tick(1)
            if not sync_ts:
                continue

            snap = ros_mgr.get_snapshot(
                frame_id,
                tick_count=sync_ts.frame_count,
                sim_elapsed_ns=sync_ts.elapsed_time,
            )
            if snap is None or snap.ego is None:
                print("frame skipped: waiting for fresh sensor data")
                continue

            ego = EgoState(
                x=snap.ego.x, y=snap.ego.y,
                theta=snap.ego.yaw, speed=snap.ego.speed,
            )
            actors = {
                str(o.obj_id): _actor_data(o) for o in snap.gt_objects
            } if snap.gt_objects else None

            measurement = build_measurement_dict(ego, actors)
            ds_writer.write_frame(frame_id, measurement)

            images = {name: cam.image for name, cam in snap.cameras.items()}
            img_writer.write(frame_id, images)
            if snap.lidar is not None:
                lidar_writer.write(frame_id, snap.lidar.points)

            frame_id += 1
    finally:
        ds_writer.finalize()
        ros_mgr.stop()
        grpc_client.stop()
        print(f"done. {frame_id} frames written to {ds_writer.output_dir}")


if __name__ == "__main__":
    main()
