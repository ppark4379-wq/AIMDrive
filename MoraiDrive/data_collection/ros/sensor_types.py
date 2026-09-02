"""
ros/sensor_types.py
--------------------
ROSManager가 주고받는 센서 데이터 컨테이너. collect.py가 이 SensorSnapshot을
schema.EgoState / schema.ActorData(LMDrive JSON 포맷)로 변환해서 저장한다.

EgoTelemetry라는 이름을 쓴 이유: schema.EgoState(LMDrive JSON 포맷, x/y/theta/speed/
command...)와 이름이 겹치면 헷갈리므로, ROS에서 받은 원시 텔레메트리는 별도 이름으로
구분한다.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CameraFrame:
    timestamp_ns: int
    image: np.ndarray          # (H, W, 3) RGB uint8


@dataclass
class GNSSData:
    timestamp_ns: int
    latitude: float
    longitude: float
    altitude: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_z: float = 0.0


@dataclass
class IMUData:
    timestamp_ns: int
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


@dataclass
class LidarFrame:
    timestamp_ns: int
    points: np.ndarray         # (N, 4): x, y, z, intensity


@dataclass
class EgoTelemetry:
    timestamp_ns: int
    x: float
    y: float
    z: float
    yaw: float                 # rad
    speed: float                # m/s
    steer: float = 0.0          # rad


@dataclass
class GTObject:
    obj_id: int
    obj_type: str               # "vehicle" | "pedestrian" | "static"
    x: float
    y: float
    z: float
    vel_x: float
    vel_y: float
    heading_deg: float = 0.0


@dataclass
class TrafficLightState:
    tl_id: str
    state: int


@dataclass
class SensorSnapshot:
    frame_id: int
    timestamp_ns: int           # 동기화 기준(front 카메라 header.stamp)

    cameras: Dict[str, CameraFrame]
    gnss: Optional[GNSSData]
    imu: Optional[IMUData]
    lidar: Optional[LidarFrame]

    ego: Optional[EgoTelemetry] = None
    gt_objects: List[GTObject] = field(default_factory=list)
    tl_states: List[TrafficLightState] = field(default_factory=list)

    tick_count: int = 0          # gRPC SyncTimestamp.frame_count
    sim_elapsed_ns: int = 0      # gRPC SyncTimestamp.elapsed_time
