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
    # MORAI publishes sensor images as sensor_msgs/CompressedImage. Keep the
    # original encoded payload so collection does not decode and then encode
    # four JPEGs on every synchronous tick. ``image`` remains available for
    # tests and non-compressed producers.
    image: Optional[np.ndarray] = None   # (H, W, 3) RGB uint8
    encoded: Optional[bytes] = None
    encoding: str = "jpeg"


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
    # 동기화에는 timestamp_ns를 사용하고 MORAI header 원본은 별도로 보존한다.
    source_timestamp_ns: Optional[int] = None


@dataclass
class ControlCommand:
    timestamp_ns: int
    steer: float
    throttle: float
    brake: float


@dataclass
class TargetVelocity:
    timestamp_ns: int
    speed_mps: float


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
    width: float = 0.0
    length: float = 0.0
    height: float = 0.0


@dataclass
class TrafficLightState:
    tl_id: str
    state: int


@dataclass
class GTObjectsSample:
    timestamp_ns: int
    objects: List[GTObject]
    source_timestamp_ns: Optional[int] = None
    source: str = "ros"
    grpc_tick_raw: Optional[int] = None
    grpc_timestamp_raw: Optional[int] = None


@dataclass
class TrafficLightSample:
    timestamp_ns: int
    states: List[TrafficLightState]
    source_timestamp_ns: Optional[int] = None
    source: str = "ros"
    grpc_tick_raw: Optional[int] = None
    grpc_timestamp_raw: Optional[int] = None


@dataclass
class SensorSnapshot:
    frame_id: int
    timestamp_ns: int           # 동기화 기준(front 카메라 header.stamp)

    cameras: Dict[str, CameraFrame]
    gnss: Optional[GNSSData]
    imu: Optional[IMUData]
    lidar: Optional[LidarFrame]

    ego: Optional[EgoTelemetry] = None
    control: Optional[ControlCommand] = None
    target_velocity: Optional[TargetVelocity] = None
    gt_objects: List[GTObject] = field(default_factory=list)
    tl_states: List[TrafficLightState] = field(default_factory=list)
    gt_objects_timestamp_ns: Optional[int] = None
    traffic_light_timestamp_ns: Optional[int] = None

    # MORAI proto가 단위/센서 프레임 대응을 명시하지 않으므로 원본값으로만 보존한다.
    grpc_frame_count_raw: int = 0
    grpc_elapsed_time_raw: int = 0
    simulator_labels_source: str = "unavailable"
    grpc_label_tick_raw: Optional[int] = None
    grpc_label_timestamp_raw: Optional[int] = None
