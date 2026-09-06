"""
Central config for the MORAI -> LMDrive dataset conversion pipeline.

센서 데이터는 ROS1(message_filters로 동기화), 시뮬레이션 tick 제어는 gRPC로 받는다
(자세한 배경은 ros/ros_manager.py, grpc_ctrl/morai_grpc_client.py 참고).
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ROSConfig:
    master_uri: str = ""                # 비우면 기본 ROS_MASTER_URI 환경변수 사용
    topics: Dict[str, str] = field(default_factory=lambda: {
        # rostopic list로 확인된 실제 토픽명 (2026-09-02, 카메라4+GPS+IMU 센서 추가 후)
        "camera_front": "/camera_front",
        "camera_left":  "/camera_left",
        "camera_right": "/camera_right",
        "camera_rear":  "/camera_rear",
        "gnss":          "/gps",
        "imu":           "/imu",
        "lidar":         "/lidar3D",
        "gt_objects":    "/Object_topic",
        "traffic_light": "/GetTrafficLightStatus",
        "ego_state":     "/Ego_topic",
        "control":       "/ctrl_cmd",
        "target_velocity": "/expert_target_velocity",
    })
    sync_slop_sec: float = 0.05          # message_filters.ApproximateTimeSynchronizer slop
    sync_queue_size: int = 50
    ego_buffer_size: int = 200            # 센서 timestamp에 가장 가까운 Ego 선택용 이력
    object_buffer_size: int = 200
    traffic_light_buffer_size: int = 200
    control_buffer_size: int = 200
    target_velocity_buffer_size: int = 200
    control_max_steer_deg: float = 40.0
    morai_to_lmdrive_steer_sign: float = 1.0


@dataclass
class GrpcConfig:
    host: str = "172.24.176.1"           # WSL에서 확인한 Windows 호스트(MORAI gRPC) 주소
    port: int = 7789
    client_key: str = "MoraiDataCollector"
    map_name: str = ""                   # 비우면 MORAI SIM에 이미 로드된 맵/ego 사용
    ego_vehicle_model: str = ""
    tick_period_ms: int = 100            # 100ms = 10Hz, 프레임 저장 Hz를 사실상 결정


@dataclass
class DatasetConfig:
    # Windows Z:\DCM\data_collector. WSL에서는 Z: 드라이브를 /mnt/z에 mount해야 한다.
    dataset_root: str = "/mnt/z/DCM/data_collector"
    job_name: str = "morai_collection"
    route_name: str = "route_moraiA_scenario1"
    append_timestamp: bool = True
    map_id: str = "unknown"
    map_source_dir: str = ""          # MGeo JSON 파일들이 있는 디렉터리
    # 저장 Hz는 이제 wall-clock이 아니라 GrpcConfig.tick_period_ms가 결정한다
    # (매 gRPC tick마다 한 프레임 저장). collect_hz는 참고용 표시값 — 실제로
    # 바꾸려면 tick_period_ms를 바꿔야 한다.
    collect_hz: float = 10.0
    image_ext: str = "jpg"
    image_quality: int = 92


@dataclass
class CameraExtrinsics:
    """단위: m / deg. 차량 원점 기준 상대 오프셋 (CARLA 컨벤션: x=전방+, y=우측+, z=상방+)."""
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    width: int = 800
    height: int = 600
    fov: float = 100.0


@dataclass
class LidarExtrinsics:
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass
class SensorConfig:
    """
    TODO: 아래 값은 아직 MORAI 실측치가 아니라 CARLA 원본
    (LMDrive/leaderboard/team_code/base_agent.py:137-224)을 그대로 옮긴
    자리표시자입니다. MORAI 차량에 실제로 카메라/LiDAR를 장착할 위치가
    정해지면 x/y/z/yaw를 교체하세요.

    원점 기준 미확정: CARLA는 이 원점이 차체 바운딩박스 중심이었지만,
    MORAI의 ego pose API가 후륜축/차체중심/GPS안테나 중 무엇을 기준으로
    좌표를 주는지 아직 확인 전입니다. 이게 CARLA와 다르면 카메라 오프셋도
    그 기준에 맞게 다시 재야 하고, 다르면 waypoint 예측에 체계적 오차가
    생길 수 있습니다 (자세한 내용은 대화 로그 참고).
    """
    origin_reference: str = "UNCONFIRMED: MORAI ego pose 원점이 차체중심/후륜축/GPS안테나 중 무엇인지 확인 필요"

    cameras: Dict[str, CameraExtrinsics] = field(default_factory=lambda: {
        "front": CameraExtrinsics(x=1.3, y=0.0, z=2.3, yaw=0.0,   width=1200, height=900),
        "left":  CameraExtrinsics(x=1.3, y=0.0, z=2.3, yaw=-60.0, width=400,  height=300),
        "right": CameraExtrinsics(x=1.3, y=0.0, z=2.3, yaw=60.0,  width=400,  height=300),
        "rear":  CameraExtrinsics(x=-1.3, y=0.0, z=2.3, yaw=180.0, width=400, height=300),
    })

    lidar: LidarExtrinsics = field(
        default_factory=lambda: LidarExtrinsics(x=1.3, y=0.0, z=2.5, yaw=-90.0)
    )


ROS = ROSConfig()
GRPC = GrpcConfig()
DATASET = DatasetConfig()
SENSORS = SensorConfig()
