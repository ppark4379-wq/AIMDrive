"""
Central config for the MORAI -> LMDrive dataset conversion pipeline.

TODO: replace the port numbers below with MORAI's actual UDP port
assignment (SDK docs / competition-kit network setting).
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class NetworkConfig:
    ego_info_port: int = 9091
    object_info_port: int = 9092
    camera_port: Dict[str, int] = field(default_factory=lambda: {
        "front": 9093, "left": 9094, "right": 9095, "rear": 9096,
    })
    lidar_port: int = 9097
    bind_ip: str = "0.0.0.0"
    recv_buffer_size: int = 65536       # UDP datagram 최대 크기 여유값
    max_staleness_sec: float = 0.15     # 프레임 동기화 시 센서값 최대 허용 지연


@dataclass
class DatasetConfig:
    output_root: str = "./output"
    route_name: str = "route_moraiA_scenario1"
    collect_hz: float = 10.0            # LMDrive 원본 데이터셋과 동일하게 ~10Hz
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


NETWORK = NetworkConfig()
DATASET = DatasetConfig()
SENSORS = SensorConfig()
