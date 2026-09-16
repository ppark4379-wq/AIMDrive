"""
Central config for the MORAI -> LMDrive dataset conversion pipeline.

기본 수집은 독립 Sync Runtime의 manifest와 MORAI 센서 저장 파일을 읽는다.
맵·경로·저장 위치·실행 옵션은 이 파일에서 변경한다.
대규모 internal Sensor Record backend는 internal_sensor_record/의 고정 계약과
전용 entrypoint를 사용하며, 이 legacy/default dispatch 설정을 덮어쓰지 않는다.
"""
from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path
import json
import os


MORAI_DRIVE = Path(__file__).resolve().parents[1]
MAP_DIR = MORAI_DRIVE / "scenario_runner/data/R_KR_PR_K-city_2025"
LOCAL_MACHINE_CONFIG = MORAI_DRIVE / ".runtime/data_collection/local_machine.json"


def _machine_setting(environment_name, local_key, default):
    value = os.environ.get(environment_name, "").strip()
    if value:
        return value
    if LOCAL_MACHINE_CONFIG.is_file():
        document = json.loads(LOCAL_MACHINE_CONFIG.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("local_machine.json must contain one JSON object")
        value = document.get(local_key, default)
        if not isinstance(value, str):
            raise ValueError(local_key + " must be a string")
        return value
    return default


MORAI_LAUNCHER = Path(
    _machine_setting("MORAI_LAUNCHER_ROOT", "launcher_root", "/mnt/c/MoraiLauncher_Win")
).expanduser().resolve()


@dataclass
class CollectorConfig:
    mode: str = "sync_native_files"
    duration: float = 120.0             # 실제 시간(초). CLI --duration으로 덮어쓰기
    global_info: str = ""
    route_spec: str = ""
    route_definition: str = str(MORAI_DRIVE / "scenario_runner/config/full_loop_links.yaml")
    global_path: str = str(MAP_DIR / "2026_molit_comp_global_path.txt")
    route_definition_key: str = "full_loop"
    route_start_index: int = None
    require_route_valid: bool = False
    route_valid_max_error: float = 3.0
    no_route_timestamp: bool = False
    # 비우면 manifest_root 아래 세션 이름(생성 시각) 기준 최신 manifest 선택.
    # 특정 실행을 다시 수집하려면 capture_manifest에 파일 경로를 지정한다.
    capture_manifest: str = ""
    manifest_root: str = str(MORAI_DRIVE / ".runtime/sync_runtime")
    label_spool_dir: str = str(MORAI_DRIVE / "dataset/.label_spool")
    morai_sync_root: str = ""
    morai_sync_base: str = ""
    # 두 센서 저장 위치에서 이름이 같은 최신 세션을 선택한다.
    morai_camera_sync_base: str = str(MORAI_LAUNCHER / "SensorData/SynchronousMode")
    morai_lidar_sync_base: str = str(MORAI_LAUNCHER / "MoraiLauncher_Win_Data/SaveFile/SensorData/SynchronousMode")
    native_file_timeout: float = 5.0
    native_poll_interval: float = 0.05
    native_stable_checks: int = 2
    manifest_poll_interval: float = 0.05
    writer_queue_size: int = 32
    max_samples: int = 0


@dataclass
class ROSConfig:
    master_uri: str = ""                # 비우면 기본 ROS_MASTER_URI 환경변수 사용
    # Navigation state is sourced from /Ego_topic. GNSS/IMU topics are not
    # required for collection or sensor synchronization.
    use_gnss: bool = False
    use_imu: bool = False
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
    traffic_light_max_age_sec: float = 1.0
    control_buffer_size: int = 200
    target_velocity_buffer_size: int = 200
    control_max_steer_deg: float = 40.0
    morai_to_lmdrive_steer_sign: float = 1.0


@dataclass
class GrpcConfig:
    host: str = "auto"                   # WSL default gateway 또는 환경변수로 해석
    port: int = 7789
    client_key: str = "MoraiDataCollector"
    map_name: str = ""                   # 비우면 MORAI SIM에 이미 로드된 맵/ego 사용
    ego_vehicle_model: str = ""
    tick_period_ms: int = 100            # 100ms = 10Hz, 프레임 저장 Hz를 사실상 결정
    # Read-only ActorStates source used to replace simulator topics that are
    # silent in SyncMode. "auto" discovers the Windows-host gateway from WSL.
    state_labels_enabled: bool = True
    state_host: str = "auto"
    state_port: int = 7789
    ego_id: str = "Ego"
    state_rpc_timeout_sec: float = 0.25
    state_retry_interval_sec: float = 0.05
    state_required_timeout_sec: float = 2.0
    state_poll_interval_sec: float = 0.005
    state_history_frames: int = 2000
    state_ego_match_distance_m: float = 0.5


@dataclass
class DatasetConfig:
    # Machine-local and ignored by Git. The internal recorder exposes its
    # Windows-backed acquisition spool through this same repository path.
    dataset_root: str = str(MORAI_DRIVE / "dataset")
    job_name: str = "competition_train"
    route_name: str = "full_loop"
    # 폴더명: <route>_<drive_tag>_<날짜·시간>_<run_note>
    # 자동으로 장애물을 감지하지 않으므로 실제 실험 조건에 맞게 지정한다.
    drive_tag: str = "npc_spon"
    run_note: str = ""  # 예: "lfd_gain0.65_max_v60". 실제 변경 사항을 직접 기록
    append_timestamp: bool = True
    map_id: str = "R_KR_PR_K-city_2025"
    map_source_dir: str = str(MAP_DIR)  # MGeo JSON 파일들이 있는 디렉터리
    # native-file 모드의 캡처 주기는 sync_runtime/config.yaml에서 설정한다.
    collect_hz: float = 10.0
    image_ext: str = "jpg"
    image_quality: int = 92


@dataclass
class CameraExtrinsics:
    """aim_sensorset.json에 저장된 MORAI 차량 상대 pose와 영상 규격."""
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
    MORAI 26.R1.H3 aim_sensorset.json의 실제 활성 센서 값이다.

    이 값은 센서 파일의 pose·FOV·해상도를 재현한다. MORAI Ego pose 원점의
    물리적 정의와 카메라의 ms 단위 노출 시각 보증은 별도 미확정 사항이다.
    """
    origin_reference: str = "MORAI vehicle-relative sensor pose; exact physical Ego origin remains unverified"

    cameras: Dict[str, CameraExtrinsics] = field(default_factory=lambda: {
        "front": CameraExtrinsics(x=1.90, y=0.00, z=1.20, pitch=2.0,
                                  yaw=0.0, width=1280, height=720, fov=90.0),
        "left":  CameraExtrinsics(x=1.15, y=0.65, z=1.20, pitch=10.0,
                                  yaw=70.0, width=640, height=480, fov=130.0),
        "right": CameraExtrinsics(x=1.15, y=-0.65, z=1.20, pitch=10.0,
                                  yaw=290.0, width=640, height=480, fov=130.0),
        "rear":  CameraExtrinsics(x=-0.14, y=0.00, z=1.20, pitch=2.0,
                                  yaw=180.0, width=640, height=480, fov=90.0),
    })

    lidar: LidarExtrinsics = field(
        default_factory=lambda: LidarExtrinsics(x=1.0, y=0.0, z=1.48, yaw=0.0)
    )


ROS = ROSConfig()
COLLECTOR = CollectorConfig()
GRPC = GrpcConfig()
DATASET = DatasetConfig()
SENSORS = SensorConfig()
