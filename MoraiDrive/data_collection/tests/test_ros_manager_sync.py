"""
tests/test_ros_manager_sync.py
--------------------------------
rospy 없이 ROSManager의 message_filters 브리징 로직(_on_synced → get_snapshot)을
검증하는 모킹 테스트.

실제 slop 매칭 자체는 message_filters.ApproximateTimeSynchronizer(외부 ROS1 패키지)가
수행하므로 여기서는 재검증하지 않는다. 대신 이 파일이 실제로 작성한 로직만 검증한다:
  1. _on_synced로 "이미 매칭된 묶음"이 들어왔을 때 카메라/GNSS/IMU/LiDAR가
     올바르게 반영되는지
  2. get_snapshot()이 동일 센서 묶음을 중복 반환하지 않는지
  3. Ego timestamp 이력에서 센서 기준시각과 가장 가까운 샘플을 선택하는지

실행:
    python tests/test_ros_manager_sync.py   (data_collection/ 에서 실행)
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ros.ros_manager import ROSManager  # noqa: E402
from ros.sensor_types import LidarFrame  # noqa: E402


def _fake_header(secs, nsecs=0):
    return SimpleNamespace(stamp=SimpleNamespace(secs=secs, nsecs=nsecs))


def _v3(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def _fake_compressed_image(secs, nsecs=0):
    ok, buf = cv2.imencode(".jpg", np.zeros((4, 4, 3), dtype=np.uint8))
    assert ok
    return SimpleNamespace(header=_fake_header(secs, nsecs), data=buf.tobytes())


def _fake_gnss(secs, nsecs=0):
    return SimpleNamespace(header=_fake_header(secs, nsecs),
                            latitude=37.1, longitude=127.1, altitude=10.0)


def _fake_imu(secs, nsecs=0):
    return SimpleNamespace(header=_fake_header(secs, nsecs),
                            linear_acceleration=_v3(0.1, 0.2, 9.8),
                            angular_velocity=_v3(0.0, 0.0, 0.01))


def _fake_lidar(secs, nsecs=0):
    return SimpleNamespace(header=_fake_header(secs, nsecs))


def _fake_ego(secs, nsecs=0):
    return SimpleNamespace(
        header=_fake_header(secs, nsecs),
        position=_v3(1.0, 2.0, 0.0),
        velocity=_v3(3.0, 0.0, 0.0),
        heading=0.0,
    )


def _fake_objects(secs, obj_id):
    obj = SimpleNamespace(
        unique_id=obj_id,
        position=_v3(float(obj_id), 0.0, 0.0),
        velocity=_v3(0.0, 0.0, 0.0),
        heading=0.0,
        size=_v3(2.0, 1.0, 1.0),
    )
    return SimpleNamespace(
        header=_fake_header(secs), npc_list=[obj],
        pedestrian_list=[], obstacle_list=[],
    )


def _fake_traffic_light(secs, light_id, status):
    return SimpleNamespace(
        header=_fake_header(secs), trafficLightIndex=light_id,
        trafficLightStatus=status,
    )


def _make_manager(normalize_ego_to_arrival=False):
    config = {
        "topics": {
            "camera_front": "/front", "camera_left": "/left",
            "camera_right": "/right", "camera_rear": "/back",
            "gnss": "/gps", "imu": "/imu", "lidar": "/lidar",
            "gt_objects": "/Object_topic", "traffic_light": "/GetTrafficLightStatus",
            "ego_state": "/Ego_topic", "control": "/ctrl_cmd",
            "target_velocity": "/expert_target_velocity",
        },
        "sync_slop_sec": 0.02,
        "sync_queue_size": 10,
        "traffic_light_max_age_sec": 1.0,
        "normalize_ego_to_arrival": normalize_ego_to_arrival,
    }
    mgr = ROSManager(config)
    # start()는 rospy가 필요해서 호출하지 않고, start()가 설정하는
    # _sync_member_keys만 동일하게 맞춰서 _on_synced를 직접 테스트한다.
    mgr._sync_member_keys = ["cam_front", "cam_left", "cam_right", "cam_rear",
                              "gnss", "imu", "lidar"]
    # PointCloud2 파싱(sensor_msgs.point_cloud2)은 실제 ROS1 환경에서만 import되고
    # 이 테스트가 검증하려는 브리징 로직과 무관하므로, header.stamp만 반영하는
    # 가벼운 스텁으로 대체한다.
    mgr._parse_lidar = lambda msg: LidarFrame(
        timestamp_ns=msg.header.stamp.secs * 1_000_000_000 + msg.header.stamp.nsecs,
        points=np.zeros((0, 4), dtype=np.float32),
    )
    return mgr


def _synced_bundle(secs, nsecs=0):
    return (
        _fake_compressed_image(secs, nsecs),
        _fake_compressed_image(secs, nsecs),
        _fake_compressed_image(secs, nsecs),
        _fake_compressed_image(secs, nsecs),
        _fake_gnss(secs, nsecs),
        _fake_imu(secs, nsecs),
        _fake_lidar(secs, nsecs),
    )


def test_no_snapshot_before_first_sync():
    mgr = _make_manager()
    assert mgr.get_snapshot(frame_id=0) is None
    print("PASS: 첫 동기화 전에는 get_snapshot()이 None")


def test_on_synced_populates_snapshot():
    mgr = _make_manager()
    mgr._on_synced(*_synced_bundle(100, 15_000_000))

    snap = mgr.get_snapshot(frame_id=1)
    assert snap is not None
    assert snap.timestamp_ns == 100 * 1_000_000_000 + 15_000_000
    assert snap.gnss is not None and snap.imu is not None and snap.lidar is not None
    assert set(snap.cameras.keys()) == {"front", "left", "right", "rear"}
    print("PASS: _on_synced 이후 get_snapshot()이 카메라+gnss+imu+lidar를 모두 채움")


def test_no_duplicate_when_no_new_sync_yet():
    """새 동기화 묶음이 없으면 동일 센서 프레임을 중복 반환하지 않는다."""
    mgr = _make_manager()
    mgr._on_synced(*_synced_bundle(200))
    assert mgr.get_snapshot(frame_id=1) is not None
    assert mgr.get_snapshot(frame_id=2) is None
    print("PASS: 새 동기화 콜백이 없는 tick엔 이전 센서 프레임을 중복 반환하지 않음")


def test_snapshot_must_be_newer_than_pre_tick_baseline():
    """tick 이전 묶음은 반환하지 않고, 이후 도착한 묶음만 tick에 결합한다."""
    mgr = _make_manager()
    mgr._on_synced(*_synced_bundle(250))
    baseline = mgr.latest_sync_timestamp()
    assert baseline == 250_000_000_000
    assert mgr.get_snapshot(frame_id=1, after_timestamp_ns=baseline) is None

    mgr._on_synced(*_synced_bundle(250, 100_000_000))
    snap = mgr.wait_for_snapshot(
        frame_id=1, grpc_frame_count_raw=7,
        after_timestamp_ns=baseline, timeout_sec=0.01
    )
    assert snap is not None
    assert snap.timestamp_ns == 250_100_000_000
    assert snap.grpc_frame_count_raw == 7
    print("PASS: tick 이후 새 센서 timestamp만 snapshot으로 반환")

def test_nearest_ego_is_selected_from_history():
    """최신 Ego가 아니라 센서 timestamp와 가장 가까운 Ego를 결합한다."""
    mgr = _make_manager()
    mgr._cb_ego(_fake_ego(299, 900_000_000))
    nearest = mgr._ego_history[-1]
    mgr._cb_ego(_fake_ego(300, 500_000_000))
    mgr._on_synced(*_synced_bundle(300))

    snap = mgr.get_snapshot(frame_id=1)
    assert snap is not None and snap.ego is nearest
    assert snap.ego.timestamp_ns == 299_900_000_000
    print("PASS: Ego 이력에서 센서 timestamp와 가장 가까운 샘플을 선택")


def test_realtime_ego_uses_arrival_clock_and_preserves_source_stamp():
    mgr = _make_manager(normalize_ego_to_arrival=True)
    with patch("ros.ros_manager.time.time_ns", return_value=999_000_000_123):
        mgr._cb_ego(_fake_ego(300, 500_000_000))
    ego = mgr._ego_history[-1]
    assert ego.timestamp_ns == 999_000_000_123
    assert ego.source_timestamp_ns == 300_500_000_000
    print("PASS: realtime Ego는 도착 clock으로 결합하고 원본 header stamp를 보존")


def test_lidar_included_in_sync_group():
    """이번 마이그레이션의 핵심 목표: LiDAR도 이제 신선도 체크 대상이다."""
    mgr = _make_manager()
    mgr._on_synced(*_synced_bundle(400))
    snap = mgr.get_snapshot(frame_id=1)
    assert snap.lidar is not None
    assert snap.lidar.timestamp_ns == snap.timestamp_ns
    print("PASS: LiDAR가 동기화 그룹에 포함되어 카메라와 동일 타임스탬프를 가짐")


def test_async_labels_select_nearest_frame_sample():
    mgr = _make_manager()
    ctrl = SimpleNamespace(front_steer=0.0, accel=0.3, brake=0.0)
    with patch("ros.ros_manager.time.time_ns", return_value=500_005_000_000):
        mgr._cb_objects(_fake_objects(1, 101))
        mgr._cb_traffic_light(_fake_traffic_light(1, "TL-near", 1))
        mgr._cb_control(ctrl)
        mgr._cb_target_velocity(SimpleNamespace(data=8.0))
    with patch("ros.ros_manager.time.time_ns", return_value=500_100_000_000):
        mgr._cb_objects(_fake_objects(2, 202))
        mgr._cb_traffic_light(_fake_traffic_light(2, "TL-latest", 16))

    mgr._on_synced(*_synced_bundle(500))
    snap = mgr.get_snapshot(frame_id=1)

    assert [obj.obj_id for obj in snap.gt_objects] == [101]
    assert [light.tl_id for light in snap.tl_states] == ["TL-near"]
    assert snap.control is not None and snap.target_velocity is not None
    assert snap.gt_objects_timestamp_ns == 500_005_000_000
    assert snap.traffic_light_timestamp_ns == 500_005_000_000


def test_stale_async_labels_are_not_reused():
    mgr = _make_manager()
    ctrl = SimpleNamespace(front_steer=0.0, accel=0.3, brake=0.0)
    with patch("ros.ros_manager.time.time_ns", return_value=600_000_000_000):
        mgr._cb_objects(_fake_objects(1, 101))
        mgr._cb_traffic_light(_fake_traffic_light(1, "TL", 1))
        mgr._cb_control(ctrl)
        mgr._cb_target_velocity(SimpleNamespace(data=8.0))
    mgr._on_synced(*_synced_bundle(601))
    snap = mgr.get_snapshot(frame_id=1)

    assert snap.gt_objects == []
    assert [light.tl_id for light in snap.tl_states] == ["TL"]
    assert snap.control is None and snap.target_velocity is None


def test_traffic_light_status_is_held_for_low_rate_topic():
    mgr = _make_manager()
    with patch("ros.ros_manager.time.time_ns", return_value=700_500_000_000):
        mgr._cb_traffic_light(_fake_traffic_light(700, "TL-held", 1))
    mgr._on_synced(*_synced_bundle(701, 300_000_000))
    snap = mgr.get_snapshot(frame_id=1)

    assert [light.tl_id for light in snap.tl_states] == ["TL-held"]
    assert snap.traffic_light_timestamp_ns == 700_500_000_000


if __name__ == "__main__":
    test_no_snapshot_before_first_sync()
    test_on_synced_populates_snapshot()
    test_no_duplicate_when_no_new_sync_yet()
    test_snapshot_must_be_newer_than_pre_tick_baseline()
    test_nearest_ego_is_selected_from_history()
    test_realtime_ego_uses_arrival_clock_and_preserves_source_stamp()
    test_lidar_included_in_sync_group()
    test_async_labels_select_nearest_frame_sample()
    test_stale_async_labels_are_not_reused()
    test_traffic_light_status_is_held_for_low_rate_topic()
    print("\nAll ROSManager sync bridging tests passed.")
