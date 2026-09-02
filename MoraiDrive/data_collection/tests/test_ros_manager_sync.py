"""
tests/test_ros_manager_sync.py
--------------------------------
rospy 없이 ROSManager의 message_filters 브리징 로직(_on_synced → get_snapshot)을
검증하는 모킹 테스트.

실제 slop 매칭 자체는 message_filters.ApproximateTimeSynchronizer(외부 ROS1 패키지)가
수행하므로 여기서는 재검증하지 않는다. 대신 이 파일이 실제로 작성한 로직만 검증한다:
  1. _on_synced로 "이미 매칭된 묶음"이 들어왔을 때 카메라/GNSS/IMU/LiDAR가
     올바르게 반영되는지
  2. get_snapshot()의 브리징 정책 — 아직 새 묶음이 안 온 tick엔 직전 값을 재사용하고,
     동기화 그룹 밖(ego)이 벌어져도 드롭하지 않고 로그만 남기는지

실행:
    python tests/test_ros_manager_sync.py   (data_collection/ 에서 실행)
"""
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _make_manager():
    config = {
        "topics": {
            "camera_front": "/front", "camera_left": "/left",
            "camera_right": "/right", "camera_rear": "/back",
            "gnss": "/gps", "imu": "/imu", "lidar": "/lidar",
            "gt_objects": "/Object_topic", "traffic_light": "/GetTrafficLightStatus",
            "ego_state": "/Ego_topic",
        },
        "sync_slop_sec": 0.02,
        "sync_queue_size": 10,
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


def test_stale_reuse_when_no_new_sync_yet():
    """다음 tick에 아직 새 동기화 묶음이 안 왔으면, get_snapshot()은 직전 값을 재사용한다
    (프레임 드롭 대신 stale 값 재사용 정책)."""
    mgr = _make_manager()
    mgr._on_synced(*_synced_bundle(200))
    snap1 = mgr.get_snapshot(frame_id=1)

    snap2 = mgr.get_snapshot(frame_id=2)
    assert snap2 is not None
    assert snap2.timestamp_ns == snap1.timestamp_ns
    print("PASS: 새 동기화 콜백이 없는 tick엔 직전 동기화 값을 재사용")


def test_ego_staleness_does_not_drop_but_is_logged():
    """ego는 동기화 그룹 밖이라, ref_ts와 많이 벌어져도 스냅샷에서 드롭되지 않고
    그대로 포함된다 (경고 로그만 get_snapshot 내부에서 출력됨)."""
    mgr = _make_manager()
    mgr._on_synced(*_synced_bundle(300))

    mgr._cb_ego(_fake_ego(300, 500_000_000))  # ref_ts보다 500ms 벌어짐 (slop 20ms 초과)
    assert mgr._ego is not None

    snap = mgr.get_snapshot(frame_id=1)
    assert snap is not None and snap.ego is not None
    print("PASS: ego가 동기화 기준과 벌어져도 스냅샷에서 드롭되지 않고 포함됨(로그만 남김)")


def test_lidar_included_in_sync_group():
    """이번 마이그레이션의 핵심 목표: LiDAR도 이제 신선도 체크 대상이다."""
    mgr = _make_manager()
    mgr._on_synced(*_synced_bundle(400))
    snap = mgr.get_snapshot(frame_id=1)
    assert snap.lidar is not None
    assert snap.lidar.timestamp_ns == snap.timestamp_ns
    print("PASS: LiDAR가 동기화 그룹에 포함되어 카메라와 동일 타임스탬프를 가짐")


if __name__ == "__main__":
    test_no_snapshot_before_first_sync()
    test_on_synced_populates_snapshot()
    test_stale_reuse_when_no_new_sync_yet()
    test_ego_staleness_does_not_drop_but_is_logged()
    test_lidar_included_in_sync_group()
    print("\nAll ROSManager sync bridging tests passed.")
