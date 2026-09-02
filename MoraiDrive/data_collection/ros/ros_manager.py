"""
ros/ros_manager.py
------------------
Windows ROS1 Noetic 환경에서 MORAI 센서 토픽을 구독.

동기화 방식 (message_filters 기반):
    카메라 4개(front/left/right/back) + GNSS + IMU + LiDAR는
    message_filters.ApproximateTimeSynchronizer로 header.stamp 기준 동기화한다.
    LiDAR도 동기화 그룹에 포함되어 신선도가 보장된다(카메라/GNSS/IMU와 slop 이내로
    맞춰진 값만 쓰게 됨).

    gt_objects / traffic_light / ego_state는 morai_msgs 정의상 Header를 갖고 있지만
    (third_party/morai_msgs의 .msg 파일로 확인됨), 매 tick 필수 항목이 아니라 동기화
    그룹에는 넣지 않고 최신값 버퍼로만 유지한다. 카메라 기준 타임스탬프와 너무
    벌어지면 로그만 남긴다(get_snapshot 참고).

프레임 트리거는 gRPC world.tick()이다 — 이 클래스는 ROS 토픽을 계속 구독/버퍼링만
하고, "언제 한 프레임을 뽑아 저장할지"는 collect.py의 메인 루프가 get_snapshot()을
호출하는 시점(gRPC tick 이후)이 결정한다.

※ 실제 morai_msgs 필드명은 설치된 패키지 버전에 따라 다를 수 있음.
  _cb_gnss, _cb_imu, _cb_objects, _cb_ego 콜백에서 필드 이름을 확인/수정 필요.
"""

import threading
import time
from typing import Dict, Optional

import numpy as np

# Windows ROS1 환경에서는 python 경로에 ROS site-packages가 잡혀 있어야 함
# (setup.bat 등으로 환경변수 설정한 상태에서 import).
try:
    import rospy
    import message_filters
    from sensor_msgs.msg import CompressedImage, Imu, PointCloud2
    from sensor_msgs import point_cloud2
    from morai_msgs.msg import (
        EgoVehicleStatus,
        ObjectStatusList,
        GPSMessage,
        GetTrafficLightStatus,
    )
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    rospy = None

from ros.sensor_types import (
    CameraFrame, GNSSData, IMUData, EgoTelemetry, LidarFrame,
    GTObject, TrafficLightState, SensorSnapshot,
)


class ROSManager:
    """
    ROS 토픽 구독 및 최신 데이터 버퍼 관리.

    get_snapshot()을 호출하면 message_filters로 이미 동기화된 카메라/GNSS/IMU/LiDAR
    묶음과, 별도로 버퍼링되는 ego/gt_objects/tl_states를 합쳐 SensorSnapshot으로 반환.
    """

    def __init__(self, config: dict):
        """
        config: {"topics": {...}, "sync_slop_sec": 0.02, "sync_queue_size": 10}
                (config.py의 ROSConfig를 dict로 넘기면 됨)
        """
        self.topics = config["topics"]
        self.slop_sec       = float(config.get("sync_slop_sec", 0.02))
        self.queue_size     = int(config.get("sync_queue_size", 10))
        self.sync_window_ns = int(self.slop_sec * 1_000_000_000)

        self._lock = threading.Lock()

        # message_filters로 동기화된 최신 묶음 (카메라4 + gnss + imu + lidar)
        self._cameras: Dict[str, CameraFrame] = {}
        self._gnss:    Optional[GNSSData]     = None
        self._imu:     Optional[IMUData]      = None
        self._lidar:   Optional[LidarFrame]   = None
        self._synced_ts: Optional[int]        = None
        self._last_sync_wall_time: float      = 0.0
        self._last_stale_warn_time: float     = 0.0

        # 동기화 그룹 밖 — 최신값만 버퍼링
        self._ego:        Optional[EgoTelemetry] = None
        self._gt_objects  = []
        self._tl_states   = []

        # message_filters.Subscriber / ApproximateTimeSynchronizer GC 방지용 보관
        self._mf_subs = []
        self._sync = None
        self._sync_member_keys = []

        self._initialized = False

    # ── 초기화 / 종료 ────────────────────────────────────────

    def start(self, node_name: str = "morai_data_collector"):
        if not ROS_AVAILABLE:
            print("[ROSManager] WARNING: rospy not available. Mock mode.")
            self._initialized = True
            return

        rospy.init_node(node_name, anonymous=True, disable_signals=True)

        # ── 동기화 그룹: 카메라 4개 + GNSS + IMU (+ LiDAR) ──────────
        sync_members = [
            ("cam_front", self.topics["camera_front"], CompressedImage),
            ("cam_left",  self.topics["camera_left"],  CompressedImage),
            ("cam_right", self.topics["camera_right"], CompressedImage),
            ("cam_rear",  self.topics["camera_rear"],  CompressedImage),
            ("gnss",      self.topics["gnss"],          GPSMessage),
            ("imu",       self.topics["imu"],           Imu),
        ]

        lidar_topic = self.topics.get("lidar")
        if lidar_topic:
            sync_members.append(("lidar", lidar_topic, PointCloud2))
        else:
            print("[ROSManager] WARNING: topics.lidar 미설정 — LiDAR는 동기화 그룹에서 제외")

        self._sync_member_keys = [key for key, _, _ in sync_members]
        self._mf_subs = [
            message_filters.Subscriber(topic, msg_type)
            for _, topic, msg_type in sync_members
        ]
        self._sync = message_filters.ApproximateTimeSynchronizer(
            self._mf_subs, queue_size=self.queue_size, slop=self.slop_sec
        )
        self._sync.registerCallback(self._on_synced)

        # 최신값 버퍼링 전용 토픽
        if self.topics.get("gt_objects"):
            rospy.Subscriber(self.topics["gt_objects"], ObjectStatusList,
                             self._cb_objects, queue_size=2)
        if self.topics.get("traffic_light"):
            rospy.Subscriber(self.topics["traffic_light"], GetTrafficLightStatus,
                             self._cb_traffic_light, queue_size=2)
        if self.topics.get("ego_state"):
            rospy.Subscriber(self.topics["ego_state"], EgoVehicleStatus,
                             self._cb_ego, queue_size=5)

        self._initialized = True
        print(f"[ROSManager] 구독 시작 완료 "
              f"(sync: {self._sync_member_keys}, slop={self.slop_sec*1000:.0f}ms, "
              f"queue_size={self.queue_size})")

        self._spin_thread = threading.Thread(target=rospy.spin, daemon=True)
        self._spin_thread.start()

    def stop(self):
        if ROS_AVAILABLE and rospy.is_shutdown() is False:
            rospy.signal_shutdown("DataCollector 종료")

    # ── 스냅샷 반환 ──────────────────────────────────────────

    def get_snapshot(self, frame_id: int, tick_count: int = 0,
                      sim_elapsed_ns: int = 0) -> Optional[SensorSnapshot]:
        """
        브리징 정책: _on_synced(push)가 아직 새 묶음을 못 만든 tick에 get_snapshot
        (pull)이 호출되면, 직전에 동기화된 값을 그대로 재사용한다(프레임 드롭 대신).
        한 번도 동기화된 적 없으면 None.
        """
        with self._lock:
            if not self._cameras or self._synced_ts is None:
                return None

            ref_ts = self._synced_ts

            ego = self._ego
            if ego is not None and abs(ego.timestamp_ns - ref_ts) > self.sync_window_ns:
                print(f"[ROSManager] WARNING: ego 데이터가 동기화 기준(front 카메라)보다 "
                      f"{abs(ego.timestamp_ns - ref_ts) / 1e6:.1f}ms 벗어남")

            snap = SensorSnapshot(
                frame_id      = frame_id,
                timestamp_ns  = ref_ts,
                cameras       = dict(self._cameras),
                gnss          = self._gnss,
                imu           = self._imu,
                lidar         = self._lidar,
                ego           = ego,
                gt_objects    = list(self._gt_objects),
                tl_states     = list(self._tl_states),
                tick_count     = tick_count,
                sim_elapsed_ns = sim_elapsed_ns,
            )

        self._warn_if_sync_stale()
        return snap

    def wait_for_data(self, timeout_sec: float = 5.0) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._cameras and self._ego:
                return True
            time.sleep(0.05)
        return False

    # ── message_filters 동기화 콜백 ───────────────────────────

    def _on_synced(self, *msgs):
        parsed = dict(zip(self._sync_member_keys, msgs))
        try:
            cameras = {
                "front": self._parse_camera(parsed["cam_front"]),
                "left":  self._parse_camera(parsed["cam_left"]),
                "right": self._parse_camera(parsed["cam_right"]),
                "rear":  self._parse_camera(parsed["cam_rear"]),
            }
            gnss  = self._parse_gnss(parsed["gnss"])
            imu   = self._parse_imu(parsed["imu"])
            lidar = self._parse_lidar(parsed["lidar"]) if "lidar" in parsed else None
        except Exception as e:
            print(f"[ROSManager] 동기화 콜백 파싱 오류: {e}")
            return

        with self._lock:
            self._cameras = cameras
            self._gnss    = gnss
            self._imu     = imu
            if lidar is not None:
                self._lidar = lidar
            self._synced_ts = cameras["front"].timestamp_ns
            self._last_sync_wall_time = time.time()

    def _warn_if_sync_stale(self):
        stale_warn_sec = max(1.0, self.slop_sec * 10)
        now = time.time()
        if not self._last_sync_wall_time:
            return
        age = now - self._last_sync_wall_time
        if age > stale_warn_sec and (now - self._last_stale_warn_time) > stale_warn_sec:
            print(f"[ROSManager] WARNING: {age:.2f}s 동안 새로운 동기화 프레임이 없음 "
                  f"(slop={self.slop_sec*1000:.0f}ms, queue_size={self.queue_size}) — "
                  f"카메라/GNSS/IMU/LiDAR 퍼블리시 주기·타임스탬프 확인 필요")
            self._last_stale_warn_time = now

    # ── 동기화 그룹 메시지 파싱 ────────────────────────────────

    def _parse_camera(self, msg: "CompressedImage") -> CameraFrame:
        import cv2
        np_arr = np.frombuffer(msg.data, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ts  = msg.header.stamp.secs * 1_000_000_000 + msg.header.stamp.nsecs
        return CameraFrame(timestamp_ns=ts, image=img)

    def _parse_gnss(self, msg: "GPSMessage") -> GNSSData:
        """morai_msgs/GPSMessage: latitude/longitude/altitude만 있고 속도 필드 없음."""
        ts = msg.header.stamp.secs * 1_000_000_000 + msg.header.stamp.nsecs
        return GNSSData(
            timestamp_ns = ts,
            latitude     = msg.latitude,
            longitude    = msg.longitude,
            altitude     = msg.altitude,
        )

    def _parse_imu(self, msg: "Imu") -> IMUData:
        ts = msg.header.stamp.secs * 1_000_000_000 + msg.header.stamp.nsecs
        return IMUData(
            timestamp_ns = ts,
            accel_x = msg.linear_acceleration.x,
            accel_y = msg.linear_acceleration.y,
            accel_z = msg.linear_acceleration.z,
            gyro_x  = msg.angular_velocity.x,
            gyro_y  = msg.angular_velocity.y,
            gyro_z  = msg.angular_velocity.z,
        )

    def _parse_lidar(self, msg: "PointCloud2") -> LidarFrame:
        ts = msg.header.stamp.secs * 1_000_000_000 + msg.header.stamp.nsecs
        pts = np.array(list(point_cloud2.read_points(
            msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
        )), dtype=np.float32)
        if pts.size == 0:
            pts = np.zeros((0, 4), dtype=np.float32)
        return LidarFrame(timestamp_ns=ts, points=pts)

    # ── 최신값 버퍼링 전용 ROS 콜백 (동기화 그룹 밖) ─────────────

    def _cb_objects(self, msg: "ObjectStatusList"):
        """morai_msgs/ObjectStatusList: npc_list/pedestrian_list/obstacle_list로 이미 분리돼서 옴."""
        try:
            objects = []
            for npc in getattr(msg, "npc_list", []):
                objects.append(GTObject(
                    obj_id=npc.unique_id, obj_type="vehicle",
                    x=npc.position.x, y=npc.position.y, z=npc.position.z,
                    vel_x=npc.velocity.x, vel_y=npc.velocity.y,
                    heading_deg=getattr(npc, "heading", 0.0),
                ))
            for ped in getattr(msg, "pedestrian_list", []):
                objects.append(GTObject(
                    obj_id=ped.unique_id, obj_type="pedestrian",
                    x=ped.position.x, y=ped.position.y, z=ped.position.z,
                    vel_x=ped.velocity.x, vel_y=ped.velocity.y,
                    heading_deg=getattr(ped, "heading", 0.0),
                ))
            for obs in getattr(msg, "obstacle_list", []):
                objects.append(GTObject(
                    obj_id=getattr(obs, "unique_id", -1), obj_type="static",
                    x=obs.position.x, y=obs.position.y, z=obs.position.z,
                    vel_x=0.0, vel_y=0.0,
                ))
            with self._lock:
                self._gt_objects = objects
        except Exception as e:
            print(f"[ROSManager] GT Objects 콜백 오류: {e}")

    def _cb_traffic_light(self, msg: "GetTrafficLightStatus"):
        try:
            with self._lock:
                self._tl_states = [TrafficLightState(
                    tl_id = str(getattr(msg, "trafficLightIndex", 0)),
                    state = int(getattr(msg, "trafficLightStatus", 0)),
                )]
        except Exception as e:
            print(f"[ROSManager] 신호등 콜백 오류: {e}")

    def _cb_ego(self, msg: "EgoVehicleStatus"):
        """
        morai_msgs/EgoVehicleStatus: position/velocity는 geometry_msgs/Vector3, heading은 deg.
        Header는 있지만(확인됨) 실제 브릿지가 stamp를 안 채울 가능성에 대비해 비정상이면
        수신 시각으로 대체한다.
        """
        try:
            ts = int(time.time() * 1e9)
            if hasattr(msg, "header") and (msg.header.stamp.secs or msg.header.stamp.nsecs):
                ts = msg.header.stamp.secs * 1_000_000_000 + msg.header.stamp.nsecs

            heading_rad = msg.heading * (np.pi / 180.0)
            speed_mps   = float(np.sqrt(msg.velocity.x ** 2 + msg.velocity.y ** 2))

            with self._lock:
                self._ego = EgoTelemetry(
                    timestamp_ns = ts,
                    x     = msg.position.x,
                    y     = msg.position.y,
                    z     = msg.position.z,
                    yaw   = heading_rad,
                    speed = speed_mps,
                    steer = getattr(msg, "front_steer_angle", 0.0),
                )
        except Exception as e:
            print(f"[ROSManager] Ego 콜백 오류: {e}")
