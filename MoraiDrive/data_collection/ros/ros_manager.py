"""
ros/ros_manager.py
------------------
Windows ROS1 Noetic 환경에서 MORAI 센서 토픽을 구독.

동기화 방식 (message_filters 기반):
    카메라 4개(front/left/right/back) + LiDAR는
    message_filters.ApproximateTimeSynchronizer로 header.stamp 기준 동기화한다.
    LiDAR도 동기화 그룹에 포함되어 신선도가 보장된다(카메라/GNSS/IMU와 slop 이내로
    맞춰진 값만 쓰게 됨).

    gt_objects / traffic_light / ego_state는 동기화 그룹 밖에서 timestamp 이력으로
    관리한다. /ctrl_cmd와 /expert_target_velocity는 Header가 없으므로 ROS 도착시각을
    기록한다. snapshot에서는 Object/control/target은 front camera 기준시각의 최근접
    샘플을 선택하고, 저주기 상태값인 traffic light는 최근 값을 짧게 hold한다.

이 클래스는 ROS 토픽을 구독/버퍼링하고, ``collect.py``가 ROS SyncMode tick 이후
``get_snapshot()``을 호출할 때 새 센서 묶음을 반환한다.

※ 실제 morai_msgs 필드명은 설치된 패키지 버전에 따라 다를 수 있음.
  _cb_gnss, _cb_imu, _cb_objects, _cb_ego 콜백에서 필드 이름을 확인/수정 필요.
"""

import threading
import time
from collections import deque
from typing import Dict, Optional

import numpy as np

# Windows ROS1 환경에서는 python 경로에 ROS site-packages가 잡혀 있어야 함
# (setup.bat 등으로 환경변수 설정한 상태에서 import).
try:
    import rospy
    import message_filters
    from sensor_msgs.msg import CompressedImage, Imu, PointCloud2
    from sensor_msgs import point_cloud2
    from std_msgs.msg import Float32
    from morai_msgs.msg import (
        EgoVehicleStatus,
        ObjectStatusList,
        GPSMessage,
        GetTrafficLightStatus,
        CtrlCmd,
    )
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    rospy = None

from ros.sensor_types import (
    CameraFrame, GNSSData, IMUData, EgoTelemetry, ControlCommand,
    TargetVelocity, LidarFrame, GTObject, TrafficLightState,
    GTObjectsSample, TrafficLightSample, SensorSnapshot,
)


class ROSManager:
    """
    ROS 토픽 구독 및 최신 데이터 버퍼 관리.

    get_snapshot()을 호출하면 message_filters로 이미 동기화된 카메라/GNSS/IMU/LiDAR
    묶음과, 별도로 버퍼링되는 ego/gt_objects/tl_states를 합쳐 SensorSnapshot으로 반환.
    """

    def __init__(self, config: dict):
        """
        config: {"topics": {...}, "sync_slop_sec": 0.05, "sync_queue_size": 50,
                 "ego_buffer_size": 200}
                (config.py의 ROSConfig를 dict로 넘기면 됨)
        """
        self.topics = config["topics"]
        self.use_gnss = bool(config.get("use_gnss", False))
        self.use_imu = bool(config.get("use_imu", False))
        self.slop_sec       = float(config.get("sync_slop_sec", 0.02))
        self.queue_size     = int(config.get("sync_queue_size", 10))
        self.ego_buffer_size = int(config.get("ego_buffer_size", 200))
        self.object_buffer_size = int(config.get("object_buffer_size", 200))
        self.traffic_light_buffer_size = int(config.get("traffic_light_buffer_size", 200))
        self.traffic_light_max_age_ns = int(
            float(config.get("traffic_light_max_age_sec", 1.0)) * 1_000_000_000
        )
        self.control_buffer_size = int(config.get("control_buffer_size", 200))
        self.target_velocity_buffer_size = int(config.get("target_velocity_buffer_size", 200))
        self.control_max_steer_rad = np.deg2rad(
            float(config.get("control_max_steer_deg", 40.0))
        )
        self.morai_to_lmdrive_steer_sign = float(
            config.get("morai_to_lmdrive_steer_sign", 1.0)
        )
        self.normalize_ego_to_arrival = bool(
            config.get("normalize_ego_to_arrival", False)
        )
        self.require_fresh_ego = bool(config.get("require_fresh_ego", False))
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
        self._ego_history = deque(maxlen=self.ego_buffer_size)
        self._ego_topic_count = 0
        self._gt_objects_history = deque(maxlen=self.object_buffer_size)
        self._gt_objects_topic_count = 0
        self._tl_states_history = deque(maxlen=self.traffic_light_buffer_size)
        self._control_history = deque(maxlen=self.control_buffer_size)
        self._target_velocity_history = deque(maxlen=self.target_velocity_buffer_size)
        self._gt_objects  = []
        self._tl_states   = []

        # get_snapshot()이 이미 반환한 센서 묶음을 다시 저장하지 않도록 추적한다.
        self._last_returned_synced_ts: Optional[int] = None

        # message_filters.Subscriber / ApproximateTimeSynchronizer GC 방지용 보관
        self._mf_subs = []
        self._sync = None
        self._sync_member_keys = []
        self._sync_input_counts = {}
        self._sync_input_latest_ts = {}

        self._initialized = False

    # ── 초기화 / 종료 ────────────────────────────────────────

    def start(self, node_name: str = "morai_data_collector"):
        if not ROS_AVAILABLE:
            print("[ROSManager] WARNING: rospy not available. Mock mode.")
            self._initialized = True
            return

        rospy.init_node(node_name, anonymous=True, disable_signals=True)

        # ── 동기화 그룹: 카메라 4개 + LiDAR ────────────────────────
        # GNSS/IMU are optional. Navigation state comes from /Ego_topic.
        sync_members = [
            ("cam_front", self.topics["camera_front"], CompressedImage),
            ("cam_left",  self.topics["camera_left"],  CompressedImage),
            ("cam_right", self.topics["camera_right"], CompressedImage),
            ("cam_rear",  self.topics["camera_rear"],  CompressedImage),
        ]

        if self.use_gnss:
            sync_members.append(("gnss", self.topics["gnss"], GPSMessage))
        if self.use_imu:
            sync_members.append(("imu", self.topics["imu"], Imu))

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
        self._sync_input_counts = {key: 0 for key in self._sync_member_keys}
        self._sync_input_latest_ts = {key: None for key in self._sync_member_keys}
        for key, subscriber in zip(self._sync_member_keys, self._mf_subs):
            # 이 콜백은 ApproximateTimeSynchronizer가 등록되기 전에 추가된다.
            # 따라서 혼합 clock을 먼저 정규화한 동일 메시지를 synchronizer가 받는다.
            subscriber.registerCallback(self._prepare_sync_input, key)
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
        if self.topics.get("control"):
            rospy.Subscriber(self.topics["control"], CtrlCmd,
                             self._cb_control, queue_size=20)
        if self.topics.get("target_velocity"):
            rospy.Subscriber(self.topics["target_velocity"], Float32,
                             self._cb_target_velocity, queue_size=20)

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

    def latest_sync_timestamp(self) -> Optional[int]:
        """현재까지 수신한 가장 최신 동기화 묶음의 front-camera timestamp."""
        with self._lock:
            return self._synced_ts

    def wait_for_new_sync_timestamp(
        self, after_timestamp_ns: Optional[int], timeout_sec: float = 0.5
    ) -> Optional[int]:
        """Wait until a synchronized sensor bundle newer than the given one arrives."""
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            with self._lock:
                timestamp_ns = self._synced_ts
            if timestamp_ns is not None and (
                after_timestamp_ns is None or timestamp_ns > after_timestamp_ns
            ):
                return timestamp_ns
            time.sleep(0.002)
        return None

    def ego_topic_count(self) -> int:
        """Number of real /Ego_topic callbacks (excluding injected telemetry)."""
        with self._lock:
            return self._ego_topic_count

    def gt_objects_topic_count(self) -> int:
        """Number of /Object_topic callbacks, including empty object lists."""
        with self._lock:
            return self._gt_objects_topic_count

    def _nearest_within_sync_window(self, history, ref_ts: int, label: str):
        sample = min(
            history,
            key=lambda item: abs(item.timestamp_ns - ref_ts),
            default=None,
        )
        if sample is None:
            return None
        delta_ns = abs(sample.timestamp_ns - ref_ts)
        if delta_ns > self.sync_window_ns:
            print(
                f"[ROSManager] WARNING: 가장 가까운 {label}가 front camera 기준시각과 "
                f"{delta_ns / 1e6:.1f}ms 벗어남"
            )
            return None
        return sample

    def _latest_before_within_age(self, history, ref_ts: int, max_age_ns: int,
                                  label: str):
        past_samples = [
            sample for sample in history
            if sample.timestamp_ns <= ref_ts
        ]
        sample = max(
            past_samples,
            key=lambda item: item.timestamp_ns,
            default=None,
        )
        if sample is None:
            return self._nearest_within_sync_window(history, ref_ts, label)
        age_ns = ref_ts - sample.timestamp_ns
        if age_ns > max_age_ns:
            print(
                f"[ROSManager] WARNING: 최신 {label}가 front camera 기준시각보다 "
                f"{age_ns / 1e6:.1f}ms 오래됨"
            )
            return None
        return sample

    def get_snapshot(self, frame_id: int, grpc_frame_count_raw: int = 0,
                     grpc_elapsed_time_raw: int = 0,
                     after_timestamp_ns: Optional[int] = None) -> Optional[SensorSnapshot]:
        """
        새 동기화 묶음만 한 번 반환한다. gRPC tick 시점에 아직 새 ROS 묶음이 없으면
        None을 반환하여 동일 이미지/LiDAR가 여러 frame_id로 저장되는 것을 막는다.

        Ego는 최신값을 무조건 붙이지 않고, 보관한 timestamp 이력 중 front camera
        기준시각과 가장 가까운 샘플을 선택한다.
        """
        with self._lock:
            if not self._cameras or self._synced_ts is None:
                return None

            ref_ts = self._synced_ts
            if after_timestamp_ns is not None and ref_ts <= after_timestamp_ns:
                return None
            if ref_ts == self._last_returned_synced_ts:
                return None

            ego = min(
                self._ego_history,
                key=lambda sample: abs(sample.timestamp_ns - ref_ts),
                default=None,
            )
            ego_delta_ns = (
                abs(ego.timestamp_ns - ref_ts) if ego is not None else None
            )
            if ego_delta_ns is not None and ego_delta_ns > self.sync_window_ns:
                print(f"[ROSManager] WARNING: 가장 가까운 ego 데이터가 동기화 기준(front 카메라)과 "
                      f"{ego_delta_ns / 1e6:.1f}ms 벗어남")
            if self.require_fresh_ego and (
                ego is None or ego_delta_ns > self.sync_window_ns
            ):
                # Do not consume this sensor bundle. A matching /Ego_topic
                # callback may still arrive before the caller's timeout.
                return None

            object_sample = self._nearest_within_sync_window(
                self._gt_objects_history, ref_ts, "/Object_topic"
            )
            light_sample = self._latest_before_within_age(
                self._tl_states_history, ref_ts, self.traffic_light_max_age_ns,
                "/GetTrafficLightStatus"
            )
            control = self._nearest_within_sync_window(
                self._control_history, ref_ts, "/ctrl_cmd"
            )
            target_velocity = self._nearest_within_sync_window(
                self._target_velocity_history, ref_ts, "/expert_target_velocity"
            )

            snap = SensorSnapshot(
                frame_id      = frame_id,
                timestamp_ns  = ref_ts,
                cameras       = dict(self._cameras),
                gnss          = self._gnss,
                imu           = self._imu,
                lidar         = self._lidar,
                ego           = ego,
                control       = control,
                target_velocity = target_velocity,
                gt_objects    = list(object_sample.objects) if object_sample else [],
                tl_states     = list(light_sample.states) if light_sample else [],
                gt_objects_timestamp_ns = object_sample.timestamp_ns if object_sample else None,
                traffic_light_timestamp_ns = light_sample.timestamp_ns if light_sample else None,
                grpc_frame_count_raw = grpc_frame_count_raw,
                grpc_elapsed_time_raw = grpc_elapsed_time_raw,
                simulator_labels_source = (
                    object_sample.source if object_sample else
                    (light_sample.source if light_sample else "unavailable")
                ),
                grpc_label_tick_raw = (
                    object_sample.grpc_tick_raw if object_sample else
                    (light_sample.grpc_tick_raw if light_sample else None)
                ),
                grpc_label_timestamp_raw = (
                    object_sample.grpc_timestamp_raw if object_sample else
                    (light_sample.grpc_timestamp_raw if light_sample else None)
                ),
            )
            self._last_returned_synced_ts = ref_ts

        self._warn_if_sync_stale()
        return snap

    def wait_for_snapshot(self, frame_id: int, grpc_frame_count_raw: int = 0,
                          grpc_elapsed_time_raw: int = 0,
                          after_timestamp_ns: Optional[int] = None,
                          timeout_sec: float = 0.5) -> Optional[SensorSnapshot]:
        """기준 timestamp보다 새로운 동기화 센서 묶음이 올 때까지 기다린다."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            snap = self.get_snapshot(
                frame_id, grpc_frame_count_raw=grpc_frame_count_raw,
                grpc_elapsed_time_raw=grpc_elapsed_time_raw,
                after_timestamp_ns=after_timestamp_ns,
            )
            if snap is not None:
                return snap
            time.sleep(0.002)
        self._warn_if_sync_stale()
        return None

    def wait_for_data(self, timeout_sec: float = 5.0) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._cameras and self._ego:
                return True
            time.sleep(0.05)
        return False

    def readiness(self) -> dict:
        """라이브 시작 실패 시 어느 입력이 아직 비어 있는지 보여주는 상태 요약."""
        with self._lock:
            return {
                "sensor_sync": self._synced_ts is not None,
                "cameras": sorted(self._cameras.keys()),
                "gnss": self._gnss is not None,
                "imu": self._imu is not None,
                "lidar": self._lidar is not None,
                "ego": self._ego is not None,
                "ego_topic_count": self._ego_topic_count,
                "gt_objects": bool(self._gt_objects_history),
                "gt_objects_topic_count": self._gt_objects_topic_count,
                "traffic_light": bool(self._tl_states_history),
                "control": bool(self._control_history),
                "target_velocity": bool(self._target_velocity_history),
                "input_counts": dict(self._sync_input_counts),
                "input_latest_ts": dict(self._sync_input_latest_ts),
            }

    def _prepare_sync_input(self, msg, key: str):
        """공통 수신시각으로 header clock을 정규화하고 도착 상태를 기록한다.

        MORAI Sync 환경에서는 카메라는 Unix epoch를, GNSS/IMU/LiDAR는 simulation
        elapsed time을 header.stamp에 넣을 수 있다. simulation clock은 wall clock과
        다른 속도로 진행될 수 있어 고정 offset으로 두 clock을 변환할 수 없다.
        따라서 같은 로컬 clock의 ROS 수신시각으로 동기화한다.
        """
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        source_timestamp_ns = None
        if stamp is not None:
            source_timestamp_ns = stamp.secs * 1_000_000_000 + stamp.nsecs
        arrival_timestamp_ns = time.time_ns()
        with self._lock:
            if stamp is not None:
                msg.header.stamp = rospy.Time(
                    secs=arrival_timestamp_ns // 1_000_000_000,
                    nsecs=arrival_timestamp_ns % 1_000_000_000,
                )
            self._sync_input_counts[key] += 1
            self._sync_input_latest_ts[key] = {
                "arrival": arrival_timestamp_ns,
                "source": source_timestamp_ns,
            }

    def update_ego(self, ego: EgoTelemetry):
        """gRPC Actor State 등 ROS 외부에서 얻은 Ego 샘플을 이력에 추가한다."""
        with self._lock:
            self._ego = ego
            self._ego_history.append(ego)

    def update_grpc_labels(self, label_snapshot, sensor_timestamp_ns: int):
        """Bind one gRPC ActorStates result to the paused SyncMode sensor frame."""
        timestamp_ns = int(sensor_timestamp_ns)
        object_sample = GTObjectsSample(
            timestamp_ns=timestamp_ns,
            source_timestamp_ns=None,
            objects=list(label_snapshot.objects),
            source="grpc",
            grpc_tick_raw=int(label_snapshot.tick),
            grpc_timestamp_raw=int(label_snapshot.simulator_timestamp_raw),
        )
        light_sample = TrafficLightSample(
            timestamp_ns=timestamp_ns,
            source_timestamp_ns=None,
            states=list(label_snapshot.traffic_lights),
            source="grpc",
            grpc_tick_raw=int(label_snapshot.tick),
            grpc_timestamp_raw=int(label_snapshot.simulator_timestamp_raw),
        )
        with self._lock:
            self._gt_objects = list(object_sample.objects)
            self._tl_states = list(light_sample.states)
            self._gt_objects_history.append(object_sample)
            self._tl_states_history.append(light_sample)

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
            gnss = self._parse_gnss(parsed["gnss"]) if "gnss" in parsed else None
            imu = self._parse_imu(parsed["imu"]) if "imu" in parsed else None
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
        ts  = msg.header.stamp.secs * 1_000_000_000 + msg.header.stamp.nsecs
        image_format = str(getattr(msg, "format", "jpeg") or "jpeg").lower()
        return CameraFrame(
            timestamp_ns=ts,
            encoded=bytes(msg.data),
            encoding=image_format,
        )

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
            arrival_ts = time.time_ns()
            stamp = getattr(getattr(msg, "header", None), "stamp", None)
            source_ts = None
            if stamp is not None and (stamp.secs or stamp.nsecs):
                source_ts = stamp.secs * 1_000_000_000 + stamp.nsecs
            objects = []
            for npc in getattr(msg, "npc_list", []):
                objects.append(GTObject(
                    obj_id=npc.unique_id, obj_type="vehicle",
                    x=npc.position.x, y=npc.position.y, z=npc.position.z,
                    vel_x=npc.velocity.x / 3.6, vel_y=npc.velocity.y / 3.6,
                    heading_deg=getattr(npc, "heading", 0.0),
                    width=getattr(npc.size, "x", 0.0),
                    length=getattr(npc.size, "y", 0.0),
                    height=getattr(npc.size, "z", 0.0),
                ))
            for ped in getattr(msg, "pedestrian_list", []):
                objects.append(GTObject(
                    obj_id=ped.unique_id, obj_type="pedestrian",
                    x=ped.position.x, y=ped.position.y, z=ped.position.z,
                    vel_x=ped.velocity.x / 3.6, vel_y=ped.velocity.y / 3.6,
                    heading_deg=getattr(ped, "heading", 0.0),
                    width=getattr(ped.size, "x", 0.0),
                    length=getattr(ped.size, "y", 0.0),
                    height=getattr(ped.size, "z", 0.0),
                ))
            for obs in getattr(msg, "obstacle_list", []):
                objects.append(GTObject(
                    obj_id=getattr(obs, "unique_id", -1), obj_type="static",
                    x=obs.position.x, y=obs.position.y, z=obs.position.z,
                    vel_x=0.0, vel_y=0.0,
                    width=getattr(obs.size, "x", 0.0),
                    length=getattr(obs.size, "y", 0.0),
                    height=getattr(obs.size, "z", 0.0),
                ))
            with self._lock:
                self._gt_objects = objects
                self._gt_objects_history.append(GTObjectsSample(
                    timestamp_ns=arrival_ts,
                    source_timestamp_ns=source_ts,
                    objects=objects,
                ))
                self._gt_objects_topic_count += 1
        except Exception as e:
            print(f"[ROSManager] GT Objects 콜백 오류: {e}")

    def _cb_traffic_light(self, msg: "GetTrafficLightStatus"):
        try:
            arrival_ts = time.time_ns()
            stamp = getattr(getattr(msg, "header", None), "stamp", None)
            source_ts = None
            if stamp is not None and (stamp.secs or stamp.nsecs):
                source_ts = stamp.secs * 1_000_000_000 + stamp.nsecs
            states = [TrafficLightState(
                tl_id=str(getattr(msg, "trafficLightIndex", 0)),
                state=int(getattr(msg, "trafficLightStatus", 0)),
            )]
            with self._lock:
                self._tl_states = states
                self._tl_states_history.append(TrafficLightSample(
                    timestamp_ns=arrival_ts,
                    source_timestamp_ns=source_ts,
                    states=states,
                ))
        except Exception as e:
            print(f"[ROSManager] 신호등 콜백 오류: {e}")

    def _cb_ego(self, msg: "EgoVehicleStatus"):
        """
        morai_msgs/EgoVehicleStatus: position/velocity는 geometry_msgs/Vector3, heading은 deg.
        Header는 있지만(확인됨) 실제 브릿지가 stamp를 안 채울 가능성에 대비해 비정상이면
        수신 시각으로 대체한다.
        """
        try:
            arrival_ts = time.time_ns()
            source_ts = None
            if hasattr(msg, "header") and (msg.header.stamp.secs or msg.header.stamp.nsecs):
                source_ts = msg.header.stamp.secs * 1_000_000_000 + msg.header.stamp.nsecs
            ts = arrival_ts if self.normalize_ego_to_arrival else (source_ts or arrival_ts)

            heading_rad = msg.heading * (np.pi / 180.0)
            speed_mps   = float(np.sqrt(msg.velocity.x ** 2 + msg.velocity.y ** 2))

            with self._lock:
                ego = EgoTelemetry(
                    timestamp_ns = ts,
                    x     = msg.position.x,
                    y     = msg.position.y,
                    z     = msg.position.z,
                    yaw   = heading_rad,
                    speed = speed_mps,
                    steer = getattr(msg, "front_steer_angle", 0.0),
                    source_timestamp_ns = source_ts,
                )
                self._ego = ego
                self._ego_history.append(ego)
                self._ego_topic_count += 1
        except Exception as e:
            print(f"[ROSManager] Ego 콜백 오류: {e}")

    def _cb_control(self, msg: "CtrlCmd"):
        """Expert가 실제 publish한 /ctrl_cmd 명령을 ROS 도착시각과 함께 보관한다."""
        try:
            if self.control_max_steer_rad <= 0:
                raise ValueError("control_max_steer_deg must be positive")
            sample = ControlCommand(
                timestamp_ns=time.time_ns(),
                steer=float(np.clip(
                    self.morai_to_lmdrive_steer_sign
                    * float(getattr(msg, "front_steer", 0.0))
                    / self.control_max_steer_rad,
                    -1.0,
                    1.0,
                )),
                throttle=float(np.clip(getattr(msg, "accel", 0.0), 0.0, 1.0)),
                brake=float(np.clip(getattr(msg, "brake", 0.0), 0.0, 1.0)),
            )
            with self._lock:
                self._control_history.append(sample)
        except Exception as e:
            print(f"[ROSManager] /ctrl_cmd 콜백 오류: {e}")

    def _cb_target_velocity(self, msg: "Float32"):
        """Expert 최종 target_velocity[m/s]를 ROS 도착시각과 함께 보관한다."""
        try:
            sample = TargetVelocity(
                timestamp_ns=time.time_ns(),
                speed_mps=max(0.0, float(msg.data)),
            )
            with self._lock:
                self._target_velocity_history.append(sample)
        except Exception as e:
            print(f"[ROSManager] /expert_target_velocity 콜백 오류: {e}")
