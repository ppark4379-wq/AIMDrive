"""ROS lifecycle handshake and fresh Ego fallback for the independent runtime."""
import math
import threading
import time


class SyncRuntimeBridge:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ready_param = cfg.get("ready_param", "/morai_drive/scenario_ready")
        self.active_param = cfg.get("runtime_active_param", "/morai_drive/sync_runtime_active")
        self._lock = threading.Lock()
        self._sample = None
        self._sequence = 0
        self.subscriber = None
        self.rospy = None

    def start(self):
        import rospy
        from morai_msgs.msg import EgoVehicleStatus

        self.rospy = rospy
        if not rospy.core.is_initialized():
            rospy.init_node("scenario_runner", anonymous=True, disable_signals=True)
        rospy.set_param(self.ready_param, False)
        if rospy.get_param(self.active_param, False):
            raise RuntimeError("Sync Runtime is already active; restart it in scenario mode")
        self.subscriber = rospy.Subscriber(
            self.cfg.get("ego_topic", "/Ego_topic"), EgoVehicleStatus,
            self._on_ego, queue_size=1,
        )

    def _on_ego(self, message):
        state = {
            "x": float(message.position.x), "y": float(message.position.y),
            "z": float(message.position.z), "yaw_deg": float(message.heading),
            "speed": math.sqrt(sum(float(getattr(message.velocity, axis)) ** 2 for axis in ("x", "y", "z"))),
            "current_link": "", "front_wheel_angle": float(getattr(message, "wheel_angle", 0.0)),
        }
        with self._lock:
            self._sequence += 1
            self._sample = (time.monotonic(), state)

    def latest_state(self):
        with self._lock:
            if self._sample is None:
                return None
            received, state = self._sample
            if time.monotonic() - received > float(self.cfg.get("ego_max_age_sec", 1.0)):
                return None
            return dict(state)

    def wait_until_ready(self, controller):
        if controller is None or not controller.is_running():
            raise RuntimeError("AIMDrive expert is not running")
        self.rospy.set_param(self.ready_param, True)
        deadline = time.monotonic() + float(self.cfg.get("startup_timeout_sec", 30.0))
        active_sequence = None
        while time.monotonic() < deadline:
            if self.rospy.is_shutdown():
                raise RuntimeError("ROS shut down while waiting for Sync Runtime")
            if not controller.is_running():
                raise RuntimeError("AIMDrive expert stopped while waiting for Sync Runtime")
            active = bool(self.rospy.get_param(self.active_param, False))
            with self._lock:
                sequence = self._sequence
            if not active:
                active_sequence = None
            elif active_sequence is None:
                active_sequence = sequence
            elif sequence > active_sequence and self.latest_state() is not None:
                return
            time.sleep(0.02)
        raise TimeoutError("Sync Runtime startup timed out: require active runtime and a new /Ego_topic message")

    def close(self):
        try:
            if self.rospy is not None:
                self.rospy.set_param(self.ready_param, False)
        finally:
            if self.subscriber is not None:
                self.subscriber.unregister()
                self.subscriber = None
            with self._lock:
                self._sample = None
