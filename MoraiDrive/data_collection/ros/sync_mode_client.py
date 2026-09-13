"""MORAI ROS1 built-in synchronous-mode service client."""

import threading
import time


class ROSSyncModeClient:
    # Generic coordination signal. Scenario Runner does not own or configure
    # SyncMode; it only delays Expert startup until collection is ready.
    READY_PARAM = "/morai_drive/data_collector_ready"

    def __init__(self, user_id="MoraiDriveCollector", time_step_ms=100,
                 service_timeout_sec=10.0, control_topic="/ctrl_cmd",
                 ego_topic="/Ego_topic"):
        import rospy
        from morai_msgs.msg import CtrlCmd, EgoVehicleStatus
        from morai_msgs.srv import (
            MoraiSyncModeCmdSrv,
            MoraiSyncModeCtrlCmdSrv,
            MoraiWaitForTickSrv,
        )

        self.rospy = rospy
        self.CtrlCmd = CtrlCmd
        self.user_id = str(user_id)
        self.time_step_ms = int(time_step_ms)
        self.service_timeout_sec = float(service_timeout_sec)
        self.frame = None
        self._started = False
        self._condition = threading.Condition()
        self._control = None
        self._control_sequence = 0
        self._ego_publish_logged = False
        self._timing_started = time.monotonic()
        self._timing_steps = 0
        self._control_rpc_total = 0.0
        self._tick_rpc_total = 0.0
        self._tick_rpc_max = 0.0

        service_types = {
            "/SyncModeCmd": MoraiSyncModeCmdSrv,
            "/SyncModeCtrlCmd": MoraiSyncModeCtrlCmdSrv,
            "/SyncModeWaitForTick": MoraiWaitForTickSrv,
        }
        for name in service_types:
            rospy.wait_for_service(name, timeout=self.service_timeout_sec)
        self._mode_service = rospy.ServiceProxy(
            "/SyncModeCmd", service_types["/SyncModeCmd"], persistent=True
        )
        self._control_service = rospy.ServiceProxy(
            "/SyncModeCtrlCmd",
            service_types["/SyncModeCtrlCmd"],
            persistent=True,
        )
        self._tick_service = rospy.ServiceProxy(
            "/SyncModeWaitForTick",
            service_types["/SyncModeWaitForTick"],
            persistent=True,
        )
        self._subscriber = rospy.Subscriber(
            control_topic, CtrlCmd, self._on_control, queue_size=10
        )
        # MORAI's normal /Ego_topic publisher is silent in built-in SyncMode.
        # WaitForTick returns the authoritative state for that exact frame, so
        # relay it through the interface consumed by Expert and ROSManager.
        self._ego_publisher = rospy.Publisher(
            ego_topic, EgoVehicleStatus, queue_size=10
        )
        self.ego_topic = str(ego_topic)

    def _on_control(self, command):
        with self._condition:
            self._control = command
            self._control_sequence += 1
            self._condition.notify_all()

    @property
    def control_sequence(self):
        with self._condition:
            return self._control_sequence

    def wait_for_control(self, after_sequence, timeout_sec=1.0):
        deadline = time.monotonic() + float(timeout_sec)
        with self._condition:
            while self._control_sequence <= after_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, after_sequence
                self._condition.wait(remaining)
            return self._control, self._control_sequence

    def start(self):
        from morai_msgs.msg import SyncModeCmd

        # A stale True value would start Scenario Runner's Expert before this
        # collector has proved that one SyncMode tick produces ROS sensors.
        self.rospy.set_param(self.READY_PARAM, False)
        request = SyncModeCmd(
            user_id=self.user_id,
            start_sync_mode=True,
            time_step=self.time_step_ms,
        )
        result = self._mode_service(request).response
        if not result.result or result.time_step <= 0:
            raise RuntimeError(
                "MORAI /SyncModeCmd start failed: "
                f"result={result.result}, frame={result.frame}, "
                f"time_step={result.time_step}"
            )
        self.user_id = result.user_id or self.user_id
        self.frame = int(result.frame)
        self._started = True
        print(
            f"[ROSSync] started user_id={self.user_id}, "
            f"frame={self.frame}, time_step={result.time_step}ms"
        )
        return result

    def mark_ready(self):
        """Release Scenario Runner only after post-SyncMode sensor validation."""
        if not self._started:
            raise RuntimeError("Cannot mark ROS SyncMode ready before start()")
        self.rospy.set_param(self.READY_PARAM, True)
        print(f"[ROSSync] collector ready at frame={self.frame}")

    def neutral_command(self):
        command = self.CtrlCmd()
        command.longlCmdType = 1
        command.accel = 0.0
        command.brake = 1.0
        command.front_steer = 0.0
        command.rear_steer = 0.0
        return command

    def step(self, command, sensor_capture=True):
        """Apply command to the current frame and wait for exactly one tick."""
        from morai_msgs.msg import SyncModeCtrlCmd, WaitForTick

        if not self._started or self.frame is None:
            raise RuntimeError("Call ROSSyncModeClient.start() first")
        control_request = SyncModeCtrlCmd(
            frame=self.frame,
            command=command,
            sensor_capture=bool(sensor_capture),
        )
        control_started = time.monotonic()
        control_result = self._control_service(control_request).response
        control_finished = time.monotonic()
        if not control_result.result:
            raise RuntimeError(
                f"MORAI /SyncModeCtrlCmd failed at frame={self.frame}"
            )

        tick_request = WaitForTick(user_id=self.user_id, frame=self.frame)
        tick_result = self._tick_service(tick_request).response
        tick_finished = time.monotonic()
        if not tick_result.tick_status:
            raise RuntimeError(
                f"MORAI /SyncModeWaitForTick failed at frame={self.frame}"
            )
        if int(tick_result.frame) <= self.frame:
            raise RuntimeError(
                "MORAI sync frame did not advance: "
                f"before={self.frame}, after={tick_result.frame}"
            )
        self.frame = int(tick_result.frame)
        self._ego_publisher.publish(tick_result.vehicle_status)
        if not self._ego_publish_logged:
            print(
                "[ROSSync] WaitForTick vehicle_status -> "
                f"{self.ego_topic} bridge active"
            )
            self._ego_publish_logged = True
        self._record_timing(
            control_finished - control_started,
            tick_finished - control_finished,
        )
        return tick_result

    def _record_timing(self, control_rpc_sec, tick_rpc_sec):
        self._timing_steps += 1
        self._control_rpc_total += control_rpc_sec
        self._tick_rpc_total += tick_rpc_sec
        self._tick_rpc_max = max(self._tick_rpc_max, tick_rpc_sec)
        now = time.monotonic()
        if now - self._timing_started < 5.0:
            return
        count = max(1, self._timing_steps)
        print(
            "[ROSSync] rpc timing: "
            f"ctrl={self._control_rpc_total * 1000.0 / count:.1f}ms avg, "
            f"wait_tick={self._tick_rpc_total * 1000.0 / count:.1f}ms avg/"
            f"{self._tick_rpc_max * 1000.0:.1f}ms max, samples={count}"
        )
        self._timing_started = now
        self._timing_steps = 0
        self._control_rpc_total = 0.0
        self._tick_rpc_total = 0.0
        self._tick_rpc_max = 0.0

    def stop(self):
        if not self._started:
            return
        from morai_msgs.msg import SyncModeCmd

        try:
            request = SyncModeCmd(
                user_id=self.user_id,
                start_sync_mode=False,
                time_step=self.time_step_ms,
            )
            result = self._mode_service(request).response
            print(f"[ROSSync] stopped result={result.result}, frame={result.frame}")
        finally:
            self._started = False
            self.rospy.set_param(self.READY_PARAM, False)
