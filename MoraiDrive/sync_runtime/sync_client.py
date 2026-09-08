"""Minimal MORAI ROS1 built-in Synchronous Mode service client."""

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SyncStepResult:
    request_frame: int
    response_frame: int
    vehicle_status: Any
    control_rpc_sec: float
    wait_for_tick_rpc_sec: float
    sensor_capture: bool


class MoraiRosSyncClient:
    """Own SyncMode lifecycle and advance it only from MORAI responses."""

    def __init__(
        self,
        user_id: str,
        time_step_ms: int,
        service_timeout_sec: float,
        services: dict,
        ego_topic: str,
        ego_publisher_queue_size: int = 1,
        gear_service_name: str = "",
    ):
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
        self.current_frame = None
        self._started = False

        if self.time_step_ms <= 0 or self.time_step_ms % 20 != 0:
            raise ValueError("time_step_ms must be a positive multiple of 20ms")

        self.mode_service_name = str(services["mode"])
        self.control_service_name = str(services["control"])
        self.wait_for_tick_service_name = str(services["wait_for_tick"])
        self.gear_service_name = str(gear_service_name)
        service_types = {
            self.mode_service_name: MoraiSyncModeCmdSrv,
            self.control_service_name: MoraiSyncModeCtrlCmdSrv,
            self.wait_for_tick_service_name: MoraiWaitForTickSrv,
        }
        for name in service_types:
            rospy.wait_for_service(name, timeout=self.service_timeout_sec)

        self._gear_service = None
        if self.gear_service_name:
            from morai_msgs.srv import MoraiSyncModeSetGearSrv

            rospy.wait_for_service(
                self.gear_service_name, timeout=self.service_timeout_sec
            )
            self._gear_service = rospy.ServiceProxy(
                self.gear_service_name,
                MoraiSyncModeSetGearSrv,
                persistent=True,
            )

        self._mode_service = rospy.ServiceProxy(
            self.mode_service_name,
            service_types[self.mode_service_name],
            persistent=True,
        )
        self._control_service = rospy.ServiceProxy(
            self.control_service_name,
            service_types[self.control_service_name],
            persistent=True,
        )
        self._wait_for_tick_service = rospy.ServiceProxy(
            self.wait_for_tick_service_name,
            service_types[self.wait_for_tick_service_name],
            persistent=True,
        )
        self._ego_publisher = rospy.Publisher(
            str(ego_topic),
            EgoVehicleStatus,
            queue_size=max(1, int(ego_publisher_queue_size)),
        )
        self.ego_topic = str(ego_topic)

    @property
    def started(self) -> bool:
        return self._started

    def start(self):
        from morai_msgs.msg import SyncModeCmd

        if self._started:
            raise RuntimeError("MORAI ROS SyncMode is already started")
        request = SyncModeCmd(
            user_id=self.user_id,
            start_sync_mode=True,
            time_step=self.time_step_ms,
        )
        result = self._mode_service(request).response
        if not result.result or int(result.time_step) <= 0:
            raise RuntimeError(
                "MORAI SyncMode start failed: "
                "result={}, frame={}, time_step={}".format(
                    result.result, result.frame, result.time_step
                )
            )

        self.user_id = str(result.user_id or self.user_id)
        self.current_frame = int(result.frame)
        self._started = True
        return result

    def set_gear(self, gear: int):
        """Set Ego gear for the current authoritative SyncMode frame."""
        from morai_msgs.msg import SyncModeSetGear

        if not self._started or self.current_frame is None:
            raise RuntimeError("Call start() before set_gear()")
        if self._gear_service is None:
            raise RuntimeError("SyncMode gear service is not configured")
        request = SyncModeSetGear(
            gear=int(gear),
            frame=int(self.current_frame),
        )
        result = self._gear_service(request).response
        if not result.result:
            raise RuntimeError(
                "MORAI {} failed at frame={} gear={}".format(
                    self.gear_service_name, self.current_frame, gear
                )
            )
        return result

    def step(self, command, sensor_capture: bool = False) -> SyncStepResult:
        """Apply one command/capture decision, then advance exactly one tick."""
        from morai_msgs.msg import SyncModeCtrlCmd, WaitForTick

        if not self._started or self.current_frame is None:
            raise RuntimeError("Call start() before step()")

        request_frame = int(self.current_frame)
        control_request = SyncModeCtrlCmd(
            frame=request_frame,
            command=command,
            sensor_capture=bool(sensor_capture),
        )
        control_started = time.monotonic()
        control_result = self._control_service(control_request).response
        control_finished = time.monotonic()
        if not control_result.result:
            raise RuntimeError(
                "MORAI {} failed at frame={}".format(
                    self.control_service_name, request_frame
                )
            )

        tick_request = WaitForTick(
            user_id=self.user_id,
            frame=request_frame,
        )
        tick_result = self._wait_for_tick_service(tick_request).response
        tick_finished = time.monotonic()
        if not tick_result.tick_status:
            raise RuntimeError(
                "MORAI {} failed at frame={} (pause_status={})".format(
                    self.wait_for_tick_service_name,
                    request_frame,
                    tick_result.pause_status,
                )
            )

        response_frame = int(tick_result.frame)
        if response_frame != request_frame + 1:
            raise RuntimeError(
                "MORAI sync frame did not advance exactly once: "
                "request={}, response={}".format(
                    request_frame, response_frame
                )
            )

        # MORAI's response is the only authority for frame progression.
        self.current_frame = response_frame
        self._ego_publisher.publish(tick_result.vehicle_status)
        return SyncStepResult(
            request_frame=request_frame,
            response_frame=response_frame,
            vehicle_status=tick_result.vehicle_status,
            control_rpc_sec=control_finished - control_started,
            wait_for_tick_rpc_sec=tick_finished - control_finished,
            sensor_capture=bool(sensor_capture),
        )

    def stop(self):
        from morai_msgs.msg import SyncModeCmd

        if not self._started:
            return None
        try:
            request = SyncModeCmd(
                user_id=self.user_id,
                start_sync_mode=False,
                time_step=self.time_step_ms,
            )
            return self._mode_service(request).response
        finally:
            self._started = False
