"""MORAI's ROS service transaction for one authoritative simulation frame."""
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class Step:
    request_frame: int
    response_frame: int
    vehicle_status: object
    control_rpc_sec: float
    wait_for_tick_rpc_sec: float
    sensor_capture: bool


class RosTickClient:
    def __init__(self, user_id, time_step_ms, service_timeout_sec, services, ego_topic,
                 ego_publisher_queue_size=1, gear_service_name=""):
        import rospy
        from morai_msgs import msg, srv
        self.ros, self.msg = rospy, msg
        self.CtrlCmd, self.user_id = msg.CtrlCmd, str(user_id)
        self.time_step_ms, self.timeout = int(time_step_ms), float(service_timeout_sec)
        if self.time_step_ms <= 0 or self.time_step_ms % 20:
            raise ValueError("time_step_ms must be a positive multiple of 20ms")
        self.current_frame, self.started = None, False
        self.names = {name:str(services[name]) for name in ("mode","control","wait_for_tick")}
        types = {"mode":srv.MoraiSyncModeCmdSrv,"control":srv.MoraiSyncModeCtrlCmdSrv,
                 "wait_for_tick":srv.MoraiWaitForTickSrv}
        for name in self.names.values():
            rospy.wait_for_service(name,timeout=self.timeout)
        self.gear_name, self.gear = str(gear_service_name), None
        if self.gear_name:
            rospy.wait_for_service(self.gear_name,timeout=self.timeout)
            self.gear = rospy.ServiceProxy(self.gear_name,srv.MoraiSyncModeSetGearSrv,persistent=True)
        self.rpc = {name:rospy.ServiceProxy(path,types[name],persistent=True)
                    for name,path in self.names.items()}
        self.publisher = rospy.Publisher(str(ego_topic),msg.EgoVehicleStatus,
                                         queue_size=max(1,int(ego_publisher_queue_size)))

    def _mode(self, enabled):
        return self.rpc["mode"](self.msg.SyncModeCmd(user_id=self.user_id,
            start_sync_mode=enabled,time_step=self.time_step_ms)).response

    def start(self):
        if self.started:
            raise RuntimeError("MORAI ROS SyncMode is already started")
        result = self._mode(True)
        if not result.result or int(result.time_step) <= 0:
            raise RuntimeError("MORAI SyncMode start failed: result={}, frame={}, time_step={}".format(
                result.result,result.frame,result.time_step))
        self.user_id = str(result.user_id or self.user_id)
        self.current_frame, self.started = int(result.frame), True
        return result

    def set_gear(self, gear):
        if not self.started or self.current_frame is None:
            raise RuntimeError("Call start() before set_gear()")
        if self.gear is None:
            raise RuntimeError("SyncMode gear service is not configured")
        result = self.gear(self.msg.SyncModeSetGear(gear=int(gear),frame=int(self.current_frame))).response
        if not result.result:
            raise RuntimeError("MORAI {} failed at frame={} gear={}".format(self.gear_name,self.current_frame,gear))
        return result

    def step(self, command, sensor_capture=False):
        if not self.started or self.current_frame is None:
            raise RuntimeError("Call start() before step()")
        frame = int(self.current_frame)
        request = self.msg.SyncModeCtrlCmd(frame=frame,command=command,sensor_capture=bool(sensor_capture))
        begun = time.monotonic()
        result = self.rpc["control"](request).response
        applied = time.monotonic()
        if not result.result:
            raise RuntimeError("MORAI {} failed at frame={}".format(self.names["control"],frame))
        tick = self.rpc["wait_for_tick"](self.msg.WaitForTick(user_id=self.user_id,frame=frame)).response
        finished = time.monotonic()
        if not tick.tick_status:
            raise RuntimeError("MORAI {} failed at frame={} (pause_status={})".format(
                self.names["wait_for_tick"],frame,tick.pause_status))
        response = int(tick.frame)
        if response != frame+1:
            raise RuntimeError("MORAI sync frame did not advance exactly once: request={}, response={}".format(frame,response))
        self.current_frame = response
        self.publisher.publish(tick.vehicle_status)
        return Step(frame,response,tick.vehicle_status,applied-begun,finished-applied,bool(sensor_capture))

    def stop(self):
        if not self.started:
            return None
        try:
            return self._mode(False)
        finally:
            self.started = False
