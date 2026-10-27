"""Scenario Runner's ROS synchronous tick owner and capture boundary.

Run in scenario mode before the existing SyncRuntimeBridge raises READY. The
owner consumes the existing Expert's /ctrl_cmd; it does not start another
Expert, sensor collector, ROS master or network bridge.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import signal
from threading import Event
import time
import uuid

from .capture import ManifestWriter
from .ros import RosTickClient
from .state import CaptureSchedule, ControlMailbox, capture_record


DEFAULTS = dict(mode="standalone",node_name="morai_sync_runtime",user_id="MoraiDriveSyncRuntime",
    time_step_ms=20,capture_enabled=True,capture_stride=5,
    capture=dict(journal_root=".runtime/sync_runtime",journal_queue_size=1024,scenario_id="full_loop",
                 expected_sensor_files=dict(camera_front="CAMERA_1/{frame}_Intensity.png",
                     camera_left="CAMERA_2/{frame}_Intensity.png",camera_right="CAMERA_3/{frame}_Intensity.png",
                     camera_rear="CAMERA_4/{frame}_Intensity.png",lidar="LIDAR_5/{frame}_Intensity.bin")),
    warn_after_frames=5,fail_safe_after_frames=25,
    fail_safe=dict(longl_cmd_type=1,accel=0.,brake=1.,front_steer=0.,rear_steer=0.),
    services=dict(mode="/SyncModeCmd",control="/SyncModeCtrlCmd",wait_for_tick="/SyncModeWaitForTick",set_gear="/SyncModeSetGear"),
    topics=dict(control="/ctrl_cmd",ego="/Ego_topic"),
    ros=dict(service_timeout_sec=10.,control_subscriber_queue_size=1,ego_publisher_queue_size=1),
    logging=dict(interval_sec=5.),standalone=dict(set_gear_drive=True,drive_gear=4),
    scenario=dict(ready_param="/morai_drive/scenario_ready",runtime_active_param="/morai_drive/sync_runtime_active",
                  wait_for_ready=True,require_fresh_ready_transition=True,ready_timeout_sec=0.,poll_interval_sec=.05,
                  scenario_id_param="/morai_drive/scenario_id"))


def load_config(path=None):
    if path is None:
        config = deepcopy(DEFAULTS)
    else:
        import yaml
        with open(path,encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
    required = ("time_step_ms","capture_enabled","capture_stride","capture","warn_after_frames",
                "fail_safe_after_frames","fail_safe","services","topics","ros","logging")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError("Missing config keys: " + ", ".join(missing))
    if int(config.get("capture_stride",0)) <= 0:
        raise ValueError("capture_stride must be > 0")
    if not dict(config["capture"]).get("expected_sensor_files"):
        raise ValueError("capture.expected_sensor_files must not be empty")
    return config


def wait_for_ready(ros, config, stop):
    scenario = config.get("scenario",{})
    if not bool(scenario.get("wait_for_ready",True)):
        ros.logwarn("[SyncRuntime] scenario mode READY wait is disabled by config")
        return
    name = str(scenario.get("ready_param","/morai_drive/scenario_ready"))
    fresh = bool(scenario.get("require_fresh_ready_transition",True))
    seconds = float(scenario.get("ready_timeout_sec",0.))
    deadline = time.monotonic()+seconds if seconds > 0 else None
    seen_false = not bool(ros.get_param(name,False))
    ros.loginfo("[SyncRuntime] waiting for scenario READY: %s (fresh_transition=%s)",name,fresh)
    while not stop.is_set() and not ros.is_shutdown():
        ready = bool(ros.get_param(name,False))
        seen_false |= not ready
        if ready and (seen_false or not fresh):
            ros.loginfo("[SyncRuntime] scenario READY received")
            return
        if deadline is not None and time.monotonic() >= deadline:
            raise RuntimeError("Timed out waiting for scenario READY: " + name)
        time.sleep(max(.01,float(scenario.get("poll_interval_sec",.05))))
    raise KeyboardInterrupt


def failsafe_factory(command_type, settings):
    settings = dict(settings)
    def command():
        result = command_type()
        result.longlCmdType = int(settings.get("longl_cmd_type",1))
        for name in ("accel","brake","front_steer","rear_steer"):
            setattr(result,name,float(settings.get(name,1. if name=="brake" else 0.)))
        result.velocity = result.acceleration = 0.
        return result
    return command


class TickDiagnostics:
    def __init__(self, interval):
        self.interval, self.total_ticks = max(.1,float(interval)), 0
        self._reset()

    def _reset(self, now=None):
        self.begun, self.count = time.monotonic() if now is None else now, 0
        self.control = self.wait = self.cycle = self.maximum = 0.

    def record(self, step, seconds):
        self.total_ticks += 1
        self.count += 1
        self.control += step.control_rpc_sec
        self.wait += step.wait_for_tick_rpc_sec
        self.cycle += seconds
        self.maximum = max(self.maximum,seconds)

    def report(self, ros, step, decision):
        now = time.monotonic()
        elapsed = now-self.begun
        if self.total_ticks > 1 and elapsed < self.interval:
            return
        count = max(1,self.count)
        ros.loginfo("[SyncRuntime] current_frame=%d request_frame=%d response_frame=%d "
                    "ticks=%d control=%s seq=%d age_frames=%d sensor_capture=%s "
                    "rpc_ctrl=%.2fms avg rpc_wait=%.2fms avg "
                    "cycle=%.2fms avg/%.2fms max tick_hz=%.2f",
                    step.response_frame,step.request_frame,step.response_frame,self.total_ticks,
                    decision.mode,decision.source_sequence,decision.age_frames,str(step.sensor_capture).lower(),
                    self.control*1000/count,self.wait*1000/count,self.cycle*1000/count,self.maximum*1000,
                    self.count/elapsed if elapsed>0 else 0.)
        self._reset(now)


def log_decision(ros, decision):
    if decision.warning_threshold_reached:
        ros.logwarn("[SyncRuntime] Expert control stale: seq=%d age_frames=%d",decision.source_sequence,decision.age_frames)
    if decision.failsafe_activated:
        ros.logwarn("[SyncRuntime] fail-safe active: seq=%d age_frames=%d; ticks continue",decision.source_sequence,decision.age_frames)
    if decision.recovered_from_failsafe:
        ros.loginfo("[SyncRuntime] fresh Expert control recovered: seq=%d",decision.source_sequence)


class TickOwner:
    """Own service progression, bounded metadata and ordered shutdown."""
    def __init__(self, config, mode=None, client_factory=RosTickClient, writer_factory=ManifestWriter,
                 session_factory=None):
        self.config, self.mode = config, mode or str(config.get("mode","standalone"))
        if self.mode not in ("standalone","scenario"):
            raise ValueError("Unsupported mode: " + self.mode)
        self.client_factory, self.writer_factory = client_factory, writer_factory
        self.session_factory = session_factory or (lambda:time.strftime("sync-%Y%m%d-%H%M%S-")+uuid.uuid4().hex[:8])

    def run(self, ros, stop=None, max_ticks=0):
        cfg, stop = self.config, stop if stop is not None else Event()
        scenario, transport, capture = cfg.get("scenario",{}), cfg["ros"], dict(cfg["capture"])
        active = str(scenario.get("runtime_active_param","/morai_drive/sync_runtime_active"))
        client = subscriber = writer = None
        reason = "Sync Runtime stopped"
        try:
            ros.set_param(active,False)
            if self.mode=="scenario":
                wait_for_ready(ros,cfg,stop)
            client = self.client_factory(user_id=str(cfg.get("user_id","MoraiDriveSyncRuntime")),
                time_step_ms=int(cfg["time_step_ms"]),service_timeout_sec=float(transport["service_timeout_sec"]),
                services=dict(cfg["services"]),ego_topic=str(cfg["topics"]["ego"]),
                ego_publisher_queue_size=int(transport.get("ego_publisher_queue_size",1)),
                gear_service_name=str(cfg["services"].get("set_gear","/SyncModeSetGear")) if self.mode=="standalone" else "")
            mailbox = ControlMailbox(failsafe_factory(client.CtrlCmd,cfg["fail_safe"]),
                                     int(cfg["warn_after_frames"]),int(cfg["fail_safe_after_frames"]))
            subscriber = ros.Subscriber(str(cfg["topics"]["control"]),client.CtrlCmd,mailbox.update,
                queue_size=max(1,int(transport.get("control_subscriber_queue_size",1))),tcp_nodelay=True)
            started = client.start()
            if self.mode=="standalone" and bool(cfg.get("standalone",{}).get("set_gear_drive",True)):
                gear = int(cfg.get("standalone",{}).get("drive_gear",4))
                client.set_gear(gear)
                ros.loginfo("[SyncRuntime] standalone Ego gear set: D(%d) frame=%d",gear,client.current_frame)
            enabled, stride, session = bool(cfg["capture_enabled"]), int(cfg["capture_stride"]), self.session_factory()
            scenario_id = str(ros.get_param(str(scenario.get("scenario_id_param","/morai_drive/scenario_id")),"")
                              or capture.get("scenario_id",self.mode))
            journal = str((Path(str(capture.get("journal_root",".runtime/sync_runtime")))/session/"capture_manifest.jsonl").absolute())
            if enabled:
                writer = self.writer_factory(journal,queue_size=int(capture.get("journal_queue_size",1024)))
            ros.set_param(active,True)
            ros.loginfo("[SyncRuntime] Sync Mode ON: user_id=%s start_frame=%d time_step=%dms capture_enabled=%s stride=%d mode=%s",
                        client.user_id,client.current_frame,int(started.time_step),enabled,stride,self.mode)
            if writer is not None:
                ros.loginfo("[SyncRuntime] capture manifest: %s",journal)
            diagnostics, schedule, ego = TickDiagnostics(cfg["logging"]["interval_sec"]), CaptureSchedule(enabled,stride), None
            while not stop.is_set() and not ros.is_shutdown():
                if self.mode=="scenario" and not ros.get_param(scenario.get("ready_param","/morai_drive/scenario_ready"),False):
                    ros.loginfo("[SyncRuntime] scenario READY cleared; stopping")
                    break
                decision = mailbox.select()
                log_decision(ros,decision)
                take = bool(ego is not None and schedule.should_capture(client.current_frame))
                begun = time.monotonic()
                step = client.step(decision.command,sensor_capture=take)
                elapsed = time.monotonic()-begun
                if ego is None:
                    schedule.observe_bootstrap_response(step.response_frame)
                elif take and not writer.submit(capture_record(session,scenario_id,stride,ego,step,decision,capture["expected_sensor_files"])):
                    ros.logwarn_throttle(5.,"[SyncRuntime] capture manifest queue unavailable; metadata dropped=%d (tick continues)",writer.dropped)
                ego = step.vehicle_status
                diagnostics.record(step,elapsed)
                diagnostics.report(ros,step,decision)
                if max_ticks>0 and diagnostics.total_ticks>=max_ticks:
                    reason = "max tick count reached"
                    break
        except KeyboardInterrupt:
            reason = "keyboard interrupt"
        finally:
            if subscriber is not None:
                subscriber.unregister()
            if client is not None and client.started:
                try:
                    result = client.stop()
                    ros.loginfo("[SyncRuntime] Sync Mode OFF: result=%s frame=%s",getattr(result,"result",None),getattr(result,"frame",None))
                except BaseException as error:
                    ros.logerr("[SyncRuntime] Sync Mode OFF failed: %s",error)
            if writer is not None:
                try:
                    writer.close()
                    ros.loginfo("[SyncRuntime] capture journal closed: written=%d dropped=%d",writer.written,writer.dropped)
                except BaseException as error:
                    ros.logerr("[SyncRuntime] capture journal close failed: %s",error)
            try:
                ros.set_param(active,False)
            except BaseException as error:
                ros.logerr("[SyncRuntime] active-state cleanup failed: %s",error)
            if not ros.is_shutdown():
                ros.signal_shutdown(reason)


def build_parser():
    parser = argparse.ArgumentParser(description="Scenario Runner MORAI ROS SyncMode tick owner")
    parser.add_argument("--mode",choices=("standalone","scenario"),default=None)
    parser.add_argument("--config",help="Existing SyncMode YAML contract; omitted uses the built-in equivalent defaults")
    parser.add_argument("--time-step-ms",type=int)
    parser.add_argument("--user-id")
    parser.add_argument("--log-interval-sec",type=float)
    parser.add_argument("--max-ticks",type=int,default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    for name,target in (("time_step_ms",config),("user_id",config),("log_interval_sec",config["logging"])):
        value = getattr(args,name)
        if value is not None and (name!="user_id" or value):
            target["interval_sec" if name=="log_interval_sec" else name] = value
    from scenario_runner.project_paths import EXPERT_ROOT
    from ..expert_process import configure_ros_message_path
    configure_ros_message_path(EXPERT_ROOT)
    import rospy
    rospy.init_node(str(config.get("node_name","morai_sync_runtime")),anonymous=False,disable_signals=True)
    stop, interrupts = Event(), []
    def request_stop(signum,_frame):
        interrupts.append(signum)
        stop.set()
        if signum==signal.SIGINT and len(interrupts)>1:
            raise KeyboardInterrupt
    for number in (signal.SIGINT,signal.SIGTERM):
        signal.signal(number,request_stop)
    TickOwner(config,args.mode).run(rospy,stop,args.max_ticks)
