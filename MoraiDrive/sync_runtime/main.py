#!/usr/bin/env python3
"""Standalone tick owner for MORAI ROS built-in Synchronous Mode."""

import argparse
import os
import signal
import sys
import threading
import time
import uuid

import yaml


if __package__ in (None, ""):
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from sync_runtime.control_buffer import LatestControlBuffer
    from sync_runtime.capture_journal import CaptureJournal
    from sync_runtime.frame_record import (
        CaptureFrameRecord,
        NativeFrameCaptureScheduler,
        control_dict,
        ego_state_dict,
    )
    from sync_runtime.sync_client import MoraiRosSyncClient
else:
    from .control_buffer import LatestControlBuffer
    from .capture_journal import CaptureJournal
    from .frame_record import (
        CaptureFrameRecord,
        NativeFrameCaptureScheduler,
        control_dict,
        ego_state_dict,
    )
    from .sync_client import MoraiRosSyncClient


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


class RollingTickMetrics:
    def __init__(self, interval_sec):
        self.interval_sec = max(0.1, float(interval_sec))
        self.total_ticks = 0
        self._window_started = time.monotonic()
        self._window_ticks = 0
        self._control_rpc_total = 0.0
        self._wait_rpc_total = 0.0
        self._cycle_total = 0.0
        self._cycle_max = 0.0

    def record(self, step_result, cycle_sec):
        self.total_ticks += 1
        self._window_ticks += 1
        self._control_rpc_total += step_result.control_rpc_sec
        self._wait_rpc_total += step_result.wait_for_tick_rpc_sec
        self._cycle_total += cycle_sec
        self._cycle_max = max(self._cycle_max, cycle_sec)

    def report_if_due(self, rospy, step_result, decision, force=False):
        now = time.monotonic()
        elapsed = now - self._window_started
        if not force and self.total_ticks > 1 and elapsed < self.interval_sec:
            return

        count = max(1, self._window_ticks)
        hz = self._window_ticks / elapsed if elapsed > 0.0 else 0.0
        rospy.loginfo(
            "[SyncRuntime] current_frame=%d request_frame=%d response_frame=%d "
            "ticks=%d control=%s seq=%d age_frames=%d sensor_capture=%s "
            "rpc_ctrl=%.2fms avg rpc_wait=%.2fms avg "
            "cycle=%.2fms avg/%.2fms max tick_hz=%.2f",
            step_result.response_frame,
            step_result.request_frame,
            step_result.response_frame,
            self.total_ticks,
            decision.mode,
            decision.source_sequence,
            decision.age_frames,
            str(step_result.sensor_capture).lower(),
            self._control_rpc_total * 1000.0 / count,
            self._wait_rpc_total * 1000.0 / count,
            self._cycle_total * 1000.0 / count,
            self._cycle_max * 1000.0,
            hz,
        )
        self._window_started = now
        self._window_ticks = 0
        self._control_rpc_total = 0.0
        self._wait_rpc_total = 0.0
        self._cycle_total = 0.0
        self._cycle_max = 0.0


def _load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    required = (
        "time_step_ms",
        "capture_enabled",
        "capture_stride",
        "capture",
        "warn_after_frames",
        "fail_safe_after_frames",
        "fail_safe",
        "services",
        "topics",
        "ros",
        "logging",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("Missing config keys: {}".format(", ".join(missing)))
    capture = dict(config["capture"])
    if int(config.get("capture_stride", 0)) <= 0:
        raise ValueError("capture_stride must be > 0")
    if not capture.get("expected_sensor_files"):
        raise ValueError("capture.expected_sensor_files must not be empty")
    return config


def _new_session_id():
    return "sync-{}-{}".format(
        time.strftime("%Y%m%d-%H%M%S"), uuid.uuid4().hex[:8]
    )


def _expected_sensor_files(patterns, frame):
    return {
        str(name): str(pattern).format(frame=int(frame))
        for name, pattern in patterns.items()
    }


def _make_failsafe_factory(ctrl_cmd_type, config):
    settings = dict(config)

    def make_command():
        command = ctrl_cmd_type()
        command.longlCmdType = int(settings.get("longl_cmd_type", 1))
        command.accel = float(settings.get("accel", 0.0))
        command.brake = float(settings.get("brake", 1.0))
        command.front_steer = float(settings.get("front_steer", 0.0))
        command.rear_steer = float(settings.get("rear_steer", 0.0))
        command.velocity = 0.0
        command.acceleration = 0.0
        return command

    return make_command


def _wait_for_scenario_ready(rospy, config, stop_requested):
    scenario = dict(config.get("scenario", {}))
    if not bool(scenario.get("wait_for_ready", True)):
        rospy.logwarn(
            "[SyncRuntime] scenario mode READY wait is disabled by config"
        )
        return

    ready_param = str(
        scenario.get("ready_param", "/morai_drive/scenario_ready")
    )
    timeout_sec = float(scenario.get("ready_timeout_sec", 0.0))
    require_fresh_transition = bool(
        scenario.get("require_fresh_ready_transition", True)
    )
    poll_interval_sec = max(
        0.01, float(scenario.get("poll_interval_sec", 0.05))
    )
    deadline = None if timeout_sec <= 0.0 else time.monotonic() + timeout_sec
    ready_observed_false = not bool(rospy.get_param(ready_param, False))
    rospy.loginfo(
        "[SyncRuntime] waiting for scenario READY: %s (fresh_transition=%s)",
        ready_param,
        require_fresh_transition,
    )
    while not stop_requested.is_set() and not rospy.is_shutdown():
        ready = bool(rospy.get_param(ready_param, False))
        if not ready:
            ready_observed_false = True
        if ready and (ready_observed_false or not require_fresh_transition):
            rospy.loginfo("[SyncRuntime] scenario READY received")
            return
        if deadline is not None and time.monotonic() >= deadline:
            raise RuntimeError(
                "Timed out waiting for scenario READY: {}".format(ready_param)
            )
        time.sleep(poll_interval_sec)
    raise KeyboardInterrupt


def _install_signal_handlers(stop_requested):
    signal_count = {"count": 0}

    def request_stop(signum, _frame):
        signal_count["count"] += 1
        stop_requested.set()
        # A second interrupt can break a stuck ROS service call. The finally
        # block still attempts SyncMode OFF when the ROS connection is usable.
        if signum == signal.SIGINT and signal_count["count"] > 1:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def _log_control_transition(rospy, decision):
    if decision.warning_threshold_reached:
        rospy.logwarn(
            "[SyncRuntime] Expert control stale: seq=%d age_frames=%d",
            decision.source_sequence,
            decision.age_frames,
        )
    if decision.failsafe_activated:
        rospy.logwarn(
            "[SyncRuntime] fail-safe active: seq=%d age_frames=%d; ticks continue",
            decision.source_sequence,
            decision.age_frames,
        )
    if decision.recovered_from_failsafe:
        rospy.loginfo(
            "[SyncRuntime] fresh Expert control recovered: seq=%d",
            decision.source_sequence,
        )


def run(args):
    config = _load_config(os.path.abspath(args.config))
    mode = args.mode or str(config.get("mode", "standalone"))
    if mode not in ("standalone", "scenario"):
        raise ValueError("Unsupported mode: {}".format(mode))
    if args.time_step_ms is not None:
        config["time_step_ms"] = int(args.time_step_ms)
    if args.user_id:
        config["user_id"] = args.user_id
    if args.log_interval_sec is not None:
        config["logging"]["interval_sec"] = float(args.log_interval_sec)

    import rospy

    rospy.init_node(
        str(config.get("node_name", "morai_sync_runtime")),
        anonymous=False,
        disable_signals=True,
    )
    stop_requested = threading.Event()
    _install_signal_handlers(stop_requested)

    sync_client = None
    control_subscriber = None
    capture_journal = None
    runtime_active_param = str(
        config.get("scenario", {}).get(
            "runtime_active_param", "/morai_drive/sync_runtime_active"
        )
    )
    shutdown_reason = "Sync Runtime stopped"
    try:
        rospy.set_param(runtime_active_param, False)
        if mode == "scenario":
            _wait_for_scenario_ready(rospy, config, stop_requested)

        sync_client = MoraiRosSyncClient(
            user_id=str(config.get("user_id", "MoraiDriveSyncRuntime")),
            time_step_ms=int(config["time_step_ms"]),
            service_timeout_sec=float(config["ros"]["service_timeout_sec"]),
            services=dict(config["services"]),
            ego_topic=str(config["topics"]["ego"]),
            ego_publisher_queue_size=int(
                config["ros"].get("ego_publisher_queue_size", 1)
            ),
            gear_service_name=(
                str(config["services"].get("set_gear", "/SyncModeSetGear"))
                if mode == "standalone" else ""
            ),
        )
        control_buffer = LatestControlBuffer(
            _make_failsafe_factory(sync_client.CtrlCmd, config["fail_safe"]),
            warn_after_frames=int(config["warn_after_frames"]),
            fail_safe_after_frames=int(config["fail_safe_after_frames"]),
        )
        control_subscriber = rospy.Subscriber(
            str(config["topics"]["control"]),
            sync_client.CtrlCmd,
            control_buffer.update,
            queue_size=max(
                1, int(config["ros"].get("control_subscriber_queue_size", 1))
            ),
            tcp_nodelay=True,
        )

        start_result = sync_client.start()
        if mode == "standalone" and bool(
            config.get("standalone", {}).get("set_gear_drive", True)
        ):
            drive_gear = int(config.get("standalone", {}).get("drive_gear", 4))
            sync_client.set_gear(drive_gear)
            rospy.loginfo(
                "[SyncRuntime] standalone Ego gear set: D(%d) frame=%d",
                drive_gear,
                sync_client.current_frame,
            )
        capture_config = dict(config["capture"])
        capture_enabled = bool(config["capture_enabled"])
        capture_stride = int(config["capture_stride"])
        scenario_id_param = str(
            config.get("scenario", {}).get(
                "scenario_id_param", "/morai_drive/scenario_id"
            )
        )
        scenario_id = str(
            rospy.get_param(scenario_id_param, "")
            or capture_config.get("scenario_id", mode)
        )
        session_id = _new_session_id()
        journal_path = os.path.abspath(
            os.path.join(
                str(capture_config.get("journal_root", ".runtime/sync_runtime")),
                session_id,
                "capture_manifest.jsonl",
            )
        )
        if capture_enabled:
            capture_journal = CaptureJournal(
                journal_path,
                queue_size=int(capture_config.get("journal_queue_size", 1024)),
            )
        rospy.set_param(runtime_active_param, True)
        rospy.loginfo(
            "[SyncRuntime] Sync Mode ON: user_id=%s start_frame=%d "
            "time_step=%dms capture_enabled=%s stride=%d mode=%s",
            sync_client.user_id,
            sync_client.current_frame,
            int(start_result.time_step),
            capture_enabled,
            capture_stride,
            mode,
        )
        if capture_journal is not None:
            rospy.loginfo("[SyncRuntime] capture manifest: %s", journal_path)
        metrics = RollingTickMetrics(config["logging"]["interval_sec"])
        # SyncMode start gives a frame number but no authoritative Ego for it.
        # The first step is therefore an unconditional non-capture bootstrap.
        ego_for_current_frame = None
        capture_scheduler = NativeFrameCaptureScheduler(
            capture_enabled, capture_stride
        )

        while not stop_requested.is_set() and not rospy.is_shutdown():
            if mode == "scenario" and not rospy.get_param(
                config.get("scenario", {}).get("ready_param", "/morai_drive/scenario_ready"), False
            ):
                rospy.loginfo("[SyncRuntime] scenario READY cleared; stopping")
                break
            decision = control_buffer.select()
            _log_control_transition(rospy, decision)
            request_frame = int(sync_client.current_frame)
            sensor_capture = bool(
                ego_for_current_frame is not None
                and capture_scheduler.should_capture(request_frame)
            )
            cycle_started = time.monotonic()
            step_result = sync_client.step(
                decision.command, sensor_capture=sensor_capture
            )
            cycle_sec = time.monotonic() - cycle_started
            if ego_for_current_frame is None:
                capture_scheduler.observe_bootstrap_response(
                    step_result.response_frame
                )
            elif sensor_capture:
                record = CaptureFrameRecord(
                    schema_version=1,
                    session_id=session_id,
                    scenario_id=scenario_id,
                    native_sync_frame=step_result.request_frame,
                    request_frame=step_result.request_frame,
                    response_frame=step_result.response_frame,
                    capture_requested=True,
                    capture_stride=capture_stride,
                    ego_state_for_frame=ego_state_dict(ego_for_current_frame),
                    ego_state_source_frame=step_result.request_frame,
                    next_ego_state=ego_state_dict(step_result.vehicle_status),
                    next_ego_state_source_frame=step_result.response_frame,
                    applied_control=control_dict(decision.command),
                    control_source_sequence=decision.source_sequence,
                    control_reused=decision.mode == "reused",
                    control_age_frames=decision.age_frames,
                    control_mode=decision.mode,
                    expected_sensor_files=_expected_sensor_files(
                        capture_config["expected_sensor_files"],
                        step_result.request_frame,
                    ),
                )
                if not capture_journal.submit(record):
                    rospy.logwarn_throttle(
                        5.0,
                        "[SyncRuntime] capture manifest queue unavailable; "
                        "metadata dropped=%d (tick continues)",
                        capture_journal.dropped,
                    )
            ego_for_current_frame = step_result.vehicle_status
            metrics.record(step_result, cycle_sec)
            metrics.report_if_due(
                rospy,
                step_result,
                decision,
                force=metrics.total_ticks == 1,
            )
            if args.max_ticks > 0 and metrics.total_ticks >= args.max_ticks:
                shutdown_reason = "max tick count reached"
                break
    except KeyboardInterrupt:
        shutdown_reason = "keyboard interrupt"
    finally:
        if control_subscriber is not None:
            control_subscriber.unregister()
        if sync_client is not None and sync_client.started:
            try:
                stop_result = sync_client.stop()
                rospy.loginfo(
                    "[SyncRuntime] Sync Mode OFF: result=%s frame=%s",
                    getattr(stop_result, "result", None),
                    getattr(stop_result, "frame", None),
                )
            except BaseException as error:
                rospy.logerr("[SyncRuntime] Sync Mode OFF failed: %s", error)
        if capture_journal is not None:
            try:
                capture_journal.close()
                rospy.loginfo(
                    "[SyncRuntime] capture journal closed: written=%d dropped=%d",
                    capture_journal.written,
                    capture_journal.dropped,
                )
            except BaseException as error:
                rospy.logerr("[SyncRuntime] capture journal close failed: %s", error)
        try:
            rospy.set_param(runtime_active_param, False)
        except BaseException as error:
            rospy.logerr("[SyncRuntime] active-state cleanup failed: %s", error)
        if not rospy.is_shutdown():
            rospy.signal_shutdown(shutdown_reason)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Independent MORAI ROS SyncMode tick runtime"
    )
    parser.add_argument(
        "--mode",
        choices=("standalone", "scenario"),
        default=None,
        help="standalone starts immediately; scenario waits for configured READY",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--time-step-ms", type=int)
    parser.add_argument("--user-id")
    parser.add_argument("--log-interval-sec", type=float)
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=0,
        help="stop cleanly after N ticks; 0 runs until interrupted",
    )
    return parser.parse_args(argv)


def main(argv=None):
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
