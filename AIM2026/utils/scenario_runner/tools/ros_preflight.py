"""Keep an installed ROS bridge ready for scenario control, without a collector."""
import argparse
import importlib.util
import json
import os
import signal
import socket
import subprocess
import time

from scenario_runner.utils.environment import build_capture_environment


def has_control_subscriber(master):
    _, subscribers, _ = master.getSystemState()
    return bool(dict(subscribers).get("/ctrl_cmd", ()))


def listener_alive(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=.5):
            return True
    except OSError:
        return False


def _check_owned_bridge(process, phase):
    if process is not None and process.poll() is not None:
        raise RuntimeError("The owned ROS bridge exited " + phase)


def _control_ready(master):
    try:
        return has_control_subscriber(master)
    except Exception:
        return False


def keep_control_ready(master, set_param, process, port, timeout, stopped):
    """Preserve the original scenario-only master/control/keepalive lifecycle."""
    deadline = time.monotonic() + 40
    while not stopped:
        try:
            master.getParamNames()
            break
        except Exception:
            _check_owned_bridge(process, "before the ROS master was ready")
            if time.monotonic() >= deadline:
                raise TimeoutError("The ROS master was not ready within 40 seconds")
            time.sleep(1)
    if stopped:
        return
    # Apply host time before waiting for control: Expert timers must not freeze
    # when Simulator Network /clock is intentionally disconnected.
    set_param("/use_sim_time", False)
    print("[PASS] ROS master ready; /use_sim_time=false", flush=True)
    deadline = time.monotonic() + timeout
    while not stopped and not _control_ready(master):
        _check_owned_bridge(process, "while waiting for /ctrl_cmd")
        if time.monotonic() >= deadline:
            raise TimeoutError("No MORAI /ctrl_cmd subscriber within " + str(timeout) + " seconds")
        time.sleep(1)
    if stopped:
        return
    print("[PASS] MORAI /ctrl_cmd subscriber ready; no sensor collector started", flush=True)
    report_at = time.monotonic() + 30
    missing_count = 0
    while not stopped:
        _check_owned_bridge(process, "during scenario validation")
        if process is None and not listener_alive(port):
            raise RuntimeError("The existing ROS bridge listener exited")
        if _control_ready(master):
            missing_count = 0
        else:
            missing_count += 1
            if missing_count >= 5:
                raise RuntimeError("The MORAI /ctrl_cmd subscription was lost for 5 consecutive checks")
        if time.monotonic() >= report_at:
            print("[KEEPALIVE] ROS master, bridge and /ctrl_cmd ready", flush=True)
            report_at = time.monotonic() + 30
        time.sleep(1)


def stop_owned_bridge(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=35)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check installed ROS modules; do not open sockets")
    parser.add_argument("--scenario-only", action="store_true", help="Scenario bridge mode (also the default)")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--timeout", type=int, default=os.environ.get("MORAI_CONNECT_TIMEOUT_SECONDS", "600"),
                        help="Control connection wait, at least 30 seconds (default: MORAI_CONNECT_TIMEOUT_SECONDS or 600)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or args.timeout < 30:
        parser.error("port must be valid and timeout must be an integer of at least 30 seconds")
    missing = [name for name in ("rosgraph", "rospy", "rosbridge_server", "morai_msgs")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Missing installed ROS packages: " + ", ".join(missing))
    if args.check:
        print(json.dumps(dict(status="CHECKED", bridge_port=args.port,
                              collector_started=False, network_started=False)))
        return 0
    import rosgraph
    import rospy
    process = None
    stopped = []
    handlers = {sig: signal.signal(sig, lambda *_: stopped.append(True))
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if not listener_alive(args.port):
            process = subprocess.Popen(["roslaunch", "rosbridge_server", "rosbridge_websocket.launch",
                "port:=" + str(args.port), "address:=" + args.bind],
                env=build_capture_environment(), start_new_session=True)
        master = rosgraph.Master("/scenario_runner_preflight")
        keep_control_ready(master, rospy.set_param, process, args.port, args.timeout, stopped)
        return 0
    finally:
        try:
            stop_owned_bridge(process)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
