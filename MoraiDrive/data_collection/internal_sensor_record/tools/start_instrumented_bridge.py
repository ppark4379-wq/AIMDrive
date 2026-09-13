#!/usr/bin/env python3
"""Start one instrumented rosbridge and a local ROS master when required."""
import argparse
import datetime
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse
import xmlrpc.client


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from network_config import ros_master_uri, runtime_environment
from runtime_paths import CATKIN_ROOT, ROS_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master",
        default=ros_master_uri(),
    )
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument(
        "--ros-setup",
        default=str(CATKIN_ROOT / "devel/setup.bash"),
    )
    parser.add_argument("--drop-camera-ingress", action="store_true")
    parser.add_argument("--log-root", type=Path)
    args = parser.parse_args()

    if os.environ.get("MORAI_BRIDGE_ROS_BOOTSTRAPPED") != "1":
        setup = Path(args.ros_setup).expanduser().resolve()
        if not setup.is_file():
            parser.error("Missing ROS/catkin setup: " + str(setup))
        env = dict(
            os.environ,
            **runtime_environment(),
            MORAI_BRIDGE_ROS_BOOTSTRAPPED="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        os.execvpe(
            "bash",
            [
                "bash", "-c",
                'source "$1"; shift; exec /usr/bin/python3 "$@"',
                "morai_bridge", str(setup), str(Path(__file__).resolve()),
            ] + sys.argv[1:],
            env,
        )

    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            check.bind(("0.0.0.0", args.port))
        except OSError:
            parser.error(
                "Bridge port %d is already in use; do not launch a second bridge"
                % args.port
            )

    from ssd_storage import SSD_ROOT, require_ssd
    log_root = args.log_root or (
        SSD_ROOT / "sensor_record_auto_v1" / "bridge_logs"
    )
    require_ssd(log_root, check_volume=True, reserve_bytes=0)
    output = log_root / datetime.datetime.now().strftime(
        "bridge_%Y%m%d_%H%M%S_%f"
    )
    output.mkdir(parents=True, exist_ok=False)

    env = os.environ.copy()
    env.update(
        ROS_MASTER_URI=args.master,
        ROS_HOME=str(output / "ros_home"),
        ROS_LOG_DIR=str(output / "ros_logs"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    children = []
    log = None
    socket.setdefaulttimeout(3)

    def master_ready():
        try:
            return xmlrpc.client.ServerProxy(args.master).getPid(
                "/morai_internal_bridge_launcher"
            )[0] == 1
        except (OSError, xmlrpc.client.Error):
            return False

    try:
        if not master_ready():
            url = urlparse(args.master)
            try:
                with socket.socket() as check:
                    check.bind((url.hostname, 0))
            except OSError as exc:
                raise RuntimeError(
                    "Configured remote ROS master is unavailable: " + args.master
                ) from exc
            log = (output / "roscore.log").open("x")
            core = subprocess.Popen(
                [
                    sys.executable,
                    str(ROS_ROOT / "bin/roscore"),
                    "-p", str(url.port or 11311),
                ],
                cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
            )
            children.append(core)
            deadline = time.monotonic() + 20
            while not master_ready():
                if core.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(
                        "ROS master did not start; see " + str(output / "roscore.log")
                    )
                time.sleep(0.2)

        command = [
            sys.executable, "-u",
            str(ROOT / "tools/run_instrumented_bridge.py"),
            "--port", str(args.port),
            "__name:=morai_internal_sensor_record_bridge",
        ]
        if args.drop_camera_ingress:
            command.append("--drop-camera-ingress")
        bridge = subprocess.Popen(command, cwd=str(ROOT), env=env)
        children.append(bridge)
        print("Evidence:", output, flush=True)
        print(
            "Connect MORAI vehicle information and sensors to port",
            args.port,
            flush=True,
        )
        return bridge.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=10)
        if log:
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())
