#!/usr/bin/env python3
"""Entrypoint for the binary-JSON-compatible MORAI rosbridge."""
import argparse
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime_paths import ROS_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument(
        "--drop-camera-ingress", action="store_true",
        help=("Keep MORAI cameras connected for Sensor Record, but discard "
              "duplicate ROS image publishes before JSON deserialization"),
    )
    args, remaining = parser.parse_known_args()

    import rosbridge_server.autobahn_websocket as websocket
    from rosbridge_library.protocol import Protocol
    from bridge.production_bridge import install

    install(
        websocket,
        Protocol,
        args.port,
        drop_camera_ingress=args.drop_camera_ingress,
        drop_lidar_ingress=False,
    )
    entry = str(ROS_ROOT / "lib/rosbridge_server/rosbridge_websocket")
    sys.argv = [entry, "--port", str(args.port)] + remaining
    runpy.run_path(entry, run_name="__main__")


if __name__ == "__main__":
    main()
