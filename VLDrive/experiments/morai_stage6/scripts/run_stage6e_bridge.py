#!/usr/bin/env python3
"""Production bridge entrypoint; normally started by start_stage6e_bridge.py."""
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9090)
    args, remaining = parser.parse_known_args()
    import rosbridge_server.autobahn_websocket as ws
    from rosbridge_library.protocol import Protocol
    from aimdrive.ros_training.production_bridge import install
    install(ws, Protocol, args.port)
    entry = "/opt/ros/noetic/lib/rosbridge_server/rosbridge_websocket"
    sys.argv = [entry, "--port", str(args.port)] + remaining
    runpy.run_path(entry, run_name="__main__")


if __name__ == "__main__":
    main()
