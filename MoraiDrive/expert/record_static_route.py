#!/usr/bin/env python3
"""Read-only MORAI ENU route recorder. No vehicle control is published."""

import argparse
import csv
import math
from pathlib import Path
import re
import socket
import sys
import threading
import time
from urllib.parse import urlparse
import os


DEFAULT_OUTPUT = (Path(__file__).resolve().parent / "src/expert/src/planning"
                  / "\uc815\uc801\uc7a5\uc560\ubb3c" / "\ud6c4\ubcf4\uc804\uc5ed\uacbd\ub85c")
FIELDS = ("x", "y", "z", "heading_deg", "yaw_rad", "s_m", "speed_mps",
          "source_stamp_s")


def sample_message(msg):
    values = (msg.position.x, msg.position.y, msg.position.z, msg.heading,
              msg.velocity.x, msg.header.stamp.to_sec())
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite ego telemetry; recording stopped.")
    return dict(zip(("x", "y", "z", "heading_deg", "speed_mps",
                     "source_stamp_s"), values), yaw_rad=math.radians(msg.heading))


def route_file(directory, name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Use an ASCII route name, e.g. straight or right_01.")
    return Path(directory) / (name + ".csv")


class RouteWriter:
    def __init__(self, stream, spacing=0.2):
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("Spacing must be positive and finite.")
        self.stream = stream
        self.writer = csv.DictWriter(stream, fieldnames=FIELDS)
        self.writer.writeheader()
        self.spacing = spacing
        self.previous = None
        self.saved = None
        self.count = 0
        self.distance = 0.0

    def _write(self, point):
        self.writer.writerow({key: point[key] for key in FIELDS})
        self.stream.flush()
        self.saved = point
        self.count += 1

    def add(self, point, received):
        point = dict(point)
        point["received"] = received
        if self.previous is not None:
            previous = self.previous
            dt = received - previous["received"]
            distance = math.hypot(point["x"] - previous["x"],
                                  point["y"] - previous["y"])
            if dt > 2.0 or dt < 0:
                raise ValueError("Telemetry gap; do not connect across missing data.")
            if point["source_stamp_s"] < previous["source_stamp_s"]:
                raise ValueError("Simulator time reset; start a new route.")
            if distance > max(3.0, dt * max(abs(point["speed_mps"]),
                                          abs(previous["speed_mps"])) + 1.0):
                raise ValueError("Position jump detected; start a new route.")
        # Accumulate only emitted chords to avoid inflating length while parked.
        if self.saved is None:
            point["s_m"] = 0.0
            self._write(point)
        else:
            distance = math.hypot(point["x"] - self.saved["x"],
                                  point["y"] - self.saved["y"])
            point["s_m"] = self.distance + distance
            if distance >= self.spacing:
                self.distance = point["s_m"]
                self._write(point)
        self.previous = point

    def finish(self):
        if self.previous is not None and self.saved is not None:
            if self.previous["s_m"] - self.saved["s_m"] > 0.01:
                self._write(self.previous)
        return self.count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", help="straight, right_01, right_02, ...")
    parser.add_argument("--check", action="store_true", help="Print ENU only; no file")
    parser.add_argument("--topic", default="/Ego_topic")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spacing", type=float, default=0.2)
    parser.add_argument("--start-x", type=float, default=-59.65)
    parser.add_argument("--start-y", type=float, default=-159.06)
    parser.add_argument("--start-radius", type=float, default=5.0)
    args = parser.parse_args()
    if not args.check and not args.name:
        parser.error("Route name required, e.g. right_01")
    if not all(math.isfinite(v) and v > 0 for v in (args.spacing, args.start_radius)):
        parser.error("spacing and start-radius must be positive and finite")
    if not all(math.isfinite(v) for v in (args.start_x, args.start_y)):
        parser.error("start coordinates must be finite")
    target = route_file(args.output_dir, args.name) if args.name else None
    if not args.check and target.exists():
        raise ValueError("File already exists; choose a new name: " + str(target))

    import rospy
    from morai_msgs.msg import EgoVehicleStatus

    master = urlparse(os.environ.get("ROS_MASTER_URI", "http://localhost:11311"))
    with socket.create_connection((master.hostname, master.port or 11311), timeout=3):
        pass
    rospy.init_node("static_route_recorder", anonymous=True, disable_signals=True)
    lock = threading.Lock()
    ready = threading.Event()
    state = {"latest": None, "writer": None, "error": None}

    def callback(msg):
        with lock:
            try:
                if state["error"] is not None:
                    return
                point, received = sample_message(msg), time.monotonic()
                state["latest"] = (point, received)
                if state["writer"] is not None:
                    state["writer"].add(point, received)
                ready.set()
            except Exception as exc:
                state["error"] = str(exc)
                ready.set()

    subscriber = rospy.Subscriber(args.topic, EgoVehicleStatus, callback, queue_size=1)
    try:
        if not ready.wait(10):
            raise ValueError("No /Ego_topic in 10 seconds. Check bridge and MORAI connection.")
        with lock:
            if state["error"]:
                raise ValueError(state["error"])
            point = state["latest"][0]
        delta = math.hypot(point["x"] - args.start_x, point["y"] - args.start_y)
        print("ENU x={x:.3f} y={y:.3f} z={z:.3f}, heading={heading_deg:.2f} deg".format(**point))
        print("Distance from expected start: {:.2f} m. Raw ENU; NO coordinate offset.".format(delta))
        if args.check:
            return
        print("Stop Expert. Drive manually in MORAI. Finish AFTER returning to the global route.")
        input("Position at the start, then press Enter to record (Ctrl+C to finish): ")
        target.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            if state["error"]:
                raise ValueError(state["error"])
            point, received = state["latest"]
            if time.monotonic() - received > 1.0:
                raise ValueError("Ego telemetry is stale. Resume simulator and retry.")
            if math.hypot(point["x"] - args.start_x, point["y"] - args.start_y) > args.start_radius:
                raise ValueError("Outside start radius. Reposition near expected start or set --start-x/--start-y.")
            stream = target.open("x", newline="", encoding="utf-8")
            writer = RouteWriter(stream, args.spacing)
            writer.add(point, received)
            state["writer"] = writer
        print("RECORDING: {}\nStart ENU: {:.3f}, {:.3f}, {:.3f}".format(
            target, point["x"], point["y"], point["z"]), flush=True)
        try:
            while not rospy.is_shutdown():
                time.sleep(0.1)
                with lock:
                    if state["error"]:
                        raise ValueError(state["error"])
                    if time.monotonic() - state["latest"][1] > 2.0:
                        raise ValueError("Telemetry stopped for 2 seconds; recording stopped.")
        except KeyboardInterrupt:
            pass
        finally:
            with lock:
                state["writer"] = None
                count = writer.finish()
                stream.close()
            print("Saved {} points: {}".format(count, target), flush=True)
            if count < 2:
                print("WARNING: Too few points; this is not a usable driving route.")
    finally:
        subscriber.unregister()
        rospy.signal_shutdown("recorder finished")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Cancelled before recording.")
    except Exception as exc:
        print("[FAIL] {} (Any saved route may be incomplete.)".format(exc), file=sys.stderr)
        sys.exit(1)
