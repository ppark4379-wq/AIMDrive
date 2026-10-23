#!/usr/bin/env python3
"""Record MORAI ENU routes for Jamming. No vehicle control is published."""

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
import json


EXPERT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERT_ROOT / "src"))
from expert.src.planning.jamming.section import (
    JammingSection, next_route_name, register_route, ROOT as JAMMING_ROOT, ROUTE_DIRECTORY,
)
from expert.src.config.paths import STATIC_OBSTACLE_PATHS

DEFAULT_OUTPUT = JAMMING_ROOT / ROUTE_DIRECTORY
STATIC_OUTPUT = STATIC_OBSTACLE_PATHS / ROUTE_DIRECTORY
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


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", help="Route name; default: next free jamming_01, jamming_02, ...")
    parser.add_argument("--section", choices=("jamming", "static"), default="jamming")
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check", action="store_true", help="Show ENU, then Enter starts Jamming recording")
    checks.add_argument("--check-only", action="store_true", help="Print ENU and exit without recording")
    parser.add_argument("--topic", default="/Ego_topic")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--spacing", type=float)
    parser.add_argument("--start-x", type=float, help="Static section only; Jamming uses config/jamming_section.json")
    parser.add_argument("--start-y", type=float, help="Static section only; Jamming uses config/jamming_section.json")
    parser.add_argument("--start-radius", type=float, help="Static section only; Jamming uses a transverse start line")
    args = parser.parse_args(argv)
    args.jamming_section = JammingSection() if args.section == "jamming" else None
    section = args.jamming_section
    if section and any(getattr(args, key) is not None for key in ("start_x", "start_y", "start_radius")):
        parser.error("Configure the Jamming start line in config/jamming_section.json; --start-* overrides are for --section static")
    defaults = dict(output_dir=section.output_dir if section else STATIC_OUTPUT,
        spacing=section.spacing_m if section else .2,
        start_x=section.start[0] if section else -59.65,
        start_y=section.start[1] if section else -159.06,
        start_radius=None if section else 5.)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    # Preserve the old coordinate-only behavior when explicitly using static.
    args.check_only = args.check_only or (args.section == "static" and args.check)
    if not args.check_only and not args.name:
        args.name = next_route_name(args.output_dir, "jamming" if section else "static")
    if not all(math.isfinite(v) and v > 0 for v in
               ((args.spacing,) if section else (args.spacing, args.start_radius))):
        parser.error("spacing and start-radius must be positive and finite")
    if not all(math.isfinite(v) for v in (args.start_x, args.start_y)):
        parser.error("start coordinates must be finite")
    return args


def validate_recording_start(point, args):
    if args.jamming_section is not None:
        args.jamming_section.validate_start((point["x"], point["y"]))
    elif math.hypot(point["x"]-args.start_x, point["y"]-args.start_y) > args.start_radius:
        raise ValueError("Outside start radius. Reposition near expected start or set --start-x/--start-y.")


def finish_recording(target, args, error=None):
    """Incomplete recordings remain inspectable but never enter routes.json."""
    registered = False
    section = args.jamming_section
    if error is None and section is not None and target.parent.resolve() == section.output_dir.resolve():
        try:
            register_route(target, section)
            registered = True
        except (OSError, ValueError, KeyError) as exc:
            error = str(exc)
    if section is not None:
        status_path = target.with_suffix(".status.json")
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        status.update(ready_for_planner=registered, error=error,
                      source_frame="ENU", start_enu=section.start, end_enu=section.end)
        status_path.write_text(
            json.dumps(status, indent=2)+"\n", encoding="utf-8")
    if registered:
        print("Registered in data/recorded_paths/jamming/routes.json. Restart Expert after finishing all recordings.")
    elif section is not None:
        print("NOT registered for avoidance: " + (error or "custom output directory"))
    return registered


def main():
    args = parse_arguments()
    target = route_file(args.output_dir, args.name) if not args.check_only else None
    if target is not None and target.exists():
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
        print("ENU x={x:.3f} y={y:.3f} z={z:.3f}, heading={heading_deg:.2f} deg".format(**point))
        if args.jamming_section is not None:
            section = args.jamming_section
            lateral, longitudinal = section.to_local((point["x"], point["y"]))
            print("Start line: forward={:+.2f} m (allowed -{:.1f} to +{:.1f}), right={:+.2f} m (allowed +/-{:.1f}).".format(
                longitudinal, section.approach_distance_m, section.start_longitudinal_tolerance_m,
                lateral, section.start_lateral_half_width_m))
        else:
            delta = math.hypot(point["x"] - args.start_x, point["y"] - args.start_y)
            print("Distance from expected start: {:.2f} m. Raw ENU; NO coordinate offset.".format(delta))
        if args.check_only:
            return
        print("Stop Expert. Drive manually in MORAI.")
        print("Output: " + str(target))
        if args.jamming_section is not None:
            section = args.jamming_section
            print("Jamming end line through ENU: {}. Finish within +/-{:.1f} m forward, +/-{:.1f} m sideways.".format(
                section.end, section.end_longitudinal_tolerance_m, section.activation_half_width_m))
            print("Lateral endings are allowed; the planner validates the connection back to the global route.")
        else:
            print("Finish AFTER returning to the global route.")
        input("Position at the start, then press Enter to record (Ctrl+C to finish): ")
        target.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            if state["error"]:
                raise ValueError(state["error"])
            point, received = state["latest"]
            if time.monotonic() - received > 1.0:
                raise ValueError("Ego telemetry is stale. Resume simulator and retry.")
            validate_recording_start(point, args)
            stream = target.open("x", newline="", encoding="utf-8")
            writer = RouteWriter(stream, args.spacing)
            writer.add(point, received)
            state["writer"] = writer
        print("RECORDING: {}\nStart ENU: {:.3f}, {:.3f}, {:.3f}".format(
            target, point["x"], point["y"], point["z"]), flush=True)
        recording_error = None
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
        except Exception as exc:
            recording_error = str(exc)
            raise
        finally:
            with lock:
                state["writer"] = None
                recording_error = recording_error or state["error"]
                count = writer.finish()
                stream.close()
            print("Saved {} points: {}".format(count, target), flush=True)
            if count < 2:
                print("WARNING: Too few points; this is not a usable driving route.")
            finish_recording(target, args, recording_error)
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
