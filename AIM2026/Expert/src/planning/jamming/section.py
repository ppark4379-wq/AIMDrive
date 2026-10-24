"""Metric ENU <-> forward-Y rotation shared by recording and planning.

This module uses only the standard library so the ROS recorder needs no ML
environment. The CSV remains raw ENU; only planner inputs are rotated.
"""
import csv
import json
import math
from pathlib import Path
import re

from ...config.paths import JAMMING_PATHS, JAMMING_SECTION_CONFIG, RECORDINGS_DIRECTORY_NAME

ROOT = JAMMING_PATHS
ROUTE_DIRECTORY = RECORDINGS_DIRECTORY_NAME


class JammingSection:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        # 기본 경로는 중앙 설정을 사용하고, 외부/테스트 라이브러리는 기존 section.json을 유지한다.
        settings_path = (JAMMING_SECTION_CONFIG if self.root.resolve() == ROOT.resolve()
                         else self.root / "section.json")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        self.start = tuple(settings["start_enu"])
        self.end = tuple(settings["end_enu"])
        for point in (self.start, self.end):
            if len(point) != 3 or not all(math.isfinite(v) for v in point):
                raise ValueError("Section endpoints must be finite ENU XYZ")
        for key in ("start_longitudinal_tolerance_m", "start_lateral_half_width_m",
                    "end_longitudinal_tolerance_m", "activation_half_width_m",
                    "approach_distance_m", "return_blend_m", "global_tail_m", "spacing_m"):
            value = float(settings[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid section setting: " + key)
            setattr(self, key, value)
        if self.start_lateral_half_width_m > self.activation_half_width_m:
            raise ValueError("Start-line width exceeds the section activation band")
        dx, dy = self.end[0]-self.start[0], self.end[1]-self.start[1]
        self.length = math.hypot(dx, dy)
        if self.length <= self.return_blend_m + 30.:
            raise ValueError("Section is too short for the recorded-route planner")
        self.forward = (dx/self.length, dy/self.length)
        self.right = (self.forward[1], -self.forward[0])
        self.angle = math.atan2(dy, dx)
        self.rotation = math.pi/2-self.angle

    @property
    def output_dir(self):
        return self.root / ROUTE_DIRECTORY

    def to_local(self, point):
        dx, dy = point[0]-self.start[0], point[1]-self.start[1]
        return (dx*self.right[0]+dy*self.right[1],
                dx*self.forward[0]+dy*self.forward[1])

    def to_world(self, point):
        x, y = point[:2]
        return (self.start[0]+x*self.right[0]+y*self.forward[0],
                self.start[1]+x*self.right[1]+y*self.forward[1])

    def yaw_to_local(self, yaw):
        return math.atan2(math.sin(yaw+self.rotation), math.cos(yaw+self.rotation))

    def validate_start(self, point):
        """Allow lateral starts and the approach before the section entry."""
        lateral, longitudinal = self.to_local(point)
        if not all(math.isfinite(v) for v in (lateral, longitudinal)):
            raise ValueError("Non-finite start position")
        if (longitudinal < -self.approach_distance_m - 1e-9
                or longitudinal > self.start_longitudinal_tolerance_m + 1e-9
                or abs(lateral) > self.start_lateral_half_width_m + 1e-9):
            raise ValueError(
                "Outside Jamming start line: forward={:+.2f} m (allowed -{:.1f} to +{:.1f}), "
                "right={:+.2f} m (allowed +/-{:.1f}). Reposition near the start line.".format(
                    longitudinal, self.approach_distance_m, self.start_longitudinal_tolerance_m,
                    lateral, self.start_lateral_half_width_m))

    def validate_recording(self, points):
        """Validate raw (x,y,yaw) without registering incomplete recordings."""
        if len(points) < 10 or not all(math.isfinite(v) for p in points for v in p):
            raise ValueError("Insufficient or non-finite route points")
        self.validate_start(points[0])
        end_side, end_forward = self.to_local(points[-1])
        if (abs(end_forward-self.length) > self.end_longitudinal_tolerance_m + 1e-9
                or abs(end_side) > self.activation_half_width_m + 1e-9):
            raise ValueError(
                "Recording has not reached the Jamming end line: forward error={:+.2f} m, "
                "right={:+.2f} m".format(end_forward-self.length, end_side))
        local = [(*self.to_local(p), self.yaw_to_local(p[2])) for p in points]
        backwards, distance = [], 0.
        for i, (a, b) in enumerate(zip(local, local[1:])):
            step = math.hypot(b[0]-a[0], b[1]-a[1])
            if step > 1.:
                raise ValueError("Recording has a gap larger than 1 m; drive more slowly")
            distance += step
            forward = (b[0]-a[0])*math.cos(a[2])+(b[1]-a[1])*math.sin(a[2])
            if forward < -.03:
                backwards.append((i, distance))
        if backwards and backwards[-1][1] > 3.:
            raise ValueError("Recording reverses after initial positioning")
        trim = backwards[-1][0]+1 if backwards else 0
        forward_points = local[trim:]
        if any(math.sin(p[2]) < .5 for p in forward_points):
            raise ValueError("Recording heading leaves the section's forward corridor")
        if any(b[1] < a[1]-.03 for a, b in zip(forward_points, forward_points[1:])):
            raise ValueError("Recording must progress from the Jamming start toward its end")
        if any(abs(p[0]) > self.activation_half_width_m for p in local):
            raise ValueError("Recording leaves the configured section activation band")
        return local


def next_route_name(directory, prefix="jamming"):
    directory = Path(directory)
    index = 1
    while (directory / ("%s_%02d.csv" % (prefix, index))).exists():
        index += 1
    return "%s_%02d" % (prefix, index)


def register_route(path, section):
    """Publish only a complete recording from this section's own directory."""
    path = Path(path).resolve()
    if path.parent != section.output_dir.resolve():
        raise ValueError("Only files in the Jamming candidate directory can be registered")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\.csv", path.name):
        raise ValueError("Invalid route filename")
    manifest = section.root / "routes.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    names = value["routes"]
    if not isinstance(names, list) or any(not isinstance(name, str) or Path(name).name != name for name in names):
        raise ValueError("Invalid Jamming route manifest")
    if value.get('directory'):
        from .prepare_routes import PREPARED_DIRECTORY, prepare_library
        if value['directory'] != PREPARED_DIRECTORY:
            raise ValueError('Invalid Jamming candidate directory')
        prepare_library(section, names if path.name in names else names+[path.name], value.get('requested_end_enu'))
        return path.name
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    section.validate_recording([tuple(float(row[k]) for k in ("x", "y", "yaw_rad")) for row in rows])
    if path.name not in names:
        names.append(path.name)
        pending = manifest.with_suffix(".pending.json")
        pending.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        pending.replace(manifest)
    return path.name
