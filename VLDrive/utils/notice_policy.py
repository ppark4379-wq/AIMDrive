"""Route notices shared by offline labels, live inference and RViz (no ML/ROS imports)."""
import hashlib
import json
import math
from bisect import bisect_right
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZONES = PROJECT_ROOT / "configs/notice_zones/kcity.json"
ZONE_SCHEMA = "morai-route-notice-zones-v1"
NOTICE_SCHEMA = "morai-route-notice-instructions-v1"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_route(path):
    points = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = [float(v) for v in line.split()[:3]]
        if len(values) < 2 or not all(math.isfinite(v) for v in values):
            raise ValueError("Invalid global path point")
        points.append(tuple(values + [0.0] * (3 - len(values))))
    if len(points) < 2:
        raise ValueError("Global path needs at least two points")
    return points


def nearest_index(points, x, y, hint=None):
    """Snap to an existing vertex, with a continuity tie-break at crossing roads."""
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Current position must be finite")
    distances = [(p[0] - x) ** 2 + (p[1] - y) ** 2 for p in points]
    best = min(range(len(points)), key=distances.__getitem__)
    if hint is not None:
        # Only break numerical ties: even tightly spaced distinct vertices must
        # snap to the actual nearest point, never lag behind a stale hint.
        candidates = [i for i, d in enumerate(distances) if abs(d - distances[best]) <= 1e-12]
        best = min(candidates, key=lambda i: (abs(i - hint), distances[i]))
    return best, math.sqrt(distances[best])


def compose_instruction(navigation, notice=""):
    if not isinstance(navigation, str) or not navigation.strip():
        raise ValueError("Navigation instruction must be nonempty")
    if not isinstance(notice, str):
        raise ValueError("Notice must be a string (empty when inactive)")
    return navigation if not notice.strip() else navigation + "\n" + notice.strip()


def compose_instruction_batch(navigation, notices=None):
    if not isinstance(navigation, (list, tuple)) or not navigation:
        raise ValueError("Navigation must be a nonempty text batch")
    if notices is None:
        notices = [""] * len(navigation)
    if not isinstance(notices, (list, tuple)) or len(notices) != len(navigation):
        raise ValueError("Notice and navigation batch sizes must match")
    return [compose_instruction(nav, notice) for nav, notice in zip(navigation, notices)]


def current_condition_suffix(conditions):
    """Exclude old-frame losses when the current navigation/notice was unavailable."""
    begin = len(conditions) - 1
    while begin > 0 and conditions[begin - 1] == conditions[-1]:
        begin -= 1
    return begin


class RouteNoticePolicy:
    def __init__(self, config=DEFAULT_ZONES, global_path=None):
        self.config_path = Path(config).resolve()
        self.spec = json.loads(self.config_path.read_text(encoding="utf-8"))
        if self.spec.get("schema_version") != ZONE_SCHEMA:
            raise ValueError("Unsupported notice zone schema")
        self.route_path = (Path(global_path).resolve() if global_path else
                           (self.config_path.parent / self.spec["global_path"]).resolve())
        if file_sha256(self.route_path) != self.spec["global_path_sha256"]:
            raise ValueError("Notice global path hash mismatch; rebuild snapped zones")
        self.points = read_route(self.route_path)
        self.progress_m = [0.0]
        for a, b in zip(self.points, self.points[1:]):
            self.progress_m.append(self.progress_m[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        if len(self.points) != self.spec["route_point_count"]:
            raise ValueError("Notice route point count mismatch")
        self.sha256 = file_sha256(self.config_path)
        self.zones = self.spec["zones"]
        self.max_route_distance_m = float(self.spec.get("max_route_distance_m", 12.0))
        if not math.isfinite(self.max_route_distance_m) or self.max_route_distance_m <= 0:
            raise ValueError("Invalid maximum route distance")
        ids = set()
        for zone in self.zones:
            if zone["id"] in ids or not zone["text"].strip():
                raise ValueError("Notice IDs must be unique and texts nonempty")
            ids.add(zone["id"])
            start, end = (int(zone[key]["route_index"]) for key in ("start", "end"))
            if not 0 <= start <= end < len(self.points):
                raise ValueError("Notice boundaries must follow global path order")
            for key in ("start", "end"):
                boundary = zone[key]
                if tuple(boundary["xyz"]) != self.points[int(boundary["route_index"])]:
                    raise ValueError("Notice boundary is not an exact global path vertex")

    def at_index(self, index):
        index = int(index)
        if not 0 <= index < len(self.points):
            raise ValueError("Notice route index out of bounds")
        # MGeo repeats link endpoints. Equal consecutive vertices describe the
        # same boundary; a notice ends only at the next physically distinct point.
        progress = self.progress_m[index]
        active = [z for z in self.zones
                  if self.progress_m[z["start"]["route_index"]] <= progress
                  <= self.progress_m[z["end"]["route_index"]]]
        return {"route_index": index, "valid": True,
                "zone_ids": [z["id"] for z in active],
                "notice_text": " ".join(dict.fromkeys(z["text"] for z in active))}

    def first_index_after(self, index):
        return bisect_right(self.progress_m, self.progress_m[index])

    def at_position(self, x, y, hint=None):
        index, distance = nearest_index(self.points, float(x), float(y), hint)
        state = self.at_index(index)
        state["route_distance_m"] = distance
        if distance > self.max_route_distance_m:
            state.update(valid=False, zone_ids=[], notice_text="")
        return state

    def check_route_points(self, points):
        if len(points) != len(self.points) or any(
                math.hypot(float(a[0]) - b[0], float(a[1]) - b[1]) > 1e-3
                for a, b in zip(points, self.points)):
            raise ValueError("Notice route differs from navigation route; do not reuse indices")


class NoticeTracker:
    """Current state plus entry/exit events; automatically handles repeated laps."""
    def __init__(self, policy):
        self.policy = policy
        self.reset()

    def reset(self):
        self.index = None
        self.active = ()

    def update(self, x, y, navigation=None):
        state = self.policy.at_position(x, y, self.index)
        active = tuple(state["zone_ids"])
        state["entered"] = [z for z in active if z not in self.active]
        state["exited"] = [z for z in self.active if z not in active]
        self.index = state["route_index"] if state["valid"] else None
        self.active = active
        if navigation is not None:
            state["instruction"] = navigation
            state["llm_text"] = compose_instruction(navigation, state["notice_text"])
        return state


class NoticeRanges:
    """Contiguous, inclusive frame ranges including explicit no-notice intervals."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.sha256 = file_sha256(self.path)
        self.routes = {}
        identities = set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("schema_version") != NOTICE_SCHEMA
                    or not isinstance(row.get("text"), str)
                    or not isinstance(row.get("zone_ids"), list)
                    or bool(row["text"]) != bool(row["zone_ids"])
                    or not 0 <= int(row["start_frame"]) <= int(row["stop_frame"])):
                raise ValueError("Invalid notice instruction record")
            identities.add((row["zones_sha256"], row["global_path_sha256"]))
            self.routes.setdefault(row["route_id"], []).append(row)
        if len(identities) != 1:
            raise ValueError("Notice file must have one zone/global-path identity")
        self.starts = {}
        for route, rows in self.routes.items():
            rows.sort(key=lambda r: r["start_frame"])
            for a, b in zip(rows, rows[1:]):
                if a["stop_frame"] + 1 != b["start_frame"]:
                    raise ValueError("Notice frame ranges must be contiguous without overlaps")
            self.starts[route] = [r["start_frame"] for r in rows]

    def for_frame(self, route, frame, route_sha256=None):
        rows = self.routes.get(route, [])
        index = bisect_right(self.starts.get(route, []), frame) - 1
        if index < 0 or frame > rows[index]["stop_frame"]:
            raise ValueError("Missing notice coverage for {} frame {}".format(route, frame))
        row = rows[index]
        if route_sha256 is not None and row["route_sha256"] != route_sha256:
            raise ValueError("Notice route hash mismatch")
        return row
