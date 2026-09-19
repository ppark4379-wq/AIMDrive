#!/usr/bin/env python3
"""Snap user boundaries to global-path vertices and label recorded frames."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.notice_policy import (DEFAULT_ZONES, NOTICE_SCHEMA, ZONE_SCHEMA,
                                 NoticeTracker, RouteNoticePolicy, file_sha256,
                                 nearest_index, read_route)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def snap_zones(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    route = (source.parent / raw["global_path"]).resolve()
    points = read_route(route)
    zones = []
    previous_start = -1
    for item in raw["zones"]:
        zone = {k: item[k] for k in ("id", "label", "type", "text", "color")}
        for key in ("start", "end"):
            xy = item[key]
            index, distance = nearest_index(points, float(xy[0]), float(xy[1]))
            if distance > float(raw.get("max_snap_distance_m", 5.0)):
                raise ValueError("{} {} is {:.2f}m from path; confirm coordinates".format(
                    item["id"], key, distance))
            zone[key] = dict(requested_xy=xy, route_index=index,
                             xyz=list(points[index]), snap_distance_m=distance)
        if not previous_start <= zone["start"]["route_index"] <= zone["end"]["route_index"]:
            raise ValueError("Notice boundaries do not follow supplied route order: " + zone["id"])
        previous_start = zone["start"]["route_index"]
        zones.append(zone)
    import os
    result = dict(schema_version=ZONE_SCHEMA, map_id=raw["map_id"],
                  coordinate_frame="MORAI world XY metres", boundary_policy="inclusive_path_vertices",
                  global_path=os.path.relpath(route, output.parent).replace("\\", "/"),
                  global_path_sha256=file_sha256(route), route_point_count=len(points),
                  source_sha256=file_sha256(source), max_route_distance_m=12.0, zones=zones)
    write_json(output, result)
    RouteNoticePolicy(output)  # Verify exact vertices and source identity before use.
    return result


def compress_notices(frames):
    """(frame_id, state) -> inclusive ranges; inactive intervals are explicit."""
    ranges = []
    for frame, state in frames:
        signature = (state["notice_text"], state["zone_ids"])
        if (ranges and ranges[-1]["stop_frame"] + 1 == frame
                and (ranges[-1]["text"], ranges[-1]["zone_ids"]) == signature):
            ranges[-1]["stop_frame"] = frame
        else:
            ranges.append(dict(start_frame=frame, stop_frame=frame,
                               text=state["notice_text"], zone_ids=state["zone_ids"]))
    return ranges


def build_dataset_notices(dataset_root, policy):
    from tools.build_instructions import dataset_routes, load
    _, routes = dataset_routes(dataset_root)
    records = []
    for route_id, route, count in routes:
        route_file = route / "route.json"
        tracker = NoticeTracker(policy)
        frames = []
        for frame in range(count):
            m = load(route / "measurements_full" / ("{:04d}.json".format(frame)))
            if int(m.get("frame_id", -1)) != frame:
                raise ValueError("Notice measurement frame mismatch")
            if m.get("navigation", {}).get("route_valid") is False:
                raise ValueError("Cannot label notice from invalid current localization")
            state = tracker.update(float(m["gps_x"]), float(m["gps_y"]))
            if not state["valid"]:
                raise ValueError("Recorded ego is outside notice route: {} {}".format(route_id, frame))
            frames.append((frame, state))
        for row in compress_notices(frames):
            row.update(schema_version=NOTICE_SCHEMA, route_id=route_id,
                       source="current_ego_snapped_global_path",
                       route_sha256=file_sha256(route_file), zones_sha256=policy.sha256,
                       global_path_sha256=policy.spec["global_path_sha256"])
            records.append(row)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, help="User boundary JSON to snap")
    parser.add_argument("--zones", type=Path, default=DEFAULT_ZONES)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.points:
        result = snap_zones(args.points, args.zones)
        for zone in result["zones"]:
            print("{}: {}..{} (snap {:.3f}m / {:.3f}m)".format(
                zone["id"], zone["start"]["route_index"], zone["end"]["route_index"],
                zone["start"]["snap_distance_m"], zone["end"]["snap_distance_m"]))
    if args.dataset_root:
        if not args.output:
            parser.error("--dataset-root requires --output")
        rows = build_dataset_notices(args.dataset_root, RouteNoticePolicy(args.zones))
        serialized = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows)
        if args.output.exists() and args.output.read_text(encoding="utf-8") != serialized:
            raise FileExistsError("Different notice file exists; choose a new --output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print("Wrote {} notice ranges to {}".format(len(rows), args.output))
    elif args.output:
        parser.error("--output requires --dataset-root")
    elif not args.points:
        parser.error("Supply --points or --dataset-root")


if __name__ == "__main__":
    main()
