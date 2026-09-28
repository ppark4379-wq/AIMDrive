#!/usr/bin/env python3
"""Pin the MGeo lane markings around the full_loop_static segment.

The repository map folder carries links and the official path only.  The
static-preset lane gate needs the painted boundaries, so this tool copies the
boundaries near the segment from the simulator's MGeo export into a small
JSON file together with the source SHA-256.  It never edits the source map.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MAP_DIR = ROOT / "scenario_runner" / "data" / "R_KR_PR_K-city_2025"
# The data folder holds the separated map inputs only; pinned extracts live in config.
DEFAULT_OUTPUT = ROOT / "scenario_runner" / "config" / "static_segment_lane_markings.json"
SCHEMA = "morai_static_segment_lane_markings_v1"
# full_loop_static drives official-route index 483 -> 753 (132.4 m).
SEGMENT_INDEX = (483, 753)
CORRIDOR_M = 12.0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def near_route(points, route, corridor_m):
    for x, y in points:
        if np.min(np.hypot(route[:, 0] - x, route[:, 1] - y)) <= corridor_m:
            return True
    return False


def extract(source, route_path, segment=SEGMENT_INDEX, corridor_m=CORRIDOR_M):
    rows = json.loads(Path(source).read_text(encoding="utf-8"))
    route = np.loadtxt(route_path)[segment[0]:segment[1] + 1, :2]
    lines = []
    for row in rows:
        points = [[float(p[0]), float(p[1])] for p in row.get("points", [])]
        if len(points) < 2 or not all(math.isfinite(v) for p in points for v in p):
            continue
        if not near_route(points, route, corridor_m):
            continue
        lines.append({
            "idx": str(row["idx"]),
            "lane_type": [int(value) for value in row.get("lane_type", [])],
            "lane_color": [str(value) for value in row.get("lane_color", [])],
            "lane_shape": [str(value) for value in row.get("lane_shape", [])],
            "lane_width_m": float(row.get("lane_width", 0.15) or 0.15),
            "points_xy": [[round(x, 4), round(y, 4)] for x, y in points],
        })
    lines.sort(key=lambda line: line["idx"])
    return {
        "schema": SCHEMA,
        "map_name": "R_KR_PR_K-city_2025",
        "source": {"file": Path(source).name, "sha256": sha256_file(source)},
        "route": {"file": Path(route_path).name, "sha256": sha256_file(route_path),
                  "segment_index": list(segment), "corridor_m": float(corridor_m)},
        "lines": lines,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mgeo-lane-boundaries", type=Path, required=True,
                        help="MORAI MGeo lane_boundary_set.json for R_KR_PR_K-city_2025")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("Existing lane marking file is preserved: {}".format(args.output))
    document = extract(args.mgeo_lane_boundaries,
                       MAP_DIR / "2026_molit_comp_global_path.txt")
    args.output.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print("[LANES] {} boundaries -> {}".format(len(document["lines"]), args.output))


if __name__ == "__main__":
    main()
