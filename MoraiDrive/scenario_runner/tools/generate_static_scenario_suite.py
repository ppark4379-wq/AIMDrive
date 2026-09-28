#!/usr/bin/env python3
"""Generate unique static-obstacle layouts and keep only Expert-proven runs.

Every attempt is an independent MORAI scenario process.  A candidate becomes
a replayable preset only when the ordinary ``full_loop_static`` runner exits
successfully, which means the real Expert reached the segment goal without a
reported collision or scenario-integrity failure.  Sensor Record is never
enabled by this tool.
"""

import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scenario_runner.tools import static_lane_compliance as lane  # noqa: E402

CONFIG_PATH = ROOT / "scenario_runner" / "config" / "full_loop.yaml"
ROUTE_PATH = ROOT / "scenario_runner" / "config" / "full_loop_links.yaml"
FOOTPRINT_PATH = ROOT / "scenario_runner" / "config" / "static_obstacle_footprints_v1.json"
OUTPUT_ROOT = (
    ROOT / "scenario_runner" / "config" / "static_scenario_presets" / "generated"
)
RUNTIME_ROOT = ROOT / "scenario_runner" / ".runtime" / "static_scenario_batch"
SCHEMA = "morai_static_obstacle_preset_v1"
SUITE_SCHEMA = "morai_static_obstacle_suite_v1"
PLACEMENT_POLICY = "lane_aware_right_edge_strata_v1"
# Narrowest ego-lane width at the intersection entry and exit (MGeo yellow
# centre line to white edge line); the intersection itself has no right edge.
LANE_WIDTH_M = 3.34
LANE_EDGE_MARGIN_M = 0.10
OBSTACLE_GAP_M = 0.30
# The Expert's recorded static routes (read only) and the planner body it
# sweeps along them: rear axle at the recorded pose, 4.635 m forward, 2.0 m
# wide (expert lattice_planning/parameters.py VEHICLE_LENGTH, VEHICLE_WIDTH).
EXPERT_ROUTE_DIR = (ROOT / "expert" / "src" / "expert" / "src" / "planning"
                    / "정적장애물" / "후보전역경로")
PLANNER_BODY_LENGTH_M = 4.635
PLANNER_BODY_HALF_WIDTH_M = 1.0
ROUTE_INDEX_SPACING_M = 0.5  # official path spacing in the static zone
# Clearance = box right edge - left edge of a route's swept planner body over
# the box's route-index span.  In 123 single-obstacle runs of
# static_native36_probe_retry right_02 was chosen only at 0.63 m or more
# (right_03 up to 0.71 m), right_03 cleared at 0.66 m or more and right_04/05
# (about 5 m) were needed up to 0.53 m.  The ambiguous bands stay unused.
SMALL_CLEARANCE_M = 0.80         # every box of a small-swing layout vs right_02
BLOCKING_MAX_CLEARANCE_M = 0.55  # the blocking box of a medium layout vs right_02
MEDIUM_CLEARANCE_M = 0.75        # every box vs right_03
# A model with less lateral room than this keeps its right edge within
# OVERHANG_RIGHT_EDGE_SPAN_M of the right_03 limit and covers the centre line,
# as a rock slide would (static36 Rockslide passed so with right_03).
MIN_LATERAL_WINDOW_M = 0.30
OVERHANG_RIGHT_EDGE_SPAN_M = 0.50
# Right-edge strata: the lower and upper half of the medium band, and the
# small band.  Their bounds follow the route envelopes at each route index.
LATERAL_STRATA = ("right", "centre", "left")
# Boxes right of the Ego path keep their left edge this far right of a
# recorded route's swept body (the same 0.75 m as MEDIUM_CLEARANCE_M), or stay
# under BLOCKING_MAX_CLEARANCE_M of right_02 to block it.  The right road
# limit keeps them 0.45 m inside the intersection corner lines (-5.45 m).
RIGHT_ROAD_LIMIT_M = -5.0
RIGHT_STRATA = ("right_near", "right_far", "right_block")
# Layout targets.  straight: every box right of right_01 (live: right_01, 0.3 m).
# late: a box just outside the lane at the zone start (index 660-662) blocks
# right_02 to right_04 and a lane box after index 676 forces the swing; the
# Expert then follows right_01 and joins right_03 (live 6 of 6, onset index
# 669, 3.1 m).  With the right-side box at index 665 or later that join has no
# room and the Expert swung 6.6 m around it on right_05/06 (live 5 of 5).  A
# third box, far right at the zone start, blocks right_05/06 as well (live 2
# of 2).  A box right of the path cannot force an S-shaped return for the
# same reason: right_04/05 pass it on the right.
BLOCKED_ROUTES = ("right_02.csv", "right_03.csv", "right_04.csv", "right_05.csv",
                  "right_06_retry.csv")
AVOIDANCE_TARGETS = ("straight", "small", "medium", "late")
LANE_ROLES = {"lane_small": ("left",), "lane_block": ("right", "centre"),
              "lane_any": ("left", "centre", "right"), "lane_late": ("centre",)}
# The late lane box leaves room for the right_01 -> right_03 join.  Live runs
# joined with its right edge at 0.0 to 0.4 m (6 runs) but also swung 6.6 m
# around the right-side box at -0.03 m (2 runs) and at -0.34 m or lower (4).
LATE_LANE_RIGHT_EDGE_MIN_M = 0.1
# Fourteen verified models cannot sit in the left stratum and appear in medium
# layouts only, so a plan that covers every model needs more medium runs.
DEFAULT_AVOIDANCE_WEIGHTS = "20,20,40,20"


def portable_path(path):
    """Store repository-owned paths without a machine-specific home path."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate different static-obstacle layouts, drive each with the "
            "real Expert, and retain collision-free completions only."
        )
    )
    parser.add_argument("--attempts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", default=None, help="Optional suite name")
    parser.add_argument("--min-obstacles", type=int, default=1)
    parser.add_argument("--max-obstacles", type=int, default=3)
    parser.add_argument(
        "--placement", choices=("lane", "anchor"), default="lane",
        help=(
            "lane: keep each LiDAR-verified model box inside the ego lane and "
            "spread its right edge over lateral strata (default). anchor: the "
            "earlier +-lateral-margin sampling around the zone anchors"
        ),
    )
    parser.add_argument("--lateral-margin-m", type=float, default=0.60,
                        help="Used by --placement anchor only")
    parser.add_argument("--footprint-catalog", type=Path, default=FOOTPRINT_PATH,
                        help="LiDAR-verified model footprints; limits the models")
    parser.add_argument("--avoidance-weights", default=DEFAULT_AVOIDANCE_WEIGHTS,
                        metavar="SMALL,MEDIUM",
                        help="Relative share of small/medium swing layouts (--placement lane)")
    parser.add_argument("--yaw-margin-deg", type=float, default=0.0,
                        help="Default keeps the configured obstacle orientation")
    parser.add_argument(
        "--scale-jitter", type=float, default=0.0,
        help="Compatibility option: only 0 is accepted; MORAI native model size [1, 1, 1] is used",
    )
    parser.add_argument("--trial-timeout-sec", type=float, default=240.0)
    parser.add_argument(
        "--invalid-plan-timeout-sec", type=float, default=15.0,
        help=(
            "Reject early when the live Expert remains blocked with a "
            "planner error and no valid candidate for this many seconds"
        ),
    )
    parser.add_argument(
        "--signal-stop-timeout-sec", type=float, default=30.0,
        help=(
            "Reject early when a valid path remains stopped only by the "
            "traffic-signal speed limit for this many seconds"
        ),
    )
    parser.add_argument(
        "--max-cross-track-error-m", type=float, default=4.0,
        help=(
            "Coarse backstop on the 1 Hz logged Ego position error; the "
            "--ego-trace lane gate decides marking contact and deviation"
        ),
    )
    parser.add_argument(
        "--max-right-deviation-m", type=float,
        default=lane.DEFAULT_POLICY["max_right_deviation_m"],
        help="Reject when the Ego body centre moves further right of the route",
    )
    parser.add_argument(
        "--min-marking-clearance-m", type=float,
        default=lane.DEFAULT_POLICY["min_marking_clearance_m"],
        help="Required clearance between the Ego body and any no-cross marking",
    )
    parser.add_argument(
        "--keep-going-after",
        type=int,
        default=None,
        metavar="ACCEPTED",
        help="Stop early after this many accepted layouts",
    )
    parser.add_argument(
        "--accepted-model-minimum", type=int, default=0,
        help="Require every configured static model in at least this many accepted layouts",
    )
    parser.add_argument(
        "--accepted-route-targets",
        metavar="SHORT,ALTERNATE,WIDE",
        help=(
            "Stop early only after accepted Expert paths reach all three "
            "route-class targets"
        ),
    )
    return parser.parse_args(argv)


def load_generation_policy(config_path=CONFIG_PATH, route_path=ROUTE_PATH):
    with Path(config_path).open("r", encoding="utf-8") as source:
        scenario = yaml.safe_load(source)["scenarios"]["full_loop"]
    with Path(route_path).open("r", encoding="utf-8") as source:
        route = yaml.safe_load(source)["full_loop"]
    spec = route["zones"]["static_obstacle"]
    anchors = list(spec.get("multi_positions", []))
    if len(anchors) < 2:
        raise ValueError("static_obstacle.multi_positions requires two anchors")
    models = [str(value) for value in scenario.get("static_obstacle_models", [])]
    if not models:
        raise ValueError("static_obstacle_models is empty")
    # Legacy route/model scales belong to old recorded layouts. New layouts
    # always use each MORAI asset at its native size.
    scale_overrides = {model: [1.0, 1.0, 1.0] for model in models}
    fallback_scale = [1.0, 1.0, 1.0]
    return {
        "models": models,
        "scale_overrides": scale_overrides,
        "fallback_scale": fallback_scale,
        "anchor_start": copy.deepcopy(anchors[0]),
        "anchor_end": copy.deepcopy(anchors[-1]),
        "map_name": "R_KR_PR_K-city_2025",
    }


def _stratified_positions(rng, count):
    """Return ordered, separated fractions across the verified route zone."""
    # Each obstacle owns one longitudinal stratum.  This avoids overlap-prone
    # center clustering while retaining random positions in every trial.
    margin = 0.04
    values = []
    for index in range(count):
        low = index / float(count) + margin / count
        high = (index + 1) / float(count) - margin / count
        values.append(rng.uniform(low, high))
    return values


def generate_layout(policy, rng, count, lateral_margin_m, yaw_margin_deg,
                    scale_jitter=0.0, required_model=None):
    if count < 1:
        raise ValueError("Obstacle count must be positive")
    if count > len(policy["models"]):
        raise ValueError("Obstacle count exceeds available unique models")
    if scale_jitter != 0.0:
        raise ValueError("Random obstacle scale is disabled; use --scale-jitter 0")

    first = policy["anchor_start"]
    last = policy["anchor_end"]
    ax = float(first["position"]["x"])
    ay = float(first["position"]["y"])
    az = float(first["position"]["z"])
    bx = float(last["position"]["x"])
    by = float(last["position"]["y"])
    bz = float(last["position"]["z"])
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1.0:
        raise ValueError("Static obstacle generation anchors are too close")
    # Left normal to the anchor segment.  Positions remain tied to the known
    # intersection zone rather than arbitrary world/map coordinates.
    nx, ny = -dy / length, dx / length
    base_yaw = 0.5 * (
        float(first.get("yaw_deg", 0.0)) + float(last.get("yaw_deg", 0.0))
    )
    route_start = int(first.get("route_path_index", 0))
    route_end = int(last.get("route_path_index", route_start))
    if required_model is not None:
        if required_model not in policy["models"]:
            raise ValueError("Required model is outside the generation policy")
        remaining = [model for model in policy["models"] if model != required_model]
        models = [required_model] + rng.sample(remaining, count - 1)
        rng.shuffle(models)
    else:
        models = rng.sample(policy["models"], count)
    obstacles = []
    for fraction, model in zip(_stratified_positions(rng, count), models):
        lateral = rng.uniform(-lateral_margin_m, lateral_margin_m)
        x = ax + fraction * dx + lateral * nx
        y = ay + fraction * dy + lateral * ny
        z = az + fraction * (bz - az)
        yaw = base_yaw + rng.uniform(-yaw_margin_deg, yaw_margin_deg)
        # Do not inherit enlargement from an old/injected generation policy.
        scale = [1.0, 1.0, 1.0]
        obstacles.append({
            "model": model,
            "position": {
                "x": round(x, 6), "y": round(y, 6), "z": round(z, 6),
            },
            "yaw_deg": round(yaw, 6),
            "scale": scale,
            "route_path_index": int(round(
                route_start + fraction * (route_end - route_start)
            )),
            "generation": {
                "longitudinal_fraction": round(fraction, 8),
                "lateral_offset_m": round(lateral, 6),
                "scale_multiplier": 1.0,
                "scale_policy": "morai_native_model_scale_v1",
            },
        })
    return obstacles


def load_footprints(path=FOOTPRINT_PATH):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != "morai_static_obstacle_footprints_v1":
        raise ValueError("Unsupported static footprint catalog: {}".format(path))
    return document


def load_lane_policy(policy, footprint_path=FOOTPRINT_PATH,
                     markings_path=lane.DEFAULT_MARKINGS):
    """Restrict models to LiDAR-verified footprints and pin the lane geometry."""
    footprints = load_footprints(footprint_path)
    allowed = [model for model in policy["models"] if model in footprints["models"]]
    if not allowed:
        raise ValueError("No configured static model has a verified footprint")
    markings = lane.load_markings(markings_path)
    route, first_index = lane.load_route()
    names = ("right_01.csv",) + BLOCKED_ROUTES
    envelopes = {name: route_envelope(EXPERT_ROUTE_DIR / name, route, first_index)
                 for name in names[1:]}
    right_envelopes = {name: route_envelope(EXPERT_ROUTE_DIR / name, route, first_index,
                                            side="right") for name in names}
    result = dict(policy)
    result.update({
        "models": allowed,
        "models_without_footprint": [
            model for model in policy["models"] if model not in footprints["models"]
        ],
        "footprints": footprints["models"],
        "footprint_catalog": {"path": portable_path(footprint_path),
                              "sha256": sha256_file(footprint_path),
                              "registry_sha256": footprints["registry"]["sha256"]},
        "markings": markings,
        "markings_file": {"path": portable_path(markings_path),
                          "sha256": sha256_file(markings_path)},
        "route": (route, first_index),
        "envelopes": envelopes,
        "right_envelopes": right_envelopes,
        "expert_routes": {name: {"path": portable_path(EXPERT_ROUTE_DIR / name),
                                 "sha256": sha256_file(EXPERT_ROUTE_DIR / name)}
                          for name in names},
    })
    return result


def route_envelope(path, route, first_index, side="left"):
    """Left edge (+left, route frame; the maximum) or right edge (the minimum)
    of the planner body swept along one recorded route, per half route index."""
    sign, pick = (1.0, max) if side == "left" else (-1.0, min)
    points = []
    with Path(path).open(encoding="utf-8") as source:
        for row in csv.DictReader(source):
            x, y, yaw = float(row["x"]), float(row["y"]), float(row["yaw_rad"])
            c, s = math.cos(yaw), math.sin(yaw)
            for forward in np.linspace(0.0, PLANNER_BODY_LENGTH_M, 24):
                half = sign * PLANNER_BODY_HALF_WIDTH_M
                points.append((x + forward * c - half * s, y + forward * s + half * c))
    index, lateral = lane.route_frame(np.array(points), route, first_index)
    envelope = {}
    for key, value in zip(np.round(index * 2.0) / 2.0, lateral):
        start = -math.inf if side == "left" else math.inf
        envelope[float(key)] = pick(envelope.get(float(key), start), float(value))
    return envelope


def envelope_over(envelope, first, last, pick=max):
    keys = [key for key in envelope
            if math.floor(first * 2.0) / 2.0 <= key <= math.ceil(last * 2.0) / 2.0]
    if not keys:
        raise ValueError("Recorded route does not cover route index {}-{}".format(first, last))
    return pick(envelope[key] for key in keys)


def footprint_route_extent(footprint, yaw_deg, tangent):
    """Lateral (+left) and along-route extent of a model box for a pivot at 0."""
    yaw = math.radians(yaw_deg)
    cx, cy = footprint["center_offset_spawn_xy"]
    centre = np.array([cx * math.cos(yaw) - cy * math.sin(yaw),
                       cx * math.sin(yaw) + cy * math.cos(yaw)])
    heading = yaw + math.radians(footprint["heading_offset_deg"])
    axis = np.array([math.cos(heading), math.sin(heading)])
    side = np.array([-axis[1], axis[0]])
    corners = [centre + a * 0.5 * footprint["length_m"] * axis
               + b * 0.5 * footprint["width_m"] * side
               for a in (-1, 1) for b in (-1, 1)]
    normal = np.array([-tangent[1], tangent[0]])
    lateral = [float(corner @ normal) for corner in corners]
    along = [float(corner @ tangent) for corner in corners]
    return (min(lateral), max(lateral)), (min(along), max(along))


def lateral_window(policy, model, route_index, yaw_deg):
    """Right-edge range keeping the model box in the ego lane and clear of right_03."""
    route, first_index = policy["route"]
    _, tangent, _ = lane.route_pose(route, first_index, route_index)
    (low, high), along = footprint_route_extent(policy["footprints"][model], yaw_deg, tangent)
    left_bound = lane.centre_line_lateral(policy["markings"], route, first_index, route_index)
    span = (route_index + along[0] / ROUTE_INDEX_SPACING_M,
            route_index + along[1] / ROUTE_INDEX_SPACING_M)
    right_02 = envelope_over(policy["envelopes"]["right_02.csv"], *span)
    right_03 = envelope_over(policy["envelopes"]["right_03.csv"], *span)
    width = high - low
    right_min = max(left_bound - LANE_WIDTH_M + LANE_EDGE_MARGIN_M,
                    right_03 + MEDIUM_CLEARANCE_M)
    right_max = left_bound - LANE_EDGE_MARGIN_M - width
    return {"right_edge_min": right_min, "right_edge_max": right_max,
            "small_min": right_02 + SMALL_CLEARANCE_M,
            "blocking_max": right_02 + BLOCKING_MAX_CLEARANCE_M,
            "right_02_envelope": right_02, "right_03_envelope": right_03,
            "pivot_to_right_edge": low, "width": width, "along": along,
            "left_bound": left_bound,
            "fits": right_max - right_min >= MIN_LATERAL_WINDOW_M}


def stratum_range(window, name):
    low, high = window["right_edge_min"], window["right_edge_max"]
    top = min(high, window["blocking_max"])
    middle = 0.5 * (low + top)
    ranges = {"right": (low, middle), "centre": (middle, top),
              "left": (max(low, window["small_min"]), high)}
    low, high = ranges[name]
    return (low, high) if low <= high else None


def feasible_strata(window):
    if not window["fits"]:
        return []
    return [name for name in LATERAL_STRATA if stratum_range(window, name)]


def right_window(policy, model, route_index, yaw_deg):
    """Left-edge limits for a model box right of the Ego path."""
    route, first_index = policy["route"]
    _, tangent, _ = lane.route_pose(route, first_index, route_index)
    (low, high), along = footprint_route_extent(policy["footprints"][model], yaw_deg, tangent)
    span = (route_index + along[0] / ROUTE_INDEX_SPACING_M,
            route_index + along[1] / ROUTE_INDEX_SPACING_M)
    edges = {name: envelope_over(envelope, *span, pick=min)
             for name, envelope in policy["right_envelopes"].items()}
    lefts = {name: envelope_over(policy["envelopes"][name], *span)
             for name in BLOCKED_ROUTES}
    width = high - low
    return {"left_min": RIGHT_ROAD_LIMIT_M + width,
            "clear_max": {name: edge - MEDIUM_CLEARANCE_M for name, edge in edges.items()},
            "block_02_min": edges["right_02.csv"] - BLOCKING_MAX_CLEARANCE_M,
            # Right edge low enough to reach into the right_05 and right_06 bodies.
            "block_far_max": min(lefts["right_05.csv"], lefts["right_06_retry.csv"])
                             + BLOCKING_MAX_CLEARANCE_M + width,
            "right_envelopes": edges, "left_envelopes": lefts,
            "pivot_to_left_edge": high, "width": width, "along": along}


def blocked_routes(window, left_edge):
    """Recorded routes whose swept body the box reaches (clearance under 0.55 m)."""
    right_edge = left_edge - window["width"]
    return [name for name in BLOCKED_ROUTES
            if right_edge < window["left_envelopes"][name] + BLOCKING_MAX_CLEARANCE_M
            and left_edge > window["right_envelopes"][name] - BLOCKING_MAX_CLEARANCE_M]


def role_ranges(policy, role, model, route_index, yaw_deg):
    """Stratum -> (low, high) of the placed edge (right edge in the lane, left edge right of it)."""
    if role in LANE_ROLES:
        window = lateral_window(policy, model, route_index, yaw_deg)
        if not window["fits"]:
            low = window["right_edge_min"]
            ranges = ({"covers": (low, low + OVERHANG_RIGHT_EDGE_SPAN_M)}
                      if role == "lane_block" else {})
        else:
            ranges = {name: stratum_range(window, name) for name in LANE_ROLES[role]}
            if role == "lane_late" and ranges["centre"]:
                low, high = ranges["centre"]
                low = max(low, LATE_LANE_RIGHT_EDGE_MIN_M)
                ranges["centre"] = (low, high) if low <= high else None
        return {k: v for k, v in ranges.items() if v}, window
    window = right_window(policy, model, route_index, yaw_deg)
    low = window["left_min"]
    if role == "block_02":
        ranges = {"right_block": (max(low, window["block_02_min"]),
                                  window["clear_max"]["right_01.csv"])}
    elif role == "block_far":
        ranges = {"right_wall": (low, min(window["block_far_max"],
                                          window["clear_max"]["right_01.csv"]))}
    else:
        high = window["clear_max"]["right_01.csv" if role == "clear_01" else "right_02.csv"]
        middle = 0.5 * (low + high)
        ranges = {"right_far": (low, middle), "right_near": (middle, high)}
    return {k: v for k, v in ranges.items() if v[0] <= v[1]}, window


def layout_roles(target, count, rng):
    """Per-slot roles and longitudinal fractions (None: stratified) for one target."""
    if target == "straight":
        return ["clear_01"] * count, None
    if target == "small":
        roles = ["lane_small"] + ["clear_02"] * (count > 1)
        roles += [rng.choice(("lane_small", "clear_02")) for _ in range(count - len(roles))]
        rng.shuffle(roles)
        return roles, None
    if target == "medium":
        roles = ["lane_block"] + ["lane_any"] * (count - 1)
        rng.shuffle(roles)
        return roles, None
    if count < 2:
        raise ValueError("A late layout needs a right-side box and a later lane box")
    # Right-side boxes at the zone start, the lane box at index 677-680.
    roles = ["block_02", "lane_late"] if count == 2 else ["block_far", "block_02", "lane_late"]
    fractions = [rng.uniform(0.0, 0.12) for _ in roles[:-1]] + [rng.uniform(0.85, 0.98)]
    return roles, fractions


def target_models_for(policy, target, yaw_deg):
    """Models that can take at least one role of the target inside the zone."""
    roles = {"straight": ("clear_01",), "small": ("lane_small",),
             "medium": ("lane_block",), "late": ("block_far", "block_02", "lane_late")}[target]
    middle = [662.0, 670.0, 678.0]
    return [m for m in policy["models"]
            if any(role_ranges(policy, role, m, k, yaw_deg)[0] for role in roles for k in middle)]


def _least_used(options, counts, rng):
    return min(options, key=lambda key: (counts.get(key, 0), rng.random()))


def generate_lane_layout(policy, rng, count, target, required_model,
                         stratum_counts, yaw_margin_deg=0.0, max_tries=300,
                         model_counts=None):
    """One layout for a trajectory target, with boxes on both sides of the path.

    Each slot has a role: lane_small / lane_block / lane_any keep the box in the
    ego lane by right edge (see lateral_window); clear_01 / clear_02 put it right
    of the Ego path so right_01 / right_02 still pass (near or far half of the
    room up to RIGHT_ROAD_LIMIT_M); block_02 puts it right of the path but close
    enough to block right_02 and right_03.  Multi-box straight layouts use both
    near and far boxes, medium layouts at least two lane strata, and small
    layouts one box right of the path.  A lane_block model with too little room
    covers the centre line.  Other slots take their stratum first, then the
    least used model that fits it.
    """
    if target not in AVOIDANCE_TARGETS:
        raise ValueError("Unknown avoidance target: {}".format(target))
    first, last = policy["anchor_start"], policy["anchor_end"]
    route_start = float(first["route_path_index"])
    route_end = float(last["route_path_index"])
    az, bz = float(first["position"]["z"]), float(last["position"]["z"])
    base_yaw = 0.5 * (float(first.get("yaw_deg", 0.0)) + float(last.get("yaw_deg", 0.0)))
    route, first_index = policy["route"]
    anchor_lateral = float(lane.route_frame(
        np.array([[first["position"]["x"], first["position"]["y"]]]), route, first_index)[1][0])
    for _ in range(max_tries):
        roles, fractions = layout_roles(target, count, rng)
        fractions = fractions or _stratified_positions(rng, count)
        yaws = [base_yaw + rng.uniform(-yaw_margin_deg, yaw_margin_deg) for _ in fractions]
        indices = [route_start + f * (route_end - route_start) for f in fractions]
        cache = {}

        def ranges(model, slot):
            key = (model, slot)
            if key not in cache:
                cache[key] = role_ranges(policy, roles[slot], model, indices[slot], yaws[slot])
            return cache[key]

        capable = [slot for slot in range(count) if ranges(required_model, slot)[0]]
        if not capable:
            continue
        required_slot = rng.choice(capable)
        order = [required_slot] + [slot for slot in range(count) if slot != required_slot]
        models, strata = [None] * count, [None] * count
        # Count strata chosen earlier in this layout so its boxes spread too.
        tentative = dict(stratum_counts)
        for slot in order:
            unused = [m for m in policy["models"] if m not in models]
            if slot == required_slot:
                pool = [required_model]
            elif roles[slot] == "lane_block":
                # The blocking box takes the least used model, wide ones included.
                pool = [m for m in unused if ranges(m, slot)[0]]
                pool = [_least_used(pool, model_counts or {}, rng)] if pool else []
            else:
                pool = [m for m in unused if ranges(m, slot)[0]]
            open_strata = sorted({name for m in pool for name in ranges(m, slot)[0]})
            if not open_strata:
                break
            key = _least_used(["{}:{}".format(target, name) for name in open_strata],
                              tentative, rng)
            tentative[key] = tentative.get(key, 0) + 1
            stratum = key.split(":", 1)[1]
            models[slot] = _least_used([m for m in pool if stratum in ranges(m, slot)[0]],
                                       model_counts or {}, rng)
            strata[slot] = stratum
        if None in models:
            continue
        lane_strata = [name for name, role in zip(strata, roles) if role in LANE_ROLES]
        if target == "straight" and count > 1 and not {"right_near", "right_far"} <= set(strata):
            continue
        if target == "medium" and count > 1 and len(set(lane_strata)) < 2:
            continue
        obstacles, boxes, blocked = [], [], set()
        for slot in range(count):
            model, stratum, role = models[slot], strata[slot], roles[slot]
            bounds, data = ranges(model, slot)
            low, high = bounds[stratum]
            edge = float(rng.uniform(low, high))
            if role in LANE_ROLES:
                right_edge = edge
                pivot_lateral = right_edge - float(data["pivot_to_right_edge"])
                fit = "covers_centre_line" if stratum == "covers" else "inside"
                stratum = "right" if stratum == "covers" else stratum
                clearances = {"right_02_clearance_m": round(right_edge - data["right_02_envelope"], 4),
                              "right_03_clearance_m": round(right_edge - data["right_03_envelope"], 4)}
                extra = {"lane_left_bound_m": round(data["left_bound"], 4)}
            else:
                pivot_lateral = edge - float(data["pivot_to_left_edge"])
                right_edge, fit = edge - data["width"], "right_of_path"
                clearances = {name.replace(".csv", "") + "_right_clearance_m":
                              round(value - edge, 4)
                              for name, value in data["right_envelopes"].items()}
                extra = {"blocks_routes": blocked_routes(data, edge)}
                blocked.update(extra["blocks_routes"])
            point, tangent, normal = lane.route_pose(route, first_index, indices[slot])
            position = point + pivot_lateral * normal
            boxes.append((indices[slot], data["along"], right_edge, right_edge + data["width"]))
            generation = {
                "longitudinal_fraction": round(fractions[slot], 8),
                "lateral_offset_m": round(pivot_lateral - anchor_lateral, 6),
                "route_lateral_m": round(pivot_lateral, 6),
                "right_edge_route_lateral_m": round(right_edge, 4),
                "left_edge_route_lateral_m": round(right_edge + data["width"], 4),
                "placement_role": role,
                "lateral_stratum": stratum,
                "lane_fit": fit,
                "scale_multiplier": 1.0,
                "scale_policy": "morai_native_model_scale_v1",
            }
            generation.update(extra)
            generation.update(clearances)
            obstacles.append({
                "model": model,
                "position": {"x": round(float(position[0]), 6),
                             "y": round(float(position[1]), 6),
                             "z": round(az + fractions[slot] * (bz - az), 6)},
                "yaw_deg": round(yaws[slot], 6),
                "scale": [1.0, 1.0, 1.0],
                "route_path_index": int(round(indices[slot])),
                "generation": generation,
            })
        if target == "late" and not set(BLOCKED_ROUTES[:3]) <= blocked:
            continue  # an open right_02-04 lets the Expert swing around the right boxes
        if _boxes_separated(boxes):
            for item in obstacles:
                key = "{}:{}".format(target, item["generation"]["lateral_stratum"])
                stratum_counts[key] = stratum_counts.get(key, 0) + 1
            return obstacles
    raise RuntimeError("Could not place {} separated obstacles for target {}".format(count, target))


def _boxes_separated(boxes, gap=OBSTACLE_GAP_M):
    # Route-frame boxes: (route index, along extent, right edge, left edge).
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (si, ai, ri, li), (sj, aj, rj, lj) = boxes[i], boxes[j]
            # Route index spacing is 0.5 m here (660 -> 680 spans 10 m).
            ci, cj = 0.5 * si, 0.5 * sj
            along_clear = (ci + ai[1] + gap <= cj + aj[0]) or (cj + aj[1] + gap <= ci + ai[0])
            lateral_clear = (li + gap <= rj) or (lj + gap <= ri)
            if not (along_clear or lateral_clear):
                return False
    return True


def parse_avoidance_weights(value):
    weights = [float(item) for item in str(value).split(",")]
    if len(weights) != len(AVOIDANCE_TARGETS) or min(weights) <= 0:
        raise ValueError("--avoidance-weights requires {} positive numbers".format(
            len(AVOIDANCE_TARGETS)))
    return dict(zip(AVOIDANCE_TARGETS, weights))


def choose_attempt_cell(manifest, counts, rng, weights=None):
    """Obstacle count and avoidance target furthest below its weighted share."""
    weights = weights or {target: 1.0 for target in AVOIDANCE_TARGETS}
    accepted = manifest["accepted_cell_counts"]
    attempted = manifest["attempted_cell_counts"]
    cells = [(count, target) for count in counts for target in AVOIDANCE_TARGETS
             if not (target == "late" and count < 2)]
    key = lambda cell: "{}:{}".format(*cell)
    cell = key(min(cells, key=lambda cell: (
        accepted.get(key(cell), 0) / weights[cell[1]],
        attempted.get(key(cell), 0) / weights[cell[1]], rng.random())))
    count, target = cell.split(":")
    return int(count), target


def layout_signature(obstacles):
    rows = []
    for item in obstacles:
        position = item["position"]
        rows.append((
            item["model"],
            round(float(position["x"]), 3),
            round(float(position["y"]), 3),
            round(float(item["yaw_deg"]), 2),
            tuple(round(float(value), 3) for value in item["scale"]),
        ))
    return tuple(rows)


def write_yaml_atomic(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def write_json_atomic(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_latest_planner_health(log_path, tail_bytes=262144):
    """Return the most recent Expert static-planner state from a live log."""
    try:
        with Path(log_path).open("rb") as source:
            source.seek(0, os.SEEK_END)
            size = source.tell()
            source.seek(max(0, size - int(tail_bytes)), os.SEEK_SET)
            text = source.read().decode("utf-8", errors="replace")
    except OSError:
        return {
            "persistent_invalid_plan": False,
            "persistent_signal_stop": False,
            "static_status": None,
            "candidate_status": None,
        }

    lines = text.splitlines()
    static_status = next(
        (line for line in reversed(lines) if line.startswith("[STATIC] ")),
        None,
    )
    candidate_status = next(
        (
            line for line in reversed(lines)
            if line.startswith("[PLAN] candidate ")
        ),
        None,
    )
    persistent_invalid_plan = bool(
        static_status
        and candidate_status
        and "'phase': 'blocked'" in static_status
        and "valid=False" in candidate_status
    )
    persistent_signal_stop = bool(
        static_status
        and candidate_status
        and "'phase': 'follow'" in static_status
        and "'path_validated': True" in static_status
        and "'speed_source': 'signal'" in static_status
        and "'final_target_velocity': 0.0" in static_status
        and "valid=True" in candidate_status
    )
    return {
        "persistent_invalid_plan": persistent_invalid_plan,
        "persistent_signal_stop": persistent_signal_stop,
        "static_status": static_status,
        "candidate_status": candidate_status,
    }


def _stop_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


def run_candidate(
    command,
    log_path,
    timeout_sec,
    invalid_plan_timeout_sec,
    signal_stop_timeout_sec,
):
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=environment,
        )
        timed_out = False
        early_invalid_plan = False
        early_signal_stop = False
        invalid_since = None
        signal_stop_since = None
        watchdog_evidence = None
        try:
            deadline = started + max(1.0, float(timeout_sec))
            while True:
                return_code = process.poll()
                if return_code is not None:
                    break
                now = time.monotonic()
                health = read_latest_planner_health(log_path)
                if health["persistent_invalid_plan"]:
                    if invalid_since is None:
                        invalid_since = now
                    elif now - invalid_since >= invalid_plan_timeout_sec:
                        early_invalid_plan = True
                        watchdog_evidence = health
                        return_code = _stop_process_group(process)
                        log.write(
                            "\n[STATIC BATCH WATCHDOG] persistent invalid "
                            "Expert plan; candidate rejected early\n"
                        )
                        log.flush()
                        break
                else:
                    invalid_since = None
                if health["persistent_signal_stop"]:
                    if signal_stop_since is None:
                        signal_stop_since = now
                    elif now - signal_stop_since >= signal_stop_timeout_sec:
                        early_signal_stop = True
                        watchdog_evidence = health
                        return_code = _stop_process_group(process)
                        log.write(
                            "\n[STATIC BATCH WATCHDOG] traffic-signal-only "
                            "stop persisted; candidate rejected early\n"
                        )
                        log.flush()
                        break
                else:
                    signal_stop_since = None
                if now >= deadline:
                    timed_out = True
                    return_code = _stop_process_group(process)
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            _stop_process_group(process)
            raise
    return {
        "return_code": int(return_code),
        "timed_out": bool(timed_out),
        "early_invalid_plan": bool(early_invalid_plan),
        "early_signal_stop": bool(early_signal_stop),
        "watchdog_evidence": watchdog_evidence,
        "duration_sec": round(time.monotonic() - started, 3),
    }


def _failure_tail(log_path, limit=18):
    try:
        lines = Path(log_path).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError as error:
        return ["log read failed: {}".format(error)]
    return lines[-limit:]


def read_closed_loop_evidence(log_path):
    """Extract completion, selected-path and lateral-deviation evidence."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    cross_track_samples = [
        float(value)
        for value in re.findall(r"\bcte=([0-9]+(?:\.[0-9]+)?)", text)
    ]
    route_matches = re.findall(
        r"'selected_routes':\s*(\[[^\]]*\])", text
    )
    selected_routes = []
    for encoded in reversed(route_matches):
        try:
            parsed = ast.literal_eval(encoded)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, list) and parsed:
            selected_routes = [str(item) for item in parsed]
            break
    return {
        "goal_reached": "[KCityFullLoop] GOAL REACHED" in text,
        "valid_plan_samples": len(re.findall(
            r"\[PLAN\] candidate .*\bvalid=True\b", text
        )),
        "cross_track_samples": len(cross_track_samples),
        "max_cross_track_error_m": (
            max(cross_track_samples) if cross_track_samples else None
        ),
        "selected_routes": selected_routes,
    }


def classify_selected_routes(routes):
    routes = list(routes)
    if len(routes) >= 3:
        return "wide"
    if routes == ["right_02.csv", "right_03.csv"]:
        return "alternate"
    if routes and routes[0] == "right_04.csv":
        return "alternate"
    return "short"


def parse_route_targets(value):
    if not value:
        return None
    try:
        numbers = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError(
            "--accepted-route-targets requires three integers"
        ) from error
    if len(numbers) != 3 or any(item < 0 for item in numbers):
        raise ValueError(
            "--accepted-route-targets requires three non-negative integers"
        )
    if sum(numbers) < 1:
        raise ValueError("At least one accepted route target must be positive")
    return dict(zip(("short", "alternate", "wide"), numbers))


def evaluate_lane_trace(trace_path, gate_policy):
    try:
        trace = lane.read_trace_csv(trace_path)
    except (OSError, KeyError, ValueError):
        trace = []
    return lane.evaluate_trace(trace, policy=gate_policy)


def lane_summary(result):
    keys = ("passed", "reasons", "raw_samples", "max_sample_gap_m", "route_index_span",
            "min_marking_clearance_m", "closest_marking", "touched_markings",
            "max_right_deviation_m", "max_left_deviation_m", "avoidance_class",
            "trajectory_class", "swing_onset_route_index", "peak_route_index",
            "markings_source_sha256")
    return {key: result.get(key) for key in keys}


def build_candidate_document(name, seed, attempt, obstacles, policy):
    return {
        "schema": SCHEMA,
        "name": name,
        "map_name": policy["map_name"],
        "scenario": "full_loop_static",
        "source": "automated_static_expert_candidate",
        "generation": {
            "batch_seed": int(seed),
            "attempt": int(attempt),
            "candidate_only_until_expert_passes": True,
            "scale_policy": "morai_native_model_scale_v1",
        },
        "obstacles": obstacles,
    }


def main(argv=None):
    args = parse_args(argv)
    if args.scale_jitter != 0.0:
        raise SystemExit("Random obstacle scale is disabled; use --scale-jitter 0")
    try:
        route_targets = parse_route_targets(args.accepted_route_targets)
    except ValueError as error:
        raise SystemExit(str(error))
    if args.attempts < 1:
        raise SystemExit("--attempts must be at least 1")
    if args.min_obstacles < 1 or args.max_obstacles < args.min_obstacles:
        raise SystemExit("Invalid obstacle-count range")
    if args.max_obstacles > 8:
        raise SystemExit("--max-obstacles cannot exceed preset schema limit 8")
    if args.max_cross_track_error_m <= 0:
        raise SystemExit("--max-cross-track-error-m must be positive")
    if args.invalid_plan_timeout_sec <= 0:
        raise SystemExit("--invalid-plan-timeout-sec must be positive")
    if args.signal_stop_timeout_sec <= 0:
        raise SystemExit("--signal-stop-timeout-sec must be positive")
    if args.keep_going_after is not None and args.keep_going_after < 1:
        raise SystemExit("--keep-going-after must be positive")
    if args.accepted_model_minimum < 0:
        raise SystemExit("--accepted-model-minimum cannot be negative")
    if args.max_right_deviation_m <= 0:
        raise SystemExit("--max-right-deviation-m must be positive")
    try:
        avoidance_weights = parse_avoidance_weights(args.avoidance_weights)
    except ValueError as error:
        raise SystemExit(str(error))
    if args.keep_going_after is not None and route_targets is not None:
        raise SystemExit(
            "--keep-going-after and --accepted-route-targets are mutually exclusive"
        )

    seed = (
        int(args.seed) if args.seed is not None
        else int.from_bytes(os.urandom(8), "big")
    )
    rng = random.Random(seed)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suite_name = args.name or "static_auto_{}_{}".format(
        args.attempts, timestamp
    )
    if not suite_name.replace("_", "").replace("-", "").isalnum():
        raise SystemExit("--name must contain letters, digits, '_' or '-' only")
    policy = load_generation_policy()
    if args.placement == "lane":
        policy = load_lane_policy(policy, args.footprint_catalog)
    lane_gate_policy = {
        "max_right_deviation_m": float(args.max_right_deviation_m),
        "min_marking_clearance_m": float(args.min_marking_clearance_m),
    }
    if args.max_obstacles > len(policy["models"]):
        raise SystemExit("Requested more obstacles than unique allowed models")

    runtime_dir = RUNTIME_ROOT / suite_name
    output_dir = OUTPUT_ROOT / suite_name
    candidate_dir = runtime_dir / "candidates"
    log_dir = runtime_dir / "logs"
    manifest_path = output_dir / "suite_manifest.json"
    if runtime_dir.exists() or output_dir.exists():
        raise SystemExit("Suite already exists: {}".format(suite_name))

    manifest = {
        "schema": SUITE_SCHEMA,
        "name": suite_name,
        "seed": seed,
        "requested_attempts": int(args.attempts),
        "accepted_count": 0,
        "attempted_model_counts": {model: 0 for model in policy["models"]},
        "accepted_model_counts": {model: 0 for model in policy["models"]},
        "accepted_model_minimum": int(args.accepted_model_minimum),
        "accepted_route_counts": {
            "short": 0, "alternate": 0, "wide": 0,
        },
        "rejected_count": 0,
        "interrupted": False,
        "created_at_unix": time.time(),
        "generation_policy": {
            "min_obstacles": int(args.min_obstacles),
            "max_obstacles": int(args.max_obstacles),
            "lateral_margin_m": float(args.lateral_margin_m),
            "yaw_margin_deg": float(args.yaw_margin_deg),
            "scale_jitter": float(args.scale_jitter),
            "scale_policy": "morai_native_model_scale_v1",
            "trial_timeout_sec": float(args.trial_timeout_sec),
            "invalid_plan_timeout_sec": float(
                args.invalid_plan_timeout_sec
            ),
            "signal_stop_timeout_sec": float(args.signal_stop_timeout_sec),
            "max_cross_track_error_m": float(args.max_cross_track_error_m),
            "models": policy["models"],
            "accepted_route_targets": route_targets,
            "placement": (PLACEMENT_POLICY if args.placement == "lane"
                          else "anchor_lateral_margin_v1"),
            "models_without_footprint": policy.get("models_without_footprint"),
            "footprint_catalog": policy.get("footprint_catalog"),
            "lane_width_m": LANE_WIDTH_M if args.placement == "lane" else None,
            "lane_edge_margin_m": LANE_EDGE_MARGIN_M if args.placement == "lane" else None,
            "lateral_strata": list(LATERAL_STRATA),
            "route_clearance_m": {"small_vs_right_02": SMALL_CLEARANCE_M,
                                  "blocking_max_vs_right_02": BLOCKING_MAX_CLEARANCE_M,
                                  "every_box_vs_right_03": MEDIUM_CLEARANCE_M},
            "expert_routes": policy.get("expert_routes"),
            "avoidance_weights": avoidance_weights,
            "lane_gate": dict(lane.DEFAULT_POLICY, **lane_gate_policy),
            "lane_markings": {"path": portable_path(lane.DEFAULT_MARKINGS),
                              "sha256": sha256_file(lane.DEFAULT_MARKINGS)},
            "ego_body": dict(lane.EGO_BODY),
        },
        "attempted_cell_counts": {},
        "accepted_cell_counts": {},
        "attempted_lateral_strata": {},
        "accepted_avoidance_counts": {name: 0 for name in lane.TRAJECTORY_CLASSES},
        "attempts": [],
        "accepted_presets": [],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    runtime_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(manifest_path, manifest)

    signatures = set()
    coverage_order = list(policy["models"])
    rng.shuffle(coverage_order)
    required_counts = {model: 0 for model in policy["models"]}
    unplaceable = set()  # (count, target, model) layouts that found no placement
    target_models = {}
    if args.placement == "lane":
        middle = 0.5 * (float(policy["anchor_start"]["route_path_index"])
                        + float(policy["anchor_end"]["route_path_index"]))
        yaw = 0.5 * (float(policy["anchor_start"].get("yaw_deg", 0.0))
                     + float(policy["anchor_end"].get("yaw_deg", 0.0)))
        target_models = {target: target_models_for(policy, target, yaw)
                         for target in AVOIDANCE_TARGETS}
        for target, models in target_models.items():
            print("[STATIC BATCH] {} models: {}".format(target, ",".join(models)))
    print("[STATIC BATCH] suite={} seed={} attempts={}".format(
        suite_name, seed, args.attempts
    ))
    print("[STATIC BATCH] Sensor Record/Synchronization Mode are not used")
    print("[STATIC BATCH] MORAI native model size [1, 1, 1] is used; no random size multiplier")
    print("[STATIC BATCH] accepted presets: {}".format(output_dir))

    try:
        for attempt in range(1, args.attempts + 1):
            target = None
            for _ in range(1000):
                if args.placement == "lane":
                    count, target = choose_attempt_cell(
                        manifest, range(args.min_obstacles, args.max_obstacles + 1), rng,
                        avoidance_weights)
                    if count > len(target_models[target]):
                        target = "medium"
                    if not target_models[target]:
                        raise RuntimeError("No verified model can take a {} layout".format(target))
                    pool = [m for m in target_models[target]
                            if (count, target, m) not in unplaceable]
                    if not pool:
                        raise RuntimeError("No model can be placed in a {}-box {} layout".format(
                            count, target))
                    required_model = min(pool, key=lambda m: (required_counts[m], rng.random()))
                    try:
                        obstacles = generate_lane_layout(
                            policy, rng, count, target, required_model,
                            manifest["attempted_lateral_strata"], args.yaw_margin_deg,
                            model_counts=manifest["attempted_model_counts"],
                        )
                    except RuntimeError:
                        # e.g. two wide right-side boxes do not fit side by side.
                        unplaceable.add((count, target, required_model))
                        continue
                else:
                    count = rng.randint(args.min_obstacles, args.max_obstacles)
                    required_model = coverage_order[(attempt - 1) % len(coverage_order)]
                    obstacles = generate_layout(
                        policy, rng, count,
                        args.lateral_margin_m,
                        args.yaw_margin_deg,
                        args.scale_jitter,
                        required_model=required_model,
                    )
                signature = layout_signature(obstacles)
                if signature not in signatures:
                    signatures.add(signature)
                    break
            else:
                raise RuntimeError("Could not generate another unique layout")
            required_counts[required_model] += 1
            cell = "{}:{}".format(count, target) if target else None
            if cell:
                manifest["attempted_cell_counts"][cell] = (
                    manifest["attempted_cell_counts"].get(cell, 0) + 1)

            candidate_name = "{}_attempt_{:03d}".format(suite_name, attempt)
            candidate_path = candidate_dir / (candidate_name + ".yaml")
            log_path = log_dir / (candidate_name + ".log")
            trace_path = runtime_dir / "traces" / (candidate_name + ".csv")
            document = build_candidate_document(
                candidate_name, seed, attempt, obstacles, policy
            )
            if target:
                document["generation"]["placement_policy"] = PLACEMENT_POLICY
                document["generation"]["avoidance_target"] = target
                document["generation"]["footprint_catalog_sha256"] = (
                    policy["footprint_catalog"]["sha256"])
            write_yaml_atomic(candidate_path, document)
            command = [
                sys.executable, "-m", "scenario_runner.main_runner",
                "--zone", "full_loop",
                "--scenario", "full_loop_static",
                "--once", "--no-sync-runtime",
                "--static-preset", str(candidate_path),
                "--ego-trace", str(trace_path),
            ]
            models = [item["model"] for item in obstacles]
            for model in models:
                manifest["attempted_model_counts"][model] += 1
            print("\n[STATIC BATCH {}/{}] count={} target={} required={} models={}".format(
                attempt, args.attempts, count, target, required_model, ",".join(models)
            ))
            print("[STATIC BATCH] log={}".format(log_path))
            result = run_candidate(
                command,
                log_path,
                args.trial_timeout_sec,
                args.invalid_plan_timeout_sec,
                args.signal_stop_timeout_sec,
            )
            evidence = read_closed_loop_evidence(log_path)
            lane_result = evaluate_lane_trace(trace_path, lane_gate_policy)
            rejection_reasons = []
            if result["return_code"] != 0:
                rejection_reasons.append("runner_return_code")
            if result["timed_out"]:
                rejection_reasons.append("trial_timeout")
            if result["early_invalid_plan"]:
                rejection_reasons.append("persistent_invalid_expert_plan")
            if result["early_signal_stop"]:
                rejection_reasons.append("persistent_signal_stop")
            if not evidence["goal_reached"]:
                rejection_reasons.append("goal_not_proven")
            if evidence["valid_plan_samples"] < 1:
                rejection_reasons.append("no_valid_expert_plan_sample")
            if not evidence["selected_routes"]:
                rejection_reasons.append("selected_route_evidence_missing")
            max_cte = evidence["max_cross_track_error_m"]
            if max_cte is None:
                rejection_reasons.append("cross_track_evidence_missing")
            elif max_cte > args.max_cross_track_error_m:
                rejection_reasons.append("cross_track_limit_exceeded")
            rejection_reasons.extend(lane_result["reasons"])
            accepted = not rejection_reasons
            row = {
                "attempt": attempt,
                "accepted": accepted,
                "obstacle_count": count,
                "models": models,
                "candidate": portable_path(candidate_path),
                "log": portable_path(log_path),
                "closed_loop_evidence": evidence,
                "lane_compliance": lane_summary(lane_result),
                "avoidance_target": target,
                "rejection_reasons": rejection_reasons,
                **result,
            }
            if accepted:
                route_class = classify_selected_routes(
                    evidence["selected_routes"]
                )
                document["source"] = "automated_static_expert_verified"
                document["generation"][
                    "candidate_only_until_expert_passes"
                ] = False
                document["verification"] = {
                    "expert_completed": True,
                    "collision_free": True,
                    "runner_return_code": 0,
                    "max_cross_track_error_m": max_cte,
                    "cross_track_limit_m": float(
                        args.max_cross_track_error_m
                    ),
                    "valid_plan_samples": evidence["valid_plan_samples"],
                    "selected_routes": evidence["selected_routes"],
                    "lane_compliance": lane_summary(lane_result),
                    "verified_at_unix": time.time(),
                    "log": portable_path(log_path),
                    "ego_trace": portable_path(trace_path),
                    "ego_trace_sha256": sha256_file(trace_path),
                }
                accepted_path = output_dir / (candidate_name + ".yaml")
                write_yaml_atomic(accepted_path, document)
                row["accepted_preset"] = portable_path(accepted_path)
                row["route_class"] = route_class
                row["sha256"] = sha256_file(accepted_path)
                manifest["accepted_count"] += 1
                for model in models:
                    manifest["accepted_model_counts"][model] += 1
                manifest["accepted_route_counts"][route_class] += 1
                avoidance = lane_result["trajectory_class"]
                row["avoidance_class"] = avoidance
                manifest["accepted_avoidance_counts"][avoidance] += 1
                achieved = "{}:{}".format(count, avoidance)
                manifest["accepted_cell_counts"][achieved] = (
                    manifest["accepted_cell_counts"].get(achieved, 0) + 1)
                manifest["accepted_presets"].append(
                    portable_path(accepted_path)
                )
                print("[ACCEPT] completed without collision: {} avoidance={} "
                      "max_right={}m marking_clearance={}m".format(
                          accepted_path, avoidance,
                          lane_result["max_right_deviation_m"],
                          lane_result["min_marking_clearance_m"]))
            else:
                row["failure_tail"] = _failure_tail(log_path)
                manifest["rejected_count"] += 1
                print("[REJECT] reasons={} return_code={} max_cte={}".format(
                    ",".join(rejection_reasons), result["return_code"], max_cte
                ))
                for line in row["failure_tail"][-5:]:
                    print("         {}".format(line))
            manifest["attempts"].append(row)
            manifest["completed_attempts"] = attempt
            write_json_atomic(manifest_path, manifest)

            model_coverage_met = all(
                count >= args.accepted_model_minimum
                for count in manifest["accepted_model_counts"].values()
            )
            if (
                args.keep_going_after is not None
                and manifest["accepted_count"] >= args.keep_going_after
                and model_coverage_met
            ):
                break
            if (
                route_targets is not None
                and model_coverage_met
                and all(
                    manifest["accepted_route_counts"][key] >= target
                    for key, target in route_targets.items()
                )
            ):
                print(
                    "[STATIC BATCH] accepted route targets reached: {}".format(
                        manifest["accepted_route_counts"]
                    )
                )
                break
    except KeyboardInterrupt:
        manifest["interrupted"] = True
        print("\n[STATIC BATCH] interrupted; completed results are preserved")
    finally:
        manifest["accepted_model_minimum_met"] = all(
            count >= args.accepted_model_minimum
            for count in manifest["accepted_model_counts"].values()
        )
        manifest["finished_at_unix"] = time.time()
        manifest["accepted_route_targets_met"] = (
            all(
                manifest["accepted_route_counts"][key] >= target
                for key, target in route_targets.items()
            ) if route_targets is not None else None
        )
        write_json_atomic(manifest_path, manifest)

    print("\n[STATIC BATCH DONE] attempts={} accepted={} rejected={}".format(
        len(manifest["attempts"]),
        manifest["accepted_count"],
        manifest["rejected_count"],
    ))
    print("[STATIC BATCH DONE] manifest={}".format(manifest_path))
    if manifest["interrupted"]:
        raise SystemExit(130)
    if not manifest["accepted_model_minimum_met"]:
        missing = {
            model: count for model, count in manifest["accepted_model_counts"].items()
            if count < args.accepted_model_minimum
        }
        print(
            "[STATIC BATCH INCOMPLETE] accepted model coverage was not met: "
            "minimum={} actual={}".format(args.accepted_model_minimum, missing),
            file=sys.stderr,
        )
        raise SystemExit(3)
    if route_targets is not None and not manifest["accepted_route_targets_met"]:
        print(
            "[STATIC BATCH INCOMPLETE] accepted route targets were not met: "
            "target={} actual={}".format(
                route_targets, manifest["accepted_route_counts"]
            ),
            file=sys.stderr,
        )
        raise SystemExit(3)


if __name__ == "__main__":
    main()
