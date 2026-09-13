#!/usr/bin/env python3
"""
Stage 6E-1 offline route enrichment.

Input:
    one run's Stage 6D-1 v2 label manifest

Training localization:
    /Ego_topic.position + /Ego_topic.heading

Output model-side route representation:
    [41,2] by default
    +x ego forward
    +y ego left

/Ego_topic is privileged training localization only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from aimdrive.route.competition_global_path import (
    CompetitionGlobalPath,
)
from aimdrive.route.route_enrichment import (
    CyclicRouteMatcher,
    RouteEnrichmentConfig,
)


LABEL_SCHEMA = (
    "morai-stage6d1-ego-teacher-v2"
)


def world_delta_to_ego(
    delta_xy,
    heading_deg,
):
    theta = math.radians(
        float(
            heading_deg
        )
    )
    c = math.cos(theta)
    s = math.sin(theta)

    dx = delta_xy[:, 0]
    dy = delta_xy[:, 1]

    forward = c * dx + s * dy
    left = -s * dx + c * dy

    return np.stack(
        [forward, left],
        axis=-1,
    ).astype(np.float32)


def wrap_deg(value):
    return (
        float(value) + 180.0
    ) % 360.0 - 180.0


def stats(values):
    if not values:
        return None
    x = np.asarray(
        values,
        dtype=np.float64,
    )
    return {
        "count": int(x.size),
        "min": float(np.min(x)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
    )
    parser.add_argument(
        "--manifest",
        default="manifest_labels.jsonl",
    )
    parser.add_argument(
        "--path",
        default="data/competition/global_path.txt",
    )
    parser.add_argument(
        "--output-manifest",
        default="manifest_ready.jsonl",
    )
    parser.add_argument(
        "--route-dir",
        default="route_ego_teacher",
    )
    parser.add_argument(
        "--stats-file",
        default="route_stats.json",
    )
    parser.add_argument(
        "--lookahead-m",
        type=float,
        default=40.0,
    )
    parser.add_argument(
        "--spacing-m",
        type=float,
        default=1.0,
    )
    args = parser.parse_args()

    dataset = Path(
        args.dataset
    ).expanduser().resolve()
    manifest = dataset / args.manifest
    path_file = Path(
        args.path
    ).expanduser().resolve()

    path = CompetitionGlobalPath.load(
        path_file
    )

    matcher = CyclicRouteMatcher(
        path,
        RouteEnrichmentConfig(
            lookahead_m=args.lookahead_m,
            spacing_m=args.spacing_m,
            continuity_backward_m=8.0,
            continuity_forward_m=30.0,
            global_reacquire_distance_m=12.0,
        ),
    )

    expected_points = int(
        round(
            args.lookahead_m
            / args.spacing_m
        )
    ) + 1

    route_dir = (
        dataset
        / args.route_dir
    )
    route_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_manifest = (
        dataset
        / args.output_manifest
    )

    distances = []
    heading_errors = []
    route_s = []
    reacquires = 0
    written = 0

    with manifest.open(
        "r",
        encoding="utf-8",
    ) as source, output_manifest.open(
        "w",
        encoding="utf-8",
    ) as output:
        for line in source:
            if not line.strip():
                continue

            row = json.loads(line)
            labels = row.get(
                "labels"
            ) or {}
            teacher = row.get(
                "ego_teacher"
            )

            if (
                labels.get(
                    "schema_version"
                )
                != LABEL_SCHEMA
            ):
                raise RuntimeError(
                    "Expected fixed Stage 6D-1 v2 labels"
                )

            if not teacher:
                raise RuntimeError(
                    "Missing training ego_teacher"
                )

            ego_xy = np.asarray(
                teacher[
                    "position_xyz"
                ][:2],
                dtype=np.float64,
            )
            heading_deg = float(
                teacher[
                    "heading_deg"
                ]
            )

            match = matcher.match(
                ego_xy
            )
            sampled_map_xyz = (
                matcher.sample_forward(
                    match
                )
            )

            delta = (
                sampled_map_xyz[:, :2]
                - ego_xy[None, :]
            )

            ego_frame_xy = (
                world_delta_to_ego(
                    delta,
                    heading_deg,
                )
            )

            if (
                tuple(
                    ego_frame_xy.shape
                )
                != (
                    expected_points,
                    2,
                )
            ):
                raise RuntimeError(
                    "Unexpected route shape {}".format(
                        tuple(
                            ego_frame_xy.shape
                        )
                    )
                )

            frame_index = int(
                row[
                    "frame_index"
                ]
            )

            route_rel = (
                Path(
                    args.route_dir
                )
                / "{:08d}.npz".format(
                    frame_index
                )
            )

            np.savez_compressed(
                dataset / route_rel,
                ego_frame_xy=(
                    ego_frame_xy
                ),
                map_xyz=(
                    sampled_map_xyz
                    .astype(
                        np.float32
                    )
                ),
                ego_position_xy=(
                    ego_xy
                ),
                ego_heading_deg=np.asarray(
                    [heading_deg],
                    dtype=np.float64,
                ),
            )

            tangent_deg = (
                math.degrees(
                    match.tangent_heading_rad
                )
            )
            heading_error = abs(
                wrap_deg(
                    heading_deg
                    - tangent_deg
                )
            )

            enriched = dict(
                row
            )
            enriched[
                "route_input"
            ] = {
                "schema_version": (
                    "morai-stage6e-route-v1"
                ),
                "path": str(
                    route_rel
                ),
                "shape": [
                    expected_points,
                    2,
                ],
                "frame": (
                    "+x ego forward, +y ego left"
                ),
                "lookahead_m": float(
                    args.lookahead_m
                ),
                "spacing_m": float(
                    args.spacing_m
                ),
                "localization_source": (
                    "/Ego_topic"
                ),
                "localization_teacher_only": True,
                "runtime_localization_dependency": False,
                "route_s_m": float(
                    match.route_s_m
                ),
                "distance_to_path_m": float(
                    match.distance_m
                ),
                "signed_lateral_offset_m": float(
                    match.signed_lateral_offset_m
                ),
                "path_tangent_heading_deg": float(
                    tangent_deg
                ),
                "heading_vs_path_abs_error_deg": float(
                    heading_error
                ),
                "reacquired_globally": bool(
                    match.reacquired_globally
                ),
            }

            output.write(
                json.dumps(
                    enriched,
                    separators=(",", ":"),
                )
                + "\n"
            )

            distances.append(
                float(
                    match.distance_m
                )
            )
            heading_errors.append(
                float(
                    heading_error
                )
            )
            route_s.append(
                float(
                    match.route_s_m
                )
            )

            if (
                match.reacquired_globally
            ):
                reacquires += 1

            written += 1

    summary = {
        "schema_version": (
            "morai-stage6e-route-stats-v1"
        ),
        "frames": written,
        "path_total_length_m": float(
            path.total_length_m
        ),
        "route_shape": [
            expected_points,
            2,
        ],
        "distance_to_path_m": stats(
            distances
        ),
        "heading_vs_path_abs_error_deg": stats(
            heading_errors
        ),
        "global_reacquires": int(
            reacquires
        ),
        "route_s_first_m": (
            float(route_s[0])
            if route_s
            else None
        ),
        "route_s_last_m": (
            float(route_s[-1])
            if route_s
            else None
        ),
        "training_localization": (
            "/Ego_topic teacher only"
        ),
    }

    (
        dataset
        / args.stats_file
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 108)
    print(
        "MORAI Stage 6E-1 — OFFLINE TRAINING ROUTE ENRICHMENT"
    )
    print("=" * 108)
    print("frames                  :", written)
    print("route shape             :", (expected_points, 2))
    print(
        "distance/path          :",
        summary[
            "distance_to_path_m"
        ],
    )
    print(
        "heading/path           :",
        summary[
            "heading_vs_path_abs_error_deg"
        ],
    )
    print(
        "global reacquires      :",
        reacquires,
    )
    print(
        "route_s first -> last  :",
        summary[
            "route_s_first_m"
        ],
        "->",
        summary[
            "route_s_last_m"
        ],
    )
    print(
        "output manifest        :",
        output_manifest,
    )
    print(
        "OFFLINE ROUTE CONTRACT : PASS"
    )
    print("=" * 108)


if __name__ == "__main__":
    main()
