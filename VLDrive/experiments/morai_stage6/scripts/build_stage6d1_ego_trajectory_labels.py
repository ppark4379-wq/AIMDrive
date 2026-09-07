#!/usr/bin/env python3
"""
Stage 6D-1 — build training-only ego trajectory labels from /Ego_topic.

Input
-----
manifest.jsonl produced by record_stage6d1_ros_teacher_dataset.py

Teacher coordinate convention
-----------------------------
MORAI EgoVehicleStatus heading is treated as:
    heading_deg = atan2(+Y_world, +X_world), degrees

This convention is empirically supported by the live probe:
- Ego moved ~60.24 m mainly in +Y
- heading stayed near 84 -> 92 deg

Canonical label frame:
    +x = forward
    +y = left

The builder performs a motion-vs-heading consistency diagnostic and reports
the angular error. If the diagnostic looks wrong, do not train from the
generated labels until the convention is rechecked.

Timebase contract
-----------------
- receive_monotonic_ns remains the causal cross-sensor synchronization clock.
- source_stamp_ns from /Ego_topic is the simulation-time clock used for
  future trajectory horizons (0.5 s, 1.0 s, ...).

This distinction is required when MORAI runs below real time.

These labels are training-only. /Ego_topic is not a runtime model input.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path

import numpy as np
from aimdrive.ros_training.teacher_integrity import validate_manifest


DEFAULT_HORIZONS_S = (
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
)


def wrap_deg(value):
    return (
        float(value) + 180.0
    ) % 360.0 - 180.0


def parse_horizons(text):
    values = tuple(
        float(x.strip())
        for x in text.split(",")
        if x.strip()
    )
    if not values:
        raise ValueError("No horizons supplied")
    if any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Horizons must be positive")
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise ValueError("Horizons must be increasing")
    return values


def interpolate_teacher(samples, target_ns):
    times = [
        sample["source_stamp_ns"]
        for sample in samples
    ]
    idx = bisect.bisect_left(
        times,
        int(target_ns),
    )

    if idx == 0:
        return samples[0] if times and int(target_ns) == times[0] else None
    if idx >= len(samples):
        return None

    a = samples[idx - 1]
    b = samples[idx]

    ta = int(
        a["source_stamp_ns"]
    )
    tb = int(
        b["source_stamp_ns"]
    )

    if tb <= ta:
        return b

    alpha = (
        int(target_ns) - ta
    ) / float(
        tb - ta
    )
    alpha = min(
        max(alpha, 0.0),
        1.0,
    )

    pa = np.asarray(
        a["position_xyz"],
        dtype=np.float64,
    )
    pb = np.asarray(
        b["position_xyz"],
        dtype=np.float64,
    )
    va = np.asarray(
        a["velocity_xyz"],
        dtype=np.float64,
    )
    vb = np.asarray(
        b["velocity_xyz"],
        dtype=np.float64,
    )

    return {
        "source_stamp_ns": int(
            target_ns
        ),
        "position_xyz": (
            pa + alpha * (pb - pa)
        ).tolist(),
        "velocity_xyz": (
            va + alpha * (vb - va)
        ).tolist(),
        "heading_deg": float(
            a["heading_deg"]
            + alpha
            * wrap_deg(
                b["heading_deg"]
                - a["heading_deg"]
            )
        ),
    }


def world_delta_to_forward_left(
    dx,
    dy,
    heading_deg,
):
    theta = math.radians(
        float(
            heading_deg
        )
    )
    c = math.cos(theta)
    s = math.sin(theta)

    forward = (
        c * dx
        + s * dy
    )
    left = (
        -s * dx
        + c * dy
    )

    return (
        float(forward),
        float(left),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
    )
    parser.add_argument(
        "--manifest",
        default="manifest.jsonl",
    )
    parser.add_argument(
        "--horizons",
        default="0.5,1.0,1.5,2.0,2.5",
    )
    parser.add_argument(
        "--output-manifest",
        default="manifest_stage6d1_labels.jsonl",
    )
    parser.add_argument(
        "--label-dir",
        default="labels_ego_teacher",
    )
    parser.add_argument(
        "--stats-file",
        default="ego_teacher_label_stats.json",
    )
    args = parser.parse_args()

    horizons = parse_horizons(
        args.horizons
    )

    dataset = Path(
        args.dataset
    ).expanduser().resolve()
    manifest = (
        dataset
        / args.manifest
    )

    rows = []
    with manifest.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    if not rows:
        raise RuntimeError(
            "No manifest rows"
        )

    # Stage6E must not bypass its teacher gate by invoking this builder directly.
    if any(row.get("schema_version") == "morai-stage6e-run-v1" for row in rows):
        integrity = validate_manifest(rows)
        if integrity["status"] != "pass":
            raise RuntimeError("Stage6E teacher validation {}: {}".format(
                integrity["status"], integrity["failures"]))

    # Build the teacher stream in source/simulation time.
    #
    # A synchronized Ego sample can be reused by more than one LiDAR-anchor
    # frame, so duplicate source stamps are expected.  Deduplicate them here.
    # We also inspect source stamps in receive order first so a backwards
    # simulation-time jump cannot be hidden by sorting.
    teacher_by_recv = {}

    for row in rows:
        teacher = row.get(
            "ego_teacher"
        )
        if not teacher:
            continue

        recv_key = int(
            teacher[
                "receive_monotonic_ns"
            ]
        )
        teacher_by_recv[
            recv_key
        ] = teacher

    teachers_receive_order = [
        teacher_by_recv[key]
        for key in sorted(
            teacher_by_recv
        )
    ]

    source_stamps_receive_order = [
        int(
            sample[
                "source_stamp_ns"
            ]
        )
        for sample in teachers_receive_order
    ]

    negative_source_jumps = sum(
        1
        for a, b in zip(
            source_stamps_receive_order,
            source_stamps_receive_order[1:],
        )
        if b < a
    )

    zero_source_steps = sum(
        1
        for a, b in zip(
            source_stamps_receive_order,
            source_stamps_receive_order[1:],
        )
        if b == a
    )

    if negative_source_jumps:
        raise RuntimeError(
            "Ego source_stamp_ns moved backwards "
            f"{negative_source_jumps} times. "
            "Do not build trajectory labels until the simulation-time "
            "source is stable."
        )

    teacher_by_source = {}

    for teacher in teachers_receive_order:
        source_key = int(
            teacher[
                "source_stamp_ns"
            ]
        )
        teacher_by_source[
            source_key
        ] = teacher

    teachers = [
        teacher_by_source[key]
        for key in sorted(
            teacher_by_source
        )
    ]

    if len(teachers) < 2:
        raise RuntimeError(
            "Need at least two unique Ego source-time samples"
        )

    receive_duration_s = (
        int(
            teachers_receive_order[-1][
                "receive_monotonic_ns"
            ]
        )
        - int(
            teachers_receive_order[0][
                "receive_monotonic_ns"
            ]
        )
    ) / 1e9

    source_duration_s = (
        int(
            teachers[-1][
                "source_stamp_ns"
            ]
        )
        - int(
            teachers[0][
                "source_stamp_ns"
            ]
        )
    ) / 1e9

    source_to_receive_ratio = (
        source_duration_s
        / receive_duration_s
        if receive_duration_s > 0.0
        else None
    )

    label_dir = (
        dataset
        / args.label_dir
    )
    label_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_manifest = (
        dataset
        / args.output_manifest
    )

    written = 0
    skipped_future = 0
    heading_errors_deg = []
    endpoint_forward_m = []
    endpoint_left_m = []

    with output_manifest.open(
        "w",
        encoding="utf-8",
    ) as output:
        for row in rows:
            teacher = row.get(
                "ego_teacher"
            )
            if not teacher:
                continue

            current_ns = int(
                teacher[
                    "source_stamp_ns"
                ]
            )
            current_position = np.asarray(
                teacher[
                    "position_xyz"
                ],
                dtype=np.float64,
            )
            current_heading = float(
                teacher[
                    "heading_deg"
                ]
            )

            future = []

            valid = True
            for horizon in horizons:
                target_ns = (
                    current_ns
                    + int(
                        horizon * 1e9
                    )
                )

                sample = interpolate_teacher(
                    teachers,
                    target_ns,
                )

                if sample is None:
                    valid = False
                    break

                future.append(
                    sample
                )

            if not valid:
                skipped_future += 1
                continue

            trajectory = []

            for sample in future:
                future_position = np.asarray(
                    sample[
                        "position_xyz"
                    ],
                    dtype=np.float64,
                )
                delta = (
                    future_position
                    - current_position
                )

                forward, left = (
                    world_delta_to_forward_left(
                        float(delta[0]),
                        float(delta[1]),
                        current_heading,
                    )
                )

                trajectory.append(
                    [
                        forward,
                        left,
                    ]
                )

            trajectory = np.asarray(
                trajectory,
                dtype=np.float32,
            )

            future_speed = np.asarray(
                [
                    math.hypot(
                        float(
                            sample[
                                "velocity_xyz"
                            ][0]
                        ),
                        float(
                            sample[
                                "velocity_xyz"
                            ][1]
                        ),
                    )
                    for sample in future
                ],
                dtype=np.float32,
            )

            # Heading-convention sanity diagnostic.
            #
            # Use the shortest requested horizon rather than the full
            # trajectory endpoint so normal curved driving does not inflate
            # the frame-convention error.
            short_delta = (
                np.asarray(
                    future[0][
                        "position_xyz"
                    ],
                    dtype=np.float64,
                )
                - current_position
            )
            short_distance = float(
                np.linalg.norm(
                    short_delta[:2]
                )
            )

            if short_distance >= 0.5:
                motion_heading = math.degrees(
                    math.atan2(
                        float(
                            short_delta[1]
                        ),
                        float(
                            short_delta[0]
                        ),
                    )
                )
                heading_errors_deg.append(
                    abs(
                        wrap_deg(
                            motion_heading
                            - current_heading
                        )
                    )
                )

            endpoint_forward_m.append(
                float(
                    trajectory[
                        -1,
                        0,
                    ]
                )
            )
            endpoint_left_m.append(
                float(
                    trajectory[
                        -1,
                        1,
                    ]
                )
            )

            frame_index = int(
                row[
                    "frame_index"
                ]
            )
            label_rel = (
                Path(
                    args.label_dir
                )
                / (
                    f"{frame_index:08d}.npz"
                )
            )

            np.savez_compressed(
                dataset
                / label_rel,
                trajectory_xy_forward_left=(
                    trajectory
                ),
                future_speed_mps=(
                    future_speed
                ),
                horizons_s=np.asarray(
                    horizons,
                    dtype=np.float32,
                ),
                current_speed_mps=np.asarray(
                    [
                        float(
                            teacher[
                                "speed_xy_mps"
                            ]
                        )
                    ],
                    dtype=np.float32,
                ),
                current_heading_deg=np.asarray(
                    [
                        current_heading
                    ],
                    dtype=np.float32,
                ),
            )

            enriched = dict(
                row
            )
            enriched[
                "labels"
            ] = {
                "schema_version": (
                    "morai-stage6d1-ego-teacher-v2"
                ),
                "teacher_only_source": (
                    "/Ego_topic"
                ),
                "runtime_model_input": False,
                "path": str(
                    label_rel
                ),
                "trajectory_frame": (
                    "+x forward, +y left"
                ),
                "trajectory_timebase": (
                    "ego_source_stamp_ns"
                ),
                "horizons_s": list(
                    horizons
                ),
                "trajectory_shape": [
                    len(horizons),
                    2,
                ],
            }

            output.write(
                json.dumps(
                    enriched,
                    separators=(
                        ",",
                        ":",
                    ),
                )
                + "\n"
            )
            written += 1

    def stats(values):
        if not values:
            return None
        x = np.asarray(
            values,
            dtype=np.float64,
        )
        return {
            "count": int(
                x.size
            ),
            "min": float(
                x.min()
            ),
            "median": float(
                np.median(x)
            ),
            "mean": float(
                x.mean()
            ),
            "p95": float(
                np.percentile(
                    x,
                    95
                )
            ),
            "max": float(
                x.max()
            ),
        }

    summary = {
        "schema_version": (
            "morai-stage6d1-ego-teacher-v2"
        ),
        "input_frames": len(
            rows
        ),
        "unique_teacher_samples": len(
            teachers
        ),
        "unique_teacher_receive_samples": len(
            teachers_receive_order
        ),
        "teacher_receive_duration_s": float(
            receive_duration_s
        ),
        "teacher_source_duration_s": float(
            source_duration_s
        ),
        "source_to_receive_duration_ratio": (
            None
            if source_to_receive_ratio is None
            else float(
                source_to_receive_ratio
            )
        ),
        "negative_source_time_jumps": int(
            negative_source_jumps
        ),
        "duplicate_source_time_steps": int(
            zero_source_steps
        ),
        "trajectory_timebase": (
            "ego_source_stamp_ns"
        ),
        "written_labels": (
            written
        ),
        "skipped_no_full_future": (
            skipped_future
        ),
        "horizons_s": list(
            horizons
        ),
        "heading_vs_motion_abs_error_deg": (
            stats(
                heading_errors_deg
            )
        ),
        "endpoint_forward_m": (
            stats(
                endpoint_forward_m
            )
        ),
        "endpoint_left_m": (
            stats(
                endpoint_left_m
            )
        ),
        "notes": [
            (
                "/Ego_topic is privileged training teacher data only."
            ),
            (
                "Do not make model/runtime inference depend on EgoVehicleStatus."
            ),
            (
                "Trajectory horizons use Ego source_stamp_ns (simulation time), "
                "not receive_monotonic_ns."
            ),
            (
                "receive_monotonic_ns remains the causal cross-sensor sync clock."
            ),
            (
                "Heading convention is validated against shortest-horizon motion."
            ),
        ],
    }

    stats_path = (
        dataset
        / args.stats_file
    )
    stats_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 104)
    print(
        "MORAI Stage 6D-1 — EGO TEACHER TRAJECTORY LABELS"
    )
    print("=" * 104)
    print(
        "input frames              :",
        summary[
            "input_frames"
        ],
    )
    print(
        "unique teacher samples    :",
        summary[
            "unique_teacher_samples"
        ],
    )
    print(
        "teacher recv duration [s] :",
        summary[
            "teacher_receive_duration_s"
        ],
    )
    print(
        "teacher src duration [s]  :",
        summary[
            "teacher_source_duration_s"
        ],
    )
    print(
        "src/recv duration ratio   :",
        summary[
            "source_to_receive_duration_ratio"
        ],
    )
    print(
        "source time duplicates    :",
        summary[
            "duplicate_source_time_steps"
        ],
    )
    print(
        "written labels            :",
        summary[
            "written_labels"
        ],
    )
    print(
        "skipped no future         :",
        summary[
            "skipped_no_full_future"
        ],
    )
    print(
        "heading-motion error [deg]:",
        summary[
            "heading_vs_motion_abs_error_deg"
        ],
    )
    print(
        "endpoint forward [m]      :",
        summary[
            "endpoint_forward_m"
        ],
    )
    print(
        "endpoint left [m]         :",
        summary[
            "endpoint_left_m"
        ],
    )
    print()
    print(
        "manifest:",
        output_manifest,
    )
    print(
        "stats   :",
        stats_path,
    )
    print("=" * 104)


if __name__ == "__main__":
    main()
