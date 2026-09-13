"""
Stage 6E-1 run-oriented MORAI dataset helpers.

This module contains no ROS dependency so run indexing / summaries can be
unit-tested independently.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


RUN_RE = re.compile(r"^run_(\d{4})$")


def next_run_id(runs_root: Path) -> str:
    runs_root = Path(runs_root)
    max_index = 0

    if runs_root.exists():
        for child in runs_root.iterdir():
            if not child.is_dir():
                continue
            match = RUN_RE.match(child.name)
            if match:
                max_index = max(
                    max_index,
                    int(match.group(1)),
                )

    return "run_{:04d}".format(
        max_index + 1
    )


def parse_tags(values: Iterable[str]) -> List[str]:
    tags = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token and token not in tags:
                tags.append(token)
    return tags


def append_jsonl(
    path: Path,
    record: Dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with path.open(
        "a",
        encoding="utf-8",
        buffering=1,
    ) as handle:
        handle.write(
            json.dumps(
                record,
                separators=(",", ":"),
            )
            + "\n"
        )


def _stats(values):
    if not values:
        return None

    x = np.asarray(
        values,
        dtype=np.float64,
    )
    finite = np.isfinite(x)
    x = x[finite]

    if x.size == 0:
        return None

    return {
        "count": int(x.size),
        "min": float(np.min(x)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def _gps_xy_m(
    latitude,
    longitude,
    lat0,
    lon0,
):
    north = (
        float(latitude) - float(lat0)
    ) * 111320.0

    east = (
        float(longitude) - float(lon0)
    ) * 111320.0 * math.cos(
        math.radians(float(lat0))
    )

    return np.asarray(
        [east, north],
        dtype=np.float64,
    )


def summarize_manifest(
    manifest_path: Path,
) -> Dict:
    manifest_path = Path(manifest_path)

    rows = []
    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    summary = {
        "frames": len(rows),
        "ego_teacher": None,
        "gps": None,
    }

    # Deduplicate synchronized reuse using receive time first.
    ego_by_recv = {}
    for row in rows:
        ego = row.get(
            "ego_teacher"
        )
        if ego:
            ego_by_recv[
                int(
                    ego[
                        "receive_monotonic_ns"
                    ]
                )
            ] = ego

    ego = [
        ego_by_recv[key]
        for key in sorted(
            ego_by_recv
        )
    ]

    if ego:
        pos = np.asarray(
            [
                x["position_xyz"][:2]
                for x in ego
            ],
            dtype=np.float64,
        )
        raw_speed = np.asarray(
            [
                float(
                    x.get(
                        "speed_xy_mps",
                        0.0,
                    )
                )
                for x in ego
            ],
            dtype=np.float64,
        )
        heading = np.asarray(
            [
                float(
                    x[
                        "heading_deg"
                    ]
                )
                for x in ego
            ],
            dtype=np.float64,
        )
        recv = np.asarray(
            [
                int(
                    x[
                        "receive_monotonic_ns"
                    ]
                )
                for x in ego
            ],
            dtype=np.float64,
        ) / 1e9
        src = np.asarray(
            [
                int(
                    x[
                        "source_stamp_ns"
                    ]
                )
                if x.get(
                    "source_stamp_ns"
                ) is not None
                else np.nan
                for x in ego
            ],
            dtype=np.float64,
        ) / 1e9

        delta = np.diff(
            pos,
            axis=0,
        )
        path_distance = float(
            np.linalg.norm(
                delta,
                axis=1,
            ).sum()
        ) if len(pos) > 1 else 0.0

        recv_duration = (
            float(
                recv[-1] - recv[0]
            )
            if len(recv) > 1
            else 0.0
        )

        valid_src = np.isfinite(src)

        if (
            np.count_nonzero(
                valid_src
            ) >= 2
        ):
            src_valid = src[
                valid_src
            ]
            source_duration = float(
                src_valid[-1]
                - src_valid[0]
            )
        else:
            source_duration = None

        if (
            source_duration is not None
            and recv_duration > 0.0
        ):
            rtf = float(
                source_duration
                / recv_duration
            )
        else:
            rtf = None

        heading_unwrapped = np.unwrap(
            np.radians(
                heading
            )
        )

        summary[
            "ego_teacher"
        ] = {
            "unique_receive_samples": len(
                ego
            ),
            "net_displacement_m": float(
                np.linalg.norm(
                    pos[-1] - pos[0]
                )
            ),
            "cumulative_path_m": (
                path_distance
            ),
            "xy_bbox_span_m": float(
                np.linalg.norm(
                    np.max(pos, axis=0)
                    - np.min(pos, axis=0)
                )
            ),
            "receive_duration_s": (
                recv_duration
            ),
            "source_duration_s": (
                source_duration
            ),
            "source_to_receive_ratio": (
                rtf
            ),
            "raw_speed": _stats(
                raw_speed.tolist()
            ),
            "heading_unwrapped_span_deg": float(
                math.degrees(
                    float(
                        np.max(
                            heading_unwrapped
                        )
                        - np.min(
                            heading_unwrapped
                        )
                    )
                )
            ),
            "first_xy": pos[
                0
            ].tolist(),
            "last_xy": pos[
                -1
            ].tolist(),
        }

    gps_by_recv = {}
    for row in rows:
        gps = row.get("gps")
        if gps:
            gps_by_recv[
                int(
                    gps[
                        "receive_monotonic_ns"
                    ]
                )
            ] = gps

    gps = [
        gps_by_recv[key]
        for key in sorted(
            gps_by_recv
        )
    ]

    if gps:
        lat0 = float(
            gps[0][
                "latitude"
            ]
        )
        lon0 = float(
            gps[0][
                "longitude"
            ]
        )

        xy = np.asarray(
            [
                _gps_xy_m(
                    x["latitude"],
                    x["longitude"],
                    lat0,
                    lon0,
                )
                for x in gps
            ],
            dtype=np.float64,
        )

        summary["gps"] = {
            "samples": len(gps),
            "unique_latlon": len(
                set(
                    (
                        float(
                            x["latitude"]
                        ),
                        float(
                            x["longitude"]
                        ),
                    )
                    for x in gps
                )
            ),
            "net_displacement_m": float(
                np.linalg.norm(
                    xy[-1] - xy[0]
                )
            ),
            "xy_bbox_span_m": float(
                np.linalg.norm(
                    np.max(
                        xy,
                        axis=0,
                    )
                    - np.min(
                        xy,
                        axis=0,
                    )
                )
            ),
            "status_values": sorted(
                set(
                    int(
                        x.get(
                            "status",
                            -1,
                        )
                    )
                    for x in gps
                )
            ),
        }

    return summary
