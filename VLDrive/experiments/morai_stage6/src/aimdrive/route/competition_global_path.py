"""
Competition Global Path contract for MORAI Stage 6C-0.

Observed file format:
    one waypoint per line
    x y z
    float float float

Important observed properties of the supplied competition path:
- approximately 0.5 m waypoint spacing;
- first and last waypoint are identical -> closed loop;
- consecutive duplicate waypoints exist at segment boundaries.

This module:
1. parses the path;
2. removes consecutive duplicate points for geometry operations;
3. keeps a closed-loop flag;
4. builds cumulative arc length;
5. supports nearest-point lookup;
6. samples forward along the route with cyclic wrap-around.

No assumption is made here about final Route Encoder architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import math

import numpy as np


@dataclass(frozen=True)
class GlobalPathStats:
    raw_count: int
    geometry_count: int
    consecutive_duplicate_count: int
    is_closed: bool
    closure_distance_m: float
    total_length_m: float
    step_min_m: float
    step_median_m: float
    step_mean_m: float
    step_max_m: float
    xyz_min: Tuple[float, float, float]
    xyz_max: Tuple[float, float, float]


@dataclass
class CompetitionGlobalPath:
    raw_xyz: np.ndarray
    xyz: np.ndarray
    is_closed: bool
    cumulative_s_m: np.ndarray
    segment_length_m: np.ndarray
    total_length_m: float

    @classmethod
    def load(
        cls,
        path: Path,
        duplicate_tolerance_m: float = 1e-8,
        closure_tolerance_m: float = 1e-6,
    ) -> "CompetitionGlobalPath":
        path = Path(path).expanduser().resolve()

        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                fields = stripped.split()

                if len(fields) != 3:
                    raise ValueError(
                        f"{path}:{line_number}: expected 3 columns x y z, "
                        f"got {len(fields)}"
                    )

                try:
                    rows.append(
                        [
                            float(fields[0]),
                            float(fields[1]),
                            float(fields[2]),
                        ]
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: non-numeric waypoint"
                    ) from exc

        raw = np.asarray(rows, dtype=np.float64)

        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError("Global Path must be Nx3")
        if len(raw) < 2:
            raise ValueError("Global Path must contain at least two points")
        if not np.isfinite(raw).all():
            raise ValueError("Global Path contains non-finite values")

        closure_distance = float(
            np.linalg.norm(raw[-1, :2] - raw[0, :2])
        )
        is_closed = closure_distance <= closure_tolerance_m

        keep = np.ones(len(raw), dtype=bool)

        if len(raw) > 1:
            delta = np.linalg.norm(
                np.diff(raw[:, :2], axis=0),
                axis=1,
            )
            keep[1:] = delta > duplicate_tolerance_m

        geometry = raw[keep].copy()

        # If the final point repeats the first point, remove that repeated
        # terminal point. Closure is represented explicitly by is_closed.
        if (
            is_closed
            and len(geometry) >= 2
            and np.linalg.norm(
                geometry[-1, :2] - geometry[0, :2]
            ) <= closure_tolerance_m
        ):
            geometry = geometry[:-1]

        if len(geometry) < 2:
            raise ValueError("Global Path collapsed after duplicate removal")

        if is_closed:
            segment_vector = np.diff(
                np.vstack(
                    [geometry[:, :2], geometry[:1, :2]]
                ),
                axis=0,
            )
        else:
            segment_vector = np.diff(
                geometry[:, :2],
                axis=0,
            )

        segment_length = np.linalg.norm(
            segment_vector,
            axis=1,
        )

        if np.any(segment_length <= duplicate_tolerance_m):
            raise ValueError(
                "Zero-length segment remained after duplicate removal"
            )

        if is_closed:
            cumulative = np.concatenate(
                [
                    np.zeros(1, dtype=np.float64),
                    np.cumsum(segment_length[:-1]),
                ]
            )
            total_length = float(segment_length.sum())
        else:
            cumulative = np.concatenate(
                [
                    np.zeros(1, dtype=np.float64),
                    np.cumsum(segment_length),
                ]
            )
            total_length = float(cumulative[-1])

        return cls(
            raw_xyz=raw,
            xyz=geometry,
            is_closed=is_closed,
            cumulative_s_m=cumulative,
            segment_length_m=segment_length,
            total_length_m=total_length,
        )

    def stats(self) -> GlobalPathStats:
        raw_step = np.linalg.norm(
            np.diff(self.raw_xyz[:, :2], axis=0),
            axis=1,
        )

        duplicate_count = int(
            np.count_nonzero(raw_step <= 1e-8)
        )

        positive = raw_step[raw_step > 1e-8]

        closure_distance = float(
            np.linalg.norm(
                self.raw_xyz[-1, :2]
                - self.raw_xyz[0, :2]
            )
        )

        return GlobalPathStats(
            raw_count=int(len(self.raw_xyz)),
            geometry_count=int(len(self.xyz)),
            consecutive_duplicate_count=duplicate_count,
            is_closed=bool(self.is_closed),
            closure_distance_m=closure_distance,
            total_length_m=float(self.total_length_m),
            step_min_m=float(positive.min()),
            step_median_m=float(np.median(positive)),
            step_mean_m=float(positive.mean()),
            step_max_m=float(positive.max()),
            xyz_min=tuple(
                float(v)
                for v in self.raw_xyz.min(axis=0)
            ),
            xyz_max=tuple(
                float(v)
                for v in self.raw_xyz.max(axis=0)
            ),
        )

    def nearest_index(
        self,
        xy_m: np.ndarray,
    ) -> Tuple[int, float]:
        xy_m = np.asarray(xy_m, dtype=np.float64).reshape(2)

        distance = np.linalg.norm(
            self.xyz[:, :2] - xy_m[None, :],
            axis=1,
        )

        index = int(np.argmin(distance))

        return index, float(distance[index])

    def tangent_heading_rad(
        self,
        index: int,
    ) -> float:
        n = len(self.xyz)

        if not 0 <= index < n:
            raise IndexError(index)

        if self.is_closed:
            previous = self.xyz[(index - 1) % n, :2]
            following = self.xyz[(index + 1) % n, :2]
        else:
            previous = self.xyz[max(index - 1, 0), :2]
            following = self.xyz[min(index + 1, n - 1), :2]

        delta = following - previous

        if np.linalg.norm(delta) <= 1e-9:
            raise ValueError("Cannot determine tangent at waypoint")

        return float(math.atan2(delta[1], delta[0]))

    def _point_at_s(
        self,
        query_s_m: float,
    ) -> np.ndarray:
        if self.is_closed:
            query = float(query_s_m % self.total_length_m)
        else:
            query = float(
                min(max(query_s_m, 0.0), self.total_length_m)
            )

        n = len(self.xyz)

        if self.is_closed:
            # cumulative_s_m has one entry per geometry waypoint.
            index = int(
                np.searchsorted(
                    self.cumulative_s_m,
                    query,
                    side="right",
                ) - 1
            )
            index = max(0, min(index, n - 1))

            next_index = (index + 1) % n

            s0 = float(self.cumulative_s_m[index])
            seg_len = float(self.segment_length_m[index])

            local = query - s0
            if local < 0.0:
                local += self.total_length_m

            alpha = min(max(local / seg_len, 0.0), 1.0)

            return (
                self.xyz[index]
                + alpha * (
                    self.xyz[next_index] - self.xyz[index]
                )
            )

        if query >= self.total_length_m:
            return self.xyz[-1].copy()

        index = int(
            np.searchsorted(
                self.cumulative_s_m,
                query,
                side="right",
            ) - 1
        )
        index = max(0, min(index, n - 2))

        s0 = float(self.cumulative_s_m[index])
        seg_len = float(self.segment_length_m[index])
        alpha = (query - s0) / seg_len

        return (
            self.xyz[index]
            + alpha * (
                self.xyz[index + 1] - self.xyz[index]
            )
        )

    def sample_forward_from_index(
        self,
        start_index: int,
        *,
        lookahead_m: float,
        spacing_m: float,
    ) -> np.ndarray:
        """
        Return Nx3 route points sampled forward along path arc length.

        This is map-frame geometry. Ego-frame conversion is deliberately
        separate because the final yaw source is not yet fixed.
        """
        if lookahead_m <= 0:
            raise ValueError("lookahead_m must be positive")
        if spacing_m <= 0:
            raise ValueError("spacing_m must be positive")

        n = len(self.xyz)
        if not 0 <= start_index < n:
            raise IndexError(start_index)

        start_s = float(self.cumulative_s_m[start_index])

        distances = np.arange(
            0.0,
            lookahead_m + 1e-9,
            spacing_m,
            dtype=np.float64,
        )

        if distances[-1] < lookahead_m - 1e-9:
            distances = np.concatenate(
                [distances, [lookahead_m]]
            )

        return np.stack(
            [
                self._point_at_s(start_s + float(ds))
                for ds in distances
            ],
            axis=0,
        )


def gps_wgs84_to_morai_local_utm52n(
    *,
    latitude_deg: float,
    longitude_deg: float,
    east_offset_m: float,
    north_offset_m: float,
) -> Tuple[float, float]:
    """
    Candidate conversion supported by the current observed GPS sample.

    WGS84 -> UTM Zone 52N
    then subtract MORAI GPSMessage eastOffset/northOffset.

    This function imports pyproj lazily so ROS acquisition itself does not
    need pyproj. The final coordinate contract should still be validated
    against multiple moving GPS samples.
    """
    try:
        from pyproj import Transformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyproj is required for WGS84 -> UTM52N conversion"
        ) from exc

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:32652",
        always_xy=True,
    )

    easting, northing = transformer.transform(
        float(longitude_deg),
        float(latitude_deg),
    )

    return (
        float(easting - east_offset_m),
        float(northing - north_offset_m),
    )
