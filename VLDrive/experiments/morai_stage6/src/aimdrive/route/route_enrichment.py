"""
Stage 6C-1 — attach competition Global Path geometry to each recorded MORAI frame.

This module works offline on the Stage 6B-2 dataset.

Key choices
-----------
1. GPS WGS84 -> UTM52N -> subtract eastOffset/northOffset.
2. Match GPS position to the CLOSED competition route by segment projection,
   not only nearest waypoint.
3. Maintain cyclic route continuity across frames to avoid accidentally
   jumping to a nearby parallel/crossing segment.
4. Sample future route from the projected arc-length position.
5. Save:
     - map-frame sampled XYZ
     - GPS-relative map-frame XY
     - path-tangent-frame XY
   The tangent-frame representation is NOT called ego frame because the final
   runtime ego yaw source has not yet been validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math

import numpy as np

from aimdrive.route.competition_global_path import (
    CompetitionGlobalPath,
    gps_wgs84_to_morai_local_utm52n,
)


@dataclass(frozen=True)
class RouteMatch:
    segment_index: int
    segment_alpha: float
    route_s_m: float
    projected_xy_m: Tuple[float, float]
    distance_m: float
    signed_lateral_offset_m: float
    tangent_heading_rad: float
    reacquired_globally: bool


@dataclass(frozen=True)
class RouteEnrichmentConfig:
    lookahead_m: float = 40.0
    spacing_m: float = 1.0
    continuity_backward_m: float = 8.0
    continuity_forward_m: float = 30.0
    global_reacquire_distance_m: float = 12.0

    def validate(self) -> None:
        if self.lookahead_m <= 0.0:
            raise ValueError("lookahead_m must be positive")
        if self.spacing_m <= 0.0:
            raise ValueError("spacing_m must be positive")
        if self.continuity_backward_m < 0.0:
            raise ValueError("continuity_backward_m must be >= 0")
        if self.continuity_forward_m <= 0.0:
            raise ValueError("continuity_forward_m must be positive")
        if self.global_reacquire_distance_m <= 0.0:
            raise ValueError("global_reacquire_distance_m must be positive")


class CyclicRouteMatcher:
    def __init__(
        self,
        path: CompetitionGlobalPath,
        config: RouteEnrichmentConfig,
    ) -> None:
        if not path.is_closed:
            raise ValueError(
                "Stage 6C-1 matcher currently expects the supplied closed-loop competition path"
            )

        config.validate()

        self.path = path
        self.config = config
        self.previous_s_m: Optional[float] = None

        self.p0 = path.xyz[:, :2].astype(np.float64)
        self.p1 = np.roll(self.p0, -1, axis=0)

        self.segment_vector = self.p1 - self.p0
        self.segment_length = np.linalg.norm(
            self.segment_vector,
            axis=1,
        )
        self.segment_length_sq = (
            self.segment_length * self.segment_length
        )

        if np.any(self.segment_length <= 1e-9):
            raise ValueError("Route contains a zero-length geometry segment")

        self.segment_start_s = path.cumulative_s_m.astype(
            np.float64
        )
        self.total_length_m = float(path.total_length_m)

    def _candidate_mask(self) -> np.ndarray:
        if self.previous_s_m is None:
            return np.ones(
                len(self.p0),
                dtype=bool,
            )

        previous = float(self.previous_s_m)

        forward = (
            self.segment_start_s - previous
        ) % self.total_length_m

        backward = (
            previous - self.segment_start_s
        ) % self.total_length_m

        return (
            (forward <= self.config.continuity_forward_m)
            | (backward <= self.config.continuity_backward_m)
        )

    def _project(
        self,
        xy_m: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> RouteMatch:
        xy = np.asarray(
            xy_m,
            dtype=np.float64,
        ).reshape(2)

        indices = np.flatnonzero(
            candidate_mask
        )

        if len(indices) == 0:
            raise RuntimeError("No candidate route segments")

        p0 = self.p0[indices]
        vector = self.segment_vector[indices]
        length_sq = self.segment_length_sq[indices]

        alpha = (
            np.sum(
                (xy[None, :] - p0) * vector,
                axis=1,
            )
            / length_sq
        )

        alpha = np.clip(
            alpha,
            0.0,
            1.0,
        )

        projection = (
            p0
            + alpha[:, None] * vector
        )

        residual = (
            xy[None, :] - projection
        )

        distance = np.linalg.norm(
            residual,
            axis=1,
        )

        local_best = int(
            np.argmin(distance)
        )
        segment_index = int(
            indices[local_best]
        )
        segment_alpha = float(
            alpha[local_best]
        )
        projected = projection[
            local_best
        ]
        distance_m = float(
            distance[local_best]
        )

        tangent = (
            self.segment_vector[
                segment_index
            ]
            / self.segment_length[
                segment_index
            ]
        )

        tangent_heading = float(
            math.atan2(
                tangent[1],
                tangent[0],
            )
        )

        # Positive = GPS is to the LEFT of route tangent.
        offset = xy - projected

        signed_lateral = float(
            tangent[0] * offset[1]
            - tangent[1] * offset[0]
        )

        route_s = (
            float(
                self.segment_start_s[
                    segment_index
                ]
            )
            + segment_alpha
            * float(
                self.segment_length[
                    segment_index
                ]
            )
        ) % self.total_length_m

        return RouteMatch(
            segment_index=segment_index,
            segment_alpha=segment_alpha,
            route_s_m=route_s,
            projected_xy_m=(
                float(projected[0]),
                float(projected[1]),
            ),
            distance_m=distance_m,
            signed_lateral_offset_m=signed_lateral,
            tangent_heading_rad=tangent_heading,
            reacquired_globally=False,
        )

    def match(
        self,
        xy_m: np.ndarray,
    ) -> RouteMatch:
        continuity_mask = (
            self._candidate_mask()
        )

        local_match = self._project(
            xy_m,
            continuity_mask,
        )

        reacquired = False

        if (
            self.previous_s_m is not None
            and local_match.distance_m
            > self.config.global_reacquire_distance_m
        ):
            global_match = self._project(
                xy_m,
                np.ones(
                    len(self.p0),
                    dtype=bool,
                ),
            )

            if (
                global_match.distance_m
                < local_match.distance_m
            ):
                local_match = global_match
                reacquired = True

        self.previous_s_m = (
            local_match.route_s_m
        )

        if not reacquired:
            return local_match

        return RouteMatch(
            segment_index=local_match.segment_index,
            segment_alpha=local_match.segment_alpha,
            route_s_m=local_match.route_s_m,
            projected_xy_m=local_match.projected_xy_m,
            distance_m=local_match.distance_m,
            signed_lateral_offset_m=local_match.signed_lateral_offset_m,
            tangent_heading_rad=local_match.tangent_heading_rad,
            reacquired_globally=True,
        )

    def _point_at_s(
        self,
        query_s_m: float,
    ) -> np.ndarray:
        query = float(
            query_s_m
            % self.total_length_m
        )

        index = int(
            np.searchsorted(
                self.segment_start_s,
                query,
                side="right",
            )
            - 1
        )

        index = max(
            0,
            min(
                index,
                len(self.p0) - 1,
            ),
        )

        s0 = float(
            self.segment_start_s[
                index
            ]
        )

        local = query - s0

        if local < 0.0:
            local += self.total_length_m

        alpha = min(
            max(
                local
                / float(
                    self.segment_length[
                        index
                    ]
                ),
                0.0,
            ),
            1.0,
        )

        next_index = (
            index + 1
        ) % len(self.path.xyz)

        return (
            self.path.xyz[index]
            + alpha
            * (
                self.path.xyz[next_index]
                - self.path.xyz[index]
            )
        )

    def sample_forward(
        self,
        match: RouteMatch,
    ) -> np.ndarray:
        distances = np.arange(
            0.0,
            self.config.lookahead_m
            + 1e-9,
            self.config.spacing_m,
            dtype=np.float64,
        )

        if (
            distances[-1]
            < self.config.lookahead_m
            - 1e-9
        ):
            distances = np.concatenate(
                [
                    distances,
                    [
                        self.config.lookahead_m
                    ],
                ]
            )

        return np.stack(
            [
                self._point_at_s(
                    match.route_s_m
                    + float(ds)
                )
                for ds in distances
            ],
            axis=0,
        )


def route_representations(
    *,
    sampled_map_xyz: np.ndarray,
    gps_local_xy_m: Tuple[float, float],
    tangent_heading_rad: float,
) -> Dict[str, np.ndarray]:
    route = np.asarray(
        sampled_map_xyz,
        dtype=np.float64,
    )

    gps_xy = np.asarray(
        gps_local_xy_m,
        dtype=np.float64,
    ).reshape(2)

    relative_map_xy = (
        route[:, :2]
        - gps_xy[None, :]
    )

    c = math.cos(
        tangent_heading_rad
    )
    s = math.sin(
        tangent_heading_rad
    )

    x_forward = (
        c * relative_map_xy[:, 0]
        + s * relative_map_xy[:, 1]
    )

    y_left = (
        -s * relative_map_xy[:, 0]
        + c * relative_map_xy[:, 1]
    )

    tangent_frame_xy = np.stack(
        [
            x_forward,
            y_left,
        ],
        axis=1,
    )

    return {
        "map_xyz": route.astype(
            np.float32
        ),
        "relative_map_xy": (
            relative_map_xy.astype(
                np.float32
            )
        ),
        "tangent_frame_xy": (
            tangent_frame_xy.astype(
                np.float32
            )
        ),
    }


def local_xy_from_manifest_gps(
    gps_record: Dict,
) -> Tuple[float, float]:
    return gps_wgs84_to_morai_local_utm52n(
        latitude_deg=float(
            gps_record["latitude"]
        ),
        longitude_deg=float(
            gps_record["longitude"]
        ),
        east_offset_m=float(
            gps_record["eastOffset"]
        ),
        north_offset_m=float(
            gps_record["northOffset"]
        ),
    )
