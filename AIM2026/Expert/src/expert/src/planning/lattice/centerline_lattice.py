"""Right-side lattice inside the user's second, straight-road segment."""
import json
import math
from pathlib import Path

from ...localization.point import Point
from .lattice_planner import LatticePlanner
from ..static_obstacle.avoidance import StaticAvoidance
from .structures import LatticePlanningResult


class CenterlineLattice:
    START = (-59.39, -100.75)
    END = (-59.11, -46.86)

    def __init__(self, map_dir=None):
        self.planner = LatticePlanner(sample_spacing=0.2)
        self.checker = StaticAvoidance(self.planner)
        self.center = []
        self.outer = []
        self.half_width = 1.55
        if map_dir and (Path(map_dir) / 'link_set.json').is_file():
            records = {str(row['idx']): row for row in json.loads((Path(map_dir) / 'link_set.json').read_text())}
            main, right = records.get('A2256W000202'), records.get('A2256W000009')
            if main and right:
                self.center = [Point(*xy[:2]) for xy in main['points']]
                self.outer = [Point(*xy[:2]) for xy in right['points']]
                self.half_width = min(float(main['width_start']), float(main['width_end']),
                                      float(right['width_start']), float(right['width_end'])) / 2 - 0.2

    def contains(self, ego):
        if not self.START[1] <= ego.position.y <= self.END[1]:
            return False
        return -62.0 <= ego.position.x <= -53.0 and math.sin(ego.yaw) > 0.5

    def _body_inside(self, points, ego):
        if not self.center:
            return False
        self.checker._set_route(self.center, ego)
        corners = []
        for i, point in enumerate(points):
            corners.extend(self.checker._corners(point, self.planner._path_heading(points, i)))
        corners.extend(self.checker._corners(ego.position, ego.yaw))
        left_offsets = self.checker._laterals(corners)
        if any(offset > self.half_width for offset in left_offsets):
            return False
        # The outer same-direction link supplies a conservative right-edge estimate.
        self.checker._set_route(self.outer, ego)
        return all(offset >= -self.half_width for offset in self.checker._laterals(corners))

    def run(self, ego, reference_path, entries):
        obstacles = [entry['object_info'] for entry in entries if getattr(entry['object_info'], 'is_static', False)]
        result = self.planner.plan(ego, reference_path, obstacles, offsets=tuple(-i * .5 for i in range(9)),
                                   short_lookahead_distances=(15.0, 20.0), prefer_right=True)
        if not result.obstacle_detected:
            return result
        valid = [candidate for candidate in result.candidates if candidate.layer == 'short' and candidate.valid
                 and self.planner._max_curvature(candidate.points) <= .24
                 and self._body_inside(candidate.points, ego)]
        result.candidates = []
        result.active = False
        result.path = reference_path
        result.acc_override = True
        result.acc_target_velocity = 0.0
        result.reason = 'centerline_lattice_blocked' if self.center else 'centerline_map_missing'
        if valid:
            selected = min(valid, key=lambda item: item.cost)
            result.path = selected.points
            result.selected_offset = selected.offset
            result.candidates = [selected]
            result.active = True
            result.acc_target_velocity = min(20 / 3.6, math.sqrt(1.5 / max(.0001, selected.curvature_cost)))
            result.reason = 'centerline_lattice_avoid'
        result.debug_details['static_avoidance'] = {
            'phase': 'lane_lattice' if result.active else 'blocked',
            'path_validated': result.active, 'blocked_by': '' if result.active else 'central_or_outer_boundary',
            'boundary_source': 'link_width_estimate', 'planner': 'centerline_lattice',
        }
        if self.center:
            result.debug_details['central_guard'] = [
                (point.x - self.half_width * math.sin(self.planner._path_heading(self.center, i)),
                 point.y + self.half_width * math.cos(self.planner._path_heading(self.center, i)))
                for i, point in enumerate(self.center)]
        return result
