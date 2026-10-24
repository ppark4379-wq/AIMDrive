"""Adapt the static-zone search to legal highway lanes, without recorded coordinates.

Search runs on immutable snapshots off the control thread. Results are only
proposals: the current highway traffic checks must approve them before use.
"""

import copy
import math
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import CubicSpline

from ...interfaces.geometry import Point
from ...vehicle_state import VehicleState
from ..static_obstacle.recorded_planner import RecordedStaticPlanner, connector
from ..static_obstacle.corridor_graph import search_corridor
from ..lattice import parameters as static_parameters
from . import parameters as p
from .structures import HighwayCandidatePath


class HighwayCorridorSearch(RecordedStaticPlanner):
    """Reuse static-zone connectors, A*, swept OBB checks and tracking reserve."""

    def __init__(self, ego, lanes, obstacles):
        # Deliberately do not load the northbound zone's recordings or limits.
        self.origin = ego.position
        self.heading = ego.yaw
        self.cosine, self.sine = math.cos(ego.yaw), math.sin(ego.yaw)
        self.planner = SimpleNamespace(vehicle_length=p.VEHICLE_LENGTH,
            vehicle_half_width=p.VEHICLE_WIDTH / 2,
            collision_margin=max(p.COLLISION_MARGIN, static_parameters.COLLISION_MARGIN))
        self.curves = []
        self.records = []
        lows, highs = [], []
        for lane in lanes:
            xy = self.to_local(np.array([(v.x, v.y) for v in lane.points]))
            if np.any(np.diff(xy[:, 1]) <= 1e-4):
                raise ValueError('Lane is not forward in the search frame')
            self.curves.append(CubicSpline(xy[:, 1], xy[:, 0]))
            self.records.append((lane.id,))
            lows.append(xy[0, 1])
            highs.append(xy[-1, 1])
        self.low, self.high = max(lows), min(highs)
        # Stay inside this section; a neighboring lane may end at the merge.
        self.rejoin_y = min(60., self.high - p.VEHICLE_LENGTH - 4.)
        self.goal_y = self.rejoin_y + 2.
        if self.low > 0 or self.rejoin_y < 15.:
            raise ValueError('Insufficient common lane corridor')
        probes = np.linspace(0., self.goal_y + p.VEHICLE_LENGTH, 80)
        centers = np.array([curve(probes) for curve in self.curves])
        widths = np.abs(np.diff(centers, axis=0))
        if np.any(widths < 2.5) or np.any(widths > 4.5):
            raise ValueError('Nonadjacent or converging lane corridor')
        self.global_curve = self.curves[-1]
        self.rejections = dict(curvature=0, envelope=0, collision=0)
        self.search_details = {}
        self.selected_names = []
        self.search_progress = 0
        self.recording_search_timeout = .4
        self.corridor_search_timeout = 2.
        self.require_goal_alignment = True
        self.lateral_step = .35
        self.heading_slopes = (-.4, -.2, 0., .2, .4)
        self.obstacles = []
        for obstacle in obstacles:
            obj = copy.copy(obstacle)
            xy = self.to_local(np.array([[obj.position.x, obj.position.y]]))[0]
            obj.position = Point(*xy)
            obj.heading_deg += 90. - math.degrees(self.heading)
            self.obstacles.append(obj)

    def to_local(self, xy):
        delta = xy - [self.origin.x, self.origin.y]
        return np.column_stack((delta[:, 0]*self.sine - delta[:, 1]*self.cosine,
                                delta[:, 0]*self.cosine + delta[:, 1]*self.sine))

    def to_world(self, xy):
        return np.column_stack((self.origin.x + xy[:, 0]*self.sine + xy[:, 1]*self.cosine,
                                self.origin.y - xy[:, 0]*self.cosine + xy[:, 1]*self.sine))

    def _envelope_mask(self, xy, yaw, body_margin=0.):
        valid = np.isfinite(xy).all(axis=1) & np.isfinite(yaw)
        for along in (-body_margin, p.VEHICLE_LENGTH + body_margin):
            for side in (-p.VEHICLE_WIDTH/2-body_margin, p.VEHICLE_WIDTH/2+body_margin):
                x = xy[:, 0] + along*np.cos(yaw) - side*np.sin(yaw)
                y = xy[:, 1] + along*np.sin(yaw) + side*np.cos(yaw)
                centers = np.array([curve(y) for curve in self.curves])
                valid &= ((y >= self.low) & (y <= self.high)
                          & (x >= centers.min(axis=0)-p.LANE_WIDTH/2)
                          & (x <= centers.max(axis=0)+p.LANE_WIDTH/2))
        return valid

    def _edge(self, curve, y0, y1, obstacles, body_margin=0., tracking_reserve=True):
        result = super()._edge(curve, y0, y1, obstacles, body_margin, tracking_reserve)
        if result is None:
            return None
        ys = result[:, 1]
        curvature = np.abs(curve(ys, 2))/(1+curve(ys, 1)**2)**1.5
        if np.max(curvature) > p.STATIC_CHANGE_MAX_CURVATURE:
            return None
        return result

    def run_search(self):
        result = search_corridor(self, VehicleState(0., 0., math.pi/2, 0.),
                                 self.obstacles, connector, p.STATIC_CHANGE_MAX_CURVATURE)
        return None if result is None else self.to_world(result)


class HighwayStaticCorridor:
    def __init__(self, planner):
        self.planner = planner
        self.executor = None
        self.pending = None
        self.key = None
        self.cached = []
        self.origin = None
        self.status = 'idle'

    def reset(self):
        if self.pending is not None:
            self.pending.cancel()
        self.pending = None
        self.key = None
        self.cached = []
        self.origin = None
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None
        self.status = 'idle'

    @staticmethod
    def _search_jobs(jobs):
        proposals = []
        for owner, direction, lanes in jobs:
            points = owner.run_search()
            if points is not None:
                proposals.append((owner, direction, lanes, points))
        return proposals

    def candidates(self, ego, source, entries, matches, allowed_targets=None):
        planner = self.planner
        obstacles = [e['object_info'] for e in entries
                     if planner.obstacle_motion.is_static(e['object_info'])]
        # Include pose changes in invalidation without chasing normal sensor noise.
        key = (source.id, tuple(sorted((round(o.position.x, 1), round(o.position.y, 1),
            round(o.heading_deg), round(o.size_x, 2), round(o.size_y, 2)) for o in obstacles)))
        if self.pending is not None:
            if not self.pending.done():
                self.status = 'searching'
                return []
            try:
                self.cached = self.pending.result()
                self.status = 'ready' if self.cached else 'no_path_within_search_budget'
            except Exception as exc:
                self.cached = []
                self.status = 'search_failed:' + type(exc).__name__
            self.pending = None
        moved = self.origin is None or ego.position.distance(self.origin) > .5
        if key != self.key or (moved and not self.cached):
            jobs = []
            for direction, adjacent in planner.lane_map.legal_neighbors(source):
                chain = [source, adjacent]
                for _ in range(2):
                    target = chain[-1]
                    if allowed_targets is None or target.lane_number in allowed_targets:
                        try:
                            owner = HighwayCorridorSearch(ego, chain, obstacles)
                            jobs.append((owner, direction, tuple(chain)))
                        except ValueError:
                            pass
                    following = dict(planner.lane_map.legal_neighbors(target)).get(direction)
                    if following is None or following.id in {v.id for v in chain}:
                        break
                    chain = chain + [following]
            # All adjacent destinations precede any farther destination.
            jobs.sort(key=lambda job: len(job[2]))
            self.key, self.origin, self.cached = key, ego.position, []
            if jobs:
                if self.executor is None:
                    self.executor = ThreadPoolExecutor(max_workers=1)
                self.pending = self.executor.submit(self._search_jobs, jobs)
                self.status = 'searching'
            else:
                self.status = 'no_legal_corridor'
            return []
        result = []
        stale = False
        for owner, direction, lanes, xy in self.cached:
            points = [Point(x, y) for x, y in xy]
            nearest = min(range(len(points)), key=lambda i: points[i].distance(ego.position))
            if nearest >= len(points)-2 or points[nearest].distance(ego.position) > .25:
                stale = True
                continue
            yaw = math.atan2(points[nearest+1].y-points[nearest].y,
                             points[nearest+1].x-points[nearest].x)
            if abs(planner.lane_map._angle_difference(yaw, ego.yaw)) > math.radians(5):
                stale = True
                continue
            # Anchor to the measured pose, then validate this actual join too.
            points = [ego.position] + points[nearest+1:]
            candidate = HighwayCandidatePath(layer='static_corridor')
            candidate.points = points
            candidate.direction = direction
            candidate.source_link_id, candidate.target_link_id = source.id, lanes[-1].id
            candidate.target_lane = lanes[-1].lane_number
            candidate.offset = (source.lane_number-candidate.target_lane)*p.LANE_WIDTH
            candidate.static_escape = candidate.compact_static = candidate.static_corridor = True
            candidate.corridor_owner = owner
            candidate.corridor_lanes = lanes
            candidate.transition_end = points[-1]
            candidate.transition_distance = planner._cumulative_distance(points)[-1]
            candidate.curvature_cost = planner._max_curvature(points)
            candidate.speed_limit = planner._static_change_speed_limit(candidate.curvature_cost)
            candidate.cost = 1. + candidate.curvature_cost*4 + len(lanes)*.25
            # Append the final lane for completion/settling prediction and tracking.
            projection = planner.lane_map.project(points[-1], lanes[-1])
            tail = planner.lane_map.route_points(projection, 100., p.PATH_SAMPLE_SPACING)
            candidate.points += [row[0] for row in tail][1:]
            self.validate_geometry(candidate, entries)
            planner._evaluate_lane_change(candidate, ego, entries, matches)
            for lane in lanes[1:]:
                if not planner._target_lane_has_clearance(lane, candidate, matches,
                        min(ego.velocity, planner._requested_speed, candidate.speed_limit)):
                    candidate.valid = False
                    candidate.cost = float('inf')
            result.append(candidate)
            if candidate.valid:
                break
        if stale and not any(c.valid for c in result):
            self.key = None
        return result

    def validate_geometry(self, candidate, entries):
        """Recheck current static poses with the same inflated static-zone OBBs."""
        owner = candidate.corridor_owner
        world = np.array([(v.x, v.y) for v in candidate.points])
        local = owner.to_local(world)
        # The tail follows the normal lane graph; check the searched corridor
        # only through its endpoint, where these section boundaries are valid.
        end = min(range(len(candidate.points)),
                  key=lambda i: candidate.points[i].distance(candidate.transition_end))
        xy = local[:end+1]
        if len(xy) < 2:
            return
        delta = np.diff(xy, axis=0)
        yaw = np.arctan2(delta[:, 1], delta[:, 0])
        yaw = np.r_[yaw, yaw[-1]]
        objects = []
        for entry in entries:
            obj = entry['object_info']
            if not self.planner.obstacle_motion.is_static(obj):
                continue
            item = copy.copy(obj)
            item.position = Point(*owner.to_local(np.array([[obj.position.x, obj.position.y]]))[0])
            item.heading_deg += 90.-math.degrees(owner.heading)
            objects.append(item)
        curvature = self.planner._max_curvature(candidate.points[:end+1])
        sweep = .5*float(np.max(np.linalg.norm(delta, axis=1)))*(1+p.VEHICLE_LENGTH*curvature)
        if (curvature > p.STATIC_CHANGE_MAX_CURVATURE
                or not np.all(owner._envelope_mask(xy, yaw, .15))
                or not owner.collision_free(xy, yaw, objects, max(.12, sweep))):
            candidate.valid = False
            candidate.cost = float('inf')
            candidate.lane_change_verified = False
            candidate.prediction_conflict = dict(time=0., rear=False, kind='static_corridor_geometry')
            candidate.collision_time = 0.
