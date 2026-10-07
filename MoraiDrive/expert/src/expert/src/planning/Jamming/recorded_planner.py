"""Use the existing static-route planner in a rotated Jamming section.

Paths, Ego and obstacle OBB headings are transformed together. The controller
always receives ENU paths. Collision margins, search, speed and tracking are
the implementations in 정적장애물, shared rather than copied.
"""
import copy
import json
import math

import numpy as np

from ...localization.point import Point
from ..lattice_planning.lattice_planner import LatticePlanner
from ..lattice_planning import parameters
from ..lattice_planning.structures import CandidatePath
from ..정적장애물.recorded_planner import (
    RecordedStaticPlanner, RecordedRouteGeometry, load_routes, connector,
)
from .section import JammingSection, ROOT, ROUTE_DIRECTORY
from .prepare_routes import PREPARED_DIRECTORY, clip_rows
from .motion import stationary, moving, key as motion_key
from .dynamic_planner import DynamicObstaclePlanner


class _LocalPlanner(RecordedStaticPlanner):
    def __init__(self, section, planner, background):
        self.section = section
        width, length = section.activation_half_width_m, section.length
        geometry = RecordedRouteGeometry(
            rejoin_point=(0., length, section.end[2]), active_x=(-width, width),
            active_start_y=-section.approach_distance_m,
            reset_y=section.start_longitudinal_tolerance_m,
            entry_y=-section.approach_distance_m, route_x=(-width, width),
            route_y=(-section.approach_distance_m-5., length+section.global_tail_m+5.),
            obstacle_x=(-width-5., width+5.), tail_length=section.global_tail_m,
        )
        super().__init__(planner, root=section.root, background=background, geometry=geometry)
        # Nine routes over 130 m need more scheduling headroom than the
        # shorter static section, especially beside the ROS control thread.
        self.recording_search_timeout = 8.

    def _search(self, ego, obstacles):
        if not obstacles:
            # For moving traffic, start with the official lane. Selecting an
            # arbitrary recording here creates an extra lateral maneuver before
            # the moving-object planner has checked the overtake.
            start=float(ego.position.y)
            for distance in (6.,8.,12.,18.):
                join=min(start+distance,self.goal_y-.5)
                if join<=start:
                    continue
                entry=connector((start,ego.position.x,
                    math.cos(ego.yaw)/max(.5,math.sin(ego.yaw)),0.),
                    self._state(self.global_curve,join))
                first=self._edge(entry,start,join,obstacles)
                rest=self._edge(self.global_curve,join,self.goal_y,obstacles)
                if first is not None and rest is not None:
                    self.selected_names=['jamming_global_reference']
                    return np.vstack((first,rest[1:]))
        return super()._search(ego,obstacles)

    def approach_result(self, ego, reference_path):
        result = super().approach_result(ego, reference_path)
        if result is not None:
            worker = self.pending.result()
            candidate = CandidatePath(layer='recorded')
            candidate.points = worker.path
            candidate.selected_preview = True
            result.candidates = [candidate]
            result.debug_details['static_avoidance'].update(
                planner='recorded_route_graph', selected_routes=list(worker.selected_names),
                corridor_search=dict(worker.search_details), preview_only=True)
        return result

    def _load_routes(self, root):
        manifest = json.loads((root / 'routes.json').read_text(encoding='utf-8'))
        directory = manifest.get('directory', ROUTE_DIRECTORY)
        if directory not in (ROUTE_DIRECTORY, PREPARED_DIRECTORY):
            raise ValueError('Invalid Jamming candidate directory')
        def transform(name, data):
            self.section.validate_recording(data)
            # Reserve the last 10 m (configurable) for the same validated C2
            # global-route return. Interpolate all candidates to the SAME line.
            rows = [dict(zip(('x','y','yaw_rad'), row)) for row in data]
            clipped = clip_rows(self.section, rows, self.section.length-self.section.return_blend_m)
            return np.asarray([(*self.section.to_local((r['x'],r['y'])),
                                self.section.yaw_to_local(r['yaw_rad'])) for r in clipped])
        records, audit = load_routes(root, directory=directory, transform=transform)
        if not records:
            raise ValueError("No complete Jamming routes; record from start to end first")
        return records, audit


class JammingPlanner:
    def __init__(self, root=ROOT, planner=None, background=True):
        self.section = JammingSection(root)
        self.local = _LocalPlanner(self.section, planner or LatticePlanner(
            sample_spacing=parameters.STATIC_SAMPLE_SPACING), background)
        self.dynamic = DynamicObstaclePlanner()
        self._motion_states = {}
        self._last_position = None

    @property
    def active(self):
        return self.local.active

    def close(self):
        self.dynamic.reset()
        self.local.reset()
        if self.local.executor is not None:
            self.local.executor.shutdown(wait=True)

    def set_live_preview(self, enabled):
        self.dynamic.reset()
        self.local.set_live_preview(enabled)

    def set_live_drive(self, enabled):
        self.local.set_live_drive(enabled)

    def _ego(self, ego):
        result = copy.copy(ego)
        result.position = Point(*self.section.to_local(ego.position))
        result.yaw = self.section.yaw_to_local(ego.yaw)
        return result

    def _path(self, path):
        return [Point(*self.section.to_local(point)) for point in path]

    def _obstacles(self, objects):
        result = []
        for item in objects:
            original = item['object_info']
            if not stationary(original):
                continue
            obj = copy.copy(original)
            obj.is_static = True
            obj.position = Point(*self.section.to_local(original.position))
            obj.heading_deg = math.degrees(self.section.yaw_to_local(math.radians(original.heading_deg)))
            # Include footprints crossing the section band, not only centers.
            radius = .5*math.hypot(max(obj.size_x, 1.), max(obj.size_y, 1.)) + 2.
            if (abs(obj.position.x) <= self.section.activation_half_width_m+radius
                    and -self.section.approach_distance_m-radius <= obj.position.y
                    <= self.section.length+self.section.global_tail_m+radius):
                result.append(obj)
        return result

    def _moving_objects(self, objects, ego):
        result=[]
        for item in objects:
            original=item['object_info']
            if not moving(original) or original.position.distance(ego.position)>120.:
                continue
            obj=copy.copy(original)
            obj.position=Point(*self.section.to_local(original.position))
            obj.heading_deg=math.degrees(self.section.yaw_to_local(math.radians(original.heading_deg)))
            result.append(obj)
        return result

    def _world_result(self, result):
        if result is None:
            return None
        converted = copy.copy(result)
        converted.path = [Point(*self.section.to_world(p)) for p in result.path]
        converted.candidates = []
        for candidate in result.candidates:
            value = copy.copy(candidate)
            # RViz identifies the selected path by object identity.
            value.points = (converted.path if candidate.points is result.path else
                            [Point(*self.section.to_world(p)) for p in candidate.points])
            converted.candidates.append(value)
        converted.debug_details = copy.deepcopy(result.debug_details)
        converted.predicted_object_paths = [
            [Point(*self.section.to_world(p)) for p in path] for path in result.predicted_object_paths]
        converted.debug_details['jamming'] = dict(
            section='Jamming', path_frame='ENU', diagnostic_frame='section_forward_y',
            start_enu=list(self.section.start), end_enu=list(self.section.end),
            library=str(self.section.root / 'routes.json'),
        )
        # Keep static_obstacle mode/reason so the existing controller applies
        # the same steering preview, braking limits and smooth handoff.
        return converted

    def contains(self, ego):
        return self.local.contains(self._ego(ego))

    def run(self, ego, reference_path, objects, prediction_path=None, global_path=None, desired_velocity=None):
        local_ego = self._ego(ego)
        x, y = local_ego.position
        states={motion_key(item['object_info']): moving(item['object_info']) for item in objects
                if item['object_info'].type in (1,2)}
        relevant={motion_key(o) for o in self.local.known}
        if self.dynamic.active:
            relevant.update(motion_key(o) for o in self.dynamic.tracker.known)
        for item in objects:
            obj=item['object_info']
            lateral,forward_y=self.section.to_local(obj.position)
            radius=.5*math.hypot(max(obj.size_x,1.),max(obj.size_y,1.))+2.
            if (abs(lateral)<=self.section.activation_half_width_m+radius
                    and -self.section.approach_distance_m-radius<=forward_y
                    <=self.section.length+self.section.global_tail_m+radius):
                relevant.add(motion_key(obj))
        if any(k in relevant and k in self._motion_states and self._motion_states[k]!=v
               for k,v in states.items()):
            # A stopped NPC is now geometry; a restarting NPC must no longer
            # be frozen in the static snapshot. Native stationary scenes keep
            # their original one-plan behavior.
            self.local.reset()
            self.dynamic.reset()
        self._motion_states=states
        if self._last_position is not None and local_ego.position.distance(self._last_position)>parameters.STATIC_RESET_DISTANCE:
            self.dynamic.reset()
        self._last_position=local_ego.position.copy()
        obstacles = self._obstacles(objects)
        movers = self._moving_objects(objects,ego)
        planned = global_path if global_path is not None else (
            prediction_path if prediction_path is not None else reference_path)
        planned = self._path(planned)
        # GT received before this section can warm the fixed entry maneuver
        # immediately, including when Ego is outside the activation rectangle.
        self.local.refresh_live_preview(local_ego, obstacles)
        self.local.preplan(local_ego, planned, obstacles)
        in_band = abs(x) < self.section.activation_half_width_m
        in_extent = (-100. <= y
                     <= self.section.length+self.section.global_tail_m)
        forward = math.sin(local_ego.yaw) > .5
        if not self.active and not (in_band and in_extent and forward):
            return None
        reference = self._path(reference_path)
        inside = self.local.contains(local_ego)
        def finish(result):
            if result is not None:
                result=self.dynamic.run(self.local,local_ego,reference,obstacles,movers,result,
                    float(desired_velocity) if desired_velocity is not None else 30./3.6)
            return self._world_result(result)
        if self.dynamic.active:
            # The overlay tracks a validated detour until it rejoins the fixed
            # static path; do not reject it as an error from the static path.
            base=self.local._result(self.local.path,True,'follow')
            return finish(base)
        # The complete maneuver now includes the approach, starting 20 m
        # before the recording line. This allows validated avoidance to own
        # static-object speed control before legacy ACC can hold Ego at y=0.
        # Never adopt a path that starts ahead of the measured vehicle.
        if (not self.active and y < self.local.geometry.entry_y
                and not self.local.interactive_preview):
            return finish(self.local.approach_result(local_ego, reference))
        if (self.active or (inside and (obstacles or movers or self.local.pending is not None))
                or (self.local.interactive_preview and self.local.pending is not None)):
            return finish(self.local.plan(local_ego, reference, obstacles, planned))
        return finish(self.local.approach_result(local_ego, reference))
