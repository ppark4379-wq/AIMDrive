"""Static-zone search proposals must still pass the live highway safety gates."""

import contextlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from expert.src.control.path_controller import PathController
from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.highway_lane_planning.lane_map import HighwayLaneMap
from expert.src.planning.highway_lane_planning.planner import HighwayLanePlanner
from expert.src.planning.highway_lane_planning.static_corridor import HighwayStaticCorridor, HighwayCorridorSearch
from expert.src.vehicle_state import VehicleState


class HighwayStaticCorridorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.lane_map = HighwayLaneMap(str(cls.root.parents[1]/'map_data/R_KR_PG_KATRI_2025'))

    def setUp(self):
        self.planner = HighwayLanePlanner(lane_map=self.lane_map)
        self.planner._requested_speed = 27.78
        self.ego = VehicleState(62.16, 250., -math.pi/2, 0.)
        self.source = self.lane_map.links['A2256W000411']
        self.entries = [self.entry(62., 215., 0., 1, True)]
        self.planner.obstacle_motion.update(self.entries, .05)
        self.search = HighwayStaticCorridor(self.planner)
        self.addCleanup(self.search.reset)

    def entry(self, x, y, speed, uid, static=False):
        obj = ObjectInfo(x, y, speed, 2 if static else 1, is_static=static,
                         size_x=4.5, size_y=2., heading_deg=-90., unique_id=uid)
        return dict(object_info=obj, local_position=(obj.position-self.ego.position).rotate(-self.ego.yaw))

    def proposals(self):
        matches = self.planner._match_objects(self.entries)
        result = self.search.candidates(self.ego, self.source, self.entries, matches)
        if self.search.pending is not None:
            self.search.pending.result(timeout=15.)
            result = self.search.candidates(self.ego, self.source, self.entries, matches)
        return result

    def test_static_search_connects_to_actual_final_lane(self):
        lanes = [SimpleNamespace(id=str(i), points=[Point(x,-20.), Point(x,100.)])
                 for i, x in enumerate((0., -3.5))]
        owner = HighwayCorridorSearch(VehicleState(0.,0.,math.pi/2,0.), lanes, [])
        # This checks geometry, not the runtime deadline. Concurrent background
        # searches in the suite must not turn CPU contention into a false failure.
        owner.corridor_search_timeout = 10.0
        path = owner.run_search()
        self.assertIsNotNone(path)
        self.assertAlmostEqual(-3.5, path[-1,0], places=5)
        self.assertLess(np.max(np.linalg.norm(np.diff(path, axis=0), axis=1)), .3)

    def test_valid_static_path_retains_highway_verification(self):
        valid = [candidate for candidate in self.proposals() if candidate.valid]
        self.assertTrue(valid)
        self.assertEqual('A2256W000409', valid[0].target_link_id)
        self.assertTrue(valid[0].lane_change_verified)
        self.assertGreater(valid[0].safety_speed, 0.)
        self.assertFalse(hasattr(valid[0], 'static_bypass_link_ids'))

    def test_fast_rear_blocks_a_geometrically_clear_cached_proposal(self):
        self.assertTrue(any(candidate.valid for candidate in self.proposals()))
        self.entries.append(self.entry(66., 252., 25., 2))
        self.planner.obstacle_motion.update(self.entries, .05)
        self.assertFalse(any(candidate.valid for candidate in self.proposals()))

    def test_new_static_body_is_rechecked_on_the_committed_path(self):
        candidate = next(c for c in self.proposals() if c.valid)
        point = candidate.points[100]
        self.entries.append(self.entry(point.x, point.y, 0., 3, True))
        self.planner.obstacle_motion.update(self.entries, .05)
        self.search.validate_geometry(candidate, self.entries)
        self.assertFalse(candidate.valid)
        self.assertFalse(candidate.lane_change_verified)

    def test_moving_lead_does_not_start_static_search(self):
        self.entries = [self.entry(62., 215., 8., 1)]
        self.ego.velocity = 15.
        match = self.lane_map.project(self.ego.position, self.source)
        path = [v[0] for v in self.lane_map.route_points(match, 180., .5)]
        self.planner.plan(self.ego, path, self.entries, desired_velocity=27.78)
        self.assertIsNone(self.planner.static_corridor)

    def test_real_controller_follows_accepted_corridor_without_contact(self):
        candidate = next(c for c in self.proposals() if c.valid)
        planner = self.planner
        planner.static_corridor = self.search
        planner.active_candidate = candidate
        planner.state = planner.CHANGE_LEFT
        planner.source_link_id = self.source.id
        planner.target_link_id = candidate.target_link_id
        config = json.loads((self.root/'src/config/config.json').read_text(encoding='utf-8'))
        controller = PathController(config)
        path = [v[0] for v in self.lane_map.route_points(
            self.lane_map.project(self.ego.position, self.source), 200., .5)]
        dt = .05
        planner.set_control_dt(dt)
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(400):
                result = planner.plan(self.ego, path, self.entries, desired_velocity=27.78)
                self.assertGreater(result.acc_target_velocity, .05, result.debug_details)
                controller._configure_runtime(sync_mode=False, control_dt=dt)
                pedal, _, steering = controller._calculate_control(
                    self.ego, result.path, min(27.78,result.acc_target_velocity), result)
                speed = max(0., self.ego.velocity+(3. if pedal >= 0 else 5.)*pedal*dt)
                yaw = self.ego.yaw
                self.ego = VehicleState(self.ego.position.x+speed*math.cos(yaw)*dt,
                    self.ego.position.y+speed*math.sin(yaw)*dt,
                    yaw+speed/3.*math.tan(steering/controller.steering_sign)*dt, speed)
                obj = self.entries[0]['object_info']
                self.assertFalse(planner._obb_collision(self.ego.position,self.ego.yaw,
                    obj.position,-math.pi/2,obj))
                if planner.changed_lane_hold:
                    break
        self.assertTrue(planner.changed_lane_hold)


if __name__ == '__main__':
    unittest.main()
