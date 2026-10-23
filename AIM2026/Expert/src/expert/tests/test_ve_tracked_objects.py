"""VE motion tracks must reach the existing Expert policies without car labels."""
import contextlib
import io
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

from inference.ve_object_tracking import VeObjectTracker
from inference.ve_objects_udp import make_object_message
from expert.src.obstacle.object_info import ObjectInfo, is_ve_tracked_unknown
from expert.src.localization.point import Point
from expert.src.vehicle_state import VehicleState
from expert.src.planning.adaptive_cruise_control import AdaptiveCruiseControl
from expert.src.planning.section_planning import LatticePlanningProcess
from expert.src.planning.roundabout.planner import RoundaboutEntryPlanner
from expert.src.planning.lattice.structures import LatticePlanningResult
from expert.src.planning.lattice import parameters as p
from expert.src.planning.highway.obstacle_motion import ObstacleMotion
from expert.src.planning.jamming.motion import stationary, moving


def unknown(x, y=0., speed=3., heading=0., identity=1, state='moving'):
    return ObjectInfo(x, y, speed, -1, name='ve_tracked_unknown:' + state,
                      is_static=state == 'stationary', size_x=4.5, size_y=2.,
                      heading_deg=heading, unique_id=identity)


class VeTrackedObjectsTests(unittest.TestCase):
    def test_validated_static_avoidance_does_not_suppress_moving_unknown_acc(self):
        from expert.src.autonomous_driving import AutonomousDriving
        driver = object.__new__(AutonomousDriving)
        entries = [dict(object_info=unknown(12., identity=1, speed=0., state='stationary')),
                   dict(object_info=unknown(20., identity=2))]
        result = NS(mode='static_obstacle', active=True,
                    debug_details={'static_avoidance': {'path_validated': True}})
        self.assertEqual([entries[1]], driver._legacy_acc_entries(VehicleState(), result, entries))
        result.debug_details = {'jamming_dynamic': {'speed_authority': True}}
        self.assertEqual([], driver._legacy_acc_entries(VehicleState(), result, entries))

    def test_unknown_lead_reduces_acc_speed_but_untagged_unknown_is_unchanged(self):
        acc = AdaptiveCruiseControl(.5, 1., .8, 4.635)
        route = [Point(float(x), 0.) for x in range(40)]
        obj = unknown(12.)
        entry = dict(object_info=obj, local_position=Point(12., 0.))
        acc.check_object(route, [entry], [])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertAlmostEqual(1.465, acc.get_target_velocity(8., 8.))
        self.assertEqual(-1, obj.type)
        obj.name = 'unrelated_unknown'
        acc.check_object(route, [entry], [])
        self.assertEqual(8., acc.get_target_velocity(8., 8.))

    def test_crossing_unknown_reaches_roundabout_collision_prediction(self):
        planner = RoundaboutEntryPlanner()
        ego = VehicleState(0., 0., 0., 5.)
        route = planner._route([Point(float(x), 0.) for x in range(81)])
        obj = unknown(15., -12., 5., 90.)
        tracks = planner._tracks(ego, [dict(object_info=obj)])
        self.assertEqual(1, len(tracks))
        predictions = [[planner._object_poses(obj, None, i * p.ROUNDABOUT_PREDICTION_STEP)]
                       for i in range(int(p.ROUNDABOUT_PREDICTION_HORIZON / p.ROUNDABOUT_PREDICTION_STEP) + 1)]
        collision = planner._collision(ego, route, 0., tracks, 5., predictions)
        self.assertIsNotNone(collision)
        self.assertGreater(collision['collision_time'], 0.)
        obj.name = 'unrelated_unknown'
        self.assertEqual([], planner._tracks(ego, [dict(object_info=obj)]))

    def test_uncertain_zero_speed_lead_still_reaches_acc(self):
        acc = AdaptiveCruiseControl(.5, 1., .8, 4.635)
        route = [Point(float(x), 0.) for x in range(40)]
        # A conservative speed fallback must not delete the bbox or rely on a
        # stationary/vehicle label to activate the existing Expert policy.
        obj = unknown(12., speed=0., state='uncertain')
        self.assertFalse(obj.is_static)
        acc.check_object(route, [dict(object_info=obj, local_position=Point(12., 0.))], [])
        with contextlib.redirect_stdout(io.StringIO()):
            stopped_target = acc.get_target_velocity(8., 8.)
        self.assertLess(stopped_target, 8.)
        obj.velocity = 13.
        acc.check_object(route, [dict(object_info=obj, local_position=Point(12., 0.))], [])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertLess(stopped_target, acc.get_target_velocity(8., 8.))

    def test_highway_receives_tagged_unknown_and_motion_tracks_do_not_collapse(self):
        process = object.__new__(LatticePlanningProcess)
        process.highway_lane_planner = Mock()
        process.highway_lane_planner.plan.return_value = LatticePlanningResult([])
        objects = [unknown(12., identity=1), unknown(25., identity=2, speed=0., state='stationary')]
        entries = [dict(object_info=o) for o in objects]
        process._run_highway_zone(VehicleState(), [Point(0., 0.), Point(30., 0.)], entries, 5.)
        self.assertEqual(entries, process.highway_lane_planner.plan.call_args.args[2])
        motion = ObstacleMotion()
        motion.update(entries, .1)
        self.assertEqual(2, len(motion.tracks))
        self.assertFalse(motion.is_static(objects[0]))
        self.assertTrue(motion.is_static(objects[1]))

    def test_transport_keeps_stationary_and_moving_objects_separate_with_unknown_type(self):
        tracker = VeObjectTracker()
        for i in range(8):
            boxes = [dict(center_world=[i * .3, 0.], speed_mps=3., heading_deg=0., length_m=4., width_m=2.),
                     dict(center_world=[10., 0.], speed_mps=0., heading_deg=0., length_m=.4, width_m=.3)]
            packet = tracker.update(dict(generation='a', sent_ns=round((1+i*.1)*1e9), boxes=boxes))
        wire = make_object_message(packet,
            lambda: NS(header=NS(), npc_list=[], obstacle_list=[], pedestrian_list=[]),
            lambda: NS(position=NS(), size=NS(), velocity=NS()), 1)
        self.assertEqual((1, 1), (wire.num_of_npcs, wire.num_of_obstacle))
        converted = []
        for static, items in ((False, wire.npc_list), (True, wire.obstacle_list)):
            for m in items:
                converted.append(ObjectInfo(m.position.x,m.position.y,m.velocity.x/3.6,m.type,
                    name=m.name,is_static=static,size_x=m.size.x,size_y=m.size.y,
                    heading_deg=m.heading,unique_id=m.unique_id))
        self.assertTrue(all(is_ve_tracked_unknown(o) and o.type == -1 for o in converted))
        self.assertNotEqual(converted[0].unique_id, converted[1].unique_id)
        self.assertTrue(moving(converted[0]))
        self.assertTrue(stationary(converted[1]))
        self.assertFalse(stationary(converted[0]))
        self.assertFalse(moving(converted[1]))


if __name__ == '__main__':
    unittest.main()
