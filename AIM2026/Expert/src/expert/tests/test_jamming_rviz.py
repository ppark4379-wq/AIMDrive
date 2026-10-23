"""Verify the existing ROS marker pipeline, without a ROS master or controls."""
import unittest

try:
    import rospy
    from visualization_msgs.msg import Marker
    from expert.src.visualization.rviz_visualizer import LatticeRvizVisualizer
except ImportError:
    LatticeRvizVisualizer = None

from expert.src.localization.point import Point
from expert.src.vehicle_state import VehicleState
from expert.src.planning.jamming.recorded_planner import JammingPlanner
from expert.src.planning.jamming.section import JammingSection
from expert.src.planning.lattice.structures import CandidatePath, LatticePlanningResult


@unittest.skipIf(LatticeRvizVisualizer is None, 'Requires ROS Noetic message packages')
class JammingRvizTests(unittest.TestCase):
    def test_early_preview_is_red_without_becoming_the_control_path(self):
        wrapper = object.__new__(JammingPlanner)
        wrapper.section = JammingSection()
        reference = [Point(0., y) for y in (-10., -5., 0.)]
        candidate = CandidatePath(layer='recorded')
        candidate.points = [Point(2., y) for y in (0., 10., 20.)]
        candidate.selected_preview = True
        result = wrapper._world_result(LatticePlanningResult(reference, mode='static_obstacle',
            candidates=[candidate], reason='static_recorded_approach'))
        self.assertFalse(result.active)
        self.assertIsNot(result.path, result.candidates[0].points)
        ego = VehicleState(*wrapper.section.to_world((0., -10.)), wrapper.section.angle, 0.)
        visualizer = object.__new__(LatticeRvizVisualizer)
        markers = visualizer._build_path_markers(result, ego, rospy.Time(1)).markers
        lines = [m for m in markers if m.ns=='candidate_paths_recorded' and m.action==Marker.ADD]
        self.assertEqual(len(lines), 1)
        self.assertEqual((lines[0].color.r,lines[0].color.g,lines[0].color.b), (1.,0.,0.))

    def test_only_generated_path_is_red_and_uses_world_coordinates(self):
        wrapper = object.__new__(JammingPlanner)
        wrapper.section = JammingSection()
        path = [Point(0., y) for y in (0., 10., 20.)]
        candidate = CandidatePath(layer='recorded')
        candidate.points = path
        result = LatticePlanningResult(path, active=True, mode='static_obstacle',
            acc_target_velocity=2., candidates=[candidate],
            debug_details={'static_avoidance':dict(planner='recorded_route_graph',
                phase='follow', selected_routes=['jamming_01.csv','jamming_03.csv'])})
        result = wrapper._world_result(result)
        ego = VehicleState(*wrapper.section.start[:2],wrapper.section.angle,0.)
        visualizer = object.__new__(LatticeRvizVisualizer)
        markers = visualizer._build_path_markers(result,ego,rospy.Time(1)).markers
        lines = [m for m in markers if m.ns=='candidate_paths_recorded' and m.action==Marker.ADD]
        self.assertEqual(len(lines),1)
        self.assertEqual((lines[0].color.r,lines[0].color.g,lines[0].color.b),(1.,0.,0.))
        self.assertAlmostEqual(lines[0].scale.x,.15)
        self.assertAlmostEqual(lines[0].points[0].x,wrapper.section.start[0])
        self.assertAlmostEqual(lines[0].points[0].y,wrapper.section.start[1])
        status = next(m for m in markers if m.ns=='lattice_status')
        self.assertIn('JAMMING: RECORDED FOLLOW',status.text)
        self.assertIn('jamming_03.csv',status.text)
        inactive=LatticePlanningResult([],reason='outside_zone')
        cleared=visualizer._build_path_markers(inactive,ego,rospy.Time(2)).markers
        self.assertTrue(any(m.action==Marker.DELETEALL or
                            (m.ns=='candidate_paths_recorded' and m.action==Marker.DELETE) for m in cleared))

    def test_dynamic_emergency_status_and_predictions_use_world_coordinates(self):
        wrapper=object.__new__(JammingPlanner)
        wrapper.section=JammingSection()
        result=LatticePlanningResult([Point(0.,0.),Point(0.,10.)],
            mode='static_obstacle',acc_target_velocity=0.,
            predicted_object_paths=[[Point(1.,3.),Point(1.,10.)]],
            debug_details={'jamming_dynamic':dict(state='EMERGENCY',
                emergency=True,lead_gap_m=2.,collision=dict(time=.5))})
        result=wrapper._world_result(result)
        expected=wrapper.section.to_world((1.,3.))
        self.assertAlmostEqual(result.predicted_object_paths[0][0].x,expected[0])
        self.assertAlmostEqual(result.predicted_object_paths[0][0].y,expected[1])
        ego=VehicleState(*wrapper.section.start[:2],wrapper.section.angle,0.)
        visualizer=object.__new__(LatticeRvizVisualizer)
        markers=visualizer._build_path_markers(result,ego,rospy.Time(1)).markers
        status=next(m for m in markers if m.ns=='lattice_status')
        self.assertIn('DYNAMIC EMERGENCY',status.text)
        self.assertGreater(status.color.r,status.color.b)


if __name__ == '__main__':
    unittest.main()
