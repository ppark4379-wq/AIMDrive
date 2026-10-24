"""Check displayed geometry and clearing without a ROS master or a vehicle."""
import importlib.util
import io
from pathlib import Path
import unittest

import rospy
from morai_msgs.msg import ObjectStatus, ObjectStatusList
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

VIEWER = Path(__file__).resolve().parents[1] / 'legacy_expert/ve_bev_viewer.py'
spec = importlib.util.spec_from_file_location('ve_bev_viewer', VIEWER)
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


class VeBevTests(unittest.TestCase):
    def sample(self):
        objects = ObjectStatusList()
        objects.header.frame_id = 'map'
        box = ObjectStatus()
        box.position.x, box.position.y = 10., 20.
        box.size.x, box.size.y = .8, .2
        box.heading, box.velocity.x = 90., 7.2
        objects.obstacle_list = [box]
        ego = Odometry()
        ego.pose.pose.position.x, ego.pose.pose.position.y = 10., 17.
        return objects, ego

    def test_small_rotated_box_preserves_world_axes_and_true_dimensions(self):
        objects, ego = self.sample()
        result = viewer.build_markers(objects, ego, .01, rospy.Time(1))
        box = next(m for m in result.markers if m.ns == 've_exact_bbox')
        xs, ys = [p.x for p in box.points], [p.y for p in box.points]
        self.assertAlmostEqual(max(xs) - min(xs), .2)
        self.assertAlmostEqual(max(ys) - min(ys), .8)
        self.assertAlmostEqual(box.points[0].x, 9.9)
        self.assertAlmostEqual(box.points[0].y, 20.4)
        self.assertEqual(box.points[0], box.points[-1])
        label = next(m for m in result.markers if m.ns == 've_bbox_labels')
        self.assertIn('3.0m', label.text)
        self.assertIn('0.80x0.20m', label.text)
        self.assertIn('7.2km/h', label.text)
        result.serialize(io.BytesIO())

    def test_empty_input_clears_old_boxes_without_being_reported_stale(self):
        objects, ego = self.sample()
        objects.obstacle_list = []
        result = viewer.build_markers(objects, ego, .01, rospy.Time(1))
        self.assertEqual(result.markers[0].action, Marker.DELETEALL)
        self.assertFalse(any(m.ns == 've_exact_bbox' for m in result.markers))
        self.assertIn('0 objects', result.markers[-1].text)

    def test_moving_npc_transport_is_displayed_with_track_identity(self):
        objects, ego = self.sample()
        box = objects.obstacle_list.pop()
        box.unique_id = 17
        box.name = 've_tracked_unknown:moving'
        objects.npc_list = [box]
        result = viewer.build_markers(objects, ego, .01, rospy.Time(1))
        drawn = next(m for m in result.markers if m.ns == 've_exact_bbox')
        self.assertGreater(drawn.color.b, drawn.color.r)
        label = next(m for m in result.markers if m.ns == 've_bbox_labels')
        self.assertIn('17', label.text)
        self.assertIn('moving', label.text)
        self.assertIn('1 objects', result.markers[-1].text)
        result.serialize(io.BytesIO())

    def test_stale_and_wrong_frame_are_not_displayed_as_current_detections(self):
        objects, ego = self.sample()
        for age, frame, status in ((1., 'map', 'STALE'), (.01, 'ego', 'UNEXPECTED FRAME')):
            objects.header.frame_id = frame
            result = viewer.build_markers(objects, ego, age, rospy.Time(1))
            self.assertEqual(result.markers[0].action, Marker.DELETEALL)
            self.assertFalse(any(m.ns == 've_exact_bbox' for m in result.markers))
            self.assertIn(status, result.markers[-1].text)
            self.assertTrue(all(m.lifetime.to_sec() > 0 for m in result.markers[1:]))

    def test_waiting_is_distinct_from_empty_detections(self):
        result = viewer.build_markers(None, None, float('inf'), rospy.Time(0))
        self.assertIn('WAITING', result.markers[-1].text)


if __name__ == '__main__':
    unittest.main()
