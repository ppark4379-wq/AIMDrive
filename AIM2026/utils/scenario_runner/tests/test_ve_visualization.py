# Check displayed geometry and clearing without a ROS master or a vehicle.
import importlib.util
import io
from pathlib import Path
import unittest

import rospy
from morai_msgs.msg import ObjectStatus, ObjectStatusList
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

VIEWER = Path(__file__).resolve().parents[1] / 'legacy_expert/ve_bev_viewer.py'
bev_spec = importlib.util.spec_from_file_location('ve_bev_viewer', VIEWER)
bev_viewer = importlib.util.module_from_spec(bev_spec)
bev_spec.loader.exec_module(bev_viewer)


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
        result = bev_viewer.build_markers(objects, ego, .01, rospy.Time(1))
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
        result = bev_viewer.build_markers(objects, ego, .01, rospy.Time(1))
        self.assertEqual(result.markers[0].action, Marker.DELETEALL)
        self.assertFalse(any(m.ns == 've_exact_bbox' for m in result.markers))
        self.assertIn('0 objects', result.markers[-1].text)

    def test_moving_npc_transport_is_displayed_with_track_identity(self):
        objects, ego = self.sample()
        box = objects.obstacle_list.pop()
        box.unique_id = 17
        box.name = 've_tracked_unknown:moving'
        objects.npc_list = [box]
        result = bev_viewer.build_markers(objects, ego, .01, rospy.Time(1))
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
            result = bev_viewer.build_markers(objects, ego, age, rospy.Time(1))
            self.assertEqual(result.markers[0].action, Marker.DELETEALL)
            self.assertFalse(any(m.ns == 've_exact_bbox' for m in result.markers))
            self.assertIn(status, result.markers[-1].text)
            self.assertTrue(all(m.lifetime.to_sec() > 0 for m in result.markers[1:]))

    def test_waiting_is_distinct_from_empty_detections(self):
        result = bev_viewer.build_markers(None, None, float('inf'), rospy.Time(0))
        self.assertIn('WAITING', result.markers[-1].text)


# Actual ROS message geometry, stale expiry and track continuity checks.
import math


path = Path(__file__).resolve().parents[1] / 'legacy_expert/ve_roi_viewer.py'
roi_spec = importlib.util.spec_from_file_location('ve_roi_viewer', path)
roi_viewer = importlib.util.module_from_spec(roi_spec)
roi_spec.loader.exec_module(roi_viewer)


class RoiTests(unittest.TestCase):
    def sample(self):
        objects = ObjectStatusList()
        objects.header.frame_id = 'map'
        objects.header.stamp = rospy.Time(100,0)
        obj = ObjectStatus()
        obj.unique_id, obj.name = 7, 've_tracked_unknown:moving'
        obj.position.x, obj.position.y = 98., 210.
        obj.heading, obj.size.x, obj.size.y, obj.velocity.x = 120., 4., 2., 18.
        objects.npc_list = [obj]
        observations = roi_viewer.Observations()
        observations.update(objects,10.)
        ego = dict(x=100.,y=200.,yaw=math.pi/2,received=10.,stamp_ns=100_050_000_000)
        return observations, ego

    def draw(self,obs,ego,now=10.05,diagnostics=None):
        return roi_viewer.build_markers(obs,ego,diagnostics,now,rospy.Time(1))

    def test_map_to_base_rotates_center_heading_without_changing_size_or_speed(self):
        obs,ego = self.sample()
        result = self.draw(obs,ego)
        box = next(m for m in result.markers if m.ns == 'current_bbox')
        self.assertEqual(box.header.frame_id,'base_link')
        a,b,c = box.points[:3]
        self.assertAlmostEqual(math.hypot(a.x-b.x,a.y-b.y),2.)
        self.assertAlmostEqual(math.hypot(b.x-c.x,b.y-c.y),4.)
        self.assertAlmostEqual(sum(p.x for p in box.points[:4])/4,10.)
        self.assertAlmostEqual(sum(p.y for p in box.points[:4])/4,2.)
        label = next(m for m in result.markers if m.ns == 'object_speed')
        self.assertIn('18.0 km/h (5.0 m/s)',label.text)
        self.assertIn('x=+10.0 y=+2.0',label.text)
        result.serialize(io.BytesIO())

    def test_roi_matches_exact_asymmetric_rectangle(self):
        obs,ego = self.sample()
        roi = next(m for m in self.draw(obs,ego).markers if m.ns == 'roi')
        self.assertEqual({p.x for p in roi.points},{-20.,30.})
        self.assertEqual({p.y for p in roi.points},{-25.,25.})

    def test_ego_bbox_matches_planner_back_reference_and_dimensions(self):
        obs,ego = self.sample()
        box = next(m for m in self.draw(obs,ego).markers if m.ns == 'ego_bbox')
        self.assertEqual({p.x for p in box.points},{0.,roi_viewer.EGO_LENGTH})
        self.assertEqual({p.y for p in box.points},{-roi_viewer.EGO_WIDTH/2,roi_viewer.EGO_WIDTH/2})
        self.assertEqual(box.header.frame_id,'base_link')

    def test_duplicate_packet_does_not_refresh_age_or_create_motion(self):
        obs,ego = self.sample()
        self.assertFalse(obs.update(obs.objects,10.3))
        self.assertEqual(obs.received,10.)
        self.assertEqual(len(obs.trails[7]),1)
        ego['received'] = 10.5
        result = self.draw(obs,ego,10.5)
        self.assertFalse(any(m.ns == 'current_bbox' for m in result.markers))
        self.assertIn('STALE',next(m.text for m in result.markers if m.ns == 'status'))

    def test_upstream_age_is_included_and_empty_update_removes_box(self):
        obs,ego = self.sample()
        diagnostics = dict(ready=True,source_stamp_ns=obs.key,metrics=dict(input_receipt_age_ms=320))
        self.assertFalse(any(m.ns == 'current_bbox' for m in self.draw(obs,ego,10.05,diagnostics).markers))
        empty = ObjectStatusList()
        empty.header.frame_id,empty.header.stamp = 'map',rospy.Time(100,100_000_000)
        obs.update(empty,10.1)
        result = self.draw(obs,ego,10.12)
        self.assertFalse(any(m.ns == 'current_bbox' for m in result.markers))
        self.assertIn('0 current objects',next(m.text for m in result.markers if m.ns == 'status'))
        self.assertEqual(len(obs.misses),1)
        self.assertEqual(result.markers[0].action,Marker.DELETEALL)

    def test_current_ego_motion_is_applied_without_extrapolating_object(self):
        obs,ego = self.sample()
        ego['y'] += 3
        label = next(m for m in self.draw(obs,ego).markers if m.ns == 'object_speed')
        self.assertIn('x=+7.0 y=+2.0',label.text)

    def test_delivery_rate_uses_matched_sender_clock_not_wsl_receipt_clock(self):
        obs,ego = self.sample()
        obs.match_health(dict(source_stamp_ns=obs.key,publish_perf_ns=1_000_000_000))
        obs.match_health(dict(source_stamp_ns=obs.key,publish_perf_ns=1_000_000_000))
        obs.objects.header.stamp = rospy.Time(100,100_000_000)
        obs.update(obs.objects,10.09)
        obs.match_health(dict(source_stamp_ns=obs.key,publish_perf_ns=1_100_000_000))
        status = next(m.text for m in self.draw(obs,ego,10.12).markers if m.ns == 'status')
        self.assertIn('ROS matched 10.00 Hz',status)
        self.assertEqual(len(obs.matched_times),2)

    def test_rviz_delta_reuses_static_geometry_and_explicitly_deletes_lost_boxes(self):
        obs,ego = self.sample()
        delta = roi_viewer.MarkerDelta()
        initial = delta.update(self.draw(obs,ego),10.)
        self.assertTrue(any(m.ns == 'roi' for m in initial.markers))
        repeated = delta.update(self.draw(obs,ego),10.05)
        self.assertFalse(any(m.ns == 'roi' for m in repeated.markers))
        obs.objects.npc_list = []
        removed = delta.update(self.draw(obs,ego),10.06)
        self.assertTrue(any(m.ns == 'current_bbox' and m.action == Marker.DELETE for m in removed.markers))


if __name__ == "__main__":
    unittest.main()
