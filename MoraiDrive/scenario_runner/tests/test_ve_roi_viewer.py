"""Actual ROS message geometry, stale expiry and track continuity checks."""
import importlib.util
import io
import math
from pathlib import Path
import unittest

import rospy
from morai_msgs.msg import ObjectStatus, ObjectStatusList
from visualization_msgs.msg import Marker

path = Path(__file__).resolve().parents[1] / 'legacy_expert/ve_roi_viewer.py'
spec = importlib.util.spec_from_file_location('ve_roi_viewer', path)
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


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
        observations = viewer.Observations()
        observations.update(objects,10.)
        ego = dict(x=100.,y=200.,yaw=math.pi/2,received=10.,stamp_ns=100_050_000_000)
        return observations, ego

    def draw(self,obs,ego,now=10.05,diagnostics=None):
        return viewer.build_markers(obs,ego,diagnostics,now,rospy.Time(1))

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
        self.assertEqual({p.x for p in box.points},{0.,viewer.EGO_LENGTH})
        self.assertEqual({p.y for p in box.points},{-viewer.EGO_WIDTH/2,viewer.EGO_WIDTH/2})
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
        delta = viewer.MarkerDelta()
        initial = delta.update(self.draw(obs,ego),10.)
        self.assertTrue(any(m.ns == 'roi' for m in initial.markers))
        repeated = delta.update(self.draw(obs,ego),10.05)
        self.assertFalse(any(m.ns == 'roi' for m in repeated.markers))
        obs.objects.npc_list = []
        removed = delta.update(self.draw(obs,ego),10.06)
        self.assertTrue(any(m.ns == 'current_bbox' and m.action == Marker.DELETE for m in removed.markers))


if __name__ == '__main__':
    unittest.main()
