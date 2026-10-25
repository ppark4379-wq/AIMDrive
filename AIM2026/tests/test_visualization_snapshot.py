import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'AIM2026/Expert/src')]
from AIM2026.Visualization.snapshot import write_snapshot, decode
from AIM2026.Expert.src.interfaces.geometry import Point
from AIM2026.Expert.src.interfaces.vehicle_state import VehicleState
from AIM2026.Expert.src.interfaces.perception_data import ObjectInfo
from AIM2026.Expert.src.planning.lattice.structures import CandidatePath, LatticePlanningResult
from AIM2026.Expert.src.planning.highway.common.types import LaneMatch


class SnapshotTests(unittest.TestCase):
    def test_new_rviz_subscriber_receives_full_frame_after_delta_cache(self):
        from AIM2026.Visualization.rviz import OriginalDisplay
        viewer = object.__new__(OriginalDisplay)
        count = [0]
        viewer.publisher = SimpleNamespace(get_num_connections=lambda: count[0])
        viewer._marker_cache = {'roi': {'old': 1}}
        viewer._marker_ids = {'roi': {'old'}}
        viewer._render_key = 'old'
        viewer.refresh_connections()
        self.assertEqual(viewer._render_key,'old')
        count[0] = 1
        viewer.refresh_connections()
        self.assertIsNone(viewer._render_key)
        self.assertEqual(viewer._marker_cache,{})
        viewer._render_key = 'new'
        viewer.refresh_connections()
        self.assertEqual(viewer._render_key,'new')

    def world_overlay(self):
        try:
            import rospy
        except ImportError:
            self.skipTest('ROS renderer runs in WSL')
        from AIM2026.Visualization.world import WorldOverlay
        from AIM2026.Expert.src.visualization.rviz_visualizer import LatticeRvizVisualizer
        return WorldOverlay(object.__new__(LatticeRvizVisualizer))

    def test_world_map_contains_course_highway_static_and_roundabout(self):
        overlay = self.world_overlay()
        namespaces = {m.ns for m in overlay.static.markers}
        self.assertNotIn('map_road_centers', namespaces)
        self.assertTrue({'highway_map_lanes','static_reference_recordings',
                         'roundabout_approach','roundabout_circulating_lane','static_planning_zone'} <= namespaces)
        self.assertTrue(overlay.signals)
        for marker in overlay.static.markers:
            marker.serialize(__import__('io').BytesIO())

    def test_world_roi_rotates_with_ego_and_keeps_dimensions(self):
        import math
        overlay = self.world_overlay()
        import rospy
        state = self.snapshot()
        state['vehicle'].yaw = math.pi/2
        markers = overlay.dynamic(state, rospy.Time(0), False)
        roi = next(m for m in markers.markers if m.ns == 've_roi_world')
        self.assertAlmostEqual(roi.points[0].x, state['vehicle'].position.x-25.)
        self.assertAlmostEqual(roi.points[0].y, state['vehicle'].position.y+30.)
        self.assertEqual(roi.points[0],roi.points[-1])

    def test_signal_color_requires_current_bound_ve_policy_and_stale_is_gray(self):
        overlay = self.world_overlay()
        import rospy
        state = self.snapshot()
        identity = overlay.signals[0][0]
        state['signal'] = dict(id=identity,code=1,bound=True,reason='test')
        for stale, bound, red in ((False,True,True),(True,True,False),(False,False,False)):
            state['signal']['bound'] = bound
            markers = overlay.dynamic(state,rospy.Time(0),stale)
            lamp = next(m for m in markers.markers if m.ns=='signal_stop_nodes' and m.id==0)
            self.assertEqual(lamp.color.r==1.,red)

    def test_second_merge_geometry_is_shared_by_map_and_roi(self):
        try:
            import rospy
        except ImportError:
            self.skipTest('ROS renderer runs in WSL')
        from AIM2026.Visualization.roi import RoiDisplay
        from AIM2026.Expert.src.visualization.rviz_visualizer import LatticeRvizVisualizer
        from AIM2026.Expert.src.config.parameters import highway as p
        state = self.snapshot()
        state['vehicle'].yaw = 0.
        renderer = object.__new__(LatticeRvizVisualizer)
        markers = renderer._build_second_merge_markers(state['result'], rospy.Time(0))
        outline = next(m for m in markers if m.ns == 'second_merge_zone')
        self.assertAlmostEqual(outline.points[0].x, p.CONSERVATIVE_MERGE_POINT[0]+p.CONSERVATIVE_MERGE_ENTRY_RADIUS)
        self.assertAlmostEqual(outline.points[0].x, outline.points[-1].x)
        display = object.__new__(RoiDisplay)
        display.renderer = renderer
        local = display.markers(state, rospy.Time(0), False).markers
        converted = next(m for m in local if m.ns == 'second_merge_zone')
        self.assertEqual(converted.header.frame_id, 'base_link')
        self.assertAlmostEqual(converted.points[0].x, outline.points[0].x-state['vehicle'].position.x)
        label = next(m for m in local if m.ns == 'second_merge_label')
        self.assertAlmostEqual(label.pose.position.y, p.CONSERVATIVE_MERGE_POINT[1]-state['vehicle'].position.y)

    def test_merge_overlay_uses_planner_target_and_world_to_ego_transform(self):
        try:
            import rospy
        except ImportError:
            self.skipTest('ROS renderer runs in WSL')
        from AIM2026.Visualization.roi import RoiDisplay
        from AIM2026.Expert.src.visualization.rviz_visualizer import LatticeRvizVisualizer
        state = self.snapshot()
        state['vehicle'].yaw = math.pi/2.
        state['objects'][0].unique_id = 42
        state['result'].debug_details['entry_merge'] = dict(lead_id=42, phase='MERGING',
            target_event='held', search_front=30., search_rear=20.,
            search_polygon=[[3.,-20.],[7.,-20.],[7.,30.],[3.,30.],[3.,-20.]])
        renderer = object.__new__(LatticeRvizVisualizer)
        boxes = renderer._build_object_markers(state['result'], state['vehicle'], state['objects'], rospy.Time(0))
        target = next(m for m in boxes.markers if m.ns == 'lattice_objects')
        self.assertEqual((target.color.r,target.color.g,target.color.b), (1.,0.,0.))
        display = object.__new__(RoiDisplay)
        display.renderer = renderer
        markers = display.markers(state, rospy.Time(0), False).markers
        outline = next(m for m in markers if m.ns == 'merge_search_range')
        self.assertEqual(outline.header.frame_id, 'base_link')
        self.assertAlmostEqual(outline.points[0].x, -20.)
        self.assertAlmostEqual(outline.points[0].y, -3.)
        target = next(m for m in markers if m.ns == 'current_bbox')
        self.assertEqual((target.color.r,target.color.g,target.color.b), (1.,0.,0.))
        stale = display.markers(state, rospy.Time(0), True).markers
        self.assertLess(next(m for m in stale if m.ns == 'current_bbox').color.a, .5)

    def test_snapshot_read_failure_does_not_blank_fresh_frame(self):
        from AIM2026.Visualization.rviz import SnapshotReader
        import os
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            snapshot = folder/'visualization.json'
            snapshot.write_text('{"objects": []}', encoding='utf-8')
            os.utime(snapshot, (100., 100.))
            reader = SnapshotReader()
            state, stale = reader.read(folder, now=100.1)
            self.assertFalse(stale)
            # Simulate a temporary access/partial-write failure on a changed file.
            snapshot.write_text('{', encoding='utf-8')
            os.utime(snapshot, (100.2, 100.2))
            retry, stale = reader.read(folder, now=100.3)
            self.assertIs(retry, state)
            self.assertFalse(stale)
            self.assertTrue(reader.read(folder, now=101.)[1])
            snapshot.write_text('{"objects": [1]}', encoding='utf-8')
            os.utime(snapshot, (101., 101.))
            updated, stale = reader.read(folder, now=101.1)
            self.assertFalse(stale)
            self.assertEqual(updated['objects'], [1])
            with patch('AIM2026.Visualization.rviz.display_state', side_effect=AssertionError('cached')):
                self.assertIs(reader.read(folder, now=101.2)[0], updated)

    def snapshot(self):
        selected = CandidatePath(); selected.points = [Point(0,0),Point(10,0)]
        other = CandidatePath(); other.points = [Point(0,1),Point(10,1)]
        lane = SimpleNamespace(id='lane',section_index=1,lane_number=2,excluded=False,
                               points=[Point(0,0),Point(20,0)])
        result = LatticePlanningResult(selected.points,candidates=[selected,other],
            highway_lane_links=[lane],current_link_id='lane',mode='highway')
        objects=[ObjectInfo(5,0,1.,1,size_x=4.,size_y=2.,heading_deg=0.)]
        result.object_lane_matches = [({'object_info': objects[0]},
                                      LaneMatch(lane, 5., 0., 5., 0., 0.))]
        result.predicted_object_paths = [[Point(5,0), Point(6,0)]]
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'state.json'
            write_snapshot(target,result,VehicleState(0,0,0,2),objects,selected.points)
            state=decode(json.loads(target.read_text()))
        self.assertFalse(hasattr(selected,'selected_preview'))
        return state

    def test_preserves_selected_candidate_lanes_objects_and_infinite_metrics(self):
        state=self.snapshot(); result=state['result']
        self.assertTrue(result.candidates[0].selected_preview)
        self.assertFalse(result.candidates[1].selected_preview)
        self.assertEqual(result.highway_lane_links[0].lane_number,2)
        self.assertEqual(state['objects'][0].position.x,5)
        self.assertEqual(result.object_lane_matches[0][1].lane_link.id, 'lane')
        self.assertTrue(math.isinf(result.ttc))

    def test_original_ros_renderer(self):
        try:
            import rospy
        except ImportError:
            self.skipTest('ROS renderer runs in WSL')
        from AIM2026.Expert.src.visualization.rviz_visualizer import LatticeRvizVisualizer
        state=self.snapshot(); renderer=object.__new__(LatticeRvizVisualizer)
        result,vehicle=state['result'],state['vehicle'];stamp=rospy.Time(0)
        paths=renderer._build_path_markers(result,vehicle,stamp)
        lanes=renderer._build_lane_markers(result,stamp)
        objects=renderer._build_object_markers(result,vehicle,state['objects'],stamp)
        candidates=[m for m in paths.markers if m.ns.startswith('candidate_paths')]
        self.assertEqual((candidates[0].color.r,candidates[0].color.g),(1.,0.))
        self.assertEqual((candidates[1].color.r,candidates[1].color.g),(0.,1.))
        self.assertTrue(any(m.ns=='ego_shape' for m in paths.markers))
        self.assertTrue(any(m.ns=='highway_lane_labels' for m in lanes.markers))
        self.assertTrue(any(m.ns=='lattice_objects' for m in objects.markers))
        self.assertTrue(any(m.ns=='object_lane_labels' and m.text=='S1 L2 lane' for m in objects.markers))
        # A pre-fix recording must still show objects/predictions without crashing.
        result.object_lane_matches[0][1] = SimpleNamespace(x=5., y=0., z=0.)
        legacy = renderer._build_object_markers(result,vehicle,state['objects'],stamp)
        self.assertTrue(any(m.ns=='lattice_objects' for m in legacy.markers))
        self.assertTrue(any(m.ns=='predicted_object_obbs' for m in legacy.markers))
        self.assertFalse(any(m.ns=='object_lane_labels' for m in legacy.markers))

    def test_viewer_updates_without_expiration_or_repeated_clear(self):
        try:
            import rospy
        except ImportError:
            self.skipTest('ROS renderer runs in WSL')
        from AIM2026.Visualization.rviz import OriginalDisplay
        from AIM2026.Expert.src.visualization.rviz_visualizer import LatticeRvizVisualizer
        from visualization_msgs.msg import Marker
        state = self.snapshot()
        renderer = object.__new__(LatticeRvizVisualizer)
        viewer = object.__new__(OriginalDisplay)
        viewer._marker_ids = {}
        stamp = rospy.Time(0)
        def draw(topic):
            if topic == 'paths':
                return renderer._build_path_markers(state['result'], state['vehicle'], stamp)
            if topic == 'objects':
                return renderer._build_object_markers(state['result'], state['vehicle'], state['objects'], stamp)
            return renderer._build_lane_markers(state['result'], stamp)
        for topic in ('paths', 'objects', 'lanes'):
            first = viewer.update_markers(topic, draw(topic), stamp)
            self.assertEqual(sum(m.action == Marker.DELETEALL for m in first.markers), 1)
            expected = {(m.ns, m.id) for m in first.markers if m.action == Marker.ADD}
            second = viewer.update_markers(topic, draw(topic), stamp)
            self.assertEqual(second.markers, [])  # unchanged geometry stays in RViz
            self.assertTrue(all(m.lifetime.to_sec() == 0 for m in first.markers))
            changed = draw(topic)
            added = next(m for m in changed.markers if m.action == Marker.ADD)
            added.color.a = .123
            delta = viewer.update_markers(topic, changed, stamp)
            self.assertEqual(len(delta.markers), 1)
            self.assertEqual(delta.markers[0].color.a, .123)
            # A temporary read delay cannot expire the markers. A real empty
            # next frame must, however, explicitly remove every old marker.
            empty = type(second)()
            cleared = viewer.update_markers(topic, empty, stamp)
            self.assertEqual({(m.ns, m.id) for m in cleared.markers}, expected)
            self.assertTrue(all(m.action == Marker.DELETE for m in cleared.markers))
            self.assertEqual(viewer.update_markers(topic, type(second)(), stamp).markers, [])

    def test_merge_target_is_red_filled_labeled_and_removed_when_released(self):
        try:
            import rospy
        except ImportError:
            self.skipTest('ROS renderer runs in WSL')
        from AIM2026.Visualization.roi import RoiDisplay
        from AIM2026.Expert.src.visualization.rviz_visualizer import LatticeRvizVisualizer
        from visualization_msgs.msg import Marker
        viewer = object.__new__(RoiDisplay)
        viewer.renderer = object.__new__(LatticeRvizVisualizer)
        viewer._marker_ids = {}
        state = self.snapshot()
        state['objects'][0].unique_id = 7
        state['result'].debug_details['entry_merge'] = dict(lead_id=7, phase='APPROACH')
        stamp = rospy.Time(0)
        markers = viewer.update_markers('roi', viewer.markers(state, stamp, False), stamp)
        fill = next(m for m in markers.markers if m.ns == 'merge_target_vehicle')
        self.assertEqual(fill.type, Marker.CUBE)
        self.assertEqual((fill.color.r,fill.color.g,fill.color.b), (1.,0.,0.))
        self.assertEqual(fill.scale.x, state['objects'][0].size_x)
        label = next(m for m in markers.markers if m.ns == 'object_speed')
        self.assertIn('MERGE FOLLOW TARGET', label.text)
        stale = viewer.markers(state, stamp, True)
        fill = next(m for m in stale.markers if m.ns == 'merge_target_vehicle')
        self.assertGreater(fill.color.r, fill.color.g)
        self.assertLess(fill.color.a, .8)
        self.assertIn('LAST RECORDED TARGET', next(m.text for m in stale.markers if m.ns == 'object_speed'))
        state['result'].debug_details['entry_merge']['lead_id'] = None
        released = viewer.update_markers('roi', viewer.markers(state, stamp, False), stamp)
        self.assertTrue(any(m.ns=='merge_target_vehicle' and m.action==Marker.DELETE for m in released.markers))
        label = next(m for m in released.markers if m.ns == 'object_speed')
        self.assertNotIn('TARGET', label.text)

    def test_october_roi_uses_udp_snapshot_and_rotates_candidates(self):
        try:
            import rospy
        except ImportError:
            self.skipTest('ROS renderer runs in WSL')
        import io
        from AIM2026.Visualization.roi import RoiDisplay
        from AIM2026.Expert.src.visualization.rviz_visualizer import LatticeRvizVisualizer
        viewer = object.__new__(RoiDisplay)
        viewer.renderer = object.__new__(LatticeRvizVisualizer)
        state = self.snapshot()
        state['vehicle'].yaw = math.pi / 2
        state['objects'][0].name = 've_tracked_unknown:moving'
        state['objects'][0].unique_id = 7
        markers = viewer.markers(state, rospy.Time(0), False)
        roi = next(m for m in markers.markers if m.ns == 'roi')
        self.assertEqual({p.x for p in roi.points}, {-20., 30.})
        self.assertEqual({p.y for p in roi.points}, {-25., 25.})
        path = next(m for m in markers.markers if m.ns == 'selected_path')
        self.assertEqual(path.header.frame_id, 'base_link')
        self.assertAlmostEqual(path.points[-1].x, 0.)
        self.assertAlmostEqual(path.points[-1].y, -10.)
        box = next(m for m in markers.markers if m.ns == 'current_bbox')
        self.assertAlmostEqual(sum(p.y for p in box.points[:4])/4, -5.)
        label = next(m for m in markers.markers if m.ns == 'object_speed')
        self.assertIn('3.6 km/h (1.0 m/s)', label.text)
        self.assertTrue(any(m.ns.startswith('candidate_paths_') for m in markers.markers))
        markers.serialize(io.BytesIO())
        state['objects'][0].speed_unknown = True
        unknown = viewer.markers(state, rospy.Time(0), False)
        self.assertIn('UNKNOWN', next(m.text for m in unknown.markers if m.ns == 'object_speed'))
        stale = viewer.markers(state, rospy.Time(0), True)
        self.assertTrue(any(m.ns == 'roi' for m in stale.markers))
        self.assertIn('LAST RECORDED', next(m.text for m in stale.markers if m.ns == 'status'))
        for ns in ('current_bbox', 'selected_path'):
            marker = next(m for m in stale.markers if m.ns == ns)
            self.assertLess(marker.color.a, 1.)
            self.assertEqual(marker.color.r, marker.color.g)
        waiting = viewer.markers(None, rospy.Time(0), True)
        self.assertIn('WAITING', next(m.text for m in waiting.markers if m.ns == 'status'))
        self.assertFalse(any(m.ns in ('current_bbox', 'selected_path') for m in waiting.markers))


if __name__=='__main__': unittest.main()
