import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'MoraiDrive/expert/src')]
from AIM2026.Visualization.snapshot import write_snapshot, decode
from expert.src.interfaces.geometry import Point
from expert.src.interfaces.vehicle_state import VehicleState
from expert.src.interfaces.perception_data import ObjectInfo
from expert.src.planning.lattice.structures import CandidatePath, LatticePlanningResult


class SnapshotTests(unittest.TestCase):
    def snapshot(self):
        selected = CandidatePath(); selected.points = [Point(0,0),Point(10,0)]
        other = CandidatePath(); other.points = [Point(0,1),Point(10,1)]
        lane = SimpleNamespace(id='lane',section_index=1,lane_number=2,excluded=False,
                               points=[Point(0,0),Point(20,0)])
        result = LatticePlanningResult(selected.points,candidates=[selected,other],
            highway_lane_links=[lane],current_link_id='lane',mode='highway')
        objects=[ObjectInfo(5,0,1.,1,size_x=4.,size_y=2.,heading_deg=0.)]
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
        self.assertTrue(math.isinf(result.ttc))

    def test_original_ros_renderer(self):
        try:
            import rospy
        except ImportError:
            self.skipTest('ROS renderer runs in WSL')
        from expert.src.visualization.rviz_visualizer import LatticeRvizVisualizer
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


if __name__=='__main__': unittest.main()
