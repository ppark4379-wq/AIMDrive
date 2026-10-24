"""Run after sourcing ROS Noetic to compare real Marker coordinates with UDP."""
import importlib.util
from pathlib import Path
import unittest

import test_wall_udp

try:
    from std_msgs.msg import Header
except ImportError:
    Header = None


@unittest.skipIf(Header is None, 'Requires ROS Noetic environment')
class MarkerTransportTest(unittest.TestCase):
    def test_actual_marker_coordinates_equal_udp(self):
        path = Path(__file__).resolve().parents[1] / 'visualization' / 'ros1_publisher.py'
        spec = importlib.util.spec_from_file_location('wall_node', path)
        node = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(node)
        fixture = test_wall_udp.WallTransportTest()
        fixture.setUp()
        try:
            fixture.sender.send(fixture.display)
            packet = fixture.receive()
            markers = node.boundary_markers(fixture.display, Header(frame_id='base_link'))
            self.assertEqual({m.ns for m in markers.markers}, {'left', 'right'})
            for marker in markers.markers:
                xyz = [[p.x, p.y, p.z] for p in marker.points]
                received = [p for pair in node.line_edges(packet['boundaries'], marker.ns) for p in pair]
                self.assertEqual(xyz, received)
        finally:
            fixture.tearDown()
