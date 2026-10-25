"""ROS-only display adapter; receives the exact same result as the UDP sender."""
import json
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, String
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from src.UDP import line_edges

STAGES = ('raw', 'roi', 'voxel', 'ground', 'nonground')

def boundary_markers(boundaries, header, previous_keys=None):
    """Two stable markers using the same display geometry as UDP."""
    markers = []
    current_keys = set()
    for side, color in (('left', (0.1, 0.5, 1.0)), ('right', (1.0, 0.55, 0.1))):
        edges = []
        for pair in line_edges(boundaries, side):
            edges.extend([Point(x=p[0], y=p[1], z=p[2]) for p in pair])
        if not edges:
            continue
        current_keys.add((side, 0))
        marker = Marker(header=header, ns=side, id=0,
                        type=Marker.LINE_LIST, action=Marker.ADD)
        marker.pose.orientation.w = 1
        marker.scale.x = 0.08
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 0.45 if all(s.get('inferred', False) for s in boundaries[side]) else 1.0
        # Transport watchdog. Missing sides are explicitly deleted below.
        marker.lifetime = rospy.Duration(0.5)
        marker.points = edges
        markers.append(marker)
    if previous_keys is not None:
        for side, index in sorted(previous_keys-current_keys):
            markers.append(Marker(header=header, ns=side, id=index, action=Marker.DELETE))
        previous_keys.clear()
        previous_keys.update(current_keys)
    return MarkerArray(markers=markers)


def cloud_message(points, header):
    """Pack XYZI as little-endian float32, including valid empty clouds."""
    points = np.asarray(points, dtype='<f4')
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError('Expected Nx4 XYZI points')
    fields = [PointField(name=name, offset=i * 4,
                         datatype=PointField.FLOAT32, count=1)
              for i, name in enumerate(('x', 'y', 'z', 'intensity'))]
    return PointCloud2(header=header, height=1, width=len(points), fields=fields,
                       is_bigendian=False, point_step=16, row_step=len(points) * 16,
                       data=points.tobytes(), is_dense=bool(np.isfinite(points).all()))


class ROS1Publisher:
    def __init__(self):
        self.publishers = {stage: rospy.Publisher('~' + stage, PointCloud2, queue_size=1)
                           for stage in STAGES}
        self.status = rospy.Publisher('~status', String, queue_size=1)
        self.boundaries_pub = rospy.Publisher('~wall_boundaries', String, queue_size=1)
        self.display_pub = rospy.Publisher('~wall_boundaries_display', String, queue_size=1)
        self.markers_pub = rospy.Publisher('~wall_markers', MarkerArray, queue_size=1)
        self.marker_keys = set()
        self.wall_pubs = {side: rospy.Publisher('~wall_' + side, PointCloud2, queue_size=1)
                          for side in ('left', 'right', 'representatives_left', 'representatives_right')}

    @staticmethod
    def now_ns():
        return rospy.Time.now().to_nsec()

    @staticmethod
    def header(result):
        return Header(stamp=rospy.Time(*divmod(result.boundaries['stamp_ns'], 1_000_000_000)),
                      frame_id=result.frame.coordinate_frame)

    def publish_boundaries(self, result):
        b = result.boundaries
        header = self.header(result)
        if b['stamp_source'] != 'sensor_mapped':
            rospy.logwarn_throttle(10, 'Unmapped sensor clock: RViz uses publish time; see ~status')
        self.boundaries_pub.publish(String(data=json.dumps(result.raw_boundaries, allow_nan=False)))
        self.display_pub.publish(String(data=json.dumps(b, allow_nan=False)))
        self.markers_pub.publish(boundary_markers(b, header, self.marker_keys))

    def publish_clouds(self, result, invalid_packets=0, dropped_frames=0):
        header = self.header(result)
        frame = result.frame
        for side, publisher in self.wall_pubs.items():
            publisher.publish(cloud_message(result.wall_clouds[side], header))
        for stage, publisher in self.publishers.items():
            publisher.publish(cloud_message(result.stages[stage], header))
        self.status.publish(String(data=json.dumps({
            'frame_id': frame.frame_id, 'coordinate_frame': frame.coordinate_frame,
            'raw_timestamp': frame.raw_timestamp, 'timestamp': frame.timestamp,
            'stamp_source': result.boundaries['stamp_source'], 'timestamp_status': frame.timestamp_status,
            'complete': frame.complete, 'issues': frame.issues, 'diagnostics': frame.diagnostics,
            'invalid_packets': invalid_packets, 'dropped_frames': dropped_frames,
        }, allow_nan=False)))
