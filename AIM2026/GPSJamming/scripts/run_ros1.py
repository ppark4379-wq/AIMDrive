#!/usr/bin/env python3
"""ROS Noetic debug output for the existing UDP LiDAR pipeline (Python 3.8+)."""
from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, String
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from gps_jamming.lidar.walls import detect_walls
from gps_jamming.lidar.wall_ema import WallEMA
from gps_jamming.lidar.wall_width import WallWidthFallback
from gps_jamming.lidar.runtime import load_config, make_assembler, DEFAULT_CONFIG
from gps_jamming.interfaces.lidar_udp import LidarUDPReceiver, LidarUDPStream
from gps_jamming.pipeline import LidarPipeline
from gps_jamming.interfaces.wall_udp import WallUDPSender, line_edges, PORT


STAGES = ('raw', 'roi', 'voxel', 'ground', 'nonground')


def boundary_markers(boundaries, header, previous_keys=None):
    """Two stable markers; observation expiry belongs to WallEMA, not RViz."""
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
        # Transport watchdog only. Missing cells expire explicitly in WallEMA.
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


def main():
    rospy.init_node('gps_jamming_lidar')
    config = load_config(Path(rospy.get_param('~config', str(DEFAULT_CONFIG))))
    udp = dict(config['lidar']['udp'])
    udp['bind'] = rospy.get_param('~bind', '0.0.0.0')
    udp['port'] = int(rospy.get_param('~port', udp['port']))
    # Preserve every intermediate stage from the existing LiDAR pipeline.
    pipeline = LidarPipeline(config['lidar'], capture_debug=True)
    publishers = {stage: rospy.Publisher('~' + stage, PointCloud2, queue_size=1)
                  for stage in STAGES}
    status = rospy.Publisher('~status', String, queue_size=1)
    boundaries_pub = rospy.Publisher('~wall_boundaries', String, queue_size=1)
    markers_pub = rospy.Publisher('~wall_markers', MarkerArray, queue_size=1)
    filtered_pub = rospy.Publisher('~wall_boundaries_filtered', String, queue_size=1)
    wall_ema = WallEMA(**config.get('wall_ema', {}))
    wall_width = WallWidthFallback(**config.get('wall_width_fallback', {}))
    display_pub = rospy.Publisher('~wall_boundaries_display', String, queue_size=1)
    marker_keys = set()
    wall_sender = WallUDPSender(rospy.get_param('~boundary_udp_host', '127.0.0.1'),
                               int(rospy.get_param('~boundary_udp_port', PORT)))
    rospy.on_shutdown(wall_sender.close)
    rospy.loginfo('RViz wall geometry -> UDP %s:%s', *wall_sender.destination)
    wall_pubs = {side: rospy.Publisher('~wall_' + side, PointCloud2, queue_size=1)
                 for side in ('left', 'right', 'representatives_left', 'representatives_right')}
    assembler = make_assembler(config)
    rospy.loginfo('LiDAR UDP %s:%s -> ROS private point-cloud topics', udp['bind'], udp['port'])
    with LidarUDPReceiver(assembler, decoder=config['lidar']['decoder'], **udp) as receiver:
        with LidarUDPStream(receiver, config['lidar'].get('runtime', {}).get('frame_queue_size', 4)) as stream:
            while not rospy.is_shutdown():
                for frame in stream.read():
                    if rospy.is_shutdown():
                        break
                    result = pipeline.process(frame)
                    # All debug layers use one stamp. Unmapped sensor time must
                    # never masquerade as a verified acquisition timestamp.
                    source_time = result.timestamp is not None and result.timestamp_status == 'mapped'
                    stamp = (rospy.Time(*divmod(result.timestamp, 1_000_000_000))
                             if source_time else rospy.Time.now())
                    if not source_time:
                        rospy.logwarn_throttle(10, 'Unmapped sensor clock: RViz uses publish time; see ~status')
                    header = Header(stamp=stamp, frame_id=result.coordinate_frame)
                    boundaries, wall_clouds = detect_walls(
                        pipeline.last_debug['nonground'],
                        result.diagnostics.get('ground_plane'), **config.get('walls', {}))
                    boundaries.update(
                        schema_version=1, frame_id=result.coordinate_frame,
                        frame_sequence=result.frame_id, stamp_ns=stamp.to_nsec(),
                        receive_start_monotonic_ns=result.receive_monotonic_time,
                        receive_end_monotonic_ns=result.receive_end_monotonic_time,
                        sensor_timestamp_ns=result.timestamp,
                        raw_timestamp=result.raw_timestamp,
                        clock_epoch=result.clock_epoch,
                        timestamp_status=result.timestamp_status,
                        packet_count=result.packet_count,
                        stamp_source='sensor_mapped' if source_time else 'ros_publish_time',
                        complete=result.complete,
                        valid={side: bool(boundaries[side]) and result.complete for side in ('left', 'right')},
                        semantics='geometric_wall_candidates_not_drivable_area')
                    boundaries_pub.publish(String(data=json.dumps(boundaries, allow_nan=False)))
                    filtered = wall_ema.update(boundaries, time.monotonic(),
                                               result.receive_end_monotonic_time / 1e9)
                    filtered_pub.publish(String(data=json.dumps(filtered, allow_nan=False)))
                    display = wall_width.update(boundaries, result.receive_end_monotonic_time / 1e9)
                    display_pub.publish(String(data=json.dumps(display, allow_nan=False)))
                    markers_pub.publish(boundary_markers(display, header, marker_keys))
                    try:
                        wall_sender.send(display)
                    except (OSError, ValueError, TypeError, KeyError) as exc:
                        rospy.logerr_throttle(5, 'Wall UDP send failed: %s', exc)
                    for side in wall_pubs:
                        wall_pubs[side].publish(cloud_message(wall_clouds[side], header))
                    for stage, publisher in publishers.items():
                        publisher.publish(cloud_message(pipeline.last_debug[stage], header))
                    status.publish(String(data=json.dumps({
                        'frame_id': result.frame_id, 'coordinate_frame': result.coordinate_frame,
                        'raw_timestamp': result.raw_timestamp, 'timestamp': result.timestamp,
                        'stamp_source': 'sensor_mapped' if source_time else 'ros_publish_time',
                        'timestamp_status': result.timestamp_status, 'complete': result.complete,
                        'issues': result.issues, 'diagnostics': result.diagnostics,
                        'invalid_packets': receiver.invalid_packets,
                        'dropped_frames': stream.dropped_frames,
                    }, allow_nan=False)))
                rospy.loginfo_throttle(5, 'UDP invalid=%s dropped=%s last_error=%s',
                                       receiver.invalid_packets, stream.dropped_frames, receiver.last_error)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
