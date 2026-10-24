"""Read-only 8-second wall publication audit in a sourced ROS 1 environment."""
import json
import time
import rospy
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker

rows = {'raw': [], 'display': [], 'markers': []}


def on_json(msg, key):
    data = json.loads(msg.data)
    rows[key].append((time.monotonic(), data))


def on_markers(msg):
    rows['markers'].append((time.monotonic(), [m.lifetime.to_sec() for m in msg.markers
                                             if m.action == Marker.ADD]))


if __name__ == '__main__':
    rospy.init_node('wall_topic_audit', anonymous=True)
    rospy.Subscriber('/gps_jamming_lidar/wall_boundaries', String, on_json, 'raw')
    rospy.Subscriber('/gps_jamming_lidar/wall_boundaries_display', String, on_json, 'display')
    rospy.Subscriber('/gps_jamming_lidar/wall_markers', MarkerArray, on_markers)
    time.sleep(8)
    for key, samples in rows.items():
        gaps = [b[0]-a[0] for a, b in zip(samples, samples[1:])]
        summary = {'topic': key, 'messages': len(samples), 'max_gap_s': max(gaps, default=0)}
        if key != 'markers':
            for side in ('left', 'right'):
                summary[side+'_empty_frames'] = sum(not d[side] for _, d in samples)
        else:
            lives = [v for _, values in samples for v in values]
            summary['min_lifetime_s'] = min(lives, default=0)
            summary['lifetimes_under_50ms'] = sum(v < 0.05 for v in lives)
        print(json.dumps(summary))
