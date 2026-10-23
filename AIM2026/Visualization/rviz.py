#!/usr/bin/env python3
"""Read UDP Expert logs for RViz. No sensor socket or vehicle control publisher."""
import argparse
import csv
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
import xmlrpc.client

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO/'AIM2026/Expert/src')]


def latest_record(path, max_age=.75):
    """Read only a bounded tail; ignore partial writes and stale files.

    File wall-clock timestamps are shared by Windows and WSL; process
    monotonic clocks are deliberately not compared across the two systems.
    """
    try:
        if max_age is not None and time.time() - path.stat().st_mtime > max_age:
            return None
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 512 * 1024))
            lines = stream.read().split(b'\n')[:-1]
        for line in reversed(lines):
            try:
                return json.loads(line)
            except (ValueError, UnicodeError):
                continue
    except OSError:
        pass
    return None


class OriginalDisplay:
    """Publish the original b761985a topics using the retained, unchanged renderer."""
    def __init__(self):
        import rospy
        from nav_msgs.msg import Path, Odometry
        from expert.src.visualization.rviz_visualizer import LatticeRvizVisualizer
        self.renderer = LatticeRvizVisualizer()
        self.global_pub = rospy.Publisher('/global_path', Path, queue_size=1, latch=True)
        self.local_pub = rospy.Publisher('/local_path', Path, queue_size=1)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=1)

    @staticmethod
    def path_message(points, stamp):
        from nav_msgs.msg import Path
        from geometry_msgs.msg import PoseStamped
        message = Path(); message.header.frame_id = 'map'; message.header.stamp = stamp
        for point in points:
            pose = PoseStamped(); pose.header = message.header
            pose.pose.position.x = float(point.x if hasattr(point,'x') else point[0])
            pose.pose.position.y = float(point.y if hasattr(point,'y') else point[1])
            pose.pose.orientation.w = 1.
            message.poses.append(pose)
        return message

    def publish(self, route, state, stamp, stale):
        from nav_msgs.msg import Odometry
        self.global_pub.publish(self.path_message(route,stamp))
        if state is None:
            return
        vehicle, result = state['vehicle'], state['result']
        paths = self.renderer._build_path_markers(result,vehicle,stamp)
        if stale:
            for marker in paths.markers:
                if marker.ns == 'lattice_status':
                    marker.text = 'STALE - LAST RECORDED STATE\n'+marker.text
        self.renderer.paths_pub.publish(paths)
        self.renderer.objects_pub.publish(self.renderer._build_object_markers(result,vehicle,state['objects'],stamp))
        self.renderer.lanes_pub.publish(self.renderer._build_lane_markers(result,stamp))
        self.local_pub.publish(self.path_message(state['selected_path'],stamp))
        odom = Odometry(); odom.header.frame_id='map'; odom.header.stamp=stamp; odom.child_frame_id='gps'
        odom.pose.pose.position.x = vehicle.position.x; odom.pose.pose.position.y=vehicle.position.y
        odom.pose.pose.orientation.z=math.sin(vehicle.yaw/2);odom.pose.pose.orientation.w=math.cos(vehicle.yaw/2)
        self.odom_pub.publish(odom)


def display_state(folder, path_row):
    from AIM2026.Visualization.snapshot import decode
    if folder and (folder/'visualization.json').is_file():
        try:
            return decode(json.loads((folder/'visualization.json').read_text(encoding='utf-8')))
        except (OSError,ValueError):
            return None
    if path_row:
        from expert.src.interfaces.vehicle_state import VehicleState
        from expert.src.interfaces.geometry import Point
        from expert.src.planning.lattice.structures import LatticePlanningResult
        points=[Point(*p) for p in path_row['path']]
        return dict(vehicle=VehicleState(*path_row['ego']),objects=[],selected_path=points,
                    result=LatticePlanningResult(points,reason='restart_expert_for_full_visualization'))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=REPO/'AIM2026/outputs/expert')
    parser.add_argument('--markers-only', action='store_true')
    parser.add_argument('--seconds', type=float, default=0.)
    args = parser.parse_args()
    import rospy
    import tf
    from visualization_msgs.msg import MarkerArray
    owned_master = viewer = None
    stopped = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    # Isolated visualization master; never joins the simulator ROS bridge.
    master_uri = 'http://127.0.0.1:11412'
    try:
        try:
            xmlrpc.client.ServerProxy(master_uri).getPid('/udp_expert_view')
        except OSError:
            owned_master = subprocess.Popen(['roscore', '-p', '11412'], stdout=subprocess.DEVNULL,
                                            stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(50):
                if stopped[0]:
                    return 0
                try:
                    xmlrpc.client.ServerProxy(master_uri).getPid('/udp_expert_view')
                    break
                except OSError:
                    time.sleep(.1)
            else:
                raise RuntimeError('Visualization ROS master failed to start')
        rospy.init_node('udp_expert_view', anonymous=True, disable_signals=True)
        display = OriginalDisplay()
        broadcaster = tf.TransformBroadcaster()
        if not args.markers_only:
            config = REPO/'AIM2026/Visualization/expert.rviz'
            viewer = subprocess.Popen(['rviz', '-d', str(config)], start_new_session=True)
        folder, route, coordinates = None, [], None
        started = time.monotonic()
        print('[RViz] Reading UDP Expert logs; no vehicle control. Start Expert with --visualize.', flush=True)
        while not stopped[0] and not rospy.is_shutdown():
            if args.seconds and time.monotonic()-started >= args.seconds:
                break
            if viewer is not None and viewer.poll() is not None:
                break
            candidates = list(args.runs_root.glob('*/inference.yaml'))
            recorded = [p for p in candidates if (p.parent/'paths.jsonl').is_file()
                        and (p.parent/'paths.jsonl').stat().st_size > 0]
            candidates = recorded or candidates
            newest = max(candidates, key=lambda p:p.stat().st_mtime).parent if candidates else None
            if newest is not None and newest != folder:
                try:
                    with (newest/'expert/path.csv').open(newline='', encoding='utf-8') as stream:
                        route = [(float(r['x']), float(r['y'])) for r in csv.DictReader(stream)]
                    folder = newest
                    print('[RViz] '+str(folder), flush=True)
                except (OSError, ValueError):
                    time.sleep(.1)
                    continue
            fresh_path = latest_record(folder/'paths.jsonl') if folder else None
            path_row = fresh_path or (latest_record(folder/'paths.jsonl', max_age=None) if folder else None)
            if path_row is not None and 'ego' not in path_row:
                path_row = None
            state = display_state(folder, path_row)
            stamp = rospy.Time.now()
            stale = fresh_path is None
            if state is not None:
                vehicle = state['vehicle']
                x,y,yaw = vehicle.position.x,vehicle.position.y,vehicle.yaw
                broadcaster.sendTransform((x,y,0.), (0.,0.,math.sin(yaw/2),math.cos(yaw/2)),
                                          stamp, 'gps', 'map')
            display.publish(route, state, stamp, stale)
            time.sleep(.1)
    finally:
        rospy.signal_shutdown('viewer closed')
        for child in (viewer, owned_master):
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
