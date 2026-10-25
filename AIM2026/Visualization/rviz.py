#!/usr/bin/env python3
"""Read UDP Expert logs for RViz. No sensor socket or vehicle control publisher."""
import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
import signal
import subprocess
import sys
import time
import xmlrpc.client

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


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
    """Publish the map-view planner topics using the retained renderer."""
    def __init__(self):
        import rospy
        from nav_msgs.msg import Path, Odometry
        from AIM2026.Expert.src.visualization.rviz_visualizer import LatticeRvizVisualizer
        self.renderer = LatticeRvizVisualizer()
        self.global_pub = rospy.Publisher('/global_path', Path, queue_size=1, latch=True)
        self.local_pub = rospy.Publisher('/local_path', Path, queue_size=1)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=1)
        self._marker_ids = {}
        self._route = None
        from AIM2026.Visualization.world import WorldOverlay
        from visualization_msgs.msg import MarkerArray
        self.world = WorldOverlay(self.renderer)
        self.world_pub = rospy.Publisher('/expert/map', MarkerArray, queue_size=1, latch=True)
        self.overlay_pub = rospy.Publisher('/expert/overlays', MarkerArray, queue_size=1)
        self.world_pub.publish(self.world.static)

    def update_markers(self, topic, markers, stamp):
        """Update existing geometry; explicitly remove only vanished markers.

        The log reader can stall on Windows/WSL filesystem I/O. Do not expire
        geometry between reads, or DELETEALL/recreate it on every frame.
        """
        import rospy
        from visualization_msgs.msg import Marker
        current = {}
        cache = getattr(self, '_marker_cache', {})
        old = cache.get(topic, {})
        for marker in markers.markers:
            key = (marker.ns, marker.id)
            if marker.action == Marker.ADD:
                marker.lifetime = rospy.Duration(0)
                # All map markers already carry world coordinates (ROI has a
                # fixed display frame). Stamp-only changes must not rebuild
                # thousands of unchanged vertices in RViz.
                marker.header.stamp = rospy.Time(0)
                current[key] = marker
            elif marker.action == Marker.DELETE:
                current.pop(key, None)
        updates = [marker for key, marker in current.items() if old.get(key) != marker]
        frame_id = updates[0].header.frame_id if updates else getattr(self, '_marker_frames', {}).get(topic, 'map')
        if not hasattr(self, '_marker_frames'):
            self._marker_frames = {}
        self._marker_frames[topic] = frame_id
        previous = self._marker_ids.get(topic)
        if previous is None:
            # Clear leftovers once when this publisher replaces an older one.
            clear = Marker(); clear.action = Marker.DELETEALL
            clear.header.frame_id = frame_id; clear.header.stamp = stamp
            updates.insert(0, clear)
        else:
            for namespace, marker_id in sorted(previous - current.keys()):
                removed = Marker(); removed.action = Marker.DELETE
                removed.ns = namespace; removed.id = marker_id
                removed.header.frame_id = frame_id; removed.header.stamp = stamp
                updates.append(removed)
        self._marker_ids[topic] = set(current)
        cache[topic] = current
        self._marker_cache = cache
        markers.markers = updates
        return markers

    def refresh_connections(self):
        """Send a full frame to newly connected RViz subscribers, then deltas."""
        pubs = ([self.publisher] if hasattr(self, 'publisher') else
                [self.renderer.paths_pub, self.renderer.objects_pub,
                 self.renderer.lanes_pub, self.overlay_pub,
                 self.local_pub, self.odom_pub])
        counts = tuple(pub.get_num_connections() for pub in pubs)
        previous = getattr(self, '_connections', (0,)*len(counts))
        if any(current > old for current, old in zip(counts, previous)):
            self._marker_cache = {}
            self._marker_ids = {}
            self._render_key = None
        self._connections = counts

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
        self.overlay_pub.publish(self.update_markers('world_overlays',
            self.world.dynamic(state, stamp, stale), stamp))
        if route is not self._route:
            self.global_pub.publish(self.path_message(route,stamp))
            self._route = route
        if state is None:
            return
        vehicle, result = state['vehicle'], state['result']
        paths = self.renderer._build_path_markers(result,vehicle,stamp)
        if stale:
            for marker in paths.markers:
                if marker.ns == 'lattice_status':
                    marker.text = 'STALE - LAST RECORDED STATE\n'+marker.text
        self.renderer.paths_pub.publish(self.update_markers('paths', paths, stamp))
        # The driving overview shows observed objects, not dozens of future
        # translucent boxes over each object. Prediction stays in the planner.
        import copy
        observed = copy.copy(result)
        observed.predicted_object_paths = []
        observed.object_lane_matches = []
        objects = self.renderer._build_object_markers(observed,vehicle,state['objects'],stamp)
        self.renderer.objects_pub.publish(self.update_markers('objects', objects, stamp))
        lanes = self.renderer._build_lane_markers(result,stamp)
        # WorldOverlay already publishes the highway lines once, latched.
        # Drawing the same lines twice at almost identical Z causes shimmer.
        lanes.markers = [m for m in lanes.markers if m.ns != 'highway_lane_centers']
        self.renderer.lanes_pub.publish(self.update_markers('lanes', lanes, stamp))
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
        from AIM2026.Expert.src.interfaces.vehicle_state import VehicleState
        from AIM2026.Expert.src.interfaces.geometry import Point
        from AIM2026.Expert.src.planning.lattice.structures import LatticePlanningResult
        points=[Point(*p) for p in path_row['path']]
        return dict(vehicle=VehicleState(*path_row['ego']),objects=[],selected_path=points,
                    result=LatticePlanningResult(points,reason='restart_expert_for_full_visualization'))
    return None


class SnapshotReader:
    """Keep the last complete snapshot across transient Windows/WSL read errors.

    Freshness follows that snapshot's write time, not paths.jsonl buffering or
    the success of one read attempt. An actually stale snapshot stays labelled.
    """
    def __init__(self):
        self.state = None
        self.signature = None
        self.written_at = 0.

    def read(self, folder, now=None, max_age=.75):
        now = time.time() if now is None else now
        if folder is not None:
            try:
                info = (folder/'visualization.json').stat()
                signature = (info.st_mtime_ns, info.st_size)
                if signature != self.signature:
                    state = display_state(folder, None)
                    if state is not None:
                        self.state = state
                        # Windows and WSL clocks can differ after a clock change.
                        # A newly observed write proves activity in our clock;
                        # the first old snapshot alone must still remain stale.
                        changed_after_read = self.signature is not None
                        self.signature = signature
                        self.written_at = now if changed_after_read else info.st_mtime
            except OSError:
                pass
        return self.state, self.state is None or now - self.written_at > max_age


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, default=REPO/'AIM2026/outputs/expert')
    parser.add_argument('--markers-only', action='store_true')
    views = parser.add_mutually_exclusive_group()
    views.add_argument('--roi', dest='roi', action='store_true', help='October 14 local VE ROI display')
    views.add_argument('--map', dest='roi', action='store_false', help='Whole-course map + moving VE ROI (default)')
    parser.set_defaults(roi=False)
    parser.add_argument('--seconds', type=float, default=0.)
    args = parser.parse_args()
    # Prevent two readers from alternating markers on the same ROS topics.
    # Linux flock is automatically released on exit, including crashes.
    import fcntl
    import os
    import tempfile
    viewer_lock = open(Path(tempfile.gettempdir()) / ('aim2026-rviz-{}.lock'.format(os.getuid())), 'a')
    try:
        fcntl.flock(viewer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('[RViz] Already running. Stop the existing viewer with Ctrl+C before restarting.', flush=True)
        return 1
    import rospy
    import tf
    from visualization_msgs.msg import MarkerArray
    owned_master = viewer = None
    exit_code = 0
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
        if args.roi:
            from AIM2026.Visualization.roi import RoiDisplay
            display = RoiDisplay()
        else:
            display = OriginalDisplay()
        broadcaster = tf.TransformBroadcaster()
        if not args.markers_only:
            config = REPO/'AIM2026/Visualization/expert.rviz'
            if args.roi:
                config = REPO/'AIM2026/Expert/ros/rviz/ve_roi_base_link.rviz'
            viewer = subprocess.Popen(['rviz', '-d', str(config)], start_new_session=True)
        folder, route, newest = None, [], None
        reader = SnapshotReader()
        next_scan = 0.
        next_status = 0.
        run_signatures = {}
        run_activity = {}
        started = time.monotonic()
        print('[RViz] Reading UDP Expert logs; no vehicle control. Start Expert with --visualize.', flush=True)
        print('[RViz] VIEW={} | data=VE detections -> tracker -> Expert snapshot'.format(
            'VE ROI (front 30m / rear 20m / left-right 25m)' if args.roi else 'MAP'), flush=True)
        while not stopped[0] and not rospy.is_shutdown():
            if args.seconds and time.monotonic()-started >= args.seconds:
                break
            if viewer is not None and viewer.poll() is not None:
                exit_code = viewer.returncode
                print('[RViz] Window process exited (code={}).'.format(exit_code), flush=True)
                break
            if time.monotonic() >= next_scan:
                candidates = list(args.runs_root.glob('*/inference.yaml'))
                # Follow the newest actual run even before its first snapshot.
                # Preferring any old snapshot pins the viewer to a previous run
                # while the new Expert is waiting for inference. Exclude checks
                # when real runs exist: --check writes config but no control log.
                started_runs = [p for p in candidates if (p.parent/'control.jsonl').is_file()]
                candidates = started_runs or candidates
                for candidate in candidates:
                    heartbeat = candidate.parent/'control.jsonl'
                    if heartbeat.is_file():
                        info = heartbeat.stat()
                        signature = (info.st_mtime_ns, info.st_size)
                        if candidate in run_signatures and signature != run_signatures[candidate]:
                            run_activity[candidate] = time.monotonic()
                        run_signatures[candidate] = signature
                active_runs = [p for p in candidates
                               if time.monotonic()-run_activity.get(p, -float('inf')) < 3.]
                candidates = active_runs or candidates
                # A simulator session may follow a system-clock correction.
                # Old runs dated in the future must not outrank today's run.
                current_clock_runs = [p for p in candidates if p.stat().st_mtime <= time.time() + 5.]
                candidates = current_clock_runs or candidates
                def run_order(path):
                    # Scenario run names originate in the WSL launcher, even
                    # when Windows writes their files with a different clock.
                    match = re.search(r'\d{8}-\d{6}', path.parent.name)
                    if match:
                        try:
                            return datetime.strptime(match.group(), '%Y%m%d-%H%M%S').timestamp()
                        except ValueError:
                            pass
                    return path.stat().st_mtime
                if active_runs:
                    newest = max(active_runs, key=lambda p: run_activity[p]).parent
                elif folder is not None and (folder/'inference.yaml').is_file():
                    # Keep the last observed run after it stops. WSL/Windows
                    # clock differences must not switch us back to an old run.
                    newest = folder
                else:
                    newest = max(candidates, key=run_order).parent if candidates else None
                next_scan = time.monotonic() + 1.
            if newest is not None and newest != folder:
                try:
                    with (newest/'expert/path.csv').open(newline='', encoding='utf-8') as stream:
                        route = [(float(r['x']), float(r['y'])) for r in csv.DictReader(stream)]
                    folder = newest
                    reader = SnapshotReader()
                    print('[RViz] '+str(folder), flush=True)
                except (OSError, ValueError):
                    time.sleep(.1)
                    continue
            state, stale = reader.read(folder)
            display.snapshot_age_s = max(0., time.time() - reader.written_at) if state else None
            if time.monotonic() >= next_status:
                print('[RViz] {} | objects={} candidates={} path_points={} snapshot_age_s={}'.format(
                    'WAITING' if state is None else 'STALE' if stale else 'LIVE',
                    len(state['objects']) if state else 0,
                    len(state['result'].candidates) if state else 0,
                    len(state['selected_path']) if state else 0,
                    round(display.snapshot_age_s, 2) if state else None), flush=True)
                next_status = time.monotonic() + 2.
            stamp = rospy.Time.now()
            if args.roi:
                # Display-only fixed frame; the ROI does not move with map TF.
                broadcaster.sendTransform((0.,0.,0.), (0.,0.,0.,1.), stamp,
                                          'base_link', 've_display_reference')
            elif state is not None:
                vehicle = state['vehicle']
                x,y,yaw = vehicle.position.x,vehicle.position.y,vehicle.yaw
                broadcaster.sendTransform((x,y,0.), (0.,0.,math.sin(yaw/2),math.cos(yaw/2)),
                                          stamp, 'gps', 'map')
            render_key = (folder, reader.signature, stale)
            display.refresh_connections()
            if render_key != getattr(display, '_render_key', None):
                display.publish(route, state, stamp, stale)
                display._render_key = render_key
            time.sleep(.05)
    finally:
        rospy.signal_shutdown('viewer closed')
        for child in (viewer, owned_master):
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
    return 0 if exit_code == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
