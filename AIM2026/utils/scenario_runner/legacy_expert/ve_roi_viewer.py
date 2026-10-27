#!/usr/bin/env python3
"""Display the exact planner object topic in current ego FLU coordinates."""

# Resolve this relocated package before importing its shared dependencies.
import sys as _runner_sys
from pathlib import Path as _RunnerPath
for _runner_ancestor in _RunnerPath(__file__).resolve().parents:
    if (_runner_ancestor / "project_paths.py").is_file():
        _runner_sys.path.insert(0, str(_runner_ancestor.parent))
        break
from scenario_runner.project_paths import project_path, activate_project_paths
activate_project_paths()

import argparse
from collections import deque
import json
import math
from pathlib import Path
import signal
import subprocess
import runpy
import threading
import time

CONFIG = project_path('AIM2026/Expert/ros/rviz/ve_roi_base_link.rviz')
MARKER_TOPIC = '/ve/roi_markers'
MAX_AGE = .35
_ego_parameters = runpy.run_path(str(project_path('AIM2026/Expert/src/config/parameters/highway.py')))
EGO_LENGTH, EGO_WIDTH = _ego_parameters['VEHICLE_LENGTH'], _ego_parameters['VEHICLE_WIDTH']


def world_to_base(x, y, ego):
    dx, dy = x - ego['x'], y - ego['y']
    c, s = math.cos(ego['yaw']), math.sin(ego['yaw'])
    return c*dx + s*dy, -s*dx + c*dy


def footprint(x, y, yaw, length, width):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x+c*a*length-s*b*width, y+s*a*length+c*b*width)
            for a,b in ((.5,.5),(.5,-.5),(-.5,-.5),(-.5,.5),(.5,.5))]


class Observations:
    """Unique source updates, without interpolation or filling missed detections."""
    def __init__(self):
        self.objects, self.received, self.key = None, None, None
        self.times, self.trails = deque(maxlen=100), {}
        self.births, self.misses = deque(), deque()
        self.last_ids = set()
        self.duplicates = 0
        self.generation = None
        self.matched_times = deque(maxlen=50)
        self.last_matched_key = None

    def match_health(self, value):
        key, sent = value.get('source_stamp_ns'), value.get('publish_perf_ns')
        if key == self.key and sent and key != self.last_matched_key:
            self.matched_times.append(int(sent))
            self.last_matched_key = key

    def reset(self, generation):
        if generation != self.generation:
            self.__init__()
            self.generation = generation

    def update(self, objects, now):
        stamp = objects.header.stamp
        key = int(stamp.secs)*10**9 + int(stamp.nsecs)
        if key == self.key:
            self.duplicates += 1
            return False
        if self.key is not None and key < self.key:
            generation = self.generation
            self.__init__()
            self.generation = generation
        self.objects, self.received, self.key = objects, now, key
        self.times.append(now)
        boxes = list(objects.npc_list) + list(objects.obstacle_list) + list(objects.pedestrian_list)
        ids = {obj.unique_id for obj in boxes}
        self.births.extend([now]*len(ids-self.last_ids))
        self.misses.extend([now]*len(self.last_ids-ids))
        self.last_ids = ids
        for obj in boxes:
            self.trails.setdefault(obj.unique_id, deque(maxlen=32)).append(
                (now, float(obj.position.x), float(obj.position.y)))
        for identity in list(self.trails):
            while self.trails[identity] and now-self.trails[identity][0][0] > 1:
                self.trails[identity].popleft()
            if not self.trails[identity]:
                del self.trails[identity]
        for events in (self.births, self.misses):
            while events and now-events[0] > 5:
                events.popleft()
        return True


class MarkerDelta:
    """Reuse RViz objects instead of destroying fonts/geometry every frame."""
    STATIC = {'roi','roi_label','axes','axes_label','ego_bbox','ego_fill','ego_origin'}

    def __init__(self):
        self.previous = {}

    def update(self, snapshot, now):
        from visualization_msgs.msg import Marker, MarkerArray
        output, present = MarkerArray(), set()
        for marker in snapshot.markers:
            if marker.action == Marker.DELETEALL:
                continue
            key = marker.ns, marker.id
            present.add(key)
            signature = (marker.type,marker.text,str(marker.pose),str(marker.scale),str(marker.color),
                         tuple((p.x,p.y,p.z) for p in marker.points))
            previous = self.previous.get(key)
            static = marker.ns in self.STATIC
            if static:
                marker.lifetime.secs,marker.lifetime.nsecs = 0,0
            if previous is None or signature != previous[0] or now-previous[1] >= (2. if static else .15):
                output.markers.append(marker)
                self.previous[key] = (signature,now)
        for ns,identity in set(self.previous)-present:
            deletion = Marker()
            deletion.header.frame_id = 'base_link'
            deletion.ns,deletion.id,deletion.action = ns,identity,Marker.DELETE
            output.markers.append(deletion)
            del self.previous[(ns,identity)]
        return output


def build_markers(observations, ego, diagnostics, now, stamp, show_trails=True):
    import rospy
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker, MarkerArray
    output = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    output.markers.append(clear)

    def marker(ns, index, kind, color):
        m = Marker()
        m.header.frame_id, m.header.stamp = 'base_link', stamp
        m.ns, m.id, m.type, m.action = ns, index, kind, Marker.ADD
        m.pose.orientation.w = 1.
        m.color.r, m.color.g, m.color.b, m.color.a = color
        # Never leave frozen boxes when the viewer itself is killed.
        m.lifetime = rospy.Duration(.3)
        output.markers.append(m)
        return m

    def line(ns, index, points, color, width=.08, z=.1):
        m = marker(ns, index, Marker.LINE_STRIP, color)
        m.scale.x = width
        m.points = [Point(x=x,y=y,z=z) for x,y in points]
        return m

    def text(ns, index, x, y, value, color=(1.,1.,1.,1.), size=1.1):
        m = marker(ns, index, Marker.TEXT_VIEW_FACING, color)
        m.pose.position = Point(x=x,y=y,z=1.)
        m.scale.z, m.text = size, value
        return m

    cyan = (.15,.9,1.,1.)
    line('roi', 0, [(-20,-25),(30,-25),(30,25),(-20,25),(-20,-25)], cyan, .18)
    text('roi_label', 0, 31.5, 0, 'FRONT +30 m | object output ROI', cyan, 1.2)
    text('roi_label', 1, -21.5, 0, 'REAR -20 m | 50 x 50 m', cyan, 1.2)
    text('roi_label', 2, 6, 27, 'LEFT +25 m', cyan)
    text('roi_label', 3, 6, -27, 'RIGHT -25 m', cyan)
    line('axes', 0, [(0,0),(6,0),(5,.5),(6,0),(5,-.5)], (1.,.3,.2,1.), .12)
    line('axes', 1, [(0,0),(0,5),(.5,4),(0,5),(-.5,4)], (.3,1.,.3,1.), .12)
    text('axes_label', 0, 7.3,0,'+X forward', (1.,.5,.3,1.), .8)
    text('axes_label', 1, 0,6,'+Y left', (.3,1.,.3,1.), .8)
    # Match the existing planner footprint: reference point at the back edge,
    # forward 0..length, lateral +/-width/2. No collision margin or guessed height.
    ego_points = footprint(EGO_LENGTH/2,0,0,EGO_LENGTH,EGO_WIDTH)
    line('ego_bbox',0,ego_points,(1.,.9,.1,1.),.16,.25)
    fill = marker('ego_fill',0,Marker.TRIANGLE_LIST,(1.,.85,.1,.3))
    fill.scale.x = fill.scale.y = fill.scale.z = 1.
    fill.points = [Point(x=ego_points[i][0],y=ego_points[i][1],z=.04) for i in (0,1,2,0,2,3)]
    ego_marker = marker('ego_origin',0,Marker.SPHERE,(1.,.9,.1,1.))
    ego_marker.scale.x = ego_marker.scale.y = ego_marker.scale.z = .6
    ego_speed = '{:.1f} km/h'.format(ego.get('speed_kmh',0)) if ego is not None else 'speed --'
    text('ego_label',0,-3,0,'EGO planner bbox {:.3f} x {:.1f} m\n{}'.format(
        EGO_LENGTH,EGO_WIDTH,ego_speed),(1.,.9,.1,1.),1.)

    age = now-observations.received if observations.received is not None else math.inf
    objects = observations.objects
    frame_ok = objects is not None and objects.header.frame_id.lstrip('/') == 'map'
    ego_ok = ego is not None and 0 <= now-ego.get('received',-math.inf) <= .35
    diag = diagnostics or {}
    matched = diag.get('source_stamp_ns') == observations.key and diag.get('ready') is True
    input_age = diag.get('metrics',{}).get('input_receipt_age_ms',0) / 1000. + age if matched else age
    # Local receipt expiry + upstream input age. ROS/simulator clocks are separate.
    fresh = frame_ok and ego_ok and 0 <= age <= MAX_AGE and input_age <= MAX_AGE
    boxes = list(objects.npc_list)+list(objects.obstacle_list)+list(objects.pedestrian_list) if fresh else []
    if show_trails and ego_ok and fresh:
        for index, (_, history) in enumerate(observations.trails.items()):
            points = [world_to_base(x,y,ego) for t,x,y in history if now-t <= .8]
            if len(points) > 1:
                line('history_only',index,points,(.55,.55,.55,.65),.045,.06)
    for index,obj in enumerate(boxes):
        values = (obj.position.x,obj.position.y,obj.heading,obj.size.x,obj.size.y,obj.velocity.x)
        if not all(math.isfinite(v) for v in values) or min(obj.size.x,obj.size.y) <= 0:
            continue
        x,y = world_to_base(obj.position.x,obj.position.y,ego)
        yaw = math.radians(obj.heading)-ego['yaw']
        state = obj.name.split(':')[-1] if obj.name.startswith('ve_tracked_unknown:') else 'unknown'
        color = {'moving':(.2,1.,.5,1.),'stationary':(1.,.65,.15,1.)}.get(state,(.9,.4,1.,1.))
        line('current_bbox',index,footprint(x,y,yaw,obj.size.x,obj.size.y),color,.09,.2)
        line('label_leader',index,[(x,y+obj.size.y/2),(x,y+4)],color,.035,.1)
        text('object_speed',index,x,y+10,
             'ID {} {}\n{:.1f} km/h ({:.1f} m/s)\nx={:+.1f} y={:+.1f} m'.format(
                 obj.unique_id,state,obj.velocity.x,obj.velocity.x/3.6,x,y),color,1.15)

    if objects is None:
        state = 'WAITING FOR /ve/Object_topic'
    elif not frame_ok:
        state = 'UNEXPECTED OBJECT FRAME: '+objects.header.frame_id
    elif not ego_ok:
        state = 'EGO MISSING/STALE - boxes hidden'
    elif not fresh:
        state = 'STALE - boxes hidden (not an empty road)'
    else:
        state = 'LIVE: {} current objects'.format(len(boxes))
    metrics = diag.get('metrics',{})
    # Physical update rate comes from the Windows inference clock, not WSL monotonic.
    inference_hz = metrics.get('new_hz',0.) if fresh else 0.
    recent = observations.matched_times
    callback_hz = (len(recent)-1)*1e9/(recent[-1]-recent[0]) if fresh and len(recent)>1 else 0.
    source_lag = max(0.,(ego['stamp_ns']-observations.key)/1e6) if ego_ok and observations.key else None
    age_label = '{:.0f}'.format(age*1000) if math.isfinite(age) else '--'
    lag_label = '{:.0f}'.format(source_lag) if source_lag is not None else '--'
    text('status',0,38,0,
         '{}\nNEW {:.2f} Hz | ROS matched {:.2f} Hz | update age {} ms\n'
         'GPU {:.0f} ms | pipeline {:.0f} ms | input->output {:.0f} ms | sim lag {} ms'.format(
             state,inference_hz,callback_hz,age_label,metrics.get('gpu_ms',0),
             metrics.get('pipeline_ms',0),metrics.get('input_receipt_age_ms',0),lag_label),
         (.3,1.,.6,1.) if fresh else (1.,.35,.3,1.),1.35)
    text('legend',0,-26,0,
         'GREEN moving | ORANGE stationary | PURPLE uncertain | GREY history (0.8 s)\n'
         '2D footprints: height unavailable | speed = planner input, not GT\n'
         'ID births {} / missed {} (5 s) | duplicate messages {} | no smoothing / extrapolation'.format(
             sum(now-t <= 5 for t in observations.births),sum(now-t <= 5 for t in observations.misses),observations.duplicates),
         (.8,.85,.9,1.),1.05)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--markers-only', action='store_true')
    parser.add_argument('--seconds', type=float, default=0)
    parser.add_argument('--no-trails', action='store_true')
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args()
    import rospy
    from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList
    from std_msgs.msg import String
    from visualization_msgs.msg import MarkerArray
    from geometry_msgs.msg import TransformStamped
    from tf2_msgs.msg import TFMessage
    # Load this geometry module directly: the training package __init__ imports
    # torch, which the ROS-only Python environment intentionally does not need.
    import importlib.util
    from AIM2026.Localization.map_geometry import MapGeometry
    # Same projection and origin as the E3 training contract.
    geometry = MapGeometry('EPSG:32652', [302595.,4124145.,0.])
    rospy.init_node('ve_roi_viewer', anonymous=False, disable_signals=True)
    stopped, lock = threading.Event(), threading.RLock()
    observations = Observations()
    marker_delta = MarkerDelta()
    state = dict(ego=None, diagnostics=None, health_by_stamp={})
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,lambda *_: stopped.set())

    def objects(message):
        with lock:
            observations.update(message,time.time())
            state['diagnostics'] = state['health_by_stamp'].get(observations.key)
            if state['diagnostics']:
                observations.match_health(state['diagnostics'])

    def health(message):
        try:
            value = json.loads(message.data)
        except ValueError:
            return
        with lock:
            if value.get('ready'):
                generation = value.get('generation')
                if observations.generation is not None and generation != observations.generation:
                    observations.reset(generation)
                    state['health_by_stamp'].clear()
                observations.generation = generation
                key = value.get('source_stamp_ns')
                state['health_by_stamp'][key] = value
                while len(state['health_by_stamp']) > 30:
                    del state['health_by_stamp'][next(iter(state['health_by_stamp']))]
                if key == observations.key:
                    state['diagnostics'] = value
                    observations.match_health(value)

    def ego(message):
        now = time.time()
        position = (message.position.x,message.position.y,message.position.z)
        yaw = math.radians(message.heading+geometry.convergence_degrees(position))
        with lock:
            state['ego'] = dict(x=position[0],y=position[1],yaw=yaw,received=now,
                                speed_kmh=math.hypot(message.velocity.x,message.velocity.y)*3.6,
                                stamp_ns=message.header.stamp.to_nsec())

    subscriptions = [rospy.Subscriber('/ve/Object_topic',ObjectStatusList,objects,queue_size=1,tcp_nodelay=True),
                     rospy.Subscriber('/ve/diagnostics',String,health,queue_size=5,tcp_nodelay=True),
                     rospy.Subscriber('/Ego_topic',EgoVehicleStatus,ego,queue_size=1,tcp_nodelay=True)]
    publisher = rospy.Publisher(MARKER_TOPIC,MarkerArray,queue_size=1)
    # Private display-only TF tree, isolated from the vehicle's /tf tree.
    frame_pub = rospy.Publisher('/ve/roi_tf_static',TFMessage,queue_size=1,latch=True)
    reference = TransformStamped()
    reference.header.frame_id, reference.child_frame_id = 've_display_reference','base_link'
    reference.transform.rotation.w = 1.
    frame_pub.publish(TFMessage(transforms=[reference]))
    viewer = None
    print('[ROI] base_link FLU. Exact /ve/Object_topic. No controls. Close RViz or Ctrl+C to stop viewer.',flush=True)
    try:
        if not args.markers_only:
            viewer = subprocess.Popen(['rviz','-d',str(CONFIG),
                                       '/tf:=/ve/roi_tf','/tf_static:=/ve/roi_tf_static'])
        start = time.time()
        while not stopped.is_set() and not rospy.is_shutdown():
            if args.seconds and time.time()-start >= args.seconds:
                break
            if args.stop_file and args.stop_file.exists():
                break
            if viewer is not None and viewer.poll() is not None:
                break
            with lock:
                markers = build_markers(observations,state['ego'],state['diagnostics'],
                                        time.time(),rospy.Time.now(),not args.no_trails)
            publisher.publish(marker_delta.update(markers,time.time()))
            stopped.wait(.05)
    finally:
        try:
            from visualization_msgs.msg import Marker
            clear = Marker()
            clear.action = Marker.DELETEALL
            publisher.publish(MarkerArray(markers=[clear]))
        except rospy.ROSException:
            pass
        for sub in subscriptions:
            sub.unregister()
        if viewer is not None and viewer.poll() is None:
            viewer.terminate()
            try:
                viewer.wait(timeout=3)
            except subprocess.TimeoutExpired:
                viewer.kill()
        rospy.signal_shutdown('ROI viewer stopped')


if __name__ == '__main__':
    main()
