"""Read one matched live frame, save evidence, and replay detector variants."""
import json
from pathlib import Path
import sys
import time
import threading
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import String

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gps_jamming.lidar.walls import detect_walls
from gps_jamming.lidar.runtime import load_config, DEFAULT_CONFIG


def summary(result):
    return {s: [dict(range=p.get('observed_range'), seed=p.get('line_fit_range'),
                    slope=p.get('line_slope'), intercept=p.get('line_intercept'),
                    count=p.get('enclosing_point_count'), inferred=p.get('inferred', False))
                for p in result[s]] for s in ('left', 'right')}


def main():
    rospy.init_node('wall_frame_readonly_audit', anonymous=True)
    lock = threading.Lock()
    clouds, boundaries, statuses = {}, {}, {}
    def on_cloud(msg):
        with lock:
            clouds[msg.header.stamp.to_nsec()] = msg
            while len(clouds)>30: clouds.pop(next(iter(clouds)))
    def on_json(msg, kind):
        value = json.loads(msg.data)
        with lock:
            target = boundaries if kind == 'boundary' else statuses
            target[value['stamp_ns'] if kind == 'boundary' else value['frame_id']] = value
            while len(target)>30: target.pop(next(iter(target)))
    subscribers = [rospy.Subscriber('/gps_jamming_lidar/nonground', PointCloud2, on_cloud),
        rospy.Subscriber('/gps_jamming_lidar/wall_boundaries', String, on_json, 'boundary'),
        rospy.Subscriber('/gps_jamming_lidar/status', String, on_json, 'status')]
    deadline = time.monotonic()+12
    matched = None
    while time.monotonic()<deadline and matched is None:
        with lock:
            for stamp, b in boundaries.items():
                if stamp in clouds and b['frame_sequence'] in statuses:
                    matched = clouds[stamp], b, statuses[b['frame_sequence']]
                    break
        time.sleep(.03)
    if matched is None:
        raise RuntimeError('No matched live cloud/boundary/status within 12 seconds')
    for subscriber in subscribers: subscriber.unregister()
    msg, live, status = matched
    points = np.asarray(list(point_cloud2.read_points(msg, field_names=('x','y','z','intensity'),
                                                     skip_nans=True)), dtype=float).reshape(-1,4)
    plane = status['diagnostics']['ground_plane']
    config = load_config(DEFAULT_CONFIG)['walls']
    folder = ROOT/'artifacts'/'wall_audit'
    folder.mkdir(parents=True, exist_ok=True)
    identity = str(live['stamp_ns'])
    np.savez_compressed(folder/(identity+'.npz'), points=points, plane=plane)
    report = dict(frame=live['frame_sequence'], config=config, live=summary(live),
                  plane=plane, total_points=len(points), variants={})
    variants = {'current': {}, 'lateral_only': {'min_lateral': .05},
        'support_only': {'boundary_support_distance': 2.},
        'fit_only': {'distance_threshold': .7},
        'gap_only': {'max_observation_gap': 5.},
        'height_only': {'min_height': 0., 'min_height_span': .1}}
    for name, changes in variants.items():
        result, supports = detect_walls(points, plane, **dict(config, **changes))
        report['variants'][name] = summary(result)
        if name == 'current':
            report['cross_axis_support'] = {}
            for side, sign in (('left',1),('right',-1)):
                if not result[side]:
                    continue
                model = result[side][0]
                m = model['line_slope']
                # Reverse the explicit clearance for a support-band diagnostic;
                # this does not assert that every nearby return is a wall.
                b = model['line_intercept']+sign*config['clearance_m']*np.hypot(1,m)
                match = np.abs(points[:,1]-m*points[:,0]-b)/np.hypot(1,m)<=config['boundary_support_distance']
                cross = match & (sign*points[:,1]<0)
                report['cross_axis_support'][side] = dict(
                    boundary_axis_crossing_x=float(-model['line_intercept']/m) if abs(m)>1e-8 else None,
                    opposite_sign_near_line=int(cross.sum()),
                    matched_x_range=[float(points[match,0].min()),float(points[match,0].max())] if match.any() else None)
            heights = points[:,2]+(plane[0]*points[:,0]+plane[1]*points[:,1]+plane[3])/plane[2]
            report['bins'] = {}
            for side, sign in (('left',1),('right',-1)):
                rows=[]
                for start in range(-10,30,5):
                    mask=(points[:,0]>=start)&(points[:,0]<start+5)&(sign*points[:,1]>0)
                    p=points[mask]
                    rows.append(dict(x=start, count=len(p), lateral_excluded=int(np.sum(sign*p[:,1]<config['min_lateral'])),
                        height_excluded=int(np.sum(heights[mask]<config['min_height'])),
                        lateral_range=[float(p[:,1].min()),float(p[:,1].max())] if len(p) else None,
                        selected=int(np.sum((supports[side][:,0]>=start)&(supports[side][:,0]<start+5)))))
                report['bins'][side]=rows
    (folder/(identity+'.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    print('Evidence: '+str(folder/identity))


if __name__ == '__main__':
    main()
