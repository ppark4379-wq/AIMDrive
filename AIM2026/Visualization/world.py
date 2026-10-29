"""Read-only whole-course overlays. Geometry is map data; lamp state is VE policy."""
import csv
import json
import math
from pathlib import Path

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from AIM2026.Expert.src.config.parameters import sections, lattice
from AIM2026.Expert.src.planning.highway.common.lane_map import HighwayLaneMap

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT/'MoraiDrive/scenario_runner/data/R_KR_PR_K-city_2025'


def marker(ns, identity, kind, color, width=.15):
    m = Marker()
    m.header.frame_id = 'map'
    m.ns, m.id, m.type, m.action = ns, identity, kind, Marker.ADD
    m.pose.orientation.w = 1.
    m.scale.x = width
    m.color.r, m.color.g, m.color.b, m.color.a = color
    return m


def line(ns, identity, points, color, width=.15, z=.02):
    m = marker(ns, identity, Marker.LINE_STRIP, color, width)
    m.points = [Point(x=float(p[0]), y=float(p[1]), z=z) for p in points]
    return m


class WorldOverlay:
    def __init__(self, renderer, recorded_left=False):
        self.renderer = renderer
        self.signals = []
        self.static = MarkerArray()
        self._add_pedestrian_zones()
        links = json.loads((MAP/'link_set.json').read_text(encoding='utf-8-sig'))
        highway = HighwayLaneMap(str(MAP)).links
        for i, row in enumerate(links):
            if len(row.get('points', [])) < 2:
                continue
            if row['idx'] in highway:
                self.static.markers.append(line('highway_map_lanes', i, row['points'], (.12,.65,.86,.8), .18, .04))
            if row['idx'] == lattice.ROUNDABOUT_ENTRY_CIRCULATING_LINK_ID:
                self.static.markers.append(line('roundabout_circulating_lane', i, row['points'], (.4,1.,.55,.9), .3, .08))
        # Load only recordings actually selected by the static planner manifest.
        folder = ROOT/'AIM2026/Expert/data/recorded_paths/static_obstacle'
        self.review_points = []
        if recorded_left:
            colors = [(1.,.25,.25,1.), (.2,1.,.3,1.), (.2,.65,1.,1.),
                      (1.,.8,.15,1.), (1.,.3,1.,1.)]
            files = sorted((folder/'recordings').glob('left_static_obstacle_*.csv'))
            if not files:
                raise ValueError('No left_static_obstacle recordings found')
            for i, path in enumerate(files):
                with path.open(newline='', encoding='utf-8-sig') as stream:
                    points = [(float(r['x']), float(r['y'])) for r in csv.DictReader(stream)]
                if len(points) < 2 or not all(math.isfinite(v) for p in points for v in p):
                    raise ValueError('Invalid recording: ' + str(path))
                self.review_points.extend(points)
                color = colors[i % len(colors)]
                ns = path.stem
                self.static.markers.append(line(ns, 0, points, color, .25, .3+i*.03))
                for j, (tag, point) in enumerate((('START', points[0]), ('END', points[-1]))):
                    label = marker(ns, j+1, Marker.TEXT_VIEW_FACING, color)
                    label.pose.position = Point(x=point[0]+i*2., y=point[1]+j*2., z=1.+i*.6)
                    label.scale.z = .7
                    label.text = f'{i+1:02d} {tag}'
                    self.static.markers.append(label)
                print(f'[Recorded paths] {path.name}: {len(points)} points', flush=True)
        for i, name in enumerate(json.loads((folder/'routes.json').read_text())['routes']):
            with (folder/'recordings'/name).open(newline='', encoding='utf-8-sig') as f:
                points = [(float(r['x']), float(r['y'])) for r in csv.DictReader(f)]
            left = name.startswith('left_static_obstacle_')
            self.static.markers.append(line(
                'static_left_reference_recordings' if left else 'static_reference_recordings',
                i, points, (.2,.75,1.,.45) if left else (.75,.4,.85,.3), .08))
        band = renderer._segment_box_points(sections.STATIC_OBSTACLE_ZONE_START,
            sections.STATIC_OBSTACLE_ZONE_END, sections.STATIC_OBSTACLE_ZONE_HALF_WIDTH, z=.06)
        self.static.markers.append(line('static_planning_zone', 0, [(p.x,p.y) for p in band], (.95,.55,.2,.9), .23))
        band = renderer._segment_box_points(lattice.ROUNDABOUT_ENTRY_APPROACH_START,
            lattice.ROUNDABOUT_ENTRY_APPROACH_END, lattice.ROUNDABOUT_ENTRY_APPROACH_HALF_WIDTH, z=.06)
        self.static.markers.append(line('roundabout_approach', 0, [(p.x,p.y) for p in band], (.2,.85,.85,.75), .22))
        # Saved map stop nodes locate signals. Unknown lamps always remain gray.
        nodes = json.loads((MAP/'node_set.json').read_text(encoding='utf-8-sig'))
        for row in nodes:
            if row.get('on_stop_line') and row.get('traffic_light_id'):
                self.signals.append((str(row['traffic_light_id']), row['point']))
        for i, (text, point, color) in enumerate((
            ('ROUNDABOUT APPROACH / ENTRY', lattice.ROUNDABOUT_ENTRY_APPROACH_START, (.2,.85,.85,1.)),
            ('STATIC AVOIDANCE\nReferences: right purple / left cyan', sections.STATIC_OBSTACLE_ZONE_START, (.95,.55,.2,1.)),
        )):
            label = marker('course_labels', i, Marker.TEXT_VIEW_FACING, color)
            label.pose.position = Point(x=point[0]-12.,y=point[1],z=2.)
            label.scale.z = 1.1
            label.text = text
            self.static.markers.append(label)

    def _add_pedestrian_zones(self):
        """Display the configured object-position ACC zone, not a stop line."""
        config = json.loads((ROOT/'AIM2026/Expert/src/config/config.json').read_text(encoding='utf-8'))
        zones = config['planning']['velocity_profile']['speed_limit_zones']
        color = (1., .3, .65, .95)
        for i, zone in enumerate(zones):
            if 'acc_time_gap' not in zone:
                continue
            start, end = zone['start_point'], zone['end_point']
            half_width = zone.get('acc_lateral_threshold', 10.)
            band = self.renderer._segment_box_points(start, end, half_width, z=.12)
            self.static.markers.append(line('pedestrian_acc_zone', i,
                [(p.x, p.y) for p in band], color, .25, .12))
            dx, dy = end[0]-start[0], end[1]-start[1]
            length = math.hypot(dx, dy)
            nx, ny = -dy/length, dx/length
            for j, (name, point) in enumerate((('START', start), ('END', end))):
                self.static.markers.append(line('pedestrian_zone_boundary', i*2+j,
                    [(point[0]-nx*half_width, point[1]-ny*half_width),
                     (point[0]+nx*half_width, point[1]+ny*half_width)], color, .4, .15))
                label = marker('pedestrian_zone_labels', i*2+j, Marker.TEXT_VIEW_FACING, color)
                label.pose.position = Point(x=point[0]+12., y=point[1], z=2.)
                label.scale.z = .85
                label.text = ('DYNAMIC CROSSING {} ({:.2f}, {:.2f})\n'
                              '{} | ACC gap {:g}s / half-width {:g}m\n'
                              '{:g} km/h; route buffers +{:g}m / +{:g}m\n'
                              'Object-position ACC zone; predictive yield also runs outside').format(
                    name, point[0], point[1], zone['name'], zone['acc_time_gap'], half_width,
                    zone['max_velocity'], zone.get('entry_buffer_m', 0.), zone.get('exit_buffer_m', 0.))
                self.static.markers.append(label)

    def dynamic(self, state, stamp, stale):
        result = MarkerArray()
        if state is None:
            signal = {}
        else:
            ego = state['vehicle']
            # Same VE output ROI as the October 14 display: x forward, y left.
            c, s = math.cos(ego.yaw), math.sin(ego.yaw)
            corners = [(30.,25.), (30.,-25.), (-20.,-25.), (-20.,25.), (30.,25.)]
            points = [(ego.position.x+c*x-s*y, ego.position.y+s*x+c*y) for x,y in corners]
            result.markers.append(line('ve_roi_world', 0, points, (0.,.9,1.,.95), .18, .15))
            label = marker('roi_description', 0, Marker.TEXT_VIEW_FACING, (0.,.9,1.,.95))
            label.pose.position = Point(x=points[0][0],y=points[0][1],z=1.3)
            label.scale.z = .65
            label.text = ('STALE / LAST SNAPSHOT\n' if stale else '') + 'VE OUTPUT ROI: +30 / -20 m, sides 25 m'
            result.markers.append(label)
            signal = state.get('signal') or {}
        for i, (identity, point) in enumerate(self.signals):
            active = not stale and bool(signal.get('bound')) and signal.get('id') == identity
            code = signal.get('code') if active else None
            color = {1:(1.,.15,.1,1.), 4:(1.,.8,.1,1.), 16:(.1,1.,.25,1.),
                     33:(.1,1.,.25,1.)}.get(code, (.5,.55,.6,.65))
            m = marker('signal_stop_nodes', i, Marker.SPHERE, color)
            m.pose.position = Point(x=float(point[0]), y=float(point[1]), z=1.)
            m.scale.x = m.scale.y = m.scale.z = 1.6 if active else 1.
            result.markers.append(m)
            if active:
                label = marker('signal_decision', i, Marker.TEXT_VIEW_FACING, color)
                label.pose.position = Point(x=float(point[0]),y=float(point[1]),z=3.)
                label.scale.z = .9
                label.text = 'VE / POLICY {}  {}\n{}'.format(identity, code or 'UNKNOWN', signal.get('reason',''))
                result.markers.append(label)
        for m in result.markers:
            m.header.stamp = stamp
        return result
