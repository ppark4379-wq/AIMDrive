"""Green bus/truck supplement outlines added to the original Expert display."""
import math

from AIM2026.Visualization.rviz import OriginalDisplay


def footprints(state, stale, local=False):
    """World or ego-frame BEV outlines. Never reuse stale or VE-only boxes."""
    if not state or stale or not state.get('ready'):
        return []
    ego = state.get('ego')
    if local and not ego:
        return []
    rows = []
    for index, box in enumerate(state.get('supplement_boxes', [])):
        x, y = box['center_world']
        angle = math.radians(box['heading_deg'])
        if local:
            c, s = math.cos(ego['yaw_grid_rad']), math.sin(ego['yaw_grid_rad'])
            dx, dy = x-ego['x'], y-ego['y']
            x, y = c*dx+s*dy, -s*dx+c*dy
            angle -= ego['yaw_grid_rad']
        c, s = math.cos(angle), math.sin(angle)
        corners = []
        for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1)):
            u, v = a*box['length_m']/2, b*box['width_m']/2
            corners.append((x+c*u-s*v, y+s*u+c*v))
        rows.append(dict(track_id=index, center=(x, y), corners=corners,
                         speed=box.get('speed_mps'), confirmed=True,
                         hits=box.get('consecutive_hits'), points=box.get('lidar_points')))
    return rows


class LidarDisplay(OriginalDisplay):
    def __init__(self, local=False):
        import rospy
        from visualization_msgs.msg import MarkerArray
        self.publisher = rospy.Publisher('/perception/final_boxes',
                                         MarkerArray, queue_size=1)
        self._marker_ids = {}
        self.local = local

    def publish(self, state, stamp, stale):
        import rospy
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker, MarkerArray
        output = MarkerArray()
        for row in footprints(state, stale, self.local):
            for namespace, kind in (('lidar_box', Marker.LINE_STRIP),
                                    ('lidar_label', Marker.TEXT_VIEW_FACING)):
                item = Marker()
                item.header.frame_id = 'base_link' if self.local else 'map'
                item.header.stamp = stamp
                item.ns, item.id, item.type = ('final_' + namespace), row['track_id'], kind
                item.pose.orientation.w = 1.
                item.color.r, item.color.g, item.color.b, item.color.a = .2, 1., .25, 1.
                if namespace == 'lidar_box':
                    item.scale.x = .22
                    item.points = [Point(x=x, y=y, z=.45) for x, y in row['corners']]
                else:
                    item.scale.z = .7
                    item.pose.position = Point(x=row['center'][0], y=row['center'][1], z=1.8)
                    item.text = 'Supplement #%d%s' % (row['track_id'],
                                '' if row['confirmed'] else ' candidate')
                    if row['speed'] is not None:
                        item.text += ' | %.2f m/s' % row['speed']
                    else:
                        item.text += ' | speed unknown'
                    if row['points'] is not None:
                        item.text += ' | %d pts' % row['points']
                output.markers.append(item)
        self.publisher.publish(self.update_markers('lidar', output, stamp))
