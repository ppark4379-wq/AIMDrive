"""October 14 VE ROI renderer fed by the current UDP Expert snapshot.

Only display data is adapted here; no sensor subscriptions or vehicle commands.
"""
from types import SimpleNamespace as NS
import math

from AIM2026.Visualization.rviz import OriginalDisplay
from MoraiDrive.scenario_runner.legacy_expert.ve_roi_viewer import build_markers, world_to_base


class RoiDisplay(OriginalDisplay):
    def __init__(self):
        import rospy
        from visualization_msgs.msg import MarkerArray
        from AIM2026.Expert.src.visualization.rviz_visualizer import LatticeRvizVisualizer
        self.publisher = rospy.Publisher('/ve/roi_markers', MarkerArray, queue_size=1)
        # Use the existing candidate colors without creating unused publishers.
        self.renderer = object.__new__(LatticeRvizVisualizer)
        self._marker_ids = {}

    def markers(self, state, stamp, stale):
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker
        vehicle = state['vehicle'] if state else None
        ego = dict(x=vehicle.position.x, y=vehicle.position.y, yaw=vehicle.yaw,
                   speed_kmh=vehicle.velocity * 3.6, received=0., stamp_ns=0) if vehicle else None
        boxes = []
        target_marker_keys = set()
        for obj in state['objects'] if state else []:
            boxes.append(NS(position=obj.position, heading=obj.heading_deg,
                            size=NS(x=obj.size_x, y=obj.size_y),
                            velocity=NS(x=obj.velocity * 3.6), name=getattr(obj, 'name', ''),
                            unique_id=getattr(obj, 'unique_id', None)))
        message = NS(header=NS(frame_id='map'), npc_list=boxes, obstacle_list=[], pedestrian_list=[])
        # These are display snapshots, not ROS sensor callbacks. Suppress the
        # old transport-rate counters instead of inventing inference timings.
        observations = NS(objects=message if state else None, received=0. if state else None,
                          key=None, trails={}, matched_times=[], births=[], misses=[], duplicates=0)
        output = build_markers(observations, ego, None, 0., stamp, show_trails=False)
        for marker in output.markers:
            if marker.ns == 'status':
                if state is None:
                    marker.text = 'WAITING FOR EXPERT SNAPSHOT\nStart driving with --visualize'
                else:
                    age = getattr(self, 'snapshot_age_s', None)
                    age_label = ' | age {:.1f}s'.format(age) if age is not None else ''
                    marker.text = ('STALE - LAST RECORDED FRAME (not live)' if stale else 'LIVE VE -> Expert')
                    marker.text += '{}\n{} objects | {} candidates | {} path points'.format(
                        age_label, len(boxes), len(state['result'].candidates), len(state['selected_path']))
                if stale:
                    marker.color.r, marker.color.g, marker.color.b = 1., .35, .3
            elif marker.ns == 'legend':
                marker.text = ('GREEN moving | ORANGE stationary | PURPLE uncertain\n'
                               'RED vehicle = selected merge leader | YELLOW outline = merge search range\n'
                               'CYAN rectangle = model output ROI, not guaranteed detection\n'
                               'Paths: RED selected candidate | GREEN valid | GREY rejected | BLUE executed\n'
                               'Object speed = Expert input, not raw model output\n'
                               'STALE: faded geometry is the last record, not live perception')
                marker.scale.z = .85
        if state:
            def local_point(point, z):
                x, y = world_to_base(point.x, point.y, ego)
                return Point(x=x, y=y, z=z)
            # Preserve actual candidate geometry and colors, then express it
            # in the same current-ego FLU frame as the old ROI/object renderer.
            candidates = self.renderer._build_path_markers(state['result'], vehicle, stamp)
            for marker in candidates.markers:
                if marker.action != Marker.ADD or not (marker.ns.startswith('candidate_paths_')
                        or marker.ns in ('merge_search_range', 'merge_follow_target')):
                    continue
                marker.header.frame_id = 'base_link'
                marker.points = [local_point(p, p.z) for p in marker.points]
                if marker.ns == 'merge_follow_target':
                    marker.pose.position = local_point(marker.pose.position, 1.)
                    marker.pose.position.y -= 5.
                output.markers.append(marker)
            selected = Marker()
            selected.header.frame_id = 'base_link'; selected.header.stamp = stamp
            selected.ns = 'selected_path'; selected.id = 0
            selected.type = Marker.LINE_STRIP; selected.action = Marker.ADD
            selected.pose.orientation.w = 1.; selected.scale.x = .16
            selected.color.b = 1.; selected.color.g = .4; selected.color.a = 1.
            selected.points = [local_point(p, .12) for p in state['selected_path']]
            output.markers.append(selected)
            # The legacy renderer labels unknown speeds as numeric. Keep their
            # planner bound distinct from a measured/model velocity in this view.
            unknown_ids = {getattr(o, 'unique_id', None) for o in state['objects']
                           if getattr(o, 'speed_unknown', False)}
            lead_id = state['result'].debug_details.get('entry_merge', {}).get('lead_id')
            for marker in output.markers:
                if (marker.ns in ('current_bbox', 'label_leader', 'object_speed')
                        and lead_id is not None and 0 <= marker.id < len(boxes)
                        and boxes[marker.id].unique_id == lead_id):
                    target_marker_keys.add((marker.ns, marker.id))
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1., 0., 0., 1.
                    if marker.ns == 'current_bbox':
                        marker.scale.x = .23
                    if marker.ns == 'object_speed':
                        marker.text = ('LAST RECORDED TARGET\n' if stale else 'MERGE FOLLOW TARGET\n') + marker.text
                if marker.ns == 'object_speed' and boxes[marker.id].unique_id in unknown_ids:
                    lines = marker.text.splitlines()
                    lines = ['speed UNKNOWN (planner bound)' if 'km/h' in line else line for line in lines]
                    marker.text = '\n'.join(lines)
            # A thick outline alone is easy to miss beneath candidate paths.
            # Fill the actual selected detection, not a guessed neighboring car.
            for index, box in enumerate(boxes):
                if lead_id is None or box.unique_id != lead_id:
                    continue
                if not all(math.isfinite(v) for v in (box.position.x, box.position.y,
                        box.heading, box.size.x, box.size.y)) or min(box.size.x, box.size.y) <= 0:
                    continue
                fill = Marker()
                fill.header.frame_id = 'base_link'; fill.header.stamp = stamp
                fill.ns = 'merge_target_vehicle'; fill.id = index
                fill.type = Marker.CUBE; fill.action = Marker.ADD
                fill.pose.position = local_point(box.position, .45)
                angle = math.radians(box.heading)-vehicle.yaw
                fill.pose.orientation.z = math.sin(angle/2.)
                fill.pose.orientation.w = math.cos(angle/2.)
                fill.scale.x = box.size.x; fill.scale.y = box.size.y; fill.scale.z = .65
                fill.color.r = 1.; fill.color.a = .8
                output.markers.append(fill)
                target_marker_keys.add((fill.ns, fill.id))
        if stale:
            # Never show an old detection as live, but retain the explicitly
            # labelled last frame so a pause/read gap does not blank the view.
            dynamic = {'current_bbox', 'label_leader', 'object_speed', 'selected_path',
                       'merge_search_range', 'merge_follow_target', 'merge_target_vehicle'}
            for marker in output.markers:
                if (marker.ns, marker.id) in target_marker_keys:
                    # Preserve the recorded target identity while the explicit
                    # STALE status and faded alpha distinguish it from live data.
                    marker.color.r = .8; marker.color.g = marker.color.b = 0.
                    marker.color.a = .35
                elif marker.ns in dynamic or marker.ns.startswith('candidate_paths_'):
                    marker.color.r = marker.color.g = marker.color.b = .65
                    marker.color.a = .45
                elif marker.ns == 'ego_label' and state:
                    marker.text = 'LAST RECORDED ' + marker.text
        return output

    def publish(self, route, state, stamp, stale):
        markers = self.markers(state, stamp, stale)
        self.publisher.publish(self.update_markers('roi', markers, stamp))
