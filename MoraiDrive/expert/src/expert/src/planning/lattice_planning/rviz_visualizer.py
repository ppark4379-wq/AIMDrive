#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RViz MarkerArray publisher for the GT-based lattice planner."""

import math

import rospy
import tf
from geometry_msgs.msg import Point as RosPoint
from visualization_msgs.msg import Marker, MarkerArray

from . import parameters
from ..highway_lane_planning import parameters as highway_parameters


class LatticeRvizVisualizer:
    """Publish candidates with the same colors used by the source planner."""

    def __init__(self):
        self.paths_pub = rospy.Publisher(
            "/lattice/paths", MarkerArray, queue_size=1
        )
        self.objects_pub = rospy.Publisher(
            "/lattice/objects", MarkerArray, queue_size=1
        )
        self.lanes_pub = rospy.Publisher(
            "/lattice/lanes", MarkerArray, queue_size=1
        )

    def publish(self, result, vehicle_state, object_info_list):
        # RViz가 연결되지 않았을 때 Marker 생성 비용을 피합니다.
        if (
            self.paths_pub.get_num_connections() == 0
            and self.objects_pub.get_num_connections() == 0
            and self.lanes_pub.get_num_connections() == 0
        ):
            return

        now = rospy.Time.now()
        self.paths_pub.publish(self._build_path_markers(result, vehicle_state, now))
        self.objects_pub.publish(
            self._build_object_markers(result, vehicle_state, object_info_list, now)
        )
        self.lanes_pub.publish(self._build_lane_markers(result, now))

    def _build_path_markers(self, result, vehicle_state, stamp):
        marker_array = MarkerArray()
        static_mode = result.mode == "static_obstacle"
        if not static_mode or not getattr(self, "_static_path_display", False):
            marker_array.markers.append(self._delete_all("map", stamp))
        self._static_path_display = static_mode
        current_ids = {('candidate_paths_{}'.format(candidate.layer), index)
                       for index, candidate in enumerate(result.candidates)}
        for namespace, marker_id in getattr(self, '_path_marker_ids', set()) - current_ids:
            marker = self._new_marker("map", stamp, namespace, marker_id, Marker.LINE_STRIP)
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        self._path_marker_ids = current_ids

        for index, candidate in enumerate(result.candidates):
            marker = self._new_marker(
                frame_id="map",
                stamp=stamp,
                namespace="candidate_paths_{}".format(candidate.layer),
                marker_id=index,
                marker_type=Marker.LINE_STRIP,
            )

            # 원본 Visualizer 규칙: 무효=회색, 선택=빨강, 유효=초록
            if not candidate.valid:
                marker.scale.x = 0.03
                self._set_color(marker, 0.5, 0.5, 0.5, 0.5)
            elif candidate.points is result.path or getattr(candidate, 'selected_preview', False):
                marker.scale.x = 0.15
                self._set_color(marker, 1.0, 0.0, 0.0, 1.0)
            else:
                marker.scale.x = 0.05
                alpha = 0.35 if candidate.layer == "long" else 0.6
                self._set_color(marker, 0.0, 1.0, 0.0, alpha)

            marker.points = [
                RosPoint(x=point.x, y=point.y, z=0.08)
                for point in candidate.points
            ]
            marker_array.markers.append(marker)

        marker_array.markers.append(
            self._build_ego_marker(vehicle_state, stamp)
        )
        marker_array.markers.append(
            self._build_status_marker(result, vehicle_state, stamp)
        )
        guard = self._new_marker('map', stamp, 'central_guard', 0, Marker.LINE_STRIP)
        guard_points = result.debug_details.get('central_guard', [])
        guard.action = Marker.ADD if guard_points else Marker.DELETE
        guard.scale.x = 0.08
        self._set_color(guard, 1.0, 0.8, 0.0, 0.9)
        guard.points = [RosPoint(x=x, y=y, z=.1) for x, y in guard_points]
        marker_array.markers.append(guard)
        if static_mode:
            for marker in marker_array.markers:
                marker.lifetime = rospy.Duration(1.0)
        return marker_array

    def _build_ego_marker(self, vehicle_state, stamp):
        marker = self._new_marker(
            frame_id="map",
            stamp=stamp,
            namespace="ego_shape",
            marker_id=0,
            marker_type=Marker.CUBE,
        )
        half_length = 0.5 * parameters.VEHICLE_LENGTH
        marker.pose.position.x = (
            vehicle_state.position.x + half_length * math.cos(vehicle_state.yaw)
        )
        marker.pose.position.y = (
            vehicle_state.position.y + half_length * math.sin(vehicle_state.yaw)
        )
        marker.pose.position.z = 0.75
        quaternion = tf.transformations.quaternion_from_euler(
            0.0, 0.0, vehicle_state.yaw
        )
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]
        marker.scale.x = parameters.VEHICLE_LENGTH
        marker.scale.y = parameters.VEHICLE_WIDTH
        marker.scale.z = 1.5
        self._set_color(marker, 1.0, 1.0, 0.0, 0.5)
        return marker

    def _build_status_marker(self, result, vehicle_state, stamp):
        marker = self._new_marker(
            frame_id="map",
            stamp=stamp,
            namespace="lattice_status",
            marker_id=0,
            marker_type=Marker.TEXT_VIEW_FACING,
        )
        marker.pose.position.x = vehicle_state.position.x
        marker.pose.position.y = vehicle_state.position.y
        marker.pose.position.z = 3.0
        marker.scale.z = 0.65

        if result.mode == "roundabout_entry":
            details = result.debug_details.get("roundabout_entry", {})
            marker.text = "ROUNDABOUT: {}  {:.1f}km/h\nsource={} front={} conflict={} TTC={:.1f}s\nhold={:.2f}m merged={}".format(
                details.get("phase", "check").upper(),
                details.get("final_target_velocity", result.acc_target_velocity) * 3.6,
                details.get("speed_source", "roundabout"), details.get("front_id"),
                details.get("blocker_unique_id"), result.ttc,
                details.get("hold_gap", 0.0), details.get("committed", False),
            )
            if result.acc_target_velocity <= 0.05:
                self._set_color(marker, 1.0, 0.15, 0.05, 1.0)
            else:
                self._set_color(marker, 0.1, 0.8, 1.0, 1.0)
        elif result.debug_details.get('jamming_dynamic'):
            details=result.debug_details['jamming_dynamic']
            conflict=details.get('collision') or {}
            person=result.debug_details.get('jamming_pedestrian',{})
            control=result.debug_details.get('static_avoidance',{})
            target=control.get('final_target_velocity',result.acc_target_velocity)
            marker.text='JAMMING: DYNAMIC {}\n{:.1f}km/h gap={}m collision={}s'.format(
                details['state'],target*3.6,
                details.get('lead_gap_m'),conflict.get('time','none'))
            if person.get('checked'):
                marker.text+='\nperson_collision={}s source={}'.format(
                    (person.get('collision') or {}).get('time','none'),control.get('speed_source','planner'))
            if details.get('tracking_error'):
                marker.text+='\nstop='+str(details['tracking_error'])
            if details.get('replanning'):
                marker.text+=' (replanning)'
            self._set_color(marker,1.0,.2,.1,1.) if (details['emergency'] or person.get('emergency')) else self._set_color(marker,.1,.8,1.,1.)
        elif result.debug_details.get("merge_safety"):
            safety = result.debug_details["merge_safety"]
            conflict = safety.get("conflict") or {}
            marker.text = "MERGE: {}  target={}\nvehicle={} collision={}s".format(
                safety["decision"],
                "AUTO" if math.isinf(result.acc_target_velocity)
                else "{:.0f}km/h".format(result.acc_target_velocity * 3.6),
                conflict.get("id", "none"), conflict.get("time", "none"),
            )
            self._set_color(marker, 1.0, 0.65, 0.0, 1.0)
        elif result.acc_override and result.mode != "static_obstacle":
            ttc_text = "INF" if math.isinf(result.ttc) else "{:.1f}s".format(result.ttc)
            acc_target = getattr(result, "acc_target_velocity", float("inf"))
            target_text = (
                "AUTO"
                if math.isinf(acc_target)
                else "{:.0f}km/h".format(acc_target * 3.6)
            )
            marker.text = "HIGHWAY: ACC  gap={:.1f}m TTC={} target={}".format(
                result.nearest_vehicle_distance, ttc_text, target_text
            )
            self._set_color(marker, 1.0, 0.2, 0.1, 1.0)
        elif result.mode == "highway_lane":
            marker.text = "{}  {} -> {}".format(
                result.highway_state,
                result.current_link_id,
                result.target_link_id,
            )
            self._set_color(marker, 0.1, 1.0, 0.2, 1.0)
        elif result.active and result.mode == "highway":
            marker.text = "HIGHWAY: LATTICE  offset={:+.1f}m".format(result.selected_offset)
            self._set_color(marker, 0.1, 1.0, 0.2, 1.0)
        elif result.mode == "static_obstacle" and result.debug_details.get("static_avoidance"):
            details = result.debug_details["static_avoidance"]
            marker.text = "STATIC: RIGHT LATTICE {} {} offset={:+.2f}m\n{:.1f}km/h source={}".format(
                details.get('selected_layer', '').upper(), details.get("phase", "check").upper(), result.selected_offset,
                details.get("final_target_velocity", result.acc_target_velocity) * 3.6,
                details.get("speed_source", "planner"),
            )
            if details.get('planner') == 'recorded_route_graph':
                section_name = 'JAMMING' if result.debug_details.get('jamming') else 'STATIC'
                marker.text = '{}: RECORDED {}\n{}\n{:.1f}km/h source={}'.format(
                    section_name, details.get('phase', '').upper(), ' -> '.join(details.get('selected_routes', [])),
                    details.get('final_target_velocity', result.acc_target_velocity)*3.6,
                    details.get('speed_source', 'planner'))
                if details.get('live_preview'):
                    entry = 'EGO POSE' if details.get('preview_ego_on_route') else 'NOMINAL ENTRY'
                    marker.text += '\nEDITOR LIVE REPLAN: {}'.format(entry)
            if details.get('blocked_by'):
                marker.text += '\nBLOCK: {}'.format(details['blocked_by'])
                if 'required_right_offset' in details:
                    marker.text += ' need={:.2f}m limit={:.2f}m'.format(
                        details['required_right_offset'], details['allowed_right_offset'])
                elif 'allowed_right_offset' in details:
                    marker.text += ' right limit={:.2f}m'.format(details['allowed_right_offset'])
            elif 'return_distance' in details:
                marker.text += '\nENDPOINT RETURN {:.1f}m'.format(details['return_distance'])
            self._set_color(marker, 0.1, 0.8, 1.0, 1.0)
        elif result.active:
            marker.text = "STATIC: RIGHT LATTICE  offset={:+.1f}m".format(
                result.selected_offset
            )
            self._set_color(marker, 0.1, 0.8, 1.0, 1.0)
        else:
            marker.text = "LATTICE: OFF ({})".format(result.reason)
            self._set_color(marker, 0.8, 0.8, 0.8, 0.8)
        return marker

    def _build_object_markers(self, result, vehicle_state, object_info_list, stamp):
        marker_array = MarkerArray()
        marker_array.markers.append(self._delete_all("map", stamp))

        marker_id = 0
        for object_info in object_info_list:
            dx = object_info.position.x - vehicle_state.position.x
            dy = object_info.position.y - vehicle_state.position.y
            if math.hypot(dx, dy) > parameters.OBSTACLE_DETECTION_DISTANCE + 10.0:
                continue

            marker = self._new_marker(
                frame_id="map",
                stamp=stamp,
                namespace="lattice_objects",
                marker_id=marker_id,
                marker_type=Marker.CUBE,
            )
            marker_id += 1
            marker.pose.position.x = object_info.position.x
            marker.pose.position.y = object_info.position.y
            marker.pose.position.z = 0.75
            object_yaw = math.radians(float(object_info.heading_deg))
            quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, object_yaw)
            marker.pose.orientation.x = quaternion[0]
            marker.pose.orientation.y = quaternion[1]
            marker.pose.orientation.z = quaternion[2]
            marker.pose.orientation.w = quaternion[3]
            marker.scale.x = max(1.0, float(object_info.size_x))
            marker.scale.y = max(1.0, float(object_info.size_y))
            marker.scale.z = 1.5
            if object_info.is_static:
                self._set_color(marker, 1.0, 0.45, 0.0, 0.75)
            elif object_info.type in (1, 2):
                self._set_color(marker, 0.0, 0.7, 1.0, 0.65)
            else:
                self._set_color(marker, 1.0, 0.0, 1.0, 0.65)
            marker_array.markers.append(marker)

        for prediction_id, predicted_path in enumerate(result.predicted_object_paths):
            if len(predicted_path) < 2:
                continue
            marker = self._new_marker(
                frame_id="map",
                stamp=stamp,
                namespace="predicted_object_paths",
                marker_id=prediction_id,
                marker_type=Marker.LINE_STRIP,
            )
            marker.scale.x = 0.12
            self._set_color(marker, 1.0, 0.0, 1.0, 0.9)
            marker.points = [
                RosPoint(x=point.x, y=point.y, z=1.7)
                for point in predicted_path
            ]
            marker_array.markers.append(marker)

        predicted_obb_id = 0
        for (entry, lane_match), predicted_path in zip(
            result.object_lane_matches, result.predicted_object_paths
        ):
            object_info = entry["object_info"]
            label = self._new_marker(
                frame_id="map",
                stamp=stamp,
                namespace="object_lane_labels",
                marker_id=predicted_obb_id,
                marker_type=Marker.TEXT_VIEW_FACING,
            )
            label.pose.position.x = object_info.position.x
            label.pose.position.y = object_info.position.y
            label.pose.position.z = 2.4
            label.scale.z = 0.42
            label.text = "S{} L{} {}".format(
                lane_match.lane_link.section_index,
                lane_match.lane_link.lane_number,
                lane_match.lane_link.id,
            )
            self._set_color(label, 1.0, 1.0, 1.0, 0.95)
            marker_array.markers.append(label)

            for point_index, point in enumerate(predicted_path[1:], 1):
                previous = predicted_path[point_index - 1]
                yaw = math.atan2(point.y - previous.y, point.x - previous.x)
                marker = self._new_marker(
                    frame_id="map",
                    stamp=stamp,
                    namespace="predicted_object_obbs",
                    marker_id=predicted_obb_id,
                    marker_type=Marker.CUBE,
                )
                predicted_obb_id += 1
                marker.pose.position.x = point.x
                marker.pose.position.y = point.y
                marker.pose.position.z = 0.5
                quaternion = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
                marker.pose.orientation.x = quaternion[0]
                marker.pose.orientation.y = quaternion[1]
                marker.pose.orientation.z = quaternion[2]
                marker.pose.orientation.w = quaternion[3]
                marker.scale.x = max(1.0, float(object_info.size_x))
                marker.scale.y = max(1.0, float(object_info.size_y))
                marker.scale.z = 1.0
                self._set_color(marker, 1.0, 0.0, 1.0, 0.16)
                marker_array.markers.append(marker)
        return marker_array

    def _build_lane_markers(self, result, stamp):
        marker_array = MarkerArray()
        marker_array.markers.append(self._delete_all("map", stamp))
        for marker_id, lane_link in enumerate(result.highway_lane_links):
            marker = self._new_marker(
                frame_id="map",
                stamp=stamp,
                namespace="highway_lane_centers",
                marker_id=marker_id,
                marker_type=Marker.LINE_STRIP,
            )
            marker.scale.x = 0.09
            if lane_link.excluded:
                self._set_color(marker, 1.0, 0.1, 0.1, 0.8)
            elif lane_link.id == result.target_link_id:
                marker.scale.x = 0.22
                self._set_color(marker, 1.0, 0.85, 0.0, 1.0)
            elif lane_link.id == result.current_link_id:
                marker.scale.x = 0.17
                self._set_color(marker, 1.0, 1.0, 1.0, 1.0)
            else:
                self._set_color(marker, 0.0, 0.75, 0.85, 0.5)
            marker.points = [
                RosPoint(x=point.x, y=point.y, z=0.03)
                for point in lane_link.points
            ]
            marker_array.markers.append(marker)

            label = self._new_marker(
                frame_id="map",
                stamp=stamp,
                namespace="highway_lane_labels",
                marker_id=marker_id,
                marker_type=Marker.TEXT_VIEW_FACING,
            )
            middle = lane_link.points[len(lane_link.points) // 2]
            label.pose.position.x = middle.x
            label.pose.position.y = middle.y
            label.pose.position.z = 1.0
            label.scale.z = 0.5
            label.text = "S{} L{} {}{}".format(
                lane_link.section_index,
                lane_link.lane_number,
                lane_link.id,
                " [X]" if lane_link.excluded else "",
            )
            if lane_link.excluded:
                self._set_color(label, 1.0, 0.2, 0.2, 1.0)
            else:
                self._set_color(label, 0.8, 1.0, 1.0, 0.9)
            marker_array.markers.append(label)
        marker_array.markers.extend(
            self._build_merge_entry_zone_markers(result, stamp)
        )
        marker_array.markers.extend(
            self._build_roundabout_entry_markers(result, stamp)
        )
        return marker_array

    def _build_merge_entry_zone_markers(self, result, stamp):
        priorities = getattr(result, "debug_details", {}).get(
            "merge_zone_priorities", {}
        )
        zones = (
            (
                "FIRST_PREDICTIVE",
                highway_parameters.PREDICTIVE_MERGE_POINT,
                highway_parameters.PREDICTIVE_MERGE_POINT_Z,
                highway_parameters.PREDICTIVE_MERGE_ENTRY_RADIUS,
            ),
            (
                "SECOND_ENTRY",
                highway_parameters.CONSERVATIVE_MERGE_POINT,
                highway_parameters.CONSERVATIVE_MERGE_POINT_Z,
                highway_parameters.CONSERVATIVE_MERGE_ENTRY_RADIUS,
            ),
        )
        markers = []
        for marker_id, (zone_name, center, world_z, radius) in enumerate(zones):
            markers.extend(
                self._build_single_merge_zone_markers(
                    zone_name,
                    center,
                    world_z,
                    radius,
                    priorities.get(zone_name, "NONE"),
                    marker_id,
                    stamp,
                )
            )
        return markers

    def _build_single_merge_zone_markers(
        self, zone_name, center, world_z, radius, priority, marker_id, stamp
    ):
        center_x, center_y = center
        colors = {
            "OTHER": (1.0, 0.15, 0.05),
            "EGO": (0.1, 1.0, 0.2),
            "NONE": (1.0, 0.85, 0.0),
        }
        red, green, blue = colors.get(priority, colors["NONE"])

        # Planning and all existing RViz paths use a flattened map frame (z=0).
        # Keep the marker on that plane and show the supplied MORAI world z in
        # the label so the circle remains visible with the current map setup.
        fill = self._new_marker(
            frame_id="map",
            stamp=stamp,
            namespace="merge_entry_zone_fill_{}".format(zone_name.lower()),
            marker_id=marker_id,
            marker_type=Marker.CYLINDER,
        )
        fill.pose.position.x = center_x
        fill.pose.position.y = center_y
        fill.pose.position.z = 0.02
        fill.scale.x = 2.0 * radius
        fill.scale.y = 2.0 * radius
        fill.scale.z = 0.04
        self._set_color(fill, red, green, blue, 0.12)

        outline = self._new_marker(
            frame_id="map",
            stamp=stamp,
            namespace="merge_entry_zone_outline_{}".format(zone_name.lower()),
            marker_id=marker_id,
            marker_type=Marker.LINE_STRIP,
        )
        outline.scale.x = 0.35
        self._set_color(outline, red, green, blue, 0.95)
        outline.points = [
            RosPoint(
                x=center_x + radius * math.cos(2.0 * math.pi * index / 72.0),
                y=center_y + radius * math.sin(2.0 * math.pi * index / 72.0),
                z=0.08,
            )
            for index in range(73)
        ]

        label = self._new_marker(
            frame_id="map",
            stamp=stamp,
            namespace="merge_entry_zone_label_{}".format(zone_name.lower()),
            marker_id=marker_id,
            marker_type=Marker.TEXT_VIEW_FACING,
        )
        label.pose.position.x = center_x
        label.pose.position.y = center_y
        label.pose.position.z = 2.2
        label.scale.z = 0.65
        label.text = (
            "{}  R={:.0f}m  priority={}\n"
            "XYZ=({:.2f}, {:.2f}, {:.2f})"
        ).format(zone_name, radius, priority, center_x, center_y, world_z)
        self._set_color(label, red, green, blue, 1.0)
        return [fill, outline, label]

    def _build_roundabout_entry_markers(self, result, stamp):
        occupied = (
            getattr(result, "reason", "") == "roundabout_entry_occupied_stop"
        )
        following = getattr(result, "reason", "") == "roundabout_entry_follow_lead"
        cruising = getattr(result, "reason", "") == "roundabout_entry_cruise"
        red, green, blue = (
            (1.0, 0.15, 0.05)
            if occupied
            else (1.0, 0.65, 0.0)
            if following
            else (0.1, 0.8, 1.0)
            if cruising
            else (0.1, 0.8, 1.0)
        )
        markers = []

        monitor = self._new_marker(
            frame_id="map",
            stamp=stamp,
            namespace="roundabout_entry_monitor",
            marker_id=0,
            marker_type=Marker.LINE_STRIP,
        )
        monitor.scale.x = 0.25
        monitor.points = self._polyline_band_points(
            parameters.ROUNDABOUT_ENTRY_MONITOR_POLYLINE,
            parameters.ROUNDABOUT_ENTRY_MONITOR_HALF_WIDTH,
            z=0.14,
        )
        self._set_color(monitor, red, green, blue, 0.95)
        markers.append(monitor)

        decision_start, decision_end = self._roundabout_decision_points()
        decision = self._new_marker(
            frame_id="map",
            stamp=stamp,
            namespace="roundabout_entry_decision_zone",
            marker_id=0,
            marker_type=Marker.LINE_STRIP,
        )
        decision.scale.x = 0.18
        decision.points = self._segment_box_points(
            decision_start,
            decision_end,
            parameters.ROUNDABOUT_ENTRY_APPROACH_HALF_WIDTH,
            z=0.12,
        )
        self._set_color(decision, 0.0, 0.7, 1.0, 0.85)
        markers.append(decision)

        stop_line = self._new_marker(
            frame_id="map",
            stamp=stamp,
            namespace="roundabout_entry_stop_line",
            marker_id=0,
            marker_type=Marker.LINE_STRIP,
        )
        stop_line.scale.x = 0.45
        stop_line.points = self._stop_line_points(z=0.2)
        self._set_color(stop_line, 1.0, 1.0, 1.0, 1.0)
        markers.append(stop_line)

        label = self._new_marker(
            frame_id="map",
            stamp=stamp,
            namespace="roundabout_entry_label",
            marker_id=0,
            marker_type=Marker.TEXT_VIEW_FACING,
        )
        label.pose.position.x = parameters.ROUNDABOUT_ENTRY_HOLD_POINT[0]
        label.pose.position.y = parameters.ROUNDABOUT_ENTRY_HOLD_POINT[1]
        label.pose.position.z = 2.3
        label.scale.z = 0.55
        label.text = (
            "ROUNDABOUT WAIT"
            if occupied
            else "ROUNDABOUT FOLLOW"
            if following
            else "ROUNDABOUT CRUISE"
            if cruising
            else "ROUNDABOUT CHECK"
        )
        self._set_color(label, red, green, blue, 1.0)
        markers.append(label)
        return markers

    @staticmethod
    def _roundabout_decision_points():
        start = parameters.ROUNDABOUT_ENTRY_APPROACH_START
        end = parameters.ROUNDABOUT_ENTRY_APPROACH_END
        stop_point = parameters.ROUNDABOUT_ENTRY_HOLD_POINT
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return stop_point, stop_point
        ux = dx / length
        uy = dy / length
        back = parameters.ROUNDABOUT_ENTRY_STOP_BACK_DISTANCE
        forward = parameters.ROUNDABOUT_ENTRY_STOP_FORWARD_DISTANCE
        return (
            (stop_point[0] - ux * back, stop_point[1] - uy * back),
            (stop_point[0] + ux * forward, stop_point[1] + uy * forward),
        )

    @staticmethod
    def _stop_line_points(z=0.0):
        start = parameters.ROUNDABOUT_ENTRY_APPROACH_START
        end = parameters.ROUNDABOUT_ENTRY_APPROACH_END
        stop = parameters.ROUNDABOUT_ENTRY_STOP_POINT
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return [RosPoint(x=stop[0], y=stop[1], z=z)]
        nx = -dy / length
        ny = dx / length
        half_width = parameters.ROUNDABOUT_ENTRY_APPROACH_HALF_WIDTH
        return [
            RosPoint(x=stop[0] - nx * half_width, y=stop[1] - ny * half_width, z=z),
            RosPoint(x=stop[0] + nx * half_width, y=stop[1] + ny * half_width, z=z),
        ]

    @staticmethod
    def _segment_box_points(start, end, half_width, z=0.0):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return [RosPoint(x=start[0], y=start[1], z=z)]
        nx = -dy / length
        ny = dx / length
        corners = (
            (start[0] - nx * half_width, start[1] - ny * half_width),
            (end[0] - nx * half_width, end[1] - ny * half_width),
            (end[0] + nx * half_width, end[1] + ny * half_width),
            (start[0] + nx * half_width, start[1] + ny * half_width),
            (start[0] - nx * half_width, start[1] - ny * half_width),
        )
        return [RosPoint(x=x, y=y, z=z) for x, y in corners]

    @staticmethod
    def _polyline_band_points(polyline, half_width, z=0.0):
        if len(polyline) < 2:
            return [RosPoint(x=polyline[0][0], y=polyline[0][1], z=z)] if polyline else []

        left = []
        right = []
        for index, point in enumerate(polyline):
            if index == 0:
                prev_point = point
                next_point = polyline[index + 1]
            elif index == len(polyline) - 1:
                prev_point = polyline[index - 1]
                next_point = point
            else:
                prev_point = polyline[index - 1]
                next_point = polyline[index + 1]
            dx = next_point[0] - prev_point[0]
            dy = next_point[1] - prev_point[1]
            length = math.hypot(dx, dy)
            if length < 1e-9:
                nx, ny = 0.0, 0.0
            else:
                nx = -dy / length
                ny = dx / length
            left.append((point[0] + nx * half_width, point[1] + ny * half_width))
            right.append((point[0] - nx * half_width, point[1] - ny * half_width))

        outline = left + list(reversed(right)) + [left[0]]
        return [RosPoint(x=x, y=y, z=z) for x, y in outline]

    @staticmethod
    def _new_marker(frame_id, stamp, namespace, marker_id, marker_type):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = rospy.Duration(0.25)
        return marker

    @staticmethod
    def _delete_all(frame_id, stamp):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.action = Marker.DELETEALL
        return marker

    @staticmethod
    def _set_color(marker, red, green, blue, alpha):
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = alpha
