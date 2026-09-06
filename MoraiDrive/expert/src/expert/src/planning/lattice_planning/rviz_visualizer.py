#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RViz MarkerArray publisher for the GT-based lattice planner."""

import math

import rospy
import tf
from geometry_msgs.msg import Point as RosPoint
from visualization_msgs.msg import Marker, MarkerArray

from . import parameters


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
        marker_array.markers.append(self._delete_all("map", stamp))

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
            elif candidate.points is result.path:
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

        if result.acc_override:
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
        return marker_array

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
