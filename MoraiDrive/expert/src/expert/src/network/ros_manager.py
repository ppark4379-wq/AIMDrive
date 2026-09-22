#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import deque
import threading
import time

import numpy as np
import rospy
import tf
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from morai_msgs.msg import (
    CtrlCmd,
    EgoVehicleStatus,
    GetTrafficLightStatus,
    ObjectStatusList,
    SetTrafficLight,
)
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float32

from ..config.config import Config
from ..obstacle.object_info import ObjectInfo
from ..planning.lattice_planning import roundabout_debug
from ..planning.lattice_planning.rviz_visualizer import LatticeRvizVisualizer
from ..vehicle_state import VehicleState
from .grpc_sync_inputs import MoraiGrpcSyncInputs
from .synchronized_inputs import (
    control_dt,
    has_matching_object,
    select_objects,
    select_traffic_light,
)


class RosManager:
    def __init__(self, autonomous_driving):
        self.autonomous_driving = autonomous_driving

        config = Config()
        rospy.init_node("morai_standard", anonymous=True)

        editor_parameter = rospy.get_param("~static_editor_mode", False)
        self.static_editor_mode = (
            editor_parameter.strip().lower() in ("1", "true", "yes", "on")
            if isinstance(editor_parameter, str)
            else bool(editor_parameter)
        )
        self.autonomous_driving.lattice_planning_process.set_static_editor_mode(
            self.static_editor_mode
        )
        if self.static_editor_mode:
            rospy.logwarn(
                "Static Scenario Editor live-replan preview is enabled; "
                "vehicle commands remain controlled by the caller remap"
            )

        self.traffic_light_control = config["map"]["traffic_light_control"]
        self.global_path = self.convert_to_ros_path(config["map"]["path"], "/map")

        common_config = config["common"]
        self.sampling_rate = float(common_config["sampling_rate"])
        self.realtime_sampling_rate = float(
            common_config.get("realtime_sampling_rate", self.sampling_rate)
        )
        self.sync_fallback_sampling_rate = float(
            common_config.get(
                "sync_fallback_sampling_rate", self.realtime_sampling_rate
            )
        )
        self.sync_runtime_active_params = tuple(
            common_config.get(
                "sync_runtime_active_params",
                ("/morai_drive/sync_runtime_active",),
            )
        )
        self.sync_fallback_sampling_rates = dict(
            common_config.get("sync_fallback_sampling_rates", {})
        )
        self._active_sync_fallback_sampling_rate = (
            self.sync_fallback_sampling_rate
        )
        self.sync_input_slop_sec = max(
            0.0, float(common_config.get("sync_input_slop_sec", 0.05))
        )
        self.sync_input_wait_sec = max(
            0.0, float(common_config.get("sync_input_wait_sec", 0.01))
        )
        self.traffic_light_max_age_sec = max(
            0.0, float(common_config.get("traffic_light_max_age_sec", 1.0))
        )
        grpc_input_config = dict(common_config.get("sync_grpc_inputs", {}))
        self._grpc_sync_inputs = (
            MoraiGrpcSyncInputs(grpc_input_config)
            if bool(grpc_input_config.get("enabled", True))
            else None
        )
        self.count = 0

        self.vehicle_state = VehicleState()
        self.object_info_list = []
        self.traffic_light = []

        self.is_status = False
        # Object data remains optional for collection. A missing synchronized
        # sample is represented by [], never by an unrelated SyncMode frame.
        self.is_object_info = True
        self.is_traffic_light = False
        self._status_sequence = 0
        self._last_controlled_status_sequence = 0
        self._status_condition = threading.Condition()
        self._ego_source_time = None
        self._ego_frame = None
        self._ego_arrival_time = None
        self._last_processed_ego_source_time = None
        self._object_history = deque(maxlen=200)
        self._traffic_light_history = deque(maxlen=200)
        self._object_message_count = 0
        self._traffic_light_message_count = 0
        self._grpc_snapshot_count = 0
        self._grpc_last_tick = None
        if self._grpc_sync_inputs is not None:
            rospy.on_shutdown(self._grpc_sync_inputs.close)

    def execute(self):
        print("start simulation")
        self._set_protocol()
        next_realtime_cycle = time.monotonic()
        while not rospy.is_shutdown():
            sync_mode = self._sync_mode_active()
            snapshot = self._wait_for_observation(sync_mode)
            if snapshot is None:
                continue
            (
                status_sequence,
                vehicle_state,
                object_info_list,
                traffic_light,
                control_dt,
                object_delta,
                traffic_age,
                input_source,
                grpc_tick,
                grpc_tick_delta,
            ) = snapshot
            roundabout_debug.input_context(
                ego_seq=status_sequence, object_count=len(object_info_list),
                object_delta_s=object_delta, control_dt_s=control_dt, source=input_source,
                sync_mode=sync_mode, grpc_tick=grpc_tick, grpc_tick_delta=grpc_tick_delta)
            control_input, local_path = self.autonomous_driving.execute(
                vehicle_state,
                object_info_list,
                traffic_light,
                sync_mode=sync_mode,
                control_dt=control_dt,
            )
            self._send_data(
                control_input, local_path, vehicle_state, object_info_list
            )
            self._last_controlled_status_sequence = status_sequence
            self._log_input_health(
                sync_mode,
                status_sequence,
                object_info_list,
                traffic_light,
                object_delta,
                traffic_age,
                control_dt,
                input_source,
                grpc_tick,
                grpc_tick_delta,
            )

            if not sync_mode:
                period = 1.0 / self.realtime_sampling_rate
                next_realtime_cycle = max(
                    next_realtime_cycle + period, time.monotonic()
                )
                remaining = next_realtime_cycle - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
        print("end simulation")

    def _wait_for_observation(self, sync_mode):
        with self._status_condition:
            while not rospy.is_shutdown():
                if self.is_status and (
                    not sync_mode
                    or self._status_sequence > self._last_controlled_status_sequence
                ):
                    break
                self._status_condition.wait(timeout=0.1)
                if sync_mode != self._sync_mode_active():
                    return None
            if rospy.is_shutdown():
                return None

            status_sequence = self._status_sequence
            vehicle_state = self.vehicle_state
            ego_source_time = self._ego_source_time
            ego_frame = self._ego_frame

            # Allow callbacks from the same simulator frame to finish. Object
            # data is optional, so this short wait cannot stall tick ownership.
            if (
                sync_mode
                and self._grpc_sync_inputs is None
                and ego_source_time is not None
            ):
                deadline = time.monotonic() + self.sync_input_wait_sec
                while (
                    not self._has_matching_object(ego_source_time)
                    and not rospy.is_shutdown()
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    self._status_condition.wait(timeout=remaining)

            object_info_list, object_delta = self._select_objects(
                sync_mode, ego_source_time
            )
            traffic_light, traffic_age = self._select_traffic_light(
                sync_mode, ego_source_time
            )

        input_source = "ros"
        grpc_tick = None
        grpc_tick_delta = None
        if sync_mode and self._grpc_sync_inputs is not None:
            grpc_snapshot = self._grpc_sync_inputs.fetch()
            if grpc_snapshot is not None:
                grpc_tick = grpc_snapshot.tick
                grpc_tick_delta = (
                    None
                    if ego_frame is None
                    else grpc_snapshot.tick - int(ego_frame)
                )
                # Keep /Ego_topic as the control-state authority, exactly as
                # the original ROS pipeline did. gRPC replaces only the ROS
                # messages unavailable in Sync Mode, using one coherent actor
                # snapshot for objects and traffic-light state.
                object_info_list = grpc_snapshot.objects
                traffic_light = grpc_snapshot.traffic_light
                object_delta = 0.0
                traffic_age = 0.0 if traffic_light else None
                input_source = "grpc"
                self._grpc_snapshot_count += 1
                self._grpc_last_tick = grpc_snapshot.tick

        control_dt = self._control_dt(sync_mode, ego_source_time)

        return (
            status_sequence,
            vehicle_state,
            object_info_list,
            traffic_light,
            control_dt,
            object_delta,
            traffic_age,
            input_source,
            grpc_tick,
            grpc_tick_delta,
        )

    def _sync_mode_active(self):
        for parameter in self.sync_runtime_active_params:
            try:
                if bool(rospy.get_param(parameter, False)):
                    self._active_sync_fallback_sampling_rate = float(
                        self.sync_fallback_sampling_rates.get(
                            parameter, self.sync_fallback_sampling_rate
                        )
                    )
                    return True
            except (KeyError, rospy.ROSException):
                continue
        return False

    @staticmethod
    def _message_source_time(data):
        stamp = getattr(getattr(data, "header", None), "stamp", None)
        if stamp is None:
            return None
        seconds = int(getattr(stamp, "secs", 0))
        nanoseconds = int(getattr(stamp, "nsecs", 0))
        if seconds == 0 and nanoseconds == 0:
            return None
        return seconds + nanoseconds / 1e9

    def _has_matching_object(self, ego_source_time):
        return has_matching_object(
            self._object_history, ego_source_time, self.sync_input_slop_sec
        )

    def _select_objects(self, sync_mode, ego_source_time):
        return select_objects(
            self._object_history,
            sync_mode,
            ego_source_time,
            self.sync_input_slop_sec,
        )

    def _select_traffic_light(self, sync_mode, ego_source_time):
        return select_traffic_light(
            self._traffic_light_history,
            sync_mode,
            ego_source_time,
            self.traffic_light_max_age_sec,
        )

    def _control_dt(self, sync_mode, ego_source_time):
        selected_dt, current_source_time = control_dt(
            sync_mode,
            ego_source_time,
            self._last_processed_ego_source_time,
            self.realtime_sampling_rate,
            self._active_sync_fallback_sampling_rate,
        )
        self._last_processed_ego_source_time = current_source_time
        return selected_dt

    def _log_input_health(
        self,
        sync_mode,
        status_sequence,
        object_info_list,
        traffic_light,
        object_delta,
        traffic_age,
        control_dt,
        input_source,
        grpc_tick,
        grpc_tick_delta,
    ):
        mode = "sync" if sync_mode else "realtime"
        rospy.loginfo_throttle(
            5.0,
            "[ExpertInput] mode=%s ego_seq=%d dt=%.3fs "
            "object_msgs=%d selected_objects=%d object_delta=%s "
            "traffic_msgs=%d traffic_selected=%s traffic_age=%s "
            "source=%s grpc_snapshots=%d grpc_tick=%s tick_delta=%s",
            mode,
            status_sequence,
            control_dt,
            self._object_message_count,
            len(object_info_list),
            "n/a" if object_delta is None else "{:.3f}s".format(object_delta),
            self._traffic_light_message_count,
            str(bool(traffic_light)).lower(),
            "n/a" if traffic_age is None else "{:.3f}s".format(traffic_age),
            input_source,
            self._grpc_snapshot_count,
            "n/a" if grpc_tick is None else str(grpc_tick),
            "n/a" if grpc_tick_delta is None else str(grpc_tick_delta),
        )
        if (
            sync_mode
            and self._grpc_sync_inputs is not None
            and input_source != "grpc"
        ):
            rospy.logwarn_throttle(
                5.0,
                "[ExpertInput] MORAI gRPC snapshot unavailable; using ROS "
                "fallback without synchronized planning data: %s",
                self._grpc_sync_inputs.last_error or "snapshot older than Ego frame",
            )
        elif sync_mode and input_source != "grpc" and self._object_message_count == 0:
            rospy.logwarn_throttle(
                5.0,
                "[ExpertInput] /Object_topic has not published; ACC and "
                "obstacle avoidance have no object input",
            )
        elif (
            sync_mode
            and object_delta is not None
            and object_delta > self.sync_input_slop_sec
        ):
            rospy.logwarn_throttle(
                5.0,
                "[ExpertInput] nearest /Object_topic is %.3fs from Ego "
                "(allowed %.3fs); object planning is disabled for this frame",
                object_delta,
                self.sync_input_slop_sec,
            )

    def _set_protocol(self):
        self.global_path_pub = rospy.Publisher("/global_path", Path, queue_size=1)
        self.local_path_pub = rospy.Publisher("/local_path", Path, queue_size=1)
        self.ctrl_pub = rospy.Publisher("/ctrl_cmd", CtrlCmd, queue_size=1)
        self.target_velocity_pub = rospy.Publisher(
            "/expert_target_velocity", Float32, queue_size=1
        )
        self.traffic_light_pub = rospy.Publisher(
            "/SetTrafficLight", SetTrafficLight, queue_size=1
        )
        self.odom_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
        self.lattice_visualizer = LatticeRvizVisualizer()

        rospy.Subscriber(
            "/Ego_topic", EgoVehicleStatus, self.vehicle_status_callback
        )
        rospy.Subscriber(
            "/Object_topic", ObjectStatusList, self.object_info_callback
        )
        rospy.Subscriber(
            "/GetTrafficLightStatus",
            GetTrafficLightStatus,
            self.traffic_light_callback,
        )
        if self.static_editor_mode:
            rospy.Subscriber(
                "/static_scenario_editor/drive_enabled",
                Bool,
                self.static_editor_drive_callback,
                queue_size=1,
            )

    def static_editor_drive_callback(self, message):
        self.autonomous_driving.lattice_planning_process.set_static_editor_driving(
            bool(message.data)
        )

    def _send_data(
        self, control_input, local_path, vehicle_state, object_info_list
    ):
        self.ctrl_pub.publish(CtrlCmd(**control_input.__dict__))
        self.target_velocity_pub.publish(
            Float32(data=float(self.autonomous_driving.previous_target_velocity))
        )
        self.local_path_pub.publish(self.convert_to_ros_path(local_path, "map"))
        self.odom_pub.publish(self.convert_to_odometry(vehicle_state))
        try:
            self.lattice_visualizer.publish(
                self.autonomous_driving.lattice_planning_process.last_result,
                vehicle_state,
                object_info_list,
            )
        except Exception as error:
            rospy.logwarn_throttle(
                5.0, "Lattice RViz publish failed: %s", error
            )

        publish_rate = max(1, int(round(self.autonomous_driving.sampling_rate)))
        if self.count >= publish_rate:
            self.global_path_pub.publish(self.global_path)
            self.count = 0
        self.count += 1

    @staticmethod
    def convert_to_ros_path(path, frame_id):
        ros_path = Path()
        ros_path.header.frame_id = frame_id
        for point in path:
            pose_stamped = PoseStamped()
            pose_stamped.pose.position = Point(x=point.x, y=point.y, z=0)
            pose_stamped.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
            ros_path.poses.append(pose_stamped)
        return ros_path

    @staticmethod
    def convert_to_odometry(vehicle_state):
        odometry = Odometry()
        odometry.header.frame_id = "map"
        odometry.child_frame_id = "gps"
        quaternion = tf.transformations.quaternion_from_euler(
            0, 0, vehicle_state.yaw
        )
        odometry.pose.pose.position = Point(
            x=vehicle_state.position.x, y=vehicle_state.position.y, z=0
        )
        odometry.pose.pose.orientation = Quaternion(
            x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
        )
        return odometry

    def vehicle_status_callback(self, data):
        vehicle_state = VehicleState(
            data.position.x,
            data.position.y,
            np.deg2rad(data.heading),
            data.velocity.x,
        )
        br = tf.TransformBroadcaster()
        br.sendTransform(
            (vehicle_state.position.x, vehicle_state.position.y, 0),
            tf.transformations.quaternion_from_euler(0, 0, vehicle_state.yaw),
            rospy.Time.now(),
            "gps",
            "map",
        )
        with self._status_condition:
            self.vehicle_state = vehicle_state
            self._ego_source_time = self._message_source_time(data)
            header = getattr(data, "header", None)
            self._ego_frame = (
                int(getattr(header, "seq", 0)) if header is not None else None
            )
            self._ego_arrival_time = time.monotonic()
            self.is_status = True
            self._status_sequence += 1
            self._status_condition.notify_all()

    def object_info_callback(self, data):
        def convert_object(item, is_static=False):
            return ObjectInfo(
                item.position.x,
                item.position.y,
                # MORAI reports NPC speed in km/h; planning uses SI units.
                item.velocity.x / 3.6,
                item.type,
                name=item.name,
                is_static=is_static,
                size_x=item.size.x,
                size_y=item.size.y,
                heading_deg=item.heading,
                unique_id=item.unique_id,
            )

        object_info_list = (
            [convert_object(item) for item in data.npc_list]
            + [convert_object(item, is_static=True) for item in data.obstacle_list]
            + [convert_object(item) for item in data.pedestrian_list]
        )
        with self._status_condition:
            self.object_info_list = object_info_list
            self.is_object_info = True
            self._object_message_count += 1
            self._object_history.append(
                {
                    "source_time": self._message_source_time(data),
                    "arrival_time": time.monotonic(),
                    "objects": object_info_list,
                }
            )
            self._status_condition.notify_all()

    def traffic_light_callback(self, data):
        if self.traffic_light_control:
            traffic_light_status = 16
            set_traffic_light = SetTrafficLight(
                trafficLightIndex=data.trafficLightIndex,
                trafficLightStatus=traffic_light_status,
            )
            self.traffic_light_pub.publish(set_traffic_light)
        else:
            traffic_light_status = data.trafficLightStatus

        traffic_light = [
            data.trafficLightIndex,
            traffic_light_status,
            data.trafficLightType,
            time.monotonic(),
        ]
        with self._status_condition:
            self.traffic_light = traffic_light
            self.is_traffic_light = True
            self._traffic_light_message_count += 1
            self._traffic_light_history.append(
                {
                    "source_time": self._message_source_time(data),
                    "arrival_time": traffic_light[3],
                    "traffic_light": traffic_light,
                }
            )
            self._status_condition.notify_all()
