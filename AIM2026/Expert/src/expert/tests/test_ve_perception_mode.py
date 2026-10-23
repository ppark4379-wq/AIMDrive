"""VE input contracts, unknown-object geometry and fail-stop control without ROS."""
import copy
import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from expert.src.autonomous_driving import AutonomousDriving
from expert.src.localization.point import Point
from expert.src.obstacle.object_info import ObjectInfo
from expert.src.planning.ve_planning import (
    GridYawConverter, VeInputGate, VePerceptionPlanningProcess, perception_source,
)
from expert.src.vehicle_state import VehicleState

try:
    from expert.src.network import ros_manager as ros_module
except ImportError:
    ros_module = None


BASIS = dict(schema="morai_projected_grid_basis_v1", position_frame="projected_grid",
             source_orientation_frame="true_enu", orientation_frame="projected_grid",
             crs="+proj=utm +zone=52 +datum=WGS84 +units=m +no_defs",
             origin_xyz=[302595., 4124145., 0.])
HEALTH = dict(schema="morai_ve_perception_health_v1", ready=True, stamp_ns=100_000_000_000,
              sequence=7, sensor_received_monotonic_ns=9_950_000_000,
              published_monotonic_ns=9_980_000_000, checkpoint_sha256="a" * 64,
              output_frame="projected_grid", map_basis=BASIS, invalid_reason=None,
              generation="publisher-session", object_count=0, persistent_tracks_available=False,
              scalar_velocity_only=True, object_size_axes="heading_forward_length_lateral_width")


def unknown(x=10., y=.8, length=1., width=1.):
    return ObjectInfo(x, y, 0., -1, is_static=False, size_x=length, size_y=width)


class VeInputGateTest(unittest.TestCase):
    def setUp(self):
        self.gate = VeInputGate(dict(expected_checkpoint_sha256="a" * 64,
                                     expected_map_basis=copy.deepcopy(BASIS)))

    def accept(self, objects=(), health=None):
        health = copy.deepcopy(HEALTH) if health is None else health
        health["object_count"] = len(objects)
        self.gate.accept_objects(objects, 7, 100_000_000_000, 9.98, "map")
        self.gate.accept_health(health, 9.99)

    def select(self, now=10., ego_ns=100_100_000_000, ego_received=9.99):
        return self.gate.select(now, ego_ns, ego_received)

    def test_original_mode_remains_default_and_ve_forbids_grpc(self):
        self.assertEqual("ros", perception_source({}))
        for enabled in (True, None):
            config = dict(perception_source="ve")
            if enabled is not None:
                config["sync_grpc_inputs"] = dict(enabled=enabled)
            with self.assertRaises(ValueError):
                perception_source(config)
        self.assertEqual("ve", perception_source(dict(perception_source="ve", sync_grpc_inputs=dict(enabled=False))))

    def test_missing_input_is_not_valid_empty(self):
        objects, health, age = self.select()
        self.assertEqual([], objects)
        self.assertFalse(health["ready"])
        self.assertEqual("missing_health", health["invalid_reason"])
        self.assertIsNone(age)

    def test_fresh_explicit_empty_is_valid(self):
        self.accept()
        objects, health, age = self.select()
        self.assertEqual([], objects)
        self.assertTrue(health["input_validated"])
        self.assertAlmostEqual(.1, age)

    def test_unknown_geometry_does_not_require_gt_class_or_static_tag(self):
        obj = unknown()
        self.accept([obj])
        objects, health, _ = self.select()
        self.assertIs(objects[0], obj)
        self.assertTrue(health["ready"])
        self.assertFalse(obj.is_static)
        self.assertEqual(-1, obj.type)

    def test_no_matching_health_objects_cannot_reuse_gt_or_old_frame(self):
        self.gate.accept_objects([unknown()], 6, 99_900_000_000, 9.98, "map")
        self.gate.accept_health(copy.deepcopy(HEALTH), 9.99)
        self.assertEqual("missing_matching_objects", self.select()[1]["invalid_reason"])

    def test_failed_health_immediately_invalidates_previous_good_output(self):
        self.accept([unknown()])
        self.assertTrue(self.select()[1]["ready"])
        self.gate.accept_health(dict(schema=self.gate.schema, ready=False, invalid_reason="camera_missing"), 9.999)
        self.assertEqual("camera_missing", self.select()[1]["invalid_reason"])

    def test_checkpoint_basis_and_axis_identity_are_required(self):
        for key, value in (("checkpoint_sha256", "b" * 64), ("output_frame", "true_enu"),
                           ("object_size_axes", "width_length"), ("persistent_tracks_available", True)):
            with self.subTest(key=key):
                h = copy.deepcopy(HEALTH)
                h[key] = value
                self.accept(health=h)
                self.assertFalse(self.select()[1]["ready"])

    def test_generation_change_requires_expert_restart(self):
        self.accept()
        self.assertTrue(self.select()[1]["ready"])
        h = copy.deepcopy(HEALTH)
        h["generation"] = "new-publisher-session"
        self.accept(health=h)
        self.assertEqual("generation_changed", self.select()[1]["invalid_reason"])

    def test_receive_and_sensor_source_all_expire(self):
        self.accept()
        self.assertFalse(self.select(now=10.5)[1]["ready"])
        for key in ("sensor_received_monotonic_ns", "published_monotonic_ns"):
            with self.subTest(key=key):
                h = copy.deepcopy(HEALTH)
                h[key] = 9_000_000_000
                self.accept(health=h)
                self.assertFalse(self.select()[1]["ready"])

    def test_missing_stale_future_ego_and_source_are_invalid(self):
        self.accept()
        for timestamp in (None, 99_999_999_999, 101_000_000_000):
            with self.subTest(timestamp=timestamp):
                self.assertFalse(self.select(ego_ns=timestamp)[1]["ready"])
        self.assertFalse(self.select(ego_received=9.)[1]["ready"])

    def test_invalid_bbox_and_gt_class_are_rejected(self):
        for obj in (unknown(length=0.), unknown(width=float("nan")), unknown()):
            if obj.size_x > 0 and math.isfinite(obj.size_y):
                obj.type = 1
            self.accept([obj])
            self.assertFalse(self.select()[1]["ready"])

    def test_health_count_and_object_frame_must_match(self):
        self.accept()
        self.gate.health["object_count"] = 1
        self.assertFalse(self.select()[1]["ready"])
        self.accept()
        self.gate.objects[-1]["frame_id"] = "ego"
        self.assertFalse(self.select()[1]["ready"])

    def test_malformed_health_fails_closed(self):
        self.accept()
        self.gate.accept_health("{invalid", 9.99)
        self.assertFalse(self.select()[1]["ready"])

    def test_newer_fresh_sample_does_not_hide_expired_used_bbox(self):
        self.accept([unknown()])
        _, used, _ = self.select()
        self.assertTrue(self.gate.used_snapshot_valid(used, 10., 100_100_000_000, 9.99))
        newer = copy.deepcopy(HEALTH)
        newer.update(sequence=8, stamp_ns=100_400_000_000, object_count=0,
                     sensor_received_monotonic_ns=10_350_000_000,
                     published_monotonic_ns=10_380_000_000)
        self.gate.accept_objects([], 8, 100_400_000_000, 10.38, "map")
        self.gate.accept_health(newer, 10.39)
        self.assertTrue(self.gate.select(10.4, 100_500_000_000, 10.39)[1]["ready"])
        self.assertFalse(self.gate.used_snapshot_valid(used, 10.4, 100_500_000_000, 10.39))


class VeUnknownPlanningTest(unittest.TestCase):
    def setUp(self):
        self.path = [Point(i * .25, 0.) for i in range(121)]
        self.ego = VehicleState(0., 0., 0., 0.)
        self.config = dict(road_lateral_bounds_m=[-4.5, 1.75], max_speed_kph=10.)
        self.process = VePerceptionPlanningProcess(self.config)
        self.health = dict(ready=True, input_validated=True)

    def run_plan(self, objects=(), **kwargs):
        return self.process.run(self.ego, self.path, list(objects), 20., self.health, **kwargs)

    def test_explicit_bounds_and_speed_ceiling_are_required(self):
        for config in ({}, dict(self.config, max_speed_kph=11.), dict(self.config, collision_margin_m=.1)):
            with self.assertRaises(ValueError):
                VePerceptionPlanningProcess(config)

    def test_unknown_object_is_checked_without_relevance_or_class_filter(self):
        self.process.planner._find_relevant_obstacles = lambda *a: self.fail("reference-only filtering used")
        result = self.run_plan([unknown()])
        self.assertEqual("ve_avoidance", result.reason)
        self.assertLess(result.selected_offset, 0.)
        self.assertTrue(result.active)
        self.assertEqual(1, result.debug_details["ve_perception"]["checked_objects"])

    def test_explicit_empty_drives_original_reference_at_low_speed(self):
        result = self.run_plan()
        self.assertEqual("ve_reference_clear", result.reason)
        self.assertAlmostEqual(10. / 3.6, result.acc_target_velocity)
        self.assertFalse(result.debug_details["ve_perception"]["motion_prediction"])

    def test_all_candidates_blocked_returns_empty_path_and_zero_target(self):
        result = self.run_plan([unknown(length=3., width=20.)])
        self.assertEqual("all_candidates_blocked", result.reason)
        self.assertEqual([], result.path)
        self.assertEqual(0., result.acc_target_velocity)
        self.assertTrue(result.debug_details["ve_perception"]["force_brake"])

    def test_entire_vehicle_body_is_kept_within_explicit_bounds(self):
        result = self.run_plan([unknown()])
        for candidate in result.candidates:
            if candidate.offset >= 1.:
                self.assertFalse(candidate.valid)
        # A centre path is also invalid when the declared corridor is narrower
        # than the physical vehicle; an empty detection must not bypass bounds.
        self.process = VePerceptionPlanningProcess(dict(self.config, road_lateral_bounds_m=[-.5, .5]))
        self.assertEqual([], self.run_plan().path)

    def test_no_ghost_obstacle_memory_after_fresh_empty(self):
        self.assertTrue(self.run_plan([unknown()]).active)
        self.assertEqual("ve_reference_clear", self.run_plan().reason)

    def test_unvalidated_or_nonfinite_inputs_stop(self):
        self.health = dict(ready=True)
        self.assertEqual([], self.run_plan().path)
        self.health = dict(ready=True, input_validated=True)
        self.ego.velocity = float("nan")
        self.assertEqual("invalid_ego_or_velocity", self.run_plan().reason)

    def test_outside_static_trial_region_stops_each_cycle(self):
        self.process = VePerceptionPlanningProcess(dict(self.config, control_region_xy=[[-1., -1.], [1., 1.]]))
        self.assertTrue(self.run_plan().path)
        self.ego.position = Point(2., 0.)
        self.assertEqual("outside_control_region", self.run_plan().reason)

    def test_actual_return_from_avoidance_is_checked_before_following_clear_gp(self):
        self.ego.position = Point(0., -3.)
        # The GP itself is clear at y=0. The object is directly on the real
        # vehicle's connecting path, so returning to the GP must be rejected.
        result = self.run_plan([unknown(x=2., y=-3.)])
        self.assertEqual("all_candidates_blocked", result.reason)
        self.assertEqual([], result.path)


class VeExpertControlTest(unittest.TestCase):
    def setUp(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "src/config/config.json").read_text())
        self.core = AutonomousDriving.__new__(AutonomousDriving)
        self.core._init_control(config)
        self.core.perception_source = "ve"
        self.core.lattice_planning_process = VePerceptionPlanningProcess(dict(road_lateral_bounds_m=[-4.5, 1.75]))
        path = [Point(i * .25, 0.) for i in range(121)]
        self.core.path_manager = SimpleNamespace(current_waypoint=0, get_local_path=lambda _: (path, 15.))
        self.signal_requests = []
        self.signal_speed = float("inf")
        def signal(**kwargs):
            self.signal_requests.append(kwargs)
            return self.signal_speed
        self.core.traffic_light_stop_controller = SimpleNamespace(get_target_velocity=signal)
        self.ego = VehicleState()
        self.health = dict(ready=True, input_validated=True)

    def test_ve_mode_keeps_original_signal_controller_and_skips_gt_class_acc(self):
        command, path = self.core.execute(self.ego, [], ["signal", 33], perception_health=self.health)
        self.assertTrue(path)
        self.assertGreater(command.accel, 0.)
        self.assertEqual(["signal", 33], self.signal_requests[0]["current_traffic_light"])
        self.assertLessEqual(self.core.previous_target_velocity, 10. / 3.6)

    def test_original_red_signal_can_still_force_full_brake(self):
        self.signal_speed = 0.
        command, path = self.core.execute(self.ego, [], ["signal", 1], perception_health=self.health)
        self.assertTrue(path)
        self.assertEqual(1., command.brake)
        self.assertEqual(0., command.accel)

    def test_invalid_or_all_blocked_perception_never_passes_empty_path_to_controller(self):
        self.core._calculate_control = lambda *a, **k: self.fail("empty path sent to path controller")
        for objects, health in (([], None), ([unknown(length=3., width=20.)], self.health)):
            command, path = self.core.execute(self.ego, objects, [], perception_health=health)
            self.assertEqual([], path)
            self.assertEqual(1., command.brake)
            self.assertEqual(0., self.core.previous_target_velocity)


class GridOrientationTest(unittest.TestCase):
    def test_cached_conversion_matches_projection_convergence(self):
        from pyproj import Proj, Transformer
        converter = GridYawConverter(BASIS)
        inverse = Transformer.from_crs(BASIS["crs"], "EPSG:4326", always_xy=True)
        projection = Proj(BASIS["crs"])
        for x, y, yaw in ((0., 0., 0.), (-60., -130., math.pi / 2), (12., 140., -math.pi)):
            lon, lat = inverse.transform(x + BASIS["origin_xyz"][0], y + BASIS["origin_xyz"][1])
            expected = yaw + math.radians(projection.get_factors(lon, lat).meridian_convergence)
            self.assertAlmostEqual(expected, converter(yaw, x, y), places=12)


@unittest.skipIf(ros_module is None, "ROS message packages unavailable; callbacks run in the system ROS environment")
class VeRosAdapterTest(unittest.TestCase):
    """Use actual ROS message definitions, with every ROS operation mocked."""
    def setUp(self):
        self.manager = ros_module.RosManager.__new__(ros_module.RosManager)
        self.manager.perception_source = "ve"
        self.manager._status_condition = threading.Condition()
        self.manager._ve_input_gate = VeInputGate(dict(expected_checkpoint_sha256="a" * 64,
                                                      expected_map_basis=BASIS))
        self.manager._object_message_count = 0
        self.manager._status_sequence = 0
        self.manager.static_editor_mode = False
        self.manager.ve_object_topic = "/ve/Object_topic"
        self.manager.ve_health_topic = "/ve/perception_health"
        self.published = {}
        self.subscribed = []
        self.shutdown = []
        def publisher(topic, *args, **kwargs):
            self.published[topic] = []
            return SimpleNamespace(publish=self.published[topic].append)
        def subscriber(topic, *args, **kwargs):
            self.subscribed.append(topic)
        rospy = SimpleNamespace(Publisher=publisher, Subscriber=subscriber,
                                on_shutdown=self.shutdown.append, Time=SimpleNamespace(now=lambda: None))
        self.rospy_patch = patch.object(ros_module, "rospy", rospy)
        self.visualizer_patch = patch.object(ros_module, "LatticeRvizVisualizer", lambda: None)
        self.rospy_patch.start()
        self.visualizer_patch.start()
        self.addCleanup(self.rospy_patch.stop)
        self.addCleanup(self.visualizer_patch.stop)
        self.manager.autonomous_driving = SimpleNamespace(force_ve_brake=lambda: SimpleNamespace(
            accel=0., brake=1., front_steer=0., rear_steer=0., longlCmdType=1))

    def test_only_ve_objects_subscribed_and_original_signal_topic_kept(self):
        self.manager._set_protocol()
        self.assertIn("/ve/Object_topic", self.subscribed)
        self.assertIn("/ve/perception_health", self.subscribed)
        self.assertIn("/GetTrafficLightStatus", self.subscribed)
        self.assertNotIn("/Object_topic", self.subscribed)
        self.assertEqual(1., self.published["/ctrl_cmd"][-1].brake)
        self.shutdown[0]()
        self.assertEqual(2, len(self.published["/ctrl_cmd"]))

    def test_unknown_wire_axes_and_frame_local_ids_are_not_static_tracks(self):
        item = SimpleNamespace(type=-1, position=SimpleNamespace(x=4., y=2.),
                               size=SimpleNamespace(x=4.5, y=1.8), heading=30., unique_id=42,
                               velocity=SimpleNamespace(x=36.))
        header = SimpleNamespace(seq=7, stamp=SimpleNamespace(secs=100, nsecs=0), frame_id="map")
        data = SimpleNamespace(npc_list=[item], obstacle_list=[], pedestrian_list=[], header=header)
        self.manager.ve_object_info_callback(data)
        obj = self.manager._ve_input_gate.objects[-1]["objects"][0]
        self.assertEqual(4.5, obj.size_x)
        self.assertEqual(1.8, obj.size_y)
        self.assertEqual(-1, obj.type)
        self.assertFalse(obj.is_static)
        self.assertIsNone(obj.unique_id)
        self.assertEqual(0., obj.velocity)  # Unknown world heading/speed is not invented.

    def test_ego_uses_same_grid_yaw_and_vector_speed_as_ve_source(self):
        data = SimpleNamespace(position=SimpleNamespace(x=-60., y=-130.), heading=90.,
                               velocity=SimpleNamespace(x=3., y=4.),
                               header=SimpleNamespace(seq=10, stamp=SimpleNamespace(secs=100, nsecs=0)))
        with patch.object(ros_module.tf, "TransformBroadcaster", lambda: SimpleNamespace(sendTransform=lambda *a: None)):
            self.manager.vehicle_status_callback(data)
        self.assertEqual(5., self.manager.vehicle_state.velocity)
        self.assertAlmostEqual(GridYawConverter(BASIS)(math.pi / 2., -60., -130.), self.manager.vehicle_state.yaw)


if __name__ == "__main__":
    unittest.main()
