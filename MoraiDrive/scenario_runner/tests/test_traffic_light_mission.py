"""공식 경로에서 신호 단계·실제 정지·원상복구의 실패 조건을 검증한다."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ..main_runner import load_configuration
from ..scenarios.segment_modes import traffic_light as mission
from ..scenarios.segment_modes.modes import get_mission_mode
from ..scenarios.segment_modes.route import RouteWindow
from ..utils.map_loader import MGeoMapLoader
from ..utils.transform_utils import polyline_length


class TrafficLightMissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()):
            cls.global_cfg, _, cls.base_cfg = load_configuration()
        cls.full = tuple(tuple(map(float, row.split())) for row in Path(cls.base_cfg["global_path"]).read_text().splitlines()
                         if row.strip() and not row.lstrip().startswith("#"))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(temporary.cleanup)
        global_cfg, cfg = copy.deepcopy(self.global_cfg), copy.deepcopy(self.base_cfg)
        global_cfg["paths"]["runtime_root"] = temporary.name
        mission.configure(cfg)
        self.approach_s = polyline_length(self.full[1031:1220])
        self.approach_seconds = self.approach_s / 5.0 + 0.5
        self.now, self.color, self.readback_color = 100.0 - self.approach_seconds, 16, None
        self.wall = 500.0
        mocked_time = patch.object(mission, "time", SimpleNamespace(
            monotonic=lambda: self.wall, sleep=self.wall_sleep))
        mocked_time.start()
        self.addCleanup(mocked_time.stop)
        grpc = Mock()
        grpc.get_traffic_light_state_by_id.side_effect = lambda identity: dict(id=identity, color=self.color)
        def command(identity, color, **kwargs):
            previous = self.color
            self.color = color if self.readback_color is None else self.readback_color
            return dict(id=identity, accepted=True, verified=self.color == color,
                        before=dict(id=identity, color=previous), after=dict(id=identity, color=self.color))
        grpc.set_traffic_light_color_verified.side_effect = command
        grpc.get_ego_motion_state.return_value = dict(x=self.full[1031][0], y=self.full[1031][1], speed=0.)
        points = list(self.full[1031:1349])
        self.context = SimpleNamespace(
            global_cfg=global_cfg, cfg=cfg, mission_mode=get_mission_mode(mission.MODE),
            mission_route=RouteWindow(self.full, polyline_length(self.full[:1032]), 1031, 1348),
            route_points=points, route_length_m=polyline_length(points), run_instance_id="test-signal",
            clock=SimpleNamespace(domain="grpc_simulation_timestamp", now=lambda: self.now),
            map_loader=MGeoMapLoader(global_cfg["paths"]["mgeo_root"]), grpc=grpc,
            stop_drive_control=Mock(),
        )
        self.inventory = patch.object(mission, "_inventory", side_effect=self.inventory_read)
        self.inventory.start()
        self.addCleanup(self.inventory.stop)
        mission.prepare(self.context)
        mission.before_drive(self.context)
        report = self.context.traffic_light_report
        report.update(expert_runtime_verified=True, start_pose_verified=True,
                      startup_observation=dict(scenario_time=self.now, route_s=0.,
                                               ego_xy=list(self.full[1031][:2]), speed_mps=0.,
                                               distance_to_start_m=0.))
        # Drive the real mapped polyline at 5m/s before entering the stop fixture.
        for index in range(1031, 1220):
            progress = polyline_length(self.full[1031:index + 1])
            self.tick(-self.approach_seconds + progress / 5., index=index, speed_mps=5.)

    def inventory_read(self, context):
        context.traffic_light_report["actor_inventory_checks"] += 1
        context.traffic_light_report["empty_road_verified"] = True

    def wall_sleep(self, seconds):
        self.wall += seconds

    def tick(self, when, index=1219, speed_mps=0.0, **fields):
        self.wall += max(0., (100.0 + when) - self.now)
        self.now = 100.0 + when
        position = self.full[index]
        state = dict(x=position[0], y=position[1], speed=speed_mps * 3.6, collision_objects=[])
        state.update(fields)
        progress = polyline_length(self.full[1031:index + 1])
        mission.observe(self.context, when + self.approach_seconds, state, progress)

    def reset_at_start(self):
        report = self.context.traffic_light_report
        report["observations"] = []
        report["right_turn_completed"] = False
        report["startup_observation"].update(scenario_time=100., route_s=0., speed_mps=0.,
                                              ego_xy=list(self.full[1031][:2]))
        self.context.traffic_light_last_check = None

    def red_hold(self):
        for step in range(13):
            self.tick(step * 0.25)
        self.assertEqual("straight_pending", self.context.traffic_light_report["phase"])
        for when in (3.25, 3.5, 3.75):
            self.tick(when)
        self.assertEqual("straight_hold", self.context.traffic_light_report["phase"])

    def straight_hold(self):
        self.red_hold()
        for step in range(17):
            self.tick(4.0 + step * 0.25)
        self.assertEqual("arrow_pending", self.context.traffic_light_report["phase"])
        for when in (8.25, 8.5, 8.75):
            self.tick(when)
        self.assertEqual("left_go", self.context.traffic_light_report["phase"])

    def completed(self):
        self.straight_hold()
        for index in range(1219, 1299):
            progress = polyline_length(self.full[1219:index + 1])
            self.tick(9.0 + progress / 5., index=index, speed_mps=5.)
        self.tick(9.25 + progress / 5., index=1298, speed_mps=2.)
        mission.finish(self.context, "passed")

    def stop_read(self, context):
        context.traffic_light_report["final_stop"] = dict(final_verified=True, velocity_reset_used=False,
                                                         policy="common_auto_d_ros_brake")

    def test_exact_route_window_and_color_permission_are_separate(self):
        report = self.context.traffic_light_report
        self.assertAlmostEqual(155.2933490847876, report["route"]["length_m"])
        self.assertAlmostEqual(25.0, report["route"]["after_left_exit_m"], places=3)
        self.assertEqual("grpc_simulation_timestamp", report["clock_domain"])
        self.assertEqual("pure_red_green_v1", report["contract"]["color_head_target_policy"])
        self.assertFalse(report["physical_simultaneity_verified"])

    def test_controlled_phases_include_actual_full_holds_and_departure(self):
        self.completed()
        with patch.object(mission, "_stop_before_restore", side_effect=self.stop_read):
            mission.restore(self.context)
        report = self.context.traffic_light_report
        self.assertEqual("passed", report["status"])
        self.assertEqual([1, 16, 33], [row["target_color"] for row in report["commands"]])
        for flag in ("red_stop_verified", "straight_green_stop_verified", "left_arrow_departure_verified", "left_turn_completed"):
            self.assertTrue(report[flag], flag)
        self.assertEqual(16, self.color)
        self.assertTrue(report["restore"]["verified"])
        self.assertTrue(all(call.kwargs["set_sibling"] is False
                            for call in self.context.grpc.set_traffic_light_color_verified.call_args_list))

    def test_produced_report_satisfies_independent_collection_behavior_checks(self):
        self.completed()
        with patch.object(mission, "_stop_before_restore", side_effect=self.stop_read):
            mission.restore(self.context)
        from data_collection.internal_sensor_record.async_capture.traffic_light_collection import _mission_checks
        report = self.context.traffic_light_report
        checks = _mission_checks(report, report["contract"])
        self.assertTrue(all(checks.values()), checks)

    def test_red_command_ack_without_actual_stop_cannot_release(self):
        self.reset_at_start()
        for step in range(25):
            self.tick(step * 0.25, index=1031, speed_mps=2.0)
        report = self.context.traffic_light_report
        self.assertFalse(report["red_stop_verified"])
        self.assertEqual("red_approach", report["phase"])
        self.assertEqual([1], [row["target_color"] for row in report["commands"]])
        with self.assertRaises(RuntimeError):
            mission.finish(self.context, "passed")

    def test_stationary_far_before_stop_is_not_a_signal_stop(self):
        self.reset_at_start()
        for step in range(25):
            self.tick(step * 0.25, index=1031, speed_mps=0.0)
        self.assertFalse(self.context.traffic_light_report["red_stop_verified"])
        self.assertEqual(1, self.color)

    def test_red_rolling_resets_continuous_hold(self):
        self.tick(0.0)
        self.tick(1.0)
        self.tick(1.25, speed_mps=0.4)
        self.tick(2.0)
        self.tick(3.0)
        self.tick(4.0)
        self.assertEqual(1, self.color)
        self.tick(5.0)
        self.assertEqual(16, self.color)

    def test_straight_green_departure_fails_without_arrow_command(self):
        self.red_hold()
        with self.assertRaisesRegex(RuntimeError, "직진 초록"):
            self.tick(4.0, speed_mps=0.7)
        self.assertEqual([1, 16], [row["target_color"] for row in self.context.traffic_light_report["commands"]])

    def test_unexpected_phase_and_invalid_readback_fail(self):
        self.color = 36
        with self.assertRaisesRegex(RuntimeError, "다른 상태"):
            self.tick(0.0)
        self.readback_color = 36
        with self.assertRaisesRegex(RuntimeError, "시간 초과"):
            mission._command(self.context, 33)
        self.assertFalse(self.context.traffic_light_report["commands"][-1]["verified"])

    def test_wrong_signal_id_is_never_used(self):
        self.context.grpc.get_traffic_light_state_by_id.side_effect = lambda identity: dict(id="다른_신호", color=1)
        with self.assertRaisesRegex(RuntimeError, "정확한 좌회전"):
            self.tick(0.0)

    def test_arrow_readback_may_be_canonical_left_phase_but_never_yellow(self):
        for color in (32, 33, 48):
            self.readback_color = color
            mission._command(self.context, 33)
            self.assertTrue(self.context.traffic_light_report["commands"][-1]["verified"])

    def test_departure_and_exit_both_required(self):
        self.straight_hold()
        self.tick(9.0, index=1219, speed_mps=0.0)
        self.assertFalse(self.context.traffic_light_report["left_turn_completed"])
        with self.assertRaises(RuntimeError):
            mission.finish(self.context, "passed")

    def test_clock_gap_and_reverse_fail_instead_of_extending_hold(self):
        self.tick(0.0)
        with self.assertRaisesRegex(RuntimeError, "끊겼"):
            self.tick(2.0)
        with self.assertRaisesRegex(RuntimeError, "역행"):
            self.tick(-0.1)

    def test_teleport_cannot_prove_left_turn_completion(self):
        self.straight_hold()
        with self.assertRaisesRegex(RuntimeError, "위치 이동"):
            self.tick(9.0, index=1297, speed_mps=2.0)
        self.assertFalse(self.context.traffic_light_report["left_turn_completed"])

    def test_collision_and_nonfinite_observations_fail(self):
        with self.assertRaisesRegex(RuntimeError, "충돌"):
            self.tick(0.0, collision_objects=["actor"])
        self.assertEqual(1, self.context.traffic_light_report["collision_count"])
        with self.assertRaisesRegex(RuntimeError, "유효하지"):
            self.tick(0.0, speed=float("nan"))

    def test_restore_runs_even_if_final_physical_stop_fails(self):
        self.completed()
        with patch.object(mission, "_stop_before_restore", side_effect=RuntimeError("정지 실패")):
            with self.assertRaisesRegex(RuntimeError, "정지 실패"):
                mission.restore(self.context)
        self.assertEqual(16, self.color)
        self.assertTrue(self.context.traffic_light_report["restore"]["verified"])
        self.assertEqual("failed", self.context.traffic_light_report["status"])

    def test_restore_readback_mismatch_fails(self):
        self.completed()
        self.readback_color = 1
        with patch.object(mission, "_stop_before_restore", side_effect=self.stop_read):
            with self.assertRaisesRegex(RuntimeError, "시간 초과"):
                mission.restore(self.context)
        self.assertFalse(self.context.traffic_light_report["restore"]["verified"])

    def test_collector_augmented_contract_is_preserved(self):
        cfg = copy.deepcopy(self.context.cfg)
        cfg["_traffic_light_collection"]["source_sha256"]["extra/source.py"] = "a" * 64
        mission.configure(cfg)
        self.assertEqual("a" * 64, cfg["_traffic_light_collection"]["source_sha256"]["extra/source.py"])

    def test_saved_report_has_failure_not_false_pass(self):
        mission.finish(self.context, "failed", "수신 실패")
        saved = json.loads(Path(self.context.traffic_light_report["report_path"]).read_text())
        self.assertEqual("failed", saved["status"])
        self.assertEqual("수신 실패", saved["failure_reason"])
        with self.assertRaisesRegex(RuntimeError, "수신 실패"):
            mission.finish(self.context, "passed")

    def test_foreign_actor_in_path_is_rejected_without_deletion(self):
        point = self.full[1219]
        xyz = lambda values: SimpleNamespace(x=values[0], y=values[1], z=values[2])
        actor = SimpleNamespace(actor_info=SimpleNamespace(id=SimpleNamespace(value="foreign")),
                                bounding_box=SimpleNamespace(location=xyz(point), extent=xyz((1., 1., 1.))))
        self.inventory.stop()
        with patch.object(mission.tunnel_baseline, "_inventory", return_value=([actor], "Ego")):
            with self.assertRaisesRegex(RuntimeError, "다른 actor"):
                mission._inventory(self.context)
        report = self.context.traffic_light_report
        self.assertFalse(report["empty_road_verified"])
        self.assertEqual(1, report["maximum_blocking_actors"])
        self.context.grpc.destroy_actor.assert_not_called()

    def test_actual_runtime_binds_crop_and_rejects_wrong_private_path(self):
        import yaml
        from ..adapters.expert_controller import ExpertController
        from ..scenarios.segment_modes import expert, route
        context = self.context
        context.route_definition = yaml.safe_load(Path(context.cfg["route_definition_path"]).read_text())["full_loop"]
        controller = ExpertController(
            expert_root=context.global_cfg["paths"]["expert_root"],
            runtime_root=context.global_cfg["paths"]["runtime_root"],
            map_name=context.cfg["expert_map_name"], mgeo_root=context.global_cfg["paths"]["mgeo_root"],
            route_points=route.build_expert_route_points(context),
            route_signal_route_links=context.route_definition["route_links"],
            start_link=context.mission_mode.start.link_id, end_link=context.mission_mode.end.link_id,
            max_velocity_kmh=30., is_closed_path=False)
        context.expert_controller = controller
        expert.prepare_speed_zones(context)
        mission.verify_runtime(context)
        self.assertTrue(context.traffic_light_report["expert_runtime_verified"])
        self.assertIsNone(controller.process)
        csv_path = Path(controller.runtime_path_csv)
        rows = csv_path.read_text().splitlines()
        rows[10] = "0,0"
        csv_path.write_text("\n".join(rows) + "\n")
        with self.assertRaisesRegex(RuntimeError, "lookahead"):
            mission.verify_runtime(context)

    def test_start_pose_is_measured_after_capture_readiness_and_before_expert_launch(self):
        from ..scenarios.segment_modes.scenario import MissionScenario
        scenario = MissionScenario.__new__(MissionScenario)
        scenario.cfg = dict(drive_control_mode="aimdrive_expert")
        scenario.mission_mode = get_mission_mode(mission.MODE)
        scenario.grpc = Mock()
        scenario.grpc.set_ego_control_mode_auto.return_value = True
        scenario.expert_controller = Mock()
        order = []
        scenario.setup_expert_controller = Mock(side_effect=lambda: order.append("capture_ready"))
        scenario.expert_controller.start_process.side_effect = lambda: order.append("expert_launch")
        with patch.object(mission, "record_start", side_effect=lambda ctx: order.append("actual_start")):
            scenario.activate_expert_drive(after_expert_setup=lambda: order.append("existing_setup_hook"))
        self.assertEqual(["capture_ready", "existing_setup_hook", "actual_start", "expert_launch"], order)

    def test_real_start_observation_rejects_motion_or_wrong_location(self):
        self.context.traffic_light_report["observations"] = []
        mission.record_start(self.context)
        self.assertTrue(self.context.traffic_light_report["start_pose_verified"])
        self.context.grpc.get_ego_motion_state.return_value["speed"] = 3.6
        with self.assertRaisesRegex(RuntimeError, "START에서 정지"):
            mission.record_start(self.context)

    def test_delayed_apply_accepts_old_immediate_readback_then_waits_for_real_color(self):
        self.color = 16
        issued = []
        def delayed(identity, color, **kwargs):
            issued.append(self.wall)
            return dict(id=identity, accepted=True, verified=False,
                        before=dict(id=identity, color=16), after=dict(id=identity, color=16))
        self.context.grpc.set_traffic_light_color_verified.side_effect = delayed
        self.context.grpc.get_traffic_light_state_by_id.side_effect = lambda identity: dict(
            id=identity, color=1 if issued and self.wall - issued[0] >= .15 else 16)
        self.context.grpc.set_traffic_light_color_verified.reset_mock()
        record = mission._command(self.context, 1)
        self.assertTrue(record["verified"])
        self.assertEqual(16, record["immediate_after"]["color"])
        self.assertEqual(1, record["after"]["color"])
        self.assertGreaterEqual(record["readback_stable_seconds"], .2 - 1e-8)
        self.assertIn(16, [row["color"] for row in record["readback_observations"]])
        self.context.grpc.set_traffic_light_color_verified.assert_called_once_with(
            mission.TARGET_LIGHT, 1, impulse=False, set_sibling=False)

    def test_ack_and_flickering_color_cannot_pass_and_command_is_not_resent(self):
        issued = []
        def flickering(identity, color, **kwargs):
            issued.append(self.wall)
            return dict(id=identity, accepted=True, verified=False,
                        before=dict(id=identity, color=16), after=dict(id=identity, color=16))
        self.context.grpc.set_traffic_light_color_verified.side_effect = flickering
        self.context.grpc.get_traffic_light_state_by_id.side_effect = lambda identity: dict(
            id=identity, color=1 if int(round((self.wall - issued[0]) / .05)) % 2 else 16)
        self.context.grpc.set_traffic_light_color_verified.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "시간 초과"):
            mission._command(self.context, 1)
        record = self.context.traffic_light_report["commands"][-1]
        self.assertFalse(record["verified"])
        self.assertGreater(len(record["readback_observations"]), 3)
        self.context.grpc.set_traffic_light_color_verified.assert_called_once()

    def test_pending_signal_apply_keeps_observing_ego_without_blocking_ticks(self):
        issued = []
        def delayed(identity, color, **kwargs):
            issued.append(self.wall)
            return dict(id=identity, accepted=True, verified=False,
                        before=dict(id=identity, color=1), after=dict(id=identity, color=1))
        self.context.grpc.set_traffic_light_color_verified.side_effect = delayed
        def read(identity):
            return dict(id=identity, color=16 if issued and self.wall - issued[0] >= .7 else 1)
        self.context.grpc.get_traffic_light_state_by_id.side_effect = read
        for step in range(13):
            self.tick(step * .25)
        self.assertEqual("straight_pending", self.context.traffic_light_report["phase"])
        for when in (3.25, 3.5, 3.75, 4.0, 4.25):
            self.tick(when)
        self.assertEqual("straight_hold", self.context.traffic_light_report["phase"])
        rows = self.context.traffic_light_report["observations"]
        self.assertTrue(all(last["scenario_time"] - first["scenario_time"] <= 1.5
                            for first, last in zip(rows, rows[1:])))
        self.assertEqual(5, sum(row["phase"] == "straight_pending" for row in rows))

    def test_restoration_failure_keeps_original_cause_and_actual_readback(self):
        mission.finish(self.context, "failed", "red 명령 적용 실패")
        self.readback_color = 1
        with patch.object(mission, "_stop_before_restore", side_effect=self.stop_read):
            with self.assertRaisesRegex(RuntimeError, "시간 초과"):
                mission.restore(self.context)
        report = self.context.traffic_light_report
        self.assertEqual("red 명령 적용 실패", report["failure_reason"])
        self.assertIn("시간 초과", report["restoration_error"])
        self.assertEqual(1, report["restore"]["after"]["color"])
        self.assertTrue(report["restore"]["accepted"])
        self.assertTrue(report["restore"]["readback_observations"])

    def test_single_missing_signal_is_recorded_as_missing_and_does_not_fill_hold(self):
        self.tick(0.)
        self.tick(.25)
        self.context.grpc.get_traffic_light_state_by_id.side_effect = lambda identity: None
        self.tick(.5)
        row = self.context.traffic_light_report["observations"][-1]
        self.assertIsNone(row["signal_color"])
        self.assertIsNone(row["signal_id"])
        self.assertTrue(row["signal_read_unavailable"])
        self.assertIsNone(self.context.traffic_light_hold_since)
        self.context.grpc.get_traffic_light_state_by_id.side_effect = lambda identity: dict(id=identity, color=1)
        for when in (.75, 1.0, 2.0, 3.0):
            self.tick(when)
        self.assertFalse(self.context.traffic_light_report["red_stop_verified"])
        self.tick(3.75)
        self.assertTrue(self.context.traffic_light_report["red_stop_verified"])

    def test_continuously_missing_signal_is_bounded_and_cannot_pass(self):
        self.context.grpc.get_traffic_light_state_by_id.side_effect = lambda identity: None
        for when in (0., .25, .5, .75):
            self.tick(when)
        with self.assertRaisesRegex(RuntimeError, "연속 누락"):
            self.tick(1.)

    def test_departure_while_arrow_apply_is_pending_and_still_green_is_rejected(self):
        self.red_hold()
        for step in range(17):
            self.tick(4.0 + step * .25)
        self.assertEqual("arrow_pending", self.context.traffic_light_report["phase"])
        self.color = 16
        with self.assertRaisesRegex(RuntimeError, "실제로 적용되기 전에"):
            self.tick(8.25, speed_mps=1.)


if __name__ == "__main__":
    unittest.main()
