# 실제 경로·Expert 설정을 이용한 신호 mission의 오프라인 회귀 검증.
import copy
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ..adapters.expert_controller import ExpertController
from ..main_runner import load_configuration
from ..tools.verify_traffic_light_expert import verify_traffic_light_expert


class TrafficLightExpertVerificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()):
            cls.global_cfg, _, cls.scenario_cfg = load_configuration()

    def test_actual_export_phase_matrix_and_gradual_stop_without_process(self):
        with patch.object(ExpertController, "start_process", side_effect=AssertionError("검증에서 프로세스를 시작할 수 없습니다.")):
            result = verify_traffic_light_expert(self.global_cfg, self.scenario_cfg)
        self.assertTrue(result["passed"])
        self.assertTrue(result["source_files_unchanged"])
        self.assertEqual((1031, 1348), (result["global_start_index"], result["global_end_index"]))
        self.assertEqual(318, result["mission_points"])
        self.assertEqual(389, result["expert_points"])
        self.assertAlmostEqual(155.2933490661302, result["mission_length_m"], places=6)
        self.assertGreaterEqual(result["expert_tail_m"], 35.0)
        targets = result["route_signal_targets"]["targets"]
        self.assertEqual([("C1256W000123", 92), ("C1256W000052", 214)],
                         [(row["signal_id"], row["path_index"]) for row in targets])
        self.assertEqual([], result["runtime_speed_zones"])
        self.assertEqual({32, 33, 36, 48}, {row["raw_signal_bits"] for row in result["phase_matrix"] if row["left_permitted"]})
        self.assertTrue(all(row["velocity_cap_mps"] == 0 for row in result["unavailable_signal_cases"]))
        self.assertGreater(result["red_approach"][0]["velocity_cap_mps"], 0.0)
        self.assertEqual(0, result["red_approach"][0]["local_path_index"])
        self.assertAlmostEqual(105.05766429524465,
                               result["red_approach"][0]["remaining_route_m"], places=6)
        self.assertEqual(0.0, result["red_approach"][-1]["velocity_cap_mps"])
        self.assertFalse(result["expert_process_started"])
        self.assertFalse(result["ros_connected"])
        self.assertFalse(result["morai_connected"])

    def test_call_restores_environment_modules_and_input_configs(self):
        global_before, scenario_before = copy.deepcopy(self.global_cfg), copy.deepcopy(self.scenario_cfg)
        path_before = list(sys.path)
        modules_before = {key: value for key, value in sys.modules.items()
                          if key == "expert" or key.startswith("expert.")}
        with patch.dict(os.environ, {"GT_BEV_CONFIG_PATH": "검증전_설정",
                                     "GT_BEV_PATH_CSV": "검증전_경로"}):
            environment_before = dict(os.environ)
            verify_traffic_light_expert(self.global_cfg, self.scenario_cfg)
            self.assertEqual(environment_before, dict(os.environ))
        self.assertEqual(path_before, sys.path)
        self.assertEqual(modules_before, {key: value for key, value in sys.modules.items()
                                         if key == "expert" or key.startswith("expert.")})
        self.assertEqual(global_before, self.global_cfg)
        self.assertEqual(scenario_before, self.scenario_cfg)

    def test_changed_route_geometry_is_rejected_before_signal_validation(self):
        cfg = copy.deepcopy(self.scenario_cfg)
        points = Path(cfg["global_path"]).read_text(encoding="utf-8-sig").splitlines()
        # 승인된 START/END는 유지하고 crop 내부의 원본 포인트 하나만 바꾼다.
        index = 1200
        point_rows = [i for i, row in enumerate(points) if row.strip() and not row.lstrip().startswith("#")]
        line = point_rows[index]
        coordinates = points[line].split()
        coordinates[0] = str(float(coordinates[0]) + 0.05)
        points[line] = " ".join(coordinates)
        with tempfile.TemporaryDirectory(prefix="morai-traffic-light-route-test-", dir="/tmp") as temporary:
            changed = Path(temporary) / "changed_route.txt"
            changed.write_text("\n".join(points) + "\n", encoding="utf-8")
            cfg["global_path"] = str(changed)
            with self.assertRaisesRegex(ValueError, "unique exact slice"):
                verify_traffic_light_expert(self.global_cfg, cfg)

    def test_missing_signal_mode_speed_zone_filter_is_rejected(self):
        from ..scenarios.segment_modes import expert as mission_expert
        without_mode = mission_expert.FILTERED_MISSIONS - {"full_loop_traffic_light"}
        with patch.object(mission_expert, "FILTERED_MISSIONS", without_mode):
            with self.assertRaisesRegex(ValueError, "speed-zone 필터"):
                verify_traffic_light_expert(self.global_cfg, self.scenario_cfg)


# 공식 경로에서 신호 단계·실제 정지·원상복구의 실패 조건을 검증한다.
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
        # B's independent consumer remains the oracle; it is verification data,
        # not an external collector installed as part of AIM2026.
        import runpy
        _mission_checks = runpy.run_path(str(Path(__file__).resolve().parent /
            "fixtures/external_consumers/traffic_light_collection.py.txt"))["_mission_checks"]
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


from ..scenarios.full_loop_parts import traffic_light_override


def make_context(color=1):
    context = SimpleNamespace(
        cfg={
            "traffic_light_wait_override": {
                "enabled": True,
                "wait_wall_sec": 10.0,
                "stopped_speed_mps": 0.5,
                "minimum_stopline_distance_m": 0.0,
                "maximum_stopline_distance_m": 25.0,
                "query_period_wall_sec": 0.25,
                "retry_period_wall_sec": 1.0,
                "verification_timeout_wall_sec": 2.0,
                "target_color": 16,
                "impulse": True,
                "set_sibling": True,
            }
        },
        grpc=Mock(),
        event=Mock(),
    )
    traffic_light_override.initialize(context)
    context.traffic_light_stops = [{"light_id": "C-test", "route_s": 100.0}]
    context.grpc.get_traffic_light_state_by_id.return_value = {
        "id": "C-test",
        "color": color,
    }
    context.grpc.set_traffic_light_color_verified.return_value = {
        "id": "C-test",
        "target_color": 16,
        "accepted": True,
        "verified": False,
        "before": {"id": "C-test", "color": 1},
        "after": {"id": "C-test", "color": 1},
        "error": None,
    }
    return context


class TrafficLightOverrideTests(unittest.TestCase):
    def test_protected_left_waits_on_straight_green_then_requests_left_green(self):
        context = make_context(color=16)
        context.traffic_light_stops[0].update(
            signal_movement="left", signal_route_link_id="LEFT-EXIT",
        )
        context.grpc.get_traffic_light_state_by_id.side_effect = [
            {"id": "C-test", "color": 16},
            {"id": "C-test", "color": 16},
            {"id": "C-test", "color": 33},
        ]
        with patch.object(traffic_light_override.time, "monotonic", side_effect=[100, 111, 112]):
            traffic_light_override.update(context, 0, {"speed": 0}, 90)
            traffic_light_override.update(context, 11, {"speed": 0}, 90)
            self.assertEqual(context.traffic_override_verified, 0)
            traffic_light_override.update(context, 12, {"speed": 1}, 91)
        context.grpc.set_traffic_light_color_verified.assert_called_once_with(
            "C-test", 32, impulse=True, set_sibling=True,
        )
        self.assertEqual(context.traffic_override_verified, 1)
        verified = context.event.call_args_list[-1]
        self.assertEqual(verified.kwargs["signal_movement"], "left")
        self.assertEqual(verified.kwargs["verification_semantics"], "protected_left_green_bit")

    def test_same_signal_phase_depends_on_ordered_route_movement(self):
        for movement, color, expected in (
            ("left", 1, 32), ("left", 16, 32),
            ("left", 32, None), ("left", 33, None), ("left", 48, None),
            ("straight", 1, 16), ("straight", 32, 16), ("straight", 33, 16),
            ("straight", 16, None), ("straight", 48, None),
        ):
            with self.subTest(movement=movement, color=color):
                context = make_context(color=color)
                context.traffic_light_stops[0]["signal_movement"] = movement
                with patch.object(traffic_light_override.time, "monotonic", side_effect=[100, 111]):
                    traffic_light_override.update(context, 0, {"speed": 0}, 90)
                    traffic_light_override.update(context, 11, {"speed": 0}, 90)
                if expected is None:
                    context.grpc.set_traffic_light_color_verified.assert_not_called()
                else:
                    self.assertEqual(context.grpc.set_traffic_light_color_verified.call_args.args,
                                     ("C-test", expected))

    def test_unknown_route_movement_or_signal_color_never_forces_straight_release(self):
        for movement, color in ((None, 1), ("right_unprotected", 1), ("left", 0), ("left", 17)):
            with self.subTest(movement=movement, color=color):
                context = make_context(color=color)
                context.traffic_light_stops[0]["signal_movement"] = movement
                with patch.object(traffic_light_override.time, "monotonic", side_effect=[100, 111]):
                    traffic_light_override.update(context, 0, {"speed": 0}, 90)
                    traffic_light_override.update(context, 11, {"speed": 0}, 90)
                context.grpc.set_traffic_light_color_verified.assert_not_called()

    def test_verified_flag_needs_correct_exact_id_movement_readback(self):
        for after in ({"id": "OTHER", "color": 32}, {"id": "C-test", "color": 16},
                      {"id": "C-test", "color": None}, None):
            with self.subTest(after=after):
                context = make_context(color=16)
                context.traffic_light_stops[0]["signal_movement"] = "left"
                result = context.grpc.set_traffic_light_color_verified.return_value
                result.update(verified=True, after=after)
                with patch.object(traffic_light_override.time, "monotonic", side_effect=[100, 111]):
                    traffic_light_override.update(context, 0, {"speed": 0}, 90)
                    traffic_light_override.update(context, 11, {"speed": 0}, 90)
                self.assertEqual(context.traffic_override_verified, 0)
                self.assertEqual(context.traffic_override_completed, set())

    def test_verified_result_for_different_requested_color_is_not_used(self):
        context = make_context(color=16)
        context.traffic_light_stops[0]["signal_movement"] = "left"
        context.grpc.set_traffic_light_color_verified.return_value.update(
            verified=True, after={"id": "C-test", "color": 32}, target_color=16,
        )
        with patch.object(traffic_light_override.time, "monotonic", side_effect=[100, 111]):
            traffic_light_override.update(context, 0, {"speed": 0}, 90)
            traffic_light_override.update(context, 11, {"speed": 0}, 90)
        self.assertEqual(context.traffic_override_verified, 0)

    def test_exact_left_readback_can_verify_immediately(self):
        context = make_context(color=16)
        context.traffic_light_stops[0]["signal_movement"] = "left"
        context.grpc.set_traffic_light_color_verified.return_value.update(
            verified=True, after={"id": "C-test", "color": 32}, target_color=32,
        )
        with patch.object(traffic_light_override.time, "monotonic", side_effect=[100, 111]):
            traffic_light_override.update(context, 0, {"speed": 0}, 90)
            traffic_light_override.update(context, 11, {"speed": 0}, 90)
        self.assertEqual(context.traffic_override_verified, 1)
        self.assertIsNone(context.traffic_wait_state)

    def test_dynamic_releases_repeated_red_just_past_stop_projection(self):
        context = make_context()
        context.dynamic_obstacle_events = [{"id": "active"}]
        context.traffic_override_completed.add(("C-test", 100.0))
        with patch.object(traffic_light_override.time, "monotonic", side_effect=[100.0, 110.1]):
            traffic_light_override.update(context, 0.0, {"speed": 0.0}, 100.08)
            traffic_light_override.update(context, 10.1, {"speed": 0.0}, 100.08)
        context.grpc.set_traffic_light_color_verified.assert_called_once()
        self.assertEqual(context.dynamic_signal_attempts[("C-test", 100.0)], 1)

    def test_dynamic_recovery_does_not_target_light_after_passing_junction(self):
        context = make_context()
        context.dynamic_obstacle_events = [{"id": "active"}]
        with patch.object(traffic_light_override.time, "monotonic", return_value=100.0):
            traffic_light_override.update(context, 0.0, {"speed": 0.0}, 103.0)
        context.grpc.get_traffic_light_state_by_id.assert_not_called()

    def test_other_modes_keep_legacy_completed_stop_behavior(self):
        context = make_context()
        context.traffic_override_completed.add(("C-test", 100.0))
        with patch.object(traffic_light_override.time, "monotonic", return_value=100.0):
            traffic_light_override.update(context, 0.0, {"speed": 0.0}, 99.0)
        context.grpc.get_traffic_light_state_by_id.assert_not_called()

    def test_dynamic_signal_recovery_retries_are_bounded(self):
        context = make_context()
        context.dynamic_obstacle_events = [{"id": "active"}]
        context.dynamic_signal_attempts[("C-test", 100.0)] = 3
        with patch.object(traffic_light_override.time, "monotonic", side_effect=[100.0, 110.1]):
            traffic_light_override.update(context, 0.0, {"speed": 0.0}, 100.08)
            with self.assertRaisesRegex(RuntimeError, "signal recovery exhausted"):
                traffic_light_override.update(context, 10.1, {"speed": 0.0}, 100.08)
        context.grpc.set_traffic_light_color_verified.assert_not_called()

    def test_dynamic_signal_recovery_uses_configured_attempt_limit(self):
        context = make_context()
        context.dynamic_obstacle_events = [{"id": "active"}]
        context.cfg["traffic_light_wait_override"]["dynamic_maximum_attempts"] = 40
        context.dynamic_signal_attempts[("C-test", 100.0)] = 3
        with patch.object(traffic_light_override.time, "monotonic", side_effect=[100.0, 110.1]):
            traffic_light_override.update(context, 0.0, {"speed": 0.0}, 100.08)
            traffic_light_override.update(context, 10.1, {"speed": 0.0}, 100.08)
        context.grpc.set_traffic_light_color_verified.assert_called_once()
        self.assertEqual(context.dynamic_signal_attempts[("C-test", 100.0)], 4)

    def test_signal_rpc_error_does_not_abort_or_command_green(self):
        context = make_context()
        context.grpc.get_traffic_light_state_by_id.side_effect = RuntimeError("RPC unavailable")
        with patch.object(traffic_light_override.time, "monotonic", return_value=100.0):
            traffic_light_override.update(context, 0.0, {"speed": 0.0}, 99.0)
        context.grpc.set_traffic_light_color_verified.assert_not_called()

    def test_red_changes_only_after_ten_host_monotonic_seconds(self):
        context = make_context()
        red = {"id": "C-test", "color": 1}
        green = {"id": "C-test", "color": 16}
        context.grpc.get_traffic_light_state_by_id.side_effect = [
            red,
            red,
            red,
            green,
        ]
        with patch.object(
            traffic_light_override.time,
            "monotonic",
            side_effect=[100.0, 109.9, 110.16, 110.42],
        ):
            traffic_light_override.update(context, 20.0, {"speed": 0.0}, 90.0)
            traffic_light_override.update(context, 21.0, {"speed": 0.0}, 90.0)
            context.grpc.set_traffic_light_color_verified.assert_not_called()
            traffic_light_override.update(context, 22.0, {"speed": 0.0}, 90.0)
            self.assertEqual(context.traffic_override_verified, 0)
            traffic_light_override.update(context, 22.1, {"speed": 1.0}, 91.0)

        context.grpc.set_traffic_light_color_verified.assert_called_once_with(
            "C-test", 16, impulse=True, set_sibling=True
        )
        self.assertEqual(context.traffic_override_attempts, 1)
        self.assertEqual(context.traffic_override_verified, 1)
        self.assertIn(("C-test", 100.0), context.traffic_override_completed)
        names = [call.args[0] for call in context.event.call_args_list]
        self.assertEqual(
            names,
            [
                "traffic_wait_started",
                "traffic_light_override",
                "traffic_light_override_verified",
            ],
        )
        for call in context.event.call_args_list[1:]:
            self.assertFalse(call.kwargs["training_visible"])
            self.assertTrue(call.kwargs["intervention"])

    def test_moving_or_green_ego_never_requests_override(self):
        moving = make_context(color=1)
        green = make_context(color=16)
        with patch.object(
            traffic_light_override.time,
            "monotonic",
            side_effect=[100.0, 100.0],
        ):
            traffic_light_override.update(moving, 20.0, {"speed": 1.0}, 90.0)
            traffic_light_override.update(green, 20.0, {"speed": 0.0}, 90.0)
        moving.grpc.set_traffic_light_color_verified.assert_not_called()
        green.grpc.set_traffic_light_color_verified.assert_not_called()
        self.assertIsNone(moving.traffic_wait_state)
        self.assertIsNone(green.traffic_wait_state)

    def test_missing_speed_fails_closed(self):
        context = make_context(color=1)
        with patch.object(traffic_light_override.time, "monotonic", return_value=100.0):
            traffic_light_override.update(context, 20.0, {}, 90.0)
        context.grpc.get_traffic_light_state_by_id.assert_not_called()
        context.grpc.set_traffic_light_color_verified.assert_not_called()

    def test_morai_bitmask_semantics(self):
        self.assertFalse(traffic_light_override.straight_movement_is_blocked(16))
        self.assertFalse(traffic_light_override.straight_movement_is_blocked(48))
        self.assertTrue(traffic_light_override.straight_movement_is_blocked(1))
        self.assertTrue(traffic_light_override.straight_movement_is_blocked(33))
        self.assertTrue(traffic_light_override.target_is_satisfied(16, 16))
        self.assertTrue(traffic_light_override.target_is_satisfied(48, 16))
        self.assertFalse(traffic_light_override.target_is_satisfied(1, 16))
        self.assertFalse(traffic_light_override.target_is_satisfied(33, 16))

    def test_prepare_binds_exact_stop_light_and_active_route_window(self):
        links = {
            "L1": {"points": [(0, 0, 0), (10, 0, 0)], "to_node_idx": "N1"},
            "L2": {"points": [(10, 0, 0), (20, 0, 0)], "to_node_idx": "N2"},
        }
        nodes = {
            "N1": {"on_stop_line": True, "traffic_light_id": "C-test"},
            "N2": {"on_stop_line": False},
        }
        context = SimpleNamespace(
            cfg={"traffic_light_wait_override": {"enabled": True}},
            route_definition={"route_links": ["L1", "L2"]},
            map_loader=SimpleNamespace(
                get_link=lambda link_id: links[link_id],
                node_set=nodes,
            ),
            route_points=[(0, 0, 0), (10, 0, 0), (20, 0, 0)],
            route_length_m=20.0,
        )
        traffic_light_override.initialize(context)
        traffic_light_override.prepare(context)
        self.assertEqual(len(context.traffic_light_stops), 1)
        self.assertEqual(context.traffic_light_stops[0]["light_id"], "C-test")
        self.assertEqual(context.traffic_light_stops[0]["route_s"], 10.0)

        context.cfg['traffic_light_wait_override']['enabled'] = False
        context.cfg['dynamic_obstacle'] = {'pre_signal_guard': {'enabled': True}}
        context.grpc = Mock()
        traffic_light_override.prepare(context)
        self.assertEqual(context.traffic_light_stops[0]['light_id'], 'C-test')
        traffic_light_override.update(context, 0.0, {}, 0.0)
        self.assertEqual(context.grpc.mock_calls, [])

        context._full_route_points = list(context.route_points)
        context._route_prefix_s = 10.0
        context.route_points = context.route_points[1:]
        context.route_length_m = 10.0
        traffic_light_override.prepare(context)
        self.assertEqual(context.traffic_light_stops[0]["route_s"], 0.0)

    def test_prepare_uses_connected_following_link_and_does_not_guess_by_geometry(self):
        links = {
            "APPROACH": {"points": [(0, 0), (10, 0)], "to_node_idx": "STOP",
                         "related_signal": "straight"},
            "EXIT": {"points": [(10, 0), (20, 0)], "from_node_idx": "STOP",
                     "to_node_idx": "END", "related_signal": "left"},
        }
        context = SimpleNamespace(
            cfg={"traffic_light_wait_override": {"enabled": True}},
            route_definition={"route_links": ["APPROACH", "EXIT"]},
            map_loader=SimpleNamespace(get_link=lambda i: links[i], node_set={
                "STOP": {"on_stop_line": True, "traffic_light_id": "C-test"},
                "END": {"on_stop_line": False},
            }),
            route_points=[(0, 0), (10, 0), (20, 0)], route_length_m=20,
        )
        traffic_light_override.initialize(context)
        traffic_light_override.prepare(context)
        stop = context.traffic_light_stops[0]
        self.assertEqual(stop["signal_movement"], "left")
        self.assertEqual(stop["signal_route_link_id"], "EXIT")
        self.assertEqual(stop["signal_movement_binding"], "connected_ordered_route_link")
        links["EXIT"]["related_signal"] = "straight"
        traffic_light_override.prepare(context)
        self.assertEqual(context.traffic_light_stops[0]["signal_movement"], "straight")
        links["EXIT"]["from_node_idx"] = "OTHER-NODE"
        traffic_light_override.prepare(context)
        stop = context.traffic_light_stops[0]
        self.assertIsNone(stop["signal_movement"])
        self.assertEqual(stop["signal_movement_binding"], "following_link_not_connected")


if __name__ == "__main__":
    unittest.main()
