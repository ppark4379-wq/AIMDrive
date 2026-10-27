"""Fail-closed trial registration, borrowed-world and runtime scene contracts."""

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from shapely.geometry import Polygon

from .. import main_runner
from ..adapters.expert_process import configure_tunnel_trial
from ..scenarios.segment_modes import tunnel_static_trial as trial
from ..scenarios.segment_modes.modes import MISSION_MODES, PRIMARY_SCENARIOS, TUNNEL_NORMAL_START, automatic_repeat_enabled
from ..scenarios.segment_modes.scenario import MissionScenario
from .test_tunnel_static_live import state


class TrialTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.boundary = NS(policy={"body": {"rear_m": .85, "front_m": 3.85, "width_m": 2.17},
                                   "approach_m": 120., "departure_m": 70.,
                                   "planning_road_margin_m": .35, "runtime_road_margin_m": .15}, policy_sha256="policy-sha",
            mask_world=lambda xy, yaw, margin: np.ones(len(xy), dtype=bool),
            collision_mask_world=lambda xy, yaw, objects, margin: np.ones(len(xy), dtype=bool),
            corners_world=lambda xy, yaw, margin: np.array([
                trial.footprint_corners([x+1.5*np.cos(angle), y+1.5*np.sin(angle)],
                    2.17+2*margin, 4.7+2*margin, np.degrees(angle)) for (x, y), angle in zip(xy, yaw)]),
            _polygon=Polygon([(-100., -5.), (300., -5.), (300., 10.), (-100., 10.)]))
        self.boundary.covers_geometry = lambda polygon: self.boundary._polygon.covers(polygon)
        self.boundary.outside_area_geometry = lambda polygon: polygon.difference(self.boundary._polygon).area
        self.fixture_state = state("1", 45., 0., model="WoodBox")
        self.fixture_state.vehicle_state.collision_objects = []
        self.ego_state = state("Ego", -40., 0., width=2., length=4.5, model="EgoCar")
        self.ego_state.vehicle_state.collision_objects = []
        observed = trial.live._observation(self.fixture_state)
        observed.update(observed["raw_bbox_normalized"])
        observed.update(actor_kind="obstacle", model="WoodBox")
        self.measured = {"status": "nominal_path_found", "path_found": True,
                         "geometry_checks": {"passed": True}, "road_boundary_policy": self.boundary.policy}
        self.preview = {"observed": [observed], "measured_route_screen": self.measured}
        self.monitor = Mock()
        self.monitor.snapshot.return_value = (None, None)

    def context(self):
        with patch.object(trial.screening, "provenance_matches", return_value=True):
            return trial.TrialContext({"count": 1}, self.preview, {}, self.output, self.boundary, self.monitor)

    def scenario(self, context):
        scenario = NS(grpc=NS(get_ego=lambda: NS(get_id=lambda: "Ego")), route_length_m=190.,
                      route_points=[(-40., 0., 28.), (0., 0., 28.), (120., 0., 28.), (150., 0., 28.)])
        context.geometry = {"own_points": [(0., 0., 28.), (120., 0., 28.)],
                            "opposing_points": [(0., 3.5, 28.), (120., 3.5, 28.)],
                            "own_lane_width_m": 3.5, "opposing_lane_width_m": 3.5}
        context.route_points = scenario.route_points
        context.entry_s, context.exit_s = 40., 160.
        return scenario

    def test_registered_trial_is_distinct_and_single_run_outside_collection_modes(self):
        mode = MISSION_MODES[trial.MODE]
        self.assertEqual(sum(asdict(mode.zones).values()), 1)
        self.assertTrue(mode.zones.tunnel_static)
        self.assertFalse(mode.zones.tunnel)
        self.assertFalse(mode.zones.static)
        self.assertNotIn(trial.MODE, PRIMARY_SCENARIOS)
        self.assertFalse(automatic_repeat_enabled(trial.MODE, {"auto_repeat": True}))
        self.assertEqual(main_runner.build_parser().parse_args(["--scenario", trial.MODE]).scenario, trial.MODE)

    def test_lane_classifier_keeps_one_long_bypass_distinct_from_an_actual_alternating_maneuver(self):
        geometry = {"own_points": [[0., 0., 0.], [120., 0., 0.]],
                    "opposing_points": [[0., 3.5, 0.], [120., 3.5, 0.]], "length_m": 120.}
        one = [[x, 0. if x < 20 or x > 90 else 3.5] for x in range(1, 119)]
        result = trial.classify_lane_maneuver(one, geometry)
        self.assertEqual(result["pattern"], "single_opposing_lane_excursion")
        self.assertFalse(result["s_maneuver_observed"])
        alternating = [[x, 3.5 if 20 <= x <= 40 or 70 <= x <= 90 else 0.] for x in range(1, 119)]
        result = trial.classify_lane_maneuver(alternating, geometry)
        self.assertEqual(result["lane_sequence"], ["own", "opposing", "own", "opposing", "own"])
        self.assertTrue(result["s_maneuver_observed"])
        self.assertEqual(result["completed_opposing_excursions"], 2)

    def test_lane_classifier_hysteresis_rejects_pixel_scale_jitter_and_handles_negative_opposing_side(self):
        geometry = {"own_points": [[0., 0., 0.], [120., 0., 0.]],
                    "opposing_points": [[0., -3.5, 0.], [120., -3.5, 0.]], "length_m": 120.}
        jitter = [[float(x), -1.75+(-.1 if x % 2 else .1)] for x in range(1, 119)]
        self.assertFalse(trial.classify_lane_maneuver(jitter, geometry)["s_maneuver_observed"])
        alternating = [[x, -3.5 if 20 <= x <= 40 or 70 <= x <= 90 else 0.] for x in range(1, 119)]
        self.assertTrue(trial.classify_lane_maneuver(alternating, geometry)["s_maneuver_observed"])

    def test_trial_report_exports_the_exact_active_policy_road_and_owned_conservative_shape(self):
        context = self.context()
        report = json.loads((self.output/"drive_trial.json").read_text())
        self.assertEqual(report["road_boundary_geometry"]["type"], "Polygon")
        self.assertEqual(report["owned_actor_geometry"][0]["width_m"], context.references["1"]["width_m"])
        self.assertEqual(report["road_boundary_policy"], self.boundary.policy)
        self.assertFalse(report["s_avoidance_verified"])
        context.report["s_avoidance_verified"] = True
        context.fail(RuntimeError("guard failed"))
        self.assertFalse(context.report["s_avoidance_verified"])

    def test_main_uses_guarded_dispatch_and_bare_mission_still_requires_an_owned_fixture(self):
        replay_name = "scenario_runner.scenarios.segment_modes.tunnel_static_replay"
        replay = ModuleType(replay_name)
        replay.run_tunnel_static_scenario = Mock(return_value=0)
        global_cfg, scenario_cfg = {"paths": {}}, {}
        with patch.dict(sys.modules, {replay_name: replay}), patch.object(main_runner, "load_configuration",
                return_value=(global_cfg, {"zone": "full_loop"}, scenario_cfg)) as load, patch.object(
                main_runner, "MoraiGrpcClient") as grpc, patch("sys.argv", ["runner", "--scenario", trial.MODE]):
            self.assertEqual(main_runner.main(), 0)
        load.assert_called_once_with()
        replay.run_tunnel_static_scenario.assert_called_once()
        self.assertEqual(replay.run_tunnel_static_scenario.call_args.args[0].scenario, trial.MODE)
        self.assertIs(replay.run_tunnel_static_scenario.call_args.args[1], global_cfg)
        self.assertIs(replay.run_tunnel_static_scenario.call_args.args[2], scenario_cfg)
        grpc.assert_not_called()
        scenario = MissionScenario(None, None, {}, {"mission_mode": trial.MODE})
        with self.assertRaisesRegex(RuntimeError, "owned fixture"):
            scenario.run()

    def test_borrowed_world_does_not_restart_simulation_and_reset_targets_runner_key(self):
        fixture = NS(world=NS(_sim_adapter=NS(destroy_all_actors=Mock(return_value=NS(status=1)))),
                     client_key="tunnel_static_trial_fixture", client=Mock())
        borrowed = trial.BorrowedTrialClient(fixture)
        transform = NS(location=NS(**dict(zip(("x", "y", "z"), TUNNEL_NORMAL_START.xyz))))
        borrowed.start_world(transform)
        self.assertTrue(borrowed.world_reused)
        self.assertIs(borrowed.world, fixture.world)
        fixture.client.start_simulation.assert_not_called()
        with patch.dict(sys.modules, {"proto.morai.common.enum_pb2": NS(STATUS_CODE_SUCCESS=1),
                                     "proto.morai.common.type_pb2": NS(StringValue=lambda: NS(value=""))}):
            self.assertTrue(borrowed.reset_actors())
        query = fixture.world._sim_adapter.destroy_all_actors.call_args.args[0]
        self.assertEqual(query.value, "tunnel_static_trial_runner")
        self.assertNotEqual(query.value, fixture.client_key)
        with self.assertRaisesRegex(RuntimeError, "outer fixture"):
            borrowed.stop()

    def test_stale_screening_contract_is_rejected_before_drive(self):
        with patch.object(trial.screening, "provenance_matches", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "current measured"):
                trial.TrialContext({"count": 1}, self.preview, {}, self.output, self.boundary, self.monitor)

    def test_fixture_pose_movement_fails_even_if_reported_velocity_remains_zero(self):
        context = self.context()
        scenario = self.scenario(context)
        self.fixture_state.transform.location.x += .5
        with patch.object(trial.live, "_inventory", return_value={"Ego": self.ego_state, "1": self.fixture_state}), self.assertRaisesRegex(RuntimeError, "pose moved"):
            context.inspect(scenario, 0.)

    def test_actual_ego_body_road_failure_aborts_before_ros_evidence_or_completion(self):
        context = self.context()
        scenario = self.scenario(context)
        self.boundary.mask_world = lambda xy, yaw, margin: np.zeros(len(xy), dtype=bool)
        with patch.object(trial.live, "_inventory", return_value={"Ego": self.ego_state, "1": self.fixture_state}), patch.object(
                trial.live, "_model", return_value="WoodBox"), self.assertRaisesRegex(RuntimeError, "Actual Ego body left"):
            context.inspect(scenario, 0.)
        self.assertFalse(context.report["goal_reached"])
        self.assertEqual(context.report["last_ego_observation"]["position_xyz"], [-40., 0., 28.])
        self.assertEqual(context.report["last_guard_stage"], "policy_body_road")
        self.assertFalse(context.report["last_geometry_check"]["policy_road_mask"][0])

    def test_raw_bbox_failure_saves_offending_polygon_without_weakening_guard(self):
        context = self.context()
        scenario = self.scenario(context)
        self.ego_state.bounding_box.location.x = 299.
        with patch.object(trial.live, "_inventory", return_value={"Ego": self.ego_state, "1": self.fixture_state}), patch.object(
                trial.live, "_model", return_value="WoodBox"), self.assertRaisesRegex(RuntimeError, "returned Ego bounding box"):
            context.inspect(scenario, .25)
        details = context.report["last_geometry_check"]
        self.assertEqual(context.report["last_guard_stage"], "raw_bbox_road")
        self.assertTrue(details["policy_road_mask"][0])
        self.assertFalse(details["raw_bbox_road_passed"])
        self.assertGreater(details["raw_bbox_outside_area_m2"], 0.)
        self.assertEqual(len(details["raw_bbox_corners_with_margin_world"]), 4)
        self.assertFalse(context.report["expert_verified"])

    def test_ego_tilt_beyond_projection_contract_fails_and_preserves_observation(self):
        context = self.context()
        scenario = self.scenario(context)
        self.ego_state.transform.rotation.x = 356.5
        with patch.object(trial.live, "_inventory", return_value={"Ego": self.ego_state, "1": self.fixture_state}), self.assertRaisesRegex(RuntimeError, "roll/pitch"):
            context.inspect(scenario, .4)
        self.assertEqual(context.report["last_guard_stage"], "ego_roll_pitch")
        self.assertEqual(context.report["last_ego_roll_pitch_abs_deg"], [3.5, 0.])
        self.assertEqual(context.report["last_ego_observation"]["rotation_xyz_deg"][0], 356.5)

    def test_immediate_input_block_and_p_gear_precede_slow_expert_cleanup(self):
        events = []
        def command(label):
            def call(request, timeout):
                self.assertEqual(timeout, .5)
                events.append(label)
                return NS(status=1)
            return call
        stub = NS(SetVehicleControlMode=command("manual"), SetVelocity=command("zero"),
                  SetVehicleGear=command("P"), GetVehicleControlMode=lambda *a, **kw: NS(mode=1),
                  GetActorState=lambda *a, **kw: self.ego_state)
        scenario = MissionScenario.__new__(MissionScenario)
        scenario.cfg = {"mission_mode": trial.MODE}
        scenario.grpc = NS(world=NS(_sim_adapter=NS(_actor_stub=stub)),
                           get_ego=lambda: NS(get_object_info=lambda: NS(id="Ego")))
        class Parameter:
            def __init__(self):
                self.actor_info = Mock()
        modules = {"proto.morai.actor.actor_enum_pb2": NS(VEHICLE_CONTROL_KEYBOARD=1, GEAR_MODE_P=1),
                   "proto.morai.actor.actor_set_pb2": NS(VehicleControlModeParam=Parameter, SetVelocityParam=Parameter, VehicleGear=Parameter),
                   "proto.morai.common.enum_pb2": NS(STATUS_CODE_SUCCESS=1)}
        from ..scenarios.full_loop_scenario import FullLoopScenario
        with patch.dict(sys.modules, modules), patch.object(FullLoopScenario, "cleanup", side_effect=lambda: events.append("wait_for_expert")):
            scenario.cleanup()
        self.assertEqual(events, ["manual", "zero", "P", "wait_for_expert"])

    def test_remaining_tail_updates_do_not_duplicate_the_same_validated_plan(self):
        context = self.context()
        import time
        raw = context.references["1"]
        objects = {"receipt_monotonic": time.monotonic(), "rows": [{"raw_actor_id": "1", "actor_kind": "obstacle",
            "position_xyz": raw["position_xyz"], "heading_deg": 0., "size_xyz": [1.2, 1.2, 1.5]}]}
        diagnostics = {"validated_path_seen": True, "scene_geometry_sha256": "scene",
                       "last_result": {"debug_details": {"static_avoidance": {
                           "plan_count": 1, "selected_routes": ["jamming_01.csv"]}}}}
        (self.output/"planner_diagnostics.json").write_text(json.dumps(diagnostics))
        for points in ([[0., 0.], [1., 0.], [2., 0.], [3., 0.]], [[1., 0.], [2., 0.], [3., 0.]]):
            selected = {"receipt_monotonic": time.monotonic(), "namespace": "candidate_paths_recorded",
                        "points": points, "sha256": hashlib.sha256(json.dumps(points).encode()).hexdigest()}
            self.monitor.snapshot.return_value = (objects, selected)
            context.inspect_ros({"position_xyz": [0., 0., 28.]})
        self.assertEqual(len(context.report["selected_paths"]), 1)
        self.assertEqual(context.report["selected_path_updates"], 2)
        self.assertEqual(context.report["last_selected_path_summary"]["point_count"], 3)

    def test_observed_ros_id_and_size_axes_must_match_exact_fixture(self):
        context = self.context()
        reference = context.references["1"]
        import time
        snapshot = {"receipt_monotonic": time.monotonic(), "rows": [
            {"raw_actor_id": "1", "actor_kind": "obstacle", "position_xyz": reference["position_xyz"],
             "heading_deg": reference["rotation_xyz_deg"][2], "size_xyz": [1.2, 1.2, 1.5]}]}
        self.monitor.snapshot.return_value = (snapshot, None)
        context.inspect_ros({"position_xyz": [-40., 0., 28.]})  # Still outside the 80 m visibility gate.
        context.inspect_ros({"position_xyz": [0., 0., 28.]})
        self.assertEqual(context.report["observed_object_ids"], ["1"])
        snapshot["rows"][0]["raw_actor_id"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "exact trial fixture"):
            context.inspect_ros({"position_xyz": [0., 0., 28.]})

    def vehicle_context(self):
        self.fixture_state = state("1", 45., 0., width=2.11531, length=4.574842, model="2019_Toyota_Prius")
        self.fixture_state.vehicle_state.collision_objects = []
        observed = trial.live._observation(self.fixture_state)
        observed.update(observed["raw_bbox_normalized"])
        observed.update(actor_kind="vehicle", model="2019_Toyota_Prius", width_m=2.13, length_m=4.579)
        self.preview["observed"] = [observed]
        context = self.context()
        scenario = self.scenario(context)
        spec = NS(length=4.575, width=1.76, height=1.47)
        stub = NS(GetOptionName=Mock(return_value=NS(value="2019_Toyota_Prius")),
                  GetVehicleSpec=Mock(return_value=spec))
        scenario.grpc.world = NS(_sim_adapter=NS(_actor_stub=stub))
        return context, scenario, stub, spec

    def static_surface_context(self):
        self.fixture_state = state("1", 45., 0., width=1.20839, length=3.441415, model="Scooter1")
        self.fixture_state.vehicle_state.collision_objects = []
        observed = trial.live._observation(self.fixture_state)
        observed.update(observed["raw_bbox_normalized"])
        observed.update(actor_kind="obstacle", model="Scooter1")
        self.preview["observed"] = [observed]
        context = self.context()
        source = {"type": "sealed_lidar_verified_static_footprint", "catalog_sha256": "current-catalog"}
        catalog = {"Scooter1": {"width_m": .81, "length_m": 2.26,
            "center_offset_spawn_xy": [0., 0.], "heading_offset_deg": 1.347,
            "geometry_verified": True, "geometry_source": source}}
        context.candidate["obstacles"] = [{"actor_kind": "obstacle", "model": "Scooter1",
            "planned_bbox": {key: catalog["Scooter1"][key] for key in (
                "width_m", "length_m", "center_offset_spawn_xy", "heading_offset_deg")},
            "geometry_source": copy.deepcopy(source)}]
        scenario = self.scenario(context)
        stub = NS(GetOptionName=Mock(return_value=NS(value="Scooter1")))
        scenario.grpc.world = NS(_sim_adapter=NS(_actor_stub=stub))
        return context, scenario, stub, catalog

    def native_static_frames(self, context, receipt, values=None):
        reference = context.references["1"]
        row = {"raw_actor_id": "1", "actor_kind": "obstacle", "name": reference["model"],
               "position_xyz": reference["position_xyz"], "heading_deg": reference["rotation_xyz_deg"][2],
               "size_xyz": list(values or [2.26, .81, 1.87]), "velocity_raw": 0.}
        return [{"receipt_monotonic": receipt-.003+index*.001, "source_sequence": 101+index,
                 "source_stamp_ns": 1_000_000_000+index*20_000_000,
                 "publisher_node": "/native_bridge", "rows": [copy.deepcopy(row)]} for index in range(3)]

    def bind_static_surface(self, context, scenario, catalog, *, transform_frames=None, values=None):
        class Clock:
            value = 100.
            def now(self):
                self.value += .01
                return self.value
            def sleep(self, duration):
                self.value += duration
        clock = Clock()
        def snapshots(after):
            frames = self.native_static_frames(context, clock.value, values)
            return transform_frames(frames) if transform_frames else frames
        self.monitor.object_snapshots.side_effect = snapshots
        with patch.object(trial, "load_static_models", return_value=catalog), patch.object(
                trial.live, "_inventory", return_value={"1": self.fixture_state, "Ego": self.ego_state}), patch.dict(
                sys.modules, {"proto.morai.actor.actor_enum_pb2": NS(GEAR_MODE_P=1)}):
            context.bind_static_surfaces(scenario, timeout_sec=.3, monotonic_fn=clock.now, sleep_fn=clock.sleep)

    def test_static_surface_binding_preserves_larger_physical_box_and_sealed_provenance(self):
        context, scenario, stub, catalog = self.static_surface_context()
        before = copy.deepcopy(context.references["1"])
        self.bind_static_surface(context, scenario, catalog)
        reference = context.references["1"]
        contract = reference["static_surface_contract"]
        self.assertEqual(contract["source"], "verified_native_ROS_static_metadata")
        self.assertEqual(contract["size_lw_m"], [2.26, .81])
        self.assertEqual(contract["size_lwh_m"], [2.26, .81, 1.87])
        self.assertEqual(contract["geometry_source"], catalog["Scooter1"]["geometry_source"])
        self.assertEqual(len(contract["native_frames"]), 3)
        self.assertTrue(all(frame["receipt_monotonic"] > contract["binding_monotonic"] for frame in contract["native_frames"]))
        self.assertEqual(reference["raw_bbox_normalized"], before["raw_bbox_normalized"])
        self.assertEqual(reference["length_m"], 3.441415)
        self.assertEqual(reference["width_m"], 1.20839)
        self.assertEqual(stub.GetOptionName.call_args.kwargs, {"timeout": .5})
        self.bind_static_surface(context, scenario, catalog)

    def test_static_surface_binding_rejects_unknown_stale_or_changed_candidate(self):
        for change in ("unknown", "stale_source", "dimensions", "wrong_model", "unverified"):
            with self.subTest(change=change):
                context, scenario, _, catalog = self.static_surface_context()
                if change == "unknown":
                    catalog.clear()
                elif change == "stale_source":
                    context.candidate["obstacles"][0]["geometry_source"]["catalog_sha256"] = "old"
                elif change == "dimensions":
                    context.candidate["obstacles"][0]["planned_bbox"]["width_m"] += .01
                elif change == "wrong_model":
                    context.candidate["obstacles"][0]["model"] = "wrong"
                else:
                    catalog["Scooter1"]["geometry_verified"] = False
                with self.assertRaisesRegex(RuntimeError, "Static surface"):
                    self.bind_static_surface(context, scenario, catalog)
                self.assertNotIn("static_surface_contract", context.references["1"])

    def test_static_surface_binding_rejects_actual_model_motion_and_changed_contract(self):
        context, scenario, stub, catalog = self.static_surface_context()
        stub.GetOptionName.return_value.value = "wrong"
        with self.assertRaisesRegex(RuntimeError, "model changed"):
            self.bind_static_surface(context, scenario, catalog)
        stub.GetOptionName.return_value.value = "Scooter1"
        self.fixture_state.transform.location.x += .5
        with self.assertRaisesRegex(RuntimeError, "pose moved"):
            self.bind_static_surface(context, scenario, catalog)
        self.fixture_state.transform.location.x -= .5
        self.bind_static_surface(context, scenario, catalog)
        context.references["1"]["static_surface_contract"]["size_lw_m"][0] += .01
        with self.assertRaisesRegex(RuntimeError, "contract changed"):
            self.bind_static_surface(context, scenario, catalog)

    def test_ros_sealed_static_dimensions_pass_and_scaled_or_wrong_model_fail(self):
        import time
        context, scenario, _, catalog = self.static_surface_context()
        self.bind_static_surface(context, scenario, catalog)
        reference = context.references["1"]
        row = {"raw_actor_id": "1", "actor_kind": "obstacle", "name": "Scooter1",
               "position_xyz": reference["position_xyz"], "heading_deg": reference["rotation_xyz_deg"][2],
               "size_xyz": [2.25999999, .810000002, 1.87]}
        snapshot = {"receipt_monotonic": time.monotonic(), "publisher_node": "/native_bridge", "rows": [row]}
        self.monitor.snapshot.return_value = (snapshot, None)
        context.inspect_ros({"position_xyz": [0., 0., 28.]})
        self.assertEqual(reference["length_m"], 3.441415)
        for index in range(2):
            changed = copy.deepcopy(row)
            changed["size_xyz"][index] += .11
            snapshot["rows"] = [changed]
            with self.subTest(axis=index), self.assertRaisesRegex(RuntimeError, "sealed static surface"):
                context.inspect_ros({"position_xyz": [0., 0., 28.]})
        snapshot["rows"] = [dict(row, name="wrong")]
        with self.assertRaisesRegex(RuntimeError, "sealed static surface"):
            context.inspect_ros({"position_xyz": [0., 0., 28.]})

    def test_runtime_native_static_metadata_rejects_publisher_replacement(self):
        import time
        context, scenario, _, catalog = self.static_surface_context()
        self.bind_static_surface(context, scenario, catalog)
        contract = context.references["1"]["static_surface_contract"]
        snapshot = {"receipt_monotonic": time.monotonic(), "publisher_node": "/native_bridge",
                    "rows": [copy.deepcopy(contract["native_frames"][0]["row"])]}
        self.monitor.snapshot.return_value = (snapshot, None)
        context.inspect_ros({"position_xyz": [0., 0., 28.]})
        snapshot["publisher_node"] = "/replacement_bridge"
        with self.assertRaisesRegex(RuntimeError, "publisher changed"):
            context.inspect_ros({"position_xyz": [0., 0., 28.]})

    def test_native_static_metadata_can_be_larger_smaller_or_use_different_model_axes(self):
        for values in ([.81, .81, 1.2], [.56, 1.21, 1.5], [4., 2., 2.5]):
            with self.subTest(native_size=values):
                context, scenario, _, catalog = self.static_surface_context()
                physical = {key: copy.deepcopy(context.references["1"][key]) for key in (
                    "center_xyz", "heading_deg", "length_m", "width_m", "raw_bbox_normalized")}
                self.bind_static_surface(context, scenario, catalog, values=values)
                reference = context.references["1"]
                self.assertEqual(reference["static_surface_contract"]["size_lwh_m"], values)
                self.assertEqual({key: reference[key] for key in physical}, physical)

    def test_native_static_capture_rejects_unverified_frame_sources_identity_and_dimensions(self):
        def modify(frames, case):
            frame, row = frames[1], frames[1]["rows"][0]
            if case == "publisher": frame["publisher_node"] = "/other_bridge"
            elif case == "empty_publisher": frame["publisher_node"] = ""
            elif case == "receipt": frame["receipt_monotonic"] = frames[0]["receipt_monotonic"]
            elif case == "stale": frame["receipt_monotonic"] -= 1.
            elif case == "future": frame["receipt_monotonic"] += 1.
            elif case == "model": row["name"] = "wrong"
            elif case == "id": row["raw_actor_id"] = "wrong"
            elif case == "kind": row["actor_kind"] = "vehicle"
            elif case == "pose": row["position_xyz"][0] += .16
            elif case == "heading": row["heading_deg"] += 2.1
            elif case == "moving": row["velocity_raw"] = .051
            elif case == "dimensions": row["size_xyz"][0] += .011
            elif case == "nonfinite": row["size_xyz"][2] = float("nan")
            return frames
        for case in ("publisher", "empty_publisher", "receipt", "stale", "future", "model",
                     "id", "kind", "pose", "heading", "moving", "dimensions", "nonfinite"):
            with self.subTest(case=case):
                context, scenario, _, catalog = self.static_surface_context()
                with self.assertRaises(RuntimeError):
                    self.bind_static_surface(context, scenario, catalog,
                        transform_frames=lambda frames: modify(frames, case))
                self.assertNotIn("static_surface_contract", context.references["1"])

    def test_native_static_capture_times_out_without_three_fresh_messages_and_requires_park(self):
        context, scenario, _, catalog = self.static_surface_context()
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.bind_static_surface(context, scenario, catalog, transform_frames=lambda frames: [])
        self.assertNotIn("static_surface_contract", context.references["1"])
        self.ego_state.vehicle_state.gear_mode = 4
        with self.assertRaisesRegex(RuntimeError, "parked in P"):
            self.bind_static_surface(context, scenario, catalog)

    def test_native_static_binding_records_same_publisher_ordering_without_relaxing_frames(self):
        for key,reason in (("source_sequence","source_sequence_regression"),("source_stamp_ns","source_stamp_regression")):
            context,scenario,_,catalog=self.static_surface_context()
            def regress(frames):
                frames[1][key]=frames[0][key]-1
                return frames
            with self.assertRaisesRegex(RuntimeError,"regressed with the same publisher"):
                self.bind_static_surface(context,scenario,catalog,transform_frames=regress)
            failure=context.preview_report["native_metadata_failure"]
            self.assertEqual(failure["reason"],reason)
            self.assertEqual(failure["previous_publisher"],"/native_bridge")
            self.assertEqual(failure["current_publisher"],"/native_bridge")
            self.assertIs(failure["retryable_same_publisher_ordering"],True)
            self.assertNotIn("static_surface_contract",context.references["1"])

    @staticmethod
    def ordered_static_frames(frames, source_pairs):
        result = []
        for index, (sequence, stamp) in enumerate(source_pairs):
            frame = copy.deepcopy(frames[0])
            frame["receipt_monotonic"] = frames[-1]["receipt_monotonic"]-(len(source_pairs)-index)*.0001
            frame["source_sequence"], frame["source_stamp_ns"] = sequence, stamp
            result.append(frame)
        return result

    def test_native_static_binding_skips_duplicates_and_requires_three_distinct_frames(self):
        cases = (
            ([(101, 10), (102, 10), (103, 20), (104, 30)],
             "source_stamp_duplicate", [101, 103, 104]),
            ([(101, 10), (101, 10), (102, 20), (103, 30)],
             "source_sequence_and_stamp_duplicate", [101, 102, 103]),
            ([(101, 10), (101, 20), (102, 30), (103, 40)],
             "source_sequence_duplicate", [101, 102, 103]),
        )
        for pairs, reason, expected_sequences in cases:
            with self.subTest(reason=reason):
                context, scenario, _, catalog = self.static_surface_context()
                delivered = []
                def duplicate_stream(frames):
                    stream = self.ordered_static_frames(frames, pairs)
                    delivered.extend(stream)
                    return stream
                self.bind_static_surface(context, scenario, catalog, transform_frames=duplicate_stream)
                contract = context.references["1"]["static_surface_contract"]
                self.assertEqual([row["source_sequence"] for row in contract["native_frames"]], expected_sequences)
                duplicates = context.report["native_metadata_duplicates"]
                self.assertEqual(duplicates["skipped_frame_count"], 1)
                self.assertEqual(duplicates["skip_reasons"], {reason: 1})
                self.assertEqual(duplicates["verified_frames_added"], 0)
                self.assertIs(duplicates["source_identity_stable"], True)
                self.assertEqual(context.preview_report["native_metadata_duplicates"], duplicates)
                self.assertNotIn("native_metadata_failure", context.report)
                # Skipping is a verifier decision, not a rewrite of native data.
                self.assertEqual([(row["source_sequence"], row["source_stamp_ns"]) for row in delivered], pairs)

    def test_native_static_duplicate_cursor_advances_and_repetition_cannot_complete_binding(self):
        context, scenario, _, catalog = self.static_surface_context()
        requested_cursors, last_receipt = [], []
        def duplicate_only(frames):
            requested_cursors.append(self.monitor.object_snapshots.call_args.args[0])
            frame = copy.deepcopy(frames[-1])
            frame["source_sequence"], frame["source_stamp_ns"] = 101, 10
            if last_receipt:
                self.assertGreater(requested_cursors[-1], requested_cursors[-2])
                self.assertEqual(requested_cursors[-1], last_receipt[-1])
            last_receipt.append(frame["receipt_monotonic"])
            return [frame]
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.bind_static_surface(context, scenario, catalog, transform_frames=duplicate_only)
        self.assertGreater(len(requested_cursors), 2)
        self.assertGreater(context.report["native_metadata_duplicates"]["skipped_frame_count"], 0)
        self.assertNotIn("static_surface_contract", context.references["1"])

    def test_native_static_duplicate_watermarks_still_reject_actual_regressions(self):
        cases = (
            ([(101, 10), (105, 10), (104, 20)], "source_sequence_regression", 105, 10),
            ([(101, 10), (101, 20), (102, 15)], "source_stamp_regression", 101, 20),
            ([(101, 10), (102, 10), (101, 9)], "source_sequence_and_stamp_regression", 102, 10),
        )
        for pairs, reason, previous_sequence, previous_stamp in cases:
            with self.subTest(reason=reason):
                context, scenario, _, catalog = self.static_surface_context()
                with self.assertRaisesRegex(RuntimeError, "regressed with the same publisher"):
                    self.bind_static_surface(context, scenario, catalog,
                        transform_frames=lambda frames: self.ordered_static_frames(frames, pairs))
                failure = context.report["native_metadata_failure"]
                self.assertEqual(failure["reason"], reason)
                self.assertEqual(failure["previous_source_sequence"], previous_sequence)
                self.assertEqual(failure["previous_source_stamp_ns"], previous_stamp)
                self.assertEqual(context.report["native_metadata_duplicates"]["skipped_frame_count"], 1)
                self.assertNotIn("static_surface_contract", context.references["1"])

    def test_native_static_unspecified_zero_stamp_retains_sequence_based_freshness(self):
        for pairs, duplicate_count in (([(101, 0), (102, 0), (103, 0)], 0),
                                      ([(101, 0), (101, 0), (102, 0), (103, 0)], 1)):
            with self.subTest(duplicate_count=duplicate_count):
                context, scenario, _, catalog = self.static_surface_context()
                self.bind_static_surface(context, scenario, catalog,
                    transform_frames=lambda frames: self.ordered_static_frames(frames, pairs))
                evidence = context.references["1"]["static_surface_contract"]["native_frames"]
                self.assertEqual([row["source_sequence"] for row in evidence], [101, 102, 103])
                self.assertEqual([row["source_stamp_ns"] for row in evidence], [0, 0, 0])
                self.assertEqual(context.report.get("native_metadata_duplicates", {}).get("skipped_frame_count", 0),
                                 duplicate_count)

    def test_native_static_binding_publisher_change_is_structured_and_fatal_even_with_regression(self):
        context,scenario,_,catalog=self.static_surface_context()
        def replace(frames):
            frames[1]["publisher_node"]="/replacement"
            frames[1]["source_sequence"]=frames[0]["source_sequence"]
            return frames
        with self.assertRaisesRegex(RuntimeError,"publisher changed during binding"):
            self.bind_static_surface(context,scenario,catalog,transform_frames=replace)
        failure=context.preview_report["native_metadata_failure"]
        self.assertEqual(failure["reason"],"publisher_changed")
        self.assertIs(failure["retryable_same_publisher_ordering"],False)
        self.assertNotIn("static_surface_contract",context.references["1"])

    def test_object_monitor_retains_actual_header_publisher_and_bounded_frame_history(self):
        monitor = trial.TrialRosEvidence()
        for index in range(70):
            item = NS(unique_id=1, name="TrafficBarrel", position=NS(x=1., y=2., z=3.),
                      heading=4., size=NS(x=.81, y=.81, z=1.2), velocity=NS(x=0.))
            message = NS(header=NS(seq=index, stamp=NS(secs=12, nsecs=index)),
                         _connection_header={"callerid": "/native_bridge"}, npc_list=[], obstacle_list=[item])
            monitor.on_objects(message)
        frames = monitor.object_snapshots(0.)
        self.assertEqual(len(frames), 64)
        self.assertEqual(frames[0]["source_sequence"], 6)
        self.assertEqual(frames[-1]["source_sequence"], 69)
        self.assertEqual(frames[-1]["source_stamp_ns"], 12_000_000_069)
        self.assertEqual(frames[-1]["publisher_node"], "/native_bridge")
        frames[-1]["rows"][0]["name"] = "mutated"
        self.assertEqual(monitor.snapshot()[0]["rows"][0]["name"], "TrafficBarrel")

    def bind_specs(self, context, scenario):
        with patch.object(trial.live, "_inventory", return_value={"1": self.fixture_state}):
            context.bind_vehicle_specs(scenario)

    def test_native_vehicle_spec_binding_keeps_larger_collision_union_and_timeout(self):
        context, scenario, stub, _ = self.vehicle_context()
        before = copy.deepcopy(context.references["1"])
        self.bind_specs(context, scenario)
        reference = context.references["1"]
        self.assertEqual(reference["vehicle_spec_contract"], {"source": "bounded_GetVehicleSpec",
            "raw_actor_id": "1", "model": "2019_Toyota_Prius", "size_lwh_m": [4.575, 1.76, 1.47]})
        self.assertEqual(reference["width_m"], 2.13)
        self.assertEqual(reference["raw_bbox_normalized"], before["raw_bbox_normalized"])
        self.assertEqual(stub.GetVehicleSpec.call_args.kwargs, {"timeout": .5})
        self.assertEqual(stub.GetOptionName.call_args.kwargs, {"timeout": .5})
        self.bind_specs(context, scenario)

    def test_native_vehicle_spec_binding_rejects_invalid_or_larger_dimensions(self):
        for key, value in (("width", float("nan")), ("height", 0.), ("width", 2.24), ("length", 4.69), ("height", float("inf"))):
            with self.subTest(key=key, value=value):
                context, scenario, _, spec = self.vehicle_context()
                setattr(spec, key, value)
                with self.assertRaisesRegex(RuntimeError, "specification"):
                    self.bind_specs(context, scenario)
                self.assertNotIn("vehicle_spec_contract", context.references["1"])

    def test_nominal_height_can_exceed_collider_without_changing_physical_geometry(self):
        context, scenario, stub, spec = self.vehicle_context()
        reference = context.references["1"]
        reference["model"] = "DEFAULT_SEDAN"
        self.fixture_state.bounding_box.extent.z = 1.347503423690796*.5
        observed = trial.live._observation(self.fixture_state)
        reference["world_bbox"] = observed["world_bbox"]
        reference["raw_bbox_normalized"] = observed["raw_bbox_normalized"]
        stub.GetOptionName.return_value.value = "DEFAULT_SEDAN"
        spec.height = 1.5
        before = copy.deepcopy(reference["world_bbox"])
        self.bind_specs(context, scenario)
        self.assertEqual(reference["vehicle_spec_contract"]["size_lwh_m"][2], 1.5)
        self.assertEqual(reference["world_bbox"], before)

    def test_native_vehicle_spec_binding_rejects_changed_model_pose_and_contract(self):
        context, scenario, stub, spec = self.vehicle_context()
        stub.GetOptionName.return_value.value = "wrong"
        with self.assertRaisesRegex(RuntimeError, "model changed"):
            self.bind_specs(context, scenario)
        stub.GetOptionName.return_value.value = "2019_Toyota_Prius"
        self.fixture_state.transform.location.x += .5
        with self.assertRaisesRegex(RuntimeError, "pose moved"):
            self.bind_specs(context, scenario)
        self.fixture_state.transform.location.x -= .5
        self.bind_specs(context, scenario)
        spec.width += .01
        with self.assertRaisesRegex(RuntimeError, "specification changed"):
            self.bind_specs(context, scenario)

    def test_native_vehicle_spec_rpc_failure_is_closed_before_drive(self):
        context, scenario, stub, _ = self.vehicle_context()
        stub.GetVehicleSpec.side_effect = TimeoutError("bounded spec timeout")
        with self.assertRaisesRegex(TimeoutError, "bounded spec"):
            self.bind_specs(context, scenario)
        self.assertNotIn("vehicle_spec_contract", context.references["1"])

    def test_ros_vehicle_native_dimensions_pass_but_changed_scale_height_and_model_fail(self):
        import time
        context, scenario, _, _ = self.vehicle_context()
        self.bind_specs(context, scenario)
        reference = context.references["1"]
        row = {"raw_actor_id": "1", "actor_kind": "vehicle", "name": "2019_Toyota_Prius",
               "position_xyz": reference["position_xyz"], "heading_deg": reference["rotation_xyz_deg"][2],
               "size_xyz": [4.574999809, 1.75999999, 1.470000029]}
        snapshot = {"receipt_monotonic": time.monotonic(), "rows": [row]}
        self.monitor.snapshot.return_value = (snapshot, None)
        context.inspect_ros({"position_xyz": [0., 0., 28.]})
        self.assertEqual(context.references["1"]["width_m"], 2.13)
        for index in range(3):
            with self.subTest(axis=index):
                changed = copy.deepcopy(row)
                changed["size_xyz"][index] += .11
                snapshot["rows"] = [changed]
                with self.assertRaisesRegex(RuntimeError, "bound native"):
                    context.inspect_ros({"position_xyz": [0., 0., 28.]})
        snapshot["rows"] = [dict(row, name="wrong")]
        with self.assertRaisesRegex(RuntimeError, "bound native"):
            context.inspect_ros({"position_xyz": [0., 0., 28.]})

    def test_mission_binds_native_specs_before_first_guard_and_expert_configuration(self):
        from ..scenarios.full_loop_scenario import FullLoopScenario
        events = []
        scenario = MissionScenario.__new__(MissionScenario)
        scenario.cfg = {"mission_mode": trial.MODE}
        context = Mock(report={})
        context.bind_static_surfaces.side_effect = lambda owner: events.append("sealed_static")
        context.bind_vehicle_specs.side_effect = lambda owner: events.append("bind")
        context.inspect.side_effect = lambda owner, elapsed: events.append("inspect")
        with patch.object(trial, "required", return_value=context), patch(
                "scenario_runner.scenarios.segment_modes.scenario.tunnel_baseline.before_drive"), patch.object(
                FullLoopScenario, "configure_drive_with_retries", side_effect=lambda: events.append("expert")):
            scenario.configure_drive_with_retries()
        self.assertEqual(events, ["sealed_static", "bind", "inspect", "expert"])

    def test_regular_expert_runtime_has_no_trial_injection(self):
        driving = Mock()
        factory = Mock()
        configure_tunnel_trial(driving, {"planning": {}}, factory)
        factory.assert_not_called()
        driving.lattice_planning_process.recorded_jamming.set_object_geometries.assert_not_called()

    def test_runtime_boundary_is_installed_before_measured_objects_and_proof_is_recorded(self):
        rows = [self.preview["observed"][0]]
        scene_hash = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        planner = Mock()
        planner.local.selected_names = ["jamming_01.csv"]
        planner.run.return_value = NS(mode="static_obstacle", reason="follow", active=True,
            debug_details={"static_avoidance": {"path_validated": True, "road_boundary": {
                "passed": True, "policy_sha256": self.boundary.policy_sha256}},
                "jamming_scene_geometry_sha256": scene_hash})
        driving = NS(lattice_planning_process=NS(recorded_jamming=planner))
        config = {"map": {"data_dir": "local-map"}, "planning": {
            "jamming_scene_geometry": rows, "jamming_trial_diagnostics_path": str(self.output/"planner.json"),
            "jamming_trial_contract": {"mode": trial.MODE, "road_boundary_policy": self.boundary.policy,
                                        "road_boundary_policy_sha256": self.boundary.policy_sha256}}}
        factory = Mock(return_value=self.boundary)
        configure_tunnel_trial(driving, config, factory)
        factory.assert_called_once_with("local-map", body_policy=self.boundary.policy["body"],
            approach_m=120., departure_m=70., planning_margin_m=.35, runtime_margin_m=.15)
        self.assertEqual(planner.mock_calls[:2], [unittest.mock.call.set_road_boundary(self.boundary),
                                                unittest.mock.call.set_object_geometries(rows)])
        planner.run(None)
        proof = json.loads((self.output/"planner.json").read_text())
        self.assertTrue(proof["validated_path_seen"])
        self.assertEqual(proof["selected_routes"], ["jamming_01.csv"])

    def test_runtime_policy_mismatch_fails_before_any_object_geometry_is_installed(self):
        driving = Mock()
        config = {"map": {"data_dir": "local-map"}, "planning": {
            "jamming_scene_geometry": [self.preview["observed"][0]],
            "jamming_trial_contract": {"mode": trial.MODE, "road_boundary_policy": self.boundary.policy,
                                        "road_boundary_policy_sha256": "different"}}}
        with self.assertRaisesRegex(RuntimeError, "differs from measured"):
            configure_tunnel_trial(driving, config, lambda *args, **kwargs: self.boundary)
        driving.lattice_planning_process.recorded_jamming.set_object_geometries.assert_not_called()

    def test_drive_callback_explicitly_uses_measured_planning_and_runtime_margins(self):
        factory = Mock(return_value=self.boundary)
        measured = dict(self.measured, measured_ego_body=self.boundary.policy["body"])
        report = {"planned": {"count": 1}, "measured_route_screen": measured}
        callback = trial.make_drive_callback(NS(), NS(), {"paths": {"mgeo_root": "current-map"}}, {}, self.output)
        with patch.object(trial.screening, "_expert_search", return_value=(None, NS(from_map_dir=factory), None, None, None)), patch.object(
                trial, "TrialContext", return_value=NS(references={"1": {}})), self.assertRaisesRegex(RuntimeError, "ownership differs"):
            callback(report, [{"raw_id": "different"}])
        factory.assert_called_once_with("current-map", approach_m=120., departure_m=70.,
            planning_margin_m=.35, runtime_margin_m=.15, body_policy=self.boundary.policy["body"])

    def departure_loader(self):
        rows = {"A2256W000126": {"points": [[0., 0., 28.], [120., 0., 28.]],
                "from_node_idx": "start", "to_node_idx": "portal", "width_start": 3.5, "width_end": 3.5},
                "A2256W000128": {"points": [[120., 0., 28.], [220., 0., 28.]],
                "from_node_idx": "portal", "to_node_idx": "departure_end", "width_start": 3.5, "width_end": 3.5}}
        return NS(get_link=lambda identity: rows[identity])

    def lane_observation(self, x, y=0.):
        return trial.live._observation(state("Ego", x, y, width=2., length=4.5))

    def test_goal_center_before_trial_end_can_return_with_body_beyond_goal_on_actual_lane(self):
        from shapely.geometry import LineString, Polygon
        lane, source = trial.mapped_original_lane(self.departure_loader())
        observed = self.lane_observation(148.3)
        checks = trial.lane_return_checks(self.boundary, lane, observed)
        shortened = LineString([[-40., 0.], [150., 0.]]).buffer(1.75, cap_style=2, join_style=2)
        self.assertFalse(shortened.covers(Polygon(checks["policy_body_corners_world"])))
        self.assertTrue(checks["whole_body_in_original_lane"])
        self.assertEqual(source["end_xyz"], [220., 0., 28.])

    def test_opposing_lane_does_not_count_as_return_to_actual_original_lane(self):
        lane, _ = trial.mapped_original_lane(self.departure_loader())
        checks = trial.lane_return_checks(self.boundary, lane, self.lane_observation(148.3, 3.5))
        self.assertFalse(checks["whole_body_in_original_lane"])

    def test_body_past_actual_mapped_departure_end_is_rejected(self):
        lane, _ = trial.mapped_original_lane(self.departure_loader())
        checks = trial.lane_return_checks(self.boundary, lane, self.lane_observation(219.))
        self.assertFalse(checks["policy_body_in_original_lane"])
        self.assertFalse(checks["whole_body_in_original_lane"])


if __name__ == "__main__":
    unittest.main()
