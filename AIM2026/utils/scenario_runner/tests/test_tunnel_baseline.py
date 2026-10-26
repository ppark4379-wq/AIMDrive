"""Tunnel regression checks: approach baseline."""

import copy, unittest, json, sys, tempfile
from types import SimpleNamespace
from scenario_runner.main_runner import load_configuration
from scenario_runner.scenarios.full_loop_scenario import FullLoopScenario
from scenario_runner.scenarios.full_loop_parts.before_tunnel import apply
from scenario_runner.utils.map_loader import MGeoMapLoader
from scenario_runner.adapters.ve_signal import build_signal_targets
from pathlib import Path
from unittest.mock import Mock, patch
from ..scenarios.segment_modes import tunnel_baseline as baseline, tunnel_static_startup as startup
from ..scenarios.full_loop_parts.tunnel_static_parts import live


# Before tunnel

class BeforeTunnelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.global_cfg, _, cls.config = load_configuration()
        cls.loader = MGeoMapLoader(cls.global_cfg['paths']['mgeo_root'])

    def context(self, shortened):
        cfg = copy.deepcopy(self.config)
        cfg.update(route_debug_enabled=False, ve_stop_before_tunnel=shortened)
        c = FullLoopScenario(SimpleNamespace(make_transform=lambda *args: args),
                             self.loader, self.global_cfg, cfg)
        c.route_definition = c.load_route_definition()
        c.prepare_full_loop_route()
        return c

    def test_actual_route_ends_at_highway_exit_before_tunnel(self):
        full = self.context(False)
        short = self.context(True)
        self.assertEqual(full.route_points[:3530], short.route_points)
        self.assertEqual(full.route_points, short._full_route_points)
        self.assertEqual(full.route_points[3529][:3], (short.goal_x, short.goal_y, short.goal_z))
        self.assertAlmostEqual(short.route_length_m, full.highway_exit_s)
        self.assertLess(short.route_length_m, short.tunnel_entry_s)
        self.assertEqual(full.highway_activation_s, short.highway_activation_s)
        self.assertEqual(short.end_link, short.route_links[-1])
        self.assertFalse(short.cfg['expert_is_closed_path'])
        for key in ('tunnel_sudden_brake', 'tunnel_dynamic', 'tunnel_vehicle_dynamic'):
            self.assertFalse(short.cfg[key]['enabled'])
        self.assertEqual(full.cfg['highway_npc'], short.cfg['highway_npc'])
        self.assertEqual(full.cfg['static_obstacle_models'], short.cfg['static_obstacle_models'])
        self.assertEqual(4430, len(full.route_points))

    def test_endpoint_cannot_be_inside_tunnel(self):
        context = self.context(False)
        definition = copy.deepcopy(context.route_definition)
        for checkpoint in definition['checkpoints']:
            if str(checkpoint['id']) == '13':
                checkpoint['path_index'] = 3900
        context.route_definition = definition
        with self.assertRaises(ValueError):
            apply(context)

    def test_signal_binding_accepts_open_prefix_and_excludes_tunnel_signal(self):
        context = self.context(True)
        config = {'map': {'data_dir': self.global_cfg['paths']['mgeo_root'],
                         'route_signal_route_links': context.route_definition['route_links'],
                         'is_closed_path': context.cfg['expert_is_closed_path']}}
        document = build_signal_targets(context, config)
        ids = {target['signal_id'] for target in document['targets']}
        self.assertIn('C1256W000052', ids)
        self.assertIn('C1256W000077', ids)
        self.assertNotIn('LCS01', ids)
        self.assertTrue(all(t['path_index'] < len(context.route_points) for t in document['targets']))
        config['map']['is_closed_path'] = True
        with self.assertRaisesRegex(ValueError, 'does not close'):
            build_signal_targets(context, config)


# Offline tests of the tunnel's global inventory and empty-road contract.

def xyz(x, y, z=28.0):
    return SimpleNamespace(x=x, y=y, z=z)


def actor(identity, x, y, *, client_key="", center=None, extent=(1.0, 2.0, 1.0)):
    return SimpleNamespace(
        actor_info=SimpleNamespace(id=SimpleNamespace(value=identity), client_key=client_key),
        transform=SimpleNamespace(location=xyz(x, y)),
        bounding_box=SimpleNamespace(
            location=xyz(*(center if center is not None else (x, y))),
            extent=xyz(*extent),
        ),
    )


class Clock:
    def __init__(self):
        self.value = 100.

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class TunnelBaselineTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.runtime = Path(temporary.name)
        self.links = {
            baseline.TUNNEL_LINK: {
                "road_type": "2", "width_start": 3.5, "width_end": 3.5,
                "points": [(0.0, 0.0, 28.0), (50.0, 0.0, 28.0), (100.0, 0.0, 28.0)],
            },
            baseline.OPPOSING_LINK: {
                "road_type": "2", "width_start": 3.5, "width_end": 3.5,
                "points": [(100.0, 3.5, 28.0), (50.0, 3.5, 28.0), (0.0, 3.5, 28.0)],
            },
        }
        grpc = Mock()
        grpc.get_ego.return_value = Mock(spec=(
            "get_id", "get_object_info", "set_velocity", "set_vehicle_gear", "get_actor_state",
        ))
        grpc.get_ego.return_value.get_id.return_value = "Ego"
        grpc.get_ego.return_value.set_velocity.return_value = True
        grpc.get_ego.return_value.set_vehicle_gear.return_value = True
        self.ego_state = actor("Ego", -40., 0.)
        self.ego_state.transform.rotation = xyz(0., 0., 0.)
        self.ego_state.bounding_box.rotation = xyz(0., 0., 0.)
        self.ego_state.global_velocity = xyz(0., 0., 0.)
        self.ego_state.angular_velocity = xyz(0., 0., 0.)
        self.ego_state.vehicle_state = SimpleNamespace(gear_mode=4, brake=1., throttle=0.)
        grpc.get_ego.return_value.get_actor_state.return_value = self.ego_state
        self.stub = grpc.world._sim_adapter._actor_stub
        self.stub.GetActorState.return_value = self.ego_state
        self.stub.GetVehicleControlMode.return_value = SimpleNamespace(mode=3)
        self.stub.SetVehicleControlMode.return_value = SimpleNamespace(status=0)
        self.stub.SetVehicleGear.return_value = SimpleNamespace(status=0)
        self.stopped = dict(live._observation(self.ego_state), brake=1., throttle=0.,
                            control_mode=3, gear_mode=4)
        self.clock = Clock()
        self.real_stopped_readback = startup.common_stopped_readback
        for name, options in (
                ("hold_common_brake", {}), ("release_common_brake", {}),
                ("_require_no_controller", {}),
                ("common_stopped_readback", {"wraps": self.real_stopped_readback})):
            mocked = patch.object(startup, name, **options)
            setattr(self, name, mocked.start())
            self.addCleanup(mocked.stop)
        grpc.world._sim_adapter._actor_stub.GetAllActorsState.return_value = SimpleNamespace(
            states=[actor("Ego", -40.0, 0.0)],
        )
        loader = SimpleNamespace(get_link=lambda identity: self.links[identity])
        self.context = SimpleNamespace(
            mission_mode=SimpleNamespace(name=baseline.MODE_NAME),
            map_loader=loader,
            route_points=[(-40.0, 0.0, 28.0), (0.0, 0.0, 28.0), (50.0, 0.0, 28.0), (100.0, 0.0, 28.0), (130.0, 0.0, 28.0)],
            global_cfg={"paths": {"runtime_root": str(self.runtime)}},
            run_instance_id="unit-test",
            grpc=grpc,
            stop_drive_control=Mock(),
        )
        self.fake_proto = patch.dict(sys.modules, {
            "proto.morai.actor.actor_enum_pb2": SimpleNamespace(
                GEAR_MODE_P=1, GEAR_MODE_D=4, VEHICLE_CONTROL_KEYBOARD=1, VEHICLE_CONTROL_AUTO_MODE=3),
            "proto.morai.actor.actor_set_pb2": SimpleNamespace(VehicleControlModeParam=SimpleNamespace, VehicleGear=SimpleNamespace),
            "proto.morai.common.enum_pb2": SimpleNamespace(STATUS_CODE_SUCCESS=0),
            "proto.morai.actor.actor_get_pb2": SimpleNamespace(
                GetAllActorsFilter=lambda **fields: SimpleNamespace(**fields),
            ),
        })
        self.fake_proto.start()
        self.addCleanup(self.fake_proto.stop)

    def prepare(self):
        baseline.prepare(self.context)
        return self.context.tunnel_baseline_report

    def response(self, states):
        self.context.grpc.world._sim_adapter._actor_stub.GetAllActorsState.return_value = SimpleNamespace(states=states)

    def saved(self):
        return json.loads((self.runtime / "tunnel_baseline/unit-test/baseline.json").read_text())

    def park(self):
        baseline.park(self.context, now=self.clock.now, sleep=self.clock.sleep)

    def test_prepare_reports_nominal_geometry_without_claiming_physical_validation(self):
        report = self.prepare()
        self.assertEqual(report["mapped_tunnel"]["nominal_combined_width_m"], 7.0)
        self.assertEqual(report["test_route"]["approach_length_m"], 40.0)
        self.assertEqual(report["test_route"]["after_exit_length_m"], 30.0)
        self.assertFalse(report["physical_portals_verified"])
        self.assertFalse(report["physical_walls_verified"])
        self.assertFalse(report["usable_physical_width_verified"])
        self.assertFalse(report["external_spawn_points_disabled"])
        self.assertEqual(self.saved()["status"], "prepared")
        self.context.grpc.spawn_vehicle.assert_not_called()

    def test_report_storage_failure_still_stops_ego(self):
        self.prepare()
        with patch.object(baseline, "_persist", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                baseline.before_drive(self.context)
        self.context.stop_drive_control.assert_called_once()

    def test_final_stop_records_actual_d_without_claiming_p_or_input_block(self):
        self.prepare()
        self.park()
        report = self.saved()
        self.assertTrue(report["final_stop_verified"])
        self.assertFalse(report["ego_stopped_and_parked"])
        self.assertEqual(report["stop_policy"], "common_auto_d_ros_brake")
        record = report["final_parking"]
        for field in ("physical_stop_verified", "owned_expert_stopped", "controller_publishers_absent",
                      "publisher_released_after_verified", "post_release_verified", "final_verified"):
            self.assertTrue(record[field], field)
        for field in ("external_input_disabled", "gear_p_verified", "velocity_reset_used"):
            self.assertFalse(record[field], field)
        self.assertEqual(record["actual_final"]["gear_mode"], 4)
        self.assertEqual(record["actual_final"]["control_mode"], 3)
        self.stub.SetVehicleGear.assert_not_called()
        self.stub.SetVehicleControlMode.assert_not_called()
        self.context.grpc.get_ego.return_value.set_velocity.assert_not_called()

    def test_actual_physical_stop_precedes_publisher_release(self):
        self.prepare()
        release_times = []
        self.release_common_brake.side_effect = lambda *a: release_times.append(self.clock.now())
        def readback(grpc, reference=None):
            self.ego_state.global_velocity.x = 2. if self.clock.now() < 110. else 0.
            return self.real_stopped_readback(grpc, reference)
        self.common_stopped_readback.side_effect = readback
        self.park()
        self.assertGreaterEqual(release_times[0], 110.)
        self.assertGreaterEqual(self.clock.now()-release_times[0], .5-1e-8)
        self.assertTrue(self.saved()["final_stop_verified"])
        self.context.grpc.get_ego.return_value.set_velocity.assert_not_called()

    def test_physical_stop_timeout_never_certifies_command_ack(self):
        self.prepare()
        self.ego_state.global_velocity.x = 2.
        with self.assertRaisesRegex(RuntimeError, "physical_stop did not settle"):
            self.park()
        self.assertLessEqual(self.clock.now(), 120.01)
        report = self.saved()
        self.assertFalse(report["final_stop_verified"])
        self.assertFalse(report["final_parking"]["physical_stop_verified"])
        self.assertFalse(report["final_parking"]["publisher_released_after_verified"])
        self.release_common_brake.assert_called()

    def test_post_release_brake_throttle_mode_and_gear_are_observed(self):
        for field, value in (("brake", 0.), ("throttle", .1), ("gear_mode", 1), ("control_mode", 1)):
            with self.subTest(field=field):
                self.prepare()
                self.release_common_brake.reset_mock()
                self.ego_state.vehicle_state = SimpleNamespace(gear_mode=4, brake=1., throttle=0.)
                self.stub.GetVehicleControlMode.return_value = SimpleNamespace(mode=3)
                def release(*args):
                    if field == "control_mode":
                        self.stub.GetVehicleControlMode.return_value.mode = value
                    else:
                        setattr(self.ego_state.vehicle_state, field, value)
                self.release_common_brake.side_effect = release
                with self.assertRaisesRegex(RuntimeError, "post_release did not settle"):
                    self.park()
                report = self.saved()
                self.assertFalse(report["final_stop_verified"])
                self.assertTrue(report["final_parking"]["physical_stop_verified"])
                self.assertTrue(report["final_parking"]["publisher_released_after_verified"])
                self.assertFalse(report["final_parking"]["post_release_verified"])

    def test_post_release_motion_or_pose_drift_resets_final_stop_verification(self):
        self.prepare()
        self.park()
        self.assertTrue(self.saved()["final_stop_verified"])
        self.release_common_brake.side_effect = lambda *a: setattr(self.ego_state.global_velocity, "x", 2.)
        with self.assertRaisesRegex(RuntimeError, "post_release did not settle"):
            self.park()
        self.assertFalse(self.saved()["final_stop_verified"])
        self.ego_state.global_velocity.x = 0.
        def release_drift(*args):
            self.ego_state.transform.location.x += .25
        self.release_common_brake.side_effect = release_drift
        with self.assertRaisesRegex(RuntimeError, "post_release did not settle"):
            self.park()
        self.assertFalse(self.saved()["final_parking"]["final_verified"])

    def test_post_release_observations_require_half_second_and_reject_foreign_controller(self):
        self.prepare()
        self.park()
        record = self.saved()["final_parking"]
        post_release = [row for row in record["checks"] if row["phase"] == "post_release"]
        self.assertGreaterEqual(len(post_release), 3)
        self.assertGreaterEqual(record["elapsed_wall_seconds"], 2.-1e-8)
        def exclusive():
            if self.release_common_brake.call_count >= 3:
                raise RuntimeError("foreign controller")
        self._require_no_controller.side_effect = exclusive
        with self.assertRaisesRegex(RuntimeError, "foreign controller"):
            self.park()
        self.assertFalse(self.saved()["final_stop_verified"])
        self.assertFalse(self.saved()["final_parking"]["controller_publishers_absent"])

    def test_nonfinite_actuator_or_unavailable_readback_fails_without_retry(self):
        for error in (baseline._ParkingPending("invalid pedals", dict(self.stopped, brake=None)), RuntimeError("RPC unavailable")):
            with self.subTest(error=str(error)):
                self.prepare()
                self.common_stopped_readback.reset_mock()
                self.common_stopped_readback.side_effect = error
                with self.assertRaises(RuntimeError):
                    self.park()
                self.assertEqual(self.common_stopped_readback.call_count, 1)
                self.assertFalse(self.saved()["final_stop_verified"])

    def test_rejected_brake_command_or_live_owned_expert_cannot_be_verified(self):
        self.prepare()
        self.hold_common_brake.side_effect = RuntimeError("command was rejected")
        with self.assertRaisesRegex(RuntimeError, "command was rejected"):
            self.park()
        self.stub.GetActorState.assert_not_called()
        self.assertFalse(self.saved()["final_stop_verified"])
        self.hold_common_brake.side_effect = None
        self.context.expert_controller = SimpleNamespace(process=SimpleNamespace(poll=lambda: None))
        with self.assertRaisesRegex(RuntimeError, "Owned Expert did not stop"):
            self.park()
        self.assertFalse(self.saved()["final_parking"]["owned_expert_stopped"])

    def test_final_publisher_cleanup_failure_revokes_verified_stop(self):
        self.prepare()
        self.release_common_brake.side_effect = [None, RuntimeError("publisher unregister failed")]
        with self.assertRaisesRegex(RuntimeError, "publisher unregister failed"):
            self.park()
        report = self.saved()
        self.assertFalse(report["final_stop_verified"])
        self.assertFalse(report["final_parking"]["final_verified"])
        self.assertEqual(report["final_parking"]["status"], "failed")

    def test_run_failure_after_setup_or_cleanup_cannot_leave_a_passed_report(self):
        from ..scenarios.full_loop_scenario import FullLoopScenario
        from ..scenarios.segment_modes.scenario import MissionScenario

        self.prepare()
        scenario = MissionScenario.__new__(MissionScenario)
        scenario.__dict__.update(vars(self.context))
        for error in (RuntimeError("readiness failed"), RuntimeError("cleanup failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                scenario.tunnel_baseline_report.update(status="passed", baseline_drive_passed=True)
                with patch.object(FullLoopScenario, "run", side_effect=error):
                    with self.assertRaises(type(error)):
                        scenario.run()
                report = self.saved()
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["baseline_drive_passed"])
                self.assertEqual(report["failure_reason"], str(error) or type(error).__name__)

    def test_inventory_is_global_deadline_bounded_and_contains_known_ego(self):
        self.prepare()
        baseline.before_drive(self.context)
        query = self.context.grpc.world._sim_adapter._actor_stub.GetAllActorsState.call_args
        self.assertEqual(query.args[0].client_key, "")
        self.assertTrue(query.args[0].vehicle)
        self.assertTrue(query.args[0].pedestrian)
        self.assertTrue(query.args[0].obstacle)
        self.assertEqual(query.kwargs["timeout"], 1.0)
        self.assertEqual(self.saved()["inventory_checks"], 1)
        self.assertTrue(self.saved()["started_before_entry"])
        self.context.grpc.reset_actors.assert_not_called()
        self.context.grpc.destroy_vehicle_by_id.assert_not_called()

    def test_foreign_actor_in_opposing_lane_aborts_without_deleting_it(self):
        self.prepare()
        self.response([actor("Ego", -40.0, 0.0), actor("manual-car", 50.0, 3.5, client_key="manual")])
        with self.assertRaisesRegex(RuntimeError, "road is occupied: manual-car"):
            baseline.before_drive(self.context)
        report = self.saved()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["last_blocking_actors"][0]["client_key"], "manual")
        self.context.stop_drive_control.assert_called_once()
        self.context.grpc.destroy_vehicle_by_id.assert_not_called()
        self.context.grpc.enable_vehicle_spawn_point.assert_not_called()

    def test_bbox_center_and_extent_guard_detects_overlap_beyond_actor_pivot(self):
        self.prepare()
        # The pivot is outside the corridor but the actual world box overlaps it.
        self.response([actor("Ego", -40.0, 0.0), actor("wide-object", 50.0, 30.0, center=(50.0, 9.0), extent=(1.0, 4.0, 1.0))])
        with self.assertRaisesRegex(RuntimeError, "wide-object"):
            baseline.before_drive(self.context)
        blocker = self.saved()["last_blocking_actors"][0]
        self.assertEqual(blocker["world_bbox_center_xyz"], [50.0, 9.0, 28.0])

    def test_approach_and_exit_are_guarded_and_far_actors_are_allowed(self):
        self.prepare()
        for position in ((-30.0, 3.5), (125.0, 0.0)):
            with self.subTest(position=position):
                self.response([actor("Ego", -40.0, 0.0), actor("traffic", *position)])
                with self.assertRaisesRegex(RuntimeError, "traffic"):
                    baseline.before_drive(self.context)
        self.prepare()
        self.response([actor("Ego", -40.0, 0.0), actor("far-away", 50.0, 100.0)])
        baseline.before_drive(self.context)
        self.assertEqual(self.saved()["last_blocking_actors"], [])

    def test_inventory_failure_empty_reply_or_missing_ego_cannot_pass(self):
        for reply in (None, SimpleNamespace(), SimpleNamespace(states=[]), SimpleNamespace(states=[actor("other", 50.0, 100.0)])):
            with self.subTest(reply=reply):
                self.prepare()
                self.context.grpc.world._sim_adapter._actor_stub.GetAllActorsState.return_value = reply
                with self.assertRaisesRegex(RuntimeError, "inventory"):
                    baseline.before_drive(self.context)
                self.assertEqual(self.saved()["status"], "failed")
        self.prepare()
        self.context.grpc.world._sim_adapter._actor_stub.GetAllActorsState.side_effect = TimeoutError("deadline")
        with self.assertRaisesRegex(RuntimeError, "RPC failed: deadline"):
            baseline.before_drive(self.context)

    def test_invalid_actor_geometry_fails_closed(self):
        self.prepare()
        self.response([actor("Ego", -40.0, 0.0), actor("invalid", 50.0, float("nan"))])
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            baseline.before_drive(self.context)
        self.assertEqual(self.saved()["status"], "failed")

    def test_later_foreign_spawned_vehicle_aborts_periodic_observation(self):
        self.prepare()
        baseline.before_drive(self.context)
        self.response([actor("Ego", 20.0, 0.0), actor("native-traffic", 70.0, 3.5)])
        baseline.observe(self.context, 0.2, {"x": 20.0, "y": 0.0}, 60.0)
        with self.assertRaisesRegex(RuntimeError, "native-traffic"):
            baseline.observe(self.context, 0.5, {"x": 25.0, "y": 0.0}, 65.0)
        self.assertEqual(self.context.grpc.world._sim_adapter._actor_stub.GetAllActorsState.call_count, 2)
        self.assertFalse(self.saved()["baseline_drive_passed"])

    def test_pass_requires_observed_entry_inside_and_exit(self):
        self.prepare()
        baseline.before_drive(self.context)
        self.response([actor("Ego", 20.0, 0.0)])
        baseline.observe(self.context, 0.5, {"x": 20.0, "y": 0.0}, 60.0)
        self.response([actor("Ego", 115.0, 0.0)])
        baseline.observe(self.context, 1.0, {"x": 115.0, "y": 0.0}, 155.0)
        baseline.finish(self.context, "passed")
        report = self.saved()
        self.assertTrue(report["mapped_entry_observed"])
        self.assertTrue(report["mapped_exit_observed"])
        self.assertTrue(report["baseline_drive_passed"])
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["physical_portals_verified"])
        self.context.stop_drive_control.assert_not_called()

    def test_unobserved_tunnel_and_prior_failure_cannot_be_promoted_to_pass(self):
        self.prepare()
        baseline.before_drive(self.context)
        with self.assertRaisesRegex(RuntimeError, "without observing"):
            baseline.finish(self.context, "passed")
        baseline.finish(self.context, "failed", "collision")
        with self.assertRaisesRegex(RuntimeError, "collision"):
            baseline.finish(self.context, "passed")
        self.assertFalse(self.saved()["baseline_drive_passed"])

    def test_other_modes_and_unprepared_failure_are_safe_noops(self):
        baseline.finish(self.context, "failed", "route failed before preparation")
        self.context.mission_mode.name = "full_loop_normal"
        baseline.prepare(self.context)
        baseline.before_drive(self.context)
        baseline.observe(self.context, 0.0, {}, 0.0)
        baseline.finish(self.context, "passed")
        self.assertFalse(hasattr(self.context, "tunnel_baseline_report"))
        self.context.grpc.get_ego.assert_not_called()

    def test_prepare_rejects_road_type_mismatch_and_missing_route_margin(self):
        self.links[baseline.OPPOSING_LINK]["road_type"] = "1"
        with self.assertRaisesRegex(RuntimeError, "road_type=2"):
            baseline.prepare(self.context)
        self.links[baseline.OPPOSING_LINK]["road_type"] = "2"
        self.context.route_points = self.links[baseline.TUNNEL_LINK]["points"]
        with self.assertRaisesRegex(RuntimeError, "approach, mapped tunnel, and exit"):
            baseline.prepare(self.context)


if __name__ == "__main__":
    unittest.main()
