"""Offline tests of the tunnel's global inventory and empty-road contract."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ..scenarios.segment_modes import tunnel_baseline as baseline


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
            "get_id", "set_velocity", "set_vehicle_gear", "get_actor_state",
        ))
        grpc.get_ego.return_value.get_id.return_value = "Ego"
        grpc.get_ego.return_value.set_velocity.return_value = True
        grpc.get_ego.return_value.set_vehicle_gear.return_value = True
        grpc.get_ego.return_value.get_actor_state.return_value = SimpleNamespace(
            global_velocity=xyz(0.0, 0.0, 0.0),
            vehicle_state=SimpleNamespace(gear_mode=1),
        )
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
            "proto.morai.actor.actor_enum_pb2": SimpleNamespace(GEAR_MODE_P=1),
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

    def test_final_parking_checks_observed_motion_and_gear(self):
        self.prepare()
        baseline.park(self.context)
        self.assertTrue(self.saved()["ego_stopped_and_parked"])
        self.context.grpc.get_ego.return_value.get_actor_state.return_value.global_velocity.x = 2.0
        with self.assertRaisesRegex(RuntimeError, "stop/parking verification failed"):
            baseline.park(self.context)

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
