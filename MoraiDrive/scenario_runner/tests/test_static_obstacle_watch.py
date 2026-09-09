"""Read-only guard tests: deterministic RPC sequences, no MORAI connection."""
import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ..main_runner import PACKAGE_ROOT, load_configuration, MoraiGrpcClient
from ..scenarios.full_loop_parts import static_obstacle_watch as watch


def obstacle_state(actor_id="static-1", x=50.0, y=0.0, z=28.0):
    return SimpleNamespace(
        actor_info=SimpleNamespace(id=SimpleNamespace(value=actor_id)),
        transform=SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=z)),
    )


class StaticObstacleWatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()):
            config, _, _ = load_configuration()
            MoraiGrpcClient(config)  # Import protobufs; no connection.

    def setUp(self):
        (PACKAGE_ROOT / ".runtime").mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(
            prefix="static-watch-test-", dir=PACKAGE_ROOT / ".runtime",
        )
        self.addCleanup(self.tmp.cleanup)
        self.clock = patch.object(watch.time, "monotonic", return_value=0.0).start()
        self.addCleanup(patch.stopall)
        self.output = io.StringIO()
        self.redirect = redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)
        self.stub = Mock()
        self.stub.GetAllActorsState.return_value = SimpleNamespace(states=[obstacle_state()])
        self.actor = Mock()
        self.actor.get_id.return_value = "static-1"
        self.context = SimpleNamespace(
            grpc=SimpleNamespace(
                client_key="static-watch-test",
                world=SimpleNamespace(_sim_adapter=SimpleNamespace(_actor_stub=self.stub)),
            ),
            route_definition={"zones": {"static_obstacle": {
                "position": dict(x=50.0, y=0.0, z=28.0),
            }}},
            route_points=[(float(i), 0.0, 28.0) for i in range(201)],
            global_cfg={"paths": {"runtime_root": self.tmp.name}},
            static_obstacle=self.actor, static_obstacle_model="RedBarrel",
            static_obstacle_scale=[2.0, 3.0, 2.0],
            scenario_name="full_loop_static", run_instance_id="test-run",
            run_seed=123, random=random.Random(123), stop_drive_control=Mock(),
        )

    def tick(self, states=None, current_s=0.0):
        self.clock.return_value += 0.21
        if states is not None:
            self.stub.GetAllActorsState.return_value = SimpleNamespace(states=states)
        watch.update(self.context, current_s)

    def records(self):
        return [json.loads(line) for line in
                self.context.static_obstacle_watch.log_path.read_text().splitlines()]

    def test_healthy_monitor_does_not_mutate_actor_or_random_state(self):
        random_state = self.context.random.getstate()
        original_points = list(self.context.route_points)
        watch.arm(self.context)
        for progress in (10.0, 20.0, 40.0, 64.0):
            self.tick(current_s=progress)
        self.assertEqual(self.context.random.getstate(), random_state)
        self.assertEqual(self.context.route_points, original_points)
        self.assertEqual(self.context.static_obstacle_model, "RedBarrel")
        self.assertEqual(self.context.static_obstacle_scale, [2.0, 3.0, 2.0])
        self.actor.destroy.assert_not_called()
        self.actor.set_transform.assert_not_called()
        self.actor.set_scale.assert_not_called()
        self.context.stop_drive_control.assert_not_called()
        self.assertEqual({entry["status"] for entry in self.records()
                          if entry["event"] == "sample"}, {"ok"})

    def test_flash_then_missing_aborts_after_three_observations(self):
        watch.arm(self.context)  # The initial actor really exists.
        self.tick([])
        self.tick([])
        self.context.stop_drive_control.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "status=missing"):
            self.tick([])
        self.context.stop_drive_control.assert_called_once()
        self.actor.destroy.assert_not_called()  # Cleanup is the existing lifecycle's job.
        self.assertEqual(self.records()[-1]["event"], "fatal")
        self.assertIn("model=RedBarrel", self.output.getvalue())

    def test_transient_missing_recovers_without_respawn_or_stop(self):
        watch.arm(self.context)
        self.tick([])
        self.tick([obstacle_state()])
        self.assertEqual(self.context.static_obstacle_watch.failures, 0)
        self.context.stop_drive_control.assert_not_called()
        self.assertEqual([row["status"] for row in self.records()
                          if row["event"] == "sample"], ["ok", "missing", "ok"])

    def test_query_failure_is_not_reported_as_missing(self):
        watch.arm(self.context)
        self.stub.GetAllActorsState.side_effect = TimeoutError("RPC deadline exceeded")
        self.tick()
        self.tick()
        with self.assertRaisesRegex(RuntimeError, "status=query_error"):
            self.tick()
        self.assertEqual(self.records()[-1]["status"], "query_error")

    def test_z_drop_is_detected(self):
        watch.arm(self.context)
        dropped = [obstacle_state(z=25.0)]
        self.tick(dropped)
        self.tick(dropped)
        with self.assertRaisesRegex(RuntimeError, "status=z_deviation"):
            self.tick(dropped)
        self.assertEqual(self.records()[-1]["actual_xyz"], [50.0, 0.0, 25.0])

    def test_xy_displacement_is_detected(self):
        watch.arm(self.context)
        moved = [obstacle_state(x=54.0)]
        self.tick(moved)
        self.tick(moved)
        with self.assertRaisesRegex(RuntimeError, "status=xy_deviation"):
            self.tick(moved)

    def test_other_actor_is_not_adopted_as_replacement(self):
        watch.arm(self.context)
        self.tick([obstacle_state(actor_id="different-id")])
        self.assertEqual(self.records()[-1]["status"], "missing")
        self.assertEqual(self.context.static_obstacle_watch.actor_id, "static-1")

    def test_invalid_and_duplicate_states_are_not_healthy(self):
        watch.arm(self.context)
        for states, status in (
            ([obstacle_state(z=float("nan"))], "invalid_position"),
            ([obstacle_state(), obstacle_state()], "duplicate_id"),
        ):
            self.tick(states)
            self.assertEqual(self.records()[-1]["status"], status)
        self.tick([obstacle_state()])
        self.context.stop_drive_control.assert_not_called()

    def test_periodic_sampling_and_rpc_deadline(self):
        watch.arm(self.context)
        count = self.stub.GetAllActorsState.call_count
        self.clock.return_value = 0.1
        watch.update(self.context, 1.0)
        self.assertEqual(self.stub.GetAllActorsState.call_count, count)
        request = self.stub.GetAllActorsState.call_args.args[0]
        self.assertEqual(request.client_key, "static-watch-test")
        self.assertTrue(request.obstacle)
        self.assertFalse(request.vehicle)
        self.assertFalse(request.pedestrian)
        self.assertEqual(self.stub.GetAllActorsState.call_args.kwargs["timeout"], 0.3)

    def test_guard_finishes_before_other_zones_and_does_not_remove_obstacle(self):
        watch.arm(self.context)
        self.tick(current_s=65.0)
        self.assertFalse(self.context.static_obstacle_watch.active)
        count = self.stub.GetAllActorsState.call_count
        self.tick([], current_s=180.0)
        self.assertEqual(self.stub.GetAllActorsState.call_count, count)
        self.actor.destroy.assert_not_called()
        self.context.stop_drive_control.assert_not_called()
        self.assertEqual(self.records()[-1]["reason"], "ego_cleared_static_section")

    def test_legacy_after_static_route_is_not_monitored(self):
        self.context.route_points = self.context.route_points[80:]
        watch.arm(self.context)
        self.assertIsNone(self.context.static_obstacle_watch)
        self.stub.GetAllActorsState.assert_not_called()

    def test_no_static_actor_means_no_guard_query(self):
        watch.update(SimpleNamespace(), 100.0)
        watch.finish(SimpleNamespace(), "cleanup")
        self.stub.GetAllActorsState.assert_not_called()

    def test_cleanup_disarms_without_a_final_query(self):
        watch.arm(self.context)
        count = self.stub.GetAllActorsState.call_count
        watch.finish(self.context, "scenario_cleanup")
        self.tick([])
        self.assertEqual(self.stub.GetAllActorsState.call_count, count)
        self.assertEqual(self.records()[-1]["reason"], "scenario_cleanup")

    def test_log_write_failure_does_not_suppress_fault_stop(self):
        watch.arm(self.context)
        with patch.object(Path, "open", side_effect=OSError("disk full")):
            self.tick([])
            self.tick([])
            with self.assertRaisesRegex(RuntimeError, "status=missing"):
                self.tick([])
        self.context.stop_drive_control.assert_called_once()
        self.assertIn("log write failed", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
