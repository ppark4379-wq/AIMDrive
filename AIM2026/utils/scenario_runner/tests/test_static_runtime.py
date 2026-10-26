import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from ..scenarios.full_loop_parts import static_blockage_release as release
from ..scenarios.full_loop_parts import static_obstacle


def obstacle_state(actor_id):
    location = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    return SimpleNamespace(actor_info=SimpleNamespace(id=SimpleNamespace(value=actor_id)),
                           transform=SimpleNamespace(location=location))


def make_context(policy=True, lights=()):
    context = SimpleNamespace(cfg={}, grpc=Mock(), event=Mock(), traffic_light_stops=list(lights))
    release.initialize(context)
    context.static_blockage_release_policy = release.validate_policy(policy)
    context.static_obstacle_expectations = [
        {"actor_id": "7", "model": "WoodBox", "route_s": 110.0},
        {"actor_id": "8", "model": "Bike1", "route_s": 112.0},
    ]
    context.grpc.destroy_obstacle_by_id.return_value = True
    context.grpc.get_all_obstacle_states.return_value = []
    return context


def events(context):
    return [call.args[0] for call in context.event.call_args_list]


def drive(context, samples):
    """samples: (monotonic, elapsed, speed, route_s)"""
    with patch.object(release.time, "monotonic", side_effect=[s[0] for s in samples]), \
            patch.object(static_obstacle.time, "sleep"):
        for _, elapsed, speed, route_s in samples:
            release.update(context, elapsed, {"speed": speed}, route_s)


class StaticBlockageReleaseTests(unittest.TestCase):
    def test_policy_defaults_bounds_and_unknown_keys(self):
        self.assertIsNone(release.validate_policy(None))
        self.assertEqual(release.validate_policy(True), release.DEFAULTS)
        self.assertEqual(release.validate_policy({"wait_wall_sec": 8})["wait_wall_sec"], 8.0)
        with self.assertRaises(ValueError):
            release.validate_policy({"wait_wall_sec": 0.1})
        with self.assertRaises(ValueError):
            release.validate_policy({"release_after": 3})

    def test_unarmed_run_never_touches_obstacles(self):
        context = make_context(policy=None)
        drive(context, [(100.0, 0.0, 0.0, 101.0), (110.0, 10.0, 0.0, 101.0)])
        context.grpc.destroy_obstacle_by_id.assert_not_called()
        context.event.assert_not_called()

    def test_stop_close_behind_layout_releases_every_obstacle_and_discloses_it(self):
        context = make_context()
        drive(context, [(100.0, 0.0, 0.0, 101.0), (106.5, 6.4, 0.1, 101.0)])
        self.assertEqual([call.args[0] for call in context.grpc.destroy_obstacle_by_id.call_args_list],
                         ["7", "8"])
        self.assertTrue(context.static_blockage_released)
        self.assertEqual(events(context), ["static_blockage_release_armed", "static_blockage_wait_started",
                                           "actor_lifecycle_started", "actor_lifecycle_finished",
                                           "static_blockage_released"])
        released = context.event.call_args_list[-1].kwargs
        self.assertTrue(released["intervention"] and released["mission_distribution_changed"])
        self.assertFalse(released["training_visible"])
        self.assertEqual(released["actor_ids"], ["7", "8"])
        self.assertAlmostEqual(released["gap_m"], 9.0)
        self.assertTrue(context.event.call_args_list[3].kwargs["success"])
        # Nothing happens after the release.
        drive(context, [(120.0, 20.0, 0.0, 101.0)])
        self.assertEqual(context.grpc.destroy_obstacle_by_id.call_count, 2)

    def test_wall_time_alone_is_not_enough_when_simulation_runs_slow(self):
        context = make_context()
        drive(context, [(100.0, 0.0, 0.0, 101.0), (106.5, 2.0, 0.0, 101.0)])   # 6.5 s wall, 2 s simulated
        context.grpc.destroy_obstacle_by_id.assert_not_called()
        drive(context, [(119.0, 6.1, 0.0, 101.0)])
        self.assertTrue(context.static_blockage_released)
        released = context.event.call_args_list[-1].kwargs
        self.assertAlmostEqual(released["wait_scenario_sec"], 6.1)
        self.assertAlmostEqual(released["scenario_elapsed_sec"], 6.1)

    def test_stop_far_before_layout_is_not_a_blockage_wait(self):
        context = make_context()
        drive(context, [(100.0, 0.0, 0.0, 90.0), (110.0, 10.0, 0.0, 90.0)])   # 20 m: a stop-line wait
        context.grpc.destroy_obstacle_by_id.assert_not_called()
        self.assertIsNone(context.static_blockage_wait)

    def test_red_stop_line_ahead_explains_the_stop(self):
        context = make_context(lights=[{"light_id": "L1", "route_s": 104.0}])
        context.grpc.get_traffic_light_state_by_id.return_value = {"id": "L1", "color": 1}
        drive(context, [(100.0, 0.0, 0.0, 101.0), (110.0, 10.0, 0.0, 101.0)])
        context.grpc.destroy_obstacle_by_id.assert_not_called()
        context.grpc.get_traffic_light_state_by_id.return_value = {"id": "L1", "color": 16}
        drive(context, [(111.0, 11.0, 0.0, 101.0), (117.5, 17.5, 0.0, 101.0)])
        self.assertTrue(context.static_blockage_released)

    def test_release_at_a_used_stop_line_lets_the_signal_override_act_again(self):
        context = make_context(lights=[{"light_id": "L1", "route_s": 101.5}])
        context.grpc.get_traffic_light_state_by_id.return_value = {"id": "L1", "color": 16}
        context.traffic_override_completed = {("L1", 101.5), ("L9", 300.0)}
        context.traffic_wait_state = {"stop_key": ("L1", 101.5)}
        drive(context, [(100.0, 0.0, 0.0, 101.0), (106.5, 6.5, 0.0, 101.0)])
        self.assertTrue(context.static_blockage_released)
        self.assertEqual(context.traffic_override_completed, {("L9", 300.0)})
        self.assertIsNone(context.traffic_wait_state)
        self.assertIn("static_blockage_signal_override_rearmed", events(context))

    def test_moving_resets_the_wait(self):
        context = make_context()
        drive(context, [(100.0, 0.0, 0.0, 101.0), (104.0, 4.0, 1.5, 101.5),
                        (105.0, 5.0, 0.0, 102.0), (110.0, 10.0, 0.0, 102.0)])
        context.grpc.destroy_obstacle_by_id.assert_not_called()
        self.assertIn("static_blockage_wait_reset", events(context))
        drive(context, [(111.5, 11.5, 0.0, 102.0)])
        self.assertTrue(context.static_blockage_released)

    def test_destroy_applied_on_a_later_simulation_step_is_awaited(self):
        context = make_context()
        both = [obstacle_state("7"), obstacle_state("8")]
        context.grpc.get_all_obstacle_states.side_effect = [both, both, [obstacle_state("8")]] + [[]] * 5
        drive(context, [(100.0, 0.0, 0.0, 101.0), (106.5, 6.5, 0.0, 101.0)])
        self.assertTrue(context.static_blockage_released)
        removed = context.event.call_args_list[-1].kwargs["removed"]
        self.assertTrue(all(row["absent_after_readback"] for row in removed))
        self.assertEqual(removed[0]["inventory_polls"], 5)

    def test_obstacle_still_in_inventory_fails_closed(self):
        context = make_context()
        context.grpc.get_all_obstacle_states.return_value = [obstacle_state("8")]
        with self.assertRaises(RuntimeError):
            drive(context, [(100.0, 0.0, 0.0, 101.0), (107.0, 7.0, 0.0, 101.0)])
        self.assertFalse(context.static_blockage_released)
        finished = [c for c in context.event.call_args_list if c.args[0] == "actor_lifecycle_finished"]
        self.assertFalse(finished[0].kwargs["success"])

    def test_released_layout_is_not_verified_as_vanished(self):
        context = SimpleNamespace(cfg={}, grpc=Mock(), static_blockage_released=True,
                                  static_obstacle_expectations=[{"actor_id": "7"}])
        static_obstacle.verify_static_obstacle(context, 10.0, 100.0)
        context.grpc.get_all_obstacle_states.assert_not_called()

    def test_preset_declares_the_release_policy(self):
        document = {"schema": "morai_static_obstacle_preset_v1", "name": "blocked_01",
                    "map_name": "TestMap",
                    "obstacles": [{"model": "WoodBox", "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                                   "scale": [1.0, 1.0, 1.0], "yaw_deg": 5.0}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.yaml"
            path.write_text(yaml.safe_dump(document))
            plain = static_obstacle.load_static_obstacle_preset(path, "TestMap", [(0.0, 0.0), (1.0, 2.0)])
            document["blockage_release"] = {"wait_wall_sec": 5.0}
            path.write_text(yaml.safe_dump(document))
            blocked = static_obstacle.load_static_obstacle_preset(path, "TestMap", [(0.0, 0.0), (1.0, 2.0)])
        self.assertIsNone(plain["blockage_release"])
        self.assertEqual(blocked["blockage_release"]["wait_wall_sec"], 5.0)
        self.assertEqual(blocked["blockage_release"]["max_gap_m"], release.DEFAULTS["max_gap_m"])


import math

import numpy as np

from ..core.full_loop_drive_base import FullLoopDriveBase
from ..tools import static_lane_compliance as lane


FIRST = lane.SEGMENT_INDEX[0]
# A straight northbound official route: index i sits 0.5 m north of i-1.
ROUTE = (np.array([[0.0, 0.5 * k] for k in range(lane.SEGMENT_INDEX[1] - FIRST + 1)]), FIRST)
NORTH = math.pi / 2


def line(idx, lane_type, shape, points, color="white"):
    return {"idx": idx, "lane_type": [lane_type], "lane_color": [color],
            "lane_shape": [shape], "lane_width_m": 0.15, "points_xy": points}


def markings(*lines):
    return {"schema": "morai_static_segment_lane_markings_v1", "source": {"sha256": "x"},
            "lines": list(lines)}


# Yellow centre 1.7 m left, white edge 1.7 m right, and one stop line.
ROAD = markings(
    line("Y", 501, "solid", [[-1.7, 0.0], [-1.7, 60.0]], "yellow"),
    line("G", 525, "broken", [[-1.7, 60.0], [-1.7, 80.0]]),
    line("Y2", 501, "solid", [[-1.7, 80.0], [-1.7, 140.0]], "yellow"),
    line("W", 505, "solid", [[1.7, 0.0], [1.7, 140.0]]),
    line("S", 530, "solid", [[-1.7, 30.0], [1.7, 30.0]]),
    line("B", 503, "broken", [[5.0, 0.0], [5.0, 140.0]]),
)


def straight(x, y0=0.0, y1=134.0, heading=NORTH, step=0.5):
    return [(x, y, heading) for y in np.arange(y0, y1, step)]


def swing(peak_m, start=45.0, end=75.0, ramp=10.0):
    """Move right by peak_m between start and end with linear ramps."""
    rows = []
    for x, y, t in straight(0.0):
        weight = min(max((y - start) / ramp, 0.0), 1.0, max((end - y) / ramp, 0.0))
        rows.append((peak_m * weight, y, t))
    return rows


class StaticLaneComplianceTest(unittest.TestCase):
    def test_no_cross_lines_are_solid_lines_and_centre_guides_but_not_stop_lines(self):
        _, _, labels = lane.no_cross_segments(ROAD)
        kinds = {label.split(":")[0]: label.split(":")[2] for label in labels}
        self.assertEqual(kinds, {"Y": "solid", "G": "centre_guide", "Y2": "solid", "W": "solid"})

    def test_centred_lane_keeping_passes(self):
        result = lane.evaluate_trace(straight(0.0), ROAD, ROUTE)
        self.assertTrue(result["passed"], result["reasons"])
        # 1.7 m to the line centre - 0.945 m half body - 0.075 m half paint.
        self.assertAlmostEqual(result["min_marking_clearance_m"], 0.68, places=2)
        self.assertEqual(result["avoidance_class"], "small")

    def test_body_on_the_paint_is_rejected_on_either_side(self):
        # x grows to the east; heading north keeps +x on the Ego's right.
        right = lane.evaluate_trace(straight(0.70), ROAD, ROUTE)
        self.assertIn("solid_lane_marking_touched", right["reasons"])
        self.assertTrue(right["touched_markings"][0]["marking"].startswith("W:"))
        inside = lane.evaluate_trace(straight(0.66), ROAD, ROUTE)
        self.assertTrue(inside["passed"], inside["reasons"])
        # Crossing the intersection guide that continues the centre line counts.
        left = [(x if not 62.0 <= y <= 70.0 else -0.8, y, t) for x, y, t in straight(0.0)]
        result = lane.evaluate_trace(left, ROAD, ROUTE)
        self.assertEqual([row["marking"].split(":")[0] for row in result["touched_markings"]], ["G"])

    def test_swing_beyond_the_limit_is_too_far_and_classified(self):
        road = markings(line("Y", 501, "solid", [[-1.7, 0.0], [-1.7, 140.0]], "yellow"))
        medium = lane.evaluate_trace(swing(3.6), road, ROUTE)
        self.assertTrue(medium["passed"], medium["reasons"])
        self.assertEqual(medium["avoidance_class"], "medium")
        result = lane.evaluate_trace(swing(4.5), road, ROUTE)
        self.assertIn("lateral_deviation_limit_exceeded", result["reasons"])
        self.assertEqual(result["avoidance_class"], "large")

    def test_samples_after_the_goal_are_not_a_deviation(self):
        # After the goal the Ego may drive on; those samples project onto the end point.
        trace = straight(0.0) + [(0.3 * k, 134.0 + 0.3 * k, NORTH) for k in range(20)]
        result = lane.evaluate_trace(trace, markings(), ROUTE)
        self.assertTrue(result["passed"], result["reasons"])
        self.assertLess(result["max_right_deviation_m"], 0.1)

    def test_missing_incomplete_or_gapped_traces_fail_closed(self):
        self.assertEqual(lane.evaluate_trace([], ROAD, ROUTE)["reasons"], ["ego_trace_missing"])
        short = lane.evaluate_trace(straight(0.0, 0.0, 40.0), ROAD, ROUTE)
        self.assertIn("ego_trace_incomplete", short["reasons"])
        gap = [row for row in straight(0.0) if not 70.0 < row[1] < 75.0]
        self.assertIn("ego_trace_gap", lane.evaluate_trace(gap, ROAD, ROUTE)["reasons"])

    def test_active_signal_rows_collapse_to_one_pose_per_source_time(self):
        row = {"role": "ego", "source_timestamp_ns": 2,
               "signal": {"position": [1.0, 2.0, 0.0], "yaw": 0.5}}
        rows = [row, dict(row), dict(row, source_timestamp_ns=1), {"role": "collision"}]
        self.assertEqual(lane.ego_trace_from_active_signals(rows),
                         [(1.0, 2.0, 0.5), (1.0, 2.0, 0.5)])

    def test_pinned_kcity_markings_match_the_route_and_centre_line(self):
        pinned = lane.load_markings()
        route = lane.load_route()
        self.assertEqual(pinned["route"]["segment_index"], list(lane.SEGMENT_INDEX))
        self.assertEqual(pinned["route"]["sha256"], lane.sha256_file(lane.DEFAULT_ROUTE))
        for index in (660, 670, 680):
            self.assertAlmostEqual(lane.centre_line_lateral(pinned, *route, index), 1.71, delta=0.05)

    def test_runner_writes_one_trace_row_per_tick(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace" / "ego.csv"
            scenario = SimpleNamespace(cfg={"ego_trace_path": str(path)})
            for elapsed in (0.0, 0.05):
                FullLoopDriveBase.record_ego_trace(
                    scenario, elapsed, {"x": 1.0, "y": 2.0 + elapsed, "yaw_deg": 90.0, "speed": 3.0})
            scenario._ego_trace_stream.close()
            rows = lane.read_trace_csv(path)
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(rows[1][1], 2.05)
            self.assertAlmostEqual(rows[0][2], NORTH)
            silent = SimpleNamespace(cfg={})
            FullLoopDriveBase.record_ego_trace(silent, 0.0, {})
            self.assertFalse(hasattr(silent, "_ego_trace_stream"))


# Read-only guard tests: deterministic RPC sequences, no MORAI connection.
import io
import json
import random
from contextlib import redirect_stdout

from ..main_runner import PACKAGE_ROOT, load_configuration, MoraiGrpcClient
from ..scenarios.full_loop_parts import static_obstacle_watch as watch


def watched_obstacle_state(actor_id="static-1", x=50.0, y=0.0, z=28.0):
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
        self.stub.GetAllActorsState.return_value = SimpleNamespace(states=[watched_obstacle_state()])
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
        self.tick([watched_obstacle_state()])
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
        dropped = [watched_obstacle_state(z=25.0)]
        self.tick(dropped)
        self.tick(dropped)
        with self.assertRaisesRegex(RuntimeError, "status=z_deviation"):
            self.tick(dropped)
        self.assertEqual(self.records()[-1]["actual_xyz"], [50.0, 0.0, 25.0])

    def test_xy_displacement_is_detected(self):
        watch.arm(self.context)
        moved = [watched_obstacle_state(x=54.0)]
        self.tick(moved)
        self.tick(moved)
        with self.assertRaisesRegex(RuntimeError, "status=xy_deviation"):
            self.tick(moved)

    def test_other_actor_is_not_adopted_as_replacement(self):
        watch.arm(self.context)
        self.tick([watched_obstacle_state(actor_id="different-id")])
        self.assertEqual(self.records()[-1]["status"], "missing")
        self.assertEqual(self.context.static_obstacle_watch.actor_id, "static-1")

    def test_invalid_and_duplicate_states_are_not_healthy(self):
        watch.arm(self.context)
        for states, status in (
            ([watched_obstacle_state(z=float("nan"))], "invalid_position"),
            ([watched_obstacle_state(), watched_obstacle_state()], "duplicate_id"),
        ):
            self.tick(states)
            self.assertEqual(self.records()[-1]["status"], status)
        self.tick([watched_obstacle_state()])
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
