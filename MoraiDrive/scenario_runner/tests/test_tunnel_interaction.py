"""Tunnel scheduling, conservative departure gates and actor lifecycle regressions."""

import contextlib
import copy
import io
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ..scenarios.full_loop_parts import tunnel_dynamic
from ..scenarios.full_loop_parts import tunnel_dynamic_timing as timing


class _TunnelFixture:
    """Shared scenario helpers with assertions owned by the running test."""

    def __init__(self, test_case):
        self.assertTrue = test_case.assertTrue

    @staticmethod
    def state(x, y):
        return SimpleNamespace(
            actor_info=SimpleNamespace(
                id=SimpleNamespace(value="ped-1"), object_type=2),
            transform=SimpleNamespace(location=SimpleNamespace(x=x, y=y)),
        )

    def context(self, candidate=None, model="Man1"):
        actor = Mock()
        actor.get_id.return_value = "ped-1"
        actor.destroy.return_value = None
        actor.control.return_value = True
        grpc = Mock()
        grpc.get_available_pedestrian_models.return_value = [model]
        grpc.get_all_pedestrian_states.return_value = []
        grpc.spawn_pedestrian.return_value = actor
        grpc.set_pedestrian_waypoint.return_value = True
        grpc.control_pedestrian.return_value = True
        pedestrian = dict(
            enabled=True, model=model, path_id="walk_then_cross",
            candidate_policy="trial", speed_mps=1.5,
            arrival_tolerance_m=.5, max_run_sec=20.,
        )
        zone = dict(dynamic_paths={"walk_then_cross": dict(
            route_path_index=50, waypoints=[
                dict(x=45., y=2., z=0.),
                dict(x=50., y=2., z=0.),
                dict(x=50., y=-2., z=0.),
            ])}, candidate_catalog=dict(
                division_count=4, endpoint_margin_m=10., side_offset_m=2.5,
                opposite_lane_1_offset_m=2.5, opposite_lane_2_offset_m=0.,
                straight_length_m=15., cross_delta_s_m=3., diagonal_delta_s_m=8.,
                approved_ids=[], disabled_ids=[],
            ))
        if candidate is not None:
            pedestrian.update(trial_candidate_id=candidate, speed_mps=2.5)
        context = SimpleNamespace(
            cfg=dict(mission_mode="full_loop_tunnel_dynamic",
                     drive_control_mode="aimdrive_expert", tunnel_dynamic=pedestrian),
            scenario_name="full_loop_tunnel_dynamic",
            mission_mode=SimpleNamespace(
                name="full_loop_tunnel_dynamic", zones=SimpleNamespace(tunnel=True)),
            route_definition={"zones": {"tunnel": zone}},
            route_points=[(float(i), 0., 0.) for i in range(201)],
            path_s_at_index=lambda index: float(index),
            tunnel_entry_s=10., tunnel_exit_s=190., grpc=grpc,
            scenario_actors=[], run_seed=123, run_instance_id="interaction-123",
        )
        self.assertTrue(tunnel_dynamic.prepare_tunnel_dynamic(context))
        tunnel_dynamic.arm_tunnel_dynamic(context)
        return context, actor

    def start(self, context, ego_s=31.):
        self.tick(context, 0., ego_s, 4.)
        self.assertTrue(context.tunnel_dynamic_result["spawned"])

    def tick(self, context, elapsed, ego_s, speed=None, npc=None, raw_speed=None):
        if npc is not None:
            context.tunnel_dynamic_actor.get_actor_state.return_value = self.state(*npc)
        ego = dict(x=ego_s, y=0.)
        if speed is not None:
            ego["speed_mps"] = speed
            ego["speed"] = speed * 3.6 if raw_speed is None else raw_speed
        return tunnel_dynamic.update_tunnel_dynamic(context, elapsed, ego, ego_s)

    def cross_to_endpoint(self, context, speed=4., ego_s=27.):
        for elapsed, npc in ((1., (46.5, 2.)), (2., (48., 2.)),
                             (10. / 3., (50., 2.)), (4., (50., 1.)),
                             (5., (50., -.5)), (6., (50., -2.))):
            self.tick(context, elapsed, ego_s, speed, npc)

    def background_before_crossing(self):
        context, actor = self.context("straight_same_direction_01")
        future, _ = self.context("cross_left_to_right_02")
        context.tunnel_dynamic_plans.append(future.tunnel_dynamic_plan)
        context.tunnel_dynamic_results.append(future.tunnel_dynamic_result)
        return context, actor


class _TunnelRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.sleep = patch.object(tunnel_dynamic.time, "sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)
        self.fixture = _TunnelFixture(self)


class _Actor:
    def __init__(self, grpc, identity, transform, velocity):
        self.grpc, self.identity = grpc, identity
        self.x, self.y, self.z = (getattr(transform.location, axis) for axis in 'xyz')
        self.yaw = math.radians(transform.rotation.z)
        self.speed = velocity / 3.6
        self.destroy_calls = 0

    def get_id(self):
        return self.identity

    def get_actor_state(self):
        if self.identity in self.grpc.unavailable_ids:
            return None
        return self.state()

    def state(self):
        return SimpleNamespace(actor_info=SimpleNamespace(
            id=SimpleNamespace(value=self.identity), object_type=2),
            transform=SimpleNamespace(location=SimpleNamespace(x=self.x, y=self.y, z=self.z)))

    def control(self, rotation, velocity):
        self.yaw, self.speed = math.radians(rotation.z), velocity / 3.6
        return True

    def destroy(self):
        self.destroy_calls += 1
        self.grpc.live.pop(self.identity, None)


class _Grpc:
    def __init__(self):
        self.elapsed = 0.0
        self.live = {}
        self.spawned = []
        self.spawn_requests = []
        self.unavailable_ids = set()
        self.stop_calls = []
        self.inventory_history = []

    @staticmethod
    def get_available_pedestrian_models():
        return ['Man1']

    @staticmethod
    def make_transform(x, y, z, yaw):
        return SimpleNamespace(location=SimpleNamespace(x=x, y=y, z=z),
                               rotation=SimpleNamespace(x=0.0, y=0.0, z=yaw))

    def spawn_pedestrian(self, **kwargs):
        actor = _Actor(self, 'owned-%d' % (len(self.spawned) + 1),
                       kwargs['transform'], kwargs['velocity'])
        self.spawned.append(actor)
        self.spawn_requests.append(kwargs)
        self.live[actor.identity] = actor
        return actor

    def get_all_pedestrian_states(self, strict=False):
        self.inventory_history.append(tuple(sorted(self.live)))
        return [actor.state() for identity, actor in self.live.items()
                if identity not in self.unavailable_ids]

    def control_pedestrian(self, actor, dx, dy, speed, **kwargs):
        self.stop_calls.append((actor.identity, speed))
        actor.speed = speed
        return True

    def advance_to(self, elapsed):
        dt = elapsed - self.elapsed
        if dt < 0:
            raise AssertionError('test clock must advance')
        for actor in self.live.values():
            actor.x += actor.speed * dt * math.cos(actor.yaw)
            actor.y += actor.speed * dt * math.sin(actor.yaw)
        self.elapsed = elapsed


class _OverlapFixture:
    def context(self, second='diagonal_cross_02', lead=None):
        context, _ = self.fixture.context('straight_same_direction_01')
        context.route_definition['zones']['tunnel']['candidate_catalog']['straight_length_m'] = 20.0
        if lead is not None:
            context.cfg['tunnel_dynamic']['interaction_trigger'] = {'interaction_lead_sec': lead}
        tunnel_dynamic.prepare_tunnel_dynamic(context)
        first = context.tunnel_dynamic_plan
        future, _ = self.fixture.context(second)
        context.tunnel_dynamic_plans = [first, future.tunnel_dynamic_plan]
        context.grpc = _Grpc()
        context.scenario_actors = []
        context.expert_controller = SimpleNamespace(max_velocity_kmh=60.0)
        tunnel_dynamic.arm_tunnel_dynamic(context)
        self.assertEqual(len(context.tunnel_dynamic_events), 2)
        return context

    @staticmethod
    def tick(context, elapsed, ego_s, speed):
        context.grpc.advance_to(elapsed)
        tunnel_dynamic.update_tunnel_dynamic(context, elapsed,
            dict(x=ego_s, y=0.0, speed_mps=speed), ego_s)

    def start_first(self, context):
        self.tick(context, 0.0, 18.0, 8.0)
        self.assertEqual(len(context.grpc.spawned), 1)
        self.assertTrue(context.tunnel_dynamic_results[0]['spawned'])

    def overlap(self, context):
        self.start_first(context)
        self.tick(context, 5.1125, 58.9, 8.0)
        self.assertEqual(len(context.grpc.spawned), 2)
        self.assertEqual(len(context.grpc.live), 2)
        return context.tunnel_dynamic_events

    def crossing_followed_by_side(self):
        plans = []
        # The same behavior sequence as the observed seed-2705 failure:
        # a crossing, two long side walks, then another crossing. Their time
        # budget can open the side's own gate before Ego passes the first.
        for candidate in ('diagonal_cross_01', 'straight_same_direction_02',
                          'straight_opposite_03', 'cross_left_to_right_04'):
            context, _ = self.fixture.context(candidate)
            context.route_definition['zones']['tunnel']['candidate_catalog']['straight_length_m'] = 20.0
            tunnel_dynamic.prepare_tunnel_dynamic(context)
            plans.append(context.tunnel_dynamic_plan)
        context.tunnel_dynamic_plans = plans
        context.grpc = _Grpc()
        context.scenario_actors = []
        context.expert_controller = SimpleNamespace(max_velocity_kmh=60.0)
        tunnel_dynamic.arm_tunnel_dynamic(context)
        self.tick(context, 0.0, 18.9, 8.0)
        self.assertEqual(len(context.grpc.spawned), 1)
        return context


class TunnelInteractionTest(_TunnelRuntimeTest):
    def test_background_before_next_crossing_waits_for_ego_approach(self):
        context, actor = self.fixture.background_before_crossing()
        self.fixture.tick(context, 0., 15., 1.67)
        self.assertEqual(context.tunnel_dynamic_phase, "waiting")
        self.assertFalse(context.tunnel_dynamic_result["interaction_required"])
        context.grpc.spawn_pedestrian.assert_not_called()
        self.assertEqual(context.scenario_actors, [])
        self.fixture.tick(context, 1., 15., 4.)
        self.assertGreater(context.tunnel_dynamic_result["last_timing"]["earliest_spawn_delay_sec"], 0.)
        context.grpc.spawn_pedestrian.assert_not_called()

    def test_early_background_keeps_maximum_distance_bound(self):
        context, _ = self.fixture.background_before_crossing()
        self.fixture.tick(context, 0., -21., 4.)
        timing = context.tunnel_dynamic_result["last_timing"]
        self.assertEqual(timing["policy"], "tunnel_trajectory_schedule")
        self.assertGreater(timing["remaining_m"], 60.)
        self.assertEqual(context.tunnel_dynamic_phase, "waiting")
        context.grpc.spawn_pedestrian.assert_not_called()

    def test_early_background_keeps_stopping_distance_guard(self):
        context, _ = self.fixture.background_before_crossing()
        with self.assertRaisesRegex(RuntimeError, "safe_stop"):
            self.fixture.tick(context, 0., 30., 12.)
        self.assertEqual(context.tunnel_dynamic_phase, "failed")
        context.grpc.spawn_pedestrian.assert_not_called()

    def test_standalone_background_keeps_its_existing_eta_and_speed_gates(self):
        context, _ = self.fixture.context("straight_same_direction_01")
        self.fixture.tick(context, 0., 15., 1.67)
        self.assertEqual(context.tunnel_dynamic_result["last_timing"]["policy"], "tunnel_interaction_eta")
        context.grpc.spawn_pedestrian.assert_not_called()
        self.fixture.tick(context, 1., 15., 4.)
        context.grpc.spawn_pedestrian.assert_not_called()
        self.fixture.tick(context, 2., 27., 4.)
        context.grpc.spawn_pedestrian.assert_called_once()

    def test_background_endpoint_deletes_before_next_crossing_and_preserves_its_gates(self):
        context, actor = self.fixture.background_before_crossing()
        self.fixture.tick(context, 0., 27., 4.)
        plan = context.tunnel_dynamic_plan
        start, end = plan["waypoints"]
        duration = plan["path_length_m"] / plan["speed_mps"]
        for ratio in (.25, .5, .75, 1.):
            self.fixture.tick(context, duration * ratio, 27. + 13. * ratio, 4., (
                start["x"] + ratio * (end["x"] - start["x"]),
                start["y"] + ratio * (end["y"] - start["y"]),
            ))
            self.assertEqual(context.grpc.spawn_pedestrian.call_count, 1)
        actor.destroy.assert_called_once()
        self.assertEqual(context.scenario_actors, [])
        self.assertEqual(context.tunnel_dynamic_event_index, 1)
        self.assertTrue(context.tunnel_dynamic_results[0]["completed"])
        self.assertTrue(context.tunnel_dynamic_results[0]["removal_observation"]["verified"])
        self.assertTrue(context.tunnel_dynamic_result["interaction_required"])
        self.fixture.tick(context, duration + .1, 74., 1.67)
        self.assertEqual(context.tunnel_dynamic_result["last_timing"]["policy"], "tunnel_interaction_eta")
        self.assertEqual(context.grpc.spawn_pedestrian.call_count, 1)
        self.fixture.tick(context, duration + .2, 70., 4.)
        self.assertEqual(context.grpc.spawn_pedestrian.call_count, 2)

    def test_tunnel_waits_past_general_floor_for_later_actual_interaction(self):
        context, _ = self.fixture.context()
        self.fixture.tick(context, 0., 38., 2.)
        context.grpc.spawn_pedestrian.assert_not_called()
        self.fixture.tick(context, 1., 40.25, 2.)
        context.grpc.spawn_pedestrian.assert_called_once()
        trigger = context.tunnel_dynamic_result["trigger"]
        self.assertGreaterEqual(trigger["remaining_m"], trigger["safe_stop_distance_m"])
        self.assertGreaterEqual(trigger["eta_delta_sec"], 0.)
        self.assertLessEqual(trigger["eta_delta_sec"], .3)

    def test_general_early_lead_and_floor_do_not_override_tunnel_defaults(self):
        context, _ = self.fixture.context()
        context.cfg["dynamic_obstacle"] = dict(
            interaction_lead_sec=1.5, minimum_trigger_distance_m=12.,
            reaction_time_sec=2., maximum_eta_delta_sec=4.)
        settings = context.tunnel_dynamic_handler._trigger_config(context)
        self.assertEqual(settings["interaction_lead_sec"], .3)
        self.assertEqual(settings["minimum_trigger_distance_m"], 0.)
        self.assertEqual(settings["reaction_time_sec"], 2.)
        self.assertEqual(settings["maximum_eta_delta_sec"], 4.)
        self.assertEqual(context.cfg["dynamic_obstacle"]["interaction_lead_sec"], 1.5)
        self.assertEqual(context.cfg["dynamic_obstacle"]["minimum_trigger_distance_m"], 12.)

    def test_endpoint_without_ego_slowdown_completes_after_removing_actor(self):
        context, actor = self.fixture.context()
        self.fixture.start(context)
        self.fixture.cross_to_endpoint(context)
        result = context.tunnel_dynamic_result
        self.assertTrue(result["interaction_required"])
        self.assertTrue(result["completed"])
        self.assertFalse(result["interaction_verified"])
        self.assertTrue(result["npc_motion_observed"])
        self.assertTrue(result["endpoint_reached"])
        self.assertTrue(result["removal_observation"]["verified"])
        self.assertEqual(result["interaction_assessment"], "not_observed")
        self.assertNotIn("failure_reason", result)
        self.assertEqual(context.tunnel_dynamic_phase, "completed")
        actor.destroy.assert_called_once()
        self.assertEqual(context.scenario_actors, [])

    def test_far_ego_slowdown_does_not_certify_interaction(self):
        context, _ = self.fixture.context()
        self.fixture.start(context)
        self.fixture.cross_to_endpoint(context, speed=0., ego_s=0.)
        self.assertFalse(context.tunnel_dynamic_result["interaction_verified"])
        self.assertTrue(context.tunnel_dynamic_result["completed"])

    def test_missing_speed_does_not_count_as_stopped_ego(self):
        context, _ = self.fixture.context()
        self.fixture.start(context)
        self.fixture.cross_to_endpoint(context, speed=None)
        self.assertFalse(context.tunnel_dynamic_result["interaction_verified"])
        self.assertTrue(context.tunnel_dynamic_result["completed"])

    def test_slowdown_before_observed_npc_motion_is_not_response(self):
        context, _ = self.fixture.context()
        self.fixture.start(context)
        self.fixture.tick(context, .01, 27., 0., (45., 2.))
        self.fixture.cross_to_endpoint(context)
        self.assertFalse(context.tunnel_dynamic_result["interaction_verified"])
        self.assertTrue(context.tunnel_dynamic_result["completed"])

    def test_abc_endpoint_teleport_cannot_certify_motion_or_ego_response(self):
        context, actor = self.fixture.context()
        self.fixture.start(context)
        with self.assertRaisesRegex(RuntimeError, "Implausible pedestrian jump"):
            self.fixture.tick(context, .1, 31., 0., (50., -2.))
        self.assertEqual(context.tunnel_dynamic_phase, "failed")
        self.assertFalse(context.tunnel_dynamic_result["npc_motion_observed"])
        self.assertFalse(context.tunnel_dynamic_result["interaction_verified"])
        self.assertFalse(context.tunnel_dynamic_result["completed"])
        actor.destroy.assert_called_once()

    def test_measured_slowdown_then_endpoint_completes_without_resume_requirement(self):
        context, actor = self.fixture.context()
        self.fixture.start(context)
        self.fixture.tick(context, 1., 31., 4., (46.5, 2.))
        self.fixture.tick(context, 2., 34., 3., (48., 2.))
        for elapsed, npc in ((10. / 3., (50., 2.)), (4., (50., 1.)),
                             (5., (50., -.5)), (6., (50., -2.))):
            self.fixture.tick(context, elapsed, 37., 0., npc)
        result = context.tunnel_dynamic_result
        self.assertTrue(result["interaction_verified"])
        self.assertTrue(result["completed"])
        self.assertEqual(context.tunnel_dynamic_phase, "completed")
        actor.destroy.assert_called_once()
        self.assertEqual(context.grpc.spawn_pedestrian.call_count, 1)
        self.assertEqual(context.scenario_actors, [])

    def test_animal_models_keep_the_same_interaction_and_endpoint_lifecycle(self):
        for model in ("Deer1", "Dog1", "Dog2", "Dog10", "Ped_Pet1"):
            with self.subTest(model=model):
                context, actor = self.fixture.context(model=model)
                self.fixture.start(context)
                self.fixture.cross_to_endpoint(context, speed=3.)
                self.assertEqual(context.grpc.spawn_pedestrian.call_args.kwargs["model_name"], model)
                self.assertTrue(context.tunnel_dynamic_result["interaction_verified"])
                self.assertTrue(context.tunnel_dynamic_result["completed"])
                actor.destroy.assert_called_once()
                self.assertEqual(context.scenario_actors, [])

    def test_explicit_si_speed_takes_precedence_over_legacy_raw_speed(self):
        context, _ = self.fixture.context()
        self.fixture.start(context)
        self.fixture.tick(context, 1., 31., 4., (46.5, 2.), raw_speed=14.4)
        self.fixture.tick(context, 2., 34., 3., (48., 2.), raw_speed=10.8)
        response = context.tunnel_dynamic_result["ego_response"]
        self.assertAlmostEqual(response["baseline_speed_mps"], 4.)
        self.assertGreaterEqual(response["maximum_drop_mps"], 1.)
        self.assertTrue(context.tunnel_dynamic_result["interaction_verified"])

    def test_parallel_offset_path_preserves_endpoint_lifecycle_without_forced_slowdown(self):
        context, actor = self.fixture.context("straight_same_direction_01")
        self.fixture.start(context, ego_s=27.)
        plan = context.tunnel_dynamic_plan
        start, end = plan["waypoints"]
        duration = plan["path_length_m"] / plan["speed_mps"]
        for ratio in (.25, .5, .75, 1.):
            self.fixture.tick(context, duration * ratio, 24., 4., (
                start["x"] + ratio * (end["x"] - start["x"]),
                start["y"] + ratio * (end["y"] - start["y"]),
            ))
        self.assertFalse(context.tunnel_dynamic_result["interaction_required"])
        self.assertTrue(context.tunnel_dynamic_result["completed"])
        self.assertFalse(context.tunnel_dynamic_result["interaction_verified"])
        self.assertEqual(context.tunnel_dynamic_result["interaction_assessment"], "non_conflicting_path")
        actor.destroy.assert_called_once()

    def test_centerline_opposite_path_completes_without_ego_response(self):
        context, actor = self.fixture.context("straight_opposite_02")
        self.fixture.start(context, ego_s=67.)
        plan = context.tunnel_dynamic_plan
        self.assertTrue(context.tunnel_dynamic_result["interaction_required"])
        start, end = plan["waypoints"]
        duration = plan["path_length_m"] / plan["speed_mps"]
        for ratio in (.25, .5, .75, 1.):
            self.fixture.tick(context, duration * ratio, 64., 4., (
                start["x"] + ratio * (end["x"] - start["x"]),
                start["y"] + ratio * (end["y"] - start["y"]),
            ))
        self.assertTrue(context.tunnel_dynamic_result["completed"])
        self.assertFalse(context.tunnel_dynamic_result["interaction_verified"])
        self.assertEqual(context.tunnel_dynamic_result["interaction_assessment"], "not_observed")
        actor.destroy.assert_called_once()


class TunnelScheduleBoundaryTest(_TunnelRuntimeTest):
    def context(self, candidate="straight_same_direction_02", length=15.0):
        context, actor = self.fixture.context(candidate)
        if length != 15.0:
            zone = context.route_definition["zones"]["tunnel"]
            zone["candidate_catalog"]["straight_length_m"] = length
            tunnel_dynamic.prepare_tunnel_dynamic(context)
            tunnel_dynamic.arm_tunnel_dynamic(context)
        context.expert_controller = SimpleNamespace(max_velocity_kmh=36.0)
        return context, actor

    @staticmethod
    def snapshot(context):
        return copy.deepcopy(dict(cfg=context.cfg,
            plans=context.tunnel_dynamic_plans, run_seed=context.run_seed,
            controller=vars(context.expert_controller)))

    @staticmethod
    def snapshot_timing(context, current_s, speed):
        plan = context.tunnel_dynamic_plan
        return tunnel_dynamic.interaction_timing(
            context.tunnel_dynamic_trigger_cfg, plan["interaction_s"],
            plan["start_to_interaction_m"], plan["speed_mps"],
            {"speed_mps": speed}, current_s)

    def test_acceleration_requires_three_samples_and_is_prediction_only(self):
        context, _ = self.context()
        handler = context.tunnel_dynamic_handler
        handler._record_ego_motion(context, 0.0, {"speed_mps": 3.0})
        handler._record_ego_motion(context, 0.25, {"speed_mps": 3.5})
        self.assertEqual(handler._motion_forecast(context), (0.0, 10.0))
        handler._record_ego_motion(context, 0.5, {"speed_mps": 4.0})
        self.assertEqual(handler._motion_forecast(context), (2.0, 10.0))
        handler._record_ego_motion(context, 0.5, {"speed_mps": 4.0})
        self.assertEqual(len(context.tunnel_dynamic_ego_history), 3)
        handler._record_ego_motion(context, 0.75, {})
        self.assertEqual(len(context.tunnel_dynamic_ego_history), 3)
        handler._record_ego_motion(context, 0.0, {"speed_mps": 4.0})
        self.assertEqual(handler._motion_forecast(context), (0.0, 10.0))
        self.assertEqual(context.expert_controller.max_velocity_kmh, 36.0)

    def test_acceleration_advances_deadline_without_violating_no_early_window(self):
        context, actor = self.context()
        handler = context.tunnel_dynamic_handler
        before = self.snapshot(context)
        inputs = self.snapshot_timing(context, 30.0, 4.0)
        steady = handler._side_schedule(context, 30.0, inputs)
        for elapsed, speed in ((0.0, 3.0), (0.25, 3.5), (0.5, 4.0)):
            handler._record_ego_motion(context, elapsed, {"speed_mps": speed})
        accelerated = handler._side_schedule(context, 30.0, inputs)
        self.assertLess(accelerated["latest_spawn_delay_sec"],
                        steady["latest_spawn_delay_sec"])
        self.assertEqual(accelerated["earliest_spawn_delay_sec"],
                         steady["earliest_spawn_delay_sec"])
        self.assertEqual(accelerated["approach_forecast_speed_ceiling_mps"], 4.0)
        # Acceleration makes the deadline urgent but cannot open the encounter
        # window until Ego's measured speed actually permits a nearby pass.
        expected = timing.encounter_start_window(
            context.tunnel_dynamic_motion_samples[0], 30.0, 4.0, 0.0, 4.0, 3.0,
            front_margin_m=3.85)
        self.assertEqual(accelerated["earliest_spawn_delay_sec"], expected[0])
        self.assertGreaterEqual(accelerated["target_spawn_delay_sec"],
                                accelerated["earliest_spawn_delay_sec"])
        self.fixture.tick(context, 0.5, 0.0, 4.0)
        context.grpc.spawn_pedestrian.assert_not_called()
        actor.control.assert_not_called()
        self.assertEqual(self.snapshot(context), before)

    def test_last_side_walk_reserves_full_path_not_half_path_to_anchor(self):
        context, _ = self.context("straight_same_direction_04")
        handler = context.tunnel_dynamic_handler
        context.tunnel_dynamic_goal_s = 186.0
        inputs = self.snapshot_timing(context, 140.0, 4.0)
        schedule = handler._side_schedule(context, 140.0, inputs)
        full = context.tunnel_dynamic_plan["path_length_m"] / 2.5
        anchor = context.tunnel_dynamic_plan["start_to_interaction_m"] / 2.5
        self.assertAlmostEqual(full, 6.0)
        self.assertAlmostEqual(anchor, 3.0)
        self.assertEqual(schedule["npc_full_duration_sec"], full)
        finish_budget = schedule["predicted_goal_eta_sec"] - full - schedule["cleanup_budget_sec"]
        self.assertLessEqual(schedule["latest_spawn_delay_sec"], finish_budget)

    def test_infeasible_sequential_budget_is_reported_without_changing_scene(self):
        context, actor = self.context("straight_same_direction_01", length=20.0)
        second, _ = self.context("straight_same_direction_02", length=20.0)
        third, _ = self.context("cross_left_to_right_03", length=20.0)
        for future in (second, third):
            context.tunnel_dynamic_plans.append(future.tunnel_dynamic_plan)
            context.tunnel_dynamic_results.append(future.tunnel_dynamic_result)
            context.tunnel_dynamic_motion_samples.extend(future.tunnel_dynamic_motion_samples)
        before = self.snapshot(context)
        inputs = self.snapshot_timing(context, 15.0, 8.0)
        with patch.object(tunnel_dynamic.random, 'Random',
                          side_effect=AssertionError('scheduler must not reroll')):
            schedule = context.tunnel_dynamic_handler._side_schedule(context, 15.0, inputs)
        self.assertFalse(schedule["schedule_feasible"])
        self.assertLess(schedule["latest_spawn_delay_sec"],
                        schedule["earliest_spawn_delay_sec"])
        self.assertGreaterEqual(schedule["target_spawn_delay_sec"],
                                schedule["earliest_spawn_delay_sec"])
        self.assertEqual(self.snapshot(context), before)
        context.grpc.spawn_pedestrian.assert_not_called()
        context.grpc.control_pedestrian.assert_not_called()
        actor.control.assert_not_called()


class TunnelDynamicOverlapTest(_OverlapFixture, _TunnelRuntimeTest):
    def test_prior_crossing_ahead_blocks_side_even_when_own_start_gate_is_open(self):
        context = self.crossing_followed_by_side()
        self.tick(context, 2.36, 37.8, 8.0)
        first, second = context.tunnel_dynamic_events[:2]
        self.assertTrue(first.tunnel_dynamic_result['interaction_required'])
        self.assertFalse(second.tunnel_dynamic_result['interaction_required'])
        self.assertLess(37.8, first.tunnel_dynamic_plan['interaction_s'])
        self.assertLessEqual(second.tunnel_dynamic_result['last_timing']['target_spawn_delay_sec'], 0.0)
        decision = second.tunnel_dynamic_result['overlap_decision']
        self.assertFalse(decision['allowed'])
        self.assertEqual(decision['reason'], 'prior_conflicting_event_still_ahead')
        self.assertEqual(decision['blocking_event_indices'], [0])
        self.assertEqual(len(context.grpc.spawned), 1)

        # Passing that encounter removes the brake-prediction hold, while the
        # usual deadline check may still prefer waiting for its nearby endpoint.
        self.tick(context, 2.7625, 41.0, 8.0)
        self.assertFalse(first.tunnel_dynamic_result['completed'])
        decision = second.tunnel_dynamic_result['overlap_decision']
        self.assertNotIn('blocking_event_indices', decision)
        self.assertIn('predicted_wait_for_deletion_sec', decision)

    def test_crossing_endpoint_releases_side_without_requiring_ego_response(self):
        context = self.crossing_followed_by_side()
        self.tick(context, 2.36, 37.8, 8.0)
        self.tick(context, 3.60, 37.8, 8.0)
        first, second = context.tunnel_dynamic_events[:2]
        self.assertTrue(first.tunnel_dynamic_result['completed'])
        self.assertFalse(first.tunnel_dynamic_result['interaction_verified'])
        self.assertTrue(first.tunnel_dynamic_result['removal_observation']['verified'])
        self.assertIsNone(first.tunnel_dynamic_actor)
        self.assertTrue(second.tunnel_dynamic_result['spawned'])
        self.assertEqual(second.tunnel_dynamic_spawn_elapsed, 3.60)
        self.assertEqual(set(context.grpc.live), {'owned-2'})

    def test_pending_waits_until_own_gate_and_then_spawns_before_prior_endpoint(self):
        context = self.context()
        self.start_first(context)
        self.tick(context, 1.0, 26.0, 8.0)
        self.assertEqual(len(context.grpc.spawned), 1)
        self.assertFalse(context.tunnel_dynamic_results[1]['spawned'])
        self.tick(context, 5.1125, 58.9, 8.0)
        first, second = context.tunnel_dynamic_events
        self.assertEqual(len(context.grpc.live), 2)
        self.assertFalse(first.tunnel_dynamic_result['completed'])
        decision = second.tunnel_dynamic_result['overlap_decision']
        self.assertTrue(decision['allowed'])
        self.assertGreaterEqual(decision['predicted_wait_for_deletion_sec'],
                                decision['latest_start_delay_sec'])
        self.assertEqual(first.tunnel_dynamic_result['actor_id'], 'owned-1')
        self.assertEqual(second.tunnel_dynamic_result['actor_id'], 'owned-2')
        self.assertEqual(first.tunnel_dynamic_spawn_elapsed, 0.0)
        self.assertEqual(second.tunnel_dynamic_spawn_elapsed, 5.1125)

    def test_same_tick_repeated_does_not_duplicate_a_spawn_or_actor_handle(self):
        context = self.context()
        self.overlap(context)
        for _ in range(3):
            self.tick(context, 5.1125, 58.9, 8.0)
        self.assertEqual(len(context.grpc.spawned), 2)
        self.assertEqual(len(context.scenario_actors), 2)
        self.assertEqual(len({request['label'] for request in context.grpc.spawn_requests}), 2)

    def test_endpoint_removal_checks_exact_id_and_leaves_other_actor_running(self):
        context = self.context()
        first, second = self.overlap(context)
        second_actor = second.tunnel_dynamic_actor
        self.tick(context, 7.81, 80.48, 8.0)
        self.assertTrue(first.tunnel_dynamic_result['completed'])
        self.assertIsNone(first.tunnel_dynamic_actor)
        self.assertTrue(first.tunnel_dynamic_result['removal_observation']['verified'])
        self.assertEqual(first.tunnel_dynamic_result['removal_observation']['absence_confirmations'], 2)
        self.assertIs(second.tunnel_dynamic_actor, second_actor)
        self.assertEqual(second_actor.destroy_calls, 0)
        self.assertEqual(set(context.grpc.live), {'owned-2'})
        self.assertEqual(context.scenario_actors, [second_actor])
        self.assertIn(('owned-2',), context.grpc.inventory_history)
        self.assertNotEqual(context.tunnel_dynamic_phase, 'completed')

    def test_two_events_complete_without_ego_response_and_keep_scene_unchanged(self):
        context = self.context()
        before = copy.deepcopy((context.cfg, context.tunnel_dynamic_plans,
                                context.run_seed, vars(context.expert_controller)))
        with patch.object(tunnel_dynamic.random, 'Random',
                          side_effect=AssertionError('no reroll during scheduling')):
            self.overlap(context)
            self.tick(context, 7.81, 80.48, 8.0)
            self.tick(context, 8.70, 87.6, 8.0)
        self.assertEqual(context.tunnel_dynamic_phase, 'completed')
        self.assertTrue(all(result['completed'] for result in context.tunnel_dynamic_results))
        self.assertFalse(any(result['interaction_verified'] for result in context.tunnel_dynamic_results))
        self.assertEqual(context.tunnel_dynamic_results[1]['interaction_assessment'], 'not_observed')
        self.assertEqual(context.scenario_actors, [])
        self.assertEqual(context.grpc.live, {})
        self.assertEqual([actor.destroy_calls for actor in context.grpc.spawned], [1, 1])
        self.assertEqual((context.cfg, context.tunnel_dynamic_plans,
                          context.run_seed, vars(context.expert_controller)), before)

    def test_ego_slowdown_rechecks_pending_gate_without_stopping_active_actor(self):
        context = self.context()
        self.start_first(context)
        self.tick(context, 5.1125, 58.9, 1.0)
        self.assertEqual(len(context.grpc.spawned), 1)
        self.assertEqual(context.grpc.spawned[0].destroy_calls, 0)
        self.assertFalse(context.tunnel_dynamic_results[1]['spawned'])
        self.tick(context, 6.1, 65.0, 6.0)
        self.assertEqual(len(context.grpc.spawned), 2)
        self.assertAlmostEqual(context.tunnel_dynamic_results[1]['trigger']['ego_speed_mps'], 6.0)

    def test_read_failures_and_last_position_are_owned_by_each_event(self):
        context = self.context()
        first, second = self.overlap(context)
        context.grpc.unavailable_ids.add('owned-1')
        self.tick(context, 5.17, 59.0, 8.0)
        self.assertEqual(first.tunnel_dynamic_state_misses, 1)
        self.assertEqual(second.tunnel_dynamic_state_misses, 0)
        self.assertEqual(first.tunnel_dynamic_result['state_read_failures'], 1)
        self.assertEqual(second.tunnel_dynamic_result['state_read_failures'], 0)
        self.assertNotEqual(first.tunnel_dynamic_last_position, second.tunnel_dynamic_last_position)
        context.grpc.unavailable_ids.clear()
        self.tick(context, 5.22, 59.4, 8.0)
        self.assertEqual(first.tunnel_dynamic_state_misses, 0)
        self.assertEqual(second.tunnel_dynamic_state_misses, 0)

    def test_one_event_failure_cleans_both_owned_actors(self):
        context = self.context()
        first, second = self.overlap(context)
        second.tunnel_dynamic_actor.x = math.nan
        with self.assertRaisesRegex(RuntimeError, 'position is not finite'):
            self.tick(context, 5.22, 59.4, 8.0)
        self.assertEqual(context.tunnel_dynamic_phase, 'failed')
        self.assertEqual(context.grpc.live, {})
        self.assertEqual(context.scenario_actors, [])
        self.assertEqual([actor.destroy_calls for actor in context.grpc.spawned], [1, 1])
        self.assertTrue(first.tunnel_dynamic_result['removal_observation']['verified'])
        self.assertTrue(second.tunnel_dynamic_result['removal_observation']['verified'])

    def test_pending_spawn_rpc_failure_cleans_previously_active_actor(self):
        context = self.context()
        self.start_first(context)
        with patch.object(context.grpc, 'spawn_pedestrian',
                          side_effect=RuntimeError('test RPC unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'spawn RPC failed'):
                self.tick(context, 5.1125, 58.9, 8.0)
        first, second = context.tunnel_dynamic_events
        self.assertEqual(context.tunnel_dynamic_phase, 'failed')
        self.assertFalse(second.tunnel_dynamic_result['spawned'])
        self.assertFalse(second.tunnel_dynamic_result['completed'])
        self.assertIsNone(first.tunnel_dynamic_actor)
        self.assertEqual(context.grpc.spawned[0].destroy_calls, 1)
        self.assertTrue(first.tunnel_dynamic_result['removal_observation']['verified'])
        self.assertEqual(context.scenario_actors, [])
        self.assertEqual(context.grpc.live, {})

    def test_server_reusing_active_actor_id_fails_and_cleans_all_handles(self):
        context = self.context()
        self.start_first(context)
        active = context.grpc.spawned[0]
        # Returning the existing handle represents a server reply identifying
        # the already active actor. It must never become a second valid event.
        with patch.object(context.grpc, 'spawn_pedestrian', return_value=active):
            with self.assertRaisesRegex(RuntimeError, 'duplicate active actor ID'):
                self.tick(context, 5.1125, 58.9, 8.0)
        self.assertEqual(context.tunnel_dynamic_phase, 'failed')
        self.assertFalse(any(result['completed'] for result in context.tunnel_dynamic_results))
        self.assertEqual(context.scenario_actors, [])
        self.assertEqual(context.grpc.live, {})
        self.assertTrue(all(event.tunnel_dynamic_actor is None
                            for event in context.tunnel_dynamic_events))

    def test_valid_pending_gate_waits_when_active_can_finish_before_deadline(self):
        context = self.context(second=None, lead=2.0)
        self.start_first(context)
        self.tick(context, 7.7, 23.4, 4.0)
        first, second = context.tunnel_dynamic_events
        self.assertFalse(first.tunnel_dynamic_result['completed'])
        self.assertFalse(second.tunnel_dynamic_result['spawned'])
        decision = second.tunnel_dynamic_result['overlap_decision']
        self.assertFalse(decision['allowed'])
        self.assertLess(decision['predicted_wait_for_deletion_sec'], decision['latest_start_delay_sec'])
        self.tick(context, 7.81, 23.4, 4.0)
        self.assertTrue(first.tunnel_dynamic_result['completed'])
        self.assertTrue(second.tunnel_dynamic_result['spawned'])
        self.assertEqual(len(context.grpc.live), 1)


class TunnelConservativeTimingTest(_TunnelRuntimeTest):
    def context(self, candidate, overrides=None):
        context, actor = self.fixture.context(candidate)
        context.route_definition['zones']['tunnel']['candidate_catalog']['straight_length_m'] = 20.0
        if overrides is not None:
            context.cfg['tunnel_dynamic']['interaction_trigger'] = overrides
        tunnel_dynamic.prepare_tunnel_dynamic(context)
        context.expert_controller = SimpleNamespace(max_velocity_kmh=60.0)
        tunnel_dynamic.arm_tunnel_dynamic(context)
        return context, actor

    @staticmethod
    def remember(context, samples):
        for elapsed, speed in samples:
            context.tunnel_dynamic_handler._record_ego_motion(
                context, elapsed, {'speed_mps': speed})

    @staticmethod
    def snapshot(context):
        return copy.deepcopy((context.cfg, context.tunnel_dynamic_plans,
                              context.run_seed, vars(context.expert_controller)))

    def test_side01_does_not_use_startup_acceleration_to_spawn_before_measured_approach(self):
        # Seed 2709's observed early side01 trigger, translated onto the
        # fixture route. Actual speed was only 2.19 m/s despite recent +a.
        context, actor = self.context('straight_same_direction_01')
        speed, acceleration = 2.1866026626581787, 2.243025563002079
        current_s = context.tunnel_dynamic_plan['interaction_s'] - 23.93821443857252
        self.remember(context, ((0.0, speed - acceleration * .5),
                                (.25, speed - acceleration * .25), (.5, speed)))
        before = self.snapshot(context)
        expected = timing.encounter_start_window(
            context.tunnel_dynamic_motion_samples[0], current_s, speed, 0.0, speed, 3.0,
            front_margin_m=3.85)
        self.assertGreater(expected[0], 0.0)
        with patch.object(tunnel_dynamic.random, 'Random',
                          side_effect=AssertionError('no seed reroll')):
            self.fixture.tick(context, .5, current_s, speed)
        actual = context.tunnel_dynamic_result['last_timing']
        self.assertAlmostEqual(actual['earliest_spawn_delay_sec'], expected[0])
        self.assertGreater(actual['target_spawn_delay_sec'], 0.0)
        context.grpc.spawn_pedestrian.assert_not_called()
        actor.control.assert_not_called()
        self.assertEqual(self.snapshot(context), before)

    def test_reverse_side03_waits_for_real_closing_speed_despite_recent_acceleration(self):
        # Seed 2710's side03 is offset from Ego and walks toward it. The
        # opposite longitudinal direction must remain in the encounter model.
        context, _ = self.context('straight_opposite_03')
        self.assertFalse(context.tunnel_dynamic_result['interaction_required'])
        speed, acceleration = 2.464788708072624, .8419337252597895
        current_s = context.tunnel_dynamic_plan['interaction_s'] - 37.80232696538158
        self.remember(context, ((0.0, speed - acceleration * .5),
                                (.25, speed - acceleration * .25), (.5, speed)))
        expected = timing.encounter_start_window(
            context.tunnel_dynamic_motion_samples[0], current_s, speed, 0.0, speed, 3.0,
            front_margin_m=3.85)
        self.assertGreater(expected[0], 0.0)
        self.fixture.tick(context, .5, current_s, speed)
        actual = context.tunnel_dynamic_result['last_timing']
        self.assertAlmostEqual(actual['earliest_spawn_delay_sec'], expected[0])
        self.assertGreater(actual['target_spawn_delay_sec'], 0.0)
        context.grpc.spawn_pedestrian.assert_not_called()

    def test_oncoming_keeps_original_lead_and_floor_and_spawns_in_original_window(self):
        for candidate in ('straight_opposite_02', 'straight_opposite_04'):
            with self.subTest(candidate=candidate):
                context, _ = self.context(candidate)
                before = copy.deepcopy(context.tunnel_dynamic_trigger_cfg)
                settings = context.tunnel_dynamic_handler._event_trigger_config(context)
                self.assertEqual(settings['interaction_lead_sec'], 1.5)
                self.assertEqual(settings['minimum_trigger_distance_m'], 12.0)
                self.assertEqual(context.tunnel_dynamic_trigger_cfg, before)
                self.fixture.tick(context, 0.0,
                    context.tunnel_dynamic_plan['interaction_s'] - 28.7, 5.27)
                context.grpc.spawn_pedestrian.assert_called_once()
                trigger = context.tunnel_dynamic_result['trigger']
                self.assertAlmostEqual(trigger['trigger_distance_m'],
                                       5.27 * (trigger['npc_eta_sec'] + 1.5))
                self.assertGreaterEqual(trigger['remaining_m'], trigger['safe_stop_distance_m'])

    def test_crossing_and_offset_reverse_keep_tunnel_defaults(self):
        for candidate in (None, 'cross_left_to_right_02', 'diagonal_cross_03',
                          'straight_opposite_01', 'straight_same_direction_02'):
            with self.subTest(candidate=candidate):
                context, _ = self.context(candidate)
                settings = context.tunnel_dynamic_handler._event_trigger_config(context)
                self.assertEqual(settings['interaction_lead_sec'], .3)
                self.assertEqual(settings['minimum_trigger_distance_m'], 0.0)

    def test_explicit_oncoming_settings_are_honored_without_config_mutation(self):
        context, _ = self.context('straight_opposite_02',
            {'interaction_lead_sec': .75, 'minimum_trigger_distance_m': 4.0})
        before = copy.deepcopy(context.cfg)
        settings = context.tunnel_dynamic_handler._event_trigger_config(context)
        self.assertEqual(settings['interaction_lead_sec'], .75)
        self.assertEqual(settings['minimum_trigger_distance_m'], 4.0)
        self.assertEqual(context.cfg, before)

    def test_live_deceleration_holds_pending_side_then_resume_releases_overlap(self):
        context = _OverlapFixture.context(
            self, second='straight_same_direction_02')
        tick = _OverlapFixture.tick
        tick(context, 0.0, 18.0, 8.0)
        self.remember(context, ((4.9, 8.4), (5.0, 8.2), (5.1125, 8.0)))
        tick(context, 5.1125, 56.2, 8.0)
        first, second = context.tunnel_dynamic_events
        self.assertEqual(len(context.grpc.spawned), 1)
        self.assertFalse(first.tunnel_dynamic_result['completed'])
        self.assertLessEqual(second.tunnel_dynamic_result['last_timing']['target_spawn_delay_sec'], 0.0)
        self.assertTrue(second.tunnel_dynamic_result['last_timing']['side_motion_hold'])
        self.assertFalse(second.tunnel_dynamic_result['spawned'])
        self.assertGreater(context.grpc.spawned[0].speed, 0.0)

        # Current telemetry now shows acceleration again; no Ego command or
        # response-success gate is used to release the pending actor.
        self.remember(context, ((5.35, 8.0), (5.475, 8.05), (5.6, 8.1)))
        tick(context, 5.6, 58.7, 8.1)
        self.assertFalse(second.tunnel_dynamic_result['last_timing']['side_motion_hold'])
        self.assertEqual(len(context.grpc.spawned), 2)
        self.assertEqual(len(context.grpc.live), 2)
        self.assertFalse(first.tunnel_dynamic_result['completed'])
        self.assertTrue(second.tunnel_dynamic_result['overlap_decision']['allowed'])
        self.assertAlmostEqual(second.tunnel_dynamic_result['trigger']['ego_speed_mps'], 8.1)

    def test_subthreshold_speed_noise_does_not_block_necessary_overlap(self):
        context = _OverlapFixture.context(
            self, second='straight_same_direction_02')
        tick = _OverlapFixture.tick
        tick(context, 0.0, 18.0, 8.0)
        self.remember(context, ((4.9, 8.08), (5.0, 8.04), (5.1125, 8.0)))
        before = self.snapshot(context)
        with patch.object(tunnel_dynamic.random, 'Random',
                          side_effect=AssertionError('no seed reroll')):
            tick(context, 5.1125, 58.9, 8.0)
        second = context.tunnel_dynamic_events[1]
        self.assertFalse(second.tunnel_dynamic_result['last_timing']['side_motion_hold'])
        self.assertTrue(second.tunnel_dynamic_result['spawned'])
        self.assertEqual(len(context.grpc.live), 2)
        self.assertEqual(self.snapshot(context), before)

    def test_previous_side_endpoint_wait_must_not_consume_restored_oncoming_lead(self):
        context = _OverlapFixture.context(
            self, second='straight_opposite_02')
        oncoming, _ = self.context('straight_opposite_02')
        context.tunnel_dynamic_plans[1] = oncoming.tunnel_dynamic_plan
        tunnel_dynamic.arm_tunnel_dynamic(context)
        tick = _OverlapFixture.tick
        tick(context, 0.0, 18.0, 8.0)

        # The side NPC has 2.75 m (1.1 s) left. Waiting for its endpoint
        # would still satisfy the old zero-ETA deadline, but would spend most
        # of the oncoming encounter's restored 1.5 s clearance lead.
        speed, remaining = 5.27, 28.9
        current_s = context.tunnel_dynamic_plans[1]['interaction_s'] - remaining
        tick(context, 6.9, current_s, speed)
        first, second = context.tunnel_dynamic_events
        self.assertFalse(first.tunnel_dynamic_result['completed'])
        self.assertTrue(second.tunnel_dynamic_result['spawned'])
        self.assertEqual(len(context.grpc.live), 2)
        decision = second.tunnel_dynamic_result['overlap_decision']
        trigger = second.tunnel_dynamic_result['trigger']
        old_eta_deadline = remaining / speed - trigger['npc_eta_sec']
        self.assertLess(decision['predicted_wait_for_deletion_sec'], old_eta_deadline)
        self.assertGreater(decision['predicted_wait_for_deletion_sec'], 1.0)
        self.assertLessEqual(decision['latest_start_delay_sec'], 0.0)
        self.assertTrue(decision['allowed'])
        self.assertAlmostEqual(trigger['trigger_distance_m'],
                               speed * (trigger['npc_eta_sec'] + 1.5))
        self.assertAlmostEqual(second.tunnel_dynamic_spawn_elapsed, 6.9)

    def test_final_side_can_start_when_ego_front_reaches_npc_before_reference_point(self):
        context, _ = self.context('straight_same_direction_04')
        speed, current_s = 4.5, 130.95
        samples = context.tunnel_dynamic_motion_samples[0]
        endpoint = samples[-1]
        predicted_ego_s = current_s + speed * endpoint['time_sec']
        gap = endpoint['route_s'] - predicted_ego_s
        self.assertGreater(gap, 3.0)
        self.assertLess(gap, 3.85)
        point_window = timing.encounter_start_window(samples, current_s, speed,
                                                     0.0, speed, 3.0)
        front_window = timing.encounter_start_window(samples, current_s, speed,
            0.0, speed, 3.0, front_margin_m=3.85)
        self.assertAlmostEqual(point_window[0], .1)
        self.assertLessEqual(front_window[0], 0.0)

        # Recent acceleration makes the final full-path time budget urgent.
        # It must not be extrapolated into this physical approach bound.
        self.remember(context, ((0.0, 4.0), (.25, 4.25), (.5, speed)))
        self.fixture.tick(context, .5, current_s, speed)
        observed = context.tunnel_dynamic_result['last_timing']
        self.assertAlmostEqual(observed['earliest_spawn_delay_sec'], front_window[0])
        self.assertAlmostEqual(observed['encounter_front_margin_m'], 3.85)
        self.assertEqual(observed['approach_forecast_speed_ceiling_mps'], speed)
        self.assertEqual(observed['approach_acceleration_horizon_sec'], 0.0)
        self.assertLessEqual(observed['target_spawn_delay_sec'], 0.0)
        context.grpc.spawn_pedestrian.assert_called_once()


if __name__ == "__main__":
    unittest.main()
