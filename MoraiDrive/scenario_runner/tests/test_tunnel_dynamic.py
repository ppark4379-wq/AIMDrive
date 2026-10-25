"""Tunnel dynamic obstacle setup and lifecycle tests."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ..scenarios.full_loop_parts import tunnel_dynamic
from ..scenarios.full_loop_parts.tunnel_dynamic import (
    arm_tunnel_dynamic,
    prepare_tunnel_dynamic,
    update_tunnel_dynamic,
)


class TunnelDynamicTest(unittest.TestCase):
    def test_enabled_crossing_requires_handler(self):
        context = SimpleNamespace(cfg={"tunnel_dynamic": {"enabled": True}})
        with patch.object(tunnel_dynamic, "TUNNEL_DYNAMIC_HANDLER", None):
            with self.assertRaisesRegex(RuntimeError, "validated placement and lifecycle handler"):
                prepare_tunnel_dynamic(context)
        self.assertIsNone(context.tunnel_dynamic_handler)

    def test_validated_handler_receives_lifecycle_calls(self):
        context = SimpleNamespace(cfg={"tunnel_dynamic": {"enabled": True}})
        handler = Mock()
        self.assertTrue(prepare_tunnel_dynamic(context, handler))
        handler.validate.assert_called_once_with(context)
        arm_tunnel_dynamic(context)
        update_tunnel_dynamic(context, 1.0, {"speed": 5.0}, 35.0)
        handler.arm.assert_called_once_with(context)
        handler.update.assert_called_once_with(context, 1.0, {"speed": 5.0}, 35.0)

    def test_disabled_crossing_does_not_touch_handler(self):
        context = SimpleNamespace(cfg={"tunnel_dynamic": {"enabled": False}})
        handler = Mock()
        self.assertFalse(prepare_tunnel_dynamic(context, handler))
        self.assertIsNone(context.tunnel_dynamic_handler)
        handler.validate.assert_not_called()

    def test_clean_road_cannot_enable_crossing(self):
        context = SimpleNamespace(
            cfg={"tunnel_dynamic": {"enabled": True}}, clean_road_mode=True
        )
        with self.assertRaisesRegex(ValueError, "clean_road_mode"):
            prepare_tunnel_dynamic(context)

    def test_other_segment_cannot_arm_tunnel_crossing(self):
        context = SimpleNamespace(
            cfg={"tunnel_dynamic": {"enabled": True}},
            mission_mode=SimpleNamespace(zones=SimpleNamespace(tunnel=False)),
        )
        self.assertFalse(prepare_tunnel_dynamic(context))
        self.assertIsNone(context.tunnel_dynamic_handler)

    def make_route_context(self):
        waypoints = [
            {"x": 45.0, "y": 2.0, "z": 0.0},
            {"x": 50.0, "y": 2.0, "z": 0.0},
            {"x": 50.0, "y": -2.0, "z": 0.0},
        ]
        actor = Mock()
        actor.get_id.return_value = "ped-1"
        actor.destroy.return_value = True
        actor.control.return_value = True
        grpc = Mock()
        grpc.get_available_pedestrian_models.return_value = ["Man1"]
        grpc.get_all_pedestrian_states.return_value = []
        grpc.spawn_pedestrian.return_value = actor
        grpc.control_pedestrian.return_value = True
        grpc.set_pedestrian_waypoint.return_value = True
        grpc.make_transform.side_effect = lambda x, y, z, yaw: SimpleNamespace(
            rotation=SimpleNamespace(x=0., y=0., z=yaw))
        context = SimpleNamespace(
            cfg={"tunnel_dynamic": {
                "enabled": True, "model": "Man1", "path_id": "walk_then_cross",
                "candidate_policy": "trial",
                "speed_mps": 1.5,
                "arrival_tolerance_m": 0.5, "max_run_sec": 15.0,
            }},
            route_definition={"zones": {"tunnel": {"dynamic_paths": {
                "walk_then_cross": {"route_path_index": 50, "waypoints": waypoints}
            }}}},
            route_points=[(float(i), 0.0, 0.0) for i in range(101)],
            path_s_at_index=lambda index: float(index),
            tunnel_entry_s=10.0, tunnel_exit_s=90.0,
            grpc=grpc, scenario_actors=[], run_seed=7, run_instance_id="run-7",
        )
        return context, actor

    def test_one_pedestrian_moves_a_to_b_to_c_then_is_removed(self):
        context, actor = self.make_route_context()
        actor.get_actor_state.side_effect = [
            SimpleNamespace(transform=SimpleNamespace(location=SimpleNamespace(x=50.0, y=2.0))),
            SimpleNamespace(transform=SimpleNamespace(location=SimpleNamespace(x=50.0, y=-2.0))),
        ]
        self.assertTrue(prepare_tunnel_dynamic(context))
        arm_tunnel_dynamic(context)
        self.assertEqual(context.tunnel_dynamic_phase, "waiting")
        update_tunnel_dynamic(context, 0.0, {"speed_mps": 4.0}, 15.0)
        context.grpc.spawn_pedestrian.assert_not_called()
        update_tunnel_dynamic(context, 1.0, {"speed_mps": 4.0}, 31.0)
        self.assertEqual(context.tunnel_dynamic_phase, "spawned_unobserved")
        self.assertFalse(context.tunnel_dynamic_result["spawn_state_verified"])
        self.assertEqual(context.tunnel_dynamic_target_index, 1)
        update_tunnel_dynamic(context, 5.0, {"speed_mps": 3.0}, 32.0)
        self.assertEqual(context.tunnel_dynamic_target_index, 2)
        update_tunnel_dynamic(context, 8.0, {"speed_mps": 0.0}, 33.0)
        self.assertEqual(context.tunnel_dynamic_phase, "completed")
        self.assertTrue(context.tunnel_dynamic_result["completed"])
        self.assertEqual(context.grpc.spawn_pedestrian.call_count, 1)
        context.grpc.set_pedestrian_waypoint.assert_not_called()
        self.assertEqual([(call.args[0].z, call.args[1])
                          for call in actor.control.call_args_list], [(0., 5.4), (-90., 5.4)])
        context.grpc.control_pedestrian.assert_called_once_with(
            actor, 0.0, 0.0, 0.0)
        actor.destroy.assert_called_once()
        self.assertEqual(context.scenario_actors, [])

    def test_heading_command_rejection_cleans_up_at_spawn_or_turn(self):
        for reject_initial in (True, False):
            with self.subTest(reject_initial=reject_initial):
                context, actor = self.make_route_context()
                prepare_tunnel_dynamic(context)
                arm_tunnel_dynamic(context)
                if reject_initial:
                    actor.control.return_value = False
                    with self.assertRaisesRegex(RuntimeError, "heading command failed"):
                        update_tunnel_dynamic(context, 1., {"speed_mps": 4.}, 31.)
                else:
                    update_tunnel_dynamic(context, 1., {"speed_mps": 4.}, 31.)
                    actor.get_actor_state.return_value = self.observed_state(x=50., y=2.)
                    actor.control.return_value = False
                    with self.assertRaisesRegex(RuntimeError, "heading command failed"):
                        update_tunnel_dynamic(context, 5., {"speed_mps": 3.}, 32.)
                self.assertEqual(context.tunnel_dynamic_phase, "failed")
                self.assertFalse(context.tunnel_dynamic_result["completed"])
                actor.destroy.assert_called_once()
                self.assertEqual(context.scenario_actors, [])
                context.grpc.set_pedestrian_waypoint.assert_not_called()

    def test_wrong_side_path_fails_before_spawn(self):
        context, _ = self.make_route_context()
        context.route_definition["zones"]["tunnel"]["dynamic_paths"][
            "walk_then_cross"]["waypoints"][2].update(x=52.0, y=2.0)
        with self.assertRaisesRegex(RuntimeError, "must cross the official route"):
            prepare_tunnel_dynamic(context)
        context.grpc.spawn_pedestrian.assert_not_called()

    def test_timeout_destroys_owned_pedestrian(self):
        context, actor = self.make_route_context()
        prepare_tunnel_dynamic(context)
        arm_tunnel_dynamic(context)
        update_tunnel_dynamic(context, 1.0, {"speed_mps": 4.0}, 31.0)
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            update_tunnel_dynamic(context, 17.0, {}, 21.0)
        self.assertEqual(context.tunnel_dynamic_phase, "failed")
        actor.destroy.assert_called_once()
        self.assertEqual(context.scenario_actors, [])

    def test_spawn_id_failure_attempts_cleanup_without_certifying_unknown_removal(self):
        for id_error in (RuntimeError("ID RPC unavailable"), None):
            with self.subTest(id_error=id_error):
                context, actor = self.make_route_context()
                actor.get_id.side_effect = id_error
                actor.get_id.return_value = ""
                prepare_tunnel_dynamic(context)
                arm_tunnel_dynamic(context)
                with self.assertRaisesRegex(RuntimeError, "ID unavailable|no actor ID"):
                    update_tunnel_dynamic(context, 1.0, {"speed_mps": 4.0}, 31.0)
                actor.destroy.assert_called_once()
                context.grpc.get_all_pedestrian_states.assert_not_called()
                self.assertEqual(context.tunnel_dynamic_phase, "failed")
                self.assertFalse(context.tunnel_dynamic_result["completed"])
                removal = context.tunnel_dynamic_result["removal_observation"]
                self.assertFalse(removal["verified"])
                self.assertIsNone(removal["actor_id"])
                self.assertIs(context.tunnel_dynamic_actor, actor)
                self.assertEqual(context.scenario_actors, [actor])
                with self.assertRaisesRegex(RuntimeError, "already failed"):
                    update_tunnel_dynamic(context, 2.0, {}, 28.0)
                self.assertEqual(context.grpc.spawn_pedestrian.call_count, 1)

    def test_explicit_straight_candidate_uses_native_yaw_and_velocity(self):
        context, actor = self.make_route_context()
        context.cfg["tunnel_dynamic"].update(
            trial_candidate_id="straight_same_direction_01",
        )
        context.route_definition["zones"]["tunnel"]["candidate_catalog"] = {
            "division_count": 4,
            "endpoint_margin_m": 10.0,
            "side_offset_m": 2.5,
            "opposite_lane_1_offset_m": 2.5,
            "opposite_lane_2_offset_m": 0.0,
            "straight_length_m": 15.0,
            "cross_delta_s_m": 3.0,
            "diagonal_delta_s_m": 8.0,
            "approved_ids": [],
            "disabled_ids": [],
        }
        actor.get_actor_state.side_effect = [
            SimpleNamespace(transform=SimpleNamespace(
                location=SimpleNamespace(x=23.0, y=2.5))),
            SimpleNamespace(transform=SimpleNamespace(
                location=SimpleNamespace(x=35.0, y=2.5))),
        ]
        prepare_tunnel_dynamic(context)
        self.assertEqual(context.tunnel_dynamic_plan["path_id"],
                         "straight_same_direction_01")
        self.assertEqual(context.tunnel_dynamic_plan["interaction_s"], 27.5)
        arm_tunnel_dynamic(context)
        update_tunnel_dynamic(context, 1.0, {"speed_mps": 2.0}, 17.0)
        context.grpc.spawn_pedestrian.assert_called_once()
        spawn = context.grpc.spawn_pedestrian.call_args.kwargs
        self.assertEqual(spawn["velocity"], 5.4)
        self.assertEqual(spawn["active_dist"], 0.0)
        self.assertEqual(spawn["move_dist"], 17.0)
        self.assertTrue(spawn["start_action"])
        context.grpc.make_transform.assert_called_once_with(20.0, 2.5, 0.0, 0.0)
        context.grpc.spawn_pedestrian_with_waypoints.assert_not_called()
        context.grpc.set_pedestrian_waypoint.assert_not_called()
        update_tunnel_dynamic(context, 2.0, {}, 16.0)
        self.assertEqual(context.tunnel_dynamic_phase, "moving")
        update_tunnel_dynamic(context, 5.0, {}, 18.0)
        self.assertEqual(context.tunnel_dynamic_phase, "completed")
        actor.destroy.assert_called_once()

    def test_implausible_candidate_jump_is_held_for_inspection(self):
        context, actor = self.make_route_context()
        context.cfg["tunnel_dynamic"].update(
            trial_candidate_id="straight_same_direction_01",
        )
        context.route_definition["zones"]["tunnel"]["candidate_catalog"] = {
            "division_count": 4,
            "endpoint_margin_m": 10.0,
            "side_offset_m": 2.5,
            "opposite_lane_1_offset_m": 2.5,
            "opposite_lane_2_offset_m": 0.0,
            "straight_length_m": 15.0,
            "cross_delta_s_m": 3.0,
            "diagonal_delta_s_m": 8.0,
        }
        actor.get_actor_state.return_value = SimpleNamespace(
            transform=SimpleNamespace(location=SimpleNamespace(x=35.0, y=2.5)))
        prepare_tunnel_dynamic(context)
        arm_tunnel_dynamic(context)
        update_tunnel_dynamic(context, 1.0, {"speed_mps": 2.0}, 17.0)
        update_tunnel_dynamic(context, 1.1, {}, 15.0)
        self.assertEqual(context.tunnel_dynamic_phase, "diagnostic_hold")
        self.assertEqual(context.tunnel_dynamic_result["failure_reason"],
                         "implausible_pedestrian_jump")
        actor.destroy.assert_not_called()
        update_tunnel_dynamic(context, 25.0, {}, 15.0)
        actor.destroy.assert_not_called()

    def test_disabled_candidate_cannot_spawn(self):
        context, _ = self.make_route_context()
        context.cfg["tunnel_dynamic"]["trial_candidate_id"] = "straight_same_direction_01"
        context.route_definition["zones"]["tunnel"]["candidate_catalog"] = {
            "division_count": 4,
            "endpoint_margin_m": 10.0,
            "side_offset_m": 2.5,
            "opposite_lane_1_offset_m": 2.5,
            "opposite_lane_2_offset_m": 0.0,
            "straight_length_m": 15.0,
            "cross_delta_s_m": 3.0,
            "diagonal_delta_s_m": 8.0,
            "disabled_ids": ["straight_same_direction_01"],
        }
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            prepare_tunnel_dynamic(context)
        context.grpc.spawn_pedestrian.assert_not_called()

    def start_observation_context(self):
        context, actor = self.make_route_context()
        prepare_tunnel_dynamic(context)
        arm_tunnel_dynamic(context)
        update_tunnel_dynamic(context, 1.0, {"speed_mps": 4.0}, 31.0)
        return context, actor

    @staticmethod
    def observed_state(actor_id="ped-1", object_type=2, x=45.1, y=2.0):
        return SimpleNamespace(
            actor_info=SimpleNamespace(
                id=SimpleNamespace(value=actor_id), object_type=object_type),
            transform=SimpleNamespace(location=SimpleNamespace(x=x, y=y)),
        )

    def test_sdk_none_uses_one_fresh_exact_pedestrian_inventory_state(self):
        context, actor = self.start_observation_context()
        actor.get_actor_state.return_value = None
        context.grpc.get_all_pedestrian_states.return_value = [
            self.observed_state("unrelated"), self.observed_state(),
        ]
        update_tunnel_dynamic(context, 1.2, {}, 27.0)
        context.grpc.get_all_pedestrian_states.assert_called_once_with(strict=True)
        self.assertEqual(context.tunnel_dynamic_phase, "moving")
        self.assertTrue(context.tunnel_dynamic_result["spawn_state_verified"])
        self.assertEqual(context.tunnel_dynamic_result["last_state_source"], "GetAllActorsState")
        self.assertEqual(context.tunnel_dynamic_result["last_position"], {"x": 45.1, "y": 2.0})
        self.assertFalse(context.tunnel_dynamic_result["completed"])

    def test_individual_rpc_exception_can_use_fresh_inventory(self):
        context, actor = self.start_observation_context()
        actor.get_actor_state.side_effect = RuntimeError("UNKNOWN handler error")
        context.grpc.get_all_pedestrian_states.return_value = [self.observed_state()]
        update_tunnel_dynamic(context, 1.2, {}, 27.0)
        self.assertEqual(context.tunnel_dynamic_result["state_observations"], {"GetAllActorsState": 1})

    def test_inventory_wrong_type_wrong_id_duplicate_and_empty_are_not_states(self):
        context, actor = self.start_observation_context()
        actor.get_actor_state.return_value = None
        for states in ([self.observed_state(object_type=1)],
                       [self.observed_state("another")],
                       [self.observed_state(), self.observed_state()], []):
            with self.subTest(states=states):
                context.grpc.get_all_pedestrian_states.return_value = states
                with self.assertRaisesRegex(RuntimeError, "expected one exact pedestrian"):
                    context.tunnel_dynamic_handler._read_state(context, actor)
                self.assertEqual(context.tunnel_dynamic_phase, "spawned_unobserved")
                self.assertFalse(context.tunnel_dynamic_result["completed"])

    def test_inventory_rpc_failure_is_not_an_empty_scene(self):
        context, actor = self.start_observation_context()
        actor.get_actor_state.return_value = None
        context.grpc.get_all_pedestrian_states.side_effect = RuntimeError("inventory unavailable")
        with self.assertRaisesRegex(RuntimeError, "inventory unavailable"):
            context.tunnel_dynamic_handler._read_state(context, actor)
        self.assertFalse(context.tunnel_dynamic_result["spawn_state_verified"])

    def test_permanently_absent_spawn_fails_at_wall_deadline_without_completion(self):
        context, actor = self.start_observation_context()
        actor.get_actor_state.return_value = None
        ack = context.tunnel_dynamic_spawn_ack_monotonic
        with patch.object(tunnel_dynamic.time, "sleep"), patch.object(
                tunnel_dynamic.time, "monotonic", return_value=ack + .2) as now:
            for elapsed in (1.05, 1.1, 1.15, 1.2):
                update_tunnel_dynamic(context, elapsed, {}, 27.0)
            self.assertEqual(context.tunnel_dynamic_phase, "spawned_unobserved")
            now.return_value = ack + 1.01
            with self.assertRaisesRegex(RuntimeError, "initial state unavailable.*wall deadline"):
                update_tunnel_dynamic(context, 1.3, {}, 27.0)
        self.assertEqual(context.tunnel_dynamic_phase, "failed")
        self.assertFalse(context.tunnel_dynamic_result["spawn_state_verified"])
        self.assertFalse(context.tunnel_dynamic_result["completed"])
        self.assertNotIn("last_position", context.tunnel_dynamic_result)
        self.assertEqual(context.grpc.spawn_pedestrian.call_count, 1)
        self.assertEqual(context.tunnel_dynamic_result["initial_state_read_misses"], 4)
        self.assertIn("found 0", context.tunnel_dynamic_result["last_initial_state_miss"]["error"])

    def test_spawn_initialization_can_survive_several_fast_misses_then_verify_pose(self):
        context, actor = self.start_observation_context()
        actor.get_actor_state.side_effect = [None, None, None, None, self.observed_state()]
        ack = context.tunnel_dynamic_spawn_ack_monotonic
        with patch.object(tunnel_dynamic.time, "monotonic", return_value=ack + .15):
            for elapsed in (1.02, 1.04, 1.06, 1.08):
                update_tunnel_dynamic(context, elapsed, {}, 27.0)
                self.assertEqual(context.tunnel_dynamic_phase, "spawned_unobserved")
                self.assertFalse(context.tunnel_dynamic_result["spawn_state_verified"])
                self.assertNotIn("last_position", context.tunnel_dynamic_result)
            update_tunnel_dynamic(context, 1.15, {}, 27.0)
        self.assertEqual(context.tunnel_dynamic_phase, "moving")
        self.assertTrue(context.tunnel_dynamic_result["spawn_state_verified"])
        self.assertEqual(context.tunnel_dynamic_result["initial_state_read_misses"], 4)
        self.assertEqual(context.tunnel_dynamic_result["first_verified_state_source"], "GetActorState")
        self.assertAlmostEqual(context.tunnel_dynamic_result["first_verified_state_delay_wall_seconds"], .15)
        self.assertEqual(context.tunnel_dynamic_result["state_read_failures"], 0)

    def test_confirmed_actor_still_fails_after_three_state_losses(self):
        context, actor = self.start_observation_context()
        actor.get_actor_state.side_effect = [self.observed_state(), None, None, None]
        with patch.object(tunnel_dynamic.time, "sleep"):
            update_tunnel_dynamic(context, 1.05, {}, 27.0)
            self.assertEqual(context.tunnel_dynamic_phase, "moving")
            update_tunnel_dynamic(context, 1.1, {}, 27.0)
            update_tunnel_dynamic(context, 1.15, {}, 27.0)
            with self.assertRaisesRegex(RuntimeError, "state unavailable.*found 0"):
                update_tunnel_dynamic(context, 1.2, {}, 27.0)
        self.assertEqual(context.tunnel_dynamic_phase, "failed")
        self.assertEqual(context.tunnel_dynamic_result["state_failure_phase"], "moving")
        self.assertEqual(context.tunnel_dynamic_result["state_read_failures"], 3)
        self.assertFalse(context.tunnel_dynamic_result["completed"])

    def test_init_deadline_cannot_extend_an_unsafe_or_missed_interaction(self):
        for current_s, ego_state in ((51.0, {}), (45.0, {"speed_mps": 8.0})):
            with self.subTest(current_s=current_s):
                context, actor = self.start_observation_context()
                actor.get_actor_state.return_value = None
                with patch.object(tunnel_dynamic.time, "sleep"), self.assertRaisesRegex(RuntimeError, "unsafe interaction window"):
                    update_tunnel_dynamic(context, 1.05, ego_state, current_s)
                self.assertEqual(context.tunnel_dynamic_phase, "failed")
                self.assertFalse(context.tunnel_dynamic_result["spawn_state_verified"])
                self.assertFalse(context.tunnel_dynamic_result["completed"])
                actor.get_actor_state.assert_called_once()

    def test_verified_first_pose_is_accepted_when_ego_reaches_stopping_bound(self):
        context, actor = self.start_observation_context()
        actor.get_actor_state.return_value = self.observed_state()
        update_tunnel_dynamic(context, 1.05, {"speed_mps": 8.0}, 45.0)
        timing = context.tunnel_dynamic_result["initial_state_interaction_timing"]
        self.assertLess(timing["remaining_m"], timing["safe_stop_distance_m"])
        self.assertTrue(context.tunnel_dynamic_result["spawn_state_verified"])
        self.assertEqual(context.tunnel_dynamic_phase, "moving")
        self.assertFalse(context.tunnel_dynamic_result["completed"])

    def test_late_first_state_return_is_rejected_even_if_direct_query_started_in_time(self):
        context, actor = self.start_observation_context()
        ack = context.tunnel_dynamic_spawn_ack_monotonic
        actor.get_actor_state.return_value = self.observed_state()
        with patch.object(tunnel_dynamic.time, "monotonic", side_effect=[ack + .1, ack + 1.1]), patch.object(
                tunnel_dynamic.time, "sleep"), self.assertRaisesRegex(RuntimeError, "first state arrived after"):
            update_tunnel_dynamic(context, 1.05, {}, 27.0)
        self.assertFalse(context.tunnel_dynamic_result["spawn_state_verified"])

    def test_transient_miss_recovers_on_direct_read_without_reusing_last_pose(self):
        context, actor = self.start_observation_context()
        actor.get_actor_state.side_effect = [None, self.observed_state(x=46.0)]
        update_tunnel_dynamic(context, 1.1, {}, 27.0)
        self.assertNotIn("last_position", context.tunnel_dynamic_result)
        self.assertEqual(context.tunnel_dynamic_phase, "spawned_unobserved")
        update_tunnel_dynamic(context, 1.2, {}, 27.0)
        self.assertEqual(context.tunnel_dynamic_result["last_position"]["x"], 46.0)
        self.assertEqual(context.tunnel_dynamic_result["state_read_failures"], 0)
        self.assertEqual(context.tunnel_dynamic_result["last_state_source"], "GetActorState")

    def test_nonfinite_inventory_pose_is_not_verified(self):
        context, actor = self.start_observation_context()
        actor.get_actor_state.return_value = None
        context.grpc.get_all_pedestrian_states.side_effect = [
            [self.observed_state(x=float("nan"))], [], [],
        ]
        with patch.object(tunnel_dynamic.time, "sleep"), self.assertRaisesRegex(RuntimeError, "not finite"):
            update_tunnel_dynamic(context, 1.2, {}, 27.0)
        self.assertFalse(context.tunnel_dynamic_result["spawn_state_verified"])
        self.assertFalse(context.tunnel_dynamic_result["completed"])

    def test_reused_id_old_pose_keeps_jump_guard_and_never_completes_next_event(self):
        context, actor = self.start_observation_context()
        context.tunnel_dynamic_plan["trial_candidate"] = True
        context.tunnel_dynamic_random_mix = True
        actor.get_actor_state.return_value = None
        context.grpc.get_all_pedestrian_states.side_effect = [
            [self.observed_state(x=50.0, y=-2.0)], [], [],
        ]
        with patch.object(tunnel_dynamic.time, "sleep"), self.assertRaisesRegex(RuntimeError, "Implausible pedestrian jump"):
            update_tunnel_dynamic(context, 1.02, {}, 27.0)
        self.assertEqual(context.tunnel_dynamic_phase, "failed")
        self.assertFalse(context.tunnel_dynamic_result["completed"])
        self.assertFalse(context.tunnel_dynamic_result["spawn_state_verified"])

    def test_removal_waits_for_two_absences_after_delayed_delete_before_next_event(self):
        context, actor = self.start_observation_context()
        context.grpc.get_all_pedestrian_states.side_effect = [
            [self.observed_state()], [], [],
        ]
        with patch.object(tunnel_dynamic.time, "sleep") as sleep:
            context.tunnel_dynamic_handler._destroy(context)
        self.assertEqual(context.grpc.get_all_pedestrian_states.call_count, 3)
        self.assertEqual(sleep.call_args_list, [((0.1,),), ((0.1,),)])
        self.assertTrue(context.tunnel_dynamic_result["removal_observation"]["verified"])
        self.assertEqual(context.tunnel_dynamic_result["removal_observation"]["absence_confirmations"], 2)
        self.assertIsNone(context.tunnel_dynamic_actor)
        self.assertEqual(context.scenario_actors, [])
        actor.destroy.assert_called_once()

    def test_missing_or_ambiguous_delete_ack_requires_actual_absence(self):
        for ack in (None, False):
            with self.subTest(ack=ack):
                context, actor = self.start_observation_context()
                actor.destroy.return_value = ack
                with patch.object(tunnel_dynamic.time, "sleep"):
                    context.tunnel_dynamic_handler._destroy(context)
                self.assertTrue(context.tunnel_dynamic_result["removal_observation"]["verified"])
                self.assertEqual(context.grpc.get_all_pedestrian_states.call_count, 2)

    def test_delete_ack_success_cannot_clear_present_actor_or_arm_next_event(self):
        context, actor = self.start_observation_context()
        context.grpc.get_all_pedestrian_states.return_value = [self.observed_state()]
        with patch.object(tunnel_dynamic.time, "sleep"), self.assertRaisesRegex(RuntimeError, "removal unverified"):
            context.tunnel_dynamic_handler._destroy(context)
        self.assertIs(context.tunnel_dynamic_actor, actor)
        self.assertEqual(context.scenario_actors, [actor])
        self.assertFalse(context.tunnel_dynamic_result["completed"])
        self.assertFalse(context.tunnel_dynamic_result["removal_observation"]["verified"])
        self.assertEqual(context.grpc.spawn_pedestrian.call_count, 1)

    def test_failed_inventory_read_cannot_certify_deletion(self):
        context, actor = self.start_observation_context()
        context.grpc.get_all_pedestrian_states.side_effect = [
            [], RuntimeError("inventory unavailable"), [],
        ]
        with patch.object(tunnel_dynamic.time, "sleep"), self.assertRaisesRegex(RuntimeError, "removal unverified"):
            context.tunnel_dynamic_handler._destroy(context)
        self.assertIs(context.tunnel_dynamic_actor, actor)
        self.assertEqual(context.tunnel_dynamic_result["removal_observation"]["absence_confirmations"], 1)

    def test_terminal_pose_cannot_arm_next_section_before_removal_verification(self):
        context, actor = self.start_observation_context()
        next_plan = dict(context.tunnel_dynamic_plan, path_id="next_section")
        context.tunnel_dynamic_plans.append(next_plan)
        context.tunnel_dynamic_results.append(
            context.tunnel_dynamic_handler._event_result(context, next_plan))
        context.tunnel_dynamic_target_index = 2
        actor.get_actor_state.return_value = self.observed_state(x=50.0, y=-2.0)
        context.grpc.get_all_pedestrian_states.return_value = [self.observed_state()]
        with patch.object(tunnel_dynamic.time, "sleep"), self.assertRaisesRegex(RuntimeError, "removal unverified"):
            update_tunnel_dynamic(context, 3.0, {}, 28.0)
        self.assertEqual(context.tunnel_dynamic_phase, "failed")
        self.assertEqual(context.tunnel_dynamic_event_index, 0)
        self.assertFalse(context.tunnel_dynamic_results[0]["completed"])
        self.assertFalse(context.tunnel_dynamic_results[1]["spawned"])
        self.assertEqual(context.grpc.spawn_pedestrian.call_count, 1)


if __name__ == "__main__":
    unittest.main()
