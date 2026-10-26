"""Stationary preview contracts exercised without a simulator or Expert."""

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from ..scenarios.full_loop_parts.tunnel_static_parts import live
from ..scenarios.full_loop_parts.tunnel_static_parts.layout import generate_candidate, TUNNEL_LINK, OPPOSING_LINK


def xyz(x=0., y=0., z=0.):
    return NS(x=x, y=y, z=z)


def state(identity, x, y, *, width=1.2, length=1.2, heading=0., model="WoodBox"):
    return NS(actor_info=NS(id=NS(value=identity), client_key="", model=model,object_type=1 if identity=="Ego" else 3),
              transform=NS(location=xyz(x, y, 28.), rotation=xyz(z=heading)),
              bounding_box=NS(location=xyz(x, y, 28.), extent=xyz(width/2, length/2, .75), rotation=xyz(z=heading)),
              global_velocity=xyz(), angular_velocity=xyz(), vehicle_state=NS(gear_mode=1))


class Actor:
    def __init__(self, env, identity):
        self.env, self.identity = env, identity

    def get_id(self):
        return self.identity

    def get_object_info(self):
        return NS(id=NS(value=self.identity), client_key="preview-client")

    def _state(self):
        return self.env.states[self.env.aliases.get(self.identity, self.identity)]

    def set_velocity(self, value):
        self.env.commands.append((self.identity, "velocity", value))
        self._state().global_velocity = xyz(value)
        return True

    def set_vehicle_gear(self, value):
        self.env.commands.append((self.identity, "gear", value))
        self._state().vehicle_state.gear_mode = value
        return True

    def set_vehicle_dynamics_speed_limit(self, enabled, value, reset=False):
        self.env.commands.append((self.identity, "speed_limit", enabled, value, reset))
        return not self.env.reject_speed_limit

    def set_pause(self, value):
        self.env.commands.append((self.identity, "pause", value))
        return True

    def set_ai(self, value):
        self.env.commands.append((self.identity, "ai", value))
        return False  # Optional API rejection must not substitute for observation.

    def control(self, **kwargs):
        self.env.commands.append((self.identity, "control", kwargs))
        return True

    def set_physics(self, value):
        self.env.commands.append((self.identity, "physics", value))
        return True

    def set_transform(self, transform):
        self.env.commands.append((self.identity, "transform"))
        current = self._state()
        current.transform.location = copy.deepcopy(transform.location)
        current.transform.rotation = xyz(z=transform.rotation.z + 1.347)
        current.bounding_box.location = copy.deepcopy(transform.location)
        current.bounding_box.rotation = copy.deepcopy(current.transform.rotation)
        return True

    def destroy(self):
        self.env.commands.append((self.identity, "destroy"))
        identity = self.env.aliases.get(self.identity, self.identity)
        if not self.env.ignore_destroy:
            self.env.states.pop(identity, None)


class Simulator:
    def __init__(self):
        self.states = {"Ego": state("Ego", -40., 0., width=2., length=4.5, model="EgoCar")}
        self.aliases, self.commands, self.spawns, self.owners = {}, [], [], {}
        self.counter = 0
        self.reject_speed_limit = self.ambiguous_spawn = self.ignore_destroy = False
        self.inventory_hook = None
        self.stub = NS(GetAllActorsState=self.inventory, GetActorState=self.one,
                       GetOptionName=lambda info, timeout: NS(value=self.states[info.id.value].actor_info.model),
                       DestroyActor=self.destroy_raw,DestroyAllActors=self.destroy_namespace)
        self.world = NS(_sim_adapter=NS(_actor_stub=self.stub), spawn_vehicle=self.vehicle,
                        spawn_obstacle=self.obstacle)
        self.grpc = NS(world=self.world, client_key="preview-client", get_ego=lambda: Actor(self, "Ego"),
                       make_transform=lambda x, y, z, yaw: NS(location=xyz(x, y, z), rotation=xyz(z=yaw)),
                       set_vehicle_route=self.route)

    def route(self, handle, links, **kwargs):
        self.commands.append((handle.identity, "route", links, kwargs))
        return True

    def inventory(self, query, timeout):
        self.assert_global(query, timeout)
        self.counter += 1
        if self.inventory_hook:
            self.inventory_hook(self)
        return NS(states=list(self.states.values()))

    @staticmethod
    def assert_global(query, timeout):
        assert query.client_key == ""
        assert query.vehicle and query.obstacle and query.pedestrian
        assert 0 < timeout <= 1.

    def one(self, info, timeout):
        assert 0 < timeout <= 1.
        return self.states.get(self.aliases.get(info.id.value, info.id.value))

    def destroy_raw(self,info,timeout):
        assert info.client_key in ("","preview-client")
        assert info.id.value in self.states and info.id.value!="Ego"
        assert 0<timeout<=1.
        self.commands.append((info.id.value,"destroy"))
        if not self.ignore_destroy:
            self.states.pop(info.id.value)
        return NS(status=1,description="owned raw actor delete accepted")

    def destroy_namespace(self,request,timeout):
        assert request.value=="preview-client"
        assert 0<timeout<=1.
        self.commands.append((request.value,"destroy_namespace"))
        if not self.ignore_destroy:
            for identity in list(self.states):
                if identity!="Ego" and self.owners.get(identity)==request.value:
                    self.states.pop(identity)
        return NS(status=1,description="fixture namespace delete accepted")

    def spawn(self, kind, kwargs):
        assert self.states["Ego"].vehicle_state.gear_mode == 1
        self.spawns.append((kind, kwargs))
        raw = str(100 + len(self.spawns))
        self.aliases[kwargs["request_id"]] = raw
        transform = kwargs["transform"]
        width, length = (1.8, 4.) if kind == "vehicle" else (1.2, 1.2)
        actor = state(raw, transform.location.x, transform.location.y, width=width, length=length,
                      heading=transform.rotation.z + 1.347, model=kwargs["model_name"])
        actor.actor_info.client_key="preview-client"
        actor.actor_info.object_type=1 if kind=="vehicle" else 3
        self.owners[raw]="preview-client"
        self.states[raw] = actor
        if self.ambiguous_spawn:
            raise RuntimeError("RPC returned no status after creating the actor")
        return Actor(self, kwargs["request_id"])

    def obstacle(self, **kwargs):
        return self.spawn("obstacle", kwargs)

    def vehicle(self, **kwargs):
        return self.spawn("vehicle", kwargs)


class LivePreviewTest(unittest.TestCase):
    def setUp(self):
        self.sim = Simulator()
        links = {TUNNEL_LINK: {"road_type": 2, "width_start": 3.5, "width_end": 3.5,
                               "points": [(0., 0., 28.), (60., 0., 28.), (120., 0., 28.)]},
                 OPPOSING_LINK: {"road_type": 2, "width_start": 3.5, "width_end": 3.5,
                                "points": [(120., 3.5, 28.), (60., 3.5, 28.), (0., 3.5, 28.)]}}
        self.loader = NS(get_link=lambda identity: links[identity])
        self.static_catalog = {"WoodBox": {"width_m": 1.2, "length_m": 1.2,
                                          "heading_offset_deg": 1.347, "center_offset_spawn_xy": [0., 0.]}}
        self.vehicle_catalog = {"Sedan": {"width_m": 1.8, "length_m": 4.,
                                         "heading_offset_deg": 1.347, "center_offset_spawn_xy": [0., 0.],
                                         "geometry_verified": True}}
        modules = {"proto.morai.actor.actor_enum_pb2": NS(GEAR_MODE_P=1, LONG_CMD_TYPE_SPEED=2),
                   "proto.morai.actor.actor_get_pb2": NS(GetAllActorsFilter=lambda **kw: NS(**kw)),
                   "proto.morai.common.type_pb2": NS(Vector3=lambda **kw: NS(**kw),StringValue=lambda **kw:NS(**kw)),
                   "proto.morai.common.enum_pb2":NS(STATUS_CODE_SUCCESS=1,OBJECT_TYPE_VEHICLE=1,OBJECT_TYPE_OBSTACLE=3)}
        self.patch_modules = patch.dict(sys.modules, modules)
        self.patch_modules.start()
        self.addCleanup(self.patch_modules.stop)
        self.handle_patch = patch.object(live, "_handle_for_request", side_effect=lambda grpc, kind, rid: Actor(self.sim, rid))
        self.handle_patch.start()
        self.addCleanup(self.handle_patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def candidate(self, kind="obstacle", count=1):
        return generate_candidate(self.loader, self.static_catalog, self.vehicle_catalog, 42,
                                  count=count, actor_kind=kind)

    def run_preview(self, candidate=None, **kwargs):
        with patch.object(live.time, "sleep"):
            return live.preview_candidate(self.sim.grpc, self.loader, candidate or self.candidate(),
                                          hold_seconds=0., output_dir=self.directory, **kwargs)

    def assert_clean(self, report):
        self.assertTrue(report["cleanup"]["verified"], report)
        self.assertEqual(list(self.sim.states), ["Ego"])
        self.assertEqual(self.sim.states["Ego"].vehicle_state.gear_mode, 1)

    def test_static_preview_freezes_physics_parks_ego_and_does_not_approve_driving(self):
        report = self.run_preview()
        self.assertEqual(report["status"], "valid_preview", report)
        self.assert_clean(report)
        self.assertFalse(report["expert_verified"])
        self.assertFalse(report["live_avoidance_verified"])
        self.assertFalse(report["route_screened"])
        self.assertGreaterEqual(report["ego_parking_commands"], 3)
        self.assertIn((report["observed"][0]["handle_id"], "physics", False), self.sim.commands)
        self.assertAlmostEqual(report["observed"][0]["observed_bbox_yaw_offset_deg"], 1.347)
        self.assertEqual(json.loads((self.directory / "preview.json").read_text())["status"], "valid_preview")
        self.assertTrue(all(command[1] in {"velocity", "gear", "physics", "transform", "destroy","destroy_namespace"} for command in self.sim.commands))

    def test_actual_parked_vehicle_uses_npc_api_and_observed_stationarity(self):
        report = self.run_preview(self.candidate("vehicle"))
        self.assertEqual(report["status"], "valid_preview", report)
        self.assert_clean(report)
        kind, args = self.sim.spawns[0]
        self.assertEqual(kind, "vehicle")
        self.assertFalse(args["multi_ego"])
        self.assertFalse(args["pause"])
        self.assertEqual(args["velocity"], 0.)
        observed = report["observed"][0]
        self.assertTrue(observed["stationary_controls_accepted"]["default_ai_retained"])
        self.assertTrue(observed["stationary_controls_accepted"]["default_physics_retained"])
        self.assertTrue(observed["stationary_controls_accepted"]["native_route"])
        self.assertFalse(any(c[0] == observed["handle_id"] and c[1] in ("ai", "pause", "gear", "physics") for c in self.sim.commands))
        self.assertEqual(observed["observed_gear_mode"], 1)
        self.assertEqual(observed["raw_actor_id"], "101")
        self.assertNotEqual(observed["handle_id"], observed["raw_actor_id"])

    def test_two_actors_are_both_removed_and_measured_screening_is_required_to_claim_path(self):
        screen = Mock(return_value={"status": "nominal_path_found", "path_found": True,
                                    "geometry_checks": {"passed": True}})
        report = self.run_preview(self.candidate("vehicle", 2), screen_fn=screen)
        self.assertEqual(report["status"], "valid_preview", report)
        self.assertEqual(len(report["observed"]), 2)
        self.assert_clean(report)
        self.assertTrue(report["route_screened"])
        measured = screen.call_args.kwargs["measured_geometry"]
        self.assertEqual(len(measured), 2)
        self.assertIn("world_bbox", measured[0])

    def test_screening_pass_label_without_path_and_collision_checks_is_rejected(self):
        for screening in ({"status": "passed"}, {"status": "nominal_path_found", "path_found": True}):
            with self.subTest(screening=screening):
                self.sim = Simulator()
                report = self.run_preview(screen_fn=lambda *a, **k: screening)
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["live_avoidance_verified"])
                self.assert_clean(report)

    def test_measured_ego_overhang_and_pose_are_sent_to_the_same_planner(self):
        self.sim.states["Ego"].bounding_box.location.x += 1.5
        screen = Mock(return_value={"status": "nominal_path_found", "path_found": True,
                                    "geometry_checks": {"passed": True}})
        report = self.run_preview(screen_fn=screen)
        self.assertEqual(report["status"], "valid_preview", report)
        footprint = screen.call_args.kwargs["actual_ego_footprint"]
        self.assertAlmostEqual(footprint["center_forward_m"], 1.5)
        self.assertAlmostEqual(footprint["center_left_m"], 0.)
        self.assertEqual(screen.call_args.kwargs["actual_ego_state"], report["ego_observed"])
        self.assert_clean(report)

    def test_tilted_ego_box_recovers_intrinsic_center_before_projection(self):
        roll, pitch, yaw = map(math.radians, (2.2, .7, -150.))
        cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
        rotation = ((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
                    (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr), (-sp, cp*sr, cp*cr))
        intrinsic, pivot = [1.53, -.036, .46], [29., -490., 28.6]
        center = [pivot[i]+sum(rotation[i][j]*intrinsic[j] for j in range(3)) for i in range(3)]
        observed = {"position_xyz": pivot, "rotation_xyz_deg": [2.2, .7, -150.],
                    "raw_bbox_normalized": {"center_xyz": center, "length_m": 4.635, "width_m": 2.16},
                    "world_bbox": {"extent_xyz": [1.08, 2.3175, .82]}}
        measured = live._ego_footprint(observed)
        for actual, expected in zip(measured["center_actor_flu_m"], intrinsic):
            self.assertAlmostEqual(actual, expected, places=11)
        self.assertAlmostEqual(measured["height_m"], 1.64)
        self.assertEqual(measured["max_abs_roll_pitch_deg"], 3.)

    def test_driving_callback_is_not_called_when_measured_path_fails(self):
        drive = Mock()
        report = self.run_preview(screen_fn=Mock(return_value={"status": "no_path_found", "path_found": False}),
                                  drive_fn=drive)
        drive.assert_not_called()
        self.assertFalse(report["expert_verified"])
        self.assert_clean(report)

    def test_driving_callback_failure_still_removes_every_owned_actor(self):
        screen = Mock(return_value={"status": "nominal_path_found", "path_found": True,
                                    "geometry_checks": {"passed": True}})
        report = self.run_preview(self.candidate("vehicle", 2), screen_fn=screen,
                                  drive_fn=Mock(side_effect=RuntimeError("Trial stopped")))
        self.assertEqual(report["status"], "failed")
        self.assertIn("Trial stopped", report["failure_reason"])
        self.assertFalse(report["live_avoidance_verified"])
        self.assert_clean(report)

    def test_trial_success_requires_all_passage_and_return_evidence(self):
        screen = Mock(return_value={"status": "nominal_path_found", "path_found": True,
                                    "geometry_checks": {"passed": True}})
        report = self.run_preview(screen_fn=screen, drive_fn=Mock(return_value={"status": "passed"}))
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["expert_verified"])
        self.assertIn("complete passage/return evidence", report["failure_reason"])
        self.assert_clean(report)

    def test_trial_callback_runs_with_owned_scene_and_full_proof_allows_success(self):
        screen = Mock(return_value={"status": "nominal_path_found", "path_found": True,
                                    "geometry_checks": {"passed": True}})
        def drive(report, owned):
            self.assertEqual(report["status"], "driving_trial")
            self.assertEqual(len(owned), 1)
            self.assertIn(owned[0]["raw_id"], self.sim.states)
            self.assertEqual(report["owned_actor_geometry"], report["observed"])
            return {"status": "passed", "completed": True, "collision_free": True,
                    "returned_to_original_lane": True, "fixture_stationary": True}
        report = self.run_preview(screen_fn=screen, drive_fn=drive)
        self.assertEqual(report["status"], "valid_driving_trial", report)
        self.assertTrue(report["live_avoidance_verified"])
        self.assertTrue(report["expert_verified"])
        self.assert_clean(report)

    def test_foreign_actor_blocks_preview_and_is_never_deleted_or_relocated(self):
        foreign = state("foreign", 35., 3.5, model="OtherCar")
        self.sim.states["foreign"] = foreign
        report = self.run_preview()
        self.assertEqual(report["status"], "failed")
        self.assertIn("Foreign actors", report["failure_reason"])
        self.assertEqual(self.sim.spawns, [])
        self.assertIs(self.sim.states["foreign"], foreign)
        self.assertEqual(foreign.transform.location.x, 35.)
        self.assertFalse(any(c[0] == "foreign" for c in self.sim.commands))

    def test_foreign_actor_far_outside_corridor_is_preserved(self):
        self.sim.states["foreign"] = state("foreign", 50., 30.)
        report = self.run_preview()
        self.assertEqual(report["status"], "valid_preview", report)
        self.assertIn("foreign", self.sim.states)
        self.assertTrue(report["cleanup"]["verified"])

    def test_ego_motion_is_detected_before_periodic_zero_velocity_can_mask_it(self):
        original = self.sim.one
        reads = 0
        def disturbed(info, timeout):
            nonlocal reads
            if info.id.value == "Ego":
                reads += 1
                if reads >= 4:
                    self.sim.states["Ego"].global_velocity = xyz(.3)
            return original(info, timeout)
        self.sim.stub.GetActorState = disturbed
        report = self.run_preview(self.candidate("vehicle"))
        self.assertEqual(report["status"], "failed")
        self.assertIn("Ego is moving", report["failure_reason"])
        self.assertFalse(report["live_avoidance_verified"])
        self.assertNotIn("101", self.sim.states)

    def test_observed_vehicle_profile_drift_fails_and_is_retained_in_report(self):
        candidate = self.candidate("vehicle")
        candidate["obstacles"][0]["expected_api_raw"] = {
            "size_lwh_m": [4., 1.8, 2.], "center_offset_actor_flu_m": [0., 0., 0.]}
        report = self.run_preview(candidate)
        self.assertEqual(report["status"], "failed")
        self.assertIn("reviewed vehicle profile", report["failure_reason"])
        self.assertEqual(len(report["observed"]), 1)
        self.assert_clean(report)

    def test_rejected_vehicle_stationary_contract_still_cleans_created_actor(self):
        self.sim.reject_speed_limit = True
        report = self.run_preview(self.candidate("vehicle"))
        self.assertEqual(report["status"], "failed")
        self.assertIn("stationary commands", report["failure_reason"])
        self.assert_clean(report)

    def test_ambiguous_spawn_recovers_exact_request_id_and_cleans_it(self):
        self.sim.ambiguous_spawn = True
        report = self.run_preview(self.candidate("vehicle"))
        self.assertEqual(report["status"], "valid_preview", report)
        self.assertIn("creating", report["observed"][0]["spawn_rpc_error"])
        self.assertEqual(len(self.sim.spawns), 1)
        self.assert_clean(report)

    def test_initial_global_registration_delay_requeries_without_duplicate_spawn(self):
        original = self.sim.stub.GetAllActorsState
        queries = 0
        def delayed(query, timeout):
            nonlocal queries
            response = original(query, timeout)
            if self.sim.spawns:
                queries += 1
                if queries <= 3:
                    response.states = [actor for actor in response.states if actor.actor_info.id.value != "101"]
            return response
        self.sim.stub.GetAllActorsState = delayed
        report = self.run_preview()
        self.assertEqual(report["status"], "valid_preview", report)
        self.assertEqual(len(self.sim.spawns), 1)
        self.assertEqual(report["spawn_readbacks"][0]["attempts"], 4)
        self.assertEqual(report["spawn_readbacks"][0]["raw_actor_id"], "101")
        self.assert_clean(report)

    def test_request_alias_readback_resolves_global_id_before_model_lookup(self):
        original = self.sim.stub.GetActorState
        model_queries = []
        original_model = self.sim.stub.GetOptionName
        def aliased(info, timeout):
            observed = original(info, timeout)
            if observed is not None and info.id.value != "Ego":
                observed = copy.deepcopy(observed)
                observed.actor_info.id.value = info.id.value
            return observed
        def authoritative_model(info, timeout):
            model_queries.append(info.id.value)
            return original_model(info, timeout)
        self.sim.stub.GetActorState = aliased
        self.sim.stub.GetOptionName = authoritative_model
        report = self.run_preview()
        self.assertEqual(report["status"], "valid_preview", report)
        self.assertEqual(len(self.sim.spawns), 1)
        self.assertEqual(report["spawn_readbacks"][0]["raw_actor_id"], "101")
        self.assertTrue(model_queries)
        self.assertEqual(set(model_queries), {"101"})
        self.assert_clean(report)

    def test_initial_direct_readback_delay_after_ambiguous_spawn_recovers_same_request(self):
        self.sim.ambiguous_spawn = True
        original = self.sim.stub.GetActorState
        reads = 0
        def delayed(info, timeout):
            nonlocal reads
            if info.id.value != "Ego":
                reads += 1
                if reads <= 2:
                    return None
            return original(info, timeout)
        self.sim.stub.GetActorState = delayed
        report = self.run_preview()
        self.assertEqual(report["status"], "valid_preview", report)
        self.assertEqual(len(self.sim.spawns), 1)
        self.assertEqual(report["spawn_readbacks"][0]["attempts"], 3)
        self.assertIn("creating", report["spawn_readbacks"][0]["spawn_rpc_error"])
        self.assert_clean(report)

    def test_initial_registration_never_visible_stops_after_ten_queries_and_cleans_request(self):
        original = self.sim.stub.GetActorState
        def absent(info, timeout):
            return original(info, timeout) if info.id.value == "Ego" else None
        self.sim.stub.GetActorState = absent
        report = self.run_preview()
        self.assertEqual(report["status"], "failed", report)
        self.assertIn("retry budget", report["failure_reason"])
        self.assertEqual(report["spawn_readbacks"][0]["attempts"], 10)
        self.assertEqual(len(self.sim.spawns), 1)
        self.assert_clean(report)

    def test_wrong_model_during_initial_registration_fails_without_retry_or_respawn(self):
        original = self.sim.world.spawn_obstacle
        def wrong_model(**kwargs):
            handle = original(**kwargs)
            handle._state().actor_info.model = "WrongBox"
            return handle
        self.sim.world.spawn_obstacle = wrong_model
        report = self.run_preview()
        self.assertEqual(report["status"], "failed", report)
        self.assertIn("model differs", report["failure_reason"])
        self.assertEqual(report["spawn_readbacks"][0]["attempts"], 1)
        self.assertEqual(report["spawn_readbacks"][0]["retry_errors"], [])
        self.assertEqual(len(self.sim.spawns), 1)
        self.assert_clean(report)

    def test_initial_static_transform_lock_receives_eight_attempt_budget(self):
        with patch.object(live, "lock_static_obstacle_transform", wraps=live.lock_static_obstacle_transform) as lock:
            report = self.run_preview()
        self.assertEqual(report["status"], "valid_preview", report)
        self.assertEqual(lock.call_args.kwargs["attempts"], 8)
        self.assertEqual(lock.call_args.kwargs["stable_samples"], 2)
        self.assert_clean(report)

    def test_initial_registration_rpcs_share_one_deadline_and_never_respawn(self):
        clock = [0.]
        original = self.sim.stub.GetAllActorsState
        timeouts = []
        def slow_inventory(query, timeout):
            timeouts.append(timeout)
            response = original(query, timeout)
            clock[0] += min(.55, timeout)
            return response
        def advance_sleep(duration):
            clock[0] += duration
        self.sim.stub.GetAllActorsState = slow_inventory
        original_one = self.sim.stub.GetActorState
        self.sim.stub.GetActorState = lambda info, timeout: (
            original_one(info, timeout) if info.id.value == "Ego" else None)
        with patch.object(live.time, "monotonic", side_effect=lambda: clock[0]), patch.object(
                live.time, "sleep", side_effect=advance_sleep):
            report = live.preview_candidate(self.sim.grpc, self.loader, self.candidate(),
                hold_seconds=0., output_dir=self.directory)
        self.assertEqual(report["status"], "failed", report)
        self.assertEqual(len(self.sim.spawns), 1)
        readback = report["spawn_readbacks"][0]
        self.assertLessEqual(readback["elapsed_sec"], 1.5)
        self.assertLess(readback["attempts"], 10)
        self.assertTrue(any(timeout < 1. for timeout in timeouts))
        self.assert_clean(report)

    def test_motion_during_hold_fails_and_cleans_actor(self):
        def motion(sim):
            if sim.counter >= 5 and "101" in sim.states:
                sim.states["101"].global_velocity = xyz(.3)
        self.sim.inventory_hook = motion
        report = self.run_preview(self.candidate("vehicle"))
        self.assertEqual(report["status"], "failed", report)
        self.assertIn("moving", report["failure_reason"])
        self.assert_clean(report)

    def test_new_foreign_occupancy_after_spawn_aborts_and_removes_only_owned(self):
        foreign = state("foreign", 50., 3.5)
        def arriving(sim):
            if sim.counter >= 5:
                sim.states["foreign"] = foreign
        self.sim.inventory_hook = arriving
        report = self.run_preview(self.candidate("vehicle"))
        self.assertEqual(report["status"], "failed")
        self.assertIn("Foreign actors", report["failure_reason"])
        self.assertEqual(set(self.sim.states), {"Ego", "foreign"})
        self.assertIs(self.sim.states["foreign"], foreign)
        self.assertTrue(report["cleanup"]["verified"])

    def test_fresh_inventory_failure_cannot_reuse_last_successful_inventory(self):
        def failed_query(sim):
            if sim.counter == 5:
                raise RuntimeError("deadline exceeded")
        self.sim.inventory_hook = failed_query
        report = self.run_preview(self.candidate("vehicle"))
        self.assertEqual(report["status"], "failed")
        self.assertIn("deadline", report["failure_reason"])
        self.assert_clean(report)

    def test_cleanup_must_be_verified_by_fresh_global_readback(self):
        self.sim.ignore_destroy = True
        report = self.run_preview()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["cleanup"]["verified"])
        self.assertIn("remain after cleanup", report["failure_reason"])

    def test_reviewed_vehicle_envelope_is_never_shrunk_to_raw_default_box(self):
        planned = {"center_xyz": [0., 0., 0.], "width_m": 2.4, "length_m": 5., "heading_deg": 0.}
        raw = {"center_xyz": [0., 0., 1.], "width_m": 1.8, "length_m": 4., "heading_deg": 0.}
        box = live._conservative_box(planned, raw)
        self.assertEqual(box["width_m"], 2.4)
        self.assertEqual(box["length_m"], 5.)

    def test_actual_wider_bbox_can_fail_road_margin_even_when_requested_bbox_fits(self):
        candidate = self.candidate("vehicle")
        original = self.sim.vehicle
        def wide(**kwargs):
            handle = original(**kwargs)
            handle._state().bounding_box.extent.x = 2.0
            return handle
        self.sim.world.spawn_vehicle = wide
        report = self.run_preview(candidate)
        self.assertEqual(report["status"], "failed")
        self.assertIn("nominal", report["failure_reason"])
        self.assertEqual(len(report["observed"]), 1)
        self.assertGreaterEqual(report["observed"][0]["width_m"], 4.0)
        self.assert_clean(report)

    def test_keyboard_interrupt_persists_failure_and_cleans_exact_owned_request(self):
        original = self.sim.vehicle
        def interrupted(**kwargs):
            original(**kwargs)
            raise KeyboardInterrupt()
        self.sim.world.spawn_vehicle = interrupted
        with self.assertRaises(KeyboardInterrupt):
            self.run_preview(self.candidate("vehicle"))
        report = json.loads((self.directory / "preview.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("KeyboardInterrupt", report["failure_reason"])
        self.assert_clean(report)

    def pivot_candidate(self):
        self.vehicle_catalog["Sedan"].update(center_offset_spawn_xy=[1.3, 0.],
            expected_api_raw={"size_lwh_m": [4., 1.8, 1.5], "center_offset_actor_flu_m": [1.3, 0., .46]},
            geometry_source={"type": "reviewed_vehicle_surface_and_api_collider_union",
                             "model_key": "npc|Sedan|1,1,1", "registry_sha256": "a" * 64})
        return self.candidate("vehicle")

    def test_known_pivot_api_variant_uses_body_and_collider_union_without_claiming_offset_match(self):
        report = self.run_preview(self.pivot_candidate())
        self.assertEqual(report["status"], "valid_preview", report)
        observed = report["observed"][0]
        profile = observed["reviewed_profile_readback"]
        self.assertFalse(profile["catalog_offset_verified"])
        self.assertFalse(profile["checked"])
        self.assertEqual(profile["current_api_center_variant"], "actor_pivot")
        self.assertTrue(profile["direct_and_global_query_agreement"])
        self.assertGreater(observed["length_m"], observed["raw_bbox_normalized"]["length_m"] + 1.)
        self.assertIn("reviewed_collider_at_observed_pose", observed)
        self.assertFalse(report["physical_mesh_verified"])
        self.assertFalse(report["collision_behavior_verified"])
        self.assert_clean(report)

    def test_arbitrary_unreviewed_api_center_offset_is_rejected_instead_of_pivot_fallback(self):
        candidate = self.pivot_candidate()
        original = self.sim.vehicle
        def shifted(**kwargs):
            handle = original(**kwargs)
            handle._state().bounding_box.location.x += .4
            return handle
        self.sim.world.spawn_vehicle = shifted
        report = self.run_preview(candidate)
        self.assertEqual(report["status"], "failed")
        self.assertIn("offset differs", report["failure_reason"])
        self.assert_clean(report)

    def test_known_pivot_api_variant_does_not_bypass_tilt_rejection(self):
        candidate = self.pivot_candidate()
        original = self.sim.vehicle
        def tilted(**kwargs):
            handle = original(**kwargs)
            actor = handle._state()
            actor.transform.rotation.x = actor.bounding_box.rotation.x = 4.
            return handle
        self.sim.world.spawn_vehicle = tilted
        report = self.run_preview(candidate)
        self.assertEqual(report["status"], "failed")
        self.assertIn("roll/pitch", report["failure_reason"])
        self.assert_clean(report)

    def test_opposing_npc_route_uses_native_reverse_lane_and_connected_departure(self):
        links = {
            "A2256W000131": {"from_node_idx": "opposite_entry", "to_node_idx": "opposite_exit",
                "points": [[120., 3.5, 28.], [60., 3.5, 28.], [0., 3.5, 28.]]},
            "A2256W000129": {"from_node_idx": "opposite_exit", "to_node_idx": "departure",
                "points": [[0., 3.5, 28.], [-20., 3.5, 28.]]}}
        loader = NS(get_link=lambda identity: links[identity])
        placement = {"placement_lane_id": "A2256W000131", "position": {"x": 60., "y": 3.5, "z": 28.},
            "yaw_deg": 180., "planned_bbox": {"width_m": 1.8, "length_m": 4.,
                "center_offset_spawn_xy": [0., 0.], "heading_offset_deg": 0.}}
        route, waypoint = live._native_npc_route(loader, placement, {})
        self.assertEqual(route, ["A2256W000131", "A2256W000129"])
        self.assertEqual(waypoint, 1)
        placement["yaw_deg"] = 0.
        with self.assertRaisesRegex(RuntimeError, "native lane flow"):
            live._native_npc_route(loader, placement, {})
        placement["yaw_deg"] = 180.
        links["A2256W000129"]["from_node_idx"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "not connected"):
            live._native_npc_route(loader, placement, {})

    def test_expanded_measured_shape_allows_opposing_lane_only_with_full_road_and_portal_guards(self):
        from shapely.geometry import Polygon
        geometry = {"placement_domain": "two_lane_physical_road", "own_points": [[0., 0., 28.], [120., 0., 28.]],
                    "opposing_points": [[0., 3.5, 28.], [120., 3.5, 28.]],
                    "own_lane_width_m": 3.5, "opposing_lane_width_m": 3.5, "length_m": 120.}
        road = Polygon([(-40., -1.75), (160., -1.75), (160., 5.25), (-40., 5.25)])
        geometry["road_boundary"] = NS(covers_geometry=lambda polygon: road.covers(polygon))
        ego = {"center_xyz": [-30., 0., 28.], "heading_deg": 0., "width_m": 2.3, "length_m": 4.7}
        box = {"center_xyz": [45., 3.5, 28.], "heading_deg": 180., "width_m": 1.8, "length_m": 4.}
        live._geometry_checks([box], ego, geometry)
        legacy = dict(geometry, placement_domain="own_lane")
        with self.assertRaisesRegex(RuntimeError, "opposing-lane corridor"):
            live._geometry_checks([box], ego, legacy)
        with self.assertRaisesRegex(RuntimeError, "shared road margin"):
            live._geometry_checks([dict(box, center_xyz=[45., 5., 28.])], ego, geometry)
        with self.assertRaisesRegex(RuntimeError, "portal"):
            live._geometry_checks([dict(box, center_xyz=[.5, 3.5, 28.])], ego, geometry)
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            live._geometry_checks([box, dict(box, center_xyz=[80., 0., 28.]), dict(box)], ego, geometry)

    def test_expanded_preview_cannot_skip_measured_jamming_passability_screen(self):
        with patch.object(live, "validate_candidate", return_value={"placement_domain": "two_lane_physical_road"}), self.assertRaisesRegex(
                ValueError, "measured full-body"):
            live.preview_candidate(self.sim.grpc, NS(), {"count": 1}, hold_seconds=0.)
        self.assertEqual(self.sim.spawns, [])


if __name__ == "__main__":
    unittest.main()
