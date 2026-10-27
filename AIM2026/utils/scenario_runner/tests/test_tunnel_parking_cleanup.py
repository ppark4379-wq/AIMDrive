"""Tunnel regression checks: parking cleanup."""

import copy, sys, unittest, hashlib, json, tempfile
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
from ..scenarios.full_loop_parts.tunnel_static_parts import live, safety
from ..scenarios.segment_modes import tunnel_static_startup as startup
from .test_tunnel_static_live import state, Actor
from . import test_tunnel_static_live as preview_fixtures
from pathlib import Path
from ..tools import tunnel_static_batch_safety as safety


# Shared brake holds retain preview immutability and bounded readbacks.

class CommonPreviewParkingTests(unittest.TestCase):
    def setUp(self):
        self.ego_state=state("Ego",0.,0.,width=2.,length=4.5)
        self.ego_state.vehicle_state=NS(gear_mode=4,brake=1.,throttle=0.)
        self.observed=live._observation(self.ego_state)
        self.observed["control_mode"]=3
        self.grpc=NS(_tunnel_static_common_start=True)
        self.ego=NS(set_velocity=Mock(),set_vehicle_gear=Mock())
        self.report={"ego_parking_commands":0}
        self.proto=patch.dict(sys.modules,{"proto.morai.actor.actor_enum_pb2":NS(GEAR_MODE_P=1,GEAR_MODE_D=4)})
        self.proto.start()
        self.addCleanup(self.proto.stop)

    def test_common_initial_hold_waits_for_readback_without_issuing_p_or_zero_velocity(self):
        with patch.object(startup,"hold_common_brake",autospec=True) as hold, \
                patch.object(startup,"common_stopped_readback",autospec=True,
                    side_effect=[RuntimeError("suspension still moving"),(self.ego_state,self.observed)]) as read, \
                patch.object(live.time,"sleep") as sleep:
            result=live._park(self.grpc,self.ego,self.report)
        self.assertEqual(result,(self.ego_state,self.observed))
        hold.assert_called_once_with(self.grpc)
        self.assertEqual(read.call_count,2)
        sleep.assert_called_once_with(.1)
        self.ego.set_velocity.assert_not_called()
        self.ego.set_vehicle_gear.assert_not_called()
        self.assertEqual(self.report["ego_parking_commands"],1)

    def test_reference_drift_is_rejected_before_any_brake_command(self):
        with patch.object(startup,"hold_common_brake",autospec=True) as hold, \
                patch.object(startup,"common_stopped_readback",autospec=True,
                    side_effect=RuntimeError("Ego pose moved")) as read:
            with self.assertRaisesRegex(RuntimeError,"pose moved"):
                live._park(self.grpc,self.ego,self.report,self.observed)
        read.assert_called_once_with(self.grpc,self.observed)
        hold.assert_not_called()
        self.assertEqual(self.report["ego_parking_commands"],0)

    def test_reference_checks_precede_hold_and_post_hold_readback_is_fail_fast(self):
        order=[]
        def read(_grpc,reference=None):
            order.append("read")
            if len(order)==3:
                raise RuntimeError("Brake hold lost")
            return self.ego_state,self.observed
        with patch.object(startup,"hold_common_brake",autospec=True,side_effect=lambda _:order.append("hold")), \
                patch.object(startup,"common_stopped_readback",autospec=True,side_effect=read), \
                patch.object(live.time,"sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError,"Brake hold lost"):
                live._park(self.grpc,self.ego,self.report,self.observed)
        self.assertEqual(order,["read","hold","read"])
        sleep.assert_not_called()

    def test_initial_hold_keeps_twenty_read_limit(self):
        with patch.object(startup,"hold_common_brake",autospec=True) as hold, \
                patch.object(startup,"common_stopped_readback",autospec=True,
                    side_effect=RuntimeError("No verified stopped state")) as read, \
                patch.object(live.time,"sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError,"No verified stopped state"):
                live._park(self.grpc,self.ego,self.report)
        self.assertEqual(read.call_count,20)
        self.assertEqual(sleep.call_count,19)
        hold.assert_called_once()

    def test_common_spawn_registration_verifies_hold_and_keeps_actual_ego_geometry(self):
        diagnostics={}
        inventory={"Ego":self.ego_state,"7":state("7",10.,3.5)}
        with patch.object(live,"_inventory",return_value=inventory), \
                patch.object(startup,"common_stopped_readback",autospec=True,return_value=(self.ego_state,self.observed)) as read, \
                patch.object(live,"_resolve_raw",return_value="7"):
            result=live._wait_spawn_readback(self.grpc,NS(),{"Ego"},"WoodBox",{},diagnostics,"Ego",self.observed)
        self.assertEqual(result,(inventory,"7"))
        read.assert_called_once_with(self.grpc,self.observed)
        self.assertEqual(diagnostics["status"],"verified")

    def test_spawn_inventory_cannot_hide_lost_gear_brake_throttle_or_motion(self):
        for field,value in (("gear_mode",1),("brake",.98),("throttle",.02),("throttle",-.02),("brake",float("nan"))):
            with self.subTest(field=field,value=value):
                wrong=copy.deepcopy(self.ego_state)
                setattr(wrong.vehicle_state,field,value)
                with patch.object(live,"_inventory",return_value={"Ego":wrong}), \
                        patch.object(startup,"common_stopped_readback",autospec=True,return_value=(self.ego_state,self.observed)), \
                        patch.object(live,"_resolve_raw") as resolve:
                    with self.assertRaisesRegex(RuntimeError,"brake hold"):
                        live._wait_spawn_readback(self.grpc,NS(),{"Ego"},"WoodBox",{}, {},"Ego",self.observed)
                resolve.assert_not_called()
        moving=copy.deepcopy(self.ego_state)
        moving.global_velocity.x=.051
        with patch.object(live,"_inventory",return_value={"Ego":moving}), \
                patch.object(startup,"common_stopped_readback",autospec=True) as read, \
                patch.object(live,"_resolve_raw") as resolve:
            with self.assertRaisesRegex(RuntimeError,"moving"):
                live._wait_spawn_readback(self.grpc,NS(),{"Ego"},"WoodBox",{}, {},"Ego",self.observed)
        read.assert_not_called()
        resolve.assert_not_called()


class CommonSafetyParkingTests(unittest.TestCase):
    def setUp(self):
        self.stub=NS()
        self.ego=NS()
        self.state=state("Ego",0.,0.,width=2.,length=4.5)
        self.state.vehicle_state=NS(gear_mode=4,brake=1.,throttle=0.)
        self.observed=dict(live._observation(self.state),control_mode=3)

    def test_shared_filter_excludes_only_exact_live_owned_brake_holder(self):
        from ..scenarios.full_loop_parts.tunnel_static_parts import connection
        owned="/tunnel_brake_hold_owned_pid"
        publishers=[owned,owned+"_other", "/foreign_controller"]
        holder=object.__new__(startup._RosBrakeHold)
        holder.owner_pid=startup.os.getpid()
        holder.node_name=owned
        holder.active=True
        holder.thread=NS(is_alive=lambda:True)
        with patch.object(connection,"_controller_publishers",return_value=publishers), \
                patch.object(startup,"_ROS_BRAKE_HOLD",holder):
            self.assertEqual(safety.competing_controller_publishers(),publishers[1:])
            holder.active=False
            self.assertEqual(safety.competing_controller_publishers(),publishers)
            holder.active=True
            holder.owner_pid+=1
            self.assertEqual(safety.competing_controller_publishers(),publishers)
            holder.owner_pid=startup.os.getpid()
            holder.thread=NS(is_alive=lambda:False)
            self.assertEqual(safety.competing_controller_publishers(),publishers)
        impostor=NS(owner_pid=startup.os.getpid(),node_name=owned,active=True,thread=NS(is_alive=lambda:True))
        with patch.object(connection,"_controller_publishers",return_value=publishers), \
                patch.object(startup,"_ROS_BRAKE_HOLD",impostor):
            self.assertEqual(safety.competing_controller_publishers(),publishers)

    def test_safety_adapter_verifies_common_hold_and_reports_actual_auto_mode(self):
        with patch.object(startup,"hold_common_brake",autospec=True) as hold, \
                patch.object(safety,"_controller_publishers",return_value=[]), \
                patch.object(safety,"competing_controller_publishers",return_value=[]), \
                patch.object(startup,"common_stopped_readback",autospec=True,return_value=(self.state,self.observed)) as read:
            result=safety.park(self.stub,self.ego)
        adapter=hold.call_args.args[0]
        self.assertIs(adapter.get_ego(),self.ego)
        self.assertIs(adapter.world._sim_adapter._actor_stub,self.stub)
        read.assert_called_once_with(adapter)
        self.assertTrue(result["parked"])
        self.assertTrue(result["brake_hold_verified"])
        self.assertTrue(result["control_publishers_absent"])
        self.assertFalse(result["external_input_disabled"])
        self.assertEqual((result["control_mode"],result["gear_mode"]),(3,4))
        self.assertEqual((result["brake"],result["throttle"]),(1.,0.))
        self.assertEqual(result["control_scope"],safety.CONTROL_SCOPE)

    def test_publisher_appearing_after_hold_cannot_be_reported_absent(self):
        with patch.object(startup,"hold_common_brake",autospec=True), \
                patch.object(startup,"common_stopped_readback",autospec=True,return_value=(self.state,self.observed)), \
                patch.object(safety,"_controller_publishers",return_value=["/foreign_controller"]), \
                patch.object(safety,"competing_controller_publishers",return_value=["/foreign_controller"]):
            with self.assertRaisesRegex(RuntimeError,"publisher appeared"):
                safety.park(self.stub,self.ego)

    def test_exact_owned_brake_publisher_is_disclosed_without_becoming_a_competitor(self):
        owned="/tunnel_static_brake_hold_owned_pid"
        with patch.object(startup,"hold_common_brake",autospec=True), \
                patch.object(startup,"common_stopped_readback",autospec=True,return_value=(self.state,self.observed)), \
                patch.object(safety,"_controller_publishers",return_value=[owned]), \
                patch.object(safety,"competing_controller_publishers",return_value=[]):
            result=safety.park(self.stub,self.ego)
        self.assertTrue(result["control_publishers_absent"])
        self.assertFalse(result["external_input_disabled"])
        self.assertEqual(result["observed_control_publishers"],[owned])
        self.assertEqual(result["control_scope"],safety.CONTROL_SCOPE)

    def test_rejected_or_contended_hold_cannot_become_verified_parking(self):
        with patch.object(startup,"hold_common_brake",autospec=True,side_effect=RuntimeError("Competing controller")), \
                patch.object(startup,"common_stopped_readback",autospec=True) as read:
            with self.assertRaisesRegex(RuntimeError,"Competing controller"):
                safety.park(self.stub,self.ego)
        read.assert_not_called()

    def test_unverified_safety_readback_retains_three_second_deadline(self):
        clock={"time":0.}
        def sleep(seconds):
            clock["time"]+=seconds
        with patch.object(startup,"hold_common_brake",autospec=True) as hold, \
                patch.object(startup,"common_stopped_readback",autospec=True,side_effect=RuntimeError("Brake not applied")) as read, \
                patch.object(safety.time,"monotonic",side_effect=lambda:clock["time"]), \
                patch.object(safety.time,"sleep",side_effect=sleep):
            with self.assertRaisesRegex(RuntimeError,"within three seconds"):
                safety.park(self.stub,self.ego)
        hold.assert_called_once()
        self.assertGreater(read.call_count,1)
        self.assertLessEqual(clock["time"],3.1)


# Native owned cleanup must prove scope, ACKs and eventual global absence.

class NativeCleanupTests(preview_fixtures.LivePreviewTest):
    # Use the isolated Simulator fixture, without duplicating its preview tests.
    def test_native_delete_uses_verified_raw_id_and_never_sdk_destroy(self):
        with patch.object(Actor,"destroy",side_effect=AssertionError("SDK delete must not be used")):
            result=self.run_preview()
        self.assert_clean(result)
        cleanup=result["cleanup"]
        self.assertEqual(cleanup["actors"][0]["raw_object_info"]["id"],"101")
        self.assertTrue(cleanup["actors"][0]["destroy_accepted"])
        self.assertEqual(cleanup["namespace_cleanup"]["client_key"],"preview-client")
        self.assertTrue(cleanup["namespace_cleanup"]["accepted"])
        self.assertEqual(cleanup["namespace_cleanup"]["settle_sec"],1.5)

    def test_blank_global_lookup_key_is_preserved_separately_from_handle_ownership_pin(self):
        original=self.sim.world.spawn_obstacle
        def blank_owner(**kwargs):
            handle=original(**kwargs)
            handle._state().actor_info.client_key=""
            return handle
        self.sim.world.spawn_obstacle=blank_owner
        result=self.run_preview()
        self.assert_clean(result)
        row=result["cleanup"]["actors"][0]
        self.assertEqual(row["returned_client_key"],"")
        self.assertEqual(row["raw_object_info"]["client_key"],"")
        self.assertEqual(row["source_client_key"],"preview-client")
        self.assertIn("ownership namespace is separate",row["client_key_basis"])

    def test_native_delete_rejection_is_retained_and_can_be_recovered_by_verified_namespace(self):
        self.sim.stub.DestroyActor=lambda info,timeout:NS(status=0,description="individual rejection")
        result=self.run_preview()
        self.assert_clean(result)
        cleanup=result["cleanup"]
        self.assertEqual(cleanup["actors"][0]["native_status"],0)
        self.assertEqual(cleanup["actors"][0]["native_description"],"individual rejection")
        self.assertFalse(cleanup["actors"][0]["destroy_accepted"])
        self.assertTrue(cleanup["individual_delete_errors"])
        self.assertTrue(cleanup["individual_errors_recovered_by_verified_namespace"])

    def test_native_delete_timeout_is_not_erased_when_namespace_recovers(self):
        self.sim.stub.DestroyActor=Mock(side_effect=TimeoutError("individual RPC timeout"))
        result=self.run_preview()
        self.assert_clean(result)
        self.assertIn("individual RPC timeout",result["cleanup"]["actors"][0]["destroy_error"])
        self.assertTrue(result["cleanup"]["individual_errors_recovered_by_verified_namespace"])

    def test_namespace_rejection_cannot_pass_even_if_individual_delete_removed_actor(self):
        self.sim.stub.DestroyAllActors=lambda request,timeout:NS(status=0,description="namespace rejection")
        result=self.run_preview()
        self.assertEqual(result["status"],"failed")
        self.assertFalse(result["cleanup"]["verified"])
        self.assertFalse(result["cleanup"]["namespace_cleanup"]["accepted"])
        self.assertIn("namespace rejection",result["failure_reason"])

    def test_delayed_namespace_deletion_is_verified_by_bounded_requeries(self):
        reads={"count":0,"pending":False}
        self.sim.stub.DestroyActor=lambda info,timeout:NS(status=1,description="delete queued")
        def queued(request,timeout):
            self.assertEqual(request.value,"preview-client")
            reads["pending"]=True
            return NS(status=1,description="namespace queued")
        def inventory(sim):
            if reads["pending"]:
                reads["count"]+=1
                if reads["count"]==3:
                    sim.states.pop("101")
        self.sim.stub.DestroyAllActors=queued
        self.sim.inventory_hook=inventory
        result=self.run_preview()
        self.assert_clean(result)
        self.assertEqual(result["cleanup"]["global_readback"]["attempts"],3)
        self.assertTrue(result["cleanup"]["global_readback"]["verified"])

    def test_actor_disappearing_near_three_second_deadline_can_still_be_verified(self):
        clock={"time":0.,"first_read":None,"namespace_called":False}
        actor=state("101",25.,0.)
        actor.actor_info.client_key="preview-client"
        self.sim.states["101"]=actor
        self.sim.aliases["owned_request"]="101"
        self.sim.ignore_destroy=True
        original_namespace=self.sim.stub.DestroyAllActors
        def namespace(request,timeout):
            clock["namespace_called"]=True
            return original_namespace(request,timeout)
        self.sim.stub.DestroyAllActors=namespace
        entry={"handle":Actor(self.sim,"owned_request"),"request_id":"owned_request","raw_id":"101",
               "source_client_key":"preview-client","source_handle_id":"owned_request",
               "actor_kind":"obstacle","requested_model":"WoodBox",
               "raw_object_info":copy.deepcopy(actor.actor_info)}
        def inventory(sim):
            if not clock["namespace_called"]:
                return
            if clock["first_read"] is None:
                clock["first_read"]=clock["time"]
            if clock["time"]-clock["first_read"]>=2.85:
                sim.states.pop("101",None)
        self.sim.inventory_hook=inventory
        report={"inventory_checks":0,"cleanup":{"verified":False,"actors":[]}}
        baseline={"Ego":copy.deepcopy(self.sim.states["Ego"])}
        def sleep(seconds):clock["time"]+=seconds
        with patch.object(live.time,"monotonic",side_effect=lambda:clock["time"]), \
                patch.object(live.time,"sleep",side_effect=sleep) as sleeping:
            live.cleanup_owned_actors(self.sim.grpc,[entry],baseline,report)
        self.assertTrue(report["cleanup"]["verified"])
        self.assertGreaterEqual(report["cleanup"]["global_readback"]["elapsed_sec"],2.85)
        self.assertLess(report["cleanup"]["global_readback"]["elapsed_sec"],3.)
        self.assertGreater(report["cleanup"]["global_readback"]["attempts"],20)
        self.assertEqual(sleeping.call_args_list[0].args,(1.5,))

    def test_explicit_same_namespace_baseline_actor_blocks_every_delete(self):
        foreign=state("foreign",500.,100.)
        foreign.actor_info.client_key="preview-client"
        self.sim.states["foreign"]=foreign
        destroy=Mock(wraps=self.sim.stub.DestroyActor)
        namespace=Mock(wraps=self.sim.stub.DestroyAllActors)
        self.sim.stub.DestroyActor,self.sim.stub.DestroyAllActors=destroy,namespace
        result=self.run_preview()
        self.assertFalse(result["cleanup"]["verified"])
        self.assertIn("preserved baseline actor",result["failure_reason"])
        destroy.assert_not_called()
        namespace.assert_not_called()
        self.assertIs(self.sim.states["foreign"],foreign)

    def test_namespace_cleanup_cannot_hide_disappearing_foreign_actor(self):
        foreign=state("foreign",500.,100.)
        self.sim.states["foreign"]=foreign
        original=self.sim.stub.DestroyAllActors
        def unsafe(request,timeout):
            response=original(request,timeout)
            self.sim.states.pop("foreign")
            return response
        self.sim.stub.DestroyAllActors=unsafe
        result=self.run_preview()
        self.assertFalse(result["cleanup"]["verified"])
        self.assertIn("baseline actors disappeared",result["failure_reason"])

    def test_alias_resolving_to_baseline_actor_refuses_bulk_even_before_raw_id_assignment(self):
        foreign=state("foreign",500.,100.)
        self.sim.states["foreign"]=foreign
        original=self.sim.stub.GetActorState
        self.sim.stub.GetActorState=lambda info,timeout:foreign if info.id.value.startswith("tunnel_preview_") else original(info,timeout)
        destroy=Mock(wraps=self.sim.stub.DestroyActor)
        namespace=Mock(wraps=self.sim.stub.DestroyAllActors)
        self.sim.stub.DestroyActor,self.sim.stub.DestroyAllActors=destroy,namespace
        result=self.run_preview()
        self.assertFalse(result["cleanup"]["verified"])
        self.assertIn("namespace cleanup refused",result["failure_reason"])
        destroy.assert_not_called()
        namespace.assert_not_called()
        self.assertIs(self.sim.states["foreign"],foreign)

    def test_exact_owned_readback_transport_failure_never_becomes_absence_or_bulk_approval(self):
        original=self.sim.stub.GetActorState
        destroy=Mock(wraps=self.sim.stub.DestroyActor)
        namespace=Mock(wraps=self.sim.stub.DestroyAllActors)
        self.sim.stub.DestroyActor,self.sim.stub.DestroyAllActors=destroy,namespace
        def drive(report,owned):
            def failed(info,timeout):
                if info.id.value!="Ego":raise TimeoutError("owned readback unavailable")
                return original(info,timeout)
            self.sim.stub.GetActorState=failed
            return {"status":"passed",**{key:True for key in ("completed","collision_free","returned_to_original_lane","fixture_stationary")}}
        screen=lambda *args,**kwargs:{"status":"nominal_path_found","path_found":True,"geometry_checks":{"passed":True}}
        result=self.run_preview(screen_fn=screen,drive_fn=drive)
        self.assertFalse(result["cleanup"]["verified"])
        self.assertIn("owned readback unavailable",result["failure_reason"])
        destroy.assert_not_called()
        namespace.assert_not_called()

    def test_borrowed_runner_namespace_cannot_replace_the_original_fixture_owner(self):
        def drive(report,owned):
            self.sim.grpc.client_key="tunnel_static_trial_runner"
            return {"status":"passed",**{key:True for key in ("completed","collision_free","returned_to_original_lane","fixture_stationary")}}
        destroy=Mock(wraps=self.sim.stub.DestroyActor)
        namespace=Mock(wraps=self.sim.stub.DestroyAllActors)
        self.sim.stub.DestroyActor,self.sim.stub.DestroyAllActors=destroy,namespace
        screen=lambda *args,**kwargs:{"status":"nominal_path_found","path_found":True,"geometry_checks":{"passed":True}}
        result=self.run_preview(screen_fn=screen,drive_fn=drive)
        self.assertFalse(result["cleanup"]["verified"])
        self.assertIn("original nonempty fixture-owner",result["failure_reason"])
        destroy.assert_not_called()
        namespace.assert_not_called()


# The inherited fixture methods are useful; inherited tests belong to their
# original suite, not this focused native-cleanup suite.
for name in list(vars(preview_fixtures.LivePreviewTest)):
    if name.startswith("test_") and name not in vars(NativeCleanupTests):
        setattr(NativeCleanupTests,name,None)


# Recovery must preserve foreign actors even when owner reads are global.

class BatchSafetyTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.calls = []
        self.stub = NS(DestroyAllActors=lambda request, timeout: self.destroy(request, timeout))
        self.grpc = NS(world=NS(_sim_adapter=NS(_actor_stub=self.stub)))
        self.proto = patch.dict(sys.modules, {
            "proto.morai.common.type_pb2": NS(StringValue=lambda **kwargs: NS(**kwargs)),
            "proto.morai.common.enum_pb2": NS(STATUS_CODE_SUCCESS=1)})
        self.proto.start()
        self.addCleanup(self.proto.stop)

    def destroy(self, request, timeout):
        self.assertIn(request.value, safety.OWNED_KEYS)
        self.assertLessEqual(timeout, 1.)
        self.calls.append(request.value)
        return NS(status=1)

    def recover(self, inventories, publishers=None):
        with patch.object(safety, "inventory", side_effect=inventories), \
                patch.object(startup, "block_common_input"), \
                patch.object(safety, "park", return_value={"parked": True}), \
                patch.object(safety, "stop_owned_experts", return_value=[]), \
                patch.object(safety, "_controller_publishers", return_value=[]), \
                patch.object(safety, "competing_controller_publishers", side_effect=publishers or [[], []]):
            return safety.recover(self.grpc, None, self.root)

    def test_first_recovery_records_global_foreign_baseline_without_owner_filter(self):
        actors = {"Ego": None, "2": None, "4": None, "5": None}
        result = self.recover([actors, actors])
        self.assertEqual(result["status"], "recovered")
        self.assertTrue(result["read_filter_is_global"])
        self.assertEqual(result["preserved_foreign_actor_ids"], ["2", "4", "5"])
        self.assertEqual(self.calls, list(safety.OWNED_KEYS))

    def test_owned_fixture_absence_is_verified_against_frozen_foreign_inventory(self):
        safety.write_json(self.root/"foreign_actor_inventory.json", {"actor_ids": ["2", "4"]})
        result = self.recover([{"Ego": None, "2": None, "4": None, "6": None}, {"Ego": None, "2": None, "4": None}])
        self.assertEqual(result["removed_actor_ids"], ["6"])
        self.assertEqual(result["preserved_foreign_actor_ids"], ["2", "4"])

    def test_retained_owned_or_new_unaccounted_actor_blocks_next_trial(self):
        safety.write_json(self.root/"foreign_actor_inventory.json", {"actor_ids": ["2"]})
        with self.assertRaisesRegex(RuntimeError, "Unaccounted actors remain"):
            self.recover([{"Ego": None, "2": None, "6": None}, {"Ego": None, "2": None, "6": None}])

    def test_disappearing_foreign_actor_is_never_reported_as_success(self):
        safety.write_json(self.root/"foreign_actor_inventory.json", {"actor_ids": ["2", "4"]})
        with self.assertRaisesRegex(RuntimeError, "Foreign actor disappeared"):
            self.recover([{"Ego": None, "2": None, "4": None}, {"Ego": None, "2": None}])

    def test_unowned_controller_blocks_namespace_cleanup(self):
        with self.assertRaisesRegex(RuntimeError, "competing or unowned"):
            self.recover([], publishers=[["/other_team_controller"]])
        self.assertFalse(self.calls)

    def test_recovery_blocks_input_before_owned_stop_and_checks_publishers_before_auto_hold(self):
        order=[]
        with patch.object(startup,"block_common_input",side_effect=lambda _:order.append("block")), \
                patch.object(safety,"stop_owned_experts",side_effect=lambda _:order.append("stop-owned") or [101]), \
                patch.object(safety,"_controller_publishers",return_value=[]), \
                patch.object(safety,"competing_controller_publishers",side_effect=lambda:order.append("publishers") or []), \
                patch.object(safety,"park",side_effect=lambda *args:order.append("hold") or {"parked":True}), \
                patch.object(safety,"inventory",return_value={"Ego":None}):
            result=safety.recover(self.grpc,None,self.root)
        self.assertEqual(order[:4],["block","stop-owned","publishers","hold"])
        self.assertEqual(result["stopped_expert_pids"],[101])
        self.assertEqual(self.calls,list(safety.OWNED_KEYS))

    def test_foreign_publisher_after_owned_stop_prevents_brake_release_and_actor_changes(self):
        with patch.object(startup,"block_common_input") as block, \
                patch.object(safety,"stop_owned_experts",return_value=[]), \
                patch.object(safety,"competing_controller_publishers",return_value=["/foreign_controller"]), \
                patch.object(safety,"park") as hold:
            with self.assertRaisesRegex(RuntimeError,"competing or unowned"):
                safety.recover(self.grpc,None,self.root)
        block.assert_called_once()
        hold.assert_not_called()
        self.assertFalse(self.calls)

    def test_owned_brake_publisher_is_recorded_while_recovery_rejects_only_competitors(self):
        owned="/tunnel_static_brake_hold_owned_pid"
        with patch.object(startup,"block_common_input"), \
                patch.object(safety,"stop_owned_experts",return_value=[]), \
                patch.object(safety,"competing_controller_publishers",return_value=[]), \
                patch.object(safety,"_controller_publishers",return_value=[owned]), \
                patch.object(safety,"park",return_value={"parked":True}), \
                patch.object(safety,"inventory",return_value={"Ego":None}):
            result=safety.recover(self.grpc,None,self.root)
        self.assertEqual(result["status"],"recovered")
        self.assertEqual(result["controller_publishers"],[])
        self.assertEqual(result["observed_control_publishers"],[owned])
        self.assertEqual(result["control_scope"],safety.CONTROL_SCOPE)

    def test_scene_noop_accepts_verified_owned_brake_publisher(self):
        with patch.dict(sys.modules,{"proto.morai.actor.actor_get_pb2":NS(ActorState=NS())}), \
                patch.object(safety,"park",return_value={"parked":True,"brake_hold_verified":True}), \
                patch.object(safety,"competing_controller_publishers",return_value=[]), \
                patch.object(safety,"_controller_publishers",return_value=["/owned_brake_hold"]), \
                patch.object(safety,"inventory",return_value={"Ego":None}):
            result=safety.relocate(self.grpc,None,self.root,[],"restore")
        self.assertEqual(result["status"],"nothing_to_restore")
        self.assertTrue(result["parking"]["brake_hold_verified"])

    def test_foreign_scene_controller_blocks_inventory_and_actor_mutation(self):
        with patch.dict(sys.modules,{"proto.morai.actor.actor_get_pb2":NS(ActorState=NS())}), \
                patch.object(safety,"park",return_value={"parked":True}), \
                patch.object(safety,"competing_controller_publishers",return_value=["/foreign"]), \
                patch.object(safety,"inventory") as inventory, \
                patch.object(safety,"set_transform") as transform:
            with self.assertRaisesRegex(RuntimeError,"Stop the Ego controller"):
                safety.relocate(self.grpc,None,self.root,[],"restore")
        inventory.assert_not_called()
        transform.assert_not_called()

    def test_helper_only_selects_expert_inside_private_batch_with_guarded_contract(self):
        proc = self.root/"proc"
        proc.mkdir()
        batch = self.root/"batch"
        batch.mkdir()
        guarded = batch/"config.json"
        safety.write_json(guarded, {"planning": {"jamming_trial_contract": {"mode": "full_loop_tunnel_static"}}})
        outside = self.root/"unrelated.json"
        safety.write_json(outside, {"planning": {"jamming_trial_contract": {"mode": "full_loop_tunnel_static"}}})
        entrypoint = Path(__file__).resolve().parents[1]/"adapters/expert_process.py"
        for pid, entry, configuration in (("101", entrypoint, guarded), ("102", entrypoint, outside), ("103", "/tmp/unrelated.py", guarded)):
            process = proc/pid
            process.mkdir()
            (process/"cmdline").write_bytes(b"python3\0"+str(entry).encode()+b"\0")
            (process/"environ").write_bytes(b"GT_BEV_CONFIG_PATH="+str(configuration).encode()+b"\0")
            (process/"stat").write_text(pid+" (python) "+" ".join(["S", "1", pid]+["0"]*16+["100"]))
        self.assertEqual(safety.owned_experts(batch, proc), [101])

    def test_pid_reuse_after_term_does_not_signal_an_unrelated_replacement_group(self):
        original = {"pid": 101, "process_group": 101, "starttime_ticks": 100, "configuration": "owned"}
        replacement = dict(original, starttime_ticks=200)
        with patch.object(safety, "owned_experts", return_value=[101]), \
                patch.object(safety, "expert_identity", side_effect=[original, original, replacement, replacement, replacement]), \
                patch.object(safety.os, "killpg") as kill, \
                patch("scenario_runner.adapters.expert_controller.ExpertController._clear_owned_ros_registrations") as cleanup:
            self.assertEqual(safety.stop_owned_experts(self.root), [101])
        kill.assert_called_once_with(101, safety.signal.SIGTERM)
        cleanup.assert_called_once_with(101)

    def test_actor_selection_rejects_ego_duplicates_and_empty_components(self):
        for value in ("Ego", "2,2", "2,,4", "-2"):
            with self.assertRaises(ValueError):
                safety.actor_ids(value)
        self.assertEqual(safety.actor_ids("2,4,5"), ["2", "4", "5"])

    def test_failed_restore_still_restores_remaining_foreign_actors(self):
        originals, snapshot = {}, {"actor_ids": ["2", "4", "5"], "actors": []}
        for identity in snapshot["actor_ids"]:
            payload = identity.encode()
            (self.root/(identity+".pb")).write_bytes(payload)
            originals[payload] = NS(actor_info=NS(identity=identity), transform=NS(location=NS(x=1., y=2., z=3.)))
            snapshot["actors"].append(dict(id=identity, model="model", state_file=identity+".pb", sha256=hashlib.sha256(payload).hexdigest()))
        safety.write_json(self.root/"scene_snapshot.json", snapshot)
        self.stub.GetOptionName = lambda info, timeout: NS(value="model")
        restored = []

        def transform(stub, info, target, *, restoring):
            restored.append(info.identity)
            if info.identity == "2":
                raise RuntimeError("injected unavailable readback")
            return {"position_error_m": 0.}

        with patch.dict(sys.modules, {"proto.morai.actor.actor_get_pb2": NS(ActorState=NS(FromString=lambda payload: originals[payload]))}), \
                patch.object(safety, "park", return_value={"parked": True}), \
                patch.object(safety, "inventory", return_value={i: originals[i.encode()] for i in snapshot["actor_ids"]}), \
                patch.object(safety, "_controller_publishers", return_value=[]), \
                patch.object(safety, "competing_controller_publishers", return_value=[]), \
                patch.object(safety, "set_transform", side_effect=transform):
            with self.assertRaisesRegex(RuntimeError, "could not be restored"):
                safety.relocate(self.grpc, None, self.root, snapshot["actor_ids"], "restore")
        self.assertEqual(restored, ["2", "4", "5"])
        result = json.loads((self.root/"restore.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual([row["id"] for row in result["actors"]], ["4", "5"])


if __name__ == "__main__":
    unittest.main()
