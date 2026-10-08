"""Native owned cleanup must prove scope, ACKs and eventual global absence."""
import copy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from ..scenarios.full_loop_parts.tunnel_static_parts import live
from .test_tunnel_static_live import Actor, state
from . import test_tunnel_static_live as preview_fixtures


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


if __name__=="__main__":unittest.main()
