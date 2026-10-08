"""Recovery must preserve foreign actors even when owner reads are global."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from ..tools import tunnel_static_batch_safety as safety
from ..scenarios.segment_modes import tunnel_static_startup as startup


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
