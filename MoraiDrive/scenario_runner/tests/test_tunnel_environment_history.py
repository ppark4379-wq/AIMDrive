"""Pre-drive retries do not consume weather history for the actual run."""

import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from ..scenarios.full_loop_parts import route_profile
from ..scenarios.full_loop_parts.tunnel_static_parts import execution, runtime
from . import test_tunnel_static_search as search_fixtures


def choose_environment(path, seed):
    context=SimpleNamespace(cfg={"profile_state_path":str(path),"profiles":[
        {"name":"sun","weather":"SUNNY"},{"name":"fog","weather":"FOGGY"}],
        "scenario_times":[11,13,15]},random=random.Random(seed))
    context.profile_state_path=lambda:route_profile.profile_state_path(context)
    context.read_profile_state=lambda:route_profile.read_profile_state(context)
    context.validate_profiles=lambda profiles:route_profile.validate_profiles(context,profiles)
    with redirect_stdout(io.StringIO()):
        index,profile,previous=route_profile.select_profile(context)
        context.previous_profile_state=previous
        hour=route_profile.select_scenario_hour(context)
    return {"run_seed":seed,"profile_index":index,"weather":profile["weather"],"time_hour":hour}


class EnvironmentHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.profile=self.root/"full_loop_tunnel_static_profile.json"

    def previous_sunny(self):
        # Preserve the exact incoming bytes, not only their interpreted fields.
        original=b'{ "profile_index": 0, "weather": "SUNNY", "time_hour": 13, "run_seed": 99 }\n'
        self.profile.write_bytes(original)
        return original

    def test_owned_failed_preflight_restores_incoming_weather_and_time_exactly(self):
        incoming=self.previous_sunny()
        attempt=choose_environment(self.profile,100)
        self.assertEqual(attempt["weather"],"FOGGY")
        self.assertNotEqual(attempt["time_hour"],13)
        runtime.atomic_json(self.profile,attempt)
        execution.restore_preflight_environment_history(self.profile,incoming,100)
        self.assertEqual(self.profile.read_bytes(),incoming)
        self.assertEqual(choose_environment(self.profile,100),attempt)

    def test_first_run_retry_removes_only_its_own_created_profile(self):
        unrelated=self.root/"full_loop_static_profile.json"
        unrelated.write_bytes(b'unchanged unrelated mode history\n')
        runtime.atomic_json(self.profile,choose_environment(self.profile,100))
        execution.restore_preflight_environment_history(self.profile,None,100)
        self.assertFalse(self.profile.exists())
        self.assertEqual(unrelated.read_bytes(),b'unchanged unrelated mode history\n')
        self.assertEqual(sorted(path.name for path in self.root.iterdir()),[unrelated.name])

    def test_external_seed_change_is_rejected_without_overwrite(self):
        incoming=self.previous_sunny()
        external={"profile_index":1,"weather":"FOGGY","time_hour":15,"run_seed":201}
        runtime.atomic_json(self.profile,external)
        before=self.profile.read_bytes()
        with self.assertRaisesRegex(RuntimeError,"outside this owned preflight"):
            execution.restore_preflight_environment_history(self.profile,incoming,100)
        self.assertEqual(self.profile.read_bytes(),before)

    def test_external_new_first_run_history_is_not_deleted(self):
        runtime.atomic_json(self.profile,{"run_seed":201,"weather":"SUNNY","time_hour":13})
        before=self.profile.read_bytes()
        with self.assertRaisesRegex(RuntimeError,"outside this owned preflight"):
            execution.restore_preflight_environment_history(self.profile,None,100)
        self.assertEqual(self.profile.read_bytes(),before)

    def test_no_change_is_a_read_only_no_op(self):
        incoming=self.previous_sunny()
        execution.restore_preflight_environment_history(self.profile,incoming,100)
        self.assertEqual(self.profile.read_bytes(),incoming)
        self.assertFalse(self.profile.with_suffix(".json.preflight_restore.tmp").exists())

    def test_ordering_retry_recovers_then_restores_so_actual_run_remains_foggy(self):
        fixture=search_fixtures.SearchTests(methodName="test_structured_ordering_retry_is_bounded_and_failed_actual_drive_counts")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        incoming=self.previous_sunny()
        fixture.args.run_seed=100
        fixture.args.profile_state_path=self.profile
        campaign,row,_events=fixture.initialize_row()
        events=[]
        environments=[]

        def process(command,*args):
            self.assertEqual(command[command.index("--run-seed")+1],"100")
            self.assertEqual(command[command.index("--profile-state-path")+1],str(self.profile))
            self.assertEqual(self.profile.read_bytes(),incoming)
            environment=choose_environment(self.profile,100)
            environments.append(environment)
            events.append("attempt_%s_%s" % (len(environments),environment["weather"]))
            runtime.atomic_json(self.profile,environment)
            return {"returncode":1 if len(environments)==1 else 0,"timed_out":False,"forced_kill":False}

        def recover(action,directory=None):
            self.assertEqual(action,"recover")
            self.assertEqual(json.loads(self.profile.read_text())["weather"],"FOGGY")
            events.append("recover")
            return {"status":"recovered"}

        success=search_fixtures.outcome("actual_fail")
        success.update(strict_pass=True,status="passed",failure_reason="")
        campaign.process=process
        campaign.helper=recover
        with patch.object(execution,"classify_trial",side_effect=[search_fixtures.outcome("ordering"),success]),patch.object(
                execution,"read_trace",return_value={"valid":True,"evidence_kind":"completed_actual_trace"}):
            campaign.trial(row)
        self.assertEqual(events,["attempt_1_FOGGY","recover","attempt_2_FOGGY","recover"])
        self.assertEqual(environments[0],environments[1])
        self.assertEqual(campaign.state["actual_drives"],1)
        self.assertEqual(campaign.state["actual_passes"],1)
        self.assertEqual(campaign.state["preflight_rejects"],1)
        self.assertFalse(campaign.state["attempts"][0]["outcome"]["actual_driving"])
        self.assertEqual(campaign.state["attempts"][0]["disposition"],"retry_ordering")
        self.assertEqual(json.loads(self.profile.read_text()),environments[1])


if __name__ == "__main__":
    unittest.main()
