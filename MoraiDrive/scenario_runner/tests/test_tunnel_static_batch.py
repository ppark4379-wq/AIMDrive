"""Batch accounting, fixed sources, recovery barriers and process ownership."""

import copy
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

from ..tools import run_tunnel_static_batch as batch


def trial_files(directory, *, passed=True, moved=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    proof = {"status": "passed" if passed else "failed", "completed": passed,
             "collision_free": True, "returned_to_original_lane": passed, "fixture_stationary": True,
             "completion_conditions": {"whole_body_in_original_lane": passed}, "ego_stopped_and_parked": True,
             "trajectory_checks": 10, "failure_reason": "" if passed else "bounded test failure"}
    batch.atomic_json(directory/"drive_trial.json", proof)
    batch.atomic_json(directory/"planner_diagnostics.json", {"calls": 20, "validated_path_seen": passed})
    batch.atomic_json(directory/"manifest.json", {"status": "passed" if passed else "failed",
        "expert_verified": passed, "live_avoidance_verified": passed,
        "fixture_result": {"status": "valid_driving_trial" if passed else "failed", "cleanup": {"verified": True}}})
    values = "0,0,0,0,0\n1,0.01,0,0,.051\n2,1,0,0,10\n" if moved else "0,0,0,0,0\n1,.001,0,0,.051\n2,.002,0,0,.02\n"
    (directory/"ego_trace.csv").write_text("elapsed_s,x,y,yaw_deg,speed\n"+values)


class FakeBatch(batch.Batch):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.events = []
        self.reject_first = False
        self.fail_recovery = False

    def helper(self, action, *, directory=None):
        self.events.append(action)
        if action == "recover" and directory and Path(directory).name == "recovery" and self.fail_recovery:
            raise RuntimeError("recovery failed")
        self.state["recovery_checks"] += int(action == "recover")
        self.save()
        return {"returncode": 0}

    def candidate(self, case, attempt):
        return {"seed": 42+attempt, "candidate_index": attempt}

    def process(self, command, log_path, timeout, **kwargs):
        self.events.append("trial")
        target = Path(command[command.index("--output")+1])
        if self.reject_first and len(self.state["attempts"]) == 1:
            target.mkdir(parents=True, exist_ok=True)
            batch.atomic_json(target/"manifest.json", {"status": "failed", "failure_reason": "screening rejection"})
        else:
            trial_files(target)
        return {"returncode": 0, "timed_out": False, "forced_kill": False}


class BatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.helper = self.root/"helper.sh"
        self.helper.write_text("#!/bin/sh\nexit 0\n")
        self.golden = {}
        for kind in batch.KINDS:
            self.golden[kind] = self.root/(kind+".json")
            self.golden[kind].write_text('{"seed":42}')
        self.snapshot = {"source": "fixed"}

    def args(self, output=None, **kwargs):
        argv = ["--output", str(output or self.root/"batch"), "--recovery-helper", str(self.helper), "--actual-runs", "3"]
        for kind, path in self.golden.items():
            argv.extend(("--"+kind+"-candidate", str(path)))
        result = batch.build_parser().parse_args(argv)
        for key, value in kwargs.items():
            setattr(result, key, value)
        return result

    def make(self, args=None):
        return FakeBatch(args or self.args(), snapshot_fn=lambda *args: copy.deepcopy(self.snapshot),
                         global_lock_path=self.root/"global.lock")

    def test_default24_plan_has_eight_of_each_and_three_known_starting_cases(self):
        plan = batch.make_plan(24)
        self.assertEqual({kind: sum(row["kind"] == kind for row in plan) for kind in batch.KINDS},
                         {"woodbox": 8, "sedan": 8, "mixed": 8})
        self.assertEqual(sum(row["golden"] for row in plan), 3)

    def test_existing_output_with_foreign_baseline_and_no_manifest_is_first_run(self):
        output = self.root/"batch"
        output.mkdir()
        (output/"foreign_actor_inventory.json").write_text('{"actor_ids":["Ego","2"]}')
        worker = self.make()
        result = worker.run()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["actual_drives"], 3)
        self.assertEqual(worker.events, ["recover", "trial", "recover", "trial", "recover", "trial", "recover", "recover"])

    def test_screen_rejection_does_not_count_and_is_recovered_before_next_trial(self):
        worker = self.make()
        worker.reject_first = True
        result = worker.run()
        self.assertEqual(result["actual_drives"], 3)
        self.assertEqual(result["preflight_rejects"], 1)
        self.assertEqual(len(result["attempts"]), 4)
        self.assertTrue(all(row["recovery_verified"] for row in result["attempts"]))

    def test_failed_recovery_blocks_next_trial_and_never_commits_pending_case(self):
        worker = self.make()
        worker.fail_recovery = True
        with self.assertRaisesRegex(RuntimeError, "recovery failed"):
            worker.run()
        self.assertEqual(worker.events.count("trial"), 1)
        self.assertEqual(worker.state["actual_drives"], 0)
        self.assertFalse(worker.state["attempts"][0]["committed"])

    def test_source_mismatch_resume_never_executes_recovery_or_scene_actions(self):
        original = self.make()
        original.acquire()
        original.release()
        self.snapshot = {"source": "changed"}
        resumed = self.make(self.args(resume=True))
        with self.assertRaisesRegex(RuntimeError, "Frozen batch sources"):
            resumed.run()
        self.assertEqual(resumed.events, [])

    def test_global_lock_prevents_a_second_output_from_controlling_same_simulator(self):
        first = self.make()
        first.acquire()
        self.addCleanup(first.release)
        second = self.make(self.args(output=self.root/"other"))
        with self.assertRaisesRegex(RuntimeError, "global MORAI"):
            second.run()
        self.assertEqual(second.events, [])

    def test_suspension_noise_cannot_count_as_actual_driving(self):
        target = self.root/"noise"
        trial_files(target, passed=False, moved=False)
        result = batch.classify_trial(target)
        self.assertTrue(result["expert_entered"])
        self.assertFalse(result["actual_driving"])
        self.assertFalse(result["strict_pass"])

    def test_strict_pass_requires_cleanup_and_all_completion_fields(self):
        target = self.root/"pass"
        trial_files(target)
        self.assertTrue(batch.classify_trial(target)["strict_pass"])
        manifest = batch.read_json(target/"manifest.json")
        manifest["fixture_result"]["cleanup"]["verified"] = False
        batch.atomic_json(target/"manifest.json", manifest)
        self.assertFalse(batch.classify_trial(target)["strict_pass"])

    def test_resumed_pending_attempt_is_recovered_and_counted_exactly_once(self):
        worker = self.make()
        worker.acquire()
        target = self.root/"pending"
        trial_files(target)
        worker.state["attempts"] = [{"case_index": 0, "kind": "woodbox", "trial_dir": str(target), "committed": False}]
        worker.save()
        worker.release()
        resumed = self.make(self.args(resume=True))
        result = resumed.run()
        self.assertEqual(result["actual_drives"], 3)
        self.assertEqual(sum(entry["case_index"] == 0 for entry in result["attempts"]), 1)

    def test_unknown_resumed_pid_identity_is_not_signaled_or_recovered(self):
        worker = self.make()
        worker.acquire()
        worker.state["inflight"] = {"stage": "trial", "process_pid": 123, "process_pgid": 123, "process_start_ticks": "old"}
        worker.save()
        worker.release()
        resumed = self.make(self.args(resume=True))
        with patch.object(resumed, "process_identity", return_value={"root": "unrelated", "pgid": 123,
                "command": ["other"], "start_ticks": "new"}), patch.object(batch.os, "killpg") as kill, self.assertRaisesRegex(RuntimeError, "identity is uncertain"):
            resumed.run()
        kill.assert_not_called()
        self.assertEqual(resumed.events, [])

    def test_process_save_exception_still_interrupts_and_waits_for_child_before_recovery(self):
        worker = self.make()
        worker.state = {"inflight": {}}
        process = Mock(pid=123, returncode=130)
        process.poll.side_effect = [None, None, 130]
        with patch.object(batch.subprocess, "Popen", return_value=process), patch.object(worker, "save", side_effect=OSError("disk full")), patch.object(
                worker, "send_group") as send, self.assertRaisesRegex(OSError, "disk full"):
            batch.Batch.process(worker, ["script"], self.root/"child.log", 240.)
        send.assert_called_with(process, signal.SIGINT)
        process.wait.assert_called()

    def test_expanded_default30_preserves_legacy24_and_explicit12_in_either_order(self):
        base = ["--output", str(self.root), "--recovery-helper", str(self.helper)]
        self.assertEqual(batch.build_parser().parse_args(base).actual_runs, 24)
        self.assertEqual(batch.build_parser().parse_args(base+["--expanded-models"]).actual_runs, 30)
        for flags in (["--expanded-models", "--actual-runs", "12"], ["--actual-runs", "12", "--expanded-models"]):
            self.assertEqual(batch.build_parser().parse_args(base+flags).actual_runs, 12)

    def test_expanded_balanced30_rotates_every_20_static_and12_vehicle_model(self):
        statics = {"static_%02d" %i: {} for i in range(20)}
        vehicles = {"vehicle_%02d" %i: {} for i in range(12)}
        plan, counts = batch.expanded_plan(30, statics, vehicles, 7)
        self.assertEqual({kind: sum(row["kind"] == kind for row in plan) for kind in batch.EXPANDED_KINDS},
                         {kind: 6 for kind in batch.EXPANDED_KINDS})
        self.assertTrue(all(value > 0 for rows in counts.values() for value in rows.values()))
        self.assertEqual(plan, batch.expanded_plan(30, statics, vehicles, 7)[0])
        for row in plan:
            if row["kind"] in ("static_two", "vehicle_two"):
                self.assertEqual(len({item["model"] for item in row["models"]}), 2)

    def test_fixed_triples_run_last_and_duplicate_species_do_not_inflate_coverage_capacity(self):
        triples = [{"kind": "static_three", "path": "triple_%d.json" %i,
                    "models": [{"actor_kind": "obstacle", "model": "S0"}]*3} for i in range(3)]
        plan, _ = batch.expanded_plan(30, {"S%d" %i: {} for i in range(22)}, {"V": {}}, 7, triples)
        self.assertTrue(all("fixed_candidate" in row for row in plan[-3:]))
        self.assertTrue(all("fixed_candidate" not in row for row in plan[:-3]))
        with self.assertRaisesRegex(ValueError, "coverage exceeds"):
            batch.expanded_plan(30, {"S%d" %i: {} for i in range(23)}, {"V": {}}, 7, triples)

    def expanded_fixture(self, *, changed_dimensions=False):
        acceptance = self.root/"acceptance.json"
        footprint = {"width_m": 1., "length_m": 2., "center_offset_spawn_xy": [0., 0.], "heading_offset_deg": 1.347}
        entries = []
        for kind, name in (("obstacle", "S"), ("vehicle", "V")):
            path = self.root/(name+"_preview.json")
            planned = dict(footprint, width_m=1.1 if changed_dimensions else 1.)
            batch.atomic_json(path, {"planned": {"obstacles": [{"planned_bbox": planned}]}})
            entries.append({"actor_kind": kind, "model": name, "preview_path": str(path)})
        batch.atomic_json(acceptance, {"models": entries})
        args = batch.build_parser().parse_args(["--output", str(self.root/"expanded"), "--recovery-helper", str(self.helper),
            "--expanded-models", "--model-acceptance", str(acceptance)])
        pools = ({"S": footprint}, {"V": footprint})
        return args, pools

    def expanded_load(self, args, pools):
        return patch.multiple(batch, load_configuration=Mock(return_value=({"paths": {"mgeo_root": "map"}}, None, None)),
            MGeoMapLoader=Mock(return_value="loader"), load_static_models=Mock(return_value=pools[0]),
            load_vehicle_models=Mock(return_value=pools[1]), tunnel_geometry=Mock(return_value={"source_sha256": "current-map"}),
            accepted_model_pools=Mock(return_value=pools))

    def test_expanded_inputs_validate_acceptance_provenance_and_pin_model_bytes(self):
        args, pools = self.expanded_fixture()
        with self.expanded_load(args, pools):
            models, contract = batch.load_expanded_inputs(args)
            batch.accepted_model_pools.assert_called_once_with(args.model_acceptance, pools[0], pools[1], source_geometry_sha256="current-map")
        self.assertEqual(len(contract["plan"]), 30)
        self.assertEqual(contract["acceptance_sha256"], batch.digest(args.model_acceptance))
        self.assertEqual(contract["model_sha256"]["obstacle"]["S"], batch.fingerprint(pools[0]["S"]))
        self.assertEqual(contract["rotation_fingerprint"], batch.fingerprint(contract["plan"]))

    def test_changed_accepted_model_dimensions_are_rejected_before_scene_helper(self):
        args, pools = self.expanded_fixture(changed_dimensions=True)
        with self.expanded_load(args, pools), self.assertRaisesRegex(ValueError, "dimensions changed"):
            batch.load_expanded_inputs(args)

    def test_expanded_requires_acceptance_and_unaccepted_selection_is_rejected(self):
        args = batch.build_parser().parse_args(["--output", str(self.root), "--recovery-helper", str(self.helper), "--expanded-models"])
        with self.assertRaisesRegex(ValueError, "requires --model-acceptance"):
            batch.validate_args(args)
        args, pools = self.expanded_fixture()
        args.static_model = ["unaccepted"]
        with self.expanded_load(args, pools), self.assertRaisesRegex(ValueError, "Unaccepted selected"):
            batch.load_expanded_inputs(args)

    def test_three_candidate_cannot_enter_plan_with_stale_or_failed_pre_screening(self):
        args, pools = self.expanded_fixture()
        path = self.root/"triple.json"
        batch.atomic_json(path, {"count": 3, "experimental_three_actor_trial": True,
            "obstacles": [{"actor_kind": "obstacle", "model": "S"}]*3})
        args.three_candidate = [path]
        batch.atomic_json(path.with_name("screening.json"), {"status": "nominal_path_found", "path_found": True,
                                                         "geometry_checks": {"passed": True}})
        with self.expanded_load(args, pools), patch.object(batch, "provenance_matches", return_value=False), self.assertRaisesRegex(ValueError, "current passed screening"):
            batch.load_expanded_inputs(args)

    def test_expanded_resume_rejects_changed_acceptance_bytes_before_recovery(self):
        args, pools = self.expanded_fixture()
        with self.expanded_load(args, pools):
            original = self.make(args)
            original.acquire()
            original.release()
        document = batch.read_json(args.model_acceptance)
        document["source_revision_changed"] = True
        batch.atomic_json(args.model_acceptance, document)
        args.resume = True
        with self.expanded_load(args, pools):
            resumed = self.make(args)
            with self.assertRaisesRegex(RuntimeError, "Resume configuration differs"):
                resumed.run()
        self.assertEqual(resumed.events, [])

    def bound_candidate(self):
        return {"seed": 42, "candidate_index": 0, "count": 3, "experimental_three_actor_trial": True,
                "obstacles": [{"position": {"x": x, "y": 0., "z": 28.}, "yaw_deg": 12.,
                    "planned_bbox": {"width_m": 2., "length_m": 5., "center_offset_spawn_xy": [0., 0.], "heading_offset_deg": 0.}}
                    for x in (25., 50., 75.)]}

    def test_bound_conservative_triple_envelopes_are_accepted_without_rewriting_catalogue(self):
        candidate = self.bound_candidate()
        original = copy.deepcopy(candidate)
        boxes = [dict(batch.bbox_world(row), width_m=2.2, length_m=6.3) for row in candidate["obstacles"]]
        screen = {"candidate_seed": 42, "candidate_index": 0, "candidate_document_sha256": batch.fingerprint(candidate),
                  "obstacle_input": boxes}
        self.assertTrue(batch.screening_binds_candidate(candidate, screen))
        self.assertEqual(candidate, original)
        screen.pop("candidate_document_sha256")
        self.assertFalse(batch.screening_binds_candidate(candidate, screen))

    def test_bound_triple_rejects_shrunk_unrelated_or_different_document_geometry(self):
        candidate = self.bound_candidate()
        good = {"candidate_seed": 42, "candidate_index": 0, "candidate_document_sha256": batch.fingerprint(candidate),
                "obstacle_input": [dict(batch.bbox_world(row), width_m=2.2, length_m=6.3) for row in candidate["obstacles"]]}
        for variation in ("shrink", "relocate", "wrong_document", "reordered_rows"):
            with self.subTest(variation=variation):
                screen = copy.deepcopy(good)
                if variation == "shrink":
                    screen["obstacle_input"][0]["width_m"] = 1.9
                elif variation == "relocate":
                    screen["obstacle_input"][0]["center_xyz"] = [500., 500., 28.]
                elif variation == "wrong_document":
                    screen["candidate_document_sha256"] = "0"*64
                else:
                    screen["obstacle_input"] = list(reversed(screen["obstacle_input"]))
                self.assertFalse(batch.screening_binds_candidate(candidate, screen))

    def expanded_policy(self):
        from ..scenarios.full_loop_parts.tunnel_static_parts.advanced_patterns import PATTERN_COUNT_CAPS
        return {"selected_pattern": "short_detour", "footprint_domain": "two_lane_physical_road",
            "station_window_m": [20., 105.], "static_yaw_offsets_deg": {"Steel_Barricade": [90.]},
            "layout": {"min_pair_gap_m": 18., "max_width_m": 6.7},
            "campaign": {"patterns": list(PATTERN_COUNT_CAPS)},
            "actor_count_caps": {name: list(counts) for name, counts in PATTERN_COUNT_CAPS.items()}}

    def test_selected_campaign_covers_every_pattern_count_and_model_deterministically(self):
        policy = self.expanded_policy()
        statics = {"S%d" %i: {"width_m": .8, "length_m": 1.2} for i in range(11)}
        vehicles = {"V%d" %i: {"width_m": 2.2, "length_m": 4.7} for i in range(24)}
        plan, counts = batch.campaign_plan(54, statics, vehicles, 17, policy)
        expected = {(name, count) for name, caps in policy["actor_count_caps"].items() for count in caps}
        self.assertEqual({(row["pattern"], row["count"]) for row in plan}, expected)
        self.assertTrue(all(n>0 for pool in counts.values() for n in pool.values()))
        self.assertEqual((plan, counts), batch.campaign_plan(54, statics, vehicles, 17, policy))
        self.assertTrue(all(row["requested_pattern"] == row["pattern"] for row in plan))
        self.assertTrue(all(item["actor_kind"]=="obstacle" for row in plan if row["pattern"]=="window_bypass" for item in row["models"]))
        self.assertTrue(all(row["pattern"]=="long_group_bypass" for row in plan if any(item["actor_kind"]=="vehicle" for item in row["models"])))

    def test_selected_campaign_refuses_insufficient_layout_or_species_slots(self):
        policy = self.expanded_policy()
        with self.assertRaisesRegex(ValueError, "pattern/count"):
            batch.campaign_plan(2, {"S": {}}, {}, 1, policy)
        with self.assertRaisesRegex(ValueError, "coverage exceeds"):
            batch.campaign_plan(9, {"S%d" %i: {"width_m": .8, "length_m": 1.2} for i in range(100)}, {}, 1, policy)

    def test_incompatible_recipe_is_recorded_without_forcing_wide_vehicle_slots(self):
        policy=self.expanded_policy(); policy["campaign"]["patterns"]=["window_bypass", "long_group_bypass"]
        plan, counts=batch.campaign_plan(8, {}, {"WideCar": {"width_m": 2.2, "length_m": 4.7}}, 1, policy)
        self.assertTrue(all(row["pattern_supported"] is False for row in plan if row["pattern"]=="window_bypass"))
        self.assertGreater(counts["vehicle"]["WideCar"], 0)

    def test_batch_passes_exact_selected_pattern_domain_yaw_and_three_actor_cap(self):
        worker = self.make(); worker.models = (Mock(), {"Steel_Barricade": {}}, {})
        worker.expanded_contract = {"selected_layout_policy": self.expanded_policy()}
        case = {"index": 0, "kind": "alternating_three_3", "golden": False, "count": 3,
                "pattern": "alternating_three", "models": [{"actor_kind": "obstacle", "model": "Steel_Barricade"}]*3}
        value = {"obstacles": copy.deepcopy(case["models"])}
        with patch.object(batch, "generate_candidate", return_value=value) as generate:
            self.assertEqual(batch.Batch.candidate(worker, case, 0), value)
        options = generate.call_args.kwargs
        self.assertEqual(options["pattern"], "alternating_three")
        self.assertEqual(options["footprint_domain"], "two_lane_physical_road")
        self.assertEqual(options["static_yaw_offsets_deg"], {"Steel_Barricade": [90.]})
        self.assertEqual(options["station_window_m"], [20., 105.])
        self.assertIs(options["allow_three"], True)

    def test_trial_requested_layout_never_substitutes_for_actual_trajectory_pattern(self):
        target = self.root/"patterns"; trial_files(target)
        document = batch.read_json(target/"drive_trial.json")
        document.update(requested_pattern="alternating_three", realized_pattern="single_opposing_lane_excursion",
                        pattern_classification={"s_avoidance_verified": False})
        batch.atomic_json(target/"drive_trial.json", document)
        result = batch.classify_trial(target)
        self.assertEqual(result["requested_pattern"], "alternating_three")
        self.assertEqual(result["realized_pattern"], "single_opposing_lane_excursion")
        self.assertFalse(result["pattern_classification"]["s_avoidance_verified"])

    def test_selected_config_sha_and_policy_validation_are_explicit(self):
        import yaml
        policy = self.expanded_policy(); policy.update(schema="aimdrive_tunnel_static_config_v1", map="R_KR_PR_K-city_2025")
        path = self.root/"expanded.yaml"; path.write_text(yaml.safe_dump(policy))
        _, contract = batch.campaign_options(path)
        self.assertEqual(contract["config_sha256"], batch.digest(path))
        policy["actor_count_caps"]["alternating_three"] = [2]
        path.write_text(yaml.safe_dump(policy))
        with self.assertRaisesRegex(ValueError, "count caps"): batch.campaign_options(path)

    def prepare_pattern_worker(self, *, geometry=True):
        worker=self.make()
        plan=[dict(index=i,kind="short_detour_1",golden=False,count=1,
            requested_pattern="short_detour",pattern="short_detour",pattern_supported=True,
            models=[{"actor_kind":"obstacle","model":"WoodBox"}]) for i in range(2)]
        worker.expanded_contract={"selected_layout_policy": self.expanded_policy(),
            "model_sha256": {"obstacle": {"WoodBox": "fixed"}, "vehicle": {}}, "plan": plan}
        worker.snapshot=lambda: copy.deepcopy(self.snapshot)
        worker.args.max_attempts_per_case=2
        worker.acquire()
        worker.state["plan"]=plan
        worker.state["goal_actual_drives"]=2; worker.args.actual_runs=2
        worker.candidate=Mock(side_effect=[batch.NoCandidateGeometry("road body rejected")]*2+
                              [{"obstacles":[{"actor_kind":"obstacle","model":"WoodBox"}]}])
        return worker

    def test_advanced_geometry_exhaustion_recovers_excludes_then_runs_next_case(self):
        worker=self.prepare_pattern_worker()
        self.addCleanup(worker.release)
        worker.execute()
        self.assertEqual(len(worker.state["excluded_placements"]),1)
        self.assertEqual(worker.state["actual_drives"],1)
        self.assertEqual(worker.state["status"],"completed_with_exclusions")
        self.assertTrue(all(row.get("recovery_verified") for row in worker.state["attempts"]))
        self.assertEqual(worker.state["actual_model_occurrences"]["obstacle"]["WoodBox"],1)
        self.assertEqual(worker.state["strict_pass_model_occurrences"]["obstacle"]["WoodBox"],1)

    def test_advanced_transport_or_protocol_preflight_never_continues_as_geometry(self):
        worker=self.prepare_pattern_worker()
        self.addCleanup(worker.release)
        worker.candidate=Mock(return_value={"obstacles":[{"actor_kind":"obstacle","model":"WoodBox"}]})
        def failed(command, log_path, timeout, **kwargs):
            target=Path(command[command.index("--output")+1]); target.mkdir(parents=True)
            batch.atomic_json(target/"manifest.json",{"status":"failed","failure_reason":"StatusCode.UNAVAILABLE"})
            return {"returncode":2,"timed_out":False,"forced_kill":False}
        worker.process=failed
        with self.assertRaisesRegex(RuntimeError,"outside bounded geometric rejection"): worker.execute()
        self.assertEqual(worker.candidate.call_count,1)
        self.assertFalse(worker.state.get("excluded_placements"))

    def metadata_ordering_files(self, target, *, changed_publisher=False):
        target=Path(target); target.mkdir(parents=True,exist_ok=True)
        failure={"schema":"aimdrive_native_static_metadata_failure_v1","stage":"preflight_static_binding",
            "actor_kind":"obstacle","reason":"publisher_changed" if changed_publisher else "source_sequence_regression",
            "previous_publisher":"/native_bridge","current_publisher":"/replacement" if changed_publisher else "/native_bridge",
            "previous_source_sequence":100,"current_source_sequence":99,
            "previous_source_stamp_ns":1000,"current_source_stamp_ns":1100,
            "retryable_same_publisher_ordering":not changed_publisher}
        batch.atomic_json(target/"manifest.json",{"status":"failed","failure_reason":"Native static metadata ordering",
            "fixture_result":{"cleanup":{"verified":True},"native_metadata_failure":failure}})
        batch.atomic_json(target/"planner_diagnostics.json",{"calls":0})
        return failure

    def test_ordering_classifier_requires_static_same_publisher_preflight_and_cleanup(self):
        target=self.root/"ordering"; self.metadata_ordering_files(target)
        self.assertTrue(batch.classify_trial(target)["retryable_native_metadata_ordering"])
        for key,value in (("actor_kind","vehicle"),("current_publisher","/replacement"),
                          ("stage","runtime"),("current_source_sequence",101)):
            self.metadata_ordering_files(target)
            document=batch.read_json(target/"manifest.json")
            document["fixture_result"]["native_metadata_failure"][key]=value
            batch.atomic_json(target/"manifest.json",document)
            self.assertFalse(batch.classify_trial(target)["retryable_native_metadata_ordering"])
        self.metadata_ordering_files(target)
        batch.atomic_json(target/"planner_diagnostics.json",{"calls":1})
        self.assertFalse(batch.classify_trial(target)["retryable_native_metadata_ordering"])
        self.metadata_ordering_files(target)
        document=batch.read_json(target/"manifest.json");document["fixture_result"]["cleanup"]["verified"]=False
        batch.atomic_json(target/"manifest.json",document)
        self.assertFalse(batch.classify_trial(target)["retryable_native_metadata_ordering"])

    def test_advanced_ordering_retries_only_after_recovery_and_then_counts_actual_pass(self):
        worker=self.prepare_pattern_worker();self.addCleanup(worker.release)
        worker.candidate=Mock(return_value={"obstacles":[{"actor_kind":"obstacle","model":"WoodBox"}]})
        attempts=[]
        def process(command, log, timeout, **kwargs):
            target=Path(command[command.index("--output")+1]);attempts.append(target);worker.events.append("trial")
            if len(attempts)==1:self.metadata_ordering_files(target)
            else:trial_files(target)
            return {"returncode":2 if len(attempts)==1 else 0,"timed_out":False,"forced_kill":False}
        worker.process=process;worker.execute()
        self.assertEqual(worker.state["actual_drives"],2)
        self.assertEqual(worker.state["preflight_rejects"],1)
        self.assertFalse(worker.state.get("excluded_placements"))
        self.assertEqual(worker.events[:5],["recover","trial","recover","trial","recover"])

    def test_ordering_retry_exhaustion_stops_and_never_becomes_geometry_exclusion(self):
        worker=self.prepare_pattern_worker();self.addCleanup(worker.release)
        worker.candidate=Mock(return_value={"obstacles":[{"actor_kind":"obstacle","model":"WoodBox"}]})
        def process(command, log, timeout, **kwargs):
            self.metadata_ordering_files(Path(command[command.index("--output")+1]))
            return {"returncode":2,"timed_out":False,"forced_kill":False}
        worker.process=process
        with self.assertRaisesRegex(RuntimeError,"bounded attempts"):worker.execute()
        self.assertEqual(worker.candidate.call_count,2)
        self.assertFalse(worker.state.get("excluded_placements"))
        self.assertTrue(all(row["recovery_verified"] for row in worker.state["attempts"]))

    def test_resumed_ordering_is_recovered_once_and_remains_same_pending_case(self):
        worker=self.prepare_pattern_worker();self.addCleanup(worker.release)
        target=self.root/"resume_ordering";self.metadata_ordering_files(target)
        worker.state["attempts"]=[{"case_index":0,"kind":"short_detour_1","trial_dir":str(target),"committed":False,
            "process":{"returncode":2,"timed_out":False,"forced_kill":False}}]
        worker.reconcile_resume()
        self.assertEqual(worker.events,["recover"])
        self.assertEqual(worker.state["preflight_rejects"],1)
        self.assertEqual(worker.state["next_case_index"],0)
        worker.reconcile_resume()
        self.assertEqual(worker.state["preflight_rejects"],1)

    def test_advanced_publisher_replacement_stops_after_first_verified_recovery(self):
        worker=self.prepare_pattern_worker();self.addCleanup(worker.release)
        worker.candidate=Mock(return_value={"obstacles":[{"actor_kind":"obstacle","model":"WoodBox"}]})
        def process(command, log, timeout, **kwargs):
            self.metadata_ordering_files(Path(command[command.index("--output")+1]),changed_publisher=True)
            return {"returncode":2,"timed_out":False,"forced_kill":False}
        worker.process=process
        with self.assertRaisesRegex(RuntimeError,"outside bounded geometric rejection"):worker.execute()
        self.assertEqual(worker.candidate.call_count,1)
        self.assertFalse(worker.state.get("excluded_placements"))


if __name__ == "__main__":
    unittest.main()
