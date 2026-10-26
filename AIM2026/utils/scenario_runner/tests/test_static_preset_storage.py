# Catalog compaction preserves old bytes, paths and editable suite behavior.
import hashlib
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scenario_runner.tools import compact_static_presets
from scenario_runner.utils import static_preset_store as store


def _concurrent_writer(generated, worker):
    store.GENERATED_ROOT = Path(generated)
    for index in range(4):
        name = "worker_{}_{}.yaml".format(worker, index)
        store.write_text(store.GENERATED_ROOT / "static_test" / name, "name: " + name + "\n")


class StaticPresetStoreTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.generated = Path(self.temporary.name) / "generated"
        self.generated.mkdir()
        patch = mock.patch.object(store, "GENERATED_ROOT", self.generated)
        patch.start()
        self.addCleanup(patch.stop)
        self.suite = self.generated / "static_test"
        self.suite.mkdir()
        self.original = {
            "z_attempt.yaml": b"# retain comments\r\nname: z\r\nvalue: 1.000\r\n",
            "a_attempt.yaml": "# 배치 승인\nname: a\nvalue: 1\n".encode("utf-8"),
            "suite_manifest.json": b'{ "accepted_presets": ["z_attempt.yaml", "a_attempt.yaml"] }\n',
        }
        for name, payload in self.original.items():
            (self.suite / name).write_bytes(payload)
        self.catalog = self.generated / "static_test.json"

    def pack(self):
        return compact_static_presets.compact(self.suite, remove_loose=True)

    def test_bytes_hashes_and_original_paths_survive_without_extraction(self):
        texts = {name: (self.suite / name).read_text(encoding="utf-8") for name in self.original}
        result = self.pack()
        self.assertEqual(result["entries"], len(self.original))
        self.assertFalse(self.suite.exists())
        self.assertTrue(store.is_dir(self.suite))
        for name, payload in self.original.items():
            logical = self.suite / name
            with self.subTest(name=name):
                self.assertTrue(store.is_file(logical))
                self.assertEqual(store.read_bytes(logical), payload)
                self.assertEqual(store.read_text(logical), texts[name])
                self.assertEqual(hashlib.sha256(store.read_bytes(logical)).digest(), hashlib.sha256(payload).digest())
        self.assertFalse(self.suite.exists())
        self.assertEqual([p.name for p in store.glob_yaml(self.suite)], ["a_attempt.yaml", "z_attempt.yaml"])
        self.assertFalse(store.is_file(self.suite / "missing.yaml"))
        with self.assertRaises(FileNotFoundError):
            store.read_bytes(self.suite / "missing.yaml")

    def test_compact_default_keeps_all_sources(self):
        result = store.compact_suite(self.suite)
        self.assertEqual(set(result["source_sha256"]), {str(self.suite / name) for name in self.original})
        self.assertEqual({p.name: p.read_bytes() for p in self.suite.iterdir()}, self.original)

    def test_loose_override_and_external_files_keep_normal_behavior(self):
        self.pack()
        self.suite.mkdir()
        override = self.suite / "a_attempt.yaml"
        override.write_text("name: override\n", encoding="utf-8")
        self.assertEqual(store.read_text(override), "name: override\n")
        store.write_text(override, "name: changed\n")
        self.assertEqual(override.read_text(), "name: changed\n")
        self.assertEqual([p.name for p in store.glob_yaml(self.suite)], ["a_attempt.yaml", "z_attempt.yaml"])
        external = Path(self.temporary.name) / "external.yaml"
        store.write_text(external, "# custom\n")
        self.assertTrue(store.is_file(external))
        self.assertEqual(store.read_bytes(external), b"# custom\n")

    def test_manual_promotion_and_manifest_update_remain_packed(self):
        self.pack()
        new = self.suite / "new_approved.yaml"
        store.write_text(new, "name: new_approved\n")
        store.write_text(self.suite / "suite_manifest.json", '{"accepted_presets":["new_approved.yaml"]}\n')
        self.assertEqual(store.read_text(new), "name: new_approved\n")
        self.assertEqual(json.loads(store.read_text(self.suite / "suite_manifest.json"))["accepted_presets"], ["new_approved.yaml"])
        self.assertEqual(store.read_bytes(self.suite / "z_attempt.yaml"), self.original["z_attempt.yaml"])
        self.assertFalse(self.suite.exists())
        self.assertEqual([p.name for p in store.glob_yaml(self.suite)], ["a_attempt.yaml", "new_approved.yaml", "z_attempt.yaml"])

    def test_failed_atomic_replace_retains_previous_catalog_and_cleans_temporary(self):
        self.pack()
        before = self.catalog.read_bytes()
        with mock.patch.object(store.os, "replace", side_effect=OSError("replacement failed")):
            with self.assertRaisesRegex(OSError, "replacement failed"):
                store.write_text(self.suite / "a_attempt.yaml", "name: replacement\n")
        self.assertEqual(self.catalog.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in self.generated.iterdir()), [self.catalog.name])
        self.assertEqual(store.read_bytes(self.suite / "a_attempt.yaml"), self.original["a_attempt.yaml"])

    def test_corrupt_entry_is_rejected_by_reads_listing_and_existence_checks(self):
        self.pack()
        document = json.loads(self.catalog.read_text())
        document["entries"]["a_attempt.yaml"]["text"] = "name: tampered\n"
        self.catalog.write_text(json.dumps(document), encoding="utf-8")
        for operation in (lambda: store.read_bytes(self.suite / "a_attempt.yaml"),
                          lambda: store.glob_yaml(self.suite),
                          lambda: store.is_file(self.suite / "a_attempt.yaml"),
                          lambda: store.is_dir(self.suite)):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    operation()

    def test_unsafe_entry_names_and_wrong_suite_are_rejected(self):
        self.pack()
        original = json.loads(self.catalog.read_text())
        for unsafe in ("../escape.yaml", "/tmp/escape.yaml", "nested/file.yaml", "nested\\file.yaml", "a\x00.yaml"):
            document = json.loads(json.dumps(original))
            document["entries"][unsafe] = document["entries"].pop("a_attempt.yaml")
            self.catalog.write_text(json.dumps(document), encoding="utf-8")
            with self.subTest(name=unsafe), self.assertRaisesRegex(ValueError, "Unsafe"):
                store.glob_yaml(self.suite)
        original["suite"] = "another_suite"
        self.catalog.write_text(json.dumps(original), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid static preset catalog"):
            store.glob_yaml(self.suite)

    def test_duplicate_json_keys_are_rejected(self):
        self.pack()
        text = self.catalog.read_text()
        text = text.replace('"schema": ', '"suite": "static_test", "schema": ', 1)
        self.catalog.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            store.glob_yaml(self.suite)

    def test_compaction_refuses_unrecognized_files_or_symlinks_without_removing_sources(self):
        extra = self.suite / "unexpected.log"
        extra.write_text("do not discard")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.pack()
        self.assertFalse(self.catalog.exists())
        extra.unlink()
        target = self.suite / "z_attempt.yaml"
        target.unlink()
        target.symlink_to(self.suite / "a_attempt.yaml")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.pack()
        self.assertFalse(self.catalog.exists())
        self.assertTrue((self.suite / "suite_manifest.json").exists())

    def test_non_utf8_write_cannot_replace_a_valid_catalog(self):
        self.pack()
        before = self.catalog.read_bytes()
        with self.assertRaises(UnicodeError):
            store.write_text(self.suite / "a_attempt.yaml", "invalid UTF-16", encoding="utf-16")
        self.assertEqual(self.catalog.read_bytes(), before)

    def test_removal_refuses_sources_changed_after_compaction(self):
        real_compact = store.compact_suite
        def changed(directory):
            result = real_compact(directory)
            (self.suite / "a_attempt.yaml").write_text("name: concurrent_edit\n")
            return result
        with mock.patch.object(store, "compact_suite", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "Source changed"):
                self.pack()
        self.assertEqual(len(list(self.suite.iterdir())), len(self.original))

    def test_recompaction_keeps_existing_packed_entries_and_new_loose_entries(self):
        self.pack()
        self.suite.mkdir()
        (self.suite / "added.yaml").write_text("name: added\n")
        result = self.pack()
        self.assertEqual(result["entries"], len(self.original) + 1)
        self.assertEqual(store.read_bytes(self.suite / "z_attempt.yaml"), self.original["z_attempt.yaml"])
        self.assertEqual(store.read_text(self.suite / "added.yaml"), "name: added\n")
        self.assertFalse(self.suite.exists())

    def test_loose_writes_are_also_atomic_and_require_existing_parent(self):
        path = self.suite / "a_attempt.yaml"
        before = path.read_bytes()
        with mock.patch.object(store.os, "replace", side_effect=OSError("replacement failed")):
            with self.assertRaisesRegex(OSError, "replacement failed"):
                store.write_text(path, "name: replacement\n")
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(list(self.suite.iterdir())), len(self.original))
        with self.assertRaises(FileNotFoundError):
            store.write_text(self.generated / "absent" / "preset.yaml", "name: new\n")

    def test_loose_directories_and_invalid_operator_paths_keep_filesystem_errors(self):
        path = self.suite / "directory.yaml"
        path.mkdir()
        self.assertIn(path, store.glob_yaml(self.suite))
        with self.assertRaises(IsADirectoryError):
            store.read_text(path)
        self.assertFalse(store.is_file(path))
        for path in (self.suite / "unknown.txt", self.generated / "unsafe suite" / "preset.yaml"):
            with self.subTest(path=path), self.assertRaises(FileNotFoundError):
                store.read_text(path)

    def test_concurrent_process_writes_do_not_lose_other_entries(self):
        self.pack()
        context = multiprocessing.get_context("spawn")
        workers = [context.Process(target=_concurrent_writer, args=(str(self.generated), worker))
                   for worker in range(6)]
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=15)
                self.assertEqual(worker.exitcode, 0)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5)
        self.assertEqual(len(store.glob_yaml(self.suite)), 2 + 6 * 4)
        for name, payload in self.original.items():
            self.assertEqual(store.read_bytes(self.suite / name), payload)


# Keep editing, review and duplicate-suite protection usable after compaction.
from unittest.mock import patch

import yaml

from scenario_runner.tools import generate_static_scenario_suite as generator
from scenario_runner.tools import review_static_scenario_candidate as review
from scenario_runner.tools import static_scenario_editor as editor


class PackedPresetConsumersTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.generated = self.root / "generated"
        self.suite = self.generated / "saved_suite"
        self.suite.mkdir(parents=True)
        self.manifest = self.suite / "suite_manifest.json"
        self.manifest.write_text("{}\n", encoding="utf-8")
        override = patch.object(store, "GENERATED_ROOT", self.generated)
        override.start()
        self.addCleanup(override.stop)

    def pack(self):
        store.compact_suite(self.suite)
        for path in self.suite.iterdir():
            path.unlink()
        self.suite.rmdir()

    def test_manual_promotion_preserves_loose_result_without_extracting_suite(self):
        candidate = self.root / "candidate.yaml"
        candidate.write_text(yaml.safe_dump({
            "schema": "morai_static_obstacle_preset_v1", "name": "candidate",
            "generation": {"candidate_only_until_expert_passes": True},
            "verification": {"expert_completed": True}, "obstacles": [],
        }), encoding="utf-8")
        document = {
            "attempts": [{
                "attempt": 1, "accepted": False, "candidate": str(candidate),
                "models": ["Box"], "return_code": 0, "timed_out": False,
                "rejection_reasons": ["cross_track_limit_exceeded"],
                "closed_loop_evidence": {"goal_reached": True, "valid_plan_samples": 4},
            }],
            "accepted_count": 0, "accepted_model_counts": {"Box": 0},
            "accepted_model_minimum": 1, "accepted_model_minimum_met": False,
            "rejected_count": 1, "accepted_presets": [],
        }
        initial = json.dumps(document)
        self.manifest.write_text(initial, encoding="utf-8")
        with patch.object(review.time, "time", return_value=123.0):
            output = review.promote_candidate(self.manifest, 1, "reviewer", "review note")
        expected_preset, expected_manifest = output.read_bytes(), self.manifest.read_bytes()
        output.unlink()
        self.manifest.write_text(initial, encoding="utf-8")
        self.pack()

        with patch.object(review.time, "time", return_value=123.0):
            actual = review.promote_candidate(self.manifest, 1, "reviewer", "review note")
        self.assertEqual(actual, output)
        self.assertEqual(store.read_bytes(actual), expected_preset)
        self.assertEqual(store.read_bytes(self.manifest), expected_manifest)
        self.assertFalse(self.suite.exists())
        _, manifest_path, loaded = review.load_suite("saved_suite", self.generated)
        self.assertEqual(manifest_path, self.manifest)
        self.assertEqual(loaded["accepted_count"], 1)
        catalog_before = self.suite.with_suffix(".json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already manually accepted"):
            review.promote_candidate(self.manifest, 1, "reviewer", "repeat")
        self.assertEqual(self.suite.with_suffix(".json").read_bytes(), catalog_before)

    def test_editor_save_and_explicit_overwrite_keep_packed_suite_packed(self):
        state = editor.LiveState(self.suite)
        state.models = ["CargoBox"]
        state.model_scales = {}
        state.default_scale = [1.0, 1.0, 1.0]
        state.global_path = [[float(index), 0.0] for index in range(21)]
        state.update_obstacles([{
            "actor_id": "obstacle-1", "model": "CargoBox", "x": 8.2,
            "y": 0.4, "z": 2.0, "yaw_deg": 15.0, "width_m": 2.0, "length_m": 3.0,
        }])
        state.on_trial_event({"event": "trial_finished", "trial": 1,
                              "outcome": "passed", "reason": "goal_reached", "collisions": []})
        state.on_trial_event({"event": "trial_ready", "trial": 1, "previous_outcome": "passed"})
        request = {"name": "layout", "obstacles": [{"actor_id": "obstacle-1", "model": "CargoBox"}]}
        with patch.object(editor.time, "time", return_value=123.0):
            original = state.save(request)
        output = Path(original["saved"])
        expected = output.read_bytes()
        output.unlink()
        self.pack()
        with patch.object(editor.time, "time", return_value=123.0):
            actual = state.save(request)
            self.assertEqual(actual, original)
            self.assertEqual(store.read_bytes(output), expected)
            with self.assertRaises(FileExistsError):
                state.save(request)
            state.save(dict(request, overwrite=True))
        self.assertEqual(store.read_bytes(output), expected)
        self.assertFalse(self.suite.exists())

    def test_generator_rejects_existing_catalog_before_starting_a_trial(self):
        self.pack()
        with patch.object(generator, "OUTPUT_ROOT", self.generated), \
                patch.object(generator, "RUNTIME_ROOT", self.root / "runtime"), \
                patch.object(generator, "load_generation_policy", return_value={"models": ["A", "B", "C"]}), \
                patch.object(generator.subprocess, "Popen", side_effect=AssertionError("No trial expected")):
            with self.assertRaisesRegex(SystemExit, "Suite already exists: saved_suite"):
                generator.main(["--name", "saved_suite", "--placement", "anchor", "--seed", "1"])
        self.assertFalse(self.suite.exists())
        self.assertFalse((self.root / "runtime").exists())


import copy
from types import SimpleNamespace


from scenario_runner.utils import static_preset_store

from MoraiDrive.scenario_runner.scenarios.full_loop_parts.static_obstacle import (
    verified_pair_presets, load_static_obstacle_preset,
)
from MoraiDrive.scenario_runner.scenarios.full_loop_parts.highway import _merge_static_vehicle_pose


class VerifiedPairTests(unittest.TestCase):
    def test_rejects_failed_unverified_and_wrong_count_layouts(self):
        source = yaml.safe_load(static_preset_store.read_text(verified_pair_presets()[0]))
        with tempfile.TemporaryDirectory() as directory:
            for index, mutate in enumerate([
                lambda d: None,
                lambda d: d['verification'].update(collision_free=False),
                lambda d: d['verification']['lane_compliance'].update(passed=False),
                lambda d: d['generation'].update(candidate_only_until_expert_passes=True),
                lambda d: d['obstacles'].pop(),
                lambda d: d.update(blockage_release={'enabled': True}),
            ]):
                document = copy.deepcopy(source)
                mutate(document)
                (Path(directory) / ('%d.yaml' % index)).write_text(yaml.safe_dump(document))
            self.assertEqual([p.name for p in verified_pair_presets(directory)], ['0.yaml'])

    def test_all_real_pairs_preserve_original_geometry(self):
        pool = verified_pair_presets()
        self.assertGreater(len(pool), 0)
        for path in pool:
            source = yaml.safe_load(static_preset_store.read_text(path))
            points = [[o['position'][k] for k in ('x', 'y', 'z')] for o in source['obstacles']]
            loaded = load_static_obstacle_preset(path, source['map_name'], points)
            self.assertEqual(len(loaded['placements']), 2)
            for actual, original in zip(loaded['placements'], source['obstacles']):
                for field in ('model', 'position', 'yaw_deg', 'scale'):
                    self.assertEqual(actual[field], original[field])

    def test_disabled_merge_car_needs_no_spawn_or_route_lookup(self):
        context = SimpleNamespace(cfg={'highway_merge_static_vehicle': {'enabled': False}})
        self.assertIsNone(_merge_static_vehicle_pose(context))


if __name__ == "__main__":
    unittest.main()
