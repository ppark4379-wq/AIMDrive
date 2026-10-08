"""Only a reviewed complete runtime migration can feed policy revalidation."""

import copy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

from ..scenarios.full_loop_parts.tunnel_static_parts import source_compatibility as compat


class RuntimeSourceCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.old = copy.deepcopy(compat.LEGACY_RUNTIME_SOURCE_SHA256)
        self.new = dict(self.old)
        self.scenario_path = next(path for path in self.new if path.endswith("adapters/expert_controller.py"))
        self.new[self.scenario_path] = "1"*64

    def test_equal_current_sources_keep_current_provenance(self):
        result = compat.runtime_source_compatibility(self.new,self.new)
        self.assertFalse(result["historical_runtime_provenance"])
        self.assertFalse(result["requires_new_policy_revalidation"])
        self.assertIsNone(result["reviewed_runtime_transition"])

    def test_exact_reviewed_complete_transition_is_candidate_only_and_never_rewrites_proof(self):
        original = copy.deepcopy(self.old)
        with patch.object(compat,"PRE_MAIN_RUNTIME_SOURCE_SHA256",self.new):
            result = compat.runtime_source_compatibility(self.old,self.new)
        self.assertTrue(result["historical_runtime_provenance"])
        self.assertTrue(result["requires_new_policy_revalidation"])
        self.assertFalse(result["historical_runtime_sources_rewritten"])
        self.assertFalse(result["reviewed_runtime_transition"]["physical_approval_transferred"])
        self.assertFalse(result["reviewed_runtime_transition"]["expert_source_changed"])
        self.assertEqual(result["reviewed_runtime_transition"]["source_file_count"],8)
        result["reviewed_runtime_transition"]["original_sha256"][self.scenario_path] = "changed-copy"
        self.assertEqual(self.old,original)

    def test_unlisted_hash_missing_key_extra_key_or_expert_change_fails_closed(self):
        variants=[]
        unknown=dict(self.new); unknown[self.scenario_path]="unreviewed"; variants.append(unknown)
        missing=dict(self.new); missing.pop(self.scenario_path); variants.append(missing)
        extra=dict(self.new); extra["/unreviewed/source.py"]="2"*64; variants.append(extra)
        expert=dict(self.new)
        path=next(path for path in expert if "/expert/src/expert/src/network/" in path)
        expert[path]="3"*64; variants.append(expert)
        with patch.object(compat,"PRE_MAIN_RUNTIME_SOURCE_SHA256",self.new):
            for current in variants:
                with self.subTest(current=current),self.assertRaisesRegex(ValueError,"Runtime source identity/hash changed"):
                    compat.runtime_source_compatibility(self.old,current)

    def test_unknown_historical_hash_and_reverse_transition_are_not_accepted(self):
        with patch.object(compat,"PRE_MAIN_RUNTIME_SOURCE_SHA256",self.new):
            old=dict(self.old); old[self.scenario_path]="unknown-history"
            with self.assertRaises(ValueError): compat.runtime_source_compatibility(old,self.new)
            with self.assertRaises(ValueError): compat.runtime_source_compatibility(self.new,self.old)

    def test_empty_or_unreviewed_transition_never_passes(self):
        with patch.object(compat,"PRE_MAIN_RUNTIME_SOURCE_SHA256",{}):
            with self.assertRaises(ValueError): compat.runtime_source_compatibility(self.old,self.new)
        for old,current in (({},{}),(None,self.new),(self.old,None)):
            with self.subTest(old=old,current=current),self.assertRaises(ValueError):
                compat.runtime_source_compatibility(old,current)

    def test_reviewed_target_is_the_exact_frozen_current_eight_files(self):
        self.assertEqual(set(compat.REVIEWED_RUNTIME_SOURCE_SHA256),set(self.old))
        self.assertEqual(len(compat.REVIEWED_RUNTIME_SOURCE_SHA256),8)
        for path,sha in compat.REVIEWED_RUNTIME_SOURCE_SHA256.items():
            with self.subTest(path=path):
                self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),sha)

    def test_scenario_migration_cannot_allow_expert_edits_even_with_an_incorrect_registry(self):
        current=dict(self.new)
        expert_path=next(path for path in current if "/expert/src/expert/src/network/" in path)
        current[expert_path]="4"*64
        with patch.object(compat,"PRE_MAIN_RUNTIME_SOURCE_SHA256",current), \
                self.assertRaisesRegex(ValueError,"Expert runtime source changed"):
            compat.runtime_source_compatibility(self.old,current)

    def test_reviewed_goal_stop_preserves_exact_producer_pins_without_transferring_approval(self):
        old = copy.deepcopy(compat.PRE_GOAL_STOP_RUNTIME_SOURCE_SHA256)
        current = dict(compat.PRE_MAIN_RUNTIME_SOURCE_SHA256)
        result = compat.reviewed_goal_stop_runtime_transition(old, current)
        self.assertIsNotNone(result)
        self.assertFalse(result['physical_approval_transferred'])
        self.assertFalse(result['historical_runtime_sources_rewritten'])
        result['original_sha256'].clear()
        self.assertEqual(old, compat.PRE_GOAL_STOP_RUNTIME_SOURCE_SHA256)

    def test_goal_stop_rejects_unknown_missing_extra_and_reverse_source_endpoints(self):
        old = dict(compat.PRE_GOAL_STOP_RUNTIME_SOURCE_SHA256)
        current = dict(compat.PRE_MAIN_RUNTIME_SOURCE_SHA256)
        path = next(path for path in old if path.endswith('/core/full_loop_drive_base.py'))
        variants = []
        unknown = dict(current); unknown[path] = 'unknown'; variants.append((old, unknown))
        missing = dict(current); missing.pop(path); variants.append((old, missing))
        extra = dict(current); extra['/extra/unreviewed.py'] = '1'*64; variants.append((old, extra))
        changed_producer = dict(old); changed_producer[path] = '2'*64; variants.append((changed_producer, current))
        variants.extend(((current, old), (old, old)))
        for saved, endpoint in variants:
            with self.subTest(saved=saved, endpoint=endpoint):
                self.assertIsNone(compat.reviewed_goal_stop_runtime_transition(saved, endpoint))

    def test_goal_stop_cannot_include_an_expert_change_even_if_registry_is_wrong(self):
        old = dict(compat.PRE_GOAL_STOP_RUNTIME_SOURCE_SHA256)
        current = dict(compat.PRE_MAIN_RUNTIME_SOURCE_SHA256)
        expert = next(path for path in old if '/expert/src/expert/src/network/' in path)
        current[expert] = '3'*64
        with patch.object(compat, 'PRE_MAIN_RUNTIME_SOURCE_SHA256', current):
            self.assertIsNone(compat.reviewed_goal_stop_runtime_transition(old, current))


if __name__ == "__main__":
    unittest.main()
