"""Reviewed source migration preserves candidate eligibility, never approval."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ..scenarios.full_loop_parts.tunnel_static_parts import replay
from ..scenarios.full_loop_parts.tunnel_static_parts import source_compatibility as compat


class MainSourceMigrationTests(unittest.TestCase):
    def setUp(self):
        self.document = compat._main_compatibility_document()
        self.previous = copy.deepcopy(self.document['previous_screening_provenance'])
        self.current = copy.deepcopy(self.document['current_screening_provenance'])

    def test_exact_upgrade_requires_current_fresh_guards_and_preserves_evidence(self):
        original = copy.deepcopy(self.previous)
        result = replay._screen_compatibility(self.previous, self.current)
        upgrade = result['reviewed_main_screening_transition']
        self.assertTrue(result['historical_screening_provenance'])
        self.assertTrue(upgrade['fresh_current_planner_and_runtime_guards_required'])
        self.assertFalse(upgrade['physical_approval_transferred'])
        self.assertFalse(upgrade['historical_evidence_rewritten'])
        self.assertEqual(self.previous, original)

    def test_historical_live_change_is_checked_before_main_upgrade(self):
        original = copy.deepcopy(self.previous)
        old, _ = next(iter(replay.REVIEWED_LIVE_PREPARATION_TRANSITIONS))
        original['planner_source_sha256'][replay.LIVE_PREPARATION_SOURCE] = old
        result = replay._screen_compatibility(original, self.current, allow_layout_preparation_change=False)
        self.assertEqual(result['reviewed_live_preparation_transition']['original_sha256'], old)
        original['planner_source_sha256'][replay.LIVE_PREPARATION_SOURCE] = 'unknown'
        with self.assertRaises(ValueError):
            replay._screen_compatibility(original, self.current)

    def test_current_source_proof_has_no_historical_upgrade(self):
        result = replay._screen_compatibility(self.current, self.current)
        self.assertFalse(result['historical_screening_provenance'])
        self.assertNotIn('reviewed_main_screening_transition', result)

    def test_unknown_expert_key_recording_route_and_policy_are_rejected(self):
        variants = []
        altered = copy.deepcopy(self.current)
        altered['planner_source_sha256']['expert/src/expert/src/autonomous_driving.py'] = '0'*64
        variants.append(altered)
        altered = copy.deepcopy(self.current)
        altered['planner_source_sha256']['extra/unreviewed.py'] = '0'*64
        variants.append(altered)
        for key in ('original_route_sha256', 'screening_policy_sha256'):
            altered = copy.deepcopy(self.current); altered[key] = '0'*64; variants.append(altered)
        altered = copy.deepcopy(self.current)
        key = next(iter(altered['recorded_library_sha256']))
        altered['recorded_library_sha256'][key] = '0'*64; variants.append(altered)
        altered = copy.deepcopy(self.current)
        altered['screening_policy']['full_body_road_margin_m'] = 0.; variants.append(altered)
        for current in variants:
            with self.subTest(current=current):
                self.assertIsNone(compat.reviewed_main_screening_transition(current))
                with self.assertRaises(ValueError):
                    replay._screen_compatibility(self.previous, current)

    def test_unknown_historical_expert_cannot_use_the_main_upgrade(self):
        old = copy.deepcopy(self.previous)
        old['planner_source_sha256']['expert/src/expert/src/autonomous_driving.py'] = '0'*64
        with self.assertRaises(ValueError):
            replay._screen_compatibility(old, self.current)

    def test_config_tampering_invalidates_migration_and_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            changed = copy.deepcopy(self.document)
            changed['current_screening_provenance'] = self.previous
            path.write_text(json.dumps(changed))
            with patch.object(compat, 'MAIN_COMPATIBILITY_PATH', path):
                self.assertIsNone(compat.reviewed_main_screening_transition(self.current))
                with self.assertRaises(ValueError):
                    replay._screen_compatibility(self.previous, self.current)
                source, row = next(iter(changed['archived_sources'].items()))
                self.assertIsNone(compat.archived_source_path(source, row['sha256']))

    def test_archive_requires_exact_root_hash_and_original_content(self):
        source, row = next(iter(self.document['archived_sources'].items()))
        for path in (source, str(compat.PROJECT_ROOT/source),
                     str(Path(self.document['previous_project_root'])/source)):
            archived = compat.archived_source_path(path, row['sha256'])
            self.assertIsNotNone(archived)
            self.assertEqual(hashlib.sha256(archived.read_bytes()).hexdigest(), row['sha256'])
        for path in ('../'+source, '/unreviewed/project/'+source):
            self.assertIsNone(compat.archived_source_path(path, row['sha256']))
        self.assertIsNone(compat.archived_source_path(source, '0'*64))

    def test_archive_tampered_bytes_are_rejected_without_editing_real_archive(self):
        source, row = next(iter(self.document['archived_sources'].items()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root/row['archive_path']; archive.parent.mkdir(parents=True)
            archive.write_bytes(b'tampered source')
            manifest = root/'manifest.json'; raw = json.dumps(self.document).encode(); manifest.write_bytes(raw)
            with patch.object(compat, 'PROJECT_ROOT', root), \
                    patch.object(compat, 'MAIN_COMPATIBILITY_PATH', manifest), \
                    patch.object(compat, 'REVIEWED_MAIN_COMPATIBILITY_SHA256', hashlib.sha256(raw).hexdigest()):
                self.assertIsNone(compat.archived_source_path(source, row['sha256']))

    def test_runtime_known_endpoints_portable_only_between_explicit_roots(self):
        current = compat.REVIEWED_RUNTIME_SOURCE_SHA256
        for old in (compat.LEGACY_RUNTIME_SOURCE_SHA256,
                    compat.PRE_GOAL_STOP_RUNTIME_SOURCE_SHA256,
                    self.document['previous_runtime_source_sha256']):
            result = compat.runtime_source_compatibility(old, current)
            self.assertTrue(result['requires_new_policy_revalidation'])
            self.assertFalse(result['historical_runtime_sources_rewritten'])
            self.assertFalse(result['reviewed_runtime_transition']['physical_approval_transferred'])
            self.assertTrue(result['reviewed_runtime_transition']['fresh_current_planner_and_runtime_guards_required'])

    def test_runtime_unknown_hash_missing_extra_key_and_unreviewed_root_reject(self):
        old, current = self.document['previous_runtime_source_sha256'], compat.REVIEWED_RUNTIME_SOURCE_SHA256
        bad = dict(current); bad[next(iter(bad))] = '0'*64
        missing = dict(current); missing.pop(next(iter(missing)))
        extra = dict(current); extra['/unknown/source.py'] = '0'*64
        for value in (bad, missing, extra):
            with self.assertRaises(ValueError):
                compat.runtime_source_compatibility(old, value)
        unrelated = {'/unknown/project/'+Path(path).relative_to(self.document['previous_project_root']).as_posix(): sha
                     for path, sha in old.items()}
        with self.assertRaises(ValueError):
            compat.runtime_source_compatibility(unrelated, current)


if __name__ == '__main__':
    unittest.main()
