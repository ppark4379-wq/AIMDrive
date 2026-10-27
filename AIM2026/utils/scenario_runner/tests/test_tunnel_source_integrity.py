"""Tunnel regression checks: integrity."""

import copy, hashlib, json, tempfile, unittest, warnings, zipfile, ast, importlib, importlib.util, os, subprocess, sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from ..scenarios.full_loop_parts.tunnel_static_parts import replay, source_compatibility as compat, evidence, execution_policy as policy
from scenario_runner.project_paths import RUNNER_ROOT
from .. import tools


# Reviewed source migration preserves candidate eligibility, never approval.

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

    def test_every_manifest_source_preserves_bytes_and_path_lookup(self):
        rows = self.document['archived_sources']
        self.assertEqual(len(rows), 52)
        for source, row in rows.items():
            for recorded in (source, str(compat.PROJECT_ROOT / source),
                             str(Path(self.document['previous_project_root']) / source)):
                with self.subTest(source=source, recorded=recorded):
                    raw = compat.archived_source_bytes(recorded, row['sha256'])
                    self.assertIsNotNone(raw)
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), row['sha256'])
                    archived = compat.archived_source_path(recorded, row['sha256'])
                    self.assertIsInstance(archived, Path)
                    self.assertEqual(archived.read_bytes(), raw)

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


class PackedSourceArchiveTests(unittest.TestCase):
    @contextmanager
    def packed_archive(self):
        """A pinned two-source fixture independent of any external project."""
        document = copy.deepcopy(compat._main_compatibility_document())
        source = 'scenario_runner/selected.py'
        unrelated = 'scenario_runner/unrelated.py'
        payload = b'original source\r\nwith exact bytes\x00\xff\n'
        other_payload = b'unrelated original source\n'
        rows = {}
        entries = {}
        for name, data in ((source, payload), (unrelated, other_payload)):
            sha = hashlib.sha256(data).hexdigest()
            basename = sha + '.source'
            rows[name] = {'sha256': sha,
                          'archive_path': 'scenario_runner/config/tunnel_static_source_archive/' + basename}
            entries[basename] = data
        document['archived_sources'] = rows
        document['previous_screening_provenance']['planner_source_sha256'] = {
            source: rows[source]['sha256']}
        document['current_screening_provenance']['planner_source_sha256'] = {source: '0' * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'scenario_runner/config/tunnel_static_source_archive.zip'
            archive.parent.mkdir(parents=True)
            manifest = root / 'manifest.json'
            raw = json.dumps(document).encode('utf-8')
            manifest.write_bytes(raw)

            def rewrite(values):
                with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as packed:
                    for name, data in values.items():
                        packed.writestr(name, data)

            rewrite(entries)
            with patch.object(compat, 'PROJECT_ROOT', root), \
                    patch.object(compat, 'MAIN_COMPATIBILITY_PATH', manifest), \
                    patch.object(compat, 'REVIEWED_MAIN_COMPATIBILITY_SHA256', hashlib.sha256(raw).hexdigest()):
                yield {'root': root, 'archive': archive, 'source': source,
                       'unrelated': unrelated, 'payload': payload, 'rows': rows,
                       'entries': entries, 'document': document, 'rewrite': rewrite}

    def test_packed_sources_preserve_bytes_and_real_path_io(self):
        with self.packed_archive() as fixture:
            source = fixture['source']
            sha = fixture['rows'][source]['sha256']
            for recorded in (source, str(fixture['root'] / source),
                             str(Path(fixture['document']['previous_project_root']) / source)):
                with self.subTest(recorded=recorded):
                    self.assertEqual(compat.archived_source_bytes(recorded, sha), fixture['payload'])
                    archived = compat.archived_source_path(recorded, sha)
                    self.assertIsInstance(archived, Path)
                    self.assertTrue(archived.is_file())
                    with open(archived, 'rb') as stream:
                        self.assertEqual(stream.read(), fixture['payload'])
                    self.assertEqual(archived.stat().st_size, len(fixture['payload']))
            self.assertFalse((fixture['root'] / fixture['rows'][source]['archive_path']).exists())

    def test_loose_content_takes_precedence_and_cannot_hide_tampering(self):
        with self.packed_archive() as fixture:
            source = fixture['source']
            row = fixture['rows'][source]
            loose = fixture['root'] / row['archive_path']
            loose.parent.mkdir()
            loose.write_bytes(b'tampered loose source')
            self.assertIsNone(compat.archived_source_bytes(source, row['sha256']))
            self.assertIsNone(compat.archived_source_path(source, row['sha256']))
            loose.unlink()
            self.assertEqual(compat.archived_source_bytes(source, row['sha256']), fixture['payload'])
            loose.write_bytes(fixture['payload'])
            self.assertEqual(compat.archived_source_path(source, row['sha256']), loose.resolve())

    def test_missing_or_corrupt_selected_source_never_uses_stale_materialization(self):
        with self.packed_archive() as fixture:
            source = fixture['source']
            sha = fixture['rows'][source]['sha256']
            cached = compat.archived_source_path(source, sha)
            self.assertEqual(cached.read_bytes(), fixture['payload'])
            selected_name = sha + '.source'
            missing = {name: data for name, data in fixture['entries'].items() if name != selected_name}
            for entries in (missing, dict(fixture['entries'], **{selected_name: b'tampered'})):
                fixture['rewrite'](entries)
                self.assertIsNone(compat.archived_source_bytes(source, sha))
                self.assertIsNone(compat.archived_source_path(source, sha))
            fixture['archive'].write_bytes(b'not a ZIP archive')
            self.assertIsNone(compat.archived_source_bytes(source, sha))
            self.assertIsNone(compat.archived_source_path(source, sha))
            fixture['archive'].unlink()
            self.assertIsNone(compat.archived_source_bytes(source, sha))
            self.assertIsNone(compat.archived_source_path(source, sha))

    def test_unrelated_source_corruption_does_not_change_selected_screening(self):
        with self.packed_archive() as fixture:
            source = fixture['source']
            unrelated = fixture['unrelated']
            other_sha = fixture['rows'][unrelated]['sha256']
            current = fixture['document']['current_screening_provenance']
            expected = compat.reviewed_main_screening_transition(current)
            self.assertIsNotNone(expected)
            changed = dict(fixture['entries'], **{other_sha + '.source': b'tampered unrelated source'})
            fixture['rewrite'](changed)
            self.assertEqual(compat.archived_source_bytes(source, fixture['rows'][source]['sha256']),
                             fixture['payload'])
            self.assertIsNone(compat.archived_source_bytes(unrelated, other_sha))
            self.assertEqual(compat.reviewed_main_screening_transition(current), expected)

    def test_duplicate_selected_zip_members_are_rejected(self):
        with self.packed_archive() as fixture:
            source = fixture['source']
            sha = fixture['rows'][source]['sha256']
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                with zipfile.ZipFile(fixture['archive'], 'a') as packed:
                    packed.writestr(sha + '.source', fixture['payload'])
            self.assertIsNone(compat.archived_source_bytes(source, sha))
            self.assertIsNone(compat.archived_source_path(source, sha))
            self.assertIsNone(compat.reviewed_main_screening_transition(
                fixture['document']['current_screening_provenance']))

    def test_runtime_screening_reads_archive_without_requesting_real_paths(self):
        with self.packed_archive() as fixture:
            before = sorted(path.relative_to(fixture['root']) for path in fixture['root'].rglob('*'))
            with patch.object(compat, 'archived_source_path', side_effect=AssertionError('unexpected extraction')):
                result = compat.reviewed_main_screening_transition(
                    fixture['document']['current_screening_provenance'])
            self.assertIsNotNone(result)
            self.assertFalse(result['physical_approval_transferred'])
            after = sorted(path.relative_to(fixture['root']) for path in fixture['root'].rglob('*'))
            self.assertEqual(after, before)

    def test_modified_materialized_file_does_not_override_verified_archive(self):
        with self.packed_archive() as fixture:
            source = fixture['source']
            sha = fixture['rows'][source]['sha256']
            cached = compat.archived_source_path(source, sha)
            cached.chmod(0o600)
            cached.write_bytes(b'tampered cached copy')
            self.assertEqual(compat.archived_source_bytes(source, sha), fixture['payload'])
            refreshed = compat.archived_source_path(source, sha)
            self.assertIsInstance(refreshed, Path)
            self.assertEqual(refreshed.read_bytes(), fixture['payload'])

    def test_packed_lookup_keeps_root_and_hash_restrictions(self):
        with self.packed_archive() as fixture:
            source = fixture['source']
            sha = fixture['rows'][source]['sha256']
            for recorded in ('../' + source, '/unreviewed/project/' + source,
                             source.replace('/', '\\'), 'scenario_runner/missing.py'):
                with self.subTest(recorded=recorded):
                    self.assertIsNone(compat.archived_source_bytes(recorded, sha))
                    self.assertIsNone(compat.archived_source_path(recorded, sha))
            self.assertIsNone(compat.archived_source_bytes(source, '0' * 64))
            self.assertIsNone(compat.archived_source_path(source, '0' * 64))


# Approval identity follows immutable execution dependencies, never run history.

class TunnelPolicySourcePinsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.expert = self.root/"expert"
        original = self.expert/"src/config/config.json"
        original.parent.mkdir(parents=True)
        original.write_text(json.dumps({"planning":{"velocity_profile":{"max_velocity":37.}}}))
        self.controller = self.expert/"src/control/path_controller.py"
        self.controller.parent.mkdir(parents=True)
        self.controller.write_text("# reviewed controller revision\n")
        self.map = self.root/"map"
        self.map.mkdir()
        (self.map/"link_set.json").write_text("[]")
        (self.map/"node_set.json").write_text("[]")
        self.library = self.root/"library.json"
        self.library.write_text(json.dumps({"schema":"isolated-policy-source-test","candidates":[]}))
        self.global_cfg = {"paths":{"expert_root":str(self.expert),"mgeo_root":str(self.map),
                                    "runtime_root":str(self.root/"runtime")}}

    def describe(self):
        return policy.describe(self.global_cfg,self.library)

    def test_policy_includes_every_shared_runtime_snapshot_dependency(self):
        helper = policy.PACKAGE_ROOT/"tools/tunnel_static_batch_safety.sh"
        expected = evidence.source_snapshot({},helper,helper)["runtime_files"]
        actual = self.describe()["source_files"]
        for path,sha in expected.items():
            with self.subTest(path=path):
                self.assertEqual(actual.get(path),sha)

    def test_transport_scenario_configuration_and_actual_map_are_pinned(self):
        actual = self.describe()["source_files"]
        paths = [policy.PACKAGE_ROOT/relative for relative in (
            "tools/__init__.py", "tools/run_tunnel_static_trial.sh", "tools/tunnel_static_batch_safety.sh",
            "config/global.yaml", "config/full_loop.yaml", "config/tunnel_static.yaml",
            "scenarios/full_loop_parts/tunnel_static_parts/connection.py", "adapters/morai_grpc.py",
            "core/base_scenario.py", "utils/map_loader.py", "utils/link_graph.py", "utils/transform_utils.py",
            "scenarios/full_loop_parts/actor_lifecycle.py", "scenarios/full_loop_parts/vehicle_common.py",
            "scenarios/full_loop_parts/static_obstacle.py",
        )]
        paths += [self.map/"link_set.json", self.map/"node_set.json", self.controller,
                  RUNNER_ROOT/"tools/project_env.sh"]
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(actual.get(str(path.resolve())),policy.digest(path))

    def test_controller_or_map_change_invalidates_policy_identity(self):
        before = policy.identity(self.describe())
        self.controller.write_text("# changed controller revision\n")
        after_controller = policy.identity(self.describe())
        self.assertNotEqual(before,after_controller)
        (self.map/"link_set.json").write_text('[{"different":"road"}]')
        self.assertNotEqual(after_controller,policy.identity(self.describe()))

    def test_mutable_weather_approval_and_log_files_do_not_change_policy_identity(self):
        before = policy.identity(self.describe())
        runtime = self.root/"runtime"
        runtime.mkdir()
        for name in ("full_loop_tunnel_static_profile.json", "current_expert_approvals.json", "trial.log"):
            (runtime/name).write_text("mutable run state\n")
        self.assertEqual(before,policy.identity(self.describe()))


# Only a reviewed complete runtime migration can feed policy revalidation.

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


# Relocated tunnel runtime keeps command compatibility and one implementation.

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent


class TunnelStaticStructureTests(unittest.TestCase):
    def command(self, *arguments):
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, *arguments], cwd=str(PROJECT_ROOT), env=environment,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15.)

    def test_legacy_imports_share_the_canonical_module_and_preserve_its_spec(self):
        for old,target in tools._TUNNEL_LEGACY_MODULES.items():
            with self.subTest(old=old):
                canonical = importlib.import_module(target)
                legacy = importlib.import_module("scenario_runner.tools."+old)
                self.assertIs(legacy,canonical)
                self.assertEqual(canonical.__spec__.name,target)
                self.assertEqual(canonical.__package__,target.rsplit(".",1)[0])

    def test_legacy_patching_reaches_the_same_runtime_function(self):
        legacy = importlib.import_module("scenario_runner.tools.replay_tunnel_static_library")
        canonical = importlib.import_module(tools._TUNNEL_LEGACY_MODULES["replay_tunnel_static_library"])
        with patch.object(legacy,"digest",return_value="patched"):
            self.assertEqual(canonical.digest("unused"),"patched")

    def test_normal_runtime_imports_do_not_load_development_tool_modules(self):
        code = ("import json,sys; "
                "from scenario_runner.scenarios.full_loop_parts.tunnel_static_parts import replay,execution,trial,safety; "
                "print(json.dumps(sorted(name for name in sys.modules "
                "if name.startswith('scenario_runner.tools.tunnel_static.'))))")
        result = self.command("-c",code)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1]),[])

    def test_every_legacy_command_and_grouped_entrypoint_still_support_help(self):
        for old in tools._TUNNEL_LEGACY_MODULES:
            with self.subTest(old=old):
                result = self.command("-m","scenario_runner.tools."+old,"--help")
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertNotIn("Traceback",result.stderr)
        for arguments in (("--help",),("trial","--help"),("verify-models","--help")):
            with self.subTest(grouped=arguments):
                result = self.command("-m","scenario_runner.tools.tunnel_static",*arguments)
                self.assertEqual(result.returncode,0,result.stderr)

    def test_flat_tools_keep_only_two_tunnel_static_runtime_launch_wrappers(self):
        files = {path.name for path in (PACKAGE_ROOT/"tools").glob("*tunnel_static*") if path.is_file()}
        self.assertEqual(files,{"run_tunnel_static_trial.sh","tunnel_static_batch_safety.sh"})
        for path in (PACKAGE_ROOT/"tools/tunnel_static").glob("*.sh"):
            result = subprocess.run(["bash","-n",str(path)],capture_output=True,text=True,timeout=5.)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_fresh_source_pins_cover_canonical_code_and_legacy_dispatch(self):
        helper = PACKAGE_ROOT/"tools/tunnel_static_batch_safety.sh"
        snapshot = evidence.source_snapshot({},helper,helper)
        files = snapshot["runtime_files"]
        for relative in ("tools/__init__.py","scenarios/full_loop_parts/tunnel_static_parts/replay.py",
                         "scenarios/full_loop_parts/tunnel_static_parts/execution.py",
                         "scenarios/full_loop_parts/tunnel_static_parts/trial.py",
                         "scenarios/full_loop_parts/tunnel_static_parts/safety.py"):
            path = PACKAGE_ROOT/relative
            self.assertEqual(files[str(path.resolve())],evidence.digest(path))
        self.assertTrue(all(Path(path).is_file() for path in files))

    def test_deferred_development_relative_imports_resolve_after_relocation(self):
        for path in (PACKAGE_ROOT/"tools/tunnel_static").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom) and node.level and node.module:
                    target = importlib.util.resolve_name("."*node.level+node.module,
                                                         "scenario_runner.tools.tunnel_static")
                    with self.subTest(file=path.name,line=node.lineno,target=target):
                        self.assertIsNotNone(importlib.util.find_spec(target))


if __name__ == "__main__":
    unittest.main()
