"""좌회전 신호 수집은 실제 단계 관측과 봉인 근거가 모두 있어야 통과한다."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[2]
INTERNAL = PROJECT / 'data_collection/internal_sensor_record'
sys.path[:0] = [str(PROJECT), str(INTERNAL), str(INTERNAL / 'tools')]
from async_capture import traffic_light_collection as signal_collection
from async_capture.route_signal_binding import bind_route_progress
from scenario_collection import configure_collection_scenario
from tools.build_sensor_record_training_dataset import required_training_coverage
from morai_pretraining.sensor_record_manifest import route_permission_metadata


class TrafficLightCollectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sources = self.root / 'producer'
        for relative in signal_collection.SOURCE_FILES:
            path = self.sources / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('수집 당시 테스트 소스: ' + relative, encoding='utf-8')
        self.contract = dict(schema=signal_collection.CONTRACT_SCHEMA, source_sha256={
            relative: hashlib.sha256((self.sources / relative).read_bytes()).hexdigest()
            for relative in signal_collection.SOURCE_FILES})
        self.contract.update(copy.deepcopy(signal_collection.BEHAVIOR_POLICY))
        self.cfg = dict(mission_mode=signal_collection.MODE, drive_control_mode='aimdrive_expert',
                        traffic_light_wait_override=dict(enabled=False), _traffic_light_collection=self.contract)
        target = signal_collection.TARGET_LIGHT
        def observation(when, progress, speed, phase, color):
            return dict(scenario_time=when, source_ns=int((when + 40.) * 1e9), speed_mps=speed,
                        route_s=progress, phase=phase, signal_id=target, signal_color=color,
                        ego_xy=[signal_collection.START_XY[0] + progress, signal_collection.START_XY[1]])
        observations = [observation(-47.5 + n * .25, n * .5, 2., 'red_approach', 1) for n in range(190)]
        observations += [observation(t, 95., 0., 'red_hold', 1) for t in (0., 1., 2., 3.)]
        observations += [observation(t, 95., 0., 'straight_hold', 16) for t in (3.5, 4.5, 5.5, 6.5, 7.5)]
        observations += [observation(7.5 + n * .25, 95. + n * .5, 2.,
                                     'completed' if n >= 70 else 'left_go', 33) for n in range(1, 121)]
        self.report = dict(
            schema=signal_collection.REPORT_SCHEMA, status='passed', phase='completed', contract=self.contract,
            scenario_clock='grpc_simulation_timestamp',
            route=dict(start_global_index=1031, end_global_index=1348, left_stop_global_index=1245,
                       left_exit_global_index=1297, left_stop_node_id='A1256W000742',
                       left_stop_signal_id=target, left_stop_route_s_m=105.05766, left_exit_route_s_m=130.293,
                       right_exit_global_index=1185, right_exit_route_s_m=77.),
            red_stop_verified=True, straight_green_stop_verified=True,
            left_arrow_departure_verified=True, left_turn_completed=True,
            empty_road_verified=True, maximum_blocking_actors=0, last_blocking_actors=[],
            actor_inventory_checks=20, collision_count=0,
            expert_runtime_verified=True,
            start_pose_verified=True, right_turn_completed=True,
            startup_observation=observation(-48., 0., 0., 'startup', None),
            final_stop=dict(final_verified=True, policy='common_auto_d_ros_brake', velocity_reset_used=False),
            observations=observations,
            commands=[dict(id=target, target_color=color, before=dict(id=target, color=before),
                           after=dict(id=target, color=color), verified=True, accepted=True,
                           set_sibling=False, impulse=False, scenario_time=t, route_s=95.,
                           command_wall_monotonic=99.9 + t,
                           readback_stable_seconds=.2,
                           readback_observations=[dict(wall_monotonic=100. + t + delta, id=target, color=color)
                                                  for delta in (0., .1, .2)])
                      for color, before, t in ((1, 16, -48.5), (16, 1, 3.5), (33, 16, 7.5))],
            restore=dict(id=target, verified=True, accepted=True, impulse=False, set_sibling=False, original_color=16,
                         command_wall_monotonic=149.9,
                         scope='exact_signal_color_only', automatic_schedule_restored=False,
                         readback_stable_seconds=.2,
                         readback_observations=[dict(wall_monotonic=150. + delta, id=target, color=16)
                                                for delta in (0., .1, .2)],
                         original=dict(id=target, color=16), after=dict(id=target, color=16)))
        self.episode = self.root / 'episode_000'
        (self.episode / 'raw').mkdir(parents=True)
        (self.episode / 'expert_runtime').mkdir()
        self.runtime_path = self.episode / 'expert_runtime/config.json'
        self.runtime_path.write_text(json.dumps(dict(planning=dict(velocity_profile=dict(max_velocity=30.,
                                                                                       speed_limit_zones=[])))),
                                     encoding='utf-8')
        self.runtime_sha = hashlib.sha256(self.runtime_path.read_bytes()).hexdigest()
        self.report['expert_runtime_config_sha256'] = self.runtime_sha
        self.acquisition = dict(traffic_light_target=signal_collection.COLOR_TARGET,
                                traffic_light_permission_target=signal_collection.PERMISSION_POLICY)
        acquisition_path = self.root / 'sensor_contract.json'
        acquisition_path.write_text(json.dumps(self.acquisition), encoding='utf-8')
        self.outcome = dict(status='mission_completed', full_mission_completed=True)
        self.context = SimpleNamespace(cfg=self.cfg, output=self.episode, run_seed=123,
                                       traffic_light_report=self.report, contract_path=acquisition_path)
        self.write('episode.json', dict(scenario_name=signal_collection.MODE, seed=123, scenario_config=self.cfg,
                                       scenario_clock='grpc_simulation_timestamp',
                                       expert_inputs={'expert_runtime/config.json': self.runtime_sha}))
        self.write('outcome.json', self.outcome)
        self.write('raw/RAW_COMPLETE.json', dict(status='completed'))

    def write(self, relative, document):
        (self.episode / relative).write_text(json.dumps(document), encoding='utf-8')

    def finalize(self):
        with patch.object(signal_collection, 'PROJECT', self.sources):
            signal_collection.finalize_traffic_light_collection(self.context, self.outcome)

    def validate(self):
        return signal_collection.validate_traffic_light_collection(self.episode, signal_collection.MODE)

    def test_completed_outcome_without_capture_proof_is_rejected(self):
        self.assertFalse(self.validate()['valid'])

    def test_full_evidence_accepts_capture_snapshots_without_current_sources(self):
        self.finalize()
        shutil.rmtree(self.sources)
        result = self.validate()
        self.assertTrue(result['valid'], result)
        self.assertTrue(result['historical_source_evidence_only'])
        self.assertFalse(result['manual_all_frame_camera_certification'])

    def test_verified_expert_config_must_match_the_actual_captured_bytes(self):
        self.finalize()
        self.runtime_path.write_text('{}', encoding='utf-8')
        result = self.validate()
        self.assertFalse(result['valid'])
        self.assertFalse(result['checks']['expert_runtime_input_identity'])

    def test_expert_report_hash_and_safe_single_config_path_are_required(self):
        metadata = json.loads((self.episode / 'episode.json').read_text())
        for inputs in ({}, {'../config.json': self.runtime_sha},
                       {'expert_runtime/config.json': self.runtime_sha, 'expert_runtime/other.json': self.runtime_sha}):
            modified = dict(metadata, expert_inputs=inputs)
            self.assertFalse(signal_collection._expert_runtime_matches(self.report, modified, self.episode))
        report = dict(self.report, expert_runtime_config_sha256='0' * 64)
        self.assertFalse(signal_collection._expert_runtime_matches(report, metadata, self.episode))

    def test_configuration_preserves_color_head_and_disables_legacy_override(self):
        cfg = dict(traffic_light_wait_override=dict(enabled=True))
        with patch.object(signal_collection, 'collection_contract', return_value=self.contract):
            configure_collection_scenario(cfg, signal_collection.MODE)
        self.assertEqual(cfg['_traffic_light_collection'], self.contract)
        self.assertFalse(cfg['traffic_light_wait_override']['enabled'])
        self.assertEqual(cfg['drive_control_mode'], 'aimdrive_expert')
        self.assertFalse(cfg['sync_runtime']['enabled'])

    def test_old_startpoint_or_temporary_signal_contract_is_not_admitted_as_the_new_plan(self):
        for key, value in (('start_global_index', 1081), ('impulse', True)):
            config = copy.deepcopy(self.cfg)
            config['_traffic_light_collection'][key] = value
            self.assertFalse(signal_collection._contract_checks(config)['declared_behavior'])

    def test_actual_runner_contract_keeps_the_collector_source_pins(self):
        from scenario_runner.scenarios.segment_modes.traffic_light import configure
        cfg = {}
        configure_collection_scenario(cfg, signal_collection.MODE)
        captured = copy.deepcopy(cfg['_traffic_light_collection'])
        configure(cfg)
        self.assertEqual(cfg['_traffic_light_collection'], captured)
        self.assertTrue(all(signal_collection._contract_checks(cfg).values()))

    def test_new_coverage_requires_both_color_and_permission_classes_without_objects(self):
        self.assertEqual(required_training_coverage(signal_collection.MODE),
                         ('waypoint_frames', 'traffic_light_0', 'traffic_light_1',
                          'route_permission_0', 'route_permission_1'))
        self.assertNotIn('object_positive_frames', required_training_coverage(signal_collection.MODE))

    def test_report_flags_alone_cannot_replace_the_physical_hold_and_departure(self):
        for field in ('red_stop_verified', 'straight_green_stop_verified', 'left_arrow_departure_verified', 'left_turn_completed'):
            with self.subTest(field=field):
                report = copy.deepcopy(self.report)
                report[field] = False
                self.assertFalse(signal_collection._mission_checks(report, self.contract)['completed_behavior'])
        report = copy.deepcopy(self.report)
        report['observations'] = []
        checks = signal_collection._mission_checks(report, self.contract)
        for key in ('red_physical_stop', 'straight_green_physical_stop', 'permitted_left_departure', 'left_exit_observed'):
            self.assertFalse(checks[key])

    def test_unverified_expert_runtime_or_final_physical_stop_cannot_pass(self):
        for key, value, check in (('expert_runtime_verified', False, 'expert_runtime_verified'),
                                  ('final_stop', {}, 'final_physical_stop'),
                                  ('final_stop', dict(final_verified=True, policy='set_velocity_zero', velocity_reset_used=True),
                                   'final_physical_stop')):
            report = copy.deepcopy(self.report)
            report[key] = value
            self.assertFalse(signal_collection._mission_checks(report, self.contract)[check])

    def test_startup_right_turn_and_continuous_motion_are_required(self):
        report = copy.deepcopy(self.report)
        report['start_pose_verified'] = False
        self.assertFalse(signal_collection._mission_checks(report, self.contract)['actual_start_pose'])
        report = copy.deepcopy(self.report)
        report['startup_observation']['ego_xy'][0] += 25.
        self.assertFalse(signal_collection._mission_checks(report, self.contract)['actual_start_pose'])
        report = copy.deepcopy(self.report)
        report['right_turn_completed'] = False
        self.assertFalse(signal_collection._mission_checks(report, self.contract)['right_turn_observed'])
        for field, value in (('route_s', 25.), ('ego_xy', [0., 0.]), ('scenario_time', -100.)):
            report = copy.deepcopy(self.report)
            report['observations'][1][field] = value
            self.assertFalse(signal_collection._mission_checks(report, self.contract)['observation_motion_continuity'])

    def test_short_holds_creep_wrong_ids_and_gaps_are_rejected(self):
        red = [i for i, r in enumerate(self.report['observations']) if r['phase'] == 'red_hold']
        green = [i for i, r in enumerate(self.report['observations']) if r['phase'] == 'straight_hold']
        cases = [
            ('red_physical_stop', lambda rows: rows[red[-1]].update(scenario_time=2.5)),
            ('red_physical_stop', lambda rows: rows[red[1]].update(speed_mps=.7)),
            ('straight_green_physical_stop', lambda rows: rows[green[2]].update(signal_id='다른_신호')),
            ('straight_green_physical_stop', lambda rows: rows[green[2]].update(signal_color=36)),
            ('straight_green_physical_stop', lambda rows: rows[green[2]].update(route_s=106.)),
        ]
        for check, mutate in cases:
            with self.subTest(check=check):
                report = copy.deepcopy(self.report)
                mutate(report['observations'])
                self.assertFalse(signal_collection._mission_checks(report, self.contract)[check])

    def test_commands_require_exact_selected_id_and_actual_readback(self):
        cases = [('id', '다른_신호'), ('verified', False), ('after', {}),
                 ('after', dict(id='다른_신호', color=33)),
                 ('after', dict(id=signal_collection.TARGET_LIGHT, color=36)),
                 ('target_color', 36)]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                report = copy.deepcopy(self.report)
                report['commands'][2][key] = value
                self.assertFalse(signal_collection._mission_checks(report, self.contract)['exact_signal_commands'])
        for color in (32, 33, 48):
            report = copy.deepcopy(self.report)
            report['commands'][2]['after']['color'] = color
            for observation in report['commands'][2]['readback_observations']:
                observation['color'] = color
            self.assertTrue(signal_collection._mission_checks(report, self.contract)['exact_signal_commands'])

    def test_delayed_readback_is_accepted_only_after_actual_stable_samples(self):
        report = copy.deepcopy(self.report)
        record = report['commands'][0]
        record['readback_observations'].insert(0, dict(wall_monotonic=51.45, id=signal_collection.TARGET_LIGHT, color=16))
        self.assertTrue(signal_collection._mission_checks(report, self.contract)['exact_signal_commands'])
        for row in record['readback_observations'][-2:]:
            row['color'] = 16
        self.assertFalse(signal_collection._mission_checks(report, self.contract)['exact_signal_commands'])

    def test_ack_temporary_command_short_or_discontinuous_readback_cannot_pass(self):
        target = signal_collection.TARGET_LIGHT
        cases = [
            ('impulse', True), ('readback_observations', []),
            ('readback_stable_seconds', .1),
            ('readback_observations', [dict(wall_monotonic=1. + d, id=target, color=1) for d in (0., .1, .15)]),
            ('readback_observations', [dict(wall_monotonic=1. + d, id=target, color=c)
                                       for d, c in ((0., 1), (.1, 16), (.2, 1))]),
            ('readback_observations', [dict(wall_monotonic=1. + d, id=target, color=1)
                                       for d in (0., .2, .1)]),
            ('readback_observations', [dict(wall_monotonic=1. + d, id=target, color=1)
                                       for d in (0., 2., 4.)]),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                report = copy.deepcopy(self.report)
                report['commands'][0][key] = value
                self.assertFalse(signal_collection._mission_checks(report, self.contract)['exact_signal_commands'])

    def test_restore_requires_its_own_stable_color_and_honest_restoration_scope(self):
        for key, value in (('readback_observations', []), ('accepted', False), ('impulse', True),
                           ('scope', 'complete_simulator_restore'), ('automatic_schedule_restored', True)):
            report = copy.deepcopy(self.report)
            report['restore'][key] = value
            self.assertFalse(signal_collection._mission_checks(report, self.contract)['signal_restored'])

    def test_collision_foreign_actor_or_incomplete_restore_cannot_pass(self):
        for key, value, check in (('collision_count', 1, 'collision_free'),
                                  ('collision_count', False, 'collision_free'),
                                  ('maximum_blocking_actors', 1, 'empty_road'),
                                  ('last_blocking_actors', ['foreign'], 'empty_road'),
                                  ('actor_inventory_checks', 1, 'empty_road'),
                                  ('restore', {}, 'signal_restored')):
            with self.subTest(key=key):
                report = copy.deepcopy(self.report)
                report[key] = value
                self.assertFalse(signal_collection._mission_checks(report, self.contract)[check])
        report = copy.deepcopy(self.report)
        report['restore']['after']['color'] = 1
        self.assertFalse(signal_collection._mission_checks(report, self.contract)['signal_restored'])

    def test_incomplete_raw_or_failed_mission_is_not_sealed_as_completed(self):
        self.write('raw/RAW_COMPLETE.json', dict(status='interrupted'))
        self.finalize()
        self.assertFalse(self.validate()['valid'])
        shutil.rmtree(self.episode / 'traffic_light')
        self.write('raw/RAW_COMPLETE.json', dict(status='completed'))
        self.outcome['status'] = 'failed'
        self.finalize()
        self.assertFalse(self.validate()['valid'])

    def test_source_snapshot_or_report_tampering_is_rejected(self):
        self.finalize()
        proof = json.loads((self.episode / signal_collection.PROOF_RELATIVE).read_text())
        source = proof['files']['source:' + signal_collection.SOURCE_FILES[0]]['path']
        (self.episode / source).write_text('변경된 소스', encoding='utf-8')
        self.assertFalse(self.validate()['valid'])

    def test_color_head_repurposing_or_changed_sensor_contract_is_rejected(self):
        self.acquisition['traffic_light_target'] = 'route_permission_stop_go'
        self.context.contract_path.write_text(json.dumps(self.acquisition), encoding='utf-8')
        self.finalize()
        self.assertFalse(self.validate()['valid'])
        shutil.rmtree(self.episode / 'traffic_light')
        self.acquisition['traffic_light_target'] = signal_collection.COLOR_TARGET
        self.context.contract_path.write_text(json.dumps(self.acquisition), encoding='utf-8')
        self.finalize()
        self.context.contract_path.write_text('{}', encoding='utf-8')
        self.assertFalse(self.validate()['valid'])

    def test_hold_clock_must_match_the_captured_scenario_clock(self):
        self.report['scenario_clock'] = 'host_monotonic'
        self.finalize()
        result = self.validate()
        self.assertFalse(result['valid'])
        self.assertFalse(result['checks']['scenario_clock_identity'])

    def test_forged_report_flags_with_updated_hash_still_fail_semantic_checks(self):
        self.finalize()
        proof = json.loads((self.episode / signal_collection.PROOF_RELATIVE).read_text())
        self.report['observations'] = []
        self.write('traffic_light/mission_report.json', self.report)
        self.outcome['traffic_light_mission']['report'] = self.report
        self.write('outcome.json', self.outcome)
        for role in ('mission_report', 'outcome'):
            entry = proof['files'][role]
            entry['sha256'] = hashlib.sha256((self.episode / entry['path']).read_bytes()).hexdigest()
        self.write(signal_collection.PROOF_RELATIVE, proof)
        self.assertFalse(self.validate()['valid'])

    def test_other_scenarios_do_not_acquire_a_signal_guard(self):
        self.assertEqual(signal_collection.validate_traffic_light_collection(self.episode, 'full_loop_static'),
                         dict(required=False, valid=True))

    def test_entrypoint_finalizes_only_after_run_and_archives_the_guard_source(self):
        source = (INTERNAL / 'tools/run_proven_three_lap_scenario.py').read_text()
        self.assertLess(source.index('result = scenario.run()'), source.index('finalize_traffic_light_collection(scenario, result)'))
        self.assertIn("ROOT / 'async_capture/traffic_light_collection.py'", source)
        ast.parse(source)
        shell = (INTERNAL / 'tools/collect_scenario_laps.sh').read_text()
        self.assertIn('full_loop_traffic_light) SCENARIO_FOLDER="traffic_light"', shell)

    def test_actual_capture_wrapper_keeps_new_mode_speed_and_other_modes_unchanged(self):
        source = ast.parse((INTERNAL / 'tools/run_sensor_record_scenario.py').read_text())
        branch = next(n for n in ast.walk(source) if isinstance(n, ast.If)
                      and isinstance(n.test, ast.Compare)
                      and isinstance(n.test.left, ast.Attribute) and n.test.left.attr == 'scenario'
                      and any(isinstance(v, ast.Constant) and v.value == signal_collection.MODE
                              for v in n.test.comparators))
        compiled = compile(ast.Module(body=[branch], type_ignores=[]), '<수집 wrapper 속도 회귀>', 'exec')
        for scenario, cap in ((signal_collection.MODE, '30'), ('full_loop_dynamic', '60'),
                              ('full_loop_tunnel_normal', '60'), ('full_loop_tunnel_static', None)):
            namespace = dict(a=SimpleNamespace(scenario=scenario), command=[],
                             TUNNEL_STATIC_SCENARIO='full_loop_tunnel_static')
            exec(compiled, namespace)
            self.assertEqual(namespace['command'], [] if cap is None else ['--max-speed-kmh', cap])

    def test_raw_constructor_preserves_the_signal_mode_limits(self):
        spec = importlib.util.spec_from_file_location('신호수집기_회귀', INTERNAL / 'tools/run_proven_three_lap_scenario.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)

        def base_init(context, *args, **kwargs):
            context.cfg = dict(mission_mode=signal_collection.MODE,
                               progress_watchdog=dict(enabled=True, timeout_sec=45.), off_route_timeout_sec=3.)

        with patch.object(runner.FullLoopScenario, '__init__', base_init):
            scenario = runner.RawScenario(output=self.root / 'constructor_episode',
                                          contract_path=self.context.contract_path, seconds=180.)
        self.assertEqual(scenario.cfg['progress_watchdog']['timeout_sec'], 45.)
        self.assertEqual(scenario.cfg['off_route_timeout_sec'], 3.)

    def test_raw_setup_preserves_the_validated_signal_config_bytes(self):
        source = ast.parse((INTERNAL / 'tools/run_proven_three_lap_scenario.py').read_text())
        raw = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'RawScenario')
        setup = next(n for n in raw.body if isinstance(n, ast.FunctionDef) and n.name == 'setup_expert_controller')
        start = next(i for i, n in enumerate(setup.body) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'runtime_path')
        end = next(i for i, n in enumerate(setup.body) if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'expert')
        compiled = compile(ast.Module(body=setup.body[start:end], type_ignores=[]), '<실제 Expert 설정 보존 회귀>', 'exec')
        original = self.runtime_path.read_bytes()
        for mode in (signal_collection.MODE, 'full_loop_dynamic'):
            self.runtime_path.write_bytes(original)
            context = SimpleNamespace(cfg=dict(mission_mode=mode),
                                      expert_controller=SimpleNamespace(runtime_config_path=self.runtime_path))
            exec(compiled, dict(self=context, Path=Path, json=json))
            self.assertEqual(self.runtime_path.read_bytes() == original, mode == signal_collection.MODE)
        runtime = json.loads(original)
        runtime['planning']['velocity_profile']['max_velocity'] = 60.
        self.runtime_path.write_text(json.dumps(runtime), encoding='utf-8')
        context.cfg['mission_mode'] = signal_collection.MODE
        with self.assertRaisesRegex(ValueError, '속도 계약'):
            exec(compiled, dict(self=context, Path=Path, json=json))

    def mission_wrapper(self, restore_verified, capture_error=None, restore_error=None, finish_error=None):
        trace = []
        self.last_wrapper_trace = trace

        class RawBase:
            def run(context):
                trace.append('raw_closed')
                if capture_error is not None:
                    raise capture_error
                return dict(status='mission_completed', full_mission_completed=True)

        class MissionBase:
            def run(context):
                raise AssertionError('이 MRO에서 MissionScenario.run이 호출되면 안 됩니다')

        namespace = dict(RawScenario=RawBase, MissionScenario=MissionBase)
        source = ast.parse((INTERNAL / 'tools/run_proven_three_lap_scenario.py').read_text())
        wrapper = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'RawMissionScenario')
        exec(compile(ast.Module(body=[wrapper], type_ignores=[]), '<수집 MRO 회귀>', 'exec'), namespace)
        context = namespace['RawMissionScenario']()
        context.cfg = dict(mission_mode=signal_collection.MODE)
        context.traffic_light_report = dict(original=dict(id=signal_collection.TARGET_LIGHT, color=16),
                                            restore=dict(verified=False))
        context.event = lambda name, **fields: trace.append(name)

        def restore(actual):
            trace.append('restore')
            if restore_error is not None:
                raise restore_error
            actual.traffic_light_report['restore']['verified'] = restore_verified

        def finish(actual, status, error):
            trace.append('restore_failed_status')
            if finish_error is not None:
                raise finish_error
            actual.traffic_light_report['status'] = status

        runner = SimpleNamespace(restore=restore, finish=finish)
        with patch('scenario_runner.scenarios.segment_modes.traffic_light', runner, create=True):
            result = context.run()
        return trace, result

    def test_collection_mro_restores_the_signal_after_raw_close(self):
        trace, result = self.mission_wrapper(True)
        self.assertEqual(trace, ['raw_closed', 'restore', 'traffic_light_restore_verified'])
        self.assertEqual(result['status'], 'mission_completed')

    def test_collection_mro_preserves_failed_restore_as_the_capture_failure(self):
        trace, result = self.mission_wrapper(False)
        self.assertEqual(trace[:2], ['raw_closed', 'restore'])
        self.assertEqual(result['status'], 'failed')
        self.assertIn('traffic_light_restore_failed', result['reason'])

    def test_cleanup_file_error_still_attempts_restoration_and_preserves_original_error(self):
        with self.assertRaisesRegex(ValueError, 'outcome 기록 실패'):
            self.mission_wrapper(True, capture_error=ValueError('outcome 기록 실패'))
        self.assertEqual(self.last_wrapper_trace[:2], ['raw_closed', 'restore'])

    def test_cleanup_and_restore_logging_errors_are_preserved_after_restore_attempt(self):
        original = ValueError('outcome 기록 실패')
        with self.assertRaises(RuntimeError) as caught:
            self.mission_wrapper(False, capture_error=original,
                                 restore_error=RuntimeError('원신호 복원 실패'),
                                 finish_error=RuntimeError('복원 보고서 저장 실패'))
        self.assertIs(caught.exception.__cause__, original)
        for reason in ('outcome 기록 실패', '원신호 복원 실패', '복원 보고서 저장 실패'):
            self.assertIn(reason, str(caught.exception))
        self.assertEqual(self.last_wrapper_trace[:2], ['raw_closed', 'restore'])


class SeparatePermissionTests(unittest.TestCase):
    policy = dict(schema='morai_route_movement_permission_v1', light_id=signal_collection.TARGET_LIGHT,
                  movement='left', separate_from_color_head=True, unsupported_phases_masked=True,
                  raw_phase_permission={'1': False, '16': False, '32': True, '33': True, '48': True})

    def row(self, color):
        progress = SimpleNamespace(route_index=10, route_link_index=1, current_link_id='approach',
            distance_to_route_m=0., affected_light_id=signal_collection.TARGET_LIGHT,
            distance_to_stopline_m=12., signal_movement='left', signal_route_link_id='left', signal_stop_node_id='stop')
        return bind_route_progress(progress, 0., 0.,
            [dict(id=signal_collection.TARGET_LIGHT, color=color, source_record_ids=[10, 11], source_bracket_ns=[100, 200])],
            [signal_collection.TARGET_LIGHT])

    def test_color_classes_are_unchanged_while_left_permission_is_separate(self):
        admission = dict(traffic_light_gt_transport_timing_candidate=True, route_signal_permission_candidate=True)
        for color, color_target, permission in ((1, 0, False), (16, 1, False),
                                               (32, -1, True), (33, -1, True), (48, -1, True), (36, -1, None)):
            with self.subTest(color=color):
                row = self.row(color)
                sidecar = route_permission_metadata(row, admission, self.policy)
                self.assertEqual(row['training_label'], color_target)
                self.assertEqual(sidecar['target'], permission)
                self.assertEqual(sidecar['valid'], permission is not None)
                self.assertTrue(sidecar['separate_from_color_head'])

    def test_wrong_signal_transport_or_permission_disagreement_stays_masked(self):
        for mutate, admission in ((lambda row: row.update(light_id='다른_신호'),
                                  dict(traffic_light_gt_transport_timing_candidate=True, route_signal_permission_candidate=True)),
                                 (lambda row: row.update(signal_permission=True),
                                  dict(traffic_light_gt_transport_timing_candidate=True, route_signal_permission_candidate=True)),
                                 (lambda row: None,
                                  dict(traffic_light_gt_transport_timing_candidate=False, route_signal_permission_candidate=True))):
            row = self.row(16)
            mutate(row)
            self.assertFalse(route_permission_metadata(row, admission, self.policy)['valid'])

    def test_unknown_permission_policy_fails_instead_of_changing_color_head(self):
        with self.assertRaises(ValueError):
            route_permission_metadata(self.row(16), {}, dict(self.policy, separate_from_color_head=False))


if __name__ == '__main__':
    unittest.main()
