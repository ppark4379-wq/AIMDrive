"""좌회전 신호 실험의 실제 정지·출발과 수집 당시 근거를 별도로 봉인한다."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path


MODE = 'full_loop_traffic_light'
CONTRACT_SCHEMA = 'morai_traffic_light_collection_contract_v1'
REPORT_SCHEMA = 'morai_traffic_light_mission_v1'
PROOF_SCHEMA = 'morai_traffic_light_collection_complete_v1'
PROOF_RELATIVE = Path('traffic_light/COLLECTION_TRAFFIC_LIGHT_COMPLETE.json')
PROJECT = Path(__file__).resolve().parents[3]
TARGET_LIGHT = 'C1256W000052'
COLOR_TARGET = 'pure red=0 / pure green=1; mixed yellow/arrow phases masked'
PERMISSION_POLICY = dict(schema='morai_route_movement_permission_v1', light_id=TARGET_LIGHT,
                        movement='left', separate_from_color_head=True,
                        raw_phase_permission={'1': False, '16': False, '32': True, '33': True, '48': True},
                        unsupported_phases_masked=True)
BEHAVIOR_POLICY = dict(scenario_name=MODE, target_light_id=TARGET_LIGHT,
                       command_colors=[1, 16, 33], set_sibling=False, impulse=False,
                       readback_timeout_wall_seconds=3., readback_stable_wall_seconds=.2,
                       readback_minimum_reads=3,
                       red_hold_seconds=3., straight_green_hold_seconds=4.,
                       stopped_speed_mps=.3, stop_window_m=[5., 30.],
                       start_global_index=1031, end_global_index=1348,
                       left_stop_global_index=1245, left_exit_global_index=1297,
                       color_head_target_policy='pure_red_green_v1', separate_route_permission_gt=True)
START_XY = (-60.53983283180065, 33.51020694520611)
SOURCE_FILES = (
    'scenario_runner/scenarios/segment_modes/traffic_light.py',
    'scenario_runner/scenarios/segment_modes/scenario.py',
    'scenario_runner/scenarios/segment_modes/modes.py',
    'scenario_runner/scenarios/segment_modes/route.py',
    'scenario_runner/scenarios/segment_modes/events.py',
    'data_collection/internal_sensor_record/scenario_collection.py',
    'data_collection/internal_sensor_record/tools/run_proven_three_lap_scenario.py',
    'data_collection/internal_sensor_record/async_capture/traffic_light_collection.py',
    'scenario_runner/adapters/morai_grpc.py',
    'data_collection/third_party/grpc_inha_univ/src/api/simulation_world.py',
    'data_collection/third_party/grpc_inha_univ/src/proto/sim_adapter.py',
    'data_collection/third_party/grpc_inha_univ/src/proto/morai/infrastructure/traffic_light_pb2.py',
    'data_collection/third_party/grpc_inha_univ/src/proto/morai/infrastructure/infrastructure_enum_pb2.py',
    'data_collection/third_party/grpc_inha_univ/src/proto/morai/common/enum_pb2.py',
    'data_collection/third_party/grpc_inha_univ/src/proto/morai/common/result_pb2.py',
    'data_collection/third_party/grpc_inha_univ/morai_standard_proto/morai/infrastructure/traffic_light.proto',
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, document):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(document, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def collection_contract():
    """Runner의 행동 계약을 유지하면서 수집기 소스까지 함께 고정한다."""
    from scenario_runner.scenarios.segment_modes.traffic_light import default_contract
    contract = copy.deepcopy(default_contract())
    if contract.get('schema') != CONTRACT_SCHEMA:
        raise ValueError('좌회전 신호 Runner 계약 버전이 지원되지 않습니다')
    pins = contract.setdefault('source_sha256', {})
    pins.update({relative: _sha(PROJECT / relative) for relative in SOURCE_FILES})
    return contract


def collection_evidence(context):
    if context.cfg.get('mission_mode') != MODE:
        return None
    report = getattr(context, 'traffic_light_report', {})
    return copy.deepcopy(dict(scenario_name=MODE, seed=getattr(context, 'run_seed', None),
                              completed=report.get('status') == 'passed', report=report,
                              color_head_target_policy='pure_red_green_v1',
                              permission_gt_schema='morai_route_movement_permission_v1'))


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _contract_checks(config):
    contract = config.get('_traffic_light_collection', {})
    pins = contract.get('source_sha256', {})
    return dict(
        contract_schema=contract.get('schema') == CONTRACT_SCHEMA,
        declared_behavior=all(contract.get(key) == expected for key, expected in BEHAVIOR_POLICY.items()),
        collection_mode=config.get('mission_mode') == MODE,
        expert_control=config.get('drive_control_mode') == 'aimdrive_expert',
        legacy_override_disabled=config.get('traffic_light_wait_override', {}).get('enabled') is False,
        captured_source_set=isinstance(pins, dict) and set(SOURCE_FILES).issubset(pins)
            and all(isinstance(name, str) and not Path(name).is_absolute() and '..' not in Path(name).parts
                    and isinstance(sha, str) and len(sha) == 64
                    and all(c in '0123456789abcdef' for c in sha) for name, sha in pins.items()),
    )


def _target_policy_checks(acquisition):
    return dict(legacy_color_head_target=acquisition.get('traffic_light_target') == COLOR_TARGET,
                separate_permission_policy=acquisition.get('traffic_light_permission_target') == PERMISSION_POLICY)


def _clock_matches(report, metadata):
    clock = report.get('scenario_clock', report.get('clock_domain'))
    return isinstance(clock, str) and bool(clock) and clock == metadata.get('scenario_clock')


def _expert_runtime_matches(report, metadata, episode):
    """Runner의 설정 검증 뒤에 실제 Expert 설정이 바뀌지 않았는지 대조한다."""
    from .sensor_record_admission import _safe_local_file
    inputs = metadata.get('expert_inputs')
    if not isinstance(inputs, dict):
        return False
    configs = [(name, expected) for name, expected in inputs.items()
               if isinstance(name, str) and Path(name).suffix == '.json']
    if len(configs) != 1:
        return False
    relative, expected = configs[0]
    path = Path(episode) / relative
    return (not Path(relative).is_absolute() and _safe_local_file(path, Path(episode))
            and isinstance(expected, str) and len(expected) == 64
            and _sha(path) == expected == report.get('expert_runtime_config_sha256'))


def _holding_span(observations, color, seconds, stop_s):
    """낮은 속도가 실제 신호와 같은 시나리오 시계에서 연속 유지됐는지 확인한다."""
    longest, start, previous = 0., None, None
    for row in observations:
        stamp, speed, route_s = (row.get(key) for key in ('scenario_time', 'speed_mps', 'route_s'))
        valid = (all(_finite(v) for v in (stamp, speed, route_s))
                 and row.get('signal_color') == color and row.get('signal_id') == TARGET_LIGHT
                 and 0. <= speed <= .3 and 5. <= stop_s - route_s <= 30.
                 and row.get('phase') in ('red_approach', 'red_hold', 'straight_hold'))
        if not valid:
            start, previous = None, None
            continue
        if start is None or previous is None or stamp < previous or stamp - previous > 1.5:
            start = stamp
        longest = max(longest, stamp - start)
        previous = stamp
    return longest >= seconds - .05


def _observation_valid(row):
    xy = row.get('ego_xy')
    return (all(_finite(row.get(key)) for key in ('scenario_time', 'speed_mps', 'route_s'))
            and row['speed_mps'] >= 0. and isinstance(xy, (list, tuple)) and len(xy) == 2
            and all(_finite(value) for value in xy))


def _motion_continuity(startup, observations):
    """시간·진행량·실제 위치가 갑자기 점프한 보고서는 행동 근거로 인정하지 않는다."""
    if not isinstance(startup, dict) or not _observation_valid(startup) or not observations:
        return False
    previous = startup
    for row in observations:
        if not _observation_valid(row):
            return False
        dt = row['scenario_time'] - previous['scenario_time']
        distance = math.dist(row['ego_xy'], previous['ego_xy'])
        progress = row['route_s'] - previous['route_s']
        maximum = 3. + 1.5 * max(previous['speed_mps'], row['speed_mps']) * dt
        if dt < 0. or dt > 1.5 or progress < -1. or abs(progress) > maximum or distance > maximum:
            return False
        previous = row
    return True


def _stable_signal_readback(record, accepted_colors):
    """ACK 이후 마지막 연속 세 번 이상 관측이 0.2초 유지된 경우만 인정한다."""
    rows = record.get('readback_observations')
    duration = record.get('readback_stable_seconds')
    issued = record.get('command_wall_monotonic')
    if (record.get('impulse') is not False or not isinstance(rows, list) or not rows
            or not _finite(duration) or duration < .2 - 1e-8 or not _finite(issued)):
        return False
    previous, start, count = None, None, 0
    for row in rows:
        if not isinstance(row, dict) or not _finite(row.get('wall_monotonic')):
            return False
        stamp = row['wall_monotonic']
        if stamp < issued or stamp - issued > 3. + 1e-6 or (previous is not None and stamp < previous):
            return False
        if (row.get('id') != TARGET_LIGHT or type(row.get('color')) is not int
                or row['color'] not in accepted_colors or row.get('read_error')):
            start, count = None, 0
        else:
            if start is None:
                start = stamp
            count += 1
        previous = stamp
    return (count >= 3 and start is not None and previous - start >= .2 - 1e-8
            and rows[-1]['wall_monotonic'] - rows[0]['wall_monotonic'] <= 3. + 1e-6)


def _mission_checks(report, contract):
    observations = report.get('observations') or []
    commands = report.get('commands') or []
    route = report.get('route') or {}
    stop_s, exit_s = route.get('left_stop_route_s_m'), route.get('left_exit_route_s_m')
    right_s = route.get('right_exit_route_s_m')
    startup = report.get('startup_observation') or {}
    route_valid = (route.get('start_global_index') == 1031 and route.get('end_global_index') == 1348
                   and route.get('left_stop_global_index') == 1245 and route.get('left_exit_global_index') == 1297
                   and route.get('left_stop_node_id') == 'A1256W000742'
                   and route.get('left_stop_signal_id') == TARGET_LIGHT
                   and _finite(stop_s) and _finite(exit_s) and 0. < stop_s < exit_s)
    command_rows = [r for r in commands if r.get('target_color') in (1, 16, 33)
                    and r.get('phase') != 'restore']
    command_valid = len(command_rows) == 3
    if command_valid:
        previous = -math.inf
        for command, target in zip(command_rows, (1, 16, 33)):
            before, after = command.get('before') or {}, command.get('after') or {}
            stamp = command.get('scenario_time')
            accepted = (1,) if target == 1 else (16,) if target == 16 else (32, 33, 48)
            command_valid = command_valid and (command.get('id') == TARGET_LIGHT
                and command.get('target_color') == target and command.get('verified') is True
                and command.get('accepted') is True and command.get('set_sibling') is False
                and before.get('id') == TARGET_LIGHT and after.get('id') == TARGET_LIGHT
                and type(before.get('color')) is int and before['color'] >= 0
                and after.get('color') in accepted and _stable_signal_readback(command, accepted)
                and _finite(stamp) and stamp >= previous)
            if _finite(stamp):
                previous = stamp
    restore = report.get('restore') or {}
    restore_after = restore.get('after') or {}
    original = restore.get('original') or {}
    restored_color = restore.get('original_color', original.get('color'))
    restore_valid = (restore.get('verified') is True and restore.get('id', TARGET_LIGHT) == TARGET_LIGHT
                     and restore_after.get('id') == TARGET_LIGHT and isinstance(restored_color, int)
                     and not isinstance(restored_color, bool) and restore_after.get('color') == restored_color
                     and restore.get('accepted') is True and restore.get('set_sibling') is False
                     and _stable_signal_readback(restore, (restored_color,))
                     and restore.get('scope') == 'exact_signal_color_only'
                     and restore.get('automatic_schedule_restored') is False)
    if commands:
        restore_valid = restore_valid and restored_color == (commands[0].get('before') or {}).get('color')
    return dict(
        mission_schema=report.get('schema') == REPORT_SCHEMA,
        completed_behavior=report.get('status') == 'passed' and report.get('phase') == 'completed'
            and all(report.get(key) is True for key in (
                'red_stop_verified', 'straight_green_stop_verified', 'left_arrow_departure_verified', 'left_turn_completed')),
        report_contract_identity=report.get('contract') == contract,
        expert_runtime_verified=report.get('expert_runtime_verified') is True,
        final_physical_stop=report.get('final_stop', {}).get('final_verified') is True
            and report['final_stop'].get('policy') == 'common_auto_d_ros_brake'
            and report['final_stop'].get('velocity_reset_used') is False,
        selected_route=route_valid,
        actual_start_pose=report.get('start_pose_verified') is True and _observation_valid(startup)
            and 0. <= startup['speed_mps'] <= .3 and abs(startup['route_s']) <= 1.
            and math.dist(startup['ego_xy'], START_XY) <= 1.,
        observation_motion_continuity=_motion_continuity(startup, observations),
        right_turn_observed=report.get('right_turn_completed') is True
            and route.get('right_exit_global_index') == 1185 and _finite(right_s)
            and route_valid and 0. < right_s < stop_s
            and any(_finite(r.get('route_s')) and r['route_s'] >= right_s for r in observations),
        exact_signal_commands=command_valid and len(commands) == 3,
        red_physical_stop=route_valid and _holding_span(observations, 1, 3., stop_s),
        straight_green_physical_stop=route_valid and _holding_span(observations, 16, 4., stop_s),
        permitted_left_departure=route_valid and any(r.get('signal_id') == TARGET_LIGHT
            and r.get('signal_color') in (32, 33, 48) and r.get('phase') in ('left_go', 'completed')
            and _finite(r.get('speed_mps')) and r['speed_mps'] > .5
            and _finite(r.get('route_s')) and r['route_s'] > stop_s + .5 for r in observations),
        left_exit_observed=route_valid and any(r.get('phase') in ('left_go', 'completed')
            and _finite(r.get('route_s')) and r['route_s'] >= exit_s for r in observations),
        empty_road=report.get('empty_road_verified') is True
            and type(report.get('maximum_blocking_actors')) is int and report['maximum_blocking_actors'] == 0
            and report.get('last_blocking_actors') == []
            and isinstance(report.get('actor_inventory_checks'), int)
            and not isinstance(report['actor_inventory_checks'], bool) and report['actor_inventory_checks'] >= 2,
        collision_free=type(report.get('collision_count')) is int and report['collision_count'] == 0,
        signal_restored=restore_valid,
    )


def finalize_traffic_light_collection(context, outcome):
    """시나리오 정리와 raw 종료 후에만 새로운 수집 증거를 봉인한다."""
    if context.cfg.get('mission_mode') != MODE:
        return
    episode = Path(context.output).resolve()
    contract = context.cfg.get('_traffic_light_collection', {})
    report = getattr(context, 'traffic_light_report', None)
    if not isinstance(report, dict) or not all(_contract_checks(context.cfg).values()):
        raise ValueError('좌회전 신호 수집의 사전 계약 또는 실제 주행 보고서가 없습니다')
    directory = episode / 'traffic_light'
    directory.mkdir(exist_ok=False)
    report_path = directory / 'mission_report.json'
    _write(report_path, report)
    acquisition_path = directory / 'sensor_contract.json'
    acquisition_path.write_bytes(Path(context.contract_path).read_bytes())
    outcome['traffic_light_mission'] = collection_evidence(context)
    _write(episode / 'outcome.json', outcome)
    paths = dict(episode=episode / 'episode.json', outcome=episode / 'outcome.json',
                 raw_seal=episode / 'raw/RAW_COMPLETE.json', mission_report=report_path,
                 acquisition_contract=acquisition_path)
    for relative, expected in contract['source_sha256'].items():
        payload = (PROJECT / relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('좌회전 신호 수집 중 소스가 변경됐습니다: ' + relative)
        target = directory / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        paths['source:' + relative] = target
    checks = dict(_contract_checks(context.cfg), **_mission_checks(report, contract))
    checks.update(_target_policy_checks(_json(acquisition_path)))
    metadata = _json(paths['episode'])
    checks['scenario_clock_identity'] = _clock_matches(report, metadata)
    checks['expert_runtime_input_identity'] = _expert_runtime_matches(report, metadata, episode)
    checks['collection_completed'] = outcome.get('status') == 'mission_completed' and outcome.get('full_mission_completed') is True
    checks['raw_capture_closed'] = _json(paths['raw_seal']).get('status') == 'completed'
    _write(episode / PROOF_RELATIVE, dict(schema=PROOF_SCHEMA,
        status='completed' if all(checks.values()) else 'failed', scenario=MODE,
        seed=getattr(context, 'run_seed', None), checks=checks, source_sha256=contract['source_sha256'],
        files={role: dict(path=path.relative_to(episode).as_posix(), sha256=_sha(path)) for role, path in paths.items()},
        color_head_target_policy='pure_red_green_v1', permission_gt_schema='morai_route_movement_permission_v1',
        physical_semantic_validation_completed=False, manual_all_frame_camera_certification=False,
        native_assets_modified=False))


def validate_traffic_light_collection(episode, scenario_name):
    """현재 코드 실행 없이 수집 당시 snapshot과 실제 관측을 확인한다."""
    if scenario_name != MODE:
        return dict(required=False, valid=True)
    episode = Path(episode).resolve()
    path = episode / PROOF_RELATIVE
    checks, errors = {}, []
    try:
        from .sensor_record_admission import _safe_local_file
        if not _safe_local_file(path, episode):
            raise ValueError('좌회전 신호 수집 완료 증거가 없습니다')
        proof = _json(path)
        checks['final_proof'] = proof.get('schema') == PROOF_SCHEMA and proof.get('status') == 'completed' and proof.get('scenario') == MODE
        documents = {}
        for role in ('episode', 'outcome', 'raw_seal', 'mission_report', 'acquisition_contract'):
            entry = proof.get('files', {}).get(role, {})
            relative = Path(entry.get('path', ''))
            source = episode / relative
            if relative.is_absolute() or not _safe_local_file(source, episode) or _sha(source) != entry.get('sha256'):
                raise ValueError('좌회전 신호 수집의 봉인 파일이 없거나 변경됐습니다: ' + role)
            documents[role] = _json(source)
        metadata, outcome, report = (documents[key] for key in ('episode', 'outcome', 'mission_report'))
        config = metadata.get('scenario_config', {})
        contract = config.get('_traffic_light_collection', {})
        checks.update(_contract_checks(config))
        checks.update(_target_policy_checks(documents['acquisition_contract']))
        original_acquisition = episode.parent / 'sensor_contract.json'
        checks['acquisition_copy_identity'] = _safe_local_file(original_acquisition, episode.parent) \
            and _sha(original_acquisition) == proof['files']['acquisition_contract']['sha256']
        pins = contract.get('source_sha256', {})
        if not checks['captured_source_set'] or pins != proof.get('source_sha256'):
            raise ValueError('좌회전 신호 수집의 소스 계약이 일치하지 않습니다')
        for relative, expected in pins.items():
            entry = proof.get('files', {}).get('source:' + relative, {})
            source = episode / entry.get('path', '')
            if (Path(entry.get('path', '')).is_absolute() or not _safe_local_file(source, episode)
                    or entry.get('sha256') != expected or _sha(source) != expected):
                raise ValueError('좌회전 신호 수집 당시 소스가 없거나 변경됐습니다: ' + relative)
        checks['captured_source_snapshot'] = True
        checks['episode_identity'] = metadata.get('scenario_name') == MODE and metadata.get('seed') == proof.get('seed')
        checks.update(_mission_checks(report, contract))
        checks['scenario_clock_identity'] = _clock_matches(report, metadata)
        checks['expert_runtime_input_identity'] = _expert_runtime_matches(report, metadata, episode)
        checks['collection_completed'] = outcome.get('status') == 'mission_completed' and outcome.get('full_mission_completed') is True
        checks['raw_capture_closed'] = documents['raw_seal'].get('status') == 'completed'
        checks['outcome_report_identity'] = outcome.get('traffic_light_mission', {}).get('report') == report
        checks['legacy_color_head_preserved'] = proof.get('color_head_target_policy') == 'pure_red_green_v1'
        checks['permission_gt_separate'] = proof.get('permission_gt_schema') == 'morai_route_movement_permission_v1'
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        errors.append(str(error))
    return dict(required=True, valid=bool(checks) and all(checks.values()) and not errors,
                checks=checks, errors=errors, proof_path=str(path), historical_source_evidence_only=True,
                physical_semantic_validation_completed=False, manual_all_frame_camera_certification=False)
