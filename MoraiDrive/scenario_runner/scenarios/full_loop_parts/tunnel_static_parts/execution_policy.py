"""Separate historical layouts from successful trials at the current Expert settings."""
import hashlib
import json
import math
from pathlib import Path

from ....adapters.expert_controller import expert_max_velocity_contract
from . import diversity, screening
from .runtime import ExecutionLock, atomic_json

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STORE = PACKAGE_ROOT / '.runtime/tunnel_static_replays/current_expert_approvals.json'
SCHEMA = 'aimdrive_tunnel_current_expert_approvals_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def describe(global_cfg, library):
    from . import evidence, replay
    speed = expert_max_velocity_contract(global_cfg['paths']['expert_root'])
    runtime = replay.current_runtime_sources()
    files = dict(runtime)
    safety = PACKAGE_ROOT / 'tools/tunnel_static_batch_safety.sh'
    # Approval identity covers the same immutable files guarded by every fresh
    # replay, including the alias dispatch, connection helpers and model data.
    # Mutable run history, logs and approval stores are deliberately excluded.
    files.update(evidence.source_snapshot({}, safety, safety)['runtime_files'])
    for relative in ('scenarios/segment_modes/tunnel_static_replay.py', 'scenarios/segment_modes/events.py',
                     'scenarios/full_loop_scenario.py', 'scenarios/full_loop_parts/route_profile.py',
                     'scenarios/full_loop_parts/actor_lifecycle.py',
                     'scenarios/full_loop_parts/vehicle_common.py',
                     'scenarios/full_loop_parts/static_obstacle.py',
                     'core/base_scenario.py', 'utils/map_loader.py', 'utils/link_graph.py',
                     'utils/transform_utils.py', 'utils/subprocess_log.py', 'utils/__init__.py',
                     'config/full_loop.yaml', 'main_runner.py'):
        path = (PACKAGE_ROOT / relative).resolve()
        files[str(path)] = digest(path)
    for name in ('replay.py', 'execution.py', 'runtime.py', 'evidence.py', 'safety.py', 'trial.py',
                 'execution_policy.py', 'source_compatibility.py'):
        path = Path(__file__).with_name(name)
        files[str(path)] = digest(path)
    paths = global_cfg.get('paths', {})
    mgeo = Path(paths.get('mgeo_root', PACKAGE_ROOT / 'data/R_KR_PR_K-city_2025'))
    if not mgeo.is_absolute():
        mgeo = PACKAGE_ROOT / mgeo
    for name in ('link_set.json', 'node_set.json'):
        path = (mgeo / name).resolve()
        files[str(path)] = digest(path)
    project = PACKAGE_ROOT.parent
    # These are read-only dependencies of the existing launcher. Optional
    # environment helpers may be absent in another supported repository layout.
    for relative in ('tools/internal_sensor_record_env.sh',
                     'internal_sensor_record/runtime_paths.py',
                     'internal_sensor_record/network_config.py',
                     'internal_sensor_record/python_environment.py'):
        path = (project / 'data_collection' / relative).resolve()
        if path.is_file():
            files[str(path)] = digest(path)
    expert_source = Path(paths['expert_root']).resolve() / 'src'
    for path in sorted(expert_source.rglob('*.py')):
        files[str(path.resolve())] = digest(path)
    sdk = Path(paths.get('grpc_src', project / 'data_collection/third_party/grpc_inha_univ/src'))
    if not sdk.is_absolute():
        sdk = PACKAGE_ROOT / sdk
    for directory in ('api', 'lib', 'proto'):
        for path in sorted((sdk.resolve() / directory).rglob('*.py')):
            files[str(path.resolve())] = digest(path)
    files[speed['expert_config_path']] = speed['expert_config_sha256']
    return dict(schema='aimdrive_tunnel_inherited_expert_execution_policy_v1',
                library_sha256=digest(library), expert_speed=dict(speed, max_velocity_kmh=speed['resolved_max_velocity_kmh']),
                unified_environment=True,
                runtime_files=runtime, screening=screening.screening_provenance(), source_files=files)


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def trace_geometry(manifest):
    from ....utils.map_loader import MGeoMapLoader
    from .layout import tunnel_geometry
    return tunnel_geometry(MGeoMapLoader(manifest['mgeo_root']))


def _proof(selection_path, policy):
    selection_path = Path(selection_path).resolve()
    manifest = _read(selection_path)
    if (not isinstance(manifest, dict) or manifest.get('status') != 'passed'
            or manifest.get('live_requested') is not True or manifest.get('replay_actual_pass') is not True
            or manifest.get('execution_policy') != policy):
        raise ValueError('Current Expert approval requires a fresh successful policy-bound replay')
    if policy.get('unified_environment'):
        environment = manifest.get('environment_profile', {})
        incoming = manifest.get('incoming_environment_history') or {}
        if (environment.get('run_seed') != manifest.get('seed')
                or environment.get('weather') not in ('SUNNY', 'FOGGY')
                or environment.get('time_hour') not in (11,13,15)
                or environment.get('weather') == incoming.get('weather')
                or environment.get('time_hour') == incoming.get('time_hour')):
            raise ValueError('Fresh seed, weather alternation or scenario time differs from the shared environment policy')
    live = selection_path.with_suffix('').with_name(selection_path.stem + '_live')
    state = _read(live / 'search_manifest.json')
    result = _read(live / 'replay_result.json')
    if (not state or state.get('actual_drives') != 1 or state.get('actual_passes') != 1
            or state.get('actual_failures') != 0 or state.get('status') != 'completed'
            or state.get('scene_restored') is not True or state.get('cleanup_verified') is not True or not result
            or result.get('scene_restored') is not True or result.get('cleanup_verified') is not True
            or result.get('outcome', {}).get('strict_pass') is not True):
        raise ValueError('Current Expert approval is missing completed cleanup/restoration')
    committed = [row for row in state.get('attempts', []) if row.get('outcome', {}).get('actual_driving') is True]
    if (len(committed) != 1 or committed[0].get('committed') is not True
            or committed[0].get('recovery_verified') is not True
            or committed[0].get('outcome', {}).get('strict_pass') is not True):
        raise ValueError('Current Expert approval must contain exactly one completed physical drive')
    process = committed[0].get('process', {})
    if (process.get('returncode') != 0 or any(process.get(key) is not False for key in
                                             ('timed_out', 'interrupted', 'forced_kill'))):
        raise ValueError('Current Expert approval requires a clean physical trial process exit')
    trial = Path(committed[0]['trial_dir']).resolve()
    if live.resolve() not in trial.parents:
        raise ValueError('Approval proof is outside its fresh replay directory')
    drive = _read(trial / 'drive_trial.json')
    preview = _read(trial / 'preview.json')
    planner = _read(trial / 'planner_diagnostics.json')
    config = _read(trial / 'runner_runtime/expert/config.json')
    from .evidence import classify_trial
    from . import replay
    classified = classify_trial(trial)
    if (classified['strict_pass'] is not True or classified['actual_driving'] is not True
            or classified['physical_motion_observed'] is not True):
        raise ValueError('Fresh completed trial classification failed')
    conditions = drive.get('completion_conditions', {})
    if (drive.get('status') != 'passed' or not conditions or not all(value is True for value in conditions.values())
            or not all(drive.get(key) is True for key in diversity.STRICT_FLAGS)
            or not all(conditions.get(key) is True for key in diversity.STRICT_CONDITIONS)
            or drive.get('collision_objects') != [] or drive.get('runtime_source_sha256') != policy['runtime_files']
            or preview.get('status') != 'valid_driving_trial' or preview.get('cleanup', {}).get('verified') is not True
            or planner.get('validated_path_seen') is not True or planner.get('calls', 0) < 1
            or config['planning']['velocity_profile']['max_velocity'] != policy['expert_speed']['max_velocity_kmh']):
        raise ValueError('Current Expert driving, source, speed, geometry or completion proof differs')
    screen = preview.get('measured_route_screen', {})
    if (screen.get('status') != 'nominal_path_found' or screen.get('path_found') is not True
            or screen.get('geometry_checks', {}).get('passed') is not True):
        raise ValueError('Fresh measured path/geometry proof is missing')
    boundary = screen.get('road_boundary_policy')
    boundary_sha = screen.get('road_boundary_policy_sha256')
    contract = config['planning'].get('jamming_trial_contract', {})
    if (not boundary or identity(boundary) != boundary_sha
            or any(obj.get('road_boundary_policy') != boundary or obj.get('road_boundary_policy_sha256') != boundary_sha
                   for obj in (drive, planner, contract))):
        raise ValueError('Fresh planner, actual body and measured boundary contracts differ')
    geometries = config['planning'].get('jamming_scene_geometry', [])
    if not geometries or identity(geometries) != planner.get('scene_geometry_sha256'):
        raise ValueError('Fresh planner did not use the measured scene geometry')
    observed_ids = {str(row['raw_actor_id']) for row in preview.get('observed', [])}
    geometry_ids = {str(row.get('raw_actor_id', row.get('raw_id', row.get('unique_id')))) for row in geometries}
    if (not observed_ids or len(observed_ids) != len(preview.get('planned', {}).get('obstacles', []))
            or observed_ids != geometry_ids or observed_ids != set(drive.get('fixture_raw_actor_ids', []))):
        raise ValueError('Fresh planner fixture identity/count differs from the measured owned actors')
    candidate = trial / 'candidate.json'
    candidate_id = manifest['selected']['id']
    source_candidate = Path(manifest['selected']['paths']['candidate'])
    if digest(source_candidate) != candidate_id or _read(candidate) != _read(source_candidate):
        raise ValueError('Fresh trial candidate differs from the selected immutable layout')
    trajectory = drive.get('trajectory', [])
    if (len(trajectory) < 3 or not all(all(isinstance(v, (float, int)) and math.isfinite(v)
                                       for v in row['position_xyz']) for row in trajectory)
            or max(math.dist(trajectory[0]['position_xyz'], row['position_xyz']) for row in trajectory) < .5):
        raise ValueError('Current Expert approval is missing finite physical motion')
    csv_points = replay._trace(trial / 'ego_trace.csv')
    geometry = trace_geometry(manifest)
    actual = diversity.path_features(csv_points, geometry, evidence_kind='completed_actual_trace')
    observed = diversity.path_features([row['position_xyz'][:2] for row in trajectory], geometry,
                                      evidence_kind='completed_actual_trace')
    if diversity.compare_paths(actual, observed)['duplicate'] is not True:
        raise ValueError('Actual CSV trace differs from the sealed physical motion')
    paths = [selection_path, live/'search_manifest.json', live/'replay_result.json', candidate, source_candidate,
             trial/'drive_trial.json', trial/'preview.json', trial/'manifest.json',
             trial/'planner_diagnostics.json', trial/'runner_runtime/expert/config.json', trial/'ego_trace.csv']
    return {'id':candidate_id, 'selection_path':str(selection_path),
            'proof_sha256':{str(path):digest(path) for path in paths}}


def approved_ids(store, policy):
    ledger = _read(store)
    if ledger is None:
        return set()
    if not isinstance(ledger, dict) or ledger.get('schema') != SCHEMA:
        raise ValueError('Current Expert approval store has an invalid schema')
    section = ledger.get('policies', {}).get(identity(policy), {})
    if not section:
        return set()
    if section.get('policy') != policy:
        raise ValueError('Current Expert approval policy does not match its identity')
    accepted = set()
    for row in section.get('layouts', []):
        if row['id'] in accepted or not row.get('proof_sha256'):
            raise ValueError('Duplicate or incomplete current Expert approval')
        if any(digest(path) != expected for path, expected in row['proof_sha256'].items()):
            raise ValueError('Sealed current Expert approval proof changed')
        if _proof(row['selection_path'], policy) != row:
            raise ValueError('Current Expert approval no longer meets its physical proof')
        accepted.add(row['id'])
    return accepted


def approve(store, policy, selection_path):
    store = Path(store)
    proof = _proof(selection_path, policy)
    with ExecutionLock(store.with_suffix('.lock')):
        ledger = _read(store, {'schema':SCHEMA, 'policies':{}})
        approved_ids(store, policy)
        section = ledger['policies'].setdefault(identity(policy), {'policy':policy, 'layouts':[]})
        if proof['id'] not in {row['id'] for row in section['layouts']}:
            section['layouts'].append(proof)
        atomic_json(store, ledger)
    return proof['id']
