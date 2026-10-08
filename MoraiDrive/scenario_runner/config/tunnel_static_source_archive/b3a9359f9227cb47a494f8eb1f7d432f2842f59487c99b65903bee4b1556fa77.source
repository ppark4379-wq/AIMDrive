"""Run one measured, screened tunnel fixture with the actual Expert."""

import argparse
import copy
from functools import partial
import json
import math
from pathlib import Path
import time

from ....main_runner import load_configuration
from ....adapters.expert_controller import expert_max_velocity_contract, validate_max_velocity_kmh
from ....utils.map_loader import MGeoMapLoader
from ..tunnel_static import (
    preview_candidate, screen_candidate, validate_candidate, park_ego, get_actor_id, observe_actor,
)
from .live import _stationary
from ...segment_modes.modes import TUNNEL_NORMAL_START
from ...segment_modes.tunnel_static_trial import make_drive_callback, _write
from ...segment_modes.tunnel_static_startup import (
    block_and_park, align_parked_ego, prepare_common_start, hold_common_brake,
)
from .connection import _controller_publishers


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hold-seconds", type=float, default=2.)
    parser.add_argument("--time-budget", type=float, default=8.)
    parser.add_argument("--expert-max-velocity-kmh", type=float, default=None,
                        help="Optional explicit cap; otherwise inherit the original Expert configuration")
    parser.add_argument("--run-seed", type=int, default=None)
    parser.add_argument("--profile-state-path", type=Path, default=None)
    parser.add_argument("--console-output", action="store_true")
    parser.add_argument("--no-sync-runtime", action="store_true", default=True,
                        help="This single trial always uses asynchronous Expert control without sensor collection")
    return parser


def settle_initial_ego(grpc, evidence=None, *, timeout_sec=5., minimum_stable_sec=.5,
                       minimum_samples=3, sample_period_sec=.1, reassert_park=False,
                       now=None, sleep=None):
    """Observe suspension settling before preview captures its strict reference.

    The default handshake only observes. Collection may reassert its initial
    input-block/zero/P request when readback shows a non-P gear or wrong control
    mode. Each correction resets the stable window, with eight total requests
    and the same five-second deadline. Pose, physics and tolerances stay intact.
    """
    from proto.morai.actor.actor_enum_pb2 import GEAR_MODE_P
    now, sleep = now or time.monotonic, sleep or time.sleep
    if (not math.isfinite(timeout_sec) or not 0 < timeout_sec <= 5.
            or not math.isfinite(minimum_stable_sec) or not .5 <= minimum_stable_sec < timeout_sec
            or isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int) or minimum_samples < 3
            or not math.isfinite(sample_period_sec) or not 0 < sample_period_sec <= .2
            or not isinstance(reassert_park, bool)):
        raise ValueError("Initial Ego settling requires at least .5 s / three samples and a maximum five-second deadline")
    record = evidence if evidence is not None else {}
    record.update(status="settling", timeout_sec=timeout_sec, minimum_stable_sec=minimum_stable_sec,
                  minimum_samples=minimum_samples, samples=0, window_resets=0)
    if reassert_park:
        from proto.morai.actor.actor_enum_pb2 import VEHICLE_CONTROL_KEYBOARD
        # The collection owner has already sent one accepted parking request.
        record.update(parking_attempts=1, parking_reassertions=[], control_observations=[])
    started = now()
    reference = None
    stable_since = None
    stable_samples = 0
    ego = grpc.get_ego()
    while True:
        if now()-started >= timeout_sec:
            record.update(status="failed", elapsed_sec=now()-started, stable_samples=stable_samples)
            raise RuntimeError("Initial Ego did not maintain a continuous parked stationary window within five seconds: "
                               + record.get("last_failure", "insufficient stable duration"))
        try:
            remaining = timeout_sec-(now()-started)
            state = grpc.world._sim_adapter._actor_stub.GetActorState(
                ego.get_object_info(), timeout=max(.001, min(1., remaining)))
            if state is None or not get_actor_id(state):
                raise RuntimeError("Initial Ego bounded readback is unavailable")
            observed = observe_actor(state)
            control_mode = None
            if reassert_park:
                remaining = timeout_sec-(now()-started)
                if remaining <= 0:
                    raise RuntimeError("Initial Ego control readback exceeded the five-second deadline")
                control_mode = int(grpc.world._sim_adapter._actor_stub.GetVehicleControlMode(
                    ego.get_object_info(), timeout=min(.5, remaining)).mode)
        except Exception as error:
            record.update(status="failed", elapsed_sec=now()-started, last_failure=str(error))
            raise RuntimeError("Initial Ego bounded readback failed: " + str(error)) from error
        sampled_at = now()
        record["samples"] += 1
        record["last_observation"] = observed
        record["last_observed_gear_mode"] = int(state.vehicle_state.gear_mode)
        if reassert_park:
            record['last_observed_control_mode'] = control_mode
            readback = dict(gear_mode=record['last_observed_gear_mode'], control_mode=control_mode,
                            speed_norm_raw=observed['speed_norm_raw'], elapsed_sec=sampled_at-started)
            record['control_observations'].append(readback)
            if record['parking_reassertions'] and 'after' not in record['parking_reassertions'][-1]:
                record['parking_reassertions'][-1]['after'] = copy.deepcopy(readback)
        try:
            if int(state.vehicle_state.gear_mode) != int(GEAR_MODE_P):
                raise RuntimeError("Initial Ego P gear is not observed")
            if reassert_park and control_mode != int(VEHICLE_CONTROL_KEYBOARD):
                raise RuntimeError("Initial Ego keyboard control mode is not observed")
            _stationary(observed, reference, "Initial Ego")
        except RuntimeError as error:
            reference, stable_since, stable_samples = None, None, 0
            record["window_resets"] += 1
            record["last_failure"] = str(error)
            if reassert_park and (record['last_observed_gear_mode'] != int(GEAR_MODE_P)
                                 or control_mode != int(VEHICLE_CONTROL_KEYBOARD)):
                remaining = timeout_sec-(now()-started)
                # block_and_park has three RPCs, each bounded to .5 seconds.
                # Reserve their full budget rather than extending the deadline.
                if record['parking_attempts'] >= 8 or remaining < 1.5:
                    record.update(status='failed', elapsed_sec=now()-started, stable_samples=0)
                    raise RuntimeError('Initial Ego P/control-mode readback not confirmed within '
                                       'eight requests and the five-second deadline: '+str(error))
                correction = dict(before=copy.deepcopy(readback), commands_accepted=False)
                record['parking_reassertions'].append(correction)
                record['parking_attempts'] += 1
                try:
                    block_and_park(grpc)
                    correction['commands_accepted'] = True
                except Exception as command_error:
                    record.update(status='failed', elapsed_sec=now()-started,
                                  last_failure=str(command_error), stable_samples=0)
                    correction['failure_reason'] = str(command_error)
                    raise RuntimeError('Initial Ego parking reassertion failed: '+str(command_error)) from command_error
        else:
            if reference is None:
                reference = copy.deepcopy(observed)
                stable_since = sampled_at
            stable_samples += 1
            duration = sampled_at-stable_since
            if stable_samples >= minimum_samples and duration >= minimum_stable_sec and sampled_at-started <= timeout_sec:
                record.update(status="settled", elapsed_sec=sampled_at-started,
                              stable_duration_sec=duration, stable_samples=stable_samples)
                return record
        sleep(min(sample_period_sec, max(0., timeout_sec-(now()-started))))


def validate_trial_settings(args):
    if (not math.isfinite(args.hold_seconds) or not 0 <= args.hold_seconds <= 60
            or not math.isfinite(args.time_budget) or args.time_budget <= 0):
        raise ValueError("Invalid parked hold or screening budget")
    if args.expert_max_velocity_kmh is not None:
        validate_max_velocity_kmh(args.expert_max_velocity_kmh)
    seed = getattr(args, "run_seed", None)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("Run seed must be an integer")


def run(args):
    validate_trial_settings(args)
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    global_cfg, _, scenario_cfg = load_configuration()
    speed_source = expert_max_velocity_contract(global_cfg["paths"]["expert_root"], args.expert_max_velocity_kmh)
    loader = MGeoMapLoader(global_cfg["paths"]["mgeo_root"])
    validate_candidate(candidate, loader)
    output = (args.output or Path(global_cfg["paths"]["runtime_root"])/"tunnel_static_trials"/str(time.time_ns())).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write(output/"candidate.json", candidate)
    report = {"schema": "aimdrive_tunnel_static_trial_manifest_v1", "status": "preparing",
              "expert_verified": False, "live_avoidance_verified": False,
              "candidate_path": str(args.candidate.resolve()), "output_dir": str(output),
              "automatic_repeat": False, "timeout_sec": 180., "sync_runtime_enabled": False,
              "expert_speed_source": speed_source}
    grpc = None
    try:
        publishers = _controller_publishers()
        if publishers:
            raise RuntimeError("Stop the competing Ego controller before a trial: " + ", ".join(publishers))
        from ....adapters.morai_grpc import MoraiGrpcClient
        fixture_cfg = copy.deepcopy(global_cfg)
        fixture_cfg["grpc"]["client_key"] = "tunnel_static_trial_fixture"
        grpc = MoraiGrpcClient(fixture_cfg)
        grpc.connect()
        grpc.start_world(grpc.make_transform(*TUNNEL_NORMAL_START.xyz, TUNNEL_NORMAL_START.yaw_deg))
        grpc.resume_world()
        if not grpc.reset_actors():
            raise RuntimeError("Could not clear previous trial-owned fixtures")
        report["initial_ego_settle"] = {}
        prepare_common_start(grpc, report["initial_ego_settle"])
        available = {"obstacle": set(grpc.get_available_obstacle_models()),
                     "vehicle": set(grpc.get_available_surround_vehicle_models())}
        if any(item["model"] not in available[item["actor_kind"]] for item in candidate["obstacles"]):
            raise RuntimeError("A requested trial model is unavailable in current MORAI")
        cfg = copy.deepcopy(scenario_cfg)
        cfg["expert_max_velocity_kmh"] = args.expert_max_velocity_kmh
        cfg["tunnel_static_console_output"] = bool(getattr(args, "console_output", False))
        if getattr(args, "run_seed", None) is not None:
            cfg["random_seed"] = args.run_seed
            report["requested_run_seed"] = args.run_seed
        if getattr(args, "profile_state_path", None) is not None:
            cfg["profile_state_path"] = str(args.profile_state_path.expanduser().resolve())
            cfg["tunnel_static_profile_state_path_explicit"] = True
            report["profile_state_path"] = cfg["profile_state_path"]
        if expert_max_velocity_contract(global_cfg["paths"]["expert_root"], args.expert_max_velocity_kmh) != speed_source:
            raise RuntimeError("Original Expert speed source changed before the guarded trial")
        drive = make_drive_callback(grpc, loader, global_cfg, cfg, output)
        result = preview_candidate(grpc, loader, candidate, hold_seconds=args.hold_seconds, output_dir=output,
            screen_fn=lambda data, map_loader, **kwargs: screen_candidate(data, map_loader,
                time_budget_sec=args.time_budget, **kwargs), drive_fn=drive)
        report["fixture_result"] = result
        if expert_max_velocity_contract(global_cfg["paths"]["expert_root"], args.expert_max_velocity_kmh) != speed_source:
            raise RuntimeError("Original Expert speed source changed during the guarded trial")
        report["status"] = "passed" if result.get("status") == "valid_driving_trial" else "failed"
        report["expert_verified"] = result.get("expert_verified") is True
        report["live_avoidance_verified"] = result.get("live_avoidance_verified") is True
        report["failure_reason"] = result.get("failure_reason", "")
        return report
    except BaseException as error:
        report.update(status="failed", failure_reason=str(error) or type(error).__name__)
        raise
    finally:
        try:
            if grpc is not None and grpc.world is not None:
                grpc._tunnel_static_common_start = True
                park_ego(grpc, grpc.get_ego(), {"ego_parking_commands": 0})
        except BaseException as error:
            report.update(status="failed", expert_verified=False, live_avoidance_verified=False,
                          cleanup_failure=str(error) or type(error).__name__)
            raise
        finally:
            try:
                if grpc is not None:
                    try:
                        grpc.stop()
                    except BaseException as error:
                        report.update(status="failed", expert_verified=False, live_avoidance_verified=False,
                                      cleanup_failure=str(error) or type(error).__name__)
                        raise
            finally:
                _write(output/"manifest.json", report)


def run_collection_episode(grpc, loader, global_cfg, scenario_cfg, *, scenario_factory,
                           scenario_kwargs, configure_scenario=None, output_dir, seed,
                           history_path=None):
    """Run the existing collector on one freshly guarded 121-library fixture.

    Sensor recording, storage, clocks and native UI remain collector-owned.
    This owner only selects/prepares the scene and preserves tunnel guards.
    """
    from . import replay, runtime, safety, diversity
    from ...segment_modes.tunnel_static_replay import DEFAULT_LIBRARY, _curation_pins
    from .layout import tunnel_geometry
    from .live import _inventory, _park
    from .screening import screening_provenance
    output_dir = Path(output_dir).resolve()
    episode = Path(scenario_kwargs['output']).resolve()
    library = DEFAULT_LIBRARY.resolve()
    pins = _curation_pins(library)
    geometry = tunnel_geometry(loader)
    verified = replay.verify_library(library, geometry, loader.link_set)
    history_path = Path(history_path).resolve() if history_path else output_dir.parent/'selection_history.json'
    history, _ = replay.read_history(history_path if history_path.exists() else None)
    selected, balance = replay.balanced_choice(verified['eligible'], seed, history)
    for path, expected in pins.items():
        if replay.digest(path) != expected:
            raise RuntimeError('Collection tunnel library changed during selection')
    output_dir.mkdir(parents=True, exist_ok=False)
    selection = dict(schema=replay.SCHEMA, status='selected', seed=seed, live_requested=True,
                     library_path=str(library), library_sha256=verified['library_sha256'],
                     selected={key:selected[key] for key in ('id','paths','file_sha256','strata')},
                     balance_decisions=balance, eligible_count=verified['eligible_count'])
    _write(output_dir/'selection.json', selection)
    candidate_path = Path(selected['paths']['candidate'])
    candidate = json.loads(candidate_path.read_text())
    (output_dir/'candidate.json').write_bytes(candidate_path.read_bytes())
    frozen = runtime.source_snapshot([Path(__file__)], [candidate_path, library])
    sc = copy.deepcopy(scenario_cfg)
    sc.update(random_seed=seed, expert_max_velocity_kmh=None,
              tunnel_static_profile_state_path_explicit=True,
              _tunnel_static_collection=True, _tunnel_static_candidate_sha256=selected['id'])
    # Keep the collector's persistent zone profile; preserve other zone files.
    sc['profile_state_path'] = str(Path(sc['profile_state_path']).with_name('full_loop_tunnel_static_profile.json'))
    sink = {}
    result = {'status':'failed', 'reason':'tunnel_setup_incomplete', 'full_mission_completed':False,
              'training_visible':False}
    preview = None
    restored = False
    restoration = None
    scene = output_dir/'scene_backup'
    proof_path = episode/'tunnel_static/COLLECTION_TUNNEL_COMPLETE.json'
    with runtime.ExecutionLock():
        scene_started = False
        startup = {'status':'preparing', 'settling':{}, 'startup_policy':'common_auto_d_brake'}
        try:
            runtime.require_sources(frozen)
            if _controller_publishers():
                raise RuntimeError('Competing Ego control publisher before collected tunnel preparation')
            grpc.start_world(grpc.make_transform(*TUNNEL_NORMAL_START.xyz, TUNNEL_NORMAL_START.yaw_deg))
            grpc.resume_world()
            if not grpc.reset_actors():
                raise RuntimeError('Could not clear owned collection fixture actors')
            prepare_common_start(grpc, startup['settling'])
            startup['status'] = 'ready'
            _write(output_dir/'startup.json', startup)
            inventory = _inventory(grpc, {'inventory_checks':0})
            # The shared guard preserves explicitly backed-up foreign fixtures;
            # actor 3 is never relocated, frozen or deleted.
            relocate_ids = [value for value in ('2','4','5') if value in inventory]
            scene_started = True
            safety.relocate(grpc, grpc.get_ego(), scene, relocate_ids, 'move')
            callback = make_drive_callback(grpc, loader, global_cfg, sc, output_dir/'guard',
                scenario_factory=scenario_factory, scenario_kwargs=scenario_kwargs,
                configure_scenario=configure_scenario, result_sink=sink, collection_episode=episode)
            preview = preview_candidate(grpc, loader, candidate, hold_seconds=5., output_dir=output_dir,
                screen_fn=lambda data, map_loader, **kwargs: screen_candidate(data, map_loader,
                    time_budget_sec=15., **kwargs), drive_fn=callback)
            result = sink.get('outcome') or result
            if preview.get('status') != 'valid_driving_trial' or preview.get('cleanup',{}).get('verified') is not True:
                result = dict(result, status='failed', full_mission_completed=False,
                              reason='tunnel_guard: '+preview.get('failure_reason','incomplete fixture proof'))
            runtime.require_sources(frozen)
        except BaseException as exc:
            result = dict(sink.get('outcome') or result, status='failed', full_mission_completed=False,
                          reason=type(exc).__name__+': '+str(exc), training_visible=False)
            if startup['status'] != 'ready':
                startup.update(status='failed', failure_reason=result['reason'])
        finally:
            try:
                if getattr(grpc,'world',None) is not None:
                    grpc._tunnel_static_common_start = True
                    _park(grpc, grpc.get_ego(), {'ego_parking_commands':0})
                if scene_started:
                    restoration = safety.relocate(grpc, grpc.get_ego(), scene, [], 'restore')
                    restored = restoration['status'] in ('restored','nothing_to_restore')
                    if restoration['status'] == 'nothing_to_restore':
                        restoration = dict(restoration, actors=[], scene_was_moved=False)
            except Exception as exc:
                primary_reason = result.get('reason') if result.get('status') == 'failed' else None
                cleanup_reason = 'tunnel cleanup failed: '+str(exc)
                result = dict(result, status='failed', full_mission_completed=False,
                              reason=primary_reason or cleanup_reason,
                              cleanup_failure_reason=cleanup_reason)
            _write(output_dir/'startup.json', startup)
            if episode.exists():
                destination = episode/'tunnel_static'
                destination.mkdir(exist_ok=True)
                (destination/'candidate.json').write_bytes(candidate_path.read_bytes())
                if preview is not None:
                    _write(destination/'preview.json', preview)
                raw_seal = episode/'raw/RAW_COMPLETE.json'
                raw_complete = raw_seal.is_file() and json.loads(raw_seal.read_text()).get('status')=='completed'
                cleanup = preview is not None and preview.get('cleanup',{}).get('verified') is True
                passed = result.get('status')=='mission_completed' and result.get('full_mission_completed') is True and cleanup and restored and raw_complete
                if not passed:
                    result.update(status='failed', full_mission_completed=False)
                # Seal the final outcome after the outer fixture owner has
                # completed cleanup and exact foreign-scene restoration.
                _write(episode/'outcome.json', result)
                if restoration is not None:
                    _write(destination/'scene_restore.json', restoration)
                files = {'candidate':destination/'candidate.json', 'preview_report':destination/'preview.json',
                         'drive_report':destination/'drive_trial.json', 'planner_diagnostics':destination/'planner_diagnostics.json',
                         'ego_trace':destination/'ego_trace.csv', 'episode':episode/'episode.json',
                         'outcome':episode/'outcome.json', 'raw_seal':raw_seal,
                         'scene_restore':destination/'scene_restore.json'}
                proof = dict(schema='aimdrive_tunnel_static_collection_complete_v1', scenario='full_loop_tunnel_static',
                    status='passed' if passed else 'failed', selected_candidate_sha256=selected['id'],
                    library_sha256=verified['library_sha256'], fixture_cleanup_verified=bool(cleanup),
                    scene_restored=restored, raw_record_complete=raw_complete,
                    files={name:{'path':str(path),'sha256':replay.digest(path)} for name,path in files.items() if path.is_file()},
                    runtime_source_sha256=replay.current_runtime_sources(),
                    source_files=frozen['production']['runtime_files'],
                    failure_reason='' if passed else result.get('reason','incomplete capture proof'))
                _write(proof_path, proof)
                result['tunnel_static']={'proof_path':str(proof_path),'proof_sha256':replay.digest(proof_path),
                                         'selected_candidate_sha256':selected['id'],'passed':passed}
                if not passed:
                    result.update(status='failed', full_mission_completed=False)
    selection.update(status='passed' if result.get('status')=='mission_completed' else 'failed',
                     replay_actual_pass=result.get('status')=='mission_completed',
                     failure_reason=result.get('reason',''))
    _write(output_dir/'selection.json', selection)
    print('[KCityRunner] tunnel collection seed=%s status=%s%s' % (
        seed, selection['status'], ' reason='+selection['failure_reason'] if selection['failure_reason'] else ''), flush=True)
    if result.get('cleanup_failure_reason'):
        print('[KCityRunner] '+result['cleanup_failure_reason'], flush=True)
    history.append(selection)
    _write(history_path, history)
    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = run(args)
    except (ValueError, RuntimeError) as error:
        parser.exit(2, str(error)+"\n")
    print("Tunnel Expert trial: " + report["status"] + " | " + report["output_dir"])
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
