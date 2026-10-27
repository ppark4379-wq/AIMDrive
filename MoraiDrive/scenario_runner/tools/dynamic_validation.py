"""Connect the dynamic launcher to existing observation-only diagnosis tools."""
import json
from pathlib import Path
import shutil

from scenario_runner.validation.run_log import ValidationRunLog
from scenario_runner.validation.finish import finish_validation_run
from scenario_runner.scenarios.full_loop_parts.dynamic_obstacle import dynamic_obstacle_reports


def start_validation(scenario):
    if getattr(scenario, '_runner_validation_log', None) is not None:
        return
    root = Path(__file__).resolve().parents[1] / '.runtime'
    scenario._runner_validation_log = log = ValidationRunLog(root / 'validation_runs', 'full_loop_dynamic')
    scenario._runner_scenario_name = 'full_loop_dynamic'
    scenario._validation_controller = 'aim2026_dynamic'
    options = ['--seed', str(scenario.run_seed), '--validation-events',
               '--config', str(scenario.args.config),
               '--dynamic-obstacle-count', str(scenario.args.dynamic_obstacle_count)]
    if getattr(scenario.args, 'expert_python', None):
        options.extend(['--expert-python', str(scenario.args.expert_python)])
    if scenario.args.dynamic_location_mix:
        options.append('--dynamic-location-mix')
    if getattr(scenario.args, 'random_locations', None):
        options.extend(['--random-locations', str(scenario.args.random_locations)])
    if getattr(scenario.args, 'roads', None):
        options.extend(['--roads', *map(str, scenario.args.roads)])
    if getattr(scenario.args, 'only_location', None) is not None:
        options.extend(['--only-location', str(scenario.args.only_location)])
    if getattr(scenario.args, 'opposing_crosswalk', None) is not None:
        options.extend(['--opposing-crosswalk', str(scenario.args.opposing_crosswalk)])
    if scenario.args.strict_collisions:
        options.append('--strict-collisions')
    if getattr(scenario.args, 'pedestrian_model', None):
        options.extend(['--pedestrian-models', scenario.args.pedestrian_model])
    if getattr(scenario.args, 'exclude_pedestrian_models', None):
        options.extend(['--exclude-pedestrian-models', *scenario.args.exclude_pedestrian_models])
    if getattr(scenario.args, 'pedestrian_trace', False):
        options.append('--pedestrian-trace')
    log.enable_incidents(root / 'incidents', scenario, run_options=options)
    log.incident_recorder.connect_pipeline(scenario.driver_output / 'pipeline.jsonl')
    # Persist the exact integer before any setup/drive failure can occur.
    (scenario.output / 'diagnostic_setup.json').write_text(json.dumps({
        'run_id': scenario.run_instance_id, 'seed': scenario.run_seed,
        'run_options': options, 'scenario_config': scenario.cfg,
        'driver_output': str(scenario.driver_output),
    }, ensure_ascii=False, indent=2), encoding='utf-8')


def finish_validation(scenario, status, error):
    log = getattr(scenario, '_runner_validation_log', None)
    if log is None:
        return
    if status == 'exception':
        log.incident_recorder.capture_dynamic_failure(scenario.run_instance_id, str(error))
    finish_validation_run(log, scenario, status, error)
    profile = Path(scenario.profile_state_path())
    if profile.is_file() and getattr(scenario, 'dynamic_obstacle_events', None):
        document = json.loads(profile.read_text(encoding='utf-8'))
        if document.get('run_seed') == scenario.run_seed:
            shutil.copyfile(profile, scenario.output / 'dynamic_profile.json')
    report = dict(status=status, error=str(error) if error else None,
                  run_id=scenario.run_instance_id,
                  plan=dict(mission='full_loop_dynamic', seed=scenario.run_seed,
                            driver_output=str(scenario.driver_output)),
                  incidents=[str(p) for p in log.incident_recorder.bundles],
                  dynamic_obstacle_events=dynamic_obstacle_reports(scenario))
    for destination in (scenario.output, scenario.driver_output):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if (scenario.driver_output / 'pipeline.jsonl').is_file():
        from scenario_runner.validation.failure_report import build_report
        html, _ = build_report(scenario.driver_output, scenario.output / 'failure_report')
        print('[Dynamic] Diagnosis report: ' + str(html), flush=True)
