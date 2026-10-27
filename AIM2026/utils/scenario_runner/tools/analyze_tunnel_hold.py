#!/usr/bin/env python3
"""Compare tunnel dead reckoning with withheld GPS, without driving the simulator.

Usage: analyze_tunnel_hold.py CAMPAIGN_OR_TRIAL_OR_DRIVER [--output-dir DIR]
Matched receipt-time comparisons and legacy age/speed approximations are always
reported separately. No application tolerance is assumed by this tool.
"""
import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

DRIVER = re.compile(r'\[GPS\+VE 주행 로그\]\s*([^\r\n]+?)/driver\.console\.log')
MATCHED = 'matched_receipt_time'
LEGACY = 'legacy_age_speed_approximation'
EXPIRY = 'sensors_expired_during_inference'


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def read_text(path):
    try:
        return path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return ''


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError, OverflowError):
        return None


def xy(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    result = [number(value[0]), number(value[1])]
    return result if None not in result else None


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values)-1)*fraction
    low = int(index)
    return values[low] + (values[min(low+1, len(values)-1)]-values[low])*(index-low)


def stats(values):
    values = [value for value in values if number(value) is not None]
    if not values:
        return None
    return dict(count=len(values), max=max(values), median=statistics.median(values),
                p95=percentile(values, .95), mean=statistics.mean(values))


def error_stats(samples):
    return {key: stats([row[key] for row in samples]) for key in
            ('error_2d_m', 'longitudinal_abs_m', 'lateral_abs_m')}


def error_sample(trial, method, real, estimated, yaw, **metadata):
    dx, dy = estimated[0]-real[0], estimated[1]-real[1]
    along = dx*math.cos(yaw) + dy*math.sin(yaw)
    lateral = -dx*math.sin(yaw) + dy*math.cos(yaw)
    return dict(trial=trial, method=method, real_x=real[0], real_y=real[1],
                estimated_x=estimated[0], estimated_y=estimated[1],
                longitudinal_m=along, lateral_m=lateral,
                longitudinal_abs_m=abs(along), lateral_abs_m=abs(lateral),
                error_2d_m=math.hypot(dx, dy), **metadata)


def find_control(root, console):
    if root.is_file():
        return root if root.name == 'control.jsonl' else None
    for path in (root/'driver/control.jsonl', root/'control.jsonl'):
        if path.is_file():
            return path
    found = DRIVER.search(console)
    if found:
        base = Path(found.group(1).strip())
        path = base/'driver/control.jsonl'
        if not path.is_absolute():
            path = root/path
        return path if path.is_file() else None
    return None


def comparison_samples(row, gps, trial, seen, diagnostics):
    """Only call receipt alignment matched when its explicit timestamps agree."""
    candidates = gps.get('simulated_comparisons')
    if not isinstance(candidates, list) or not candidates:
        single = gps.get('simulated_comparison')
        candidates = [single] if isinstance(single, dict) else []
    samples = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            diagnostics['invalid_comparisons'] += 1
            continue
        receipt = candidate.get('gps_received_ns')
        dr_receipt = candidate.get('dr_received_ns')
        real, estimate = xy(candidate.get('real_gps_xy')), xy(candidate.get('estimated_xy'))
        yaw = number(candidate.get('yaw_grid_rad'))
        alignment = candidate.get('alignment_method', candidate.get('alignment'))
        if (not isinstance(receipt, int) or isinstance(receipt, bool) or receipt <= 0
                or dr_receipt != receipt or candidate.get('time_basis') != 'receipt_monotonic'
                or alignment != 'interpolated_dr_history' or real is None
                or estimate is None or yaw is None):
            diagnostics['invalid_comparisons'] += 1
            continue
        key = (MATCHED, receipt)
        if key in seen:
            diagnostics['duplicate_comparisons'] += 1
            continue
        seen.add(key)
        samples.append(error_sample(trial, MATCHED, real, estimate, yaw,
            gps_received_ns=receipt, monotonic_s=receipt/1e9,
            hold_s=number(candidate.get('hold_s')), distance_m=number(candidate.get('distance_m')),
            speed_mps=number(candidate.get('speed_mps')), alignment_method=alignment))
    # New-format metadata, even if malformed, must never silently become an
    # approximate comparison with a different position and time.
    if candidates:
        return samples
    position = row.get('position') or {}
    real, estimate = xy(gps.get('simulated_real_gps_xy')), xy([position.get('x'), position.get('y')])
    yaw, speed = number(position.get('yaw_grid_rad')), number(position.get('speed_mps'))
    age = number(gps.get('simulated_real_gps_age_s'))
    if gps.get('position_mode') != 'dead_reckoning' or None in (yaw, speed, age) or age < 0:
        return []
    if real is None or estimate is None:
        return []
    receipt = gps.get('simulated_real_gps_received_ns')
    if isinstance(receipt, int) and receipt > 0:
        key = (LEGACY, receipt)
        if key in seen:
            diagnostics['duplicate_comparisons'] += 1
            return []
        seen.add(key)
    # Historical logs lack source/receipt correspondence: this is an explicitly
    # labelled constant-speed/heading approximation, not a matched sample.
    real = [real[0] + speed*age*math.cos(yaw), real[1] + speed*age*math.sin(yaw)]
    return [error_sample(trial, LEGACY, real, estimate, yaw,
        gps_received_ns=receipt, monotonic_s=number(row.get('monotonic')),
        hold_s=number(gps.get('hold_s')),
        distance_m=number((gps.get('dead_reckoning') or {}).get('distance_m')),
        speed_mps=speed, alignment_method='constant_current_speed_heading_age',
        gps_age_s=age)]


def analyze_trial(root, result=None):
    root = Path(root)
    if root.is_file():
        trial_root = root.parent.parent if root.parent.name == 'driver' else root.parent
    elif root.name == 'driver' and (root/'control.jsonl').is_file():
        trial_root = root.parent
    else:
        trial_root = root
    console = read_text(trial_root/'console.log')
    result = read_json(trial_root/'result.json') or result or {}
    control = find_control(root, console)
    driver_root = control.parent.parent if control and control.parent.name == 'driver' else trial_root
    runtime_report = read_json(driver_root/'report.json')
    text = console + '\n' + read_text(driver_root/'driver.console.log')
    diagnostics, seen, samples, transitions = Counter(), set(), [], []
    modes, held_s, distance, speeds = Counter(), [], [], []
    previous, hold_frames = None, 0
    if control:
        with control.open(encoding='utf-8', errors='replace') as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('control row must be an object')
                except (ValueError, TypeError):
                    diagnostics['malformed_control_lines'] += 1
                    continue
                if EXPIRY in line:
                    diagnostics[EXPIRY] += 1
                gps = (row.get('control_pose_selected_samples') or {}).get('gps') or {}
                mode = gps.get('position_mode')
                if mode:
                    modes[mode] += 1
                    position, moment = row.get('position') or {}, number(row.get('monotonic'))
                    current = (mode, moment, xy([position.get('x'), position.get('y')]))
                    if previous and previous[0] != mode:
                        step = (math.dist(previous[2], current[2])
                                if previous[2] and current[2] else None)
                        transitions.append(dict(from_mode=previous[0], to_mode=mode,
                            monotonic_s=moment, observed_position_step_m=step,
                            elapsed_since_previous_pose_s=(moment-previous[1]
                                if moment is not None and previous[1] is not None else None)))
                    previous = current
                if mode == 'dead_reckoning':
                    hold_frames += 1
                    hold = number(gps.get('hold_s'))
                    dist = number((gps.get('dead_reckoning') or {}).get('distance_m'))
                    speed = number((row.get('position') or {}).get('speed_mps'))
                    if hold is not None:
                        held_s.append(hold)
                    if dist is not None:
                        distance.append(dist)
                    if speed is not None:
                        speeds.append(abs(speed)*3.6)
                samples.extend(comparison_samples(row, gps, trial_root.name, seen, diagnostics))
    samples.sort(key=lambda row: row.get('monotonic_s') or 0.)
    for sample in samples:
        if sample['hold_s'] is not None:
            held_s.append(sample['hold_s'])
        if sample['distance_m'] is not None:
            distance.append(sample['distance_m'])
    baselines = []
    for base in dict.fromkeys((trial_root, driver_root)):
        if (base/'baseline.json').is_file():
            baselines.append(read_json(base/'baseline.json'))
        baselines.extend(read_json(path) for path in sorted((base/'tunnel_baseline').glob('*/baseline.json')))
    baseline = baselines[-1] if baselines else {}
    outcome = result.get('result') or runtime_report.get('status') or 'unknown'
    clean = str(outcome).lower() in ('success', 'complete', 'passed')
    expired = EXPIRY in text or EXPIRY in json.dumps(result) or diagnostics[EXPIRY] > 0
    recovery = any(t['from_mode'] == 'dead_reckoning' and t['to_mode'] == 'gps' for t in transitions)
    boundary_complete = (baseline.get('started_before_entry') is True
                         and baseline.get('mapped_entry_observed') is True
                         and baseline.get('mapped_exit_observed') is True)
    # An explicit missing boundary observation overrides a success label. Older
    # runs without a boundary report can only use the weaker GPS recovery evidence.
    traversal = boundary_complete if baseline else recovery
    complete = clean and traversal and hold_frames > 0
    completion = 'complete' if complete else ('partial' if hold_frames or samples else 'no_hold_data')
    evidence = ('mapped_entry_exit_report' if boundary_complete else
                'gps_recovery_after_dr' if not baseline and recovery else 'insufficient')
    exclusions = []
    if expired:
        exclusions.append(EXPIRY)
    if not clean:
        exclusions.append('trial_not_successful:' + str(outcome))
    if not complete:
        exclusions.append('full_tunnel_traversal_unverified')
    if not samples:
        exclusions.append('no_comparison_samples')
    groups = {}
    for method in dict.fromkeys(sample['method'] for sample in samples):
        selected = [sample for sample in samples if sample['method'] == method]
        groups[method] = dict(samples=len(selected), errors=error_stats(selected),
            last_observed_error=selected[-1],
            exit_last_observed_error=selected[-1] if complete else None)
    for sample in samples:
        sample.update(completion=completion, accuracy_aggregate_eligible=not exclusions,
                      result=outcome, exclusion_reasons=';'.join(exclusions))
    warnings = []
    if any(sample['method'] == LEGACY for sample in samples):
        warnings.append('legacy time correction is approximate; rows may repeat the same GPS fix')
    if not baseline:
        warnings.append('no mapped entry/exit report; recovery is weaker traversal evidence')
    warnings.append('GPS is a simulator measurement; receipt alignment does not establish acquisition-time accuracy')
    return dict(trial=trial_root.name, result=outcome, elapsed_s=result.get('elapsed_s'),
        failure_reason=result.get('failure_reason', result.get('reason', runtime_report.get('error'))),
        control_path=str(control) if control else None, completion=completion,
        completion_evidence=evidence, baseline_report_path=baseline.get('report_path'),
        accuracy_aggregate_eligible=not exclusions, exclusions=exclusions,
        gps_recovery_observed=recovery,
        hold_frames=hold_frames, hold_episode_count=sum(t['to_mode'] == 'dead_reckoning'
            for t in transitions) + int(bool(modes.get('dead_reckoning'))
                and (not transitions or transitions[0]['from_mode'] == 'dead_reckoning')),
        hold_s=max(held_s) if held_s else None,
        distance_m=max(distance) if distance else None, actual_speed_kmh=stats(speeds),
        position_mode_frames=dict(modes), transitions=transitions,
        transition_step_note='position steps include motion between log rows; not pure GPS correction',
        comparisons=groups, diagnostics=dict(diagnostics), warnings=warnings), samples


def analyze(root):
    root = Path(root)
    report = read_json(root/'report.json') if root.is_dir() else {}
    if not report and root.is_dir():
        report = read_json(root/'status.json')
    trial_paths = sorted(root.glob('trial_*')) if root.is_dir() else []
    trial_paths = [path for path in trial_paths if path.is_dir()]
    if not trial_paths and (root.is_file() or find_control(root, read_text(root/'console.log'))
                            or (root/'result.json').is_file() or (root/'console.log').is_file()):
        trial_paths = [root]
    recorded_results = {str(row.get('trial')): row for row in report.get('trials', []) if isinstance(row, dict)}
    trials, all_samples = [], []
    for path in trial_paths:
        match = re.search(r'trial_(\d+)$', path.name)
        fallback = recorded_results.get(str(int(match.group(1)))) if match else None
        trial, samples = analyze_trial(path, fallback)
        trials.append(trial)
        all_samples.extend(samples)
    grouped = defaultdict(list)
    eligible = {trial['trial'] for trial in trials if trial['accuracy_aggregate_eligible']}
    states = {trial['trial']: trial['completion'] for trial in trials}
    for sample in all_samples:
        scope = 'eligible_complete' if sample['trial'] in eligible else ('excluded_' + states[sample['trial']])
        grouped[(scope, sample['method'])].append(sample)
    aggregates = []
    for (scope, method), samples in sorted(grouped.items()):
        aggregates.append(dict(scope=scope, method=method,
            trials=sorted({row['trial'] for row in samples}), samples=len(samples), errors=error_stats(samples)))
    exclusions = Counter(reason for trial in trials for reason in trial['exclusions'])
    summary = dict(schema='aimdrive_tunnel_hold_analysis_v2', input=str(root.resolve()),
        requested_runs=report.get('requested'), campaign_status=report.get('status'),
        campaign_failure_reason=report.get('failure_reason', report.get('error')), observed_trials=len(trials),
        eligible_complete_trials=len(eligible),
        completion_counts=dict(Counter(trial['completion'] for trial in trials)),
        excluded_trials=sum(bool(trial['exclusions']) for trial in trials),
        exclusion_reason_counts=dict(exclusions),
        assessment='not_assessed_no_application_tolerance',
        comparison_note='matched_receipt_time aligns DR to GPS host receipt time, not GPS acquisition time; GPS has its own error',
        aggregation_note='methods and incomplete/failed trials are never pooled into eligible complete accuracy',
        trials=trials, aggregates=aggregates)
    return summary, all_samples


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['trial'])
        writer.writeheader()
        writer.writerows(rows)


def plot_samples(samples, path):
    if not samples:
        return 'no comparison samples'
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return 'matplotlib unavailable; JSON and CSV were saved'
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    groups = defaultdict(list)
    for sample in samples:
        groups[(sample['trial'], sample['method'])].append(sample)
    for (trial, method), values in groups.items():
        label = trial + (' (legacy approx)' if method == LEGACY else ' (matched receipt)')
        if not values[0].get('accuracy_aggregate_eligible', False):
            label += ' [excluded]'
        for column, xkey in enumerate(('hold_s', 'distance_m')):
            for index, ykey in enumerate(('longitudinal_m', 'lateral_m', 'error_2d_m')):
                valid = [row for row in values if row[xkey] is not None]
                axes[index, column].plot([row[xkey] for row in valid], [row[ykey] for row in valid], label=label)
                axes[index, column].set_xlabel('DR hold time (s)' if column == 0 else 'DR distance (m)')
                axes[index, column].set_ylabel(ykey)
                axes[index, column].grid(alpha=.3)
    axes[0, 0].legend(fontsize=7)
    figure.suptitle('DR minus withheld GPS (excluded trials remain diagnostic curves)')
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return 'saved'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args(argv)
    if not args.input.exists():
        parser.error('입력 경로가 없습니다: ' + str(args.input))
    summary, samples = analyze(args.input)
    output = args.output_dir or ((args.input if args.input.is_dir() else args.input.parent)/'tunnel_hold_analysis')
    output.mkdir(parents=True, exist_ok=True)
    summary['plot_status'] = ('disabled' if args.no_plot else plot_samples(samples, output/'errors.png'))
    (output/'analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    write_csv(output/'samples.csv', samples)
    write_csv(output/'trials.csv', [{key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
        for key, value in trial.items()} for trial in summary['trials']])
    print('시험 %d회 | 터널 완주 확인 %d회 | 정확도 합산 대상 %d회 | 제외 %d회' % (
        summary['observed_trials'], summary['completion_counts'].get('complete', 0),
        summary['eligible_complete_trials'], summary['excluded_trials']))
    for trial in summary['trials']:
        print('%s: %s / %s | 유지 %ss, %sm | 실제 속도 중앙값 %s km/h' % (
            trial['trial'], trial['result'], trial['completion'], trial['hold_s'], trial['distance_m'],
            round(trial['actual_speed_kmh']['median'], 2) if trial['actual_speed_kmh'] else '?'))
        for method, group in trial['comparisons'].items():
            cells = ['%s %.3f/%.3f/%.3f' % (key, value['max'], value['median'], value['p95'])
                     for key, value in group['errors'].items() if value]
            print('  %s (%d개): 최대/중앙/P95 m | %s' % (method, group['samples'], ' | '.join(cells)))
        if trial['exclusions']:
            print('  합산 제외: ' + ', '.join(trial['exclusions']))
    for group in summary['aggregates']:
        if group['scope'] != 'eligible_complete':
            continue
        values = group['errors']
        print('완주 합산 %s (%d회/%d개) | 2D 최대 %.3fm | 앞뒤 최대 %.3fm | 옆 최대 %.3fm' % (
            group['method'], len(group['trials']), group['samples'], values['error_2d_m']['max'],
            values['longitudinal_abs_m']['max'], values['lateral_abs_m']['max']))
    print('허용오차가 지정되지 않아 GPS 대체 가능 여부는 자동 판정하지 않습니다.')
    print('결과: ' + str(output.resolve()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
