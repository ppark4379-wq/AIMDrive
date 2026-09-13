"""Train-only gates for a controlled, one-factor CCDP backward experiment."""
import math


def category(points):
    if max(math.hypot(*point) for point in points) < .5:
        return 'stationary'
    if points[-1][0] >= 2.:
        return 'moving_left'
    if points[-1][0] <= -2.:
        return 'moving_right'
    return 'moving_straight'


def summarize(rows):
    result = {}
    for group in ('all', 'stationary', 'moving_straight', 'moving_left', 'moving_right', 'curved'):
        selected = [r for r in rows if group == 'all' or r['category'] == group
                    or group == 'curved' and r['category'] in ('moving_left', 'moving_right')]
        if not selected:
            continue
        count = len(selected)
        result[group] = dict(windows=count,
            ade_m=sum(sum(r['errors_m']) / 5 for r in selected) / count,
            fde_m=sum(r['errors_m'][-1] for r in selected) / count,
            lateral_error_m=sum(abs(r['prediction'][-1][0] - r['actual_future'][-1][0]) for r in selected) / count,
            direction_accuracy=sum(r['prediction'][-1][0] * r['actual_future'][-1][0] > 0
                                   and abs(r['prediction'][-1][0]) >= .5 for r in selected) / count)
    return result


def gate(before, after, telemetry, limits, updates):
    """An enforced output count alone can never pass this gate."""
    curved = after.get('curved', {})
    tests = dict(
        enough_updates=updates >= limits['minimum_updates'],
        both_turns=all(k in after for k in ('moving_left', 'moving_right')),
        curved_ade=curved.get('ade_m', math.inf) <= limits['curved_ade_max_m'],
        curved_fde=curved.get('fde_m', math.inf) <= limits['curved_fde_max_m'],
        curved_lateral=curved.get('lateral_error_m', math.inf) <= limits['curved_lateral_max_m'],
        curved_improves=curved.get('fde_m', math.inf) <= before['curved']['fde_m'] * limits['curved_ratio_max'],
        stationary=after.get('stationary', {}).get('fde_m', math.inf) <= limits['stationary_fde_max_m'],
        straight=after.get('moving_straight', {}).get('ade_m', math.inf) <= limits['straight_ade_max_m'],
        left_direction=after.get('moving_left', {}).get('direction_accuracy', 0.) >= limits['direction_min'],
        right_direction=after.get('moving_right', {}).get('direction_accuracy', 0.) >= limits['direction_min'],
        sampled_retention=limits['retention_min'] <= telemetry['sampled_retention'] <= limits['retention_max'],
        soft_retention=limits['retention_min'] <= telemetry['soft_retention'] <= limits['retention_max'],
        inference_dynamic=limits['retention_min'] <= telemetry['inference_retention'] <= limits['retention_max'],
        unsaturated=telemetry['saturated_keep_fraction'] <= limits['saturation_max'],
    )
    return dict(passed=all(tests.values()), checks=tests, telemetry=telemetry)


def compare(baseline, candidate, minimum_gradient_ratio=10.):
    """Fail closed on unequal initialization/data or a count-only improvement."""
    matched = all(baseline[k] == candidate[k] for k in
                  ('initial_trainable_sha256', 'sampling', 'seed', 'optimizer', 'schedule_updates',
                   'provenance', 'updates', 'controlled_settings'))
    same_forward = baseline['initial_task_probe']['trajectory'] == candidate['initial_task_probe']['trajectory']
    aux_same = all(math.isclose(baseline['initial_task_probe'][key], candidate['initial_task_probe'][key],
                               rel_tol=1e-6, abs_tol=1e-10) for key in
                   ('selector_weighted_pruning_l2', 'selector_weighted_reconstruction_l2'))
    old = baseline['initial_task_probe']['selector_waypoint_l2']
    new = candidate['initial_task_probe']['selector_waypoint_l2']
    gradient_ok = math.isfinite(new) and new > 0 and new >= max(old, 1e-12) * minimum_gradient_ratio
    candidate_better = candidate['after']['curved']['fde_m'] <= baseline['after']['curved']['fde_m'] * .95
    selected = None
    if matched and same_forward and aux_same and baseline['frozen_unchanged'] and candidate['frozen_unchanged']:
        if candidate['gate']['passed'] and gradient_ok and candidate_better:
            selected = 'retained_output_st'
        elif baseline['gate']['passed']:
            selected = 'legacy'
    return dict(passed=selected is not None, selected_mode=selected,
                matched_experiment=matched, identical_initial_forward=same_forward,
                auxiliary_selector_gradient_unchanged=aux_same, auxiliary_gradient_relative_tolerance=1e-6,
                task_gradient_recovered=gradient_ok, candidate_curved_fde_improves_5pct=candidate_better,
                selector_gradient_ratio=new / max(old, 1e-12),
                note='A pass permits the next bounded encoder stage, not a driving-safety claim. '
                     'The ST candidate is an independent backward surrogate, not verified author code.')
