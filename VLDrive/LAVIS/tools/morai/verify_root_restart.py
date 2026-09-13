#!/usr/bin/env python3
"""Validate the two real-model zero-update reports before bounded training."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError('Use a new verification artifact inside LAVIS')
    reports = [json.loads((path / 'report.json').read_text()) for path in (args.baseline, args.candidate)]
    preflights = [json.loads((path / 'preflight.json').read_text()) for path in (args.baseline, args.candidate)]
    baseline, candidate = reports
    checks = {
        'both_full_model_preflights_passed': all(p['passed'] for p in preflights),
        'both_performed_zero_updates': all(r['optimizer_updates'] == 0 for r in reports),
        'ten_real_history_frames': all(p['real_history_frames'] == 10 for p in preflights),
        'future_to_past_exact_zero': all(p['future_to_past_max_delta'] == 0 for p in preflights),
        'same_initial_weights': baseline['initial_trainable_sha256'] == candidate['initial_trainable_sha256'],
        'same_configuration_except_explicit_arm': baseline['controlled_settings'] == candidate['controlled_settings'],
        'same_source_data_backend': baseline['provenance'] == candidate['provenance'],
        'same_draws_and_evaluation_windows': baseline['sampling'] == candidate['sampling'],
        'actual_forward_exactly_equal': baseline['initial_task_probe']['trajectory'] == candidate['initial_task_probe']['trajectory'],
    }
    first, second = baseline['initial_task_probe'], candidate['initial_task_probe']
    for key in ('selector_weighted_pruning_l2', 'selector_weighted_reconstruction_l2'):
        checks[key + '_unchanged'] = math.isclose(first[key], second[key], rel_tol=1e-6, abs_tol=1e-10)
    ratio = second['selector_waypoint_l2'] / max(first['selector_waypoint_l2'], 1e-12)
    checks['actual_waypoint_selector_gradient_improved'] = math.isfinite(ratio) and ratio >= 10.
    result = dict(passed_for_bounded_pilot=all(checks.values()), checks=checks,
        selector_waypoint_gradient_baseline=first['selector_waypoint_l2'],
        selector_waypoint_gradient_candidate=second['selector_waypoint_l2'],
        selector_waypoint_gradient_ratio=ratio,
        baseline=str(args.baseline.resolve()), candidate=str(args.candidate.resolve()),
        next_stage='matched train-only A/B; not full training or performance certification',
        note='Task-gradient magnitude alone is insufficient. The pilot must still learn dynamic retention and turning.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))
    if not result['passed_for_bounded_pilot']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
