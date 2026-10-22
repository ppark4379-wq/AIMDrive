"""Opt-in object/signal checkpoint selection for the existing LAV head.

Every validation sensor is forwarded once. Decoded speed uses the same fixed
confidence/NMS/IoU matches as bbox diagnostics, with explicit missing-speed and
matching coverage. Thresholds are experiment regression tolerances, not a
claim of road safety. Waypoint errors never enter eligibility or ranking.
"""
import math

import numpy as np

from .morai_scenario_validation import scenario_loaders
from .morai_transfer_metrics import evaluate


SCENARIOS = {
    'full_loop_dynamic': 'dynamic_obstacle',
    'full_loop_static': 'static_obstacle',
    'full_loop_roundabout': 'roundabout',
    'full_loop_highway': 'highway',
    'full_loop_normal': 'basic_driving',
    'full_loop_tunnel_static': 'tunnel_static',
    'full_loop_tunnel_normal': 'tunnel_normal',
    'full_loop_traffic_light': 'traffic_light',
}
DEFAULT_LIMITS = dict(
    cell_ap_drop=.02, cell_recall_drop=.03,
    object_recall_drop=.03, object_precision_drop=.03,
    geometry_relative_increase=.10, geometry_absolute_increase_m=.05,
    speed_relative_increase=.10, speed_absolute_increase_mps=.20,
    speed_coverage_drop=.03, invalid_speed_fraction_increase=.01,
    light_accuracy_drop=.01, light_class_recall_drop=.02,
    empty_false_boxes_per_frame_increase=.005,
)
SPEED_POLICY = dict(
    schema='morai_matched_scalar_speed_v1', prediction='raw_detection_channel_7_times_8_mps',
    ground_truth='hypot(vel_x, vel_y) in manifest world frame; not relative velocity',
    matching_iou_thresholds=['0.3', '0.5'],
    match_source='same confidence-0.5 NMS boxes and one-to-one matches as object_detection',
    stationary_max_mps=.25, moving_min_mps=.8,
    negative_prediction='count invalid for adapter use, but retain finite raw signed value in MAE/P95; no clamping',
    vehicle_scope='GT instance_key contains OBJECT_TYPE_VEHICLE, not a predicted class',
    denominator='frame-object observations, no visibility filter; MAE/P95 conditional on finite matched raw speed including negative values',
)
SPEED_GROUPS = ('all', 'stationary', 'slow', 'moving', 'vehicle',
                'vehicle_stationary', 'vehicle_slow', 'vehicle_moving')


def _speed_group():
    return dict(gt_boxes=0, valid_gt_speed=0, invalid_gt_speed=0, matched_boxes=0,
                invalid_prediction_speed=0, negative_prediction_speed=0,
                nonfinite_prediction_speed=0, valid_matched_speed=0, errors=[])


class MatchedSpeedAccumulator:
    """Exact speed percentiles using existing object matches, without another forward."""
    def __init__(self):
        self.frames = 0
        self.supervised_frames = 0
        self.prediction_cells = dict(confidence_selected=0, nonfinite_speed=0, negative_speed=0)
        self.groups = {threshold: {name: _speed_group() for name in SPEED_GROUPS}
                       for threshold in ('0.3', '0.5')}

    def update(self, prediction, entry, object_row):
        self.frames += 1
        if not object_row['supervised']:
            return
        self.supervised_frames += 1
        if hasattr(prediction, 'detach'):
            prediction = prediction.detach().cpu().numpy()
        prediction = np.asarray(prediction)
        selected = np.isfinite(prediction[:, 0]) & (prediction[:, 0] >= 0.)
        raw_cell_speeds = prediction[selected, 7] * 8.
        self.prediction_cells['confidence_selected'] += int(selected.sum())
        self.prediction_cells['nonfinite_speed'] += int((~np.isfinite(raw_cell_speeds)).sum())
        self.prediction_cells['negative_speed'] += int((np.isfinite(raw_cell_speeds) & (raw_cell_speeds < 0.)).sum())
        for row in object_row['gt_diagnostics']:
            obj = entry['objects'][row['object_index']]
            try:
                speed = math.hypot(float(obj['vel_x']), float(obj['vel_y']))
                valid_gt = math.isfinite(speed)
            except (KeyError, TypeError, ValueError):
                speed, valid_gt = None, False
            names = ['all']
            vehicle = 'OBJECT_TYPE_VEHICLE' in str(obj.get('instance_key', ''))
            if vehicle:
                names.append('vehicle')
            if valid_gt:
                motion = 'stationary' if speed <= .25 else 'moving' if speed >= .8 else 'slow'
                names.append(motion)
                if vehicle:
                    names.append('vehicle_' + motion)
            for threshold, match in row['matches'].items():
                raw = (float(prediction[match['prediction_cell_index'], 7]) * 8.
                       if match is not None else None)
                for name in names:
                    group = self.groups[threshold][name]
                    group['gt_boxes'] += 1
                    group['valid_gt_speed' if valid_gt else 'invalid_gt_speed'] += 1
                    if match is None:
                        continue
                    group['matched_boxes'] += 1
                    if not math.isfinite(raw) or raw < 0:
                        group['invalid_prediction_speed'] += 1
                        group['nonfinite_prediction_speed' if not math.isfinite(raw)
                              else 'negative_prediction_speed'] += 1
                    if math.isfinite(raw) and valid_gt:
                        group['errors'].append(abs(raw - speed))
                        if raw >= 0:
                            group['valid_matched_speed'] += 1

    def compute(self):
        thresholds = {}
        for threshold, groups in self.groups.items():
            thresholds[threshold] = {}
            for name, group in groups.items():
                errors = group['errors']
                matched, support = group['matched_boxes'], group['gt_boxes']
                thresholds[threshold][name] = dict(
                    **{k: v for k, v in group.items() if k != 'errors'},
                    unmatched_gt_boxes=support - matched,
                    finite_matched_speed=len(errors),
                    matched_recall=matched / support if support else None,
                    finite_speed_coverage=len(errors) / group['valid_gt_speed'] if group['valid_gt_speed'] else None,
                    valid_speed_coverage=group['valid_matched_speed'] / group['valid_gt_speed'] if group['valid_gt_speed'] else None,
                    invalid_prediction_fraction=group['invalid_prediction_speed'] / matched if matched else None,
                    mae_mps=float(np.mean(errors)) if errors else None,
                    p95_mps=float(np.percentile(errors, 95)) if errors else None,
                )
        return dict(policy=dict(SPEED_POLICY), frames=self.frames,
                    supervised_frames=self.supervised_frames, thresholds=thresholds,
                    prediction_cells_before_geometry_rejection_and_nms=dict(self.prediction_cells))


def scenario_groups(dataset, manifest_scenarios):
    groups = {}
    for index, path in enumerate(dataset.row_manifest_paths):
        scenario = manifest_scenarios.get(path)
        if scenario not in SCENARIOS:
            raise ValueError('Missing/unknown perception publication scenario: ' + str(scenario))
        groups.setdefault(SCENARIOS[scenario], []).append(index)
    if not groups:
        raise ValueError('Empty perception validation groups')
    return groups


def _macro(reports, names, accessor):
    values = [accessor(reports[name]) for name in names]
    if not values or any(value is None or not math.isfinite(value) for value in values):
        raise ValueError('Missing supported perception ranking metric')
    return sum(values) / len(values)


def evaluate_scenarios(model, loaders, device, *, progress_phase=None,
                       detection_loss_policy='lmdrive_lav_v1', waypoint_loss_weight=0.0,
                       **kwargs):
    if waypoint_loss_weight != 0:
        raise ValueError('Perception-guarded evaluation requires waypoint loss weight zero')
    if not loaders or set(loaders) - set(SCENARIOS.values()):
        raise ValueError('Unknown/empty perception scenario loaders')
    reports = {}
    from utils.common.runtime import deterministic_pointpillar_reductions
    for name, loader in loaders.items():
        subset = loader.dataset
        entries = [subset.dataset.rows[i][1] for i in subset.indices]
        with deterministic_pointpillar_reductions():
            reports[name] = evaluate(
                model, loader, device, progress_phase=(progress_phase or 'evaluation') + ':' + name,
                detection_loss_policy=detection_loss_policy, object_entries=entries,
                waypoint_loss_weight=0.0, perception_metrics=True, **kwargs)
    frames = sum(row['frames'] for row in reports.values())
    object_groups = sorted(name for name, row in reports.items() if row['object_detection']['gt_boxes'])
    if not object_groups:
        raise ValueError('Perception selection requires positive object GT')
    return dict(
        frames=frames, loss=sum(row['loss'] * row['frames'] for row in reports.values()) / frames,
        per_scenario=reports, object_score_scenarios=object_groups,
        numerical_policy='FP32 eval with scoped deterministic PointPillar CSR reductions when torch_scatter is available',
        macro_detection_cell_average_precision=_macro(reports, object_groups,
            lambda row: row['detection_cell_average_precision']),
        macro_detection_cell_recall=_macro(reports, object_groups,
            lambda row: row['detection_cell_recall']),
        macro_object_f1_iou_0_5=_macro(reports, object_groups,
            lambda row: row['object_detection']['thresholds']['0.5']['f1'] or 0.),
        macro_object_recall_iou_0_5=_macro(reports, object_groups,
            lambda row: row['object_detection']['thresholds']['0.5']['recall']),
        selection_scope='Equal scenario weight for object IoU0.5 F1/recall ranking; per-scenario bbox, decoded speed and signal regression guards. No waypoint selection. Not independent test.',
    )


def selection_decision(candidate, baseline, limits=None):
    limits = dict(DEFAULT_LIMITS if limits is None else limits)
    if set(limits) != set(DEFAULT_LIMITS) or any(not math.isfinite(v) or v < 0 for v in limits.values()):
        raise ValueError('Invalid perception regression limits')
    scenarios = set(baseline['per_scenario'])
    if set(candidate['per_scenario']) != scenarios or not scenarios:
        raise ValueError('Selection requires identical nonempty scenario coverage')
    violations = []

    def finite(value):
        return value is not None and math.isfinite(value)

    def lower(name, metric, current, previous, allowance):
        if finite(previous) and (not finite(current) or current < previous - allowance):
            violations.append(dict(scenario=name, metric=metric, baseline=previous,
                                   candidate=current, minimum=previous - allowance))

    def upper(name, metric, current, previous, absolute, relative=0.):
        if not finite(previous):
            return
        ceiling = previous + max(absolute, abs(previous) * relative)
        if not finite(current) or current > ceiling:
            violations.append(dict(scenario=name, metric=metric, baseline=previous,
                                   candidate=current, maximum=ceiling))

    for name in sorted(scenarios):
        c, b = candidate['per_scenario'][name], baseline['per_scenario'][name]
        for key in ('frames', 'detection_observed_frames', 'traffic_light_observed_frames',
                    'traffic_light_class_counts', 'detection_loss_policy'):
            if c[key] != b[key]:
                raise ValueError('Selection requires identical validation coverage/policy: ' + key)
        lower(name, 'cell_ap', c['detection_cell_average_precision'], b['detection_cell_average_precision'], limits['cell_ap_drop'])
        lower(name, 'cell_recall', c['detection_cell_recall'], b['detection_cell_recall'], limits['cell_recall_drop'])
        for metric in ('box_center_l2_m', 'box_length_mae_m', 'box_width_mae_m'):
            upper(name, metric, c[metric], b[metric], limits['geometry_absolute_increase_m'], limits['geometry_relative_increase'])
        lower(name, 'traffic_light_accuracy', c['traffic_light_accuracy'], b['traffic_light_accuracy'], limits['light_accuracy_drop'])
        for label in ('red', 'green'):
            lower(name, 'traffic_light_' + label + '_recall', c['traffic_light_class_recall'][label],
                  b['traffic_light_class_recall'][label], limits['light_class_recall_drop'])
        co, bo = c['object_detection'], b['object_detection']
        if any(co[key] != bo[key] for key in ('policy', 'gt_boxes', 'empty_frames')):
            raise ValueError('Selection requires identical object policy and GT')
        for threshold in ('0.3', '0.5'):
            if bo['gt_boxes']:
                for metric in ('recall', 'precision'):
                    current, previous = co['thresholds'][threshold][metric], bo['thresholds'][threshold][metric]
                    # No predictions with existing GT is zero precision for ranking/guards,
                    # never an undefined value that escapes a regression guard.
                    lower(name, 'object_' + metric + '_iou_' + threshold,
                          0. if current is None else current, 0. if previous is None else previous,
                          limits['object_' + metric + '_drop'])
                for metric in ('center_l2_m', 'length_abs_error_m', 'width_abs_error_m'):
                    upper(name, 'matched_' + metric + '_iou_' + threshold,
                          co['overall_gt']['matching'][threshold]['matched_geometry_mean'][metric],
                          bo['overall_gt']['matching'][threshold]['matched_geometry_mean'][metric],
                          limits['geometry_absolute_increase_m'], limits['geometry_relative_increase'])
        upper(name, 'empty_false_boxes_per_frame', co['false_positive_boxes_per_empty_frame'],
              bo['false_positive_boxes_per_empty_frame'], limits['empty_false_boxes_per_frame_increase'])
        cs, bs = c['object_speed_decoded'], b['object_speed_decoded']
        if cs['policy'] != bs['policy']:
            raise ValueError('Selection requires identical speed diagnostic policy')
        for threshold in ('0.3', '0.5'):
            for motion in SPEED_GROUPS:
                cc, bb = cs['thresholds'][threshold][motion], bs['thresholds'][threshold][motion]
                for key in ('gt_boxes', 'valid_gt_speed', 'invalid_gt_speed'):
                    if cc[key] != bb[key]:
                        raise ValueError('Selection requires identical speed GT coverage')
                prefix = 'speed_' + motion + '_iou_' + threshold + '_'
                if cc['invalid_gt_speed']:
                    violations.append(dict(scenario=name, metric=prefix + 'invalid_gt_speed',
                                           candidate=cc['invalid_gt_speed'], maximum=0))
                lower(name, prefix + 'matched_recall', cc['matched_recall'], bb['matched_recall'], limits['speed_coverage_drop'])
                lower(name, prefix + 'valid_coverage', cc['valid_speed_coverage'], bb['valid_speed_coverage'], limits['speed_coverage_drop'])
                # With newly recovered matches, no baseline matched denominator exists;
                # require all of their speeds to be valid rather than exempting them.
                if cc['matched_boxes']:
                    upper(name, prefix + 'invalid_fraction', cc['invalid_prediction_fraction'],
                          bb['invalid_prediction_fraction'] or 0., limits['invalid_speed_fraction_increase'])
                for metric in ('mae_mps', 'p95_mps'):
                    upper(name, prefix + metric, cc[metric], bb[metric],
                          limits['speed_absolute_increase_mps'], limits['speed_relative_increase'])
    expected_groups = sorted(name for name in scenarios if baseline['per_scenario'][name]['object_detection']['gt_boxes'])
    if not expected_groups or candidate['object_score_scenarios'] != expected_groups or baseline['object_score_scenarios'] != expected_groups:
        raise ValueError('Selection requires identical supported object ranking groups')
    # Recompute from scenario reports so caller-supplied macro fields cannot hide a regression.
    score = [
        _macro(candidate['per_scenario'], expected_groups,
               lambda row: row['object_detection']['thresholds']['0.5']['f1'] or 0.),
        _macro(candidate['per_scenario'], expected_groups,
               lambda row: row['object_detection']['thresholds']['0.5']['recall']),
    ]
    # Keep the supported scenario set fixed by the baseline. A candidate cannot
    # improve this conditional error by silently dropping a whole scenario.
    # Existing matched-recall, speed-coverage and invalid-value guards still
    # apply before any score can promote a checkpoint.
    speed_scenarios = sorted(name for name in scenarios if finite(
        baseline['per_scenario'][name]['object_speed_decoded']['thresholds']['0.3']['all']['mae_mps']))
    speed_values = [candidate['per_scenario'][name]['object_speed_decoded']['thresholds']['0.3']['all']['mae_mps']
                    for name in speed_scenarios]
    if not speed_values:
        score.append(0.)  # Explicitly inactive when baseline has no matched speed support.
    elif all(finite(value) for value in speed_values):
        score.append(-sum(speed_values) / len(speed_values))
    else:
        # Missing conditional speed already violates a per-scenario MAE guard.
        # JSON null avoids inventing an error value or serializing infinity.
        if not violations:
            raise ValueError('Missing supported speed tie-break metric')
        score.append(None)
    return dict(eligible=not violations, violations=violations, score=score, limits=limits,
                speed_tiebreak_scenarios=speed_scenarios,
                speed_tiebreak_policy='negative equal-scenario macro raw signed speed MAE of all IoU0.3 matched objects; fixed baseline-finite scenario support; matched coverage and invalid guards retained; zero if no baseline support',
                score_policy='lexicographic macro object F1 IoU0.5, macro object recall IoU0.5, negative macro matched speed MAE IoU0.3; speed/signal guarded; waypoint excluded')
