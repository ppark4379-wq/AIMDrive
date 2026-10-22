"""Metrics for an opt-in, route-conditioned deny/allow head.

This module does not interpret a red/not-red head as route permission. Targets
must come from the separately validated route/phase parser. Missing classes or
critical signal conditions remain unmeasured; these diagnostics never certify
deployment readiness or eliminate signal-distance confounding.
"""
import hashlib
import json
import math
from copy import deepcopy
from collections import defaultdict

import numpy as np


POLICY = dict(
    schema='morai_route_permission_metrics_v1',
    class_order=['deny', 'allow'], excluded_target=-1,
    prediction='argmax of two finite logits; exact ties predict deny (index 0)',
    aggregation='frame counts across all batches; no batch-average metrics',
    unsafe_allow_rate='GT deny predicted allow / all valid GT deny',
    false_deny_rate='GT allow predicted deny / all valid GT allow',
    critical_subgroups={'left_straight_green_16': 'movement=left, raw_signal_bits=16, expected deny',
                        'left_arrow_33': 'movement=left, raw_signal_bits=33, expected allow'},
    note='Observed classification diagnostics; not a closed-loop safety or deployment certificate',
)
DISTANCE_BINS = ('behind_stopline', '0_10m', '10_20m', '20_30m', '30_45m', '45m_plus', 'unknown')
_MOVEMENT_ALIASES = {
    'left': 'left', 'left_turn': 'left', 'turn_left': 'left', '좌회전': 'left',
    'straight': 'straight', 'go_straight': 'straight', '직진': 'straight',
    'right': 'right', 'right_turn': 'right', 'turn_right': 'right', '우회전': 'right',
}


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _new_counts():
    return {'frames': 0, 'excluded': 0, 'confusion_matrix': [[0, 0], [0, 0]]}


def _record(counts, target, predicted):
    counts['frames'] += 1
    if target == -1:
        counts['excluded'] += 1
    else:
        counts['confusion_matrix'][target][predicted] += 1


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _report(counts):
    matrix = counts['confusion_matrix']
    dd, da, ad, aa = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    deny_support, allow_support = dd + da, ad + aa
    predicted_deny, predicted_allow = dd + ad, da + aa
    valid = deny_support + allow_support
    deny_recall, allow_recall = _ratio(dd, deny_support), _ratio(aa, allow_support)
    both_classes = bool(deny_support and allow_support)
    constant = ('always_deny' if predicted_deny == valid else
                'always_allow' if predicted_allow == valid else None) if valid else None
    return dict(
        frames=counts['frames'], valid=valid, excluded=counts['excluded'],
        measured=bool(valid), measurement_status='measured' if valid else 'unmeasured',
        confusion_matrix=[list(matrix[0]), list(matrix[1])],
        confusion_matrix_axes='rows=GT [deny,allow]; columns=prediction [deny,allow]',
        classes={
            'deny': dict(support=deny_support, predicted=predicted_deny,
                         recall=deny_recall, precision=_ratio(dd, predicted_deny)),
            'allow': dict(support=allow_support, predicted=predicted_allow,
                          recall=allow_recall, precision=_ratio(aa, predicted_allow)),
        },
        accuracy=_ratio(dd + aa, valid),
        balanced_accuracy=(deny_recall + allow_recall) / 2. if both_classes else None,
        both_target_classes_measured=both_classes,
        predicted_both_classes=bool(predicted_deny and predicted_allow),
        constant_prediction=constant,
        unsafe_allow=da, unsafe_allow_rate=_ratio(da, deny_support),
        false_deny=ad, false_deny_rate=_ratio(ad, allow_support),
    )


def _metadata(row):
    if not isinstance(row, dict):
        raise ValueError('Each route-permission metadata row must be a dict')
    movement = row.get('movement')
    if movement is None:
        movement = 'unknown'
    else:
        movement = _MOVEMENT_ALIASES.get(str(movement).strip().lower(), str(movement).strip().lower() or 'unknown')
    bits = row.get('raw_signal_bits')
    if bits is not None:
        if isinstance(bits, (bool, np.bool_)):
            raise ValueError('raw_signal_bits must be a nonnegative integer or None')
        try:
            integer_bits = int(bits)
            if float(bits) != integer_bits or integer_bits < 0:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise ValueError('raw_signal_bits must be a nonnegative integer or None')
        bits = integer_bits
    distance = row.get('stopline_distance_m')
    if distance is not None:
        try:
            distance = float(distance)
        except (TypeError, ValueError, OverflowError):
            raise ValueError('stopline_distance_m must be finite or None')
        if not math.isfinite(distance):
            raise ValueError('stopline_distance_m must be finite or None')
    distance_bin = ('unknown' if distance is None else 'behind_stopline' if distance < 0 else
                    '0_10m' if distance < 10 else '10_20m' if distance < 20 else
                    '20_30m' if distance < 30 else '30_45m' if distance < 45 else '45m_plus')
    return dict(
        movement=movement, raw_signal_bits=bits,
        raw_state='unknown' if bits is None else str(bits),
        light_id=str(row.get('light_id') or 'unknown'),
        scenario_id=str(row.get('scenario_id') or row.get('scenario_name') or 'unknown'),
        stopline_distance_m=distance, distance_bin=distance_bin,
        manifest_path=str(row.get('manifest_path') or ''),
        frame_id=str(row.get('frame_id', '')), sensor_time_ns=str(row.get('sensor_time_ns', '')),
    )


class RoutePermissionAccumulator:
    """Accumulate deny/allow predictions and condition diagnostics on GT metadata."""
    def __init__(self):
        self.total = _new_counts()
        self.by_movement = defaultdict(_new_counts)
        self.by_raw_signal_bits = defaultdict(_new_counts)
        self.by_light_id = defaultdict(_new_counts)
        self.by_distance = defaultdict(_new_counts)
        self.by_scenario_id = defaultdict(_new_counts)
        self.joint = defaultdict(lambda: defaultdict(lambda: defaultdict(_new_counts)))
        self.critical = {'left_straight_green_16': _new_counts(), 'left_arrow_33': _new_counts()}
        self.input_digest = hashlib.sha256()

    def update(self, logits, targets, metadata):
        scores, labels = _numpy(logits), _numpy(targets)
        if scores.ndim != 2 or scores.shape[1] != 2:
            raise ValueError('Route-permission logits must have shape [N,2]')
        if labels.ndim != 1 or labels.shape[0] != scores.shape[0]:
            raise ValueError('Route-permission targets must have shape [N] matching logits')
        if not np.isin(labels, [-1, 0, 1]).all():
            raise ValueError('Route-permission targets must be -1, 0 or 1')
        if not isinstance(metadata, (list, tuple)) or len(metadata) != len(labels):
            raise ValueError('Metadata length must match route-permission batch')
        if not np.isfinite(scores[labels != -1]).all():
            raise ValueError('Nonfinite route-permission logits on supervised frames')
        # Validate the complete batch before changing counts or the input digest.
        rows = [_metadata(row) for row in metadata]
        for original, label in zip(metadata, labels):
            if 'target' in original and original['target'] != int(label):
                raise ValueError('Route-permission target differs from aligned metadata')
            if 'valid' in original and bool(original['valid']) != (int(label) != -1):
                raise ValueError('Route-permission validity mask differs from aligned metadata')
        predicted = np.argmax(scores, axis=1)  # numpy returns index 0 on exact ties.
        for row, target, prediction in zip(rows, labels, predicted):
            target, prediction = int(target), int(prediction)
            _record(self.total, target, prediction)
            for groups, key in ((self.by_movement, 'movement'),
                                (self.by_raw_signal_bits, 'raw_state'),
                                (self.by_light_id, 'light_id'),
                                (self.by_distance, 'distance_bin'),
                                (self.by_scenario_id, 'scenario_id')):
                _record(groups[row[key]], target, prediction)
            _record(self.joint[row['movement']][row['raw_state']][row['distance_bin']], target, prediction)
            if row['movement'] == 'left':
                if row['raw_signal_bits'] == 16:
                    _record(self.critical['left_straight_green_16'], target, prediction)
                elif row['raw_signal_bits'] == 33:
                    _record(self.critical['left_arrow_33'], target, prediction)
            identity = dict(row, target=target)
            self.input_digest.update((json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                                 ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))

    def compute(self):
        critical = {}
        for name, target in (('left_straight_green_16', 0), ('left_arrow_33', 1)):
            row = _report(self.critical[name])
            row['expected_target'] = target
            row['label_phase_conflicts'] = row['classes']['allow' if target == 0 else 'deny']['support']
            critical[name] = row
        joint = {movement: {bits: {distance: _report(counts)
                                   for distance, counts in sorted(distances.items())}
                            for bits, distances in sorted(states.items())}
                 for movement, states in sorted(self.joint.items())}
        left = joint.get('left', {})
        shared_bins = []
        distance_counts = {}
        for distance in DISTANCE_BINS:
            supports = {bits: left.get(bits, {}).get(distance, {}).get('valid', 0) for bits in ('16', '33')}
            distance_counts[distance] = supports
            if distance not in ('unknown', 'behind_stopline') and supports['16'] and supports['33']:
                shared_bins.append(distance)
        measured_both = all(critical[name]['measured'] for name in critical)
        return dict(
            policy=deepcopy(POLICY), **_report(self.total),
            input_order_sha256=self.input_digest.hexdigest(),
            by_movement={key: _report(value) for key, value in sorted(self.by_movement.items())},
            by_raw_signal_bits={key: _report(value) for key, value in sorted(self.by_raw_signal_bits.items())},
            by_light_id={key: _report(value) for key, value in sorted(self.by_light_id.items())},
            by_stopline_distance={key: _report(value) for key, value in sorted(self.by_distance.items())},
            by_scenario_id={key: _report(value) for key, value in sorted(self.by_scenario_id.items())},
            by_movement_raw_state_distance=joint, critical_subgroups=critical,
            distance_confound=dict(
                left_16_and_33_valid_frames_by_distance=distance_counts,
                shared_approach_distance_bins=shared_bins,
                measurement_status=('unmeasured' if not measured_both else
                                    'coarse_overlap_observed' if shared_bins else 'no_measured_approach_overlap'),
                confounding_ruled_out=False,
                note='Phase and stop-line distance may correlate. Shared coarse bins do not prove visual signal use; no counterfactual or matched-distance certification was performed.',
            ),
            deployment_ready=False,
        )


def selection_decision(candidate, baseline=None):
    """Coverage eligibility and an experimental ranking, never a safety gate.

Ranking: balanced accuracy, then fewer left-on-straight-green false allows,
    then fewer overall false allows. At least one GT observation of each class
    and each critical condition is necessary for measurement, not statistical
    sufficiency. A model predicting only deny or only allow is not eligible.
    """
    if baseline is not None:
        for key in ('policy', 'input_order_sha256', 'frames', 'valid', 'excluded'):
            if candidate[key] != baseline[key]:
                raise ValueError('Route-permission selection requires identical input coverage/policy: ' + key)
    violations = []
    if candidate['policy'] != POLICY:
        raise ValueError('Unsupported route-permission metric policy')
    for label in ('deny', 'allow'):
        if not candidate['classes'][label]['support']:
            violations.append(dict(reason='unmeasured_target_class', target_class=label))
    if not candidate['predicted_both_classes']:
        violations.append(dict(reason='missing_predicted_class', constant_prediction=candidate['constant_prediction']))
    for name in ('left_straight_green_16', 'left_arrow_33'):
        group = candidate['critical_subgroups'][name]
        if not group['measured']:
            violations.append(dict(reason='unmeasured_critical_subgroup', subgroup=name))
        if group['label_phase_conflicts']:
            violations.append(dict(reason='label_phase_conflict', subgroup=name, count=group['label_phase_conflicts']))
    balanced = candidate['balanced_accuracy']
    left_false_allow = candidate['critical_subgroups']['left_straight_green_16']['unsafe_allow_rate']
    global_false_allow = candidate['unsafe_allow_rate']
    score = ([balanced, -left_false_allow, -global_false_allow]
             if all(value is not None and math.isfinite(value)
                    for value in (balanced, left_false_allow, global_false_allow)) else None)
    return dict(
        eligible=not violations and score is not None, violations=violations, score=score,
        score_policy='lexicographic balanced accuracy, negative left+16 unsafe allow rate, negative overall unsafe allow rate',
        coverage_note='Nonempty class/subgroup support is measurement eligibility, not a statistical quality threshold.',
        deployment_ready=False,
    )
