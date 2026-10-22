"""Recorded signal-phase waypoint diagnostics from the evaluation forward pass.

The two-class head predicts pure signal color, not route permission. Future
waypoints can include a departure after the current signal changes; source-time
masks keep those forecasts out of prerelease holding diagnostics.
"""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import torch


TARGET_LIGHT_ID = 'C1256W000052'
WAYPOINT_OFFSETS_SEC = (.5, 1., 1.5, 2., 2.5)
STOP_SPEED_MPS = .3
STATIONARY_FUTURE_M = .05
DISPLACEMENT_TOLERANCES_M = (.1, .5, 1.)
GROUPS = ('target_red_moving', 'target_red_stopped',
          'green_no_left_stable_before_release', 'green_horizon_overlaps_release',
          'left_arrow_startup', 'left_arrow_moving', 'other_context')


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError('Invalid ' + name)
    return value


def _bounds(transition):
    if transition.get('light_id') != TARGET_LIGHT_ID:
        raise ValueError('Arrow transition belongs to a different signal')
    if transition.get('from_color') != 16 or transition.get('to_color') != 33:
        raise ValueError('Expected a straight-green to left-arrow transition')
    lower = _integer(transition.get('lower_source_ns'), 'transition lower source time')
    upper = _integer(transition.get('upper_source_ns'), 'transition upper source time')
    if lower >= upper:
        raise ValueError('Invalid arrow transition uncertainty interval')
    return lower, upper


def load_arrow_transition(episode):
    """Verify sealed transport evidence and return its unique C052 16 -> 33 event."""
    folder = Path(episode) / 'traffic_audit_v1'
    seal_path = folder / 'COMPLETE.json'
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    members = seal.get('files')
    if not isinstance(members, dict) or not {'report.json', 'transitions.json'} <= members.keys():
        raise ValueError('Traffic audit seal is missing report/transition members')
    for name, expected in members.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError('Unsafe traffic audit seal member')
        if _digest(folder / name) != expected:
            raise ValueError('Changed sealed traffic evidence: ' + name)
    report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    if (report.get('schema') != 'morai_sensor_record_direct_traffic_audit_v1'
            or report.get('structural_valid') is not True
            or report.get('training_visible') is not False
            or not seal.get('raw_manifest_sha256')
            or report.get('raw_manifest_sha256') != seal['raw_manifest_sha256']):
        raise ValueError('Traffic audit is not valid sealed source evidence')
    events = json.loads((folder / 'transitions.json').read_text(encoding='utf-8'))
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ValueError('Malformed traffic transitions')
    selected = [event for event in events if event.get('id') == TARGET_LIGHT_ID
                and event.get('from_color') == 16 and event.get('to_color') == 33]
    if len(selected) != 1:
        raise ValueError('Expected exactly one target-signal 16 -> 33 transition')
    event = selected[0]
    interval = event.get('uncertainty_source_ns')
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError('Invalid arrow transition uncertainty interval')
    result = dict(light_id=TARGET_LIGHT_ID, from_color=16, to_color=33,
                  lower_source_ns=interval[0], upper_source_ns=interval[1],
                  clock='sensor_record_simulation_time',
                  sensor_anchor_indices=event.get('sensor_anchor_indices', []),
                  seal_path=str(seal_path.resolve()), seal_sha256=_digest(seal_path),
                  transitions_sha256=members['transitions.json'],
                  report_sha256=members['report.json'],
                  raw_manifest_sha256=seal['raw_manifest_sha256'])
    _bounds(result)
    return result


def annotate_entry(entry, transition, offsets=WAYPOINT_OFFSETS_SEC):
    """Select context by current bound GT, using clocks rather than model errors."""
    lower, upper = _bounds(transition)
    if (len(offsets) != 5 or any(not math.isfinite(v) or v <= 0 for v in offsets)
            or any(a >= b for a, b in zip(offsets, offsets[1:]))):
        raise ValueError('Expected five increasing positive waypoint time offsets')
    stamp = _integer(entry.get('sensor_time_ns'), 'sensor time')
    _integer(entry.get('frame_id'), 'frame ID')
    speed = entry.get('ego_pose', {}).get('speed')
    if (not isinstance(speed, (int, float)) or isinstance(speed, bool)
            or not math.isfinite(speed) or speed < 0):
        raise ValueError('Invalid recorded GT speed')
    waypoints = torch.tensor(entry.get('waypoints'), dtype=torch.float64)
    if waypoints.shape != (5, 2) or not bool(torch.isfinite(waypoints).all()):
        raise ValueError('Invalid recorded GT waypoint shape/values')
    distances = waypoints.norm(dim=-1)
    permission = entry.get('route_permission', {})
    signal = entry.get('traffic_signal', {})
    eligible = (permission.get('valid') is True
                and permission.get('light_id') == TARGET_LIGHT_ID
                and permission.get('signal_movement') == 'left')
    raw = permission.get('raw_signal_bits')
    before = [stamp + round(offset * 1_000_000_000) < lower for offset in offsets]
    after = [stamp + round(offset * 1_000_000_000) > upper for offset in offsets]
    group = 'other_context'
    if eligible:
        if (signal.get('light_id') != TARGET_LIGHT_ID
                or signal.get('signal_movement') != 'left'
                or signal.get('raw_signal_bits') != raw
                or signal.get('permission_valid') is not True
                or signal.get('signal_permission') is not permission.get('target')):
            raise ValueError('Signal and route-permission GT disagree')
        if raw in (1, 16) and permission.get('target') is False:
            if raw == 1:
                group = 'target_red_stopped' if speed <= STOP_SPEED_MPS else 'target_red_moving'
            else:
                group = ('green_no_left_stable_before_release' if all(before)
                         else 'green_horizon_overlaps_release')
        elif raw == 33 and permission.get('target') is True:
            group = 'left_arrow_startup' if speed <= STOP_SPEED_MPS else 'left_arrow_moving'
        elif raw in (1, 16, 33):
            raise ValueError('Raw signal disagrees with left-route permission')
    return dict(group=group, bound_left_permission_valid=eligible,
                light_id=signal.get('light_id'), raw_signal_bits=signal.get('raw_signal_bits'),
                route_permission=permission.get('target'), current_gt_speed_mps=float(speed),
                currently_stopped=speed <= STOP_SPEED_MPS,
                gt_endpoint_distance_m=float(distances[-1]),
                gt_max_waypoint_distance_m=float(distances.max()),
                stationary_future_gt=bool(distances.max() <= STATIONARY_FUTURE_M),
                waypoint_before_possible_release=before, waypoint_after_confirmed_release=after,
                horizon_overlaps_possible_release=(
                    stamp <= upper and stamp + round(offsets[-1] * 1_000_000_000) >= lower))


class SignalWaypointMetricsAccumulator:
    """Join official outputs to the exact admitted evaluation metadata order."""
    def __init__(self, ordered_entries, transition):
        self.entries = list(ordered_entries)
        if not self.entries:
            raise ValueError('Empty signal waypoint diagnostic selection')
        self.transition = dict(transition)
        self.annotations = [annotate_entry(entry, transition) for entry in self.entries]
        self.rows = []

    @property
    def policy(self):
        return dict(schema='morai_signal_waypoint_diagnostic_v1',
                    target_light_id=TARGET_LIGHT_ID, target_movement='left',
                    waypoint_offsets_sec=list(WAYPOINT_OFFSETS_SEC),
                    stop_speed_mps=STOP_SPEED_MPS,
                    stationary_future_gt_max_distance_m=STATIONARY_FUTURE_M,
                    displacement_tolerances_m=list(DISPLACEMENT_TOLERANCES_M),
                    before_release_comparison='waypoint_source_time_ns < lower_source_ns',
                    transition=self.transition,
                    group_selection='recorded bound permission, current GT speed and source timestamps',
                    stable_hold_selection=('stopped red or fully prerelease green; '
                                           'current GT speed <= 0.3 and all GT waypoint distances <= 0.05 m'),
                    color_head_policy='pure color only; route permission is separate; masked colors unscored',
                    thresholds_are='descriptive displacement sensitivity, not deployment pass/fail',
                    predictions_source='same forward pass as official aggregate evaluation')

    def update(self, outputs, targets, inputs):
        predicted = outputs[1].detach().cpu().double()
        gt = targets['waypoints'].detach().cpu()
        color = outputs[2].detach().cpu().double()
        velocity = inputs['velocity'].detach().cpu().reshape(-1)
        batch = len(predicted)
        if (predicted.shape != (batch, 5, 2) or gt.shape != predicted.shape
                or color.shape != (batch, 2) or len(velocity) != batch):
            raise ValueError('Unexpected signal waypoint diagnostic tensor shapes')
        if not all(bool(torch.isfinite(value).all()) for value in (predicted, gt, color, velocity)):
            raise ValueError('Nonfinite signal waypoint diagnostic tensors')
        if len(self.rows) + batch > len(self.entries):
            raise ValueError('Signal metadata shorter than evaluation order')
        probabilities = color.softmax(dim=-1)
        incoming = []
        for j in range(batch):
            index = len(self.rows) + j
            entry, annotation = self.entries[index], self.annotations[index]
            if any(int(targets[key][j]) != int(entry[key]) for key in ('frame_id', 'sensor_time_ns')):
                raise ValueError('Signal entry frame/time differs from batch')
            expected = torch.tensor(entry['waypoints'], dtype=gt.dtype)
            if not torch.equal(gt[j], expected) or not bool(targets['waypoint_valid'][j]):
                raise ValueError('Signal entry waypoint GT differs from batch')
            label = int(targets['traffic_light'][j])
            if label != int(entry['traffic_light']) or label not in (-1, 0, 1):
                raise ValueError('Signal entry color supervision differs from batch')
            target = gt[j].double()
            error = (predicted[j] - target).norm(dim=-1)
            distances = predicted[j].norm(dim=-1)
            incoming.append(dict(annotation,
                frame_id=int(entry['frame_id']), sensor_time_ns=int(entry['sensor_time_ns']),
                input_velocity_mps=float(velocity[j]),
                predicted_waypoints_m=predicted[j].tolist(), gt_waypoints_m=target.tolist(),
                waypoint_l2_errors_m=error.tolist(), ade_m=float(error.mean()), fde_m=float(error[-1]),
                predicted_endpoint_distance_m=float(distances[-1]),
                predicted_max_waypoint_distance_m=float(distances.max()),
                predicted_first_waypoint_distance_m=float(distances[0]),
                traffic_gt_class=label, predicted_traffic_class=int(probabilities[j].argmax()),
                predicted_color_probabilities=probabilities[j].tolist()))
        self.rows.extend(incoming)

    @staticmethod
    def _sensitivity(rows):
        return {str(tolerance): dict(
            any_waypoint_exceeds_count=sum(row['predicted_max_waypoint_distance_m'] > tolerance for row in rows),
            first_waypoint_exceeds_count=sum(row['predicted_first_waypoint_distance_m'] > tolerance for row in rows),
            all_waypoints_within_count=sum(row['predicted_max_waypoint_distance_m'] <= tolerance for row in rows))
            for tolerance in DISPLACEMENT_TOLERANCES_M}

    def compute(self):
        if len(self.rows) != len(self.entries):
            raise ValueError('Signal metadata length differs from evaluated frames')
        summaries = {}
        stable_holds = []
        average_fields = ('ade_m', 'fde_m', 'predicted_endpoint_distance_m',
                          'predicted_max_waypoint_distance_m', 'gt_endpoint_distance_m',
                          'gt_max_waypoint_distance_m', 'current_gt_speed_mps', 'input_velocity_mps')
        for group in GROUPS:
            rows = [row for row in self.rows if row['group'] == group]
            holds = [row for row in rows if group in (
                'target_red_stopped', 'green_no_left_stable_before_release')
                and row['currently_stopped'] and row['stationary_future_gt']]
            stable_holds.extend(holds)
            observed = [row for row in rows if row['traffic_gt_class'] >= 0]
            summaries[group] = dict(frames=len(rows),
                **{key + ('_mean' if key not in ('ade_m', 'fde_m') else ''):
                   mean(row[key] for row in rows) if rows else None for key in average_fields},
                color_observed_frames=len(observed),
                color_accuracy=mean(row['traffic_gt_class'] == row['predicted_traffic_class']
                                    for row in observed) if observed else None,
                stationary_future_gt_frames=sum(row['stationary_future_gt'] for row in rows),
                stable_hold_frames=len(holds),
                stable_hold_displacement_sensitivity=self._sensitivity(holds))
        green = [row for row in self.rows if row['group'] in (
            'green_no_left_stable_before_release', 'green_horizon_overlaps_release')]
        per_offset, total_count, total_error = [], 0, 0.
        for index, offset in enumerate(WAYPOINT_OFFSETS_SEC):
            selected = [row for row in green if row['waypoint_before_possible_release'][index]]
            count = len(selected)
            error = sum(row['waypoint_l2_errors_m'][index] for row in selected)
            per_offset.append(dict(offset_sec=offset, observed_points=count,
                                   point_l2_sum_m=error, point_mean_l2_m=error / count if count else None))
            total_count += count
            total_error += error
        return dict(policy=self.policy, frames=len(self.rows), groups=summaries,
                    stable_holds=dict(frames=len(stable_holds),
                                      displacement_sensitivity=self._sensitivity(stable_holds)),
                    green_before_release=dict(frames=len(green), per_offset=per_offset,
                        masked_point_count=total_count, masked_point_l2_sum_m=total_error,
                        masked_point_mean_l2_m=total_error / total_count if total_count else None),
                    notes=[
                        'Pure green may correctly have color class 1 while left-route permission is false.',
                        'Red+left-arrow color class -1 is masked; its left-route permission is true.',
                        'Overlapping horizons retain ordinary GT ADE/FDE; after-release points are not holding errors.',
                        'These offline waypoint diagnostics do not certify closed-loop safety or physical sensor simultaneity.'])
