"""Offline scoring. GT is consumed here, never supplied to VE planning."""
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon


def footprint(box):
    x, y = box['center_world']
    c, s = math.cos(math.radians(box['heading_deg'])), math.sin(math.radians(box['heading_deg']))
    return Polygon([(x+c*dx-s*dy, y+s*dx+c*dy) for dx, dy in (
        (box['length_m']/2, box['width_m']/2), (box['length_m']/2, -box['width_m']/2),
        (-box['length_m']/2, -box['width_m']/2), (-box['length_m']/2, box['width_m']/2))])


def local_xy(box, ego):
    dx, dy = box['center_world'][0]-ego['x'], box['center_world'][1]-ego['y']
    yaw = math.radians(ego['yaw_deg'])
    return math.cos(yaw)*dx+math.sin(yaw)*dy, -math.sin(yaw)*dx+math.cos(yaw)*dy


def match_centers(gt, ve, gate=2.):
    if not gt or not ve:
        return []
    d = np.array([[math.dist(g['center_world'], v['center_world']) for v in ve] for g in gt])
    # Gate BEFORE assignment; dummy columns prevent an invalid pair stealing a match.
    cost = np.full((len(gt), len(ve)+len(gt)), 1e6)
    cost[:, :len(ve)] = np.where(d <= gate, d, 1e6)
    for i in range(len(gt)):
        cost[i, len(ve)+i] = (len(gt)+1)*gate+1.
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(d[i, j])) for i, j in zip(rows, cols) if j < len(ve) and d[i, j] <= gate]


def score(rows, case, goal_reached, error=None):
    min_center = min_body = None
    collisions, switches, matched, eligible, invalid, ve_stale = set(), 0, 0, 0, 0, 0
    previous_matches, speed_errors, center_errors = {}, [], []
    stop_candidates = stopped = moving_exposure = 0.
    max_gap = 0.
    longest_miss = current_miss = 0.
    prior = None
    for row in rows:
        dt = 0. if prior is None else max(0., row['t']-prior['t'])
        max_gap = max(max_gap, dt)
        # Unknown gaps must not inflate observed stop duration.
        observed_dt = dt if dt <= .5 else 0.
        ego = row['ego']
        collisions.update(ego.get('collision_objects', []))
        gt = row['gt'] if row['gt_fresh'] else []
        invalid += not row['gt_fresh']
        ve_stale += not row['ve_fresh']
        ego_box = dict(center_world=[ego['x'],ego['y']], heading_deg=ego['yaw_deg'], length_m=4.635, width_m=2.)
        body = footprint(ego_box)
        for box in gt:
            center = math.dist(box['center_world'], ego_box['center_world'])
            min_center = center if min_center is None else min(center, min_center)
            distance = body.distance(footprint(box))
            min_body = distance if min_body is None else min(distance, min_body)
        targets = [b for b in gt if -20. <= local_xy(b, ego)[0] <= 30.
                   and abs(local_xy(b, ego)[1]) <= 25.]
        if any(b['speed_mps'] > .8 for b in targets):
            moving_exposure += observed_dt
        if row['gt_fresh']:
            eligible += len(targets)
            pairs = match_centers(targets, row['ve'] if row['ve_fresh'] else [])
            matched += len(pairs)
            for i, j, distance in pairs:
                g, v = targets[i], row['ve'][j]
                key, track = str(g['track_id']), v['track_id']
                last = previous_matches.get(key)
                if last and row['t']-last[0] < .75 and last[1] != track:
                    switches += 1
                previous_matches[key] = (row['t'], track)
                center_errors.append(distance)
                speed_errors.append(abs(g['speed_mps']-v['speed_mps']))
            if targets and len(pairs) < len(targets):
                current_miss += observed_dt
                longest_miss = max(longest_miss, current_miss)
            else:
                current_miss = 0.
        if ego['speed_mps'] < .3:
            stopped += observed_dt
            # This is a review candidate, not proof of a false stop. Signals,
            # curvature, lateral traffic and the selected path may justify it.
            near_ahead = any(0. <= local_xy(b, ego)[0] < 25. and abs(local_xy(b, ego)[1]) < 4. for b in gt)
            if row['t'] > 3. and row['remaining_s'] > 10. and row['gt_fresh'] and not near_ahead:
                stop_candidates += observed_dt
        prior = row
    duration = rows[-1]['t'] if rows else 0.
    encounter = (case == 'empty' or (min_center is not None and min_center < 30.))
    if case in ('slow_lead','stop_restart','roundabout','oncoming','cut_in'):
        encounter = encounter and moving_exposure >= .5
    coverage = bool(rows) and invalid == 0 and max_gap <= .5
    return dict(goal_reached=bool(goal_reached), error=error, observations=len(rows), elapsed_s=duration,
        collision_objects=sorted(collisions), collision_observed=bool(collisions),
        min_center_distance_m=min_center, min_estimated_footprint_distance_m=min_body,
        footprint_policy='ROS object dimensions; Ego 4.635x2m centered at reported pose; approximate, not collider clearance',
        stopped_s=stopped, stop_without_near_ahead_object_candidate_s=stop_candidates,
        unnecessary_stops_confirmed=False, moving_object_exposure_s=moving_exposure,
        encounter_observed=encounter, gt_stale_observations=invalid, ve_stale_observations=ve_stale,
        max_observation_gap_s=max_gap,
        measurement_coverage_ok=coverage, gt_center_observations_in_bev=eligible,
        center_matched_observations=matched, center_match_2m_rate=matched/eligible if eligible else None,
        center_error_mean_m=float(np.mean(center_errors)) if center_errors else None,
        scalar_speed_error_mean_mps=float(np.mean(speed_errors)) if speed_errors else None,
        estimated_id_switches=switches, longest_any_object_center_miss_s=longest_miss,
        criterion='goal + no observed collision + encounter + complete sampled measurements; not safety certification',
        basic_trial_passed=bool(goal_reached and not error and not collisions and encounter and coverage))


def compare_pair(gt, ve):
    fields = ('elapsed_s','stopped_s','stop_without_near_ahead_object_candidate_s',
              'min_center_distance_m','min_estimated_footprint_distance_m')
    same_plan = gt.get('pair_contract_sha256') == ve.get('pair_contract_sha256')
    same_config = gt.get('expert_config_sha256') is not None and gt.get('expert_config_sha256') == ve.get('expert_config_sha256')
    signatures = gt.get('initial_scene'), ve.get('initial_scene')
    comparable = same_plan and same_config and all(signatures)
    if comparable:
        a,b = signatures
        comparable = len(a['objects']) == len(b['objects']) and math.dist(a['ego'],b['ego']) <= 1.
        if comparable and a['objects']:
            comparable = len(match_centers(a['objects'],b['objects'],gate=2.)) == len(a['objects'])
    return dict(same_pair_contract=same_plan, same_expert_config=same_config,
        initial_scene_comparable=bool(comparable),
        comparison_ready=bool(comparable and gt['metrics']['basic_trial_passed'] and ve['metrics']['measurement_coverage_ok']),
        gt=gt['metrics'], ve=ve['metrics'],
        ve_minus_gt={key: (ve['metrics'][key]-gt['metrics'][key]
                          if gt['metrics'][key] is not None and ve['metrics'][key] is not None else None) for key in fields},
        interpretation='Matched seeded scenario policy; asynchronous actor trajectories can diverge. No automatic winner.')
