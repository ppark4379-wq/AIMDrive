"""Replay recorded perception/control poses without connecting to MORAI.

This compares decisions on the recorded trajectory, not a closed-loop rerun.
Use a dynamic mission log: specialized section planners are not replayed here.
"""
import argparse
import contextlib
import csv
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from AIM2026.Expert.adapter import planner_inputs
from AIM2026.Expert.src.interfaces.geometry import Point
from AIM2026.Expert.src.planning.common.adaptive_cruise_control import AdaptiveCruiseControl
from AIM2026.Expert.src.planning.common.dynamic_crossing import DynamicCrossingGuard
from AIM2026.Expert.src.planning.common.object_selection import ForwardObjectDetector
from AIM2026.Expert.src.planning.common.reference_path import PathManager


def records(path):
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            yield json.loads(line)


def replay(expert_dir, context_path, output, track_id):
    packet_by_key = {(p['generation'], p['sequence']): p for p in records(expert_dir/'inputs.jsonl')}
    saved = expert_dir/'expert/config.json'
    if not saved.exists():
        saved = expert_dir/'config.json'
    old = json.loads(saved.read_text(encoding='utf-8'))
    new = json.loads((ROOT/'AIM2026/Expert/src/config/config.json').read_text(encoding='utf-8'))
    context = json.loads(context_path.read_text(encoding='utf-8'))
    if context['scenario_config'].get('mission_mode') != 'full_loop_dynamic':
        raise ValueError('This replay requires a dynamic mission with the reference path')
    route = [Point(*p[:2]) for p in context['route_points']]
    manager = PathManager(route, False, old['map']['local_path_size'])
    profile = dict(old['planning']['velocity_profile'])
    replacements = {z['name']: z for z in new['planning']['velocity_profile']['speed_limit_zones']}
    profile['speed_limit_zones'] = [replacements.get(z['name'], z) if 'start_point' in z else z
                                  for z in profile['speed_limit_zones']]
    manager.set_velocity_profile(**profile)
    acc = AdaptiveCruiseControl(vehicle_length=old['common']['vehicle_length'],
        time_gap_zones=old['planning']['velocity_profile']['speed_limit_zones'],
        **old['planning']['adaptive_cruise_control'])
    guard = DynamicCrossingGuard(vehicle_length=new['common']['vehicle_length'],
        vehicle_width=new['planning']['adaptive_cruise_control']['vehicle_width'],
        **new['planning']['dynamic_crossing'])
    following = AdaptiveCruiseControl(vehicle_length=new['common']['vehicle_length'],
        time_gap_zones=new['planning']['velocity_profile']['speed_limit_zones'],
        **new['planning']['adaptive_cruise_control'])
    detector = ForwardObjectDetector([])
    rows = []
    for control in records(expert_dir/'control.jsonl'):
        pose = control.get('control_pose')
        if not pose or control.get('stale_hold'):
            continue
        packet = packet_by_key.get((pose['generation'], control.get('inference_sequence')))
        if packet is None or not packet.get('ready'):
            continue
        # Match the runtime's accepted localization sources during GPS gaps.
        ego, objects = planner_inputs(packet, pose, position_sources=('GPS', 'GPS_dead_reckoning'))
        path, desired = manager.get_local_path(ego)
        preview = route[manager.current_waypoint:manager.current_waypoint+120]
        detector.dynamic_object_list = objects
        entries = detector.detect_object(ego)
        acc.check_object(path, entries, None)
        limit, details = guard.evaluate(preview, ego, entries, desired,
            now=control['monotonic_ns']/1e9)
        following.check_object(path, guard.following_entries(entries), None)
        following_limit = following.get_target_velocity(ego.velocity, desired)
        rows.append(dict(sequence=packet['sequence'], time_s=control['monotonic_ns']/1e9,
            ego_y=float(ego.position.y), speed_kph=ego.velocity*3.6,
            recorded_target_kph=control['planning']['target_speed']*3.6,
            recorded_brake=control['brake'], old_acc_id=acc.object_id,
            new_profile_kph=desired*3.6, crossing_id=details.get('object_id'),
            crossing_limit_kph=limit*3.6 if math.isfinite(limit) else None,
            state=details['state'], lateral_m=details.get('lateral_m'),
            entry_time_s=details.get('entry_time_s'), exit_time_s=details.get('exit_time_s'),
            earliest_arrival_s=details.get('earliest_arrival_s'),
            clears_before_arrival=details.get('clears_before_arrival'),
            front_gap_m=details.get('front_gap_m'), new_acc_id=following.object_id,
            object_and_profile_limit_kph=max(0., min(desired, limit, following_limit))*3.6))
    focused = [r for r in rows if r['crossing_id'] == track_id]
    slowed = [r for r in focused if r['crossing_limit_kph'] < r['speed_kph']-.36]
    recorded = [r for r in rows if r['old_acc_id'] == track_id and r['recorded_brake'] > .5]
    stopped = [r for r in focused if r['crossing_limit_kph'] == 0.]
    summary = dict(expert_run=str(expert_dir), track_id=track_id, frames=len(rows),
        method='Recorded poses and detections; no simulator or closed-loop outcome prediction',
        first_crossing_yield=focused[0] if focused else None,
        first_lower_speed_request=slowed[0] if slowed else None,
        first_zero_request=stopped[0] if stopped else None,
        recorded_first_brake=recorded[0] if recorded else None)
    if slowed and recorded:
        summary['earlier_seconds'] = recorded[0]['time_s']-slowed[0]['time_s']
        summary['earlier_route_y_m'] = recorded[0]['ego_y']-slowed[0]['ego_y']
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if rows:
        with (output/'decisions.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('expert_dir', type=Path)
    parser.add_argument('--scenario-context', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--track-id', type=int, required=True)
    args = parser.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        report = replay(args.expert_dir, args.scenario_context, args.output, args.track_id)
    print(json.dumps(report, indent=2))
