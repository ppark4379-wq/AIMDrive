"""Reproduce the notice-data audit without modifying datasets or training code.

Run from AIMDrive: python VLDrive/docs/reports/notice_instruction_audit_20260916/audit_data.py
Candidates demonstrate data sufficiency; they are not production training labels.
Requires the pinned upstream notice_rules.py downloaded into .runtime/notice_audit.
"""
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
DATASET = WORKSPACE / 'MoraiDrive/dataset_local/competition_train'
UPSTREAM = WORKSPACE / 'VLDrive/.runtime/notice_audit/upstream'
RULES = UPSTREAM / 'tools/data_parsing/notice_rules.py'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def classify(frame):
    nav = frame.get('navigation', {})
    light_id = nav.get('affected_light_id_raw')
    raw_state = nav.get('affected_light_state_raw')
    if not light_id:
        return None, 'no_route_signal'
    if not nav.get('route_valid'):
        return None, 'invalid_route'
    if (frame.get('simulator_labels_source') != 'grpc'
            or frame.get('native_sync_frame') != frame.get('grpc_label_tick_raw')):
        return None, 'unaligned_source'
    if not any(x.get('tl_id') == light_id and x.get('state') == raw_state
               for x in frame.get('traffic_lights', [])):
        return None, 'unverified_signal'
    # Deliberately exclude arrow-only and mixed red/green/yellow states.
    # Existing MORAI conversion is checked against the raw observation.
    mapping = {1: (0, 57, 'R', 'red'), 4: (1, 59, 'Y', 'yellow'),
               16: (2, 58, 'G', 'green'), 48: (2, 58, 'G', 'green')}
    if raw_state not in mapping:
        return None, 'composite_or_unsupported_state'
    state, instruction_id, suffix, color = mapping[raw_state]
    if nav.get('affected_light_state_lmdrive') != state:
        return None, 'state_mapping_disagrees'
    return (light_id, instruction_id, suffix, color), 'verified_route_signal'


def main():
    spec = importlib.util.spec_from_file_location('upstream_notice_rules', RULES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    templates = load(UPSTREAM / 'leaderboard/leaderboard/envs/instruction_dict.json')
    report = {'upstream_notice_rules_sha256': hashlib.sha256(RULES.read_bytes()).hexdigest(),
              'candidate_policy': 'aligned-current-route-simple-signal-transitions-v1',
              'candidate_limitations': ['No CARLA distance threshold is reproduced.',
                  'Candidate counts are event starts, not training-eligible clip counts.',
                  'No future ego, future controls, or synthetic actors are used.',
                  'Candidates are not installed into the dataset or training pipeline.'],
              'existing_notice_file': load(DATASET / 'notice_instruction_list.json'),
              'routes': {}}
    candidates = {}
    for line in (DATASET / 'dataset_index.txt').read_text().splitlines():
        relative, count = line.split()
        relative = relative.rstrip('/')
        route = DATASET / relative
        frames = [load(route / 'measurements_full' / f'{i:04d}.json') for i in range(int(count))]
        assert all(f['frame_id'] == i for i, f in enumerate(frames))
        reasons = Counter(); states = Counter(); fields = Counter(); events = []; previous = None
        for frame in frames:
            for key in ('actors_data', 'gt_objects_raw', 'traffic_lights',
                        'is_vehicle_present', 'is_pedestrian_present', 'is_bike_present'):
                fields[key + '_nonempty'] += bool(frame.get(key))
            fields['affected_light_id_valid'] += frame.get('affected_light_id', -1) != -1
            states[str(frame.get('navigation', {}).get('affected_light_state_raw'))] += 1
            signal, reason = classify(frame)
            reasons[reason] += 1
            if signal is not None and signal != previous:
                light_id, instruction_id, suffix, color = signal
                events.append({'route_path': relative, 'frame_id': frame['frame_id'],
                    'last_verified_frame': frame['frame_id'],
                    'native_sync_frame': frame['native_sync_frame'],
                    'instruction': 'Notice-08-' + suffix, 'instruction_id': instruction_id,
                    'instruction_args': [], 'text': templates[str(instruction_id)][2],
                    'status': 'audit_candidate_not_training_label',
                    'source': {'measurement': f'{relative}/measurements_full/{frame["frame_id"]:04d}.json',
                        'light_id': light_id,
                        'raw_state': frame['navigation']['affected_light_state_raw'],
                        'grpc_label_tick_raw': frame['grpc_label_tick_raw'],
                        'route_valid': frame['navigation']['route_valid']}})
            elif signal is not None:
                events[-1]['last_verified_frame'] = frame['frame_id']
            previous = signal
        # All actor maps are empty: the upstream rule returns before any town-specific
        # distance lookup. Town01 here is only a harness argument, not a MORAI map claim.
        upstream_counts = {color: len(module.Notice08(color).process({'data': frames}, {}, '01'))
                           for color in ('red', 'green', 'yellow')}
        candidates[relative] = events
        report['routes'][relative] = {'frames': len(frames), 'nonempty_fields': dict(fields),
            'raw_signal_states': dict(states), 'candidate_frame_reasons': dict(reasons),
            'upstream_notice08_counts_on_unmodified_frames': upstream_counts,
            'candidate_events': len(events),
            'candidate_events_by_instruction': dict(Counter(e['instruction'] for e in events))}
        for frame_id in (243, 1000):
            d = frames[frame_id]
            report['routes'][relative].setdefault('actual_examples', []).append(
                {k: d[k] for k in ('frame_id', 'native_sync_frame', 'grpc_label_tick_raw',
                 'x', 'y', 'affected_light_id', 'actors_data', 'navigation', 'traffic_lights')})

    campaign = WORKSPACE / 'VLDrive/data/scenario_campaign_20260914_155854_lap_001/episode_000'
    manifest = load(campaign / 'raw/raw_manifest.json')
    campaign_report = {'training_visible': manifest['training_visible'],
                       'record_counts': manifest['statistics']['written'], 'examples': {}}
    streams = {'actor_inventory', 'actor_identity', 'traffic_light_inventory'}
    actor_counts = Counter()
    near_front_examples = {}
    with (campaign / 'quality/record_index.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            index = json.loads(line)
            role = index.get('stream')
            if role not in streams or (role != 'actor_inventory' and role in campaign_report['examples']):
                continue
            with (campaign / 'raw' / index['chunk_path']).open('rb') as raw:
                raw.seek(index['payload_offset'])
                payload = json.loads(raw.read(index['payload_bytes']))
            campaign_report['examples'].setdefault(role, {'record_id': index['record_id'],
                'chunk_path': index['chunk_path'], 'payload_offset': index['payload_offset'],
                'payload_bytes': index['payload_bytes'], 'payload': payload})
            if role != 'actor_inventory':
                continue
            states = payload['response'].get('states', [])
            ego = next((a for a in states if a.get('actor_info', {}).get('id', {}).get('value') == 'Ego'), None)
            if ego is None:
                continue
            p = ego['transform']['location']
            yaw_degrees = ego['transform']['rotation']['z']
            yaw = math.radians(yaw_degrees)
            for actor in states:
                info = actor.get('actor_info', {})
                kind, actor_id = info.get('object_type'), info.get('id', {}).get('value')
                if actor_id == 'Ego':
                    continue
                actor_counts[kind] += 1
                q = actor['transform']['location']
                dx, dy = q['x'] - p['x'], q['y'] - p['y']
                forward = math.cos(yaw) * dx + math.sin(yaw) * dy
                lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
                if 0 < forward < 25 and abs(lateral) < 3 and kind not in near_front_examples:
                    near_front_examples[kind] = {'record_id': index['record_id'],
                        'timestamp_raw': payload['response']['timestamp'],
                        'tick_count': payload['response']['sync_mode_tick_count'],
                        'actor_id': actor_id, 'ego_xyz': p, 'ego_yaw_deg': yaw_degrees,
                        'actor_xyz': q, 'forward_m': forward, 'lateral_m': lateral,
                        'instance': payload.get('actor_instances', {}).get(actor_id),
                        'source_index': index}
    report['newer_campaign'] = campaign_report
    dump(OUTPUT / 'audit_results.json', report)
    dump(OUTPUT / 'notice_candidates.json', candidates)
    dump(OUTPUT / 'newer_actor_examples.json', {'actor_observation_counts': dict(actor_counts),
        'near_front_examples': near_front_examples,
        'scope': 'Geometric raw-snapshot candidates only; not camera-aligned training labels.'})
    print(json.dumps(report['routes'], indent=2))
    print('Saved audit_results.json and notice_candidates.json')


if __name__ == '__main__':
    main()
