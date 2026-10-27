"""Opt-in observation and model isolation; never controls pedestrian movement."""
import json
import math
import time


def configure_models(config, model, excluded):
    if not model and not excluded:
        return
    pool = config.get('dynamic_obstacle_models', [])
    if model and model not in pool:
        raise ValueError('Unknown pedestrian model: ' + model)
    compatible = config.get('dynamic_obstacle', {}).get('compatible_models')
    if model and compatible is not None and model not in compatible:
        raise ValueError('Pedestrian model failed the MORAI survival audit: ' + model)
    selected = [m for m in ([model] if model else pool) if m not in excluded]
    if not selected:
        raise ValueError('No pedestrian models remain')
    # Keep the model registry intact: exclusion rules validate against it.
    config.setdefault('dynamic_obstacle', {})['diagnostic_models'] = selected


def record_snapshot(scenario, route_s):
    now = time.monotonic()
    if now - getattr(scenario, '_pedestrian_trace_last', float('-inf')) < 0.1:
        return
    scenario._pedestrian_trace_last = now
    row = dict(time_unix=time.time(), monotonic=now, ego_route_s=route_s)
    try:
        states = scenario.grpc.get_global_pedestrian_states()
        row['inventory'] = [dict(id=str(s.actor_info.id.value),
                                 object_type=int(s.actor_info.object_type),
                                 client_key=str(s.actor_info.client_key),
                                 x=s.transform.location.x, y=s.transform.location.y,
                                 z=s.transform.location.z) for s in states]
        by_id = {s['id']: s for s in row['inventory']}
        row['events'] = []
        for event in getattr(scenario, 'dynamic_obstacle_events', []):
            item = {key: event.get(key) for key in
                    ('id', 'actor_id', 'model', 'phase', 'start', 'end', 'speed_mps',
                     'path_length_m', 'state_read_error', 'failure_reason', 'spawn_request')}
            # A lost actor's numeric ID may be recycled by a later spawn.
            # Never attribute the replacement's position to the old event.
            state = (by_id.get(str(event.get('actor_id')))
                     if event.get('actor') is not None else None)
            if state and event.get('start'):
                item['distance_from_spawn_m'] = math.hypot(
                    state['x'] - event['start']['x'], state['y'] - event['start']['y'])
            row['events'].append(item)
    except Exception as error:
        row['query_error'] = str(error)
    try:
        with (scenario.output / 'pedestrian_trace.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    except OSError as error:
        print('[Dynamic] pedestrian trace write failed: ' + str(error), flush=True)
