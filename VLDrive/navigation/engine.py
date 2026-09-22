"""Map-independent navigation semantics. No future ego motion enters this module."""
import bisect
import hashlib
import json
import math

from .catalog import IDS, RULES, render, variant_for

DEFAULT_POLICY = {
    'lookahead_m': 30.0, 'compound_lookahead_m': 100.0,
    'completion_margin_m': 2.0, 'follow_chunk_m': 50.0,
    'max_route_error_m': 10.0, 'max_heading_error_deg': 60.0,
    'max_sensor_gap_ms': 150.0, 'projection_forward_m': 80.0,
    'projection_backward_m': 3.0, 'seed': 42,
    'prefer_compound': True, 'prefer_distance': False,
    'stopped_speed_mps': .05, 'moving_speed_mps': .2, 'brake_threshold': .05,
    'curve_lookahead_m': 20.0, 'curve_threshold_deg': 20.0,
    'junction_heading_sample_m': 5.0, 'junction_turn_threshold_deg': 30.0,
    'junction_uturn_threshold_deg': 150.0,
}


class Route:
    def __init__(self, data):
        self.data = data
        self.points = [tuple(float(v) for v in p[:2]) for p in data['points']]
        if len(self.points) < 2 or any(len(p) != 2 or not all(map(math.isfinite, p)) for p in self.points):
            raise ValueError('Route requires finite XY points')
        self.arc = [0.0]
        for a, b in zip(self.points, self.points[1:]):
            self.arc.append(self.arc[-1] + math.dist(a, b))
        self.length = self.arc[-1]
        if self.length <= 0:
            raise ValueError('Zero length route')
        self.closed = data.get('closed', False)
        if self.closed and math.dist(self.points[0], self.points[-1]) > .01:
            raise ValueError('Closed route must duplicate its first point')
        self.events = []
        ids = set()
        for event in data.get('events', []):
            event = dict(event)
            if event['id'] in ids:
                raise ValueError('Duplicate route event ID')
            ids.add(event['id'])
            start, end = event['start_index'], event['end_index']
            if type(start) is not int or type(end) is not int or not 0 <= start <= end < len(self.points):
                raise ValueError('Invalid route event bounds')
            if event['kind'] not in ('turn', 'roundabout', 'lane_change', 'highway_entry',
                                     'highway_exit', 'highway', 'blocked'):
                raise ValueError('Unknown route event kind')
            if event['kind'] in ('turn', 'lane_change', 'highway_entry', 'highway_exit'):
                if event.get('direction') not in ('left', 'right', 'straight'):
                    raise ValueError('Event lacks planned direction')
                if event['kind'] == 'lane_change' and event['direction'] == 'straight':
                    raise ValueError('Lane change must specify left or right')
            if event['kind'] == 'roundabout' and event.get('exit_number') not in (1, 2, 3):
                raise ValueError('LMDrive supports roundabout exits 1..3; do not guess')
            if not event.get('evidence'):
                raise ValueError('Route event requires semantic evidence')
            event.update(start_m=self.arc[start], end_m=self.arc[end])
            self.events.append(event)
        self.events.sort(key=lambda e: (e['start_m'], e['id']))
        self.identity = data.get('route_sha256') or hashlib.sha256(
            json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    def project(self, x, y, yaw=None, previous=None, hint=None, policy=None):
        policy = policy or DEFAULT_POLICY
        if not all(math.isfinite(float(v)) for v in (x, y)) or (yaw is not None and not math.isfinite(yaw)):
            return None
        if hint is not None:
            if type(hint) is not int or not 0 <= hint < len(self.points):
                raise ValueError('Route index outside route geometry')
            indices = range(max(0, hint - 1), min(len(self.points) - 1, hint + 1))
        elif previous is None:
            indices = range(len(self.points) - 1)
        else:
            begin = max(0, bisect.bisect_left(self.arc, previous - policy['projection_backward_m']) - 1)
            end = min(len(self.points) - 1, bisect.bisect_right(self.arc, previous + policy['projection_forward_m']))
            indices = list(range(begin, end))
            if self.closed and previous > self.length - policy['projection_forward_m']:
                indices += list(range(min(len(self.points) - 1,
                                         bisect.bisect_right(self.arc, policy['projection_forward_m']))))
        candidates = []
        for i in indices:
            a, b = self.points[i:i + 2]
            dx, dy = b[0] - a[0], b[1] - a[1]
            length2 = dx * dx + dy * dy
            if length2 < 1e-12:
                continue
            f = max(0., min(1., ((x-a[0])*dx + (y-a[1])*dy) / length2))
            error = math.hypot(x-a[0]-f*dx, y-a[1]-f*dy)
            heading = math.atan2(dy, dx)
            angle = abs(math.degrees(math.atan2(math.sin(yaw-heading), math.cos(yaw-heading)))) if yaw is not None else 0.
            progress = self.arc[i] + f * math.sqrt(length2)
            if previous is not None and progress < previous - policy['projection_backward_m']:
                if not (self.closed and previous > self.length-policy['projection_forward_m']
                        and progress < policy['projection_forward_m']):
                    continue
            if error <= policy['max_route_error_m'] and angle <= policy['max_heading_error_deg']:
                candidates.append((error + angle / 30., error, angle, progress, i))
        if not candidates:
            return None
        _, error, angle, progress, i = min(candidates)
        return {'progress_m': progress, 'route_index': i, 'route_error_m': error, 'heading_error_deg': angle}

    def point_at(self, progress):
        progress = progress % self.length if self.closed else max(0, min(self.length, progress))
        i = min(len(self.points)-2, max(0, bisect.bisect_right(self.arc, progress)-1))
        length = self.arc[i+1]-self.arc[i]
        f = (progress-self.arc[i])/length if length else 0
        a, b = self.points[i:i+2]
        return a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1])

    def is_straight(self, progress, horizon, threshold):
        if not self.closed: horizon=min(horizon,self.length-progress)
        if horizon<=0: return True
        before=self.point_at(progress);headings=[]
        for i in range(1,max(2,math.ceil(horizon/2))+1):
            after=self.point_at(progress+horizon*i/max(2,math.ceil(horizon/2)))
            if math.dist(before,after)>.01:
                headings.append(math.atan2(after[1]-before[1],after[0]-before[0]))
            before=after
        return not headings or all(abs(math.degrees(math.atan2(math.sin(h-headings[0]),
                                   math.cos(h-headings[0]))))<=threshold for h in headings)


class NavigationEngine:
    """Stable instructions throughout a maneuver; invalid localization clears output."""
    def __init__(self, route, policy=None, episode_id='live'):
        self.route = route if isinstance(route, Route) else Route(route)
        self.policy = dict(DEFAULT_POLICY, **(policy or {}))
        unknown = set(self.policy)-set(DEFAULT_POLICY)
        if unknown:
            raise ValueError('Unknown policy keys: ' + str(sorted(unknown)))
        for k, value in self.policy.items():
            if k.startswith('prefer_'):
                if type(value) is not bool: raise ValueError(k+' must be boolean')
            elif k == 'seed':
                if type(value) is not int: raise ValueError('seed must be integer')
            elif type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(k+' must be finite and positive')
        if self.policy['max_heading_error_deg'] > 180:
            raise ValueError('Heading limit exceeds 180 degrees')
        if self.policy['curve_threshold_deg'] >= 180:
            raise ValueError('Curve threshold must be below 180 degrees')
        if not self.policy['junction_turn_threshold_deg'] < self.policy['junction_uturn_threshold_deg'] <= 180:
            raise ValueError('Junction thresholds must satisfy turn < uturn <= 180')
        self.episode_id = episode_id
        self.previous = None
        self.last_time = None
        self.lap = 0
        self.active = None
        self.serial = 0
        self.instances = {}

    def _candidate(self, rule, event_key, args, target_m, end_m, priority, progress, timestamp):
        rule_id = IDS[rule]
        key = f'{self.episode_id}|{self.lap}|{event_key}|{rule_id}'
        if key not in self.instances:
            self.serial += 1
            instance_id = hashlib.sha256(f'{key}|{timestamp}|{self.serial}'.encode()).hexdigest()[:24]
            variant = variant_for(self.policy['seed'], instance_id)
            self.instances[key] = {'instruction_instance_id': instance_id, 'instruction_id': rule_id,
                'rule': rule, 'instruction_args': args, 'variant_id': variant,
                'text': render(rule_id, args, variant), 'issued_source_ns': timestamp,
                'issue_progress_m': progress, 'target_progress_m': target_m,
                'end_progress_m': end_m, 'event_id': event_key, 'priority': priority}
        return self.instances[key]

    def update(self, frame):
        stamp = frame['sensor_source_ns']
        if type(stamp) is not int or stamp < 0:
            raise ValueError('sensor_source_ns must be a nonnegative integer')
        reset = self.last_time is not None and (stamp <= self.last_time or
                stamp-self.last_time > self.policy['max_sensor_gap_ms']*1e6)
        if reset:
            self.instances.clear()
            self.active = None
            if stamp <= self.last_time:
                self.previous = None
                self.lap = 0
        self.last_time = stamp
        position = None if frame.get('valid') is False else self.route.project(
            frame['x'], frame['y'], frame.get('yaw_rad'), self.previous,
            frame.get('route_index'), self.policy)
        if position is None:
            self.active = None
            self.instances.clear()
            return {'sensor_source_ns': stamp, 'valid': False, 'reason': frame.get('reason', 'localization_invalid'),
                    'instruction': None, 'candidates': []}
        progress = position['progress_m']
        if self.previous is not None and progress < self.previous-self.policy['projection_backward_m']:
            self.lap += 1
            self.instances.clear()
            self.active = None
        self.previous = progress
        look = self.policy['lookahead_m']
        events = []
        for event in self.route.events:
            e = dict(event)
            if self.route.closed and e['end_m']+self.policy['completion_margin_m'] < progress:
                e['start_m'] += self.route.length
                e['end_m'] += self.route.length
            if e['end_m']+self.policy['completion_margin_m'] >= progress:
                events.append(e)
        events.sort(key=lambda e: (e['start_m'], e['id']))
        candidates = []

        def add(rule, event_key, target, end, priority=0, args=None):
            if rule.endswith('-dis') and target < progress:
                # Existing issued distance instructions remain valid during execution.
                key = f'{self.episode_id}|{self.lap}|{event_key}|{IDS[rule]}'
                if key in self.instances:
                    candidates.append(self.instances[key])
                return
            candidates.append(self._candidate(rule, event_key, args or {'x': max(0, target-progress)},
                                               target, end, priority, progress, stamp))

        nearby = [e for e in events if e['start_m']-progress <= look and e['kind'] != 'highway']
        if nearby:
            e = nearby[0]  # Finish the active maneuver before announcing the next one.
            start, end, key, kind = e['start_m'], e['end_m'], e['id'], e['kind']
            if kind == 'blocked':
                self.active = None
                self.instances.clear()
                return dict(position, sensor_source_ns=stamp, valid=False,
                            reason=e.get('reason', 'unresolved_route_semantics'), instruction=None, candidates=[])
            if kind == 'turn':
                direction = {'left':'L','right':'R','straight':'S'}[e['direction']]
                families = ['Turn-02'] if e.get('junction', True) else []
                if direction in ('L','R'): families.insert(0, 'Turn-01')
                if e.get('traffic_light'): families.append('Turn-03')
                if e.get('t_junction'): families.append('Turn-04')
                for family in families:
                    priority = {'Turn-01':10,'Turn-02':20,'Turn-03':30,'Turn-04':40}[family]
                    add(f'{family}-{direction}', key, start, end, priority)
                    add(f'{family}-{direction}-dis', key, start, end, priority + (1 if self.policy['prefer_distance'] else -1))
                next_turn = next((t for t in events if t['kind']=='turn' and t['start_m'] > end
                                  and t['start_m']-start <= self.policy['compound_lookahead_m']), None)
                if next_turn and e.get('junction', True) and next_turn.get('junction', True):
                    second = {'left':'L','right':'R','straight':'S'}[next_turn['direction']]
                    add(f'Turn-06-{direction}-{second}', key+'+'+next_turn['id'], start,
                        next_turn['end_m'], 60 if self.policy['prefer_compound'] else 5)
            elif kind == 'roundabout':
                add(f"Turn-05-{e['exit_number']}", key, start, end, 80)
            elif kind == 'lane_change':
                side = 'L' if e['direction']=='left' else 'R'
                add('Follow-01-'+side, key, start, end, 50)
                add('Follow-01-'+side+'-dis', key, start, end, 51 if self.policy['prefer_distance'] else 49)
            elif kind in ('highway_entry','highway_exit'):
                # LMDrive's available semantics are left entry / right exit.
                side = 'L' if kind=='highway_entry' else 'R'
                if e['direction'] != ('left' if side=='L' else 'right'):
                    return dict(position, sensor_source_ns=stamp, valid=False,
                                reason='highway_direction_not_in_lmdrive_catalog', instruction=None, candidates=[])
                add('Follow-04-'+side, key, start, end, 50)
                add('Follow-04-'+side+'-dis', key, start, end, 51 if self.policy['prefer_distance'] else 49)
        else:
            chunk = int(progress/self.policy['follow_chunk_m'])
            target = min(self.route.length, (chunk+1)*self.policy['follow_chunk_m'])
            next_junction = next((e for e in events if e['kind'] in ('turn','roundabout','blocked')), None)
            if next_junction: target = min(target, next_junction['start_m'])
            key = 'follow:'+str(chunk)
            add('Follow-02-s1', key, target, target, 1)
            add('Follow-02-s1-dis', key, target, target)
            if self.route.is_straight(progress,self.policy['curve_lookahead_m'],self.policy['curve_threshold_deg']):
                add('Follow-03-s1', key, target, target)
                add('Follow-03-s1-dis', key, target, target)
                if next_junction:
                    follow_key='until:'+next_junction['id']
                    add('Follow-03-s2', follow_key, next_junction['start_m'], next_junction['start_m'], 2)
                    add('Follow-03-s2-dis', follow_key, next_junction['start_m'], next_junction['start_m'])
            if any(e['kind']=='highway' and e['start_m']<=progress<=e['end_m'] for e in events):
                add('Follow-02-s2', key, target, target, 3)
                add('Follow-02-s2-dis', key, target, target)
            add('Other-04', key, target, target, -20)
            if frame.get('yaw_rad') is not None:
                tx, ty = self.route.point_at(target)
                yaw = frame['yaw_rad']; dx, dy = tx-frame['x'], ty-frame['y']
                forward, left = math.cos(yaw)*dx+math.sin(yaw)*dy, -math.sin(yaw)*dx+math.cos(yaw)*dy
                if forward >= 0:
                    add('Other-05', key, target, target, -10,
                        {'x': forward, 'y': abs(left), 'side':'left' if left>=0 else 'right'})
        # Explicit externally requested behavior, not a diagnosis inferred online
        # from a brake pedal or a stopped vehicle. Offline adapters may annotate
        # these with recorded maneuver evidence.
        for action, rule in [('start','Other-01'),('slow','Other-02'),('stop','Other-03')]:
            if frame.get('requested_action') == action:
                add(rule, 'action:'+str(frame.get('action_id', action)), progress, progress, 100)
        # Preserve an issued compound instruction through its second maneuver.
        if self.active and self.active['rule'].startswith('Turn-06'):
            if progress <= self.active['end_progress_m']+self.policy['completion_margin_m']:
                candidates.append(self.active)
        if not candidates:
            return dict(position, sensor_source_ns=stamp, valid=False, reason='no_supported_instruction',
                        instruction=None, candidates=[])
        if self.active:
            selected = next((c for c in candidates if c['instruction_instance_id']==self.active['instruction_instance_id']), None)
        else: selected = None
        best = max(candidates, key=lambda c:(c['priority'], -c['instruction_id']))
        if selected is None or best['priority'] > selected['priority']:
            selected = best
        changed = self.active is None or selected['instruction_instance_id'] != self.active['instruction_instance_id']
        self.active = selected
        live_keys = {c['instruction_instance_id'] for c in candidates}
        self.instances = {k:v for k,v in self.instances.items() if v['instruction_instance_id'] in live_keys}
        return dict(position, sensor_source_ns=stamp, valid=True, changed=changed, lap=self.lap,
                    instruction=selected, candidates=candidates)
