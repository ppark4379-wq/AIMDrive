"""Conservative source-time continuity checks for ROS actor and light GT.

These checks detect transport-visible discontinuities around native anchors.
They cannot establish when a camera exposed its pixels or make unobserved
simulator state atomic with a sensor frame.
"""
from bisect import bisect_left,bisect_right
import json
import math


def collapse_source_rows(rows, value_key='value'):
    """Collapse identical repeated source stamps and retain every record ID."""
    result=[]
    for row in rows:
        stamp=int(row['source_ns'])
        if result and stamp<result[-1]['source_ns']:
            raise ValueError('GT source clock reversed')
        value=row[value_key]
        canonical=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
        if result and stamp==result[-1]['source_ns']:
            result[-1]['record_ids'].append(row['record_id'])
            result[-1]['receive_monotonic_ns'].append(row['receive_monotonic_ns'])
            if canonical!=result[-1]['canonical']:
                result[-1]['conflicting_values'].append(value)
            continue
        result.append(dict(source_ns=stamp,value=value,canonical=canonical,
            record_ids=[row['record_id']],receive_monotonic_ns=[row['receive_monotonic_ns']],
            conflicting_values=[]))
    return result


def object_discontinuities(timeline, *, max_gap_ns=50000000,
                           position_tolerance_m=.25, max_acceleration_mps2=20.):
    """Find births/deaths and motion impossible under reported km/h velocity."""
    events=[]
    for before,after in zip(timeline,timeline[1:]):
        dt_ns=after['source_ns']-before['source_ns']
        if dt_ns<=0:raise ValueError('Collapsed object timeline is not increasing')
        if before['conflicting_values'] or after['conflicting_values']:
            events.append(dict(kind='conflicting_same_source_state',before_ns=before['source_ns'],
                after_ns=after['source_ns'],actor_id=None,positions=[]))
            continue
        if dt_ns>max_gap_ns:
            events.append(dict(kind='object_source_gap',before_ns=before['source_ns'],
                after_ns=after['source_ns'],actor_id=None,positions=[]))
            continue
        a,b=before['value']['objects'],after['value']['objects'];dt=dt_ns/1e9
        for key in sorted(set(a)|set(b)):
            if key not in a or key not in b:
                present=a.get(key) or b.get(key)
                events.append(dict(kind='actor_disappeared' if key in a else 'actor_appeared',
                    before_ns=before['source_ns'],after_ns=after['source_ns'],actor_id=key,
                    positions=[present['position']],name=present['name']))
                continue
            aa,bb=a[key],b[key]
            distance=math.dist(aa['position'][:2],bb['position'][:2])
            speed=max(math.hypot(*aa['velocity_kmh'][:2]),math.hypot(*bb['velocity_kmh'][:2]))/3.6
            maximum=speed*dt+.5*max_acceleration_mps2*dt*dt+position_tolerance_m
            if distance>maximum:
                events.append(dict(kind='actor_motion_discontinuity',before_ns=before['source_ns'],
                    after_ns=after['source_ns'],actor_id=key,positions=[aa['position'],bb['position']],
                    name=aa['name'],distance_m=distance,reported_speed_upper_mps=speed,
                    allowed_distance_m=maximum))
    return events


def source_times(timeline):
    times=[row['source_ns'] for row in timeline]
    if any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Collapsed source timeline is not strictly increasing')
    return times


def exact_source_row(timeline,stamp,*,times=None):
    times=source_times(timeline) if times is None else times
    i=bisect_left(times,stamp)
    return timeline[i] if i<len(timeline) and times[i]==stamp else None


def source_bracket(timeline,stamp,*,max_span_ns=30000000,times=None):
    """Return an exact row or two source rows that tightly surround a stamp."""
    times=source_times(timeline) if times is None else times
    i=bisect_left(times,stamp)
    if i<len(times) and times[i]==stamp:return (timeline[i],timeline[i],'exact')
    if i==0 or i==len(times) or times[i]-times[i-1]>max_span_ns:return (None,None,'unbracketed')
    return (timeline[i-1],timeline[i],'interpolated')


def interpolate_object_value(before,after,stamp):
    """Interpolate a continuous full snapshot; raw endpoints remain separate."""
    a,b=before['value'],after['value']
    if (before['conflicting_values'] or after['conflicting_values'] or
            a['possibly_truncated'] or b['possibly_truncated'] or set(a['objects'])!=set(b['objects'])):
        return None
    if before is after:return a
    f=(stamp-before['source_ns'])/(after['source_ns']-before['source_ns'])
    result=dict(a);result['objects']={};result['possibly_truncated']=False
    if a['counts']!=b['counts']:return None
    for key,aa in a['objects'].items():
        bb=b['objects'][key]
        if (aa['id'],aa['kind'],aa['type'],aa['name'])!=(bb['id'],bb['kind'],bb['type'],bb['name']):return None
        row=dict(aa)
        for name in ('position','velocity_kmh','acceleration_mps2','size_wlh_m'):
            row[name]=[(1-f)*x+f*y for x,y in zip(aa[name],bb[name])]
        delta=(bb['heading_deg']-aa['heading_deg']+180)%360-180
        row['heading_deg']=aa['heading_deg']+f*delta
        pose_fields=('actor_pivot_world_xyz','actor_rotation_rpy_deg','rotation_rpy_deg')
        if all(name in aa and name in bb for name in pose_fields):
            row['actor_pivot_world_xyz']=[(1-f)*x+f*y for x,y in zip(aa['actor_pivot_world_xyz'],bb['actor_pivot_world_xyz'])]
            for name in ('actor_rotation_rpy_deg','rotation_rpy_deg'):
                row[name]=[x+f*((y-x+180)%360-180) for x,y in zip(aa[name],bb[name])]
        else:
            for name in pose_fields:row.pop(name,None)
        result['objects'][key]=row
    return result


def stable_light_value(before,after):
    if before is None or before['conflicting_values'] or after['conflicting_values']:return None
    return before['value'] if before['value']==after['value'] else None


def event_times(events):
    """Build an interval index once so long recordings are not scanned per frame."""
    before=[event['before_ns'] for event in events]
    after=[event['after_ns'] for event in events]
    if any(b<a for a,b in zip(before,before[1:])) or any(b<a for a,b in zip(after,after[1:])):
        raise ValueError('GT events are not ordered')
    return before,after


def _events_in_window(events,stamp,window_ns,time_index):
    before,after=event_times(events) if time_index is None else time_index
    lo=bisect_left(after,stamp-window_ns)
    hi=bisect_right(before,stamp+window_ns)
    return events[lo:hi]


def nearby_events(events,stamp,ego_xy,*,window_ns=20000000,relevance_radius_m=60.,time_index=None):
    selected=[]
    for event in _events_in_window(events,stamp,window_ns,time_index):
        positions=event.get('positions',[])
        relevant=event['actor_id'] is None or ego_xy is None or any(
            math.dist(position[:2],ego_xy)<=relevance_radius_m for position in positions)
        if relevant:selected.append(event)
    return selected


def traffic_transitions(timeline):
    events=[]
    for before,after in zip(timeline,timeline[1:]):
        if before['conflicting_values'] or after['conflicting_values']:
            events.append(dict(kind='conflicting_same_source_state',before_ns=before['source_ns'],
                after_ns=after['source_ns'],light_ids=[]))
            continue
        a,b=before['value'],after['value']
        if a!=b:
            ids=sorted(set(x for x in (a.get('id'),b.get('id')) if x))
            events.append(dict(kind='closest_light_changed' if a.get('id')!=b.get('id') else 'light_state_changed',
                before_ns=before['source_ns'],after_ns=after['source_ns'],light_ids=ids,
                before=a,after=b))
    return events


def nearby_light_events(events,stamp,light_id,*,window_ns=100000000,time_index=None):
    return [event for event in _events_in_window(events,stamp,window_ns,time_index)
            if light_id in event['light_ids']]
