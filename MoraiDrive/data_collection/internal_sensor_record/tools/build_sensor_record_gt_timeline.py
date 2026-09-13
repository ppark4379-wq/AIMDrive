"""Link original ROS actor/light snapshots to exact native LiDAR source time.

Outputs timing candidates and conservative masks. It never changes timestamps,
fills missing GT, or publishes the result for training.
"""
import argparse
from bisect import bisect_left,bisect_right
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from async_capture.alignment import PayloadReader,decode,recorded_classes
from async_capture.journal import digest,verify_run,write_json
from async_capture.native_gt_timeline import (collapse_source_rows,event_times,
    interpolate_object_value,nearby_events,nearby_light_events,object_discontinuities,
    source_bracket,source_times,stable_light_value,traffic_transitions)
from ssd_storage import require_ssd


def vector(v):return [float(v.x),float(v.y),float(v.z)]


def object_value(msg):
    groups=(('npc_list','npc'),('pedestrian_list','pedestrian'),('obstacle_list','obstacle'))
    expected=(msg.num_of_npcs,msg.num_of_pedestrian,msg.num_of_obstacle)
    if tuple(len(getattr(msg,name)) for name,_ in groups)!=expected:
        raise ValueError('ObjectStatusList counters differ from payload arrays')
    objects={}
    for (field,kind),count in zip(groups,expected):
        for obj in getattr(msg,field):
            key='%s:%d'%(kind,int(obj.unique_id))
            if key in objects:raise ValueError('Duplicate actor identity in one snapshot')
            row=dict(id=int(obj.unique_id),kind=kind,type=int(obj.type),name=str(obj.name),
                position=vector(obj.position),heading_deg=float(obj.heading),
                velocity_kmh=vector(obj.velocity),acceleration_mps2=vector(obj.acceleration),
                size_wlh_m=vector(obj.size))
            if not all(math.isfinite(x) for x in (row['position']+row['velocity_kmh']+
                    row['acceleration_mps2']+row['size_wlh_m']+[row['heading_deg']])):
                raise ValueError('Nonfinite actor state')
            objects[key]=row
    return dict(objects=objects,counts=dict(npc=int(expected[0]),pedestrian=int(expected[1]),
        obstacle=int(expected[2])),possibly_truncated=sum(expected)>=20,
        complete_inventory=False,model_identity_complete=False,
        geometry_source='ros_object_status_list')


def grpc_object_value(payload, model_identities):
    """Convert a full GetAllActorsState snapshot without hiding identity gaps."""
    response=payload.get('response')
    if not isinstance(response,dict):return None
    instances=payload.get('actor_instances',{})
    kinds={'OBJECT_TYPE_VEHICLE':'npc','OBJECT_TYPE_PEDESTRIAN':'pedestrian',
           'OBJECT_TYPE_OBSTACLE':'obstacle'}
    objects={};counts=Counter();missing=[]
    for actor in response.get('states',[]):
        info=actor.get('actor_info',{});actor_id=str(info.get('id',{}).get('value',''))
        if actor_id=='Ego':continue
        kind=kinds.get(info.get('object_type'))
        instance=instances.get(actor_id,{})
        instance_key=instance.get('instance_key')
        identity=model_identities.get(instance_key,{}) if instance_key else {}
        embedded_valid=instance.get('model_identity_valid') is True
        model_name=str(identity.get('model_name') or
                       (instance.get('model_name') if embedded_valid else '') or '').strip()
        identity_valid=bool(model_name and (identity.get('valid_nonempty_model_name') is True
                                            or embedded_valid))
        if not instance_key or not identity_valid:
            missing.append(actor_id)
        box=actor.get('bounding_box',{})
        try:
            extent=[float(box['extent'][axis]) for axis in ('x','y','z')]
            position=[float(box['location'][axis]) for axis in ('x','y','z')]
            rotation=[float(box['rotation'][axis]) for axis in ('x','y','z')]
            acceleration=[float(actor.get('acceleration',{}).get(axis,0)) for axis in ('x','y','z')]
            velocity=[float(actor.get('global_velocity',{}).get(axis,0)) for axis in ('x','y','z')]
            if kind!='npc':
                measured=payload.get('motion_estimates',{}).get(actor_id)
                if measured is not None:
                    velocity=[3.6*float(value) for value in measured['velocity_xyz_mps']]
            values=extent+position+rotation+acceleration+velocity
            if (not actor_id or kind is None or min(extent)<=0 or
                    not all(math.isfinite(value) for value in values)):
                raise ValueError('invalid actor geometry')
        except (KeyError,TypeError,ValueError):
            raise ValueError('Invalid full-inventory actor: '+actor_id)
        key=kind+':'+(instance_key or actor_id)
        if key in objects:raise ValueError('Duplicate full-inventory actor identity')
        objects[key]=dict(id=actor_id,kind=kind,type=info.get('object_type'),
            name=model_name,model_name=model_name,instance_key=instance_key,
            position=position,position_semantics='grpc_bounding_box_world_center',
            heading_deg=rotation[2],velocity_kmh=velocity,
            acceleration_mps2=acceleration,size_wlh_m=[2*value for value in extent],
            geometry_source='GetAllActorsState.bounding_box')
        counts[kind]+=1
    return dict(objects=objects,counts=dict(npc=counts['npc'],
        pedestrian=counts['pedestrian'],obstacle=counts['obstacle']),
        possibly_truncated=False,complete_inventory=True,
        model_identity_complete=not missing,missing_model_identity_actor_ids=sorted(missing),
        geometry_source='grpc_get_all_actors_state_bounding_box')


def selected_traffic_query(query):
    """Return the selected final observation from traffic schema v2 or v3."""
    if query.get('attempts') is None:return query
    if query.get('final_valid_response_identity') is not True:return query
    valid=[row for row in query['attempts']
           if row.get('valid_response_identity') is True]
    return valid[-1] if valid else query


def ego_xy(timeline,stamp,times,max_gap_ns=50000000):
    i=int(np.searchsorted(times,stamp))
    if i<len(times) and times[i]==stamp:
        return None if timeline[i]['conflicting_values'] else timeline[i]['value']['xy']
    if i==0 or i==len(times) or times[i]-times[i-1]>max_gap_ns:return None
    if timeline[i-1]['conflicting_values'] or timeline[i]['conflicting_values']:return None
    f=(stamp-times[i-1])/(times[i]-times[i-1])
    return [(1-f)*timeline[i-1]['value']['xy'][k]+f*timeline[i]['value']['xy'][k] for k in (0,1)]


def direct_light_index(rows):
    by_id={}
    for row in rows:by_id.setdefault(row['value']['id'],[]).append(row)
    result={}
    for light_id,same in by_id.items():
        same.sort(key=lambda row:(row['before_ns'],row['after_ns'],row['record_id']))
        if any(row['after_ns']<row['before_ns'] for row in same):
            raise ValueError('Reversed direct light query source bracket')
        before=[row['before_ns'] for row in same];after=[row['after_ns'] for row in same]
        if any(b<a for a,b in zip(before,before[1:])) or any(b<a for a,b in zip(after,after[1:])):
            raise ValueError('Direct light query source time reversed')
        result[light_id]=dict(rows=same,before=before,after=after)
    return result


def direct_light_value(rows_or_index, stamp, max_span_ns=250000000):
    """Return stable direct gRPC observations that source-bracket ``stamp``.

    GetTrafficLightInfo has no timestamp.  Each observation therefore retains
    the GetAllActorsState timestamps immediately before and after the RPC.  A
    sensor time is admitted only when one completed observation is wholly
    before it and another is wholly after it, with the same ID/color and a
    bounded combined source interval.
    """
    index=(direct_light_index(rows_or_index) if isinstance(rows_or_index,list)
           else rows_or_index)
    values=[]
    for light_id,item in sorted(index.items()):
        same=item['rows']
        li=bisect_right(item['after'],stamp)-1
        ri=bisect_left(item['before'],stamp)
        if li<0 or ri>=len(same):continue
        middle=None
        if li==ri:
            middle=same[li];li-=1;ri+=1
            if li<0 or ri>=len(same):continue
        before,after=same[li],same[ri]
        observed=sorted([before,after]+([middle] if middle is not None else []),
                        key=lambda row:(row['before_ns'],row['after_ns'],row['record_id']))
        if after['after_ns']-before['before_ns']>max_span_ns:continue
        if any(row['value']!=before['value'] for row in observed[1:]):continue
        values.append(dict(before['value'],source_bracket_ns=[before['before_ns'],after['after_ns']],
            source_record_ids=[row['record_id'] for row in observed],
            query_intervals_ns=[[row['before_ns'],row['after_ns']] for row in observed]))
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for path in (a.episode,a.output):require_ssd(path,reserve_bytes=0)
    load=lambda path:json.loads(path.read_text())
    for folder,sealname in (('quality','QUALITY_COMPLETE.json'),('native_audit','COMPLETE.json')):
        for name,sha in load(a.episode/folder/sealname)['files'].items():
            if Path(name).name!=name or digest(a.episode/folder/name)!=sha:raise ValueError('Changed '+folder+' evidence')
    cat=load(a.episode/'native_audit/report.json');index=load(a.episode/'quality/INDEX_COMPLETE.json')
    acquisition=load(a.episode.parent/'sensor_contract.json')
    actor_source_units=int(acquisition.get('capture',{}).get('actor_ground_truth',{}).get(
        'source_timestamp_units_per_second',1000000))
    if actor_source_units<=0:raise ValueError('Invalid actor source clock units')
    if cat['raw_manifest_sha256']!=index['raw_manifest_sha256']:raise ValueError('Different raw recordings')
    exact={r['relative_path']:r for r in cat['comparison'] if r['role']=='lidar' and r.get('exact_xyzi_array_equal')}
    raw=Path(index['raw_path']);require_ssd(raw,reserve_bytes=0)
    manifest,_=verify_run(raw,payloads=False);classes=recorded_classes(manifest);reader=PayloadReader(raw)
    active=load(a.episode/'quality/active_signals.json')
    model_identities={};identity_records=[]
    for meta in active:
        if meta['role']!='actor_identity':continue
        identity=json.loads(reader.read(meta));identity_records.append(identity)
        key=identity.get('instance_key')
        if identity.get('valid_nonempty_model_name') is not True or not key:continue
        previous=model_identities.get(key)
        if previous is not None and previous.get('model_name')!=identity.get('model_name'):
            raise ValueError('Conflicting model names for one actor incarnation')
        model_identities[key]=identity
    objrows=[];grpc_objrows=[];lightrows=[];direct_light_rows=[];egorows=[]
    try:
        for meta in active:
            role=meta['role']
            if role=='objects':
                msg=decode(meta,reader.read(meta),classes)
                objrows.append(dict(source_ns=meta['source_timestamp_ns'],record_id=meta['record_id'],
                    receive_monotonic_ns=meta['receive_monotonic_ns'],value=object_value(msg)))
            elif role=='actor_inventory':
                row=json.loads(reader.read(meta));value=grpc_object_value(row,model_identities)
                if value is not None:
                    grpc_objrows.append(dict(source_ns=int(row['response']['timestamp'])*1000000000//actor_source_units,
                        record_id=meta['record_id'],receive_monotonic_ns=meta['receive_monotonic_ns'],
                        value=value))
            elif role=='traffic_light':
                msg=decode(meta,reader.read(meta),classes)
                lightrows.append(dict(source_ns=meta['source_timestamp_ns'],record_id=meta['record_id'],
                    receive_monotonic_ns=meta['receive_monotonic_ns'],value=dict(
                        id=str(msg.trafficLightIndex),type=int(msg.trafficLightType),status=int(msg.trafficLightStatus))))
            elif role=='traffic_light_inventory':
                row=json.loads(reader.read(meta))
                units=actor_source_units
                before=row.get('before_actor_source_raw');after=row.get('after_actor_source_raw')
                queries=row.get('queries')
                if queries is None: # Preserve the first diagnostic schema.
                    queries={row.get('light_id'):dict(response=row.get('response',{}),
                              valid_response_identity=row.get('valid_response_identity'))}
                if before is not None and after is not None and int(after)>=int(before):
                    for requested,query in queries.items():
                        query=selected_traffic_query(query)
                        response=query.get('response',{});ident=response.get('id',{}).get('value')
                        if query.get('valid_response_identity') is True and ident==requested:
                            direct_light_rows.append(dict(before_ns=int(before)*1000000000//units,
                                after_ns=int(after)*1000000000//units,record_id=meta['record_id'],
                                receive_monotonic_ns=meta['receive_monotonic_ns'],
                                value=dict(id=str(ident),color=int(response.get('color',0)))))
            elif role=='ego':
                signal=meta.get('signal');egorows.append(dict(source_ns=meta['source_timestamp_ns'],
                    record_id=meta['record_id'],receive_monotonic_ns=meta['receive_monotonic_ns'],
                    value=dict(xy=[float(x) for x in signal['position'][:2]])))
    finally:reader.close()
    ros_objects=collapse_source_rows(objrows)
    objects=collapse_source_rows(grpc_objrows) if grpc_objrows else ros_objects
    object_source=('grpc_full_inventory' if grpc_objrows else 'ros_top20_object_topic')
    lights=collapse_source_rows(lightrows)
    egos=collapse_source_rows(egorows)
    direct_light_queries=direct_light_index(direct_light_rows)
    object_max_span_ns=(150000000 if object_source=='grpc_full_inventory' else 30000000)
    discontinuities=object_discontinuities(objects,max_gap_ns=object_max_span_ns)
    transitions=traffic_transitions(lights)
    object_times,light_times,ego_times=source_times(objects),source_times(lights),source_times(egos)
    discontinuity_times,transition_times=event_times(discontinuities),event_times(transitions)
    frames=[]
    for i,bundle in enumerate(cat['bundles']):
        lidar=exact.get(bundle['assets'].get('lidar'))
        if lidar is None:raise ValueError('Native frame lacks exact ROS LiDAR identity')
        stamp=lidar['ros_source_timestamp_ns'];ego=ego_xy(egos,stamp,ego_times)
        obefore,oafter,omethod=source_bracket(objects,stamp,times=object_times,
                                             max_span_ns=object_max_span_ns)
        oval=interpolate_object_value(obefore,oafter,stamp) if obefore else None
        changes=nearby_events(discontinuities,stamp,ego,time_index=discontinuity_times)
        reasons=[];identity_reasons=[]
        if obefore is None:reasons.append('object_source_not_bracketed_within_%dms'%(
            object_max_span_ns//1000000))
        elif obefore['conflicting_values'] or oafter['conflicting_values']:
            reasons.append('conflicting_object_states_at_source')
        elif obefore['value']['possibly_truncated'] or oafter['value']['possibly_truncated']:
            reasons.append('possibly_truncated_top20_object_list')
        elif oval is None:reasons.append('object_identity_changed_across_source_bracket')
        if oval is not None and not oval.get('complete_inventory',False):
            reasons.append('object_inventory_not_complete')
        if oval is not None and not oval.get('model_identity_complete',False):
            identity_reasons.append('actor_model_identity_incomplete')
        if ego is None:reasons.append('ego_not_bracketed')
        if changes:reasons.append('nearby_actor_lifecycle_or_motion_discontinuity')
        direct_values=direct_light_value(direct_light_queries,stamp)
        lbefore,lafter,lmethod=source_bracket(lights,stamp,times=light_times)
        lvalue=stable_light_value(lbefore,lafter) if lbefore else None
        light_reasons=[];light_changes=[]
        if direct_values:
            lmethod='grpc_query_source_bracket'
            lvalue=(dict(id=direct_values[0]['id'],status=direct_values[0]['color'])
                    if len(direct_values)==1 else None)
            direct_ids=[value['id'] for value in direct_values]
            direct_records=sorted(set(r for value in direct_values for r in value['source_record_ids']))
            direct_bracket=[min(value['source_bracket_ns'][0] for value in direct_values),
                            max(value['source_bracket_ns'][1] for value in direct_values)]
        else:
            direct_ids=[];direct_records=[];direct_bracket=[]
        if direct_values:
            pass
        elif lbefore is None:light_reasons.append('light_source_not_bracketed_within_30ms')
        elif lbefore['conflicting_values'] or lafter['conflicting_values']:
            light_reasons.append('conflicting_light_states_at_source')
        elif lvalue is None:light_reasons.append('light_changed_across_source_bracket')
        else:
            light_changes=nearby_light_events(transitions,stamp,lvalue['id'],time_index=transition_times)
            if not lvalue['id'] or lvalue['status'] not in (1,16):light_reasons.append('not_pure_red_or_green')
            if light_changes:light_reasons.append('closest_light_or_state_transition_within_100ms')
        frames.append(dict(index=i,native_filename_time_ns=bundle['filename_simulation_time_hypothesis_ns'],
            lidar_source_ns=stamp,ego_xy=ego,object_gt=dict(
                source=object_source,
                association_method=omethod,
                source_record_ids=(obefore['record_ids']+(oafter['record_ids'] if oafter is not obefore else [])) if obefore else [],
                source_bracket_ns=[obefore['source_ns'],oafter['source_ns']] if obefore else [],
                objects=list(oval['objects'].values()) if oval else [],
                reasons=reasons,identity_reasons=identity_reasons,
                nearby_discontinuities=changes,transport_timing_candidate=not reasons,
                model_identity_candidate=not identity_reasons,
                complete_inventory=bool(oval and oval.get('complete_inventory')),
                geometry_verified=False,camera_association_verified=False),
            traffic_light_gt=dict(association_method=lmethod,
                source='direct_grpc_query' if direct_values else 'ros_query_response_topic',
                source_record_ids=direct_records if direct_values else ((lbefore['record_ids']+(lafter['record_ids'] if lafter is not lbefore else [])) if lbefore else []),
                source_bracket_ns=direct_bracket if direct_values else ([lbefore['source_ns'],lafter['source_ns']] if lbefore else []),
                values=direct_values,value=lvalue,reasons=light_reasons,nearby_transitions=light_changes,
                transport_timing_candidate=not light_reasons,camera_association_verified=False),
            source_timestamps_modified=False,training_visible=False))
    report=dict(schema='morai_sensor_record_gt_timeline_v1',frames=len(frames),
        object_source=object_source,object_source_snapshots=len(objects),
        ros_object_source_snapshots=len(ros_objects),grpc_object_source_snapshots=len(grpc_objrows),
        actor_identity_records=len(identity_records),verified_actor_incarnations=len(model_identities),
        light_source_snapshots=len(lights),
        direct_light_query_snapshots=len(direct_light_rows),
        ego_source_snapshots=len(egos),
        object_same_source_conflicts=sum(bool(r['conflicting_values']) for r in objects),
        light_same_source_conflicts=sum(bool(r['conflicting_values']) for r in lights),
        ego_same_source_conflicts=sum(bool(r['conflicting_values']) for r in egos),
        detected_object_events=dict(Counter(r['kind'] for r in discontinuities)),
        detected_light_events=dict(Counter(r['kind'] for r in transitions)),
        object_association_methods=dict(Counter(r['object_gt']['association_method'] for r in frames)),
        light_association_methods=dict(Counter(r['traffic_light_gt']['association_method'] for r in frames)),
        object_frame_status=dict(Counter('candidate' if r['object_gt']['transport_timing_candidate'] else
            '+'.join(r['object_gt']['reasons']) for r in frames)),
        object_identity_frame_status=dict(Counter('candidate' if r['object_gt']['model_identity_candidate'] else
            '+'.join(r['object_gt']['identity_reasons']) for r in frames)),
        light_frame_status=dict(Counter('candidate' if r['traffic_light_gt']['transport_timing_candidate'] else
            '+'.join(r['traffic_light_gt']['reasons']) for r in frames)),
        object_discontinuity_window_ms=20,object_relevance_radius_m=60,
        source_interpolation_max_span_ms=object_max_span_ns//1000000,
        light_transition_window_ms=100,direct_light_source_bracket_max_span_ms=250,
        direct_light_contract='GetTrafficLightInfo has no timestamp; stable same-ID/color queries on both sides of sensor source time',
        object_velocity_unit='km/h from captured MORAI message definition',
        source_timestamps_modified=False,physical_simultaneity_verified=False,training_visible=False,
        limitation=('Transport-visible continuity screen only. Exact source time does not prove camera exposure/GT atomicity. '
                    'New captures prefer complete gRPC inventory; legacy captures without actor incarnation/model records remain masked.'))
    a.output.mkdir(exist_ok=False)
    write_json(a.output/'report.json',report);write_json(a.output/'frames.json',frames)
    write_json(a.output/'object_events.json',discontinuities);write_json(a.output/'light_events.json',transitions)
    write_json(a.output/'COMPLETE.json',dict(files={n:digest(a.output/n) for n in
        ('report.json','frames.json','object_events.json','light_events.json')},
        raw_manifest_sha256=index['raw_manifest_sha256'],native_audit_sha256=digest(a.episode/'native_audit/report.json'),
        script_sha256=digest(Path(__file__)),training_visible=False))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
