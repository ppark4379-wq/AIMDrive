import copy
import json
import math

import pytest

from navigation.catalog import RULES, render, variant_for
from navigation.engine import DEFAULT_POLICY, NavigationEngine
from navigation.build import build_episode, build
from navigation.mgeo import roundabout_exit, junction_context
from navigation.sampling import NavigationLabels
from inference.navigation_observer import replay


def route(events=(), points=None):
    return dict(points=points or [[i,0] for i in range(201)],events=list(events),closed=False)


def event(kind='turn',start=50,end=60,**kwargs):
    return dict(id=kind+str(start),kind=kind,start_index=start,end_index=end,
                evidence='synthetic_planned_topology',**kwargs)


def frame(i,x,y=0,**kwargs):
    return dict(index=i,sensor_source_ns=1_000_000_000+i*100_000_000,x=x,y=y,yaw_rad=0.,**kwargs)


def test_all_55_meanings_are_reachable_and_all_440_templates_render():
    observed=set()
    scenarios=[]
    for direction in ('left','right','straight'):
        scenarios.append((route([event(direction=direction,junction=True,traffic_light=True,t_junction=True)]),frame(0,20)))
        for second in ('left','right','straight'):
            scenarios.append((route([event(direction=direction),event(start=80,end=90,direction=second)]),frame(0,20)))
    for side in ('left','right'):
        scenarios.append((route([event('lane_change',direction=side)]),frame(0,20)))
    for number in (1,2,3):
        scenarios.append((route([event('roundabout',exit_number=number)]),frame(0,20)))
    for kind,side in [('highway_entry','left'),('highway_exit','right')]:
        scenarios.append((route([event(kind,direction=side)]),frame(0,20)))
    scenarios.append((route([event('highway',0,100),event(start=120,end=130,direction='straight')]),frame(0,20)))
    for action in ('start','slow','stop'):
        scenarios.append((route(),frame(0,20,requested_action=action)))
    for data,pose in scenarios:
        result=NavigationEngine(data).update(pose)
        assert result['valid']
        for row in result['candidates']:
            observed.add(row['instruction_id'])
            for v in range(8):
                text=render(row['instruction_id'],row['instruction_args'],v)
                assert '[' not in text and 'left/right' not in text
    assert observed==set(RULES)
    assert len(observed)==55 and sum(len(r['templates']) for r in RULES.values())==440


def test_instruction_is_held_during_stops_and_two_maneuvers():
    engine=NavigationEngine(route([event(direction='right'),event(start=80,end=90,direction='left')]))
    results=[engine.update(frame(i,x)) for i,x in enumerate([20,25,25,25,52,61,70,85,93])]
    first=results[0]['instruction']
    assert first['rule']=='Turn-06-R-L'
    assert all(r['instruction']['instruction_instance_id']==first['instruction_instance_id'] for r in results[:-1])
    assert results[-1]['instruction']['instruction_instance_id']!=first['instruction_instance_id']


def test_invalid_pose_and_gaps_do_not_keep_stale_instruction():
    engine=NavigationEngine(route([event(direction='right')]))
    first=engine.update(frame(0,20))['instruction']
    invalid=engine.update(frame(1,22,y=100))
    assert not invalid['valid'] and invalid['instruction'] is None
    second=engine.update(frame(2,24))['instruction']
    assert first['instruction_instance_id']!=second['instruction_instance_id']
    third=engine.update(frame(20,25))['instruction']
    assert second['instruction_instance_id']!=third['instruction_instance_id']


def test_unresolved_roundabout_is_explicitly_unavailable():
    data=route([event('blocked',reason='roundabout_exit_unresolved')])
    result=NavigationEngine(data).update(frame(0,20))
    assert not result['valid'] and result['reason']=='roundabout_exit_unresolved'


def test_rotated_route_and_new_episode_have_no_city_coordinate_dependency():
    data=route([event(direction='left')],[[1000,2000+i] for i in range(201)])
    pose=frame(0,1000,2020);pose['yaw_rad']=math.pi/2
    result=NavigationEngine(data,episode_id='new_city/hand_drive').update(pose)
    assert result['valid'] and result['instruction']['rule']=='Turn-02-L'
    assert result['progress_m']==20


def test_recorded_future_fields_cannot_change_navigation():
    data=route([event(direction='left')]);pose=frame(0,20)
    a=NavigationEngine(data).update(pose)
    pose.update(future_trajectory=[[9999,0]],future_ego={'x':-999},brake=True,steer=1)
    assert NavigationEngine(data).update(pose)==a


def test_epoch_sampling_is_reproducible_and_uses_all_variants():
    values=[variant_for(42,'same_instruction',epoch) for epoch in range(100)]
    assert set(values)==set(range(8))
    assert values==[variant_for(42,'same_instruction',epoch) for epoch in range(100)]
    assert all(variant_for(42,'x',epoch,variants=[6,7]) in (6,7) for epoch in range(10))


def test_roundabout_exit_count_uses_directed_ring_and_merges_parallel_lanes():
    def link(a,b,points): return dict(from_node_idx=a,to_node_idx=b,points=points)
    links={'r1':link('a','b',[[0,0],[10,0]]),'r2':link('b','c',[[10,0],[10,10]]),
           'r3':link('c','a',[[10,10],[0,0]]),'entry':link('in','a',[[-10,0],[0,0]]),
           'exit1':link('a','out1',[[0,0],[0,-10]]),'exit1lane2':link('a','out2',[[0,0],[.5,-10]]),
           'exit2':link('b','out3',[[10,0],[20,0]])}
    planned=[('entry',0,1),('r1',2,3),('exit2',4,5)]
    assert roundabout_exit(planned,links,['r1','r2','r3'],0,5)==2


def test_multi_dataset_build_replay_and_sampling(tmp_path):
    datasets=[]
    for number in (1,2):
        episode=dict(schema='navigation-episode-v1',episode_id='city'+str(number),
                     route=route([event(direction='right')]),frames=[frame(i,20+i) for i in range(35)])
        path=tmp_path/f'episode{number}.json';path.write_text(json.dumps(episode))
        datasets.append(dict(root=str(path),format='portable'))
        records,trace,_=build_episode(episode,DEFAULT_POLICY)
        for (_,result),expected in zip(replay(episode,DEFAULT_POLICY),trace):
            assert result['instruction']['instruction_instance_id']==expected['active_instruction_instance_id']
            assert [r['instruction_instance_id'] for r in result['candidates']]==expected['candidate_ids']
    cfg=tmp_path/'config.json';out=tmp_path/'output'
    cfg.write_text(json.dumps(dict(schema='lmdrive-navigation-build-v1',datasets=datasets,output=str(out),policy=DEFAULT_POLICY)))
    assert build(cfg)['episodes']==2
    assert build(cfg)['status']=='verified_existing'
    labels=NavigationLabels(out)
    times=[frame(i,20+i)['sensor_source_ns'] for i in range(5)]
    values=[labels.sample('city1',times,epoch=e) for e in range(50)]
    assert len({r['text'] for r in values})>8
    assert all(r['instruction_id'] in RULES for r in values)
    assert labels.sample('city1',times,epoch=3)==labels.sample('city1',times,epoch=3)
    # The distance argument is re-anchored at the beginning of a cropped window.
    later=[frame(i,20+i)['sensor_source_ns'] for i in range(5,10)]
    distance=next(labels.sample('city1',later,epoch=e) for e in range(100)
                  if labels.sample('city1',later,epoch=e)['instruction_id']==7+1)
    assert distance['instruction_args']['x']==25


def test_missing_frame_excluded_from_any_training_window():
    episode=dict(episode_id='a',route=route(),frames=[frame(i,i,valid=i!=2) for i in range(5)])
    records,trace,excluded=build_episode(episode,DEFAULT_POLICY)
    assert not trace[2]['candidate_ids']
    assert set(trace[1]['candidate_ids']).isdisjoint(trace[3]['candidate_ids'])


def test_template_correction_is_semantically_entry():
    for i in range(8):
        text=render(48,{'x':30},i)
        assert 'exit' not in text and 'leave' not in text and 'off' not in text


def test_t_junction_topology_does_not_count_parallel_lanes_as_fourth_arm():
    def link(a,b,p,j=False): return dict(from_node_idx=a,to_node_idx=b,points=p,is_junction=j)
    links={'inside':link('a','b',[[0,0],[2,0]],True),
           'west':link('w','a',[[-10,0],[0,0]]),
           'east':link('b','e',[[2,0],[12,0]]),
           'north':link('a','n',[[0,0],[0,10]]),
           'north_lane':link('a','n2',[[0,0],[.5,10]])}
    assert junction_context(links,{})['a']['t_junction']
    links['south']=link('a','s',[[0,0],[0,-10]])
    assert not junction_context(links,{})['a']['t_junction']


def test_recorded_actions_are_offline_only_and_require_brake_evidence():
    from navigation.adapters import recorded_action_labels
    frames=[frame(i,i,speed_mps=s,brake=b) for i,(s,b) in enumerate([(0,0),(1,0),(3,1),(2,1),(0,1)])]
    annotated=recorded_action_labels(frames,DEFAULT_POLICY)
    assert annotated[1]['requested_action']=='start'
    assert all(f['requested_action']=='stop' for f in annotated[2:])
    assert all('requested_action' not in f for f in frames)
    assert NavigationEngine(route()).update(frames[2])['instruction']['instruction_id']!=62
    no_brake=[dict(f,brake=0) for f in frames]
    assert all(f.get('requested_action')!='stop' for f in recorded_action_labels(no_brake,DEFAULT_POLICY))


def test_streaming_signals_match_json_without_loading_whole_document(tmp_path):
    from navigation.adapters import json_array_items
    rows=[dict(record_id=i,text='x'*100000,role='ego') for i in range(30)]
    path=tmp_path/'signals.json';path.write_text(json.dumps(rows,indent=2))
    assert list(json_array_items(path))==rows


def test_curved_road_does_not_receive_straight_course_instruction():
    points=[[20*math.sin(i/40),20*(1-math.cos(i/40))] for i in range(100)]
    result=NavigationEngine(route(points=points)).update(frame(0,0))
    assert result['valid']
    assert not any(c['rule'].startswith('Follow-03') for c in result['candidates'])
    assert result['instruction']['rule']=='Follow-02-s1'


def test_until_intersection_instruction_is_not_reissued_each_distance_chunk():
    engine=NavigationEngine(route([event(start=150,end=160,direction='right')]))
    first=engine.update(frame(0,20))['instruction']
    second=engine.update(frame(1,55))['instruction']
    assert first['rule']=='Follow-03-s2'
    assert first['instruction_instance_id']==second['instruction_instance_id']


def test_live_observer_stale_pause_recovery_and_clock_reset(monkeypatch):
    import sys
    from types import ModuleType
    from inference import navigation_observer as observer

    # Exercise the live loop with receiver outputs, without a simulator or socket.
    udp=ModuleType('inference.morai_udp');projection=ModuleType('pyproj')
    closed=[]
    class Receiver:
        def __init__(self,port,bind): self.port=port;self.index=0
        def latest(self):
            index=self.index;self.index+=1
            stamp=[1.,1.,1.,1.1,.1][index]
            age=1. if index==1 else 0.
            if self.port==1: return dict(longitude=20,latitude=0),age
            return dict(sim_time=stamp,yaw=0.,speed=0.,brake=0.),age
        def close(self): closed.append(self.port)
    class Transformer:
        @staticmethod
        def from_crs(*args,**kwargs): return Transformer()
        def transform(self,x,y): return x,y
    udp.GpsReceiver=udp.VehicleStatusReceiver=Receiver
    projection.Transformer=Transformer
    monkeypatch.setitem(sys.modules,'inference.morai_udp',udp)
    monkeypatch.setitem(sys.modules,'pyproj',projection)
    monkeypatch.setattr(observer.time,'sleep',lambda _:None)
    config=dict(runtime=dict(hz=20,max_age_s=.5,max_pair_skew_s=.2,startup_timeout_s=30),
                udp=dict(gps_port=1,vehicle_status_port=2,bind='127.0.0.1'),
                position=dict(map_crs='test',east_offset_m=0,north_offset_m=0))
    stream=observer.live(config,dict(route=route(),episode_id='live-test'),DEFAULT_POLICY)
    try:
        initial=next(stream)
        stale=next(stream)
        paused=next(stream)
        recovered=next(stream)
        rewound=next(stream)
        assert initial[1]['valid']
        assert not stale[1]['valid'] and stale[1]['instruction'] is None
        assert stale[1]['reason']=='gps_stale+status_stale'
        assert stale[1]['reception']['gps']['state']=='stale'
        assert paused[0]['display_only'] and not paused[1]['valid']
        assert recovered[1]['valid'] and rewound[1]['valid']
        assert recovered[1]['instruction']['instruction_instance_id']!=initial[1]['instruction']['instruction_instance_id']
        assert rewound[1]['instruction']['instruction_instance_id']!=recovered[1]['instruction']['instruction_instance_id']
    finally:
        stream.close()
    assert sorted(closed)==[1,2]


def test_live_route_loads_configured_files_without_dataset_and_rejects_wrong_geometry(tmp_path):
    from inference.navigation_observer import load_live_route
    map_root=tmp_path/'map';map_root.mkdir()
    points=[[0,0],[10,0],[20,0]]
    (map_root/'link_set.json').write_text(json.dumps([dict(idx='road',from_node_idx='a',
        to_node_idx='b',points=points)]))
    (map_root/'node_set.json').write_text('[]')
    global_path=tmp_path/'selected_path.txt';global_path.write_text('# local metric coordinates\n0 0\n10 0\n20 0\n')
    definition=tmp_path/'routes.yaml';definition.write_text('chosen:\n  route_links: [road]\n')
    policy=tmp_path/'policy.json'
    policy.write_text(json.dumps(dict(schema='lmdrive-navigation-build-v1',
        datasets=[dict(root=str(tmp_path/'not-collected-yet'),format='portable')],
        output=str(tmp_path/'unused-output'),policy=DEFAULT_POLICY)))
    source=dict(global_path=str(global_path),map_root=str(map_root),route_definition=str(definition),
                route_key='chosen',policy_config=str(policy))
    episode,manifest,entry=load_live_route(source)
    assert episode['route']['points']==points and episode['frames']==[] and entry is None
    assert manifest['policy']==DEFAULT_POLICY
    assert episode['sources']['global_path']['path']==str(global_path)
    assert NavigationEngine(episode['route'],manifest['policy']).update(frame(0,5))['valid']
    global_path.write_text('0 0\n10 5\n20 0\n')
    with pytest.raises(ValueError,match='geometry must match'):
        load_live_route(source)
    global_path.write_text('0 0\nNaN 0\n20 0\n')
    with pytest.raises(ValueError,match='finite XY'):
        load_live_route(source)


@pytest.mark.parametrize('gps,status',[(0,10909),(10506,65536),(10506,10506),(True,10909),('10506',10909)])
def test_observer_rejects_invalid_or_conflicting_ports(gps,status):
    from inference.navigation_observer import validate_udp
    with pytest.raises(ValueError,match='distinct integer UDP ports'):
        validate_udp(dict(bind='0.0.0.0',gps_port=gps,vehicle_status_port=status))


@pytest.mark.parametrize('packets,sample,age,state',[
    (0,None,float('inf'),'no_packets'),(10,None,float('inf'),'no_valid_sample'),
    (10,{'x':1},1.,'stale'),(10,{'x':1},.01,'ok')])
def test_reception_diagnostics_distinguish_transport_parsing_and_freshness(packets,sample,age,state):
    from types import SimpleNamespace
    from inference.navigation_observer import reception, reception_text
    receiver=SimpleNamespace(port=10506,packets=packets,samples=0 if sample is None else 10)
    status=reception(receiver,sample,age,.5)
    assert status['state']==state
    assert state.upper() in reception_text(dict(gps=status,status=status))
    json.dumps(status,allow_nan=False)


def test_live_observer_keeps_waiting_after_startup_timeout(monkeypatch):
    import sys
    from types import SimpleNamespace
    from inference import navigation_observer as observer
    class Receiver:
        def __init__(self,port,bind): self.port=port
        def latest(self): return None,float('inf')
        def close(self): pass
    monkeypatch.setitem(sys.modules,'inference.morai_udp',SimpleNamespace(
        GpsReceiver=Receiver,VehicleStatusReceiver=Receiver))
    monkeypatch.setitem(sys.modules,'pyproj',SimpleNamespace(Transformer=SimpleNamespace(
        from_crs=lambda *args,**kwargs:None)))
    times=iter([0,31,32])
    monkeypatch.setattr(observer,'time',SimpleNamespace(monotonic=lambda:next(times),
        monotonic_ns=lambda:32000000000,sleep=lambda _:None))
    config=dict(runtime=dict(hz=20,max_age_s=.5,max_pair_skew_s=.2,startup_timeout_s=30),
                udp=dict(gps_port=10506,vehicle_status_port=10909,bind='0.0.0.0'),
                position=dict(map_crs='test'))
    stream=observer.live(config,dict(route=route(),episode_id='waiting'),DEFAULT_POLICY)
    try:
        for _ in range(2):
            _,result=next(stream)
            assert not result['valid'] and result['instruction'] is None
            assert result['reason']=='gps_no_packets+status_no_packets'
            assert result['reception']['startup_wait_exceeded']
    finally:
        stream.close()


@pytest.mark.parametrize('rotation',[0.,1.2,-2.4])
@pytest.mark.parametrize('direction,sign',[('left',1),('right',-1)])
def test_junction_geometry_overrides_incorrect_straight_metadata(rotation,direction,sign):
    from navigation.mgeo import route_from_mgeo
    def rotate(points):
        return [[100+math.cos(rotation)*x-math.sin(rotation)*y,
                 -50+math.sin(rotation)*x+math.cos(rotation)*y] for x,y in points]
    def link(a,b,points,junction=False):
        return dict(from_node_idx=a,to_node_idx=b,points=rotate(points),
                    is_junction=junction,related_signal='straight' if junction else None)
    links=dict(approach=link('a','b',[[-50,0],[0,0]]),
               first=link('b','c',[[0,0],[5,0],[10,5*sign],[10,10*sign]],True),
               between=link('c','d',[[10,10*sign],[10,30*sign]]),
               second=link('d','e',[[10,30*sign],[10,40*sign]],True),
               exit=link('e','f',[[10,40*sign],[10,60*sign]]))
    points=[p for link in links.values() for p in link['points']]
    data=route_from_mgeo(points,list(links),links,{},dict(), 'synthetic')
    first,second=data['events']
    assert first['direction']==direction and second['direction']=='straight'
    assert first['declared_direction']=='straight'
    assert first['turn_angle_deg']==pytest.approx(90*sign)
    assert data['diagnostics'][0]['kind']=='junction_direction_metadata_conflict'
    pose=frame(0,*rotate([[-20,0]])[0]);pose['yaw_rad']=rotation
    instruction=NavigationEngine(data).update(pose)['instruction']
    assert instruction['rule']=='Turn-06-'+('L' if sign==1 else 'R')+'-S'


def test_junction_without_signal_uses_plan_but_ordinary_curve_is_not_intersection():
    from navigation.mgeo import route_from_mgeo
    link=dict(from_node_idx='a',to_node_idx='b',is_junction=True,
              points=[[0,0],[10,0],[10,10],[10,20]])
    data=route_from_mgeo(link['points'],['curve'],dict(curve=link),{},dict(),'synthetic')
    assert data['events'][0]['direction']=='left'
    link['is_junction']=False
    data=route_from_mgeo(link['points'],['curve'],dict(curve=link),{},dict(),'synthetic')
    assert not data['events']


def test_junction_uturn_is_blocked_instead_of_mislabeled_as_left_or_straight():
    from navigation.mgeo import route_from_mgeo
    link=dict(from_node_idx='a',to_node_idx='b',is_junction=True,related_signal='straight',
              points=[[0,0],[10,0],[10,10],[0,10]])
    data=route_from_mgeo(link['points'],['uturn'],dict(uturn=link),{},dict(),'synthetic')
    assert data['events'][0]['kind']=='blocked'
    assert data['events'][0]['reason']=='uturn_not_in_lmdrive_catalog'
