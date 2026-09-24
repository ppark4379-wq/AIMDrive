"""Anchor/route correctness and a read-only, expiring local visualization."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest
import torch

from inference.live_pipeline import current_window
from inference.realtime_vldrive import validate_config
from inference.vldrive_runtime import TokenHistory
from navigation import live
from utils.inference_dashboard.server import Dashboard

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def provider(monkeypatch):
    route = dict(points=[[i,0] for i in range(201)],closed=False,events=[dict(
        id='left',kind='turn',start_index=50,end_index=60,direction='left',junction=True,
        evidence='synthetic_route_fixture')])
    episode = dict(route=route,episode_id='test',sources={})
    monkeypatch.setattr(live,'load_live_route',lambda _: (episode,{'policy':{}},None))
    document = dict(schema='morai-navigation-observer-v1',route={'fixture':True},
        position=dict(map_crs='EPSG:32652',east_offset_m=0.,north_offset_m=0.),
        runtime=dict(max_age_s=.5,max_pair_skew_s=.2))
    nav = live.LiveNavigation(document)
    nav.converter = SimpleNamespace(transform=lambda longitude,latitude:(longitude,latitude))
    return nav


def snapshots(x=25.,sim_time=1.,stamp=10.,seq=1):
    gps=(dict(longitude=x,latitude=0.),.01,seq,stamp)
    status=(dict(yaw=0.,speed=4.,brake=0.,sim_time=sim_time),.01,seq,stamp)
    return gps,status


def test_navigation_uses_current_pose_and_keeps_same_timestamp_stable(provider):
    gps,status=snapshots()
    result=provider.update(gps,status,anchor_stamp=10.,now=10.02)
    assert result['valid'] and 'left' in result['instruction']['text'].lower()
    duplicate=provider.update(gps,status,anchor_stamp=10.,now=10.03)
    assert duplicate['instruction']['instruction_instance_id']==result['instruction']['instruction_instance_id']
    gps,status=snapshots(x=100.,sim_time=1.1,stamp=10.1,seq=2)
    later=provider.update(gps,status,anchor_stamp=10.1,now=10.12)
    assert later['valid'] and later['instruction']['rule']!=result['instruction']['rule']
    assert later['pose']['x']==100.


@pytest.mark.parametrize('failure', ['missing','stale','skew','offroute','invalid_pose'])
def test_navigation_failure_never_reuses_old_instruction(provider,failure):
    gps,status=snapshots()
    assert provider.update(gps,status,anchor_stamp=10.,now=10.01)['valid']
    anchor,now=10.1,10.11
    gps,status=snapshots(sim_time=1.1,stamp=10.1,seq=2)
    if failure=='missing': gps=(None,float('inf'),0,None)
    if failure=='stale': now=10.8
    if failure=='skew': gps=(*gps[:3],9.8)
    if failure=='offroute': gps=(dict(longitude=25.,latitude=50.),.01,2,10.1)
    if failure=='invalid_pose': status=(dict(status[0],yaw=float('nan')),.01,2,10.1)
    result=provider.update(gps,status,anchor_stamp=anchor,now=now)
    assert not result['valid'] and result['instruction'] is None


def test_simulator_rewind_creates_new_navigation_instance(provider):
    first=provider.update(*snapshots(sim_time=30.),anchor_stamp=10.,now=10.01)
    reset=provider.update(*snapshots(sim_time=1.,stamp=10.1,seq=2),anchor_stamp=10.1,now=10.11)
    assert reset['valid'] and reset['instruction']['instruction_instance_id']!=first['instruction']['instruction_instance_id']


def test_token_history_keeps_anchor_text_and_camera_not_latest_global_state():
    history=TokenHistory()
    for i in range(11):
        history.append(torch.zeros(106,256),stamp=10.+.2*i,ids={'lidar':i},
                       context=dict(instruction=str(i),front_jpeg=bytes([i])))
    _,rows=history.snapshot()
    assert rows[-1]['context']==dict(instruction='10',front_jpeg=bytes([10]))
    assert current_window(rows,history,.3,now=12.1)
    assert not current_window(rows,history,.3,now=12.5)
    history.clear()
    assert not current_window(rows,history,.3,now=12.1)


def test_rejected_intermediate_sample_only_expires_history_when_gap_exceeds_limit():
    history=TokenHistory()
    history.append(torch.zeros(106,256),stamp=10.,ids={'lidar':1})
    # A rejected 10.1-second sample does not destroy the prior valid history.
    history.expire(now=10.1)
    assert len(history.snapshot()[1])==1
    history.append(torch.ones(106,256),stamp=10.2,ids={'lidar':3})
    assert len(history.snapshot()[1])==2
    history.expire(now=10.47)
    assert not history.snapshot()[1] and history.resets==1


@pytest.mark.parametrize('raw_history,navigation,missing,expected', [
    (True, None, [], False),
    (True, {'valid':False}, [], False),
    (True, {'valid':True}, [], True),
    (True, {'valid':True}, ['rgb_right'], False),
    (False, None, [], True),
    (False, {'valid':False}, [], False),
    (False, {'valid':True}, [], True),
    (False, {'valid':True}, ['rgb_right'], True),
])
def test_input_availability_rejects_invalid_raw_input_and_keeps_legacy_semantics(
        raw_history,navigation,missing,expected):
    from inference.input_windows import RawWindowHistory
    from inference.live_pipeline import input_available
    history=RawWindowHistory() if raw_history else TokenHistory()
    assert input_available(history,dict(navigation=navigation,missing=missing)) is expected


def test_live_navigation_remains_stable_across_accepted_200ms_frames(provider):
    first=provider.update(*snapshots(sim_time=1.),anchor_stamp=10.,now=10.01)
    second=provider.update(*snapshots(sim_time=1.2,stamp=10.2,seq=2),anchor_stamp=10.2,now=10.21)
    assert first['instruction']['instruction_instance_id']==second['instruction']['instruction_instance_id']


def result_fixture():
    return dict(frame=25,waypoints_raw=[[1.,-5.],[2.,-10.],[3.,-15.],[4.,-20.],[5.,-25.]],
        selected_tokens=33,horizons_s=[.5,1.,1.5,2.,2.5],total_inference_ms=100.,
        anchor_received_at=10.,instruction='Turn left.',navigation=dict(valid=True,
        instruction=dict(text='Turn left.',rule='Turn-01'),pose=None,progress_m=25.))


def dashboard(clock):
    return Dashboard(dict(enabled=True,open_browser=False,port=0,refresh_hz=5.,max_prediction_age_s=1.),clock=lambda:clock[0])


def test_dashboard_expiry_exact_camera_metric_axes_and_get_only_http():
    clock=[10.1]
    view=dashboard(clock)
    view.show_prediction(result_fixture(),b'anchor-camera')
    url=view.start()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def get(path):
        with opener.open(url+path,timeout=2) as response:
            return response.read()
    try:
        state=json.loads(get('/api/state'))
        assert state['prediction_valid'] and state['receive_only']
        assert state['result']['plot_waypoints_xy'][0]==[-1.,5.]
        assert get('/api/frame.jpg?frame=25')==b'anchor-camera'
        with pytest.raises(urllib.error.HTTPError): get('/api/frame.jpg?frame=24')
        with pytest.raises(urllib.error.HTTPError): get('/../../configs/vldrive_realtime.json')
        with pytest.raises(urllib.error.HTTPError) as error:
            opener.open(urllib.request.Request(url+'/api/state',data=b'{}',method='POST'),timeout=2)
        assert error.value.code==501
        clock[0]=11.2
        assert json.loads(get('/api/state'))['result'] is None
        with pytest.raises(urllib.error.HTTPError): get('/api/frame.jpg?frame=25')
    finally:
        view.close()


def test_dashboard_clear_and_bounded_camera_memory():
    view=dashboard([10.1])
    for i in range(10): view.show_prediction(dict(result_fixture(),frame=i),b'jpeg')
    assert len(view.images)==3
    view.update(status='navigation_unavailable',clear_prediction=True,navigation=None)
    assert view.snapshot()['result'] is None and not view.images


@pytest.mark.parametrize('kind',['gps_missing','control_port','bad_refresh','nan_skew'])
def test_live_config_rejects_unsafe_or_incomplete_stream_settings(kind):
    cfg=json.loads((ROOT/'configs/vldrive_realtime.json').read_text())
    validate_config(cfg)
    if kind=='gps_missing': del cfg['ports']['gps']
    if kind=='control_port': cfg['ports']['control']=9090
    if kind=='bad_refresh': cfg['visualization']['refresh_hz']=1000
    if kind=='nan_skew': cfg['navigation']['max_anchor_skew_s']=float('nan')
    with pytest.raises(ValueError): validate_config(cfg)


@pytest.mark.parametrize('rate',[None,.5,5.])
def test_prediction_limit_can_be_removed_or_explicitly_restored(rate):
    cfg=json.loads((ROOT/'configs/vldrive_realtime.json').read_text())
    cfg['runtime']['prediction_hz']=rate
    validate_config(cfg)


@pytest.mark.parametrize('rate',[0,-1,True,float('nan'),float('inf'),6.])
def test_prediction_limit_rejects_invalid_values(rate):
    cfg=json.loads((ROOT/'configs/vldrive_realtime.json').read_text())
    cfg['runtime']['prediction_hz']=rate
    with pytest.raises(ValueError): validate_config(cfg)


def test_navigation_and_pipeline_do_not_transmit_to_simulator():
    for name in ('inference/live_pipeline.py','navigation/live.py','navigation/route_source.py'):
        tree=ast.parse((ROOT/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                assert node.func.attr not in {'send','sendto','sendall','publish','connect'}


@pytest.mark.parametrize('interval,batch_size',[(.04,1),(.08,1),(.2,1),(.2,2)])
def test_live_pipeline_passes_anchor_navigation_and_camera_with_real_history(tmp_path,monkeypatch,interval,batch_size):
    import time
    from inference import live_pipeline as pipeline
    cfg=json.loads((ROOT/'configs/vldrive_realtime.json').read_text())
    cfg.pop('navigation_token_config', None)  # This fixture mocks the language model.
    cfg['runtime'].update(device='cpu',history_tolerance_s=.06 if interval==.2 else .03,prediction_hz=25.,
                          resample_history=interval==.2, capture_hz=10., encoder_max_hz=20., encoder_batch_size=batch_size)
    instances=[]
    class Source:
        def __init__(self,*args,**kwargs):
            self.started=time.monotonic();self.closed=False;self.builds=0
            self.lidar=SimpleNamespace(snapshot=self.snapshot)
            instances.append(self)
        def snapshot(self):
            now=time.monotonic();seq=int((now-self.started)/.04)+1
            stamp=self.started+(seq-1)*.04
            return True,now-stamp,seq,stamp
        def stream_report(self):
            _,age,seq,_=self.snapshot()
            return [(name,port,seq,seq,age) for name,port in cfg['ports'].items()]
        def build(self,*args):
            self.builds+=1
            _,_,seq,stamp=self.snapshot()
            ids={key:seq for key in cfg['ports'] if key not in ('gps','imu')}
            return {'rgb_front':torch.tensor(float(seq))},dict(missing=[],received_at={k:stamp for k in ids},
                sample_ids=ids,sim_time=seq*.04,front_jpeg=str(seq).encode())
        def snapshot_raw(self, **kwargs):
            _,_,seq,stamp=self.snapshot()
            ids={key:seq for key in cfg['ports'] if key not in ('gps','imu')}
            meta=dict(missing=[],received_at={k:stamp for k in ids},sample_ids=ids,
                      sim_time=seq*.04,front_jpeg=str(seq).encode())
            return dict(meta=meta), meta
        def prepare_snapshot(self, raw, device='cpu'):
            self.builds+=1
            return {'rgb_front':torch.tensor([float(raw['meta']['sample_ids']['lidar'])])},raw['meta']
        def close(self): self.closed=True
    monkeypatch.setattr(pipeline,'LiveFrameSource',Source)
    monkeypatch.setattr(pipeline,'load_stage2',lambda *args:(None,{'updates':3730}))
    monkeypatch.setattr(pipeline,'load_encoder',lambda *args:None)
    monkeypatch.setattr(pipeline,'encode_frame',lambda _,inputs:(
        torch.full((106,256),float(inputs['rgb_front'])),1.))
    monkeypatch.setattr(pipeline,'encode_frames',lambda _,inputs:(
        torch.stack([torch.full((106,256),float(value)) for value in inputs['rgb_front']]),1.))
    monkeypatch.setattr(pipeline,'navigation_for_frame',lambda provider,cfg,meta:dict(
        valid=True,instruction=dict(text='Instruction '+str(meta['sample_ids']['lidar']),rule='fixture')))
    predicted=[]
    def predict(model,features,text,**kwargs):
        assert kwargs['optimized'] is True
        assert features.shape==(1,11,106,256)
        index=int(features[0,-1,0,0])
        assert text=='Instruction '+str(index)
        assert bool((features[0,1:,0,0]>features[0,:-1,0,0]).all())
        predicted.append(index)
        return dict(waypoints_raw=[[0.,-float(i)] for i in range(1,6)],selected_tokens=32,vldrive_ms=1.)
    monkeypatch.setattr(pipeline,'predict_window',predict)
    class View:
        def __init__(self): self.frames=[]
        def update(self,**kwargs): pass
        def show_prediction(self,result,jpeg):
            assert str(result['frame']).encode()==jpeg
            assert result['instruction']=='Instruction '+str(result['frame'])
            self.frames.append(result['frame'])
    view=View();report={}
    pipeline.run_live(cfg,dict(meta={'encoder_contract':{}},history_interval_s=interval),tmp_path,report,
                      duration=4.0 if interval==.2 else 1.7,navigation=None,view=view)
    rows=[json.loads(line) for line in (tmp_path/'waypoints.jsonl').read_text().splitlines()]
    assert len(rows)>3 and instances[0].closed
    assert view.frames==[r['frame'] for r in rows]
    assert report['full_live_forward_tested'] and all(len(r['history_sample_ids'])==11 for r in rows)
    if interval==.08:
        # The 25Hz fake receiver has >40 frames; only sampled frames are decoded.
        assert instances[0].builds<30


@pytest.mark.parametrize('failure', ['healthy','navigation_none','navigation_invalid','sensor_missing'])
def test_live_pipeline_rechecks_input_after_forward_before_publishing(tmp_path,monkeypatch,failure):
    """A current timed window is insufficient if inputs fail during LLM work.

    Run the real live output loop and thread lifecycle with CPU tensors only.
    The fixture producer publishes exactly one valid window, then changes input
    availability while the mocked forward is in progress. No ports/controllers
    are created, and the original current-window age remains valid throughout.
    """
    import threading
    import time
    from inference import live_pipeline as pipeline

    cfg=json.loads((ROOT/'configs/vldrive_realtime.json').read_text())
    cfg.pop('navigation_token_config',None)
    cfg['runtime'].update(device='cpu',resample_history=True,prediction_hz=None)
    cfg['visualization']['max_prediction_age_s']=10.
    valid_nav=dict(valid=True,instruction=dict(text='Continue straight.',rule='fixture'))
    forward_started=threading.Event()
    inputs_changed=threading.Event()
    input_ready=threading.Event()
    calls=[]
    sources=[]

    class Source:
        def __init__(self,*args,**kwargs):
            self.closed=False
            sources.append(self)
        def stream_report(self):
            return [(name,port,11,11,.01) for name,port in cfg['ports'].items()]
        def close(self):
            self.closed=True

    class FixtureProducer:
        def __init__(self,cfg,source,history,state,stop,*args,**kwargs):
            self.history,self.state,self.stop=history,state,stop
        def collect(self):
            self.state['navigation']=valid_nav
            input_ready.set()
            if not forward_started.wait(2.):
                return
            if failure=='navigation_none':
                self.state['navigation']=None
            elif failure=='navigation_invalid':
                self.state['navigation']=dict(valid=False,reason='gps_stale',instruction=None)
            elif failure=='sensor_missing':
                self.state['missing']=['rgb_right']
            inputs_changed.set()
            self.stop.wait(2.)
        def encode(self):
            if not input_ready.wait(2.):
                return
            anchor=time.monotonic()
            context=dict(navigation=valid_nav,front_jpeg=b'anchor',sim_time=2.,preprocessing_ms=1.)
            rows=[dict(stamp=anchor-(10-i)*.2,ids={'lidar':i+1},generation=0,
                       feature=torch.full((106,256),float(i)),ready=None,vision_ms=1.,
                       context=context) for i in range(11)]
            self.history.publish(rows)
            self.stop.wait(2.)

    def predict(model,features,text,**kwargs):
        assert features.shape==(1,11,106,256) and features.device.type=='cpu'
        assert text=='Continue straight.'
        calls.append(True)
        forward_started.set()
        assert inputs_changed.wait(1.), 'Fixture never changed availability during forward'
        return dict(waypoints_raw=[[0.,-float(i)] for i in range(1,6)],
                    selected_tokens=32,vldrive_ms=1.)

    class View:
        def __init__(self): self.results=[]
        def update(self,**kwargs): pass
        def show_prediction(self,result,jpeg): self.results.append(result)

    monkeypatch.setattr(pipeline,'LiveFrameSource',Source)
    monkeypatch.setattr(pipeline,'WindowInputProducer',FixtureProducer)
    monkeypatch.setattr(pipeline,'load_stage2',lambda *args:(None,{'fixture':True}))
    monkeypatch.setattr(pipeline,'load_encoder',lambda *args:None)
    monkeypatch.setattr(pipeline,'predict_window',predict)
    view,report=View(),{}
    pipeline.run_live(cfg,dict(meta={'encoder_contract':{}},history_interval_s=.2),
                      tmp_path,report,duration=.3,navigation=None,view=view)
    rows=(tmp_path/'waypoints.jsonl').read_text().splitlines()
    assert len(calls)==1 and sources[0].closed
    if failure=='healthy':
        assert len(rows)==len(view.results)==report['outputs']==1
        assert report['discarded_predictions']==0
    else:
        assert rows==[] and view.results==[] and report['outputs']==0
        assert report['discarded_predictions']==1
