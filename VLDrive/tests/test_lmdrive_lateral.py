"""Independent upstream-math parity and split lateral/longitudinal cadence."""
import copy
from pathlib import Path

import numpy as np
import pytest

from control.config import load_config, validate_config
from control.expert_controller import ExpertWaypointController
from control.lmdrive_lateral import LMDriveLateralPID


@pytest.fixture
def cfg():
    cfg=load_config(Path(__file__).resolve().parents[1]/'../AIM2026/Control/vldrive_control.json')
    cfg['lateral']['target']['mode']='first_two_midpoint'  # Original-reference regression.
    return cfg


def test_original_lmdrive_steering_math_over_full_window(cfg):
    ctrl=LMDriveLateralPID(cfg['lateral'])
    window=[0.]*40
    for k in range(160):
        points=np.array([[np.sin(k/7)*i*3.,i*2.] for i in range(1,6)])
        speed=0. if k in (0,41,42) else 5.
        aim=(points[0]+points[1])/2.
        error=float(np.degrees(np.pi/2-np.arctan2(aim[1],aim[0]))/90)
        if speed<.01:error=0.
        window=window[1:]+[error]
        expected=1.25*error+.75*np.mean(window)+.3*(window[-1]-window[-2])
        out=ctrl.step(points,speed,k/10)
        assert out['lateral_updated']
        assert out['steering_pid_raw']==pytest.approx(expected,abs=1e-12)
        assert out['steer_pid_left_positive']==pytest.approx(np.clip(expected,-1,1)*.8)
        assert out['steering_p']+out['steering_i']+out['steering_d']==pytest.approx(expected)


def test_ten_hz_lateral_holds_while_expert_longitudinal_updates_thirty_hz(cfg):
    ctrl=ExpertWaypointController(cfg)
    rows=[ctrl.step([[i,4*i] for i in range(1,6)],1.,[.5,1,1.5,2,2.5],now=k/30)
          for k in range(90)]
    assert [k for k,r in enumerate(rows) if r['lateral_updated']]==list(range(0,90,3))
    # Longitudinal updates continue between lateral ticks; a speed cap can
    # make throttle decrease, so monotonic acceleration is not required.
    assert len({row['accel'] for row in rows[:3]})==3
    assert rows[0]['steer']==rows[1]['steer']==rows[2]['steer']
    assert all(r['expert_sampling_time_s']==1/30 for r in rows)


def test_scheduling_jitter_and_stall_do_not_replay_fake_pid_samples(cfg):
    ctrl=LMDriveLateralPID(cfg['lateral']);points=np.array([[1.,4.],[2.,8.]])
    times=[0.,.034,.068,.102,.136,.17,.204,.238,.272,.306,1.2,1.21,1.3]
    updated=[t for t in times if ctrl.step(points,2.,t)['lateral_updated']]
    assert updated==[0.,.102,.204,.306,1.2,1.3]
    assert np.count_nonzero(ctrl.pid._window)==6


def test_sign_scale_and_reset(cfg):
    for side in (-1,1):
        ctrl=ExpertWaypointController(cfg)
        out=ctrl.step([[side*100.,.1]]*5,2.,[.5,1,1.5,2,2.5])
        assert out['steer']==side*.8 and out['steering_saturated']
        assert out['steering_requested_deg']==-side*32.
        ctrl.reset()
        out=ctrl.step([[0.,i] for i in range(1,6)],2.,[.5,1,1.5,2,2.5])
        assert out['steer']==0. and out['lateral_updated']


@pytest.mark.parametrize('key,value',[('turn_KP',float('nan')),('turn_KD',-1),
    ('turn_n',1),('turn_n',2.5),('update_hz',31),('update_hz',0),('steering_scale',1.1)])
def test_bad_lateral_config(cfg,key,value):
    cfg=copy.deepcopy(cfg);cfg['lateral'][key]=value
    with pytest.raises(ValueError):validate_config(cfg)
