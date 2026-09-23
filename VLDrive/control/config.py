"""Transport, LMDrive lateral settings and the expert's longitudinal parameters.

Keep this loader standard-library-only: the Windows UDP bridge needs no NumPy.
"""
import copy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    cfg=json.loads(Path(path).read_text(encoding='utf-8'))
    source=ROOT/cfg['expert_config_path']
    expert=json.loads(source.read_text(encoding='utf-8'))
    cfg['expert_config']={k:copy.deepcopy(expert[k]) for k in ('common','control')}
    rate=expert['common'].get('realtime_sampling_rate',expert['common']['sampling_rate'])
    cfg['runtime'].update(pid_hz=rate,send_hz=rate)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg.get('schema')!='vldrive-morai-expert-v1':
        raise ValueError('Expected shared MORAI expert control configuration')
    if any(key in cfg for key in ('pid','longitudinal','vehicle','controller')):
        raise ValueError('Set controller parameters in the expert config only')
    def positive(value,name):
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
            raise ValueError('Expected positive finite '+name)
    if cfg.get('max_speed_kmh') is not None:
        positive(cfg['max_speed_kmh'],'max_speed_kmh')
    expert=cfg['expert_config'];common=expert['common'];control=expert['control']
    for key in ('sampling_rate','realtime_sampling_rate','sync_fallback_sampling_rate','wheelbase','max_steering'):
        positive(common[key],key)
    if not 0<common['max_steering']<90:
        raise ValueError('Invalid expert maximum wheel angle')
    for key in ('realtime_steering_sign','sync_steering_sign'):
        if type(common[key]) not in (int,float) or common[key] not in (-1.,1.):
            raise ValueError('Invalid expert steering sign')
    for key in ('lfd_gain','min_lfd','max_lfd'):
        positive(control['pure_pursuit'][key],key)
    if control['pure_pursuit']['min_lfd']>control['pure_pursuit']['max_lfd']:
        raise ValueError('Invalid expert lookahead limits')
    for value in control['pid'].values():
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError('Invalid expert PID gain')
    run,bridge=cfg['runtime'],cfg['bridge']
    for key,value in run.items():positive(value,key)
    rate=common['realtime_sampling_rate']
    if not 1<=rate<=50 or run['pid_hz']!=rate or run['send_hz']!=rate:
        raise ValueError('Control and send rate must equal expert realtime_sampling_rate')
    lateral=cfg['lateral']
    if lateral.get('type')!='lmdrive_pid':
        raise ValueError('Expected LMDrive lateral PID')
    for key in ('turn_KP','turn_KI','turn_KD'):
        value=lateral[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError('Invalid lateral PID gain')
    if type(lateral['turn_n']) is not int or lateral['turn_n']<2:
        raise ValueError('Invalid lateral PID window')
    positive(lateral['update_hz'],'lateral.update_hz')
    positive(lateral['steering_scale'],'lateral.steering_scale')
    if lateral['update_hz']>rate or lateral['steering_scale']>1:
        raise ValueError('Invalid lateral update rate or steering scale')
    target=lateral['target']
    if target['mode'] not in ('remaining_path','first_two_midpoint'):
        raise ValueError('Invalid lateral target mode')
    for key in ('time_s','min_distance_m','max_distance_m','min_forward_m'):
        positive(target[key],'lateral.target.'+key)
    if not target['min_forward_m']<=target['min_distance_m']<=target['max_distance_m']:
        raise ValueError('Invalid lateral target distances')
    for key in ('watchdog_s','idle_exit_s','startup_timeout_s'):positive(bridge[key],key)
    if not 1/run['send_hz']<bridge['watchdog_s']<bridge['idle_exit_s']:
        raise ValueError('Invalid bridge watchdog timing')
    for key in ('listen_port','target_port'):
        if type(bridge[key]) is not int or not 1024<=bridge[key]<=65535:
            raise ValueError('Invalid control UDP port')
    if bridge['listen_port']==bridge['target_port'] or bridge['target_host']!='127.0.0.1':
        raise ValueError('Bridge target must be a distinct local MORAI receive port')
    if type(bridge['auto_start']) is not bool or not isinstance(bridge['host'],str) or not bridge['host']:
        raise ValueError('Invalid bridge host/auto_start')
