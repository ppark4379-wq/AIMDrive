#!/usr/bin/env python3
"""MORAI sensors -> VLDrive waypoints; LMDrive lateral / expert speed control."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Same isolated Linux dependency set as train.py; never install at runtime.
if sys.platform.startswith('linux') and (ROOT / '.runtime/stage2_wsl_deps').is_dir():
    sys.path.insert(0, str(ROOT / '.runtime/stage2_wsl_deps'))
if sys.platform.startswith('linux') and (ROOT / '.runtime/offline_viz_wsl_deps').is_dir():
    sys.path.insert(0, str(ROOT / '.runtime/offline_viz_wsl_deps'))
from utils.runtime import configure_runtime
configure_runtime()

from inference.vldrive_runtime import load_contract, local_path, read_json
from inference.live_frame import CAMERA_KEYS
from inference.live_pipeline import run_live, write_report
import torch


def validate_config(cfg):
    if cfg['schema'] != 'morai-vldrive-realtime-v1':
        raise ValueError('Unsupported realtime config')
    if not isinstance(cfg['navigation_instruction'], str) or not cfg['navigation_instruction'].strip():
        raise ValueError('navigation_instruction must be nonempty')
    required = set(CAMERA_KEYS) | {'lidar', 'vehicle_status'}
    if not required <= set(cfg['ports']) or set(cfg['ports']) - required - {'gps', 'imu'}:
        raise ValueError('Only MORAI sensor receive ports are accepted')
    ports = list(cfg['ports'].values())
    if len(set(ports)) != len(ports) or any(type(p) is not int or not 0 < p < 65536 for p in ports):
        raise ValueError('Sensor ports must be unique valid UDP port numbers')
    for key in ('target_hz', 'max_age_s', 'max_stream_skew_s', 'history_tolerance_s',
                'startup_timeout_s', 'cpu_threads'):
        if type(cfg['runtime'][key]) not in (int,float) or not math.isfinite(cfg['runtime'][key]) or cfg['runtime'][key] <= 0:
            raise ValueError('Runtime values must be positive: ' + key)
    if type(cfg['runtime']['encoder_queue_size']) is not int or not 1<=cfg['runtime']['encoder_queue_size']<=5:
        raise ValueError('encoder_queue_size must be an integer in 1..5')
    rate=cfg['runtime']['prediction_hz']
    if rate is not None and (type(rate) not in (int,float) or not math.isfinite(rate) or rate<=0):
        raise ValueError('prediction_hz must be null (unlimited) or positive and finite')
    runtime = cfg['runtime']
    capacity = runtime.get('input_buffer_frames', 64)
    if type(capacity) is not int or not 11 <= capacity <= 128:
        raise ValueError('input_buffer_frames must be an integer in 11..128')
    for key in ('resample_history', 'optimized_window'):
        if type(runtime.get(key, False)) is not bool:
            raise ValueError(key+' must be boolean')
    capture_hz = runtime.get('capture_hz', runtime['target_hz'])
    if type(capture_hz) not in (int,float) or not math.isfinite(capture_hz) or not 0 < capture_hz <= 20:
        raise ValueError('capture_hz must be positive and <= 20')
    encoder_hz = runtime.get('encoder_max_hz', 10.)
    if type(encoder_hz) not in (int,float) or not math.isfinite(encoder_hz) or not 0 < encoder_hz <= 20:
        raise ValueError('encoder_max_hz must be positive and <= 20')
    gap = runtime.get('history_reset_gap_s', 1.)
    if type(gap) not in (int,float) or not math.isfinite(gap) or gap <= 1/runtime['target_hz'] + runtime['history_tolerance_s']:
        raise ValueError('history_reset_gap_s must exceed one trained interval plus tolerance')
    if capture_hz != runtime['target_hz'] and not runtime.get('resample_history', False):
        raise ValueError('Changing capture_hz requires resample_history')
    if capture_hz < runtime['target_hz']:
        raise ValueError('capture_hz must be at least the trained history target_hz')
    if runtime.get('resample_history', False) and runtime['history_tolerance_s'] >= .5/runtime['target_hz']:
        raise ValueError('history_tolerance_s must be below half the trained interval')
    for key, minimum, maximum in (
        ('history_max_span_s', 10/runtime['target_hz'], 12/runtime['target_hz']),
        ('history_max_adjacent_gap_s', 1/runtime['target_hz'], 2/runtime['target_hz']+runtime['history_tolerance_s']),
    ):
        if key in runtime and (type(runtime[key]) not in (int,float) or not math.isfinite(runtime[key])
                               or not minimum <= runtime[key] <= maximum+1e-8):
            raise ValueError(f'{key} must be within {minimum:g}..{maximum:g}')
    if type(runtime.get('pin_encoder_inputs', False)) is not bool:
        raise ValueError('pin_encoder_inputs must be boolean')
    if type(runtime.get('encoder_stream_priority', 0)) is not int or runtime.get('encoder_stream_priority', 0) not in (-1, 0):
        raise ValueError('encoder_stream_priority must be -1 or 0')
    if type(runtime.get('encoder_batch_size', 1)) is not int or runtime.get('encoder_batch_size', 1) not in (1, 2):
        raise ValueError('encoder_batch_size must be 1 or 2')
    batch_wait = runtime.get('encoder_batch_wait_s', 1/runtime['target_hz']+.02)
    if type(batch_wait) not in (int,float) or not math.isfinite(batch_wait) or not 0 < batch_wait <= 1/runtime['target_hz']+runtime['history_tolerance_s']:
        raise ValueError('encoder_batch_wait_s must be positive and within one interval plus tolerance')
    if rate is not None and rate > capture_hz:
        raise ValueError('prediction_hz must not exceed capture_hz')
    if type(cfg['runtime'].get('separate_encoder_stream',False)) is not bool:
        raise ValueError('separate_encoder_stream must be boolean')
    nav = cfg['navigation']
    if set(nav)!={'mode','config','max_anchor_skew_s'} or nav['mode'] not in ('route','fixed'):
        raise ValueError('navigation requires route/fixed mode, config and max_anchor_skew_s')
    if type(nav['max_anchor_skew_s']) not in (int,float) or not math.isfinite(nav['max_anchor_skew_s']) or nav['max_anchor_skew_s']<=0:
        raise ValueError('navigation.max_anchor_skew_s must be positive and finite')
    if nav['mode']=='route' and ('gps' not in cfg['ports'] or not isinstance(nav['config'],str) or not nav['config'].strip()):
        raise ValueError('Route navigation requires GPS receive port and navigation config')
    view = cfg['visualization']
    if set(view)!={'enabled','open_browser','port','refresh_hz','max_prediction_age_s'}:
        raise ValueError('Invalid visualization settings')
    if any(type(view[k]) is not bool for k in ('enabled','open_browser')):
        raise ValueError('visualization enabled/open_browser must be boolean')
    if type(view['port']) is not int or not 0<=view['port']<65536:
        raise ValueError('Invalid local visualization TCP port')
    for key in ('refresh_hz','max_prediction_age_s'):
        if type(view[key]) not in (int,float) or not math.isfinite(view[key]) or view[key]<=0:
            raise ValueError('Invalid visualization '+key)
    if view['refresh_hz']>15:
        raise ValueError('Display refresh_hz must not exceed 15')
    control=cfg.get('control',dict(enabled=False,config='configs/vldrive_control.json'))
    if set(control)!={'enabled','config'} or type(control['enabled']) is not bool or not isinstance(control['config'],str) or not control['config']:
        raise ValueError('Invalid control settings')
    if control['enabled'] and nav['mode']!='route':
        raise ValueError('PID control requires live route navigation and GPS pose')


def load_navigation(cfg):
    if cfg['navigation']['mode']=='fixed':
        return None
    from navigation.live import LiveNavigation
    document = read_json(local_path(cfg['navigation']['config']))
    for sensor,key in [('gps','gps_port'),('vehicle_status','vehicle_status_port')]:
        if document['udp'][key]!=cfg['ports'][sensor]:
            raise ValueError('Shared '+sensor+' port must match navigation observer config')
    return LiveNavigation(document,max_anchor_skew_s=cfg['navigation']['max_anchor_skew_s'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/vldrive_realtime.json'))
    parser.add_argument('--instruction', help='Override the fixed navigation instruction')
    parser.add_argument('--offline-smoke', action='store_true', help='Verify A-D on recorded validation sensors')
    parser.add_argument('--probe-only', action='store_true', help='Check receive streams without loading networks')
    parser.add_argument('--duration', type=float, help='Limit live acquisition seconds (also caps startup wait)')
    parser.add_argument('--no-browser', action='store_true', help='Serve the local display without opening a browser')
    control_options = parser.add_mutually_exclusive_group()
    control_options.add_argument('--control', action='store_true', help='Enable LMDrive lateral PID + expert longitudinal PID and UDP control')
    control_options.add_argument('--no-control', action='store_true', help='Disable UDP control and only display inference during manual driving')
    args = parser.parse_args()
    cfg = read_json(args.config if args.config.is_file() else local_path(str(args.config)))
    cfg.setdefault('control',dict(enabled=False,config='configs/vldrive_control.json'))
    if args.control:
        cfg['control']['enabled']=True
    elif args.no_control:
        cfg['control']['enabled']=False
    if args.instruction is not None:
        cfg['navigation_instruction'] = args.instruction
        cfg['navigation']['mode'] = 'fixed'
    if args.no_browser:
        cfg['visualization']['open_browser'] = False
    validate_config(cfg)
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration must be positive')
    if args.offline_smoke and args.probe_only:
        parser.error('--offline-smoke and --probe-only are mutually exclusive')
    if cfg['control']['enabled'] and (args.offline_smoke or args.probe_only):
        parser.error('Control cannot be combined with offline smoke or sensor probe')
    torch.set_num_threads(cfg['runtime']['cpu_threads'])
    output = local_path(cfg['output_dir']) / datetime.now().strftime('vldrive_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='vldrive-inference-report-v1', config=cfg, output=str(output),
                  status='starting', stage='contract', reconstruction=False,
                  selection_mode='gumbel', model_mode='eval', receive_only=not cfg['control']['enabled'])
    print('Output: ' + str(output), flush=True)
    exit_code = 0
    view = None
    controller = None
    try:
        if not args.offline_smoke and cfg['visualization']['enabled']:
            from utils.inference_dashboard.server import Dashboard
            view = Dashboard(cfg['visualization'],model_label=cfg['stage2_checkpoint'])
            report['visualization_url'] = view.start()
            view.update(status='contract',receive_only=report['receive_only'])
        if cfg['control']['enabled']:
            from control.config import load_config as load_control_config
            from control.runtime import ControllerSession
            control_path=local_path(cfg['control']['config'])
            control_config=load_control_config(control_path)
            report['control_config']=control_config
            controller=ControllerSession(control_config,control_path,output,view=view)
        navigation = load_navigation(cfg) if not args.offline_smoke else None
        if navigation is not None:
            planned = dict(episode_id=navigation.episode['episode_id'],route=navigation.episode['route'],
                           sources=navigation.episode['sources'],policy=navigation.policy)
            write_report(output/'planned_route.json',planned)
            report['navigation_sources'] = planned['sources']
            if view is not None:
                view.set_route(navigation.episode)
        contract = load_contract(cfg)
        report.update(stage1_sha256=contract['meta']['checkpoint_sha256'],
                      encoder_recipe=contract['meta']['encoder_contract'],
                      history_interval_s=contract['history_interval_s'],
                      training_complete=contract['training_complete'])
        if args.offline_smoke:
            from inference.vldrive_smoke import run_offline
            run_offline(cfg, contract, output, report)
        else:
            run_live(cfg, contract, output, report, args.duration, args.probe_only,
                     navigation=navigation,view=view,controller=controller)
    except KeyboardInterrupt:
        report['status'] = 'stopped_by_user'
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        print(report['error'], file=sys.stderr)
        exit_code = 1
    finally:
        if controller is not None:
            controller.close()
            report['control']=controller.state
        write_report(output / 'inference_report.json', report)
        if view is not None:
            view.close()
    print('Result: ' + report['status'], flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
