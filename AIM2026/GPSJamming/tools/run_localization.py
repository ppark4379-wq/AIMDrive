#!/usr/bin/env python3
"""Observe/record GPS, IMU and speed through the shared UDP layer; no model or commands."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from AIM2026.UDP import UDPManager
from AIM2026.GPSJamming.gps_jamming.localization.service import create_localization


def load_configuration(path,record_dir=None):
    import yaml
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    config=yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(config,dict) or (config.get('Localization') or {}).get('enabled') is not True:
        raise ValueError('Localization.enabled must be true for this location-only tool')
    config=deepcopy(config)
    config['Detection']={}
    config['Ego']['cmd_port']=None
    config['Ego']['cmd_source_port']=None
    if record_dir is not None:config['Localization']['record_dir']=str(Path(record_dir).resolve())
    return config,path.resolve()


def location_manager(config):
    manager=UDPManager(config)
    # Reuse the same manager/receiver implementations, with only the three
    # localization inputs enabled in this standalone process.
    manager.sensor_config={key:value for key,value in manager.sensor_config.items()
                           if key in ('gps','imu','vehicle_status')}
    manager._validate_ports()
    return manager


def check_configuration(config):
    manager=location_manager(config)
    dry=deepcopy(config)
    dry['Localization']['record_dir']=None
    empty=lambda:()
    receivers={key:SimpleNamespace(port=value['dest_port'],packets=0,samples=0,
               recent=empty,diagnostics=lambda:{},close=lambda:None)
               for key,value in manager.sensor_config.items()}
    service=None
    try:
        service=create_localization(SimpleNamespace(receivers=receivers),dry)
        _,_,meta=service.latest_snapshot()
        return dict(status='CHECKED',mode='localization_only',opened_sockets=0,model_loaded=False,
            command_sender_enabled=False,sensors={key:value['dest_port'] for key,value in manager.sensor_config.items()},
            test_blackout_enabled=service.hold.zone is not None,
            initial_position_mode=meta.get('position_mode'),reason=meta.get('reason'))
    finally:
        if service is not None:service.close()
        manager.close()


def run(config,seconds):
    manager=location_manager(config)
    begin=time.monotonic()
    try:
        manager.start()
        if manager.cmd is not None or manager.detection is not None:
            raise RuntimeError('location-only sender isolation failed')
        while seconds==0 or time.monotonic()-begin<seconds:
            print(json.dumps(dict(status='OBSERVING',elapsed_s=round(time.monotonic()-begin,3),
                localization=manager.localization.diagnostics()),ensure_ascii=False,allow_nan=False),flush=True)
            remaining=seconds-(time.monotonic()-begin) if seconds else 1.
            time.sleep(max(0.,min(1.,remaining)))
    except KeyboardInterrupt:
        pass
    finally:
        manager.close()
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('AIM2026/config.yaml'))
    parser.add_argument('--record-dir',type=Path)
    parser.add_argument('--seconds',type=float,default=0.)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds<0:parser.error('--seconds must be finite and nonnegative')
    try:
        config,path=load_configuration(args.config,args.record_dir)
        if args.check:
            result=check_configuration(config)
            result['config']=str(path)
            print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
            return 0
        def interrupt(*unused):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,interrupt)
        return run(config,args.seconds)
    except (ValueError,RuntimeError,OSError,KeyError) as exc:
        parser.exit(1,'localization: '+str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
