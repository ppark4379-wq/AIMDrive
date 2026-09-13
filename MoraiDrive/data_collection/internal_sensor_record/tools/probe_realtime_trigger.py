#!/usr/bin/env python3
"""Bounded RealTime SaveSensorData diagnostic with verified local redirection.

Does not configure links, drive Ego, change time mode, or approve exposure GT.
RPC failure stops issuance; a timeout is never retried. File stability is an
observation, not proof that the producer can never reopen the file.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PureWindowsPath
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from async_capture.journal import write_json, digest
from async_capture.sim_mode import require_free_running_mode
from native_capture.ground_truth import GrpcGroundTruth
from native_capture.native_text import parse_native_text
from nas_storage import powershell
from process_guard import process_lock
from ssd_storage import require_ssd, SSD_ROOT, windows_path
from runtime_paths import WINDOWS_LAUNCHER_ROOT

ALIAS='MoraiRealtimeTriggerD_v1'
DESTINATION=SSD_ROOT/'realtime_trigger_native_v1'
WINDOWS_DESTINATION=windows_path(DESTINATION)
LINKS=[str(PureWindowsPath(WINDOWS_LAUNCHER_ROOT,ALIAS)),
       str(PureWindowsPath(WINDOWS_LAUNCHER_ROOT,
                           'MoraiLauncher_Win_Data','SaveFile',ALIAS))]


def verify_working_directory():
    source=(ROOT/'tools/read_simulator_working_directory.ps1').read_text()
    temporary=windows_path(DESTINATION/'diagnostic_compiler_tmp')
    result=json.loads(powershell('& {\n'+source+"\n} -TempRoot '"+temporary+"'"))
    if result['working_directory'].rstrip('\\').casefold()!=WINDOWS_LAUNCHER_ROOT.rstrip('\\').casefold():
        raise ValueError('Unexpected simulator working directory; no native trigger issued')
    return result


def verify_links():
    script="$ErrorActionPreference='Stop'\n[Console]::OutputEncoding=New-Object System.Text.UTF8Encoding($false)\n$rows=@()\n"
    for source in LINKS:
        script+="$i=Get-Item -LiteralPath '"+source+"' -Force\n$rows+=@{source=$i.FullName;type=$i.LinkType;target=@($i.Target)}\n"
    rows=json.loads(powershell(script+'ConvertTo-Json -InputObject $rows -Depth 4 -Compress'))
    if len(rows)!=2:raise ValueError('Two verified junctions required')
    for row,expected in zip(rows,LINKS):
        if row['source']!=expected or row['type']!='Junction' or row['target']!=[WINDOWS_DESTINATION]:
            raise ValueError('Unexpected native redirect; no capture issued')
    return rows


def expected_files(name):
    return {**{'camera_'+str(i):'CAMERA_'+str(i)+'/'+name+'_Intensity.png' for i in range(1,5)},
            'lidar':'LIDAR_5/'+name+'_Intensity.bin','gps':'GPS_6/'+name+'.txt','imu':'IMU_7/'+name+'.txt'}


def validate_bytes(role,raw):
    import numpy as np
    from PIL import Image
    result={'size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    if role.startswith('camera_'):
        with Image.open(io.BytesIO(raw)) as im:
            if im.format!='PNG':raise ValueError('Expected native PNG')
            im.verify()
        with Image.open(io.BytesIO(raw)) as im:
            im.load();result.update(resolution=list(im.size),mode=im.mode)
            expected=(1280,720) if role=='camera_1' else (640,480)
            if im.size!=expected:raise ValueError('Unexpected camera resolution')
    elif role=='lidar':
        if len(raw)%16 or not 1000<=len(raw)//16<=2000000:raise ValueError('Invalid XYZI extent')
        points=np.frombuffer(raw,dtype='<f4').reshape(-1,4)
        if not np.isfinite(points).all():raise ValueError('Nonfinite XYZI')
        result['point_count']=len(points)
    else:result.update(parse_native_text(role,raw))
    return result


def observe_files(root,name,timeout=8.):
    expected=expected_files(name);previous=None;deadline=time.monotonic()+timeout
    observations=[];error='not observed'
    while time.monotonic()<deadline:
        try:
            rows={role:dict(relative_path=relative,**validate_bytes(role,(root/relative).read_bytes()))
                  for role,relative in expected.items()}
            hashes={role:r['sha256'] for role,r in rows.items()}
            observations.append({'monotonic_ns':time.monotonic_ns(),'sha256':hashes})
            if previous==hashes:
                return dict(assets=rows,observations=observations,
                            completion_evidence='two_equal_full_reads_100ms_apart; no producer-close guarantee')
            previous=hashes
        except (OSError,ValueError) as exc:error=str(exc);previous=None
        time.sleep(.1)
    raise TimeoutError('Native files incomplete/invalid: '+error)


def stationary(gt):
    row=gt.actors()
    if not row['available']:raise ValueError('No actor observation')
    ego=next(s for s in row['response']['states'] if s['actor_info']['id']['value']=='Ego')
    if math.sqrt(sum(v*v for v in ego['velocity'].values()))>.1:
        raise ValueError('This storage probe requires stationary Ego')
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=int,default=1)
    p.add_argument('--check',action='store_true',help='Read-only environment checks; no sensor trigger')
    args=p.parse_args()
    if not 1<=args.samples<=10:p.error('Use 1..10 bounded samples')
    require_ssd(DESTINATION,check_volume=True)
    links=verify_links()
    cwd=verify_working_directory()
    from native_capture.ground_truth import default_grpc_target
    gt=GrpcGroundTruth(default_grpc_target(),timeout=5)
    try:
        with process_lock('capture'):
            mode=gt.call(gt.simulation.GetSynchronousMode,gt.Empty())
            require_free_running_mode(mode);before=stationary(gt)
            if args.check:
                print(json.dumps(dict(mode=mode,links=links,stationary=True)));return
            from morai.sensor.sensor_data_save_config_pb2 import SensorDataSaveConfig
            ident='probe_'+uuid.uuid4().hex
            out=DESTINATION/ident;out.mkdir(exist_ok=False)
            spec=dict(schema='morai_realtime_trigger_probe_v1',mode=mode,links=links,cwd=cwd,before=before,
                      training_visible=False,physical_simultaneity_verified=False,
                      sample_count=args.samples,tool_sha256=digest(Path(__file__)))
            write_json(out/'run_spec.json',spec)
            results=[];error=None
            try:
                for i in range(args.samples):
                    require_ssd(out);name='frame_%06d'%i
                    # Both producer roots resolve this relative directory to D.
                    request=SensorDataSaveConfig(is_custom_file_name=True,custom_file_name=name,
                        file_dir=ALIAS+'\\'+ident)
                    intent=dict(index=i,request=gt.to_dict(request),actors_before=stationary(gt),
                        request_prepared_monotonic_ns=time.monotonic_ns(),wall_ns=time.time_ns())
                    write_json(out/(name+'_intent.json'),intent)
                    result=gt.call(gt.sensor.SaveSensorData,request)
                    write_json(out/(name+'_rpc.json'),result)
                    if not result['available'] or result['response'].get('status')!='STATUS_CODE_SUCCESS':
                        raise RuntimeError('Trigger not acknowledged; no retry or further trigger')
                    files=observe_files(out,name)
                    row=dict(index=i,rpc=result,files=files,actors_after=stationary(gt),
                             observation_completed_monotonic_ns=time.monotonic_ns())
                    write_json(out/(name+'_observed.json'),row);results.append(row)
                    print(json.dumps(dict(output=str(out),index=i,assets=len(files['assets']),
                        rpc_seconds=(result['receive_monotonic_ns']-result['request_monotonic_ns'])/1e9)),flush=True)
            except Exception as exc:error=repr(exc)
            final=dict(status='failed' if error else 'storage_probe_completed',error=error,
                samples_observed=len(results),requested_samples=args.samples,
                mode_after=gt.call(gt.simulation.GetSynchronousMode,gt.Empty()),
                training_visible=False,physical_simultaneity_verified=False,
                evidence_sha256={p.name:digest(p) for p in out.glob('*.json')})
            write_json(out/'RESULT.json',final)
            print(json.dumps(dict(output=str(out),**final)),flush=True)
            if error:raise RuntimeError(error)
    finally:gt.close()


if __name__=='__main__':main()
