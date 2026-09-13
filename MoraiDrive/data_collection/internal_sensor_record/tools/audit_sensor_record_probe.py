"""Audit direct native files against unchanged ROS bytes and independent clocks.

Filename-clock matching is a tested hypothesis, not a camera exposure seal.
"""
import argparse
from bisect import bisect_left
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
from async_capture.alignment import decode, recorded_classes, distribution
from async_capture.journal import verify_run, records, write_json, digest
from async_capture.sensor_record import audit_assets
from async_capture.native_session import recorded_native_root
from probe_realtime_trigger import validate_bytes
from ssd_storage import require_ssd, SSD_ROOT


def nearest(rows, stamp):
    times=[r[0]['source_timestamp_ns'] for r in rows]
    i=bisect_left(times,stamp)
    return min(rows[max(0,i-2):i+2],key=lambda r:abs(r[0]['source_timestamp_ns']-stamp)) if rows else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--raw',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args();require_ssd(a.run,reserve_bytes=0)
    out=a.output or a.run/'audit';require_ssd(out);out.mkdir(exist_ok=False)
    raw=a.raw or a.run/'raw';require_ssd(raw,reserve_bytes=0)
    verified=verify_run(raw);classes=recorded_classes(verified[0])
    clocks=[];streams={k:[] for k in ('gnss','imu','ego','lidar')}
    for meta,payload in records(raw,verified=verified):
        role=meta['role']
        if role=='clock':clocks.append((meta,decode(meta,payload,classes).clock.to_nsec()))
        elif role in streams:streams[role].append((meta,decode(meta,payload,classes)))
    if len(clocks)<2 or any(b[1]<x[1] for x,b in zip(clocks,clocks[1:])):
        raise ValueError('No continuous non-reversing independent ROS /clock')
    for role,rows in streams.items():
        if any(b[0]['source_timestamp_ns']<x[0]['source_timestamp_ns'] for x,b in zip(rows,rows[1:])):
            raise ValueError('Source clock reversed: '+role)
    # Original recording spans this window; never use file mtime as source time.
    native=recorded_native_root(a.run, SSD_ROOT)
    require_ssd(native,reserve_bytes=0)
    report=audit_assets(native,clocks[0][1],clocks[-1][1],validator=validate_bytes)
    comparison=[]
    for asset in report['assets']:
        stamp=asset['filename_simulation_time_hypothesis_ns'];role=asset['role']
        if role not in ('gps','imu','lidar'):continue
        stream='gnss' if role=='gps' else role
        match=nearest(streams[stream],stamp)
        if match is None:continue
        meta,msg=match
        row=dict(role=role,relative_path=asset['relative_path'],
                 filename_simulation_time_hypothesis_ns=stamp,
                 ros_record_id=meta['record_id'],ros_source_timestamp_ns=meta['source_timestamp_ns'],
                 ros_receive_monotonic_ns=meta['receive_monotonic_ns'],
                 filename_minus_ros_source_ns=stamp-meta['source_timestamp_ns'])
        if role=='gps':
            f=asset['fields']
            row['field_differences']={k:f[k]-float(getattr(msg,v)) for k,v in
                {'Latitude':'latitude','Longitude':'longitude','Altitude':'altitude',
                 'EastOffset':'eastOffset','NorthOffset':'northOffset'}.items()}
        elif role=='imu':
            f=asset['fields']
            row['orientation_max_abs_error']=max(abs(f['Orientation '+k.upper()]-getattr(msg.orientation,k)) for k in 'xyzw')
            row['raw_native_secs']=f['TimeStamp.Secs'];row['raw_native_nsecs']=f['TimeStamp.nSecs']
        else:
            blob=(native/asset['relative_path']).read_bytes()
            if digest(native/asset['relative_path'])!=asset['sha256']:raise ValueError('Native changed during comparison')
            pts=np.frombuffer(blob,dtype='<f4').reshape(-1,4)
            fields={f.name:f for f in msg.fields}
            if any(k not in fields or fields[k].datatype!=7 or fields[k].count!=1 for k in ('x','y','z','intensity')):
                raise ValueError('Expected four FLOAT32 ROS LiDAR fields')
            dt=np.dtype(dict(names=['x','y','z','intensity'],formats=['>f4' if msg.is_bigendian else '<f4']*4,
                             offsets=[fields[k].offset for k in ('x','y','z','intensity')],itemsize=msg.point_step))
            points=np.ndarray((msg.height,msg.width),dtype=dt,buffer=msg.data,strides=(msg.row_step,msg.point_step))
            xyz=np.column_stack([points[k].reshape(-1) for k in ('x','y','z','intensity')])
            row['same_shape']=pts.shape==xyz.shape
            if row['same_shape']:
                row['max_abs_xyzi_difference']=np.max(abs(pts-xyz),axis=0).tolist()
                row['exact_xyzi_array_equal']=bool(np.array_equal(pts,xyz))
        comparison.append(row)
    clock_span=(clocks[-1][1]-clocks[0][1])/1e9
    wall_span=(clocks[-1][0]['receive_monotonic_ns']-clocks[0][0]['receive_monotonic_ns'])/1e9
    report.update(raw_run=str(raw),raw_manifest_sha256=digest(raw/'raw_manifest.json'),
                  clock_samples=len(clocks),clock_span_simulation_seconds=clock_span,
                  clock_span_receive_seconds=wall_span,observed_rtf=clock_span/wall_span,
                  raw_recorder_statistics=verified[0]['statistics'],comparison=comparison,
                  comparison_note='Nearest-source candidates only; raw values and signed differences retained; no exposure approval')
    write_json(out/'report.json',report)
    write_json(out/'COMPLETE.json',dict(files={'report.json':digest(out/'report.json')},training_visible=False))
    print(json.dumps(dict(output=str(out),files_valid=report['files_valid'],counts=report['counts'],
         complete_bundles=report['complete_bundles'],errors=len(report['errors']),cadence_exceptions=report['cadence_exceptions'],
         observed_rtf=report['observed_rtf'],source_differences={role:distribution([r['filename_minus_ros_source_ns']/1e6 for r in comparison if r['role']==role]) for role in ('gps','imu','lidar')},
         lidar_exact=sum(r.get('exact_xyzi_array_equal',False) for r in comparison))),flush=True)


if __name__=='__main__':main()
