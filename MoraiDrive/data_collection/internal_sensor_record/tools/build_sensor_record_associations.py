"""Native images/cloud + original same-source ROS GPS/teacher diagnostic index.

The native GPS file is retained as its own artifact. It is NEVER used as the
same-time GPS merely because its filename matches the cameras. No restamping,
image shifts, pose offsets, or replacement of original files is performed.
"""
import argparse
from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pyproj import Transformer
from async_capture.alignment import PayloadReader,decode,recorded_classes,distribution
from async_capture.journal import verify_run,digest,write_json
from async_capture.sensor_record import gps_text_matches_ros
from async_capture.portable_paths import recorded_episode_directory
from ssd_storage import require_ssd,SSD_ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True)
    a=p.parse_args();require_ssd(a.episode,reserve_bytes=0)
    def load(n):return json.loads((a.episode/n).read_text())
    cat=load('native_audit/report.json');seal=load('native_audit/COMPLETE.json')
    if digest(a.episode/'native_audit/report.json')!=seal['files']['report.json']:raise ValueError('Native evidence changed')
    for n,sha in load('quality/QUALITY_COMPLETE.json')['files'].items():
        if Path(n).name!=n or digest(a.episode/'quality'/n)!=sha:raise ValueError('Quality evidence changed')
    info=load('quality/INDEX_COMPLETE.json')
    if info['raw_manifest_sha256']!=cat['raw_manifest_sha256']:raise ValueError('Different recordings')
    raw=recorded_episode_directory(info['raw_path'],a.episode,'raw',SSD_ROOT);require_ssd(raw,reserve_bytes=0)
    manifest,_=verify_run(raw,payloads=False)
    if digest(raw/'raw_manifest.json')!=info['raw_manifest_sha256']:raise ValueError('Raw identity changed')
    classes=recorded_classes(manifest);reader=PayloadReader(raw)
    signals=load('quality/active_signals.json');ego=[];gps={};gps_rows=[]
    try:
        for row in signals:
            if row['role']=='ego':
                if ego and row['source_timestamp_ns']<=ego[-1]['source_timestamp_ns']:
                    if row['source_timestamp_ns']==ego[-1]['source_timestamp_ns'] and row['signal']==ego[-1]['signal']:continue
                    raise ValueError('Ego clock reset or conflicting state')
                ego.append(row)
            elif row['role']=='gnss':
                msg=decode(row,reader.read(row),classes)
                values={k:float(getattr(msg,v)) for k,v in {'Latitude':'latitude','Longitude':'longitude',
                    'Altitude':'altitude','EastOffset':'eastOffset','NorthOffset':'northOffset'}.items()}
                t=row['source_timestamp_ns']
                if t in gps:raise ValueError('Duplicate GPS source time requires explicit disambiguation')
                gps[t]=(row,values);gps_rows.append((t,row,values))
    finally:reader.close()
    if any(b[0]<=x[0] for x,b in zip(gps_rows,gps_rows[1:])):raise ValueError('GPS source clock not strictly increasing')
    origin=ego[0]['source_timestamp_ns'];times=np.array([(r['source_timestamp_ns']-origin)/1e9 for r in ego])
    xy=np.array([r['signal']['position'][:2] for r in ego]);utm=Transformer.from_crs('EPSG:4326','EPSG:32652',always_xy=True)
    assets={r['relative_path']:r for r in cat['assets']}
    lc={r['relative_path']:r for r in cat['comparison'] if r['role']=='lidar'}
    gt=[r['source_timestamp_ns'] for r,values in gps.values()]
    associations=[];phase=[];errors=[];positions=[]
    for i,bundle in enumerate(cat['bundles']):
        if bundle['missing']:errors.append(dict(index=i,reason='missing_native_inputs'));continue
        lidar=lc.get(bundle['assets']['lidar'])
        if lidar is None or not lidar.get('exact_xyzi_array_equal'):
            errors.append(dict(index=i,reason='native_ros_lidar_content_not_equal'));continue
        t=lidar['ros_source_timestamp_ns'];native_time=bundle['filename_simulation_time_hypothesis_ns']
        if not 0<=native_time-t<1000000:
            errors.append(dict(index=i,reason='filename_lidar_source_difference_exceeds_ms_format'));continue
        if t not in gps:errors.append(dict(index=i,reason='same_source_ros_gps_missing'));continue
        current,values=gps[t]
        native_gps=assets[bundle['assets']['gps']]
        j=bisect_left(gt,t)
        candidates=[(stamp,r) for stamp,r,v in gps_rows[max(0,j-2):j+2] if gps_text_matches_ros(native_gps['fields'],v)]
        row=dict(index=i,native_filename_time_ns=native_time,native_gps_file=native_gps['relative_path'],
                 gps_content_matches=[dict(source_timestamp_ns=s,record_id=r['record_id']) for s,r in candidates])
        if len(candidates)==1:
            matched,record=candidates[0]
            row.update(status='unique_content_match',native_file_minus_measurement_ns=native_time-matched,
                       matched_gps_source_timestamp_ns=matched)
        else:row['status']='ambiguous_repeated_values' if candidates else 'unmatched_native_gps'
        phase.append(row)
        sec=(t-origin)/1e9;k=int(np.searchsorted(times,sec))
        if not times[0]<=sec<=times[-1] or k==0 or times[k]-times[k-1]>.05:
            errors.append(dict(index=i,reason='teacher_not_bracketed_within_50ms'));continue
        x,y=utm.transform(values['Longitude'],values['Latitude'])
        actual=np.array([x-values['EastOffset'],y-values['NorthOffset']])
        pred=np.array([np.interp(sec,times,xy[:,z]) for z in (0,1)])
        positions.append(float(np.linalg.norm(actual-pred)))
        associations.append(dict(index=i,scene_time_hypothesis_ns=t,native_filename_time_ns=native_time,
            camera_files={k:v for k,v in bundle['assets'].items() if k.startswith('camera_')},
            lidar_file=bundle['assets']['lidar'],lidar_ros_record_id=lidar['ros_record_id'],
            gps=dict(source='original_ros_message',record_id=current['record_id'],source_timestamp_ns=t,
                receive_monotonic_ns=current['receive_monotonic_ns'],native_gps_file_used_as_current=False),
            ego_bracket_record_ids=[ego[k-1]['record_id'],ego[k]['record_id']],
            native_imu_file=bundle['assets']['imu'],
            gps_ego_position_error_m=positions[-1],native_gps_phase_status=row['status'],
            training_visible=False,physical_simultaneity_verified=False))
    summary=dict(bundles=len(cat['bundles']),association_candidates=len(associations),errors=errors,
        native_gps_phase_counts=dict(Counter(r['status'] for r in phase)),
        native_gps_lag_ns=distribution([r['native_file_minus_measurement_ns'] for r in phase if r['status']=='unique_content_match']),
        current_ros_gps_vs_ego_error_m=distribution(positions),
        native_gps_policy='preserve file and observed phase; use original same-source ROS GPS reference for current scene',
        all_original_timestamps_preserved=True,physical_simultaneity_verified=False,training_visible=False)
    out=a.episode/'native_associations';out.mkdir(exist_ok=False)
    write_json(out/'report.json',summary);write_json(out/'associations.json',associations);write_json(out/'native_gps_phase.json',phase)
    write_json(out/'COMPLETE.json',dict(files={p.name:digest(p) for p in out.iterdir()},
        native_audit_sha256=digest(a.episode/'native_audit/report.json'),raw_manifest_sha256=cat['raw_manifest_sha256'],training_visible=False))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
