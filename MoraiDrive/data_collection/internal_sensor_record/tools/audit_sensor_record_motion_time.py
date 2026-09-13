"""Test native-frame teacher time against dense Ego motion, without restamping.

Kinematic consistency is separate from camera exposure and dynamic-object GT.
The thresholds below are declared diagnostic tolerances, not paper parameters.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from async_capture.journal import digest, write_json
from async_capture.alignment import distribution
from ssd_storage import require_ssd


def summarize_motion(ego, native_times, *, max_gap_s=.05, absolute_error_m=.05, relative_error=.05):
    unique = []
    repeats = 0
    for row in ego:
        if unique and row['source_timestamp_ns'] <= unique[-1]['source_timestamp_ns']:
            if row['source_timestamp_ns'] == unique[-1]['source_timestamp_ns'] and row['signal'] == unique[-1]['signal']:
                repeats += 1
                continue
            raise ValueError('Ego source reversal or conflicting states at the same timestamp')
        if row.get('signal') is None:
            raise ValueError('Missing Ego signal')
        unique.append(row)
    if len(unique) < 2 or len(native_times) < 2 or any(b<=a for a,b in zip(native_times,native_times[1:])):
        raise ValueError('Ordered Ego and native frame sequence required')
    origin = unique[0]['source_timestamp_ns']
    t = np.array([(r['source_timestamp_ns']-origin)/1e9 for r in unique])
    xy = np.array([r['signal']['position'][:2] for r in unique], dtype=float)
    speed = np.linalg.norm([r['signal']['velocity'][:2] for r in unique], axis=1)
    if not np.isfinite(xy).all() or not np.isfinite(speed).all():
        raise ValueError('Nonfinite teacher values')
    dt = np.diff(t)
    dx = np.linalg.norm(np.diff(xy,axis=0),axis=1)
    dense_integral = float(np.sum((speed[1:]+speed[:-1])*.5*dt))
    intervals = []
    for i,(a,b) in enumerate(zip(native_times,native_times[1:])):
        lo,hi = (a-origin)/1e9,(b-origin)/1e9
        row = dict(before_index=i,after_index=i+1,before_source_ns=a,after_source_ns=b,
                   source_interval_s=hi-lo,training_visible=False,camera_exposure_verified=False)
        if lo<t[0] or hi>t[-1]:
            row['status']='teacher_not_bracketed';intervals.append(row);continue
        left=max(0,int(np.searchsorted(t,lo,side='right'))-1)
        right=min(len(t)-1,int(np.searchsorted(t,hi)))
        gaps=np.diff(t[left:right+1])
        if len(gaps) and gaps.max()>max_gap_s:
            row.update(status='teacher_gap',max_teacher_gap_s=float(gaps.max()))
            intervals.append(row);continue
        knots=np.r_[lo,t[(t>lo)&(t<hi)],hi]
        pos=np.column_stack([np.interp(knots,t,xy[:,k]) for k in (0,1)])
        vel=np.interp(knots,t,speed)
        integral=float(np.sum((vel[1:]+vel[:-1])*.5*np.diff(knots)))
        path=float(np.linalg.norm(np.diff(pos,axis=0),axis=1).sum())
        error=abs(path-integral)
        bad=error>absolute_error_m and error>relative_error*max(integral,.01)
        row.update(status='kinematic_mismatch' if bad else 'kinematically_consistent',
                   dense_ego_points=right-left+1,position_path_m=path,speed_integral_m=integral,
                   absolute_error_m=error,path_over_speed_integral=path/integral if integral>.01 else None,
                   ego_bracket_record_ids=[unique[left]['record_id'],unique[right]['record_id']])
        intervals.append(row)
    checked=[r for r in intervals if 'absolute_error_m' in r]
    moving=[r for r in checked if r['speed_integral_m']>.1]
    statuses={status:sum(r['status']==status for r in intervals) for status in sorted({r['status'] for r in intervals})}
    report=dict(schema='morai_native_motion_time_v1',ego_samples=len(ego),unique_source_samples=len(unique),
        identical_source_repeats=repeats,native_frames=len(native_times),interval_status_counts=statuses,
        thresholds=dict(max_teacher_gap_s=max_gap_s,absolute_error_m=absolute_error_m,relative_error=relative_error,
                        mismatch_rule='both absolute and relative error must exceed tolerance'),
        moving_interval_path_ratio=distribution([r['path_over_speed_integral'] for r in moving]),
        interval_absolute_error_m=distribution([r['absolute_error_m'] for r in checked]),
        full_ego_stream=dict(position_path_m=float(dx.sum()),speed_integral_m=dense_integral,
            relative_difference=abs(float(dx.sum())-dense_integral)/dense_integral if dense_integral else None,
            moving_position_holds=int(((dx<.001)&((speed[:-1]+speed[1:])*.5>1)).sum()),
            source_gap_count=int((dt>max_gap_s).sum())),
        all_native_intervals_consistent=bool(intervals) and all(r['status']=='kinematically_consistent' for r in intervals),
        source_timestamps_modified=False,physical_simultaneity_verified=False,training_visible=False,
        limitation='Dense Ego position and velocity share a simulator source; this is not an independent camera/LiDAR or all-actor exposure certificate')
    return report,intervals


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    for p in (args.episode,args.output):require_ssd(p,reserve_bytes=0)
    load=lambda p:json.loads(p.read_text())
    quality=args.episode/'quality';native=args.episode/'native_audit'
    for folder,name in ((quality,'QUALITY_COMPLETE.json'),(native,'COMPLETE.json')):
        for name,sha in load(folder/name)['files'].items():
            if Path(name).name!=name or digest(folder/name)!=sha:raise ValueError('Changed input evidence')
    cat=load(native/'report.json');index=load(quality/'INDEX_COMPLETE.json')
    if cat['raw_manifest_sha256']!=index['raw_manifest_sha256']:raise ValueError('Different source runs')
    exact={r['relative_path']:r for r in cat['comparison'] if r['role']=='lidar' and r.get('exact_xyzi_array_equal')}
    times=[]
    for bundle in cat['bundles']:
        if bundle['missing'] or bundle['assets']['lidar'] not in exact:raise ValueError('Incomplete/unmatched native bundle')
        times.append(exact[bundle['assets']['lidar']]['ros_source_timestamp_ns'])
    ego=[r for r in load(quality/'active_signals.json') if r['role']=='ego']
    report,intervals=summarize_motion(ego,times)
    report.update(native_catalog_sha256=digest(native/'report.json'),quality_seal_sha256=digest(quality/'QUALITY_COMPLETE.json'),
                  raw_manifest_sha256=index['raw_manifest_sha256'])
    args.output.mkdir(exist_ok=False)
    write_json(args.output/'report.json',report);write_json(args.output/'intervals.json',intervals)
    write_json(args.output/'COMPLETE.json',dict(files={n:digest(args.output/n) for n in ('report.json','intervals.json')},
        script_sha256=digest(Path(__file__)),training_visible=False))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
