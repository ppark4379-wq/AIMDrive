"""Independent image/point-cloud motion evidence for raw alignment diagnostics.

Motion estimators are diagnostic measurements with confidence, not GT. A lag
search on one short drive does not calibrate camera exposure or LiDAR firing.
"""
from collections import defaultdict
import math

import cv2
import numpy as np
from scipy.spatial import cKDTree

from async_capture.alignment import PayloadReader, decode, distribution, group_streams, recorded_classes
from async_capture.journal import verify_run


def cloud_xyz(msg):
    fields = {f.name: f for f in msg.fields}
    if any(k not in fields or fields[k].datatype != 7 or fields[k].count != 1 for k in ('x','y','z')):
        raise ValueError('XYZ must be float32 scalars')
    if len(msg.data) != msg.row_step*msg.height or msg.row_step < msg.width*msg.point_step:
        raise ValueError('Malformed PointCloud2 layout')
    endian = '>' if msg.is_bigendian else '<'
    dtype = np.dtype(dict(names=['x','y','z'], formats=[endian+'f4']*3,
                         offsets=[fields[k].offset for k in ('x','y','z')], itemsize=msg.point_step))
    arr = np.ndarray((msg.height,msg.width), dtype=dtype, buffer=msg.data,
                     strides=(msg.row_step,msg.point_step))
    return np.column_stack([arr[k].ravel() for k in ('x','y','z')])


def scan_features(xyz):
    r = np.linalg.norm(xyz[:,:2],axis=1)
    points = xyz[np.isfinite(xyz).all(axis=1) & (r>4) & (r<60) & (xyz[:,2]>-.9) & (xyz[:,2]<4)]
    if not len(points):
        return np.empty((0,2))
    _, ix = np.unique(np.floor(points/.3).astype(np.int32),axis=0,return_index=True)
    points = points[np.sort(ix)]
    if len(points)>2500:
        points=points[np.linspace(0,len(points)-1,2500).astype(int)]
    return points[:,:2].astype(float)


def planar_icp(previous, current):
    """Map current cloud into previous cloud without consulting Ego/IMU."""
    if min(len(previous),len(current))<80:
        return dict(valid=False, reason='few_scene_points')
    tree = cKDTree(previous)
    transformed = current.copy()
    rotation, translation = np.eye(2), np.zeros(2)
    for _ in range(25):
        dist, ix = tree.query(transformed)
        keep = dist < min(2., max(.001, float(np.percentile(dist,75))))
        if keep.sum()<60:
            return dict(valid=False,reason='few_correspondences')
        a,b = transformed[keep],previous[ix[keep]]
        ma,mb = a.mean(axis=0),b.mean(axis=0)
        u,_,vt = np.linalg.svd((a-ma).T@(b-mb))
        rot = vt.T@u.T
        if np.linalg.det(rot)<0:
            vt[-1]*=-1
            rot=vt.T@u.T
        shift = mb-rot@ma
        transformed=transformed@rot.T+shift
        translation=rot@translation+shift
        rotation=rot@rotation
        if np.linalg.norm(shift)<1e-5 and abs(math.atan2(rot[1,0],rot[0,0]))<1e-6:
            break
    dist,_ = tree.query(transformed)
    fraction = float(np.mean(dist<.5))
    trimmed_rmse = float(np.sqrt(np.mean(np.sort(dist)[:max(1,int(.75*len(dist)))]**2)))
    return dict(valid=fraction>=.6 and trimmed_rmse<=.4,
                translation_xy=translation.tolist(), yaw_delta_rad=math.atan2(rotation[1,0],rotation[0,0]),
                displacement_m=float(np.linalg.norm(translation)), inlier_fraction_0_5m=fraction,
                trimmed_rmse_m=trimmed_rmse)


def optical_motion(previous, current):
    points = cv2.goodFeaturesToTrack(previous, maxCorners=300, qualityLevel=.02,minDistance=7)
    if points is None or len(points)<20:
        return dict(valid=False,reason='few_visual_features')
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(previous,current,points,None,winSize=(21,21),maxLevel=3)
    back, status2, _ = cv2.calcOpticalFlowPyrLK(current,previous,nxt,None,winSize=(21,21),maxLevel=3)
    a,b=points.reshape(-1,2),nxt.reshape(-1,2)
    good=(status.ravel()>0)&(status2.ravel()>0)&(np.linalg.norm(back.reshape(-1,2)-a,axis=1)<1)
    good &= (b[:,0]>=0)&(b[:,0]<current.shape[1])&(b[:,1]>=0)&(b[:,1]<current.shape[0])
    if good.sum()<20:
        return dict(valid=False,reason='few_consistent_tracks')
    d=b[good]-a[good]
    return dict(valid=True,tracks=int(good.sum()),median_flow_pixels=float(np.median(np.linalg.norm(d,axis=1))),
                median_horizontal_pixels=float(np.median(d[:,0])))


def extract_scene_signals(raw_run, rows, info):
    cv2.setNumThreads(2)
    manifest,_=verify_run(raw_run,payloads=False,require_success=not info.get('diagnostic_failed_run',False))
    reader, classes, streams = PayloadReader(raw_run), recorded_classes(manifest), group_streams(rows)
    signals, summary = {}, {}
    try:
        for name in ('camera:Camera-1','camera:Camera-2','camera:Camera-3','camera:Camera-4','lidar'):
            previous, output, errors = None, [], []
            for r in streams.get(name,[]):
                if not info['active_start_ns']<=r['receive_monotonic_ns']<=info['active_end_ns']:
                    continue
                row=dict(record_id=r['record_id'],receive_monotonic_ns=r['receive_monotonic_ns'],
                         source_timestamp_ns=r.get('source_timestamp_ns'))
                try:
                    msg=decode(r,reader.read(r),classes)
                    if name.startswith('camera:'):
                        img=cv2.imdecode(np.frombuffer(msg.data,dtype=np.uint8),cv2.IMREAD_COLOR)
                        if img is None or img.ndim!=3:
                            raise ValueError('Image decode failed')
                        gray=cv2.cvtColor(cv2.resize(img,(384,216)),cv2.COLOR_BGR2GRAY)
                        row.update(width=int(img.shape[1]),height=int(img.shape[0]),
                                   luma_mean=float(gray.mean()),luma_std=float(gray.std()))
                        # All four views get actual motion checks, not only a JPEG hash.
                        feature=gray
                        if previous:
                            row['motion']=optical_motion(previous[1],feature)
                    else:
                        xyz=cloud_xyz(msg)
                        row.update(points=len(xyz),finite_points=int(np.isfinite(xyz).all(axis=1).sum()),
                                   fields=[f.name for f in msg.fields],
                                   per_point_time_available=any(f.name in ('time','t','timestamp','offset_time') for f in msg.fields))
                        feature=scan_features(xyz)
                        if previous:
                            row['motion']=planar_icp(previous[1],feature)
                    if previous:
                        dt=(r['receive_monotonic_ns']-previous[0]['receive_monotonic_ns'])/1e9
                        row.update(interval_start_ns=previous[0]['receive_monotonic_ns'],interval_s=dt)
                        row['motion']['interval_valid']=0<dt<.3
                    previous=(r,feature)
                except Exception as exc:
                    # Keep every failed input as a diagnostic failure, never an empty scene.
                    row['error']=type(exc).__name__+': '+str(exc)
                    errors.append(dict(record_id=r['record_id'],error=row['error']))
                    previous=None
                output.append(row)
            signals[name]=output
            summary[name]=dict(records=len(output),errors=errors,
                valid_motion_intervals=sum(r.get('motion',{}).get('valid',False) and r.get('motion',{}).get('interval_valid',False) for r in output))
    finally:
        reader.close()
    return signals,summary


def compare_yaw(ego, imu):
    """Shape agreement under trial receive offsets; no timestamp modification."""
    if min(len(ego),len(imu))<20:
        return dict(status='insufficient_samples')
    origin=ego[0]['receive_monotonic_ns']
    te=np.array([(r['receive_monotonic_ns']-origin)/1e9 for r in ego])
    ti=np.array([(r['receive_monotonic_ns']-origin)/1e9 for r in imu])
    ye=np.unwrap([r['signal']['yaw'] for r in ego])
    yi=np.unwrap([r['signal']['yaw'] for r in imu])
    excitation=float(np.ptp(ye))
    if excitation<math.radians(5):
        return dict(status='insufficient_rotation',ego_heading_range_degrees=math.degrees(excitation))
    # Same evaluation interval for every lag. Arbitrary constant yaw frame offset
    # is removed; a sign reversal is not silently fitted away.
    # Flat headings / the operator pause must not dominate a lag fit.
    rate=np.abs(np.gradient(ye,te))
    use=(ti>=te[0]+.5)&(ti<=te[-1]-.5)&(np.interp(ti,te,rate)>math.radians(3))
    if use.sum()<30:
        return dict(status='insufficient_rotating_samples',samples=int(use.sum()),
                    ego_heading_range_degrees=math.degrees(excitation))
    errors=[]
    for lag in np.linspace(-.3,.3,121):
        diff=yi[use]-np.interp(ti[use]+lag,te,ye)
        diff-=np.median(diff)
        errors.append(dict(ego_time_minus_imu_receive_ms=round(float(lag*1000),6),
                           median_abs_error_degrees=float(np.median(np.abs(diff))*180/np.pi)))
    best=min(errors,key=lambda r:r['median_abs_error_degrees'])
    return dict(status='diagnostic_only',ego_heading_range_degrees=math.degrees(excitation),
                best=best,zero_offset=errors[60],curve=errors,rotating_samples=int(use.sum()),
                caveat='Single-drive receive-clock shape fit, not an exposure calibration or an absolute latency estimate')


def scene_report(signals, rows, timing):
    streams=group_streams(rows)
    report=dict(physical_simultaneity_verified=False,
                ego_imu_yaw=compare_yaw(streams.get('ego',[]),streams.get('imu',[])),holds=[])
    for h in timing['moving_pose_holds']:
        item=dict(hold=h,streams={})
        # Omit boundary pairs; quantify the interior of the independently found hold.
        for name,rs in signals.items():
            values=[]
            for r in rs:
                if not h['start_ns']+200000000<=r['receive_monotonic_ns']<=h['end_ns']-200000000:
                    continue
                m=r.get('motion',{})
                if m.get('valid') and m.get('interval_valid'):
                    key='displacement_m' if name=='lidar' else 'median_flow_pixels'
                    values.append(m[key]/r['interval_s'])
            item['streams'][name]=dict(units='m/s diagnostic ICP' if name=='lidar' else '384x216 pixels/s',
                                      measured_motion=distribution(values))
        report['holds'].append(item)
    report['lidar_point_timing']=dict(per_point_time_available=any(r.get('per_point_time_available',False) for r in signals.get('lidar',[])),
        deskew_applied=False,scan_start_end_semantics_verified=False)
    return report
