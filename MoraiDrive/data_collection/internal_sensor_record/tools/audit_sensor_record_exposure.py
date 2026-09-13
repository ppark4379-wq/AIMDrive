"""Native PNG/BIN cycle association, with deliberately wrong adjacent images.

Uses the existing fixed calibration and edge diagnostic. No offsets are fitted,
and no numerical filename is promoted to an exposure certificate.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import cv2
import numpy as np
from async_capture.camera_geometry import projection_transforms
from async_capture.journal import digest,write_json
from native_capture.geometry_validation import transform_matrix,project
from native_capture.lidar_edges import foreground_endpoints
from async_capture.exposure_metrics import edge_summary,VIEWS,SHIFTS,OPTICAL
from ssd_storage import require_ssd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--quality',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    for path in (a.audit,a.quality,a.output):require_ssd(path,reserve_bytes=0)
    a.output.mkdir(exist_ok=False)
    load=lambda p:json.loads(p.read_text())
    seal=load(a.audit/'COMPLETE.json')
    if digest(a.audit/'report.json')!=seal['files']['report.json']:raise ValueError('Native audit changed')
    catalog=load(a.audit/'report.json')
    quality=load(a.quality/'QUALITY_COMPLETE.json')
    for n,sha in quality['files'].items():
        if Path(n).name!=n or digest(a.quality/n)!=sha:raise ValueError('Quality evidence changed')
    info=load(a.quality/'INDEX_COMPLETE.json')
    if info['raw_manifest_sha256']!=catalog['raw_manifest_sha256']:raise ValueError('Different raw runs')
    source=Path(catalog['source_root']);require_ssd(source,reserve_bytes=0)
    assets={r['relative_path']:r for r in catalog['assets']}
    bundles=catalog['bundles'];contract=load(a.quality/'sensor_contract.json')
    transforms,provenance=projection_transforms(contract,
        ROOT/'evidence/native_geometry_v1/persisted_sensor_settings.json',
        ROOT/'evidence/native_geometry_v1/camera_projection_evidence.json')
    rows=load(a.quality/'active_signals.json');ego=[]
    for row in rows:
        if row['role']!='ego':continue
        if ego and row['source_timestamp_ns']<=ego[-1]['source_timestamp_ns']:
            if row['source_timestamp_ns']==ego[-1]['source_timestamp_ns'] and row['signal']==ego[-1]['signal']:continue
            raise ValueError('Conflicting Ego time/state')
        ego.append(row)
    origin=ego[0]['source_timestamp_ns'];et=np.array([(r['source_timestamp_ns']-origin)/1e9 for r in ego])
    yaw=np.unwrap([r['signal']['yaw'] for r in ego]);speed=np.array([r['signal']['speed'] for r in ego])
    cache=OrderedDict();cv2.setNumThreads(2)
    def payload(relative):
        raw=(source/relative).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=assets[relative]['sha256']:raise ValueError('Native payload changed')
        return raw
    def image_and_edges(relative):
        if relative not in cache:
            im=cv2.imdecode(np.frombuffer(payload(relative),np.uint8),cv2.IMREAD_GRAYSCALE)
            if im is None:raise ValueError('Invalid native image')
            im=cv2.resize(im,(640,round(im.shape[0]*640/im.shape[1])))
            cache[relative]=(im,cv2.distanceTransform(255-cv2.Canny(im,100,200),cv2.DIST_L2,cv2.DIST_MASK_PRECISE))
        cache.move_to_end(relative);value=cache[relative]
        while len(cache)>40:cache.popitem(last=False)
        return value
    lr,lt=transform_matrix(contract['transforms']['/base_link->Lidar3D-5']['transform']['transform'])
    edge_rows=[];excluded=[];illustrated=set()
    for gi in range(3,len(bundles)-3):
        g=bundles[gi];ts=g['filename_simulation_time_hypothesis_ns']
        # Filename has six fractional digits. Keep its original value and allow
        # only accumulated 1us quantization per adjacent interval, never a cycle.
        if any(bundles[gi+s]['missing'] or abs(bundles[gi+s]['filename_simulation_time_hypothesis_ns']-ts-s*100000000)>abs(s)*1000 for s in SHIFTS):
            excluded.append(gi);continue
        sec=(ts-origin)/1e9
        if sec-.15<et[0] or sec+.15>et[-1]:excluded.append(gi);continue
        vel=float(np.interp(sec,et,speed));yr=float(np.rad2deg(np.interp(sec+.15,et,yaw)-np.interp(sec-.15,et,yaw))/.3)
        category='turning' if vel>1 and abs(yr)>3 else 'moving' if vel>1 else 'stationary_or_slow'
        pts=np.frombuffer(payload(g['assets']['lidar']),dtype='<f4').reshape(-1,4)
        base=foreground_endpoints(pts)@lr.T+lt
        for vi,view in enumerate(VIEWS,1):
            paths=[bundles[gi+s]['assets']['camera_'+str(vi)] for s in SHIFTS]
            ims=[image_and_edges(path) for path in paths];h,w=ims[0][0].shape
            cr,ct=transform_matrix(transforms[view])
            uv,depth=project(base,cr@OPTICAL.T,ct,w,h,contract['camera_fov_degrees'][view])
            valid=np.isfinite(uv).all(axis=1)&(depth>2)&(uv[:,0]>=3)&(uv[:,0]<w-3)&(uv[:,1]>=h*.15)&(uv[:,1]<h*.76)
            if view=='left':valid&=~((uv[:,0]>w*.58)&(uv[:,1]>h*.48))
            if view=='right':valid&=~((uv[:,0]<w*.42)&(uv[:,1]>h*.48))
            xy=np.rint(uv[valid]).astype(int)
            scores=[float(np.minimum(dist[xy[:,1],xy[:,0]],20).mean()) if len(xy) else None for _,dist in ims]
            edge_rows.append(dict(view=view,category=category,group_index=gi,source_ns=ts,speed_mps=vel,
                yaw_rate_deg_s=yr,time_block=int(sec//10),points=len(xy),valid=len(xy)>=20,scores_px=scores,
                native_camera_files=paths,native_lidar_file=g['assets']['lidar']))
            if category=='turning' and view not in illustrated:
                panels=[]
                for shift,(im,_) in zip(SHIFTS,ims):
                    panel=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
                    for point in xy:cv2.circle(panel,tuple(point),1,(0,255,255),-1)
                    cv2.putText(panel,'%s native image %+d cycles'%(view,shift),(8,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
                    panels.append(panel)
                cv2.imwrite(str(a.output/(view+'_shift_controls.jpg')),np.concatenate(panels))
                illustrated.add(view)
        if gi%100==3:print('native groups checked',gi,flush=True)
    result={v:{c:edge_summary([r for r in edge_rows if r['view']==v and
                (r['category']==c or c=='all_moving' and r['category'] in ('moving','turning'))])
                for c in ('all_moving','turning','stationary_or_slow')} for v in VIEWS}
    report=dict(schema='morai_native_sensor_record_exposure_review_v1',shifted_edge_controls=result,
        shifts_in_100ms_cycles=SHIFTS,excluded_group_indices=excluded,calibration=provenance,
        input_native_audit_sha256=digest(a.audit/'report.json'),raw_manifest_sha256=catalog['raw_manifest_sha256'],
        physical_simultaneity_verified=False,teacher_time_alignment_verified=False,training_visible=False,
        limitation='Cycle sensitivity only. Occlusion/calibration/dynamic scene can affect edges; no sub-cycle exposure bound or automatic training approval.')
    write_json(a.output/'report.json',report);write_json(a.output/'samples.json',edge_rows)
    write_json(a.output/'COMPLETE.json',dict(files={p.name:digest(p) for p in a.output.iterdir()},training_visible=False))
    for view,value in result.items():print(view,json.dumps(value['all_moving']),flush=True)


if __name__=='__main__':main()
