#!/usr/bin/env python3
import argparse,sys,importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
R=Path('/home/dydwns/AIMDrive/MoraiDrive/data_collection/internal_sensor_record');L=Path('/home/dydwns/workspace/MoraiDrive/local_integration/morai_visual_pretraining');sys.path[:0]=[str(R),str(L)]
from morai_pretraining.center_geometry import pivot_local_offset, pedestrian_velocity_parameter
from morai_pretraining.fused_geometry import check_fused_object_geometry
from morai_pretraining.sensor_record_manifest import objects_for_entry
from tools.build_dynamic_surface_profiles import samples,context,load,write,sha

spec=importlib.util.spec_from_file_location('production_gt_timeline',R/'tools/build_sensor_record_gt_timeline.py')
timeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(timeline)

sys.path.insert(0, '/home/dydwns/AIMDrive/VLDrive/data/src')
from loaders.morai_map_derivation import correct_entry

def main(a):
 registry=load(a.registry);sources=load(a.root/'configured_sources.json')['confirmation'];out=[]
 for source in sources:
  root=Path(source);geo,_=context(root)
  for row in load(root/'report.json')['results']:
   for s in samples(root,row,geo):
    actor=load(s['path']/'gt.json')['before'];box=actor['bounding_box'];actor_id=actor['actor_info']['id']['value']
    payload={'response':{'states':[actor]},'actor_instances':{actor_id:{'instance_key':'probe','model_name':row['model'],'model_identity_valid':True}}}
    normalized=timeline.grpc_object_value(payload,{})
    assert normalized['complete_inventory'] and normalized['model_identity_complete']
    item=next(iter(normalized['objects'].values()));kind=item['kind']
    pivot={'valid':True,'offset_xyz_m':pivot_local_offset(actor),'offset_range_xyz_m':[0,0,0],'max_actor_roll_pitch_deg':max(abs((actor['transform']['rotation'][k]+180)%360-180) for k in 'xy'),'max_box_pivot_yaw_difference_deg':abs((actor['transform']['rotation']['z']-box['rotation']['z']+180)%360-180)}
    if kind=='pedestrian':pivot['pedestrian_velocity_parameters_kmh']=[pedestrian_velocity_parameter(actor)]
    result=check_fused_object_geometry(item,{'api_box_geometry_candidate':False,'reasons':['old_geometry']},registry,None,None,pivot_evidence=pivot)
    if not result['api_box_geometry_candidate']:raise ValueError((row['model'],result))
    result['instance_key']='probe';target=objects_for_entry({'object_gt':{'objects':[item]}},{'objects':[result]})[0]
    # Match the existing production manifest pipeline, including its map-basis stage.
    # No fitted yaw correction: use the independently recorded map CRS/origin.
    item['instance_key']='probe'
    before={'frame_id':0,'sensor_time_ns':0,'ego_pose':{'x':float(s['pivot'][0]),'y':float(s['pivot'][1]),'yaw':0.},'waypoints':[[0.,0.]]*5,'objects':[target],'object_valid':True,'traffic_light':-1}
    frame={'index':0,'lidar_source_ns':0,'object_gt':{'objects':[item]}}
    audit={'index':0,'lidar_source_ns':0,'objects':[result]}
    corrected,flags=correct_entry(before,frame,audit,geo,{})
    assert not flags['object_mask_reasons']
    target=corrected['objects'][0]
    points=s['points']@s['ar'].T+s['pivot'];center=np.array([target[k] for k in ('x','y','z')]);rot=Rotation.from_euler('xyz',target['rotation_rpy_deg'],degrees=True).as_matrix();ext=np.array([target['length'],target['width'],target['height']])/2;outside=np.maximum(np.abs((points-center)@rot)-ext,0)
    containment=float((outside.max(1)<=.1).mean()) if len(points) else None;expected=s['pivot']+s['ar']@registry['models'][kind+'|'+row['model']+'|1,1,1']['geometry_3d']['center_offset_flu_m'];error=float(np.linalg.norm(center-expected))
    if len(points)>=10 and containment<.95:raise ValueError((row['model'],'manifest containment',containment))
    if error>=.05:raise ValueError((row['model'],'center projection discrepancy',error))
    out.append({'model':row['model'],'source':str(root),'directory':s['directory'],'points':len(points),'containment_10cm':containment,'center_difference_from_grid_projection_m':error,'manifest_geometry_valid':target['geometry_valid']})
 report={'normalization':'production grpc_object_value from the recorded actor state; no raw bbox or velocity replacement','map_correction':'production loaders.morai_map_derivation.correct_entry', 'registry_sha256':sha(a.registry),'source_reports':[{'path':str(Path(s)/'report.json'),'sha256':sha(Path(s)/'report.json')} for s in sources],'models':len({r['model'] for r in out}),'views':len(out),'usable_views':sum(r['points']>=10 for r in out),'minimum_containment_10cm':min(r['containment_10cm'] for r in out if r['points']>=10),'max_center_difference_m':max(r['center_difference_from_grid_projection_m'] for r in out),'results':out};write(a.root/'pipeline_application_report.json',report);print({k:v for k,v in report.items() if k not in ('results','source_reports')})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--registry',type=Path,required=True);main(p.parse_args())
