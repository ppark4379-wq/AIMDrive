#!/usr/bin/env python3
"""Install reviewed dynamic 3D envelopes additively, preserving static evidence."""
import argparse,copy,hashlib,json
from pathlib import Path

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_selection(selection, profiles, confirm, camera, base, extend_existing=False):
 expected={x['model']:x['kind'] for x in selection}
 if len(expected)!=len(selection) or not expected:raise ValueError('Empty or duplicate model selection')
 if any(set(data['models'])!=set(expected) for data in (profiles,confirm,camera)):raise ValueError('Selected models and evidence must match exactly')
 if extend_existing:
  if not base.get('dynamic_default_3d_validation'):raise ValueError('Extension requires an existing reviewed dynamic registry')
  if any(kind!='vehicle' for kind in expected.values()):raise ValueError('Extension currently supports vehicles only')
  for name in expected:
   if base['models'].get('npc|'+name+'|1,1,1',{}).get('approved'):raise ValueError('Cannot replace an already approved model: '+name)
 elif len(expected)!=65 or sum(x=='pedestrian' for x in expected.values())!=29 or sum(x=='vehicle' for x in expected.values())!=36:
  raise ValueError('Expected exactly the active 29 pedestrians and 36 vehicles')
 return expected

def main():
 p=argparse.ArgumentParser();p.add_argument('--profiles',type=Path,required=True);p.add_argument('--runtime-contracts',type=Path,required=True);p.add_argument('--confirmation',type=Path,required=True);p.add_argument('--camera-review',type=Path,required=True);p.add_argument('--base-registry',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--extend-existing',action='store_true',help='Add previously unapproved vehicles with the same independent evidence requirements');a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 profiles,confirm,camera,base=map(load,[a.profiles,a.confirmation,a.camera_review,a.base_registry]);expected=validate_selection(load(a.selection),profiles,confirm,camera,base,a.extend_existing);ph=sha(a.profiles)
 if confirm['profile_sha256']!=ph or camera['profile_sha256']!=ph or camera['confirmation_sha256']!=sha(a.confirmation):raise ValueError('Review/profile mismatch')
 calibration={profiles['source_sha256']}|{x['sha256'] for x in profiles.get('additional_sources',[])}
 for source in confirm['inputs']:
  for role in ('capture','evaluation','integrity'):
   if sha(source[role])!=source[role+'_sha256']:raise ValueError('Changed evidence: '+source[role])
  capture=load(source['capture']);evaluation=load(source['evaluation']);integrity=load(source['integrity'])
  if not capture['execution_complete'] or capture.get('fatal_error') or source['capture_sha256'] in calibration or evaluation['profile_sha256']!=ph or integrity['source_sha256']!=source['capture_sha256']:raise ValueError('Incomplete/dependent confirmation')
  for row in capture['results']:
   for path,h in row['evidence_sha256'].items():
    if sha(Path(source['capture']).parent/path)!=h:raise ValueError('Changed raw capture')
  if not all(x['all_held_integrity_candidates'] for x in integrity['results']):raise ValueError('Unstable held pose')
 contracts=load(a.runtime_contracts)
 if contracts['surface_profiles_sha256']!=ph or contracts['bounds_unchanged'] is not True or set(contracts['models'])!=set(expected):raise ValueError('Runtime API contract mismatch')
 for source in contracts['source_reports']:
  if source['sha256'] not in calibration or sha(source['path'])!=source['sha256']:raise ValueError('Runtime contract must come from unchanged calibration captures')
 registry=copy.deepcopy(base)
 for name,kind in sorted(expected.items()):
  c=confirm['models'][name];v=camera['models'][name];profile=copy.deepcopy(profiles['models'][name]);metrics=[m for m in c['metrics'] if m['points']>=10]
  if not (c['lidar_pass'] and c['configured_speed_coverage_pass'] and len(metrics)>=6 and len({m['view'] for m in metrics})>=4 and len({m['distance_m'] for m in metrics})>=2 and min(m['containment'] for m in metrics)>=.95):raise ValueError('LiDAR/motion evidence failed: '+name)
  if v['pass'] is not True or v['motion_pose_review_pass'] is not True or not v['images']:raise ValueError('Missing image/pose review: '+name)
  for im in v['images']:
   if sha(im['path'])!=im['sha256']:raise ValueError('Changed reviewed image')
  contract=contracts['models'][name]
  if contract['kind']!=kind or contract['views']<8 or max(contract['offset_range_xyz_m'])>.01:raise ValueError('Unstable current API contract: '+name)
  if kind=='pedestrian' and (contract.get('pedestrian_velocity_parameters_kmh')!=c['motion_speeds_kmh'] or contract.get('native_velocity_parameter_max_error_kmh',1)>.001):raise ValueError('Unverified native pedestrian velocity parameter: '+name)
  profile['historical_reference_raw_offset_flu_m']=profile['raw_offset_flu_m'];profile['historical_reference_raw_size_lwh_m']=profile['raw_size_lwh_m']
  profile['raw_offset_flu_m']=contract['raw_offset_flu_m'];profile['raw_size_lwh_m']=contract['raw_size_lwh_m'];profile['runtime_api_contract_sha256']=sha(a.runtime_contracts)
  profile.update(approved=True,validation_scope='dynamic_default_held_motion_surface_multiview_3d',default_scale_contract='morai_native_actor_spawn_no_scale',evidence_sha256=sha(a.confirmation),camera_review_pass=True,lidar_pass=True,motion_pose_review_pass=True,reviewed_motion_speeds_kmh=c['motion_speeds_kmh'])
  key=('npc' if kind=='vehicle' else kind)+'|'+name+'|1,1,1'
  registry['models'][key]={'approved':True,'evidence_sufficient':True,'reason':None,'center_source':'dynamic_default_surface_3d','size_source':'dynamic_default_surface_3d','geometry_3d':profile,'confirmation':{'views':len(metrics),'min_containment':min(m['containment'] for m in metrics)}}
 registry['approved_models']=sorted(k for k,v in registry['models'].items() if v.get('approved'))
 validation={'models':expected,'approved_pedestrian_count':sum(k=='pedestrian' for k in expected.values()),'approved_vehicle_count':sum(k=='vehicle' for k in expected.values()),'scope':confirm['scope'],'physical_simultaneity_verified':False,'profiles_sha256':ph,'confirmation_sha256':sha(a.confirmation),'camera_review_sha256':sha(a.camera_review),'base_registry_sha256':sha(a.base_registry),'selection_sha256':sha(a.selection),'runtime_api_contracts_sha256':sha(a.runtime_contracts)}
 if a.extend_existing:
  registry.setdefault('dynamic_default_3d_extensions',[]).append(validation)
 else:
  registry['dynamic_default_3d_validation']=validation
 implementation_root=Path(__file__).resolve().parents[1]
 registry['inherited_builder_fingerprints']={k:registry.get(k) for k in ('module_sha256','script_sha256')}
 registry['module_sha256']=sha(implementation_root/'morai_pretraining/fused_geometry.py');registry['script_sha256']=sha(__file__)
 validation['implementation_sha256']={str(p.relative_to(implementation_root)):sha(p) for p in [implementation_root/name for name in ['morai_pretraining/dynamic_default_geometry.py','morai_pretraining/fused_geometry.py','morai_pretraining/center_geometry.py','morai_pretraining/sensor_record_manifest.py','async_capture/native_gt_timeline.py','tools/build_sensor_record_gt_timeline.py']]}
 registry['inherited_registry_provenance']={'path':str(a.base_registry),'sha256':sha(a.base_registry),'policy_applies_to':'Legacy entries without geometry_3d; nested geometry_3d evidence governs measured full 3D entries'}
 registry['limitation']='Geometry profiles certify reviewed default-scale visible-surface bounds, not moving sensor exposure synchronization. Each nested geometry_3d profile defines its validation scope; inherited policy describes legacy box-source selection.'
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(registry,indent=2,ensure_ascii=False,allow_nan=False)+'\n');print('Installed %d reviewed default 3D profiles:'%len(expected),a.output)
if __name__=='__main__':main()
