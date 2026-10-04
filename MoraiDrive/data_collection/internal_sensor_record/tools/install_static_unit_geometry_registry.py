#!/usr/bin/env python3
"""Build an additive V5 registry from 36 reviewed, independently checked 3D profiles."""
import argparse,copy,hashlib,json
from pathlib import Path

def load(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--profiles',type=Path,required=True);p.add_argument('--confirmation',type=Path,required=True);p.add_argument('--camera-review',type=Path,required=True);p.add_argument('--base-registry',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 profiles,confirm,camera,base=map(load,[a.profiles,a.confirmation,a.camera_review,a.base_registry])
 expected=set(load(a.selection)['models']);ph=digest(a.profiles)
 if len(expected)!=36 or set(profiles['models'])!=expected:raise ValueError('Expected exactly the selected 36 unit models')
 if confirm['profile_sha256']!=ph or confirm['capture_complete'] is not True:raise ValueError('Incomplete or different-profile confirmation')
 evidence_path=Path(confirm['source_report']);source=load(evidence_path)
 if digest(evidence_path)!=confirm['source_sha256'] or source.get('fatal_error') or source.get('execution_complete') is not True:raise ValueError('Changed or incomplete capture')
 calibration={profiles['source_sha256']}|{x['sha256'] for x in profiles.get('additional_sources',[])}
 if confirm['source_sha256'] in calibration:raise ValueError('Confirmation was used for calibration')
 if camera['profile_sha256']!=ph or camera['confirmation_sha256']!=digest(a.confirmation):raise ValueError('Camera review belongs to different evidence')
 checks={x['model']:x for x in confirm['results']};reviews=camera['models']
 if set(checks)!=expected or set(reviews)!=expected:raise ValueError('Missing reviewed model')
 registry=copy.deepcopy(base)
 for name in sorted(expected):
  check=checks[name];review=reviews[name];profile=profiles['models'][name]
  metrics=[m for m in check['metrics'] if m['points']>=10]
  if not (check['lidar_pass'] is True and len(metrics)>=6 and len({m['view'] for m in metrics})>=4 and len({m['distance_m'] for m in metrics})>=2 and min(m['containment'] for m in metrics)>=.95):raise ValueError('LiDAR check failed: '+name)
  if review['pass'] is not True or len(review['images'])<1:raise ValueError('Camera review incomplete: '+name)
  for image in review['images']:
   if digest(Path(image['path']))!=image['sha256']:raise ValueError('Changed reviewed image')
  profile=copy.deepcopy(profile);profile.update(approved=True,validation_scope='static_unit_visible_surface_multiview_3d',evidence_sha256=digest(a.confirmation),camera_review_pass=True,lidar_pass=True)
  registry['models']['obstacle|'+name+'|1,1,1']={'approved':True,'evidence_sufficient':True,'reason':None,'center_source':'static_unit_surface_3d','size_source':'static_unit_surface_3d','geometry_3d':profile,'confirmation':{'views':len(metrics),'min_containment':min(m['containment'] for m in metrics)}}
 registry['approved_models']=sorted(k for k,v in registry['models'].items() if v.get('approved'))
 registry['static_unit_3d_validation']={'models':sorted(expected),'approved_count':36,'scope':'Static unit-scale visible-surface geometry, 0.10m containment tolerance; temporal/animation/moving sensor synchronization remains separately audited.','profiles_sha256':ph,'confirmation_sha256':digest(a.confirmation),'camera_review_sha256':digest(a.camera_review),'base_registry_sha256':digest(a.base_registry),'selection_sha256':digest(a.selection)}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(registry,indent=2,ensure_ascii=False,allow_nan=False)+'\n');print('Installed 36 unit static 3D profiles:',a.output)
if __name__=='__main__':main()
