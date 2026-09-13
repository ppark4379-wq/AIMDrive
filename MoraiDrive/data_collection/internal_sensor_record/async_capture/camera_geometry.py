"""Explicit, provenance-checked projection hypotheses; not exposure certification."""
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from .journal import digest


VIEWS=('front','left','right','rear')


def check_observed_camera_tf(message,contract):
    """Check recorded live transforms against the raw contract, before correction."""
    result={}
    for transform in message.transforms:
        child=transform.child_frame_id.lstrip('/')
        if transform.header.frame_id.lstrip('/')!='base_link' or child not in ('Camera-1','Camera-2','Camera-3','Camera-4'):continue
        expected=contract['transforms']['/base_link->'+child]['transform']['transform']
        observed=transform.transform
        a=np.array([getattr(observed.translation,k) for k in ('x','y','z')]);b=np.array([expected['translation'][k] for k in ('x','y','z')])
        q=np.array([getattr(observed.rotation,k) for k in ('x','y','z','w')]);p=np.array([expected['rotation'][k] for k in ('x','y','z','w')])
        if not np.isfinite(a).all() or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>.01:
            raise ValueError('Invalid live camera transform: '+child)
        distance=float(np.linalg.norm(a-b));angle=float((Rotation.from_quat(p).inv()*Rotation.from_quat(q)).magnitude())
        if distance>.001 or angle>np.deg2rad(.01):raise ValueError('Recorded camera TF differs from calibration contract: '+child)
        result[child]=dict(translation_error_m=distance,rotation_error_rad=angle)
    return result


def projection_transforms(contract, preset_path=None, evidence_path=None):
    recorded={view:copy.deepcopy(contract['transforms']['/base_link->Camera-'+str(i)]['transform']['transform'])
              for i,view in enumerate(VIEWS,1)}
    if preset_path is None and evidence_path is None:
        return recorded,dict(mode='recorded_ros_tf',warning='Observed TF can differ from rendering rotation; hypothesis only')
    if preset_path is None or evidence_path is None:
        raise ValueError('Both recorded sensor preset and measured projection evidence are required')
    preset=json.loads(Path(preset_path).read_text());evidence=json.loads(Path(evidence_path).read_text())
    if digest(preset_path)!=evidence['sensor_file_sha256'] or evidence['fov_axis']!='horizontal':
        raise ValueError('Projection evidence does not match this preset')
    for i,view in enumerate(VIEWS,1):
        cameras=[r for r in preset['cameraList'] if r['m_SensorUniqueID']==i]
        if len(cameras)!=1:raise ValueError('Expected one camera for each unique ID')
        cam=cameras[0];tf=evidence['physical_transforms'][view]
        xyz=np.array([float(cam['pos'][k]) for k in ('x','y','z')])
        observed=np.array([recorded[view]['translation'][k] for k in ('x','y','z')])
        physical=np.array([tf['translation'][k] for k in ('x','y','z')])
        if not np.isfinite(xyz).all() or max(np.linalg.norm(xyz-observed),np.linalg.norm(xyz-physical))>.001:
            raise ValueError('Preset/captured/evidence camera positions differ')
        if abs(float(cam['cc']['cameraFOV'])-contract['camera_fov_degrees'][view])>1e-6:
            raise ValueError('Preset and captured FOV differ')
        rotation=Rotation.from_euler('xyz',[float(cam['rot'][k]) for k in ('roll','pitch','yaw')],degrees=True)
        measured=Rotation.from_quat([tf['rotation'][k] for k in ('x','y','z','w')])
        if (rotation.inv()*measured).magnitude()>1e-6:raise ValueError('Evidence/preset rotation differs')
        recorded[view]['rotation']=dict(zip(('x','y','z','w'),rotation.as_quat().tolist()))
    return recorded,dict(mode='measured_preset_xyz',preset_sha256=digest(preset_path),
        projection_evidence_sha256=digest(evidence_path),scope=evidence['scope'],
        original_ros_tf_preserved=True,physical_exposure_verified=False)
