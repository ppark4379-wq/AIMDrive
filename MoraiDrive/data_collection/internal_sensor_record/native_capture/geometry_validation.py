"""Independent geometric checks; none of these functions authorizes training."""
import itertools
import copy
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def preset_camera_transform(contract, index, preset_path=None):
    """Explicit rendering-rotation hypothesis; preserve the observed ROS TF.

    A stored preset is evidence only for the matching SHA256. This function
    never silently reads a changed current MORAI configuration.
    """
    from async_capture.journal import digest
    native=contract['native']
    path=Path(preset_path or native['sensor_file_source'])
    if digest(path)!=native['sensor_file_sha256']:
        raise ValueError('Camera preset SHA256 differs from the capture contract')
    preset=json.loads(path.read_text())
    camera=next(c for c in preset['cameraList'] if c['m_SensorUniqueID']==index)
    result=copy.deepcopy(contract['transforms']['/base_link->Camera-'+str(index)]['transform']['transform'])
    q=Rotation.from_euler('xyz',[float(camera['rot'][k]) for k in ('roll','pitch','yaw')],degrees=True).as_quat()
    result['rotation']=dict(zip(('x','y','z','w'),q.tolist()))
    return result


def transform_matrix(transform):
    r=Rotation.from_quat([transform['rotation'][k] for k in ('x','y','z','w')]).as_matrix()
    t=np.array([transform['translation'][k] for k in ('x','y','z')])
    return r,t


def lidar_ring_error(points,translation=(0.,0.,0.),rotation=None):
    """Test measured interleaved 16-beam pattern without fitting Ego poses.

    Near-origin zero-intensity returns independently expose a rotating beam
    origin (radius .04191 m, z .0378 m) in the recorded model. Positions are
    not corrected for learning: these offsets only reconstruct emitted rays.
    """
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]<4 or not len(points) or len(points)%16 or not np.isfinite(points).all():
        raise ValueError('This measured ray test requires complete interleaved 16-channel scans')
    xyz=(points[:,:3]-np.asarray(translation)) @ (np.eye(3) if rotation is None else rotation)
    valid=np.linalg.norm(points[:,:3],axis=-1)>2.
    if not valid.any():raise ValueError('No usable LiDAR returns for the ray comparison')
    expected=np.array([-15,1,-13,3,-11,5,-9,7,-7,9,-5,11,-3,13,-1,15])
    elevation=np.rad2deg(np.arctan2(xyz[:,2]-.0378,np.linalg.norm(xyz[:,:2],axis=-1)-.04191))
    errors=np.abs(elevation-np.tile(expected,len(points)//16))
    near=(np.linalg.norm(points[:,:3],axis=-1)<.2)&(points[:,3]==0.)
    return dict(usable_returns=int(valid.sum()),near_origin_no_return_candidates=int(near.sum()),
        error_deg_p50_p95_p99_max=np.percentile(errors[valid],[50,95,99,100]).tolist(),
        beam_angles_deg=expected.tolist(),channels=16,azimuth_samples=len(points)//16,
        observed_near_origin_radius_m=None if not near.any() else np.percentile(np.linalg.norm(points[near,:2],axis=-1),[0,50,100]).tolist(),
        observed_near_origin_z_m=None if not near.any() else np.percentile(points[near,2],[0,50,100]).tolist())


def actor_box(actor,extent_order='length_width_height'):
    b=actor['bounding_box']
    center=np.array([b['location'][k] for k in ('x','y','z')])
    extent=np.array([b['extent'][k] for k in ('x','y','z')])
    if extent_order=='length_width_height':extent=extent[[1,0,2]]
    elif extent_order!='raw_xyz':raise ValueError('Unknown bounding box hypothesis')
    rotation=Rotation.from_euler('xyz',[b['rotation'][k] for k in ('x','y','z')],degrees=True).as_matrix()
    corners=np.array(list(itertools.product((-1.,1.),repeat=3)))*extent
    return center,rotation,extent,corners@rotation.T+center


def world_points(points,ego_rotation,ego_center,sensor_transform,interpretation='sensor_frame'):
    xyz=np.asarray(points)[:,:3]
    r,t=transform_matrix(sensor_transform)
    if interpretation=='sensor_frame':xyz=xyz@r.T+t
    elif interpretation!='already_base_frame':raise ValueError('Unknown LiDAR frame hypothesis')
    return xyz@ego_rotation.T+ego_center


def project(points,rotation,center,width,height,fov_degrees,fov_axis='horizontal'):
    if fov_axis not in ('horizontal','vertical'):raise ValueError('Explicit FOV axis required')
    span=width if fov_axis=='horizontal' else height
    focal=span/(2*np.tan(np.deg2rad(fov_degrees)/2))
    xyz=(np.asarray(points)-center)@rotation
    depth=xyz[:,2]
    uv=xyz[:,:2]/np.maximum(depth[:,None],1e-9)*focal+[width/2,height/2]
    return uv,depth
