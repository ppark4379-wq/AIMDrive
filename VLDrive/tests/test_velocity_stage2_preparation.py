"""GPS-conditioned cache admission and partial-route navigation regressions."""
import hashlib
import math
from pathlib import Path

import pytest
import torch

from data.src.loaders.navigation_velocity_sensors import gps_velocities, VelocityNativeNavigationSensors
from data.src.loaders.navigation_sensor_record import NativeNavigationSensors
from data.src.loaders.morai_encoder_contract import sensor_record_encoder_contract
from data.src.loaders.morai_transfer_adapter import MoraiVisionAdapterConfig
from data.src.processing.build_instructions import verified_expert_path_span


def test_causal_gps_speed_preserves_missing_and_gapped_indices():
    fixes=[(0,0.,0.),(100_000_000,0.,.000001),None,
           (300_000_000,0.,.000003),(700_000_000,0.,.000007),
           (800_000_000,0.,.000008)]
    speeds,reasons=gps_velocities(fixes)
    assert reasons==['gps_no_previous_fix',None,'gps_missing_or_invalid_fix',
                     'gps_no_previous_fix','gps_invalid_interval',None]
    assert speeds[0] is None and speeds[2] is None and speeds[4] is None
    assert speeds[1]==pytest.approx(1.111949266,rel=1e-6)
    assert speeds[-1]==pytest.approx(speeds[1])


def test_stationary_gps_is_valid_zero_and_not_missing():
    speeds,reasons=gps_velocities([(0,37.,127.),(100_000_000,37.,127.)])
    assert speeds==[None,0.] and reasons==['gps_no_previous_fix',None]


def test_only_explicit_existing_episodes_are_excluded(tmp_path):
    from navigation.adapters import sensor_record_episode_paths
    for name in ['complete','incomplete']:
        folder=tmp_path/name;folder.mkdir();(folder/'episode.json').write_text('{}')
    assert len(sensor_record_episode_paths(tmp_path))==2
    assert sensor_record_episode_paths(tmp_path,['incomplete'])==[tmp_path/'complete']
    for exclusions in [['absent'],['../outside'],['incomplete','incomplete']]:
        with pytest.raises(ValueError):sensor_record_episode_paths(tmp_path,exclusions)


def test_gps_teleport_is_rejected():
    speeds,reasons=gps_velocities([(0,37.,127.),(100_000_000,38.,127.)])
    assert speeds[-1] is None and reasons[-1]=='gps_invalid_interval'


def test_only_explicit_unique_forward_subpaths_are_allowed():
    path=[[i,0] for i in range(10)]
    assert verified_expert_path_span(path,path)==(0,9)
    assert verified_expert_path_span(path[3:7],path,allow_subset=True)==(3,6)
    for bad in [path[3:7][::-1],path[3:7:2],[[3,0],[4,.01],[5,0]],[[3,0]]]:
        with pytest.raises(ValueError):
            verified_expert_path_span(bad,path,allow_subset=True)
    with pytest.raises(ValueError):
        verified_expert_path_span(path[3:7],path)
    with pytest.raises(ValueError):
        verified_expert_path_span([[0,0],[1,0]],[[0,0],[1,0],[0,0],[1,0]],allow_subset=True)


def test_velocity_reader_replaces_scalar_and_checks_actual_gps_bytes(tmp_path,monkeypatch):
    import data.src.loaders.navigation_velocity_sensors as module
    assets={};bundles=[];scenes=[]
    for i in range(3):
        name=f'gps_{i}.txt';raw=f'Latitude : 0\nLongitude : {i*.000001}\n'.encode()
        (tmp_path/name).write_bytes(raw)
        assets[name]=dict(size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        bundles.append(dict(assets=dict(lidar=f'lidar_{i}',gps=name),filename_simulation_time_hypothesis_ns=i*100_000_000))
        scenes.append(dict(lidar_file=f'lidar_{i}',native_filename_time_ns=i*100_000_000))
    def initialize(self,episode,recipe,namespace,max_gap):
        self.native=tmp_path;self.episode=tmp_path/'episode';self.identity='test/episode'
        self.scenes=scenes;self.times=[i*100_000_000 for i in range(3)];self.assets=assets
    monkeypatch.setattr(NativeNavigationSensors,'__init__',initialize)
    monkeypatch.setattr(VelocityNativeNavigationSensors,'geometry_inputs',lambda self,index:dict(velocity=torch.zeros(1)))
    monkeypatch.setattr(module,'read_json',lambda path: dict(bundles=bundles) if path.parent.name=='native_audit' else {})
    contract=sensor_record_encoder_contract([MoraiVisionAdapterConfig(use_previous_lidar_sweep=False)],'gps')
    data=VelocityNativeNavigationSensors(tmp_path,contract,'test')
    with pytest.raises(ValueError,match='Unavailable encoder velocity'): data.inputs(0)
    assert data.inputs(1)['velocity'].item()==pytest.approx(1.111949266,rel=1e-6)
    assert data.velocity_summary['ego_fallback'] is False
    (tmp_path/'gps_1.txt').write_text('Latitude : 2\nLongitude : 3\n')
    with pytest.raises(ValueError,match='Changed native asset'):
        VelocityNativeNavigationSensors(tmp_path,contract,'test')


def test_verified_asset_reader_rejects_escape_and_changed_bytes(tmp_path):
    from data.src.loaders.navigation_velocity_sensors import VerifiedNativeAssets
    folder=tmp_path/'native';folder.mkdir();(folder/'camera').mkdir()
    path=folder/'camera/frame';path.write_bytes(b'actual sensor bytes')
    reader=VerifiedNativeAssets(folder,{'camera/frame':dict(size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())})
    assert reader('camera/frame')==b'actual sensor bytes'
    with pytest.raises(ValueError,match='Unsafe'):reader('../outside')
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed native'):reader('camera/frame')
    if hasattr(__import__('os'),'O_NOFOLLOW'):
        outside=tmp_path/'outside';outside.write_bytes(b'actual sensor bytes')
        link=folder/'link';link.symlink_to(outside)
        with pytest.raises(OSError):reader('link')
