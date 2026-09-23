"""Offline history/plot/contracts without selecting any real Stage-2 weights."""
import ast
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from inference import offline_vldrive as offline
from inference.vldrive_offline_config import ConfigError,load_config,require_checkpoint
from inference.vldrive_runtime import seed_runtime
from inference.vldrive_waypoint_visualization import render_vldrive_waypoints
from models.qformer_mefa import QFormerMEFAConfig,QFormerMEFAConnector
from models.vldrive import VLDriveTrainingConfig,VLDriveTrainingModel
from utils import vision_encoder_visualization as viz

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    # Null-checkpoint tests stay independent of the user's selected trained model.
    return replace(load_config(ROOT/'configs/vldrive_offline_inference.json'),vldrive_checkpoint=None)


def records(count=36):
    return [dict(global_index=i,episode_index=0,frame_id=i,timestamp_ns=i*100_000_000,route='route')
            for i in range(count)]


def test_missing_checkpoint_is_rejected_and_history_is_consecutive(config,monkeypatch):
    assert config.vldrive_checkpoint is None
    assert config.frame_sampling_interval==1 and config.sequence_length==11
    assert {'png','pdf'}<=set(config.formats)
    def forbidden(*args): raise AssertionError('Validation or checkpoint search ran before null check')
    monkeypatch.setattr(offline,'visual_contract',forbidden)
    with pytest.raises(ConfigError,match='VLDrive checkpoint is not configured'):
        offline.run(config)


@pytest.mark.parametrize('section,key,value',[
    ('history','sequence_length',10),('history','frame_sampling_interval',0),
    ('frames','stride',0),('runtime','seed',True),('output','formats',['jpg']),
    ('dataset','max_sensor_gap_ms',float('nan')),('frames','ids',[20,10]),
    (None,'instruction',' '),('dataset','format','carla_style')])
def test_strict_configuration_rejects_bad_values(tmp_path,section,key,value):
    document=json.loads((ROOT/'configs/vldrive_offline_inference.json').read_text())
    (document[section] if section else document)[key]=value
    path=tmp_path/'config.json'
    path.write_text(json.dumps(document))
    with pytest.raises(ConfigError): load_config(path)


def test_unknown_config_key_is_not_silently_ignored(tmp_path):
    document=json.loads((ROOT/'configs/vldrive_offline_inference.json').read_text())
    document['frame_stride']=10
    path=tmp_path/'config.json'
    path.write_text(json.dumps(document))
    with pytest.raises(ConfigError,match='unknown keys'): load_config(path)


def test_native_sensor_record_auto_and_explicit_detection(tmp_path):
    capture=tmp_path/'capture'
    episode=capture/'episode_000'
    for name in ('CAMERA_1','CAMERA_2','CAMERA_3','CAMERA_4','LIDAR_5','GPS_6','IMU_7',
                 'episode_000/native_associations'):
        (capture/name).mkdir(parents=True)
    (episode/'episode.json').write_text('{}')
    (episode/'native_associations/associations.json').write_text('[]')
    (capture/'sensor_contract.json').write_text('{}')
    for mode in ('auto','morai_native_sensor_record'):
        assert offline.detect_dataset_format(tmp_path,mode)==('morai_native_sensor_record',[episode])
    with pytest.raises(ConfigError,match='Explicit dataset format'):
        offline.detect_dataset_format(tmp_path,'carla_style')
    (capture/'LIDAR_5').rmdir()
    with pytest.raises(ConfigError,match='audited Sensor Record'):
        offline.detect_dataset_format(tmp_path)


def test_window_ids_are_consecutive_causal_and_separate_from_figure_stride(config):
    rows=records()
    chosen=offline.select_indices(replace(config,stride=10,limit=3),rows)
    assert chosen==[10,20,30]
    assert offline.window_indices(rows,10)==list(range(11))
    assert offline.window_indices(rows,20)==list(range(10,21))
    assert offline.window_indices(rows,9) is None
    for anchor in range(10,36):
        ids=offline.window_indices(rows,anchor)
        assert max(ids)==anchor and np.diff(ids).tolist()==[1]*10
    rows[17]['route']='new-episode'
    assert offline.window_indices(rows,20) is None
    rows=records()
    rows[17]['frame_id']=99
    assert offline.window_indices(rows,20) is None
    rows=records()
    rows[17]['timestamp_ns']+=70_000_000
    assert offline.window_indices(rows,20) is None


def test_invalid_selected_anchor_reports_missing_history(config):
    with pytest.raises(ConfigError,match='no complete causal history'):
        offline.select_indices(replace(config,frame_ids=(9,20)),records())


class RawFrames:
    def __init__(self,count=36):
        self.read=[]
        self.times=[i*100_000_000 for i in range(count)]
        self.identity='fixture'
    def __len__(self): return len(self.times)
    def inputs(self,index):
        self.read.append(index)
        return {'rgb_front':torch.tensor([float(index)])}
    def target(self,index):
        raise AssertionError('Future labels must not be read by the inference loop')


class FeatureEncoder(nn.Module):
    def forward(self,inputs):
        # Identifiable causal frame values, not a trained visual encoder.
        return inputs['rgb_front'].reshape(1,1,1).expand(1,106,256)


class LanguageReadout(nn.Module):
    def forward(self,texts,features,mask,predict_per_frame):
        assert texts==['Continue straight along the current road.'] and predict_per_frame
        return SimpleNamespace(predicted_waypoints=features.sum(2)[...,:10].reshape(-1,5,2))


def stage2_fixture():
    net=VLDriveTrainingModel.__new__(VLDriveTrainingModel)
    nn.Module.__init__(net)
    net.config=VLDriveTrainingConfig(precision='fp32')
    net.connector=QFormerMEFAConnector(QFormerMEFAConfig(visual_dim=256,num_visual_tokens=106,
        language_dim=16,memory_size=10,qformer_hidden_size=32,qformer_num_layers=1,
        qformer_num_attention_heads=4,qformer_intermediate_size=64,reconstruction_layers=1,reconstruction_heads=8))
    net.language_model=LanguageReadout()
    return net.eval()


def test_all_windows_inferred_with_eval_gumbel_no_future_inputs_and_no_reconstruction(config):
    model=stage2_fixture()
    dataset=RawFrames()
    observed=[]
    def inspect_inputs(module,args,kwargs):
        assert kwargs['selection_mode']=='gumbel' and kwargs['compute_reconstruction'] is False
        window=args[0]
        assert window.shape==(1,11,106,256)
        ids=window[0,:,0,0].tolist()
        assert ids==list(range(int(ids[-1])-10,int(ids[-1])+1))
        assert dataset.read[-1]==int(ids[-1])  # No reads beyond current anchor.
        observed.append(ids)
    model.connector.register_forward_pre_hook(inspect_inputs,with_kwargs=True)
    def forbidden(*args): raise AssertionError('Reconstruction executed')
    model.connector.reconstruction_transformer.register_forward_pre_hook(forbidden)
    cfg=replace(config,device='cpu')
    seed_runtime(607)
    output=list(offline.infer_sequence(cfg,[dataset],records(14),[10,13],FeatureEncoder(),model))
    assert [r['frame_id'] for r in output]==[10,11,12,13]
    assert dataset.read==list(range(14)) and len(observed)==4
    assert all(np.asarray(r['waypoints_raw']).shape==(5,2) for r in output)
    assert all(min(r['selected_tokens_history'])>=1 for r in output)
    assert all(not m.training for m in model.modules())


def test_publication_renderer_axes_horizons_and_png_pdf(tmp_path):
    waypoints=np.array([[1,-4],[2,-8],[3,-12],[4,-16],[5,-20]],dtype=float)
    forward,right=viz.forward_right(waypoints)
    np.testing.assert_array_equal(forward,[4,8,12,16,20])
    np.testing.assert_array_equal(right,[-1,-2,-3,-4,-5])
    figure=render_vldrive_waypoints(waypoints,instruction='Turn left.',frame_id=10,route='route',
        selected_tokens=35,camera_image=np.zeros((72,128,3)),actual_trajectory=waypoints)
    np.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(),np.r_[0,right])
    np.testing.assert_array_equal(figure.axes[0].lines[0].get_ydata(),np.r_[0,forward])
    assert {t.get_text() for t in figure.axes[0].texts}>={'0.5s','1.0s','1.5s','2.0s','2.5s'}
    paths=viz.save_figure(figure,tmp_path/'waypoints',('png','pdf'),100)
    assert paths[0].read_bytes().startswith(b'\x89PNG')
    assert paths[1].read_bytes().startswith(b'%PDF')


def test_synthetic_preview_cannot_be_mistaken_for_model_prediction(tmp_path):
    fig=render_vldrive_waypoints(np.zeros((5,2)),instruction='Continue.',frame_id=10,route='route',
                                  selected_tokens=None,example=True)
    assert 'synthetic' in fig.axes[0].get_title(loc='left')
    assert any('NOT VLDRIVE PREDICTIONS' in text.get_text() for text in fig.texts)
    viz.plt.close(fig)


def test_manifest_is_json_safe_and_contains_unselected_checkpoint(config,tmp_path):
    contract={'meta':{'checkpoint_sha256':'abc','encoder_contract':{'feature_shape':[106,256]}}}
    manifest=offline.base_manifest(config,contract,{'root':'fixture','format':'morai_native_sensor_record','number_of_frames':21})
    offline.write_manifest(tmp_path/'inference_manifest.json',manifest)
    stored=json.loads((tmp_path/'inference_manifest.json').read_text())
    assert stored['vldrive']['checkpoint'] is None and stored['vldrive']['loaded'] is False
    assert stored['history']['figure_stride_changes_history'] is False
    assert stored['prediction_source']=='VLDrive stage-2 trajectory[0,-1]'


def test_runner_records_all_predictions_but_renders_only_selected_anchors(config,tmp_path,monkeypatch):
    from inference import vldrive_runtime
    dataset=RawFrames(13)
    cfg=replace(config,device='cpu',vldrive_checkpoint=tmp_path/'user-selected-fixture.pt',
                output=tmp_path/'figures',stride=2,limit=2)
    # No real .pt is selected or loaded: model/digest stand-ins are confined to this test.
    cfg.vldrive_checkpoint.write_text('not a model; unit test fixture')
    contract={'meta':{'checkpoint_sha256':'fixture','encoder_contract':{}},
              'training':{'frame_sampling_interval':2},'description':{'config':{'input_mode':'cached'}}}
    monkeypatch.setattr(offline,'visual_contract',lambda _:contract)
    monkeypatch.setattr(offline,'build_dataset',lambda *args:([dataset],{'number_of_frames':13}))
    monkeypatch.setattr(offline,'load_offline_stage2_contract',lambda *args:contract)
    monkeypatch.setattr(vldrive_runtime,'load_stage2',lambda *args:(stage2_fixture(),{'file_digest':{'sha256':'fixture'}}))
    monkeypatch.setattr(vldrive_runtime,'load_encoder',lambda *args:FeatureEncoder())
    drawn=[]
    def render(config,record,result,dataset,folder):
        drawn.append(record['frame_id'])
        return [{'path':f'frame_{record["frame_id"]}.png','sha256':'fixture'}],None
    monkeypatch.setattr(offline,'render_frame',render)
    manifest=offline.run(cfg)
    rows=[json.loads(line) for line in (cfg.output/'waypoints.jsonl').read_text().splitlines()]
    assert [row['frame_id'] for row in rows]==[10,11,12]
    assert drawn==[10,12] and manifest['model_windows_inferred']==3
    assert [w['frame_id'] for w in manifest['windows']]==[10,12]
    assert manifest['status']=='completed' and manifest['vldrive']['loaded']
    assert not manifest['history']['matches_training']
    assert (cfg.output/'inference_manifest.json').is_file()


def test_offline_entry_never_imports_udp_or_transmits():
    for name in ('offline_vldrive.py','vldrive_offline_config.py','vldrive_waypoint_visualization.py'):
        tree=ast.parse((ROOT/'inference'/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                assert not any(word in (node.module or '') for word in ('morai_udp','live_frame','realtime'))
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                assert node.func.attr not in {'send','sendto','sendall','publish','socket'}
