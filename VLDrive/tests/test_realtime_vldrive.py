"""Runtime contracts: real history, eval Gumbel, receive-only, shared recipe."""
import ast
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image

from inference.sensors.live_frame import CAMERA_KEYS, LiveFrameSource
from inference.runtime.vldrive_runtime import TokenHistory, encode_frame, predict_window, seed_runtime
from models.connector.ccdp import CCDPTokenSparsifier
from models.connector.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector
from models.vldrive import VLDriveTrainingConfig, VLDriveTrainingModel


def recipe():
    return dict(schema='morai_encoder_recipe_v1', feature_shape=[106,256],
        token_order='bev100_light1_waypoint5', velocity_input='constant_zero',
        target_point_input='constant_zero', pillar_decoration='legacy_lmdrive',
        adapter=dict(preprocessing='lmdrive_compat',max_lidar_points=40000,remove_ego_box_m=1.2,
            use_previous_lidar_sweep=False,lidar_axis_policy='morai_flu_to_lmdrive_fru',
            lidar_translation_xyz=[1.,0.,1.48],lidar_rotation_xyzw=[0.,0.,0.,1.],
            lidar_intensity_policy='reflectance_uint8_unit'))


class Readout(nn.Module):
    def forward(self, texts, features, mask, predict_per_frame):
        assert texts == ['Continue straight.'] and predict_per_frame
        return SimpleNamespace(predicted_waypoints=features.sum(2)[..., :10].reshape(-1,5,2))


@pytest.fixture
def model():
    # Real stage-2 predict/CCDP/MEFA flow; tiny QFormer and readout avoid assets.
    net = VLDriveTrainingModel.__new__(VLDriveTrainingModel)
    nn.Module.__init__(net)
    net.config = VLDriveTrainingConfig(precision='fp32')
    net.connector = QFormerMEFAConnector(QFormerMEFAConfig(
        visual_dim=256,num_visual_tokens=106,language_dim=16,memory_size=10,
        qformer_hidden_size=32,qformer_num_layers=1,qformer_num_attention_heads=4,
        qformer_intermediate_size=64,reconstruction_layers=1,reconstruction_heads=8))
    net.language_model = Readout()
    return net.eval()


def test_history_uses_eleven_distinct_timed_frames_without_padding():
    history = TokenHistory()
    for i in range(11):
        assert history.append(torch.full((106,256), float(i)), stamp=i*.2, ids={'lidar':i,'camera':i})
        window, rows = history.snapshot()
        assert len(rows) == i+1
        if i < 10:
            assert window is None
    assert window.shape == (1,11,106,256)
    assert window[0,:,0,0].tolist() == list(range(11))
    assert not history.append(torch.zeros(106,256),stamp=2.2,ids={'lidar':10,'camera':11})
    assert not history.append(torch.zeros(106,256),stamp=2.05,ids={'lidar':11,'camera':11})
    assert history.append(torch.ones(106,256),stamp=3.,ids={'lidar':20,'camera':20})
    assert history.snapshot()[0] is None and history.resets == 1


def test_history_metadata_poll_does_not_materialize_gpu_work(monkeypatch):
    history=TokenHistory()
    for i in range(11):
        history.append(torch.full((106,256),float(i)),stamp=i*.2,ids={'lidar':i})
    original=torch.stack
    def forbidden(*args,**kwargs):
        raise AssertionError('Metadata-only polling stacked the feature tensors')
    monkeypatch.setattr(torch,'stack',forbidden)
    features,rows=history.snapshot(materialize=False)
    assert features is None and len(rows)==11
    monkeypatch.setattr(torch,'stack',original)
    window=history.materialize(rows)
    assert window[0,:,0,0].tolist()==list(range(11))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA stream handoff requires GPU')
def test_history_producer_stream_to_consumer_stream_preserves_values():
    history=TokenHistory()
    producer=torch.cuda.Stream()
    consumer=torch.cuda.Stream()
    with torch.cuda.stream(producer):
        for i in range(11):
            history.append(torch.full((106,256),float(i),device='cuda'),stamp=i*.2,ids={'lidar':i})
    with torch.cuda.stream(consumer):
        window,_=history.snapshot()
        # Clear producer references while the consumer can still be using them.
        history.clear()
        actual=window[0,:,0,0].cpu()
    assert actual.tolist()==list(range(11))


def test_runtime_eval_gumbel_shapes_rng_and_reconstruction_disabled(model):
    features = torch.randn(1,11,106,256)
    masks = []
    handle = model.connector.ccdp.register_forward_hook(lambda m,a,o: masks.append(o['mask'].clone()))
    def forbidden(*args):
        raise AssertionError('Inference executed reconstruction')
    rec = model.connector.reconstruction_transformer.register_forward_pre_hook(forbidden)
    seed_runtime(607)
    first = predict_window(model, features, 'Continue straight.')
    second = predict_window(model, features, 'Continue straight.')
    assert not torch.equal(masks[0],masks[1])  # RNG advances; no per-frame reset.
    seed_runtime(607)
    repeat = predict_window(model, features, 'Continue straight.')
    assert torch.equal(masks[0],masks[2])
    assert first['waypoints_raw'] == repeat['waypoints_raw']
    assert np.asarray(first['waypoints_raw']).shape == (5,2)
    assert min(first['selected_tokens_history']) >= 1
    assert all(not module.training for module in model.modules())
    raw = np.asarray(first['waypoints_raw'])
    readable = np.asarray(first['waypoints_lateral_left_longitudinal_forward'])
    np.testing.assert_array_equal(readable[:,0],raw[:,0])
    np.testing.assert_array_equal(readable[:,1],-raw[:,1])
    handle.remove()
    rec.remove()


def test_default_predict_remains_argmax_and_future_inputs_rejected(model):
    features = torch.randn(1,11,106,256)
    inputs = dict(visual_feature=features,instruction=['Continue straight.'])
    first = model.predict(inputs,compute_auxiliary=False)
    torch.manual_seed(123)
    second = model.predict(inputs,compute_auxiliary=False)
    assert first['trajectory'].shape == (1,11,5,2)
    torch.testing.assert_close(first['trajectory'],second['trajectory'],atol=0,rtol=0)
    with pytest.raises(ValueError,match='allowlist'):
        model.predict(dict(inputs,future_trajectory=torch.zeros(1,11,5,2)))
    with pytest.raises(ValueError,match='nonempty'):
        predict_window(model,features,' ')
    model.train()
    with pytest.raises(ValueError,match='eval'):
        predict_window(model,features,'Continue straight.')


def test_gumbel_empty_frame_rescue_in_eval():
    selector=CCDPTokenSparsifier().eval()
    with torch.no_grad():
        for parameter in selector.parameters():
            parameter.zero_()
        selector.score_mlp[-1].bias.copy_(torch.tensor([1000.,-1000.]))
    out=selector(torch.randn(1,11,106,256),selection_mode='gumbel')
    assert torch.equal(out['mask'].sum(-1),torch.ones(1,11))


def test_encoder_shape_and_float16_cache_boundary():
    class Encoder(nn.Module):
        def forward(self, inputs):
            return torch.full((1,106,256),.123456)
    feature, ms = encode_frame(Encoder().eval(), {'rgb_front':torch.zeros(1)})
    assert feature.shape == (106,256) and feature.dtype == torch.float32 and ms >= 0
    assert feature[0,0] == torch.tensor(.123456).half().float()
    class BadEncoder(nn.Module):
        def forward(self, inputs):
            return torch.zeros(1,5,2)
    with pytest.raises(ValueError,match='106,256'):
        encode_frame(BadEncoder(),{'rgb_front':torch.zeros(1)})


def test_shared_adapter_applies_cache_recipe_once_without_route_or_velocity():
    source=LiveFrameSource({},open_receivers=False,encoder_recipe=recipe())
    images={key:np.zeros((600,800,3),np.uint8) for key in CAMERA_KEYS}
    points=np.array([[5.,2.,0.,255.],[7.,3.,-1.,127.]],dtype=np.float32)
    original=points.copy()
    inputs=source.prepare_encoder_frame(images,points,rng_seed=5)
    reference=source.adapter.camera_tensors(**{key[4:]+'_rgb':value for key,value in images.items()})
    for key,value in reference.items():
        torch.testing.assert_close(inputs[key][0],value,rtol=0,atol=0)
    expected,count=source.adapter.lidar_to_raw_features(points,rng_seed=5)
    np.testing.assert_array_equal(inputs['lidar'][0].numpy(),expected)
    np.testing.assert_array_equal(points,original)
    assert inputs['num_points'].item()==count and inputs['velocity'].item()==0
    assert 'target_point' not in inputs
    assert not list(source._receivers())


def test_live_build_decodes_packets_through_shared_recipe_and_records_identity():
    source=LiveFrameSource({},open_receivers=False,encoder_recipe=recipe())
    stream=io.BytesIO()
    Image.fromarray(np.full((600,800,3),80,dtype=np.uint8)).save(stream,format='JPEG')
    class Receiver:
        def __init__(self,payload): self.payload=payload
        def snapshot(self): return self.payload,.01,17,123.
    source.cameras={key:Receiver(stream.getvalue()) for key in CAMERA_KEYS}
    source.lidar=Receiver(np.array([[5.,2.,0.,255.],[7.,3.,-1.,127.]],dtype=np.float32))
    source.status=Receiver(dict(speed=22.,yaw_degrees=0.,sim_time=42.))
    inputs,meta=source.build('cpu',max_age=.3)
    assert inputs['rgb_front'].shape==(1,3,224,224)
    assert inputs['rgb_center'].shape==(1,3,128,128)
    assert inputs['velocity'].item()==0 and 'target_point' not in inputs
    assert set(meta['sample_ids'])==set(CAMERA_KEYS)|{'lidar','vehicle_status'}
    assert meta['sim_time']==42. and meta['speed']==22.
    source.cameras['rgb_front'].payload=None
    inputs,meta=source.build('cpu',max_age=.3)
    assert inputs is None and 'rgb_front' in meta['missing']


def test_receiver_sockets_only_bind_sensor_ports(monkeypatch):
    from inference.sensors import morai_udp
    sockets=[]
    class Socket:
        def __init__(self,*args): self.bound=None;sockets.append(self)
        def setsockopt(self,*args): pass
        def bind(self,address): self.bound=address
        def settimeout(self,value): pass
        def close(self): pass
        # Deliberately no transmission/connection methods.
    class Thread:
        def __init__(self,**kwargs): pass
        def start(self): pass
        def join(self,timeout=None): pass
    monkeypatch.setattr(morai_udp.socket,'socket',Socket)
    monkeypatch.setattr(morai_udp.threading,'Thread',Thread)
    ports=dict(zip([*CAMERA_KEYS,'lidar','vehicle_status'],range(10001,10007)))
    source=LiveFrameSource(ports,encoder_recipe=recipe())
    assert {socket.bound for socket in sockets}=={('0.0.0.0',p) for p in ports.values()}
    assert source.lidar.snapshot()==(None,float('inf'),0,None)
    source.close()


def test_inference_modules_have_no_transmit_calls():
    root=Path(__file__).resolve().parents[1]/'inference'
    for name in ('realtime/realtime_vldrive.py','runtime/vldrive_runtime.py','diagnostics/vldrive_smoke.py','sensors/live_frame.py','sensors/morai_udp.py'):
        tree=ast.parse((root/name).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                assert node.func.attr not in {'send','sendto','sendall','publish','connect'}, (name,node.lineno)
