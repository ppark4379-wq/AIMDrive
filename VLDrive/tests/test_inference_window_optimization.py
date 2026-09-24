"""Inference speed changes must preserve stochastic selection and real history."""
import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from inference.fast_window import shared_ddia
from inference.timed_history import TimedTokenHistory
from inference.vldrive_runtime import predict_window
from models.ddia_attention import install_ddia_attention
from models.language_model import VLDriveModel
from models.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector
from models.vldrive import VLDriveTrainingConfig, VLDriveTrainingModel


def test_shared_geometry_preserves_gumbel_mask_and_latest_waypoints():
    torch.manual_seed(17)
    model = VLDriveTrainingModel.__new__(VLDriveTrainingModel)
    nn.Module.__init__(model)
    model.config = VLDriveTrainingConfig(precision='fp32')
    model.connector = QFormerMEFAConnector(QFormerMEFAConfig(
        visual_dim=256, num_visual_tokens=106, language_dim=16, memory_size=10,
        qformer_hidden_size=32, qformer_num_layers=1, qformer_num_attention_heads=4,
        qformer_intermediate_size=64, reconstruction_layers=1, reconstruction_heads=8))
    language = VLDriveModel.__new__(VLDriveModel)
    nn.Module.__init__(language)
    language.tinyllama = install_ddia_attention(LlamaForCausalLM(LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=32)))
    language.tokenizer = lambda *a, **k: dict(input_ids=torch.tensor([[1, 5, 7, 2]]),
                                            attention_mask=torch.ones(1, 4, dtype=torch.long))
    language.visual_to_language_projection = nn.Linear(16, 32)
    language.waypoints_predictor = nn.Linear(32, 10)
    language.end_predictor = nn.Linear(32, 2)
    language.max_instruction_length = 128
    language.debug_vldrive = False
    model.language_model = language
    model.eval()
    masks = []
    handle = model.connector.ccdp.register_forward_hook(lambda m, a, out: masks.append(out['mask'].clone()))
    keys = list(model.state_dict())
    features = torch.randn(1, 11, 106, 256)
    for seed in (3, 7):
        torch.manual_seed(seed)
        reference = predict_window(model, features, 'Continue straight.')
        torch.manual_seed(seed)
        optimized = predict_window(model, features, 'Continue straight.', optimized=True)
        assert torch.equal(masks[-2], masks[-1])
        torch.testing.assert_close(torch.tensor(reference['waypoints_raw']),
                                   torch.tensor(optimized['waypoints_raw']), atol=0, rtol=0)
        assert optimized['selected_tokens_history'] == reference['selected_tokens_history']
    assert list(model.state_dict()) == keys
    assert all('forward' not in layer.self_attn.__dict__ for layer in language.tinyllama.model.layers)
    with pytest.raises(RuntimeError):
        with shared_ddia(language):
            raise RuntimeError('interrupted')
    assert all('forward' not in layer.self_attn.__dict__ for layer in language.tinyllama.model.layers)
    handle.remove()


def append(history, stamp, identity):
    return history.append(torch.full((106, 256), float(identity)), stamp=stamp, ids={'frame': identity})


def test_dense_buffer_outputs_ten_hz_but_preserves_two_second_history():
    history = TimedTokenHistory(capture_hz=10.)
    for i in range(41):
        assert append(history, i * .1, i)
        features, rows = history.snapshot()
        if i < 20:
            assert features is None
        else:
            assert features[0, :, 0, 0].tolist() == list(range(i-20, i+1, 2))
            assert rows[-1]['stamp']-rows[0]['stamp'] == pytest.approx(2.)
    assert history.resets == 0
    assert len(history.frames) <= history.frames.maxlen


def test_short_gap_does_not_erase_reusable_history_or_fabricate_samples():
    history = TimedTokenHistory(capture_hz=10.)
    for i in range(11):
        append(history, i * .2, i)
    assert append(history, 2.27, 11)
    assert history.resets == 0
    assert history.snapshot()[0] is None  # No sufficiently close real frame.
    assert append(history, 2.41, 12)
    features, rows = history.snapshot()
    assert features is not None and history.resets == 0
    assert len({row['ids']['frame'] for row in rows}) == 11
    assert all(abs(row['stamp']-(2.41-(10-i)*.2)) <= .06+1e-8 for i,row in enumerate(rows))
    assert not append(history, 2.51, 12)  # repeated sensor identity
    assert append(history, 3.6, 20)
    assert history.resets == 1 and history.snapshot()[0] is None


def test_dense_history_reset_and_expiry_cannot_reuse_previous_generation():
    history = TimedTokenHistory()
    for i in range(21): append(history, i * .1, i)
    _, rows = history.snapshot(materialize=False)
    history.clear()
    assert history.resets != rows[-1]['generation']
    append(history, 3., 30)
    history.expire(now=3.3)
    assert len(history.frames) == 1
    history.expire(now=4.1)
    assert len(history.frames) == 0


def test_optimized_configuration_rejects_invalid_cadences():
    from inference.realtime_vldrive import validate_config
    from inference.vldrive_runtime import read_json, ROOT
    cfg = read_json(ROOT/'configs/vldrive_realtime.json')
    cfg['runtime'].update(resample_history=True, capture_hz=10., optimized_window=True,
                          history_reset_gap_s=1., prediction_hz=10.)
    validate_config(cfg)
    cfg['runtime']['resample_history'] = False
    with pytest.raises(ValueError, match='resample_history'): validate_config(cfg)


def test_no_control_overrides_enabled_config_and_flags_are_exclusive(monkeypatch):
    import argparse
    import ast
    import sys
    from pathlib import Path
    from inference.vldrive_runtime import ROOT
    source = ROOT/'inference/realtime_vldrive.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    end = next(i for i,n in enumerate(main.body) if isinstance(n, ast.If)
               and ast.unparse(n.test) == 'args.instruction is not None')
    code = compile(ast.Module(body=main.body[:end], type_ignores=[]), str(source), 'exec')
    for flags, expected in (([], True), (['--control'], True), (['--no-control'], False)):
        monkeypatch.setattr(sys, 'argv', ['test', *flags])
        context = dict(argparse=argparse, Path=Path, __doc__='', local_path=lambda p: Path(p),
                       read_json=lambda p: dict(control=dict(enabled=True, config='control.json')))
        exec(code, context)
        assert context['cfg']['control']['enabled'] is expected
    monkeypatch.setattr(sys, 'argv', ['test', '--control', '--no-control'])
    with pytest.raises(SystemExit) as error:
        exec(code, context)
    assert error.value.code == 2
