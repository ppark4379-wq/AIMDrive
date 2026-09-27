"""Regression checks for training package boundaries and preserved setup policies."""
from pathlib import Path

import pytest
import torch
from torch import nn


def test_legacy_training_imports_share_canonical_modules():
    from training import training, vision_encoder_training, morai_transfer_training
    from training.joint import training as joint
    from training.vision_encoder.recipes import pretrain, transfer
    assert training is joint
    assert vision_encoder_training is pretrain
    assert morai_transfer_training is transfer
    assert joint.ROOT == pretrain.ROOT == transfer.ROOT == Path(__file__).resolve().parents[2]


def test_setup_reuses_existing_types_without_new_defaults():
    from training.connector import setup as connector
    from training.llm import setup as llm
    from training.joint import setup as joint
    from training.vision_encoder import setup as vision
    from models.connector.qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector
    from models.vldrive import VLDriveTrainingConfig, VLDriveTrainingModel
    from utils.config.vision_encoder_config import VisionEncoderTrainConfig
    assert connector.QFormerMEFAConfig is QFormerMEFAConfig
    assert connector.QFormerMEFAConnector is QFormerMEFAConnector
    assert llm.VLDriveTrainingConfig is joint.VLDriveTrainingConfig is VLDriveTrainingConfig
    assert joint.VLDriveTrainingModel is VLDriveTrainingModel
    assert vision.VisionEncoderTrainConfig is VisionEncoderTrainConfig


@pytest.mark.parametrize('scope', ['full', 'object_head'])
def test_transfer_freeze_policy_and_batchnorm_mode(scope):
    from training.vision_encoder.setup import configure_trainable_scope, train_mode
    model = nn.Module()
    model.rgb_backbone = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
    model.traffic_pred_head = nn.Linear(3, 2)
    model.stop_sign_head = nn.Linear(3, 1)
    configure_trainable_scope(model, scope)
    train_mode(model, scope)
    assert all(p.requires_grad for p in model.traffic_pred_head.parameters())
    assert not any(p.requires_grad for p in model.stop_sign_head.parameters())
    assert all(p.requires_grad == (scope == 'full') for p in model.rgb_backbone.parameters())
    assert not model.rgb_backbone[1].training
    assert model.traffic_pred_head.training


def test_new_implementation_files_are_in_checkpoint_source_inventory():
    from training.common.checkpoints import source_hashes
    root = Path(__file__).resolve().parents[2]
    sources = {name.replace('\\', '/') for name in source_hashes(root)}
    required = {
        'training/common/' + name + '.py'
        for name in ('batching', 'precision', 'reproducibility', 'optimizers', 'checkpoints', 'logging')
    }
    required.update('training/' + part + '/setup.py' for part in ('vision_encoder', 'connector', 'llm', 'joint'))
    assert required <= sources
    assert 'training/tests/test_training_layout.py' not in sources
