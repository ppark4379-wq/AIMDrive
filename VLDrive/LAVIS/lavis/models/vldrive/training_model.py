"""Frozen LMDrive -> dynamic CCDP/MEFA -> TinyLlama/DDIA -> frame-wise L1.

Independent Eq. (1)-(13) implementation. Sensor preprocessing, checkpoint,
Q-Former dimensions and last-valid-token readout are documented choices.
MORAI teacher labels enter loss() only, never predict().
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.common.morai_runtime import sha256, complete_visual_checkpoint_initialization
from lavis.models.base_model import BaseModel
from .qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector
from .visual_encoder import MoraiVisualEncoder
from .vldrive_model import VLDriveModel


@dataclass
class VLDriveTrainingConfig:
    input_mode: str = 'cached'
    visual_checkpoint: str = ''
    visual_sha256: str = ''
    language_checkpoint: str = ''
    precision: str = 'bf16'
    max_instruction_tokens: int = 128
    memory_size: int = 10
    target_retention: float = .3
    qformer_hidden: int = 256
    qformer_layers: int = 2
    qformer_heads: int = 8
    reconstruction_layers: int = 4
    pruning_weight: float = 10.
    reconstruction_weight: float = 1.
    checkpoint_language: bool = True

    def validate(self):
        if self.input_mode not in ('raw', 'cached'):
            raise ValueError('input_mode must be raw or cached frozen features')
        if self.precision not in ('bf16', 'fp32'):
            raise ValueError('Training uses FP32 AdamW masters and bf16/fp32 computation')
        if self.max_instruction_tokens < 1 or self.memory_size < 1:
            raise ValueError('Instruction and memory limits must be positive')
        if self.reconstruction_layers != 4 or self.target_retention != .3:
            raise ValueError('Paper path requires reconstruction_layers=4 and R=0.3')
        if self.pruning_weight != 10. or self.reconstruction_weight != 1.:
            raise ValueError('Paper path requires lambda_pruning=10 and lambda_reconstruction=1')


@registry.register_model('vldrive_training')
class VLDriveTrainingModel(BaseModel):
    """Complete training graph, also exposing a strict inference-only input API."""

    PRETRAINED_MODEL_CONFIG_DICT = {}
    checkpoint_schema = 'vldrive-training-v1'

    def __init__(self, config, *, device='cuda'):
        super().__init__()
        config.validate()
        self.config = config
        self.visual_encoder = None
        if config.input_mode == 'raw':
            with complete_visual_checkpoint_initialization():
                self.visual_encoder = MoraiVisualEncoder(config.visual_checkpoint, policy='frozen',
                    device=device, expected_sha256=config.visual_sha256)
        asset = Path(config.language_checkpoint)
        self.language_manifest = json.loads((asset / 'asset_manifest.json').read_text())
        for name, digest in self.language_manifest['files'].items():
            if sha256(asset / name) != digest:
                raise ValueError('Pretrained language asset hash mismatch: ' + name)
        # Inspect the actual local checkpoint width; the connector outputs it directly.
        hidden = json.loads((asset / 'config.json').read_text())['hidden_size']
        self.language_model = VLDriveModel(str(asset), local_files_only=True,
            freeze_tinyllama=False, use_ddia=True, visual_feature_dim=hidden,
            use_visual_projection=False, max_instruction_length=config.max_instruction_tokens,
            precision=config.precision)
        # No vocabulary or instruction-completion supervision exists in Eq. 10.
        # Keep legacy modules for API compatibility, but exclude unused heads.
        self.language_model.tinyllama.lm_head.requires_grad_(False)
        self.language_model.end_predictor.requires_grad_(False)
        # The runner keeps FP32 AdamW master weights on CPU when the decoder
        # computes in BF16. Small updates accumulate in FP32 before casting.
        if config.checkpoint_language:
            self.language_model.tinyllama.model.gradient_checkpointing_enable()
        self.connector = QFormerMEFAConnector(QFormerMEFAConfig(
            visual_dim=256, num_visual_tokens=106, language_dim=hidden,
            target_retention=config.target_retention, memory_size=config.memory_size,
            qformer_hidden_size=config.qformer_hidden, qformer_num_layers=config.qformer_layers,
            qformer_num_attention_heads=config.qformer_heads,
            qformer_intermediate_size=config.qformer_hidden * 4,
            reconstruction_layers=config.reconstruction_layers,
            reconstruction_formulation='eq8', reconstruction_reduction='sum',
            lambda_pruning=config.pruning_weight, lambda_reconstruction=config.reconstruction_weight))
        for name, parameter in self.connector.named_parameters():
            if name.startswith(('learnable_query_tokens', 'qformer.bert.embeddings.word_embeddings.',
                                'qformer.bert.embeddings.position_embeddings.',
                                'qformer.bert.embeddings.token_type_embeddings.', 'qformer.bert.pooler.')):
                parameter.requires_grad_(False)
        self.to(device)

    def encode_visual(self, sensors):
        # Encode one real frame at a time to bound frozen-encoder peak memory.
        return torch.cat([self.visual_encoder({k: v[:, t:t+1] for k, v in sensors.items()})
                          for t in range(sensors['rgb_front'].shape[1])], dim=1)

    def predict(self, inputs, *, compute_auxiliary=None):
        feature_key = 'sensors' if self.config.input_mode == 'raw' else 'visual_feature'
        if set(inputs) != {feature_key, 'instruction'}:
            raise ValueError('VLDrive input allowlist permits only visual inputs and instruction')
        texts = inputs['instruction']
        if (not isinstance(texts, (list, tuple)) or not texts
                or any(not isinstance(t, str) or not t.strip() for t in texts)):
            raise ValueError('Paper training requires actual nonempty navigation instructions')
        features = self.encode_visual(inputs[feature_key]) if feature_key == 'sensors' else inputs[feature_key]
        if not bool(torch.isfinite(features).all()):
            raise ValueError('Nonfinite visual features')
        if features.requires_grad:
            raise ValueError('The paper path uses a frozen encoder or detached frozen feature cache')
        connected = self.connector(features.float(), compute_reconstruction=compute_auxiliary)
        language = self.language_model(texts, connected['language_tokens'],
            connected['selected_valid_mask'], predict_per_frame=True)
        batch, frames = features.shape[:2]
        return dict(trajectory=language.predicted_waypoints.reshape(batch, frames, 5, 2),
                    pruning_loss=connected['pruning_loss'], reconstruction_loss=connected['reconstruction_loss'],
                    cycle_loss=connected['cycle_loss'], selected_count=connected['selected_count_per_frame'])

    @staticmethod
    def loss(output, targets):
        if set(targets) != {'future_trajectory'}:
            raise ValueError('Paper supervision expects only future_trajectory[B,T,5,2]')
        expected = targets['future_trajectory']
        if expected.shape != output['trajectory'].shape or not bool(torch.isfinite(expected).all()):
            raise ValueError('Waypoint target shape or values are invalid')
        # Eq. 10: L1 trajectory + Eq. 11 cycle consistency, over every real frame.
        waypoint = F.l1_loss(output['trajectory'].float(), expected.float())
        return dict(loss=waypoint + output['cycle_loss'], waypoint_loss=waypoint,
                    pruning_loss=output['pruning_loss'], reconstruction_loss=output['reconstruction_loss'])

    def forward(self, samples):
        """LAVIS task-compatible sample -> scalar loss dictionary."""
        output = self.predict(samples['model_inputs'])
        return self.loss(output, samples['training_targets'])

    def gradient_groups(self):
        groups = dict(selector='connector.ccdp.', mefa='connector.temporal_encoder.',
            qformer='connector.qformer.', connector_projection='connector.qformer_to_language.',
            reconstruction='connector.reconstruction_transformer.',
            waypoint='language_model.waypoints_predictor.',
            text_embeddings='language_model.tinyllama.model.embed_tokens.')
        for index in (0, len(self.language_model.tinyllama.model.layers) - 1):
            for projection in ('q_proj', 'k_proj', 'v_proj'):
                groups['lm_{}_{}'.format(index, projection)] = (
                    'language_model.tinyllama.model.layers.{}.self_attn.{}.'.format(index, projection))
        return groups

    def describe(self):
        return dict(config=asdict(self.config), connector=self.connector.config.to_dict(),
            language_asset=self.language_manifest, frozen_encoder=True,
            visual_encoder=self.visual_encoder.describe() if self.visual_encoder else 'cached frozen features',
            training_policy='all used TinyLlama decoder parameters including text embeddings',
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            inference_policy='per-token argmax with at least one token per frame',
            waypoint_readout='last valid visual token per frame', frame_padding_supported=False)

    @classmethod
    def from_config(cls, cfg):
        names = VLDriveTrainingConfig.__dataclass_fields__
        return cls(VLDriveTrainingConfig(**{k: cfg[k] for k in names if k in cfg}),
                   device=cfg.get('device', 'cuda'))
