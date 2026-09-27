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

from utils.common.runtime import sha256, complete_visual_checkpoint_initialization
from .qformer_mefa import QFormerMEFAConfig, QFormerMEFAConnector
from .frozen_vision_encoder import MoraiVisualEncoder
from .language_model import VLDriveModel


def _verify_local_asset(directory, *, expected_kind):
    """Verify every file declared by a prepared local model asset."""
    directory = Path(directory)
    manifest_path = directory / 'asset_manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Missing prepared {} asset manifest: {}'.format(expected_kind, manifest_path))
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('kind') not in (None, expected_kind):
        raise ValueError('{} asset kind mismatch'.format(expected_kind))
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('{} asset manifest has no files'.format(expected_kind))
    for name, digest in files.items():
        path = (directory / name).resolve()
        if directory.resolve() not in path.parents or not path.is_file() or sha256(path) != digest:
            raise ValueError('Pretrained {} asset hash mismatch: {}'.format(expected_kind, name))
    return manifest


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
    qformer_checkpoint: str = ''
    reconstruction_layers: int = 4
    pruning_weight: float = 10.
    reconstruction_weight: float = 1.
    checkpoint_language: bool = True
    training_variant: str = 'paper_independent'
    reconstruction_reduction: str = 'sum'
    reconstruction_feature_space: str = 'encoder'
    detach_reconstruction_loss_mask: bool = False
    inference_selection_mode: str = 'argmax'
    retention_logit_weight: float = 0.
    selector_task_gradient_mode: str = 'legacy'

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
        if self.training_variant not in ('paper_independent', 'morai_repair_v1', 'paper_objective_st_probe', 'fixed_nc_debug', 'pruned_mean_debug'):
            raise ValueError('Unknown downstream training variant')
        if self.selector_task_gradient_mode not in ('legacy', 'retained_output_st'):
            raise ValueError('Unknown selector task-gradient mode')
        if (self.training_variant == 'paper_objective_st_probe') != (
                self.selector_task_gradient_mode == 'retained_output_st'):
            raise ValueError('retained_output_st requires explicit training_variant=paper_objective_st_probe')
        if self.reconstruction_reduction not in ('sum', 'mean', 'fixed_nc', 'pruned_mean'):
            raise ValueError('Unsupported reconstruction reduction')
        if (self.training_variant == 'fixed_nc_debug') != (self.reconstruction_reduction == 'fixed_nc'):
            raise ValueError('fixed_nc requires explicit training_variant=fixed_nc_debug')
        if (self.training_variant == 'pruned_mean_debug') != (self.reconstruction_reduction == 'pruned_mean'):
            raise ValueError('pruned_mean requires explicit training_variant=pruned_mean_debug')
        if self.training_variant in ('fixed_nc_debug', 'pruned_mean_debug') and (
                self.detach_reconstruction_loss_mask or self.reconstruction_feature_space != 'encoder'
                or self.inference_selection_mode != 'argmax' or self.retention_logit_weight != 0.
                or self.selector_task_gradient_mode != 'legacy'):
            raise ValueError('Reduction debug variants change reconstruction reduction only')
        if self.reconstruction_feature_space not in ('encoder', 'normalized'):
            raise ValueError('Unsupported reconstruction feature space')
        if self.inference_selection_mode not in ('argmax', 'topk'):
            raise ValueError('Unsupported inference token selection')
        if not 0. <= self.retention_logit_weight <= 100.:
            raise ValueError('Retention logit stability weight must be between 0 and 100')
        if self.training_variant in ('paper_independent', 'paper_objective_st_probe') and (
                self.reconstruction_reduction != 'sum' or self.detach_reconstruction_loss_mask
                or self.reconstruction_feature_space != 'encoder'
                or self.inference_selection_mode != 'argmax' or self.retention_logit_weight != 0.):
            raise ValueError('Repair choices require explicit training_variant=morai_repair_v1')


class VLDriveTrainingModel(nn.Module):
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
        self.language_manifest = _verify_local_asset(asset, expected_kind='tinyllama')
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
        self.qformer_manifest = None
        if config.qformer_checkpoint:
            self.qformer_manifest = _verify_local_asset(
                config.qformer_checkpoint, expected_kind='bert_qformer_backbone')
        self.connector = QFormerMEFAConnector(
            QFormerMEFAConfig(
                visual_dim=256, num_visual_tokens=106, language_dim=hidden,
                target_retention=config.target_retention, memory_size=config.memory_size,
                qformer_hidden_size=config.qformer_hidden, qformer_num_layers=config.qformer_layers,
                qformer_num_attention_heads=config.qformer_heads,
                qformer_intermediate_size=config.qformer_hidden * 4,
                reconstruction_layers=config.reconstruction_layers,
                reconstruction_formulation='eq8', reconstruction_reduction=config.reconstruction_reduction,
                reconstruction_feature_space=config.reconstruction_feature_space,
                detach_reconstruction_loss_mask=config.detach_reconstruction_loss_mask,
                inference_selection_mode=config.inference_selection_mode,
                retention_logit_weight=config.retention_logit_weight,
                selector_task_gradient_mode=config.selector_task_gradient_mode,
                lambda_pruning=config.pruning_weight, lambda_reconstruction=config.reconstruction_weight,
            ),
            qformer_local_pretrained_path=config.qformer_checkpoint or None,
        )
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

    def predict(self, inputs, *, compute_auxiliary=None, token_selection_mode=None):
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
        connected = self.connector(features.float(), selection_mode=token_selection_mode,
                                   compute_reconstruction=compute_auxiliary)
        language = self.language_model(texts, connected['language_tokens'],
            connected['selected_valid_mask'], predict_per_frame=True)
        batch, frames = features.shape[:2]
        selector = connected['selector']
        return dict(trajectory=language.predicted_waypoints.reshape(batch, frames, 5, 2),
                    pruning_loss=connected['pruning_loss'], reconstruction_loss=connected['reconstruction_loss'],
                    cycle_loss=connected['cycle_loss'], selected_count=connected['selected_count_per_frame'],
                    retention_logit_loss=connected['retention_logit_loss'],
                    sampled_retention=selector['retention_per_frame'].detach(),
                    soft_retention=selector['soft_retention_per_frame'].detach(),
                    saturated_keep_fraction=(selector['probabilities'][..., 1] > .999).float().mean().detach())

    @staticmethod
    def loss(output, targets):
        if set(targets) not in ({'future_trajectory'}, {'future_trajectory', 'frame_loss_mask'}):
            raise ValueError('Trajectory supervision accepts only future_trajectory and optional frame_loss_mask')
        expected = targets['future_trajectory']
        if expected.shape != output['trajectory'].shape or not bool(torch.isfinite(expected).all()):
            raise ValueError('Waypoint target shape or values are invalid')
        # Eq. 10: L1 trajectory + Eq. 11 cycle consistency, over every real frame.
        element_loss = F.l1_loss(output['trajectory'].float(), expected.float(), reduction='none')
        if 'frame_loss_mask' in targets:
            mask = targets['frame_loss_mask']
            if (mask.dtype != torch.bool or mask.shape != expected.shape[:2]
                    or not bool(mask[:, -1].all())
                    or bool((mask[:, :-1] & ~mask[:, 1:]).any())):
                raise ValueError('frame_loss_mask must select a nonempty contiguous current-instruction suffix')
            waypoint = (element_loss * mask[..., None, None]).sum() / (mask.sum() * 10)
        else:
            waypoint = element_loss.mean()
        result = dict(loss=waypoint + output['cycle_loss'], waypoint_loss=waypoint,
                    pruning_loss=output['pruning_loss'], reconstruction_loss=output['reconstruction_loss'])
        for key in ('sampled_retention', 'soft_retention', 'saturated_keep_fraction'):
            if key in output:
                result[key] = output[key].float().mean().detach()
        if 'retention_logit_loss' in output:
            result['retention_logit_loss'] = output['retention_logit_loss'].detach()
        return result

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
            language_asset=self.language_manifest, qformer_asset=self.qformer_manifest, frozen_encoder=True,
            visual_encoder=self.visual_encoder.describe() if self.visual_encoder else 'cached frozen features',
            training_policy='all used TinyLlama decoder parameters including text embeddings',
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            inference_policy=('fixed budget top-k; ranking learned, token count enforced'
                if self.config.inference_selection_mode == 'topk'
                else 'per-token argmax with at least one token per frame'),
            selector_task_gradient=('experimental retained-output ST bypass; original forward and auxiliary gradients; '
                'not author-source-identical, not yet validated on MORAI'
                if self.config.selector_task_gradient_mode == 'retained_output_st'
                else 'original hard-gather query gradient path'),
            waypoint_readout='last valid visual token per frame', frame_padding_supported=False)

    @classmethod
    def from_config(cls, cfg):
        names = VLDriveTrainingConfig.__dataclass_fields__
        return cls(VLDriveTrainingConfig(**{k: cfg[k] for k in names if k in cfg}),
                   device=cfg.get('device', 'cuda'))
