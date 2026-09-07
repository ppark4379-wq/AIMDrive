"""Trainable MORAI visual encoder -> CCDP/MEFA -> pretrained LM/DDIA -> XY/speed.

Bootstrapping goal_backends must precede import. The small Q-Former and metric
decoder are explicit experiment choices. The LM is loaded from real pretrained
weights. No semantic prompt is fabricated when annotations are unavailable.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from lavis.models.vldrive.ddia_attention import (
    DDIALlamaAttention, build_ddia_attention_mask, set_ddia_modality_ids,
)
from lavis.models.vldrive.multimodal_sequence import build_multimodal_sequence
from lavis.models.vldrive.tinyllama_backbone import load_tinyllama
from .goal_backends import sha256, complete_visual_checkpoint_initialization, deterministic_pointpillar_reductions
from .goal_selection import FixedBudgetCCDP
from .metric_trajectory import MetricTrajectoryDecoder
from .route_context_encoder import FlattenRouteContextEncoder
from .stage6f_visual_encoder import Stage6FVisualEncoder, FUSION_PREFIXES
from .vldrive_contract.paper_qformer_mefa import PaperQFormerMEFAConfig, PaperQFormerMEFAConnector


class EmptySafeDDIAAttention(DDIALlamaAttention):
    """Keep the team's logits; remove fallback self-attention on empty-key rows.

Numerical softmax fallback is allowed internally, but its output and reported
probabilities must be zero for EVERY query with no allowed keys, not just pads.
Residual connections in the LLaMA layer remain intact.
"""
    def forward(self, *args, **kwargs):
        output, probabilities, cache = super().forward(*args, **kwargs)
        if self.use_ddia:
            valid = build_ddia_attention_mask(self.modality_ids).any(-1)
            output = output * valid.unsqueeze(-1)
            if probabilities is not None:
                probabilities = probabilities * valid[:, None, :, None]
        return output, probabilities, cache


class GoalVisualEncoder(Stage6FVisualEncoder):
    """Explicit feature-backbone adaptation, with small-batch BN held in eval."""
    def __init__(self, path, *, policy, **kwargs):
        super().__init__(path, policy='fusion' if policy == 'all_features' else policy, **kwargs)
        if policy == 'all_features':
            prefixes = FUSION_PREFIXES + ('rgb_backbone.', 'rgb_patch_embed.proj.', 'lidar_backbone.')
            for name, parameter in self.encoder.named_parameters():
                parameter.requires_grad_(name.startswith(prefixes))
            self.policy = policy

    def forward(self, sensors):
        with deterministic_pointpillar_reductions():
            return super().forward(sensors)

    def describe(self):
        report = super().describe()
        report.pop('frozen_backbone_eval', None)
        report.update(backbone_normalization='eval, running statistics fixed',
                      backbone_weights_trainable=self.policy == 'all_features',
                      pointpillar_reduction='sorted deterministic segment_csr, mean/max')
        return report


@dataclass
class GoalStackConfig:
    input_mode: str = 'raw'
    encoder_policy: str = 'fusion'
    language_mode: str = 'ddia'  # ddia / causal / bypass
    language_policy: str = 'last_layer'  # frozen / last_layer / full
    ccdp: bool = True
    selection_policy: str = 'fixed_budget'  # fixed_budget / reference_bernoulli
    mefa: bool = True
    memory_size: int = 10
    query_count: int = 32
    qformer_hidden: int = 256
    qformer_layers: int = 2
    connector_dim: int = 512
    reconstruction_weight: float = 0.01
    pruning_weight: float = 10.0
    speed_weight: float = 1.0
    visual_checkpoint: str = ''
    visual_sha256: str = ''
    language_checkpoint: str = ''
    precision: str = 'bf16'
    checkpoint_visual: bool = True
    max_instruction_tokens: int = 128

    def validate(self):
        if self.input_mode not in ('raw', 'cached') or self.encoder_policy not in ('fusion', 'frozen', 'all_features'):
            raise ValueError('Choose raw/cached and explicit fusion/frozen/all_features visual policy')
        if self.input_mode == 'cached' and self.encoder_policy != 'frozen':
            raise ValueError('Encoder learning requires raw sensors; cached features cannot update an encoder')
        if self.language_mode not in ('ddia', 'causal', 'bypass'):
            raise ValueError('Unknown language ablation')
        if self.selection_policy not in ('fixed_budget', 'reference_bernoulli'):
            raise ValueError('Unknown CCDP selection policy')
        if self.language_policy not in ('frozen', 'last_layer', 'full'):
            raise ValueError('Unknown LM trainability policy')
        if self.precision not in ('bf16', 'fp32'):
            raise ValueError('Training supports bf16 or fp32, not unscaled fp16 Adam')
        if not (1 <= self.query_count <= 106):
            raise ValueError('query_count must be in [1,106]')
        if any(v < 0 for v in (self.reconstruction_weight, self.pruning_weight, self.speed_weight)):
            raise ValueError('Loss weights must be nonnegative')


class GoalStack(nn.Module):
    def __init__(self, config, *, device='cuda'):
        super().__init__()
        self.config = config
        config.validate()
        self.visual_encoder = None
        if config.input_mode == 'raw':
            with complete_visual_checkpoint_initialization():
                self.visual_encoder = GoalVisualEncoder(config.visual_checkpoint, policy=config.encoder_policy,
                    device=device, expected_sha256=config.visual_sha256)
        self.selector_route = FlattenRouteContextEncoder()
        self.connector = PaperQFormerMEFAConnector(PaperQFormerMEFAConfig(
            visual_dim=256, num_visual_tokens=106, language_dim=config.connector_dim,
            query_count=config.query_count, target_retention=config.query_count / 106.,
            memory_size=config.memory_size, route_context_dim=128,
            qformer_hidden_size=config.qformer_hidden, qformer_num_layers=config.qformer_layers,
            qformer_num_attention_heads=8, qformer_intermediate_size=config.qformer_hidden * 4,
            qformer_cross_attention_freq=2, hidden_dropout_prob=0., attention_dropout_prob=0.,
            reconstruction_layers=2, reconstruction_heads=8, reconstruction_dropout=0.))
        if config.selection_policy == 'fixed_budget':
            self.connector.ccdp = FixedBudgetCCDP(self.connector.ccdp)
        for name, parameter in self.connector.named_parameters():
            unused = name.startswith(('learnable_query_tokens', 'qformer.bert.embeddings.word_embeddings.',
                'qformer.bert.embeddings.position_embeddings.', 'qformer.bert.embeddings.token_type_embeddings.',
                'qformer.bert.pooler.'))
            unused |= (config.reconstruction_weight == 0 or not config.ccdp) and name.startswith(('reconstruction_', 'mask_embedding'))
            unused |= not config.mefa and name.startswith('temporal_encoder.')
            unused |= not config.ccdp and name.startswith('ccdp.')
            if unused:
                parameter.requires_grad_(False)
        if not config.ccdp:
            self.connector.ccdp.target_retention = 1.0
            self.selector_route.requires_grad_(False)
        self.language = None
        self.tokenizer = None
        self.language_manifest = None
        hidden = config.connector_dim
        if config.language_mode != 'bypass':
            path = Path(config.language_checkpoint)
            self.language_manifest = json.loads((path / 'asset_manifest.json').read_text())
            for name, digest in self.language_manifest['files'].items():
                if sha256(path / name) != digest:
                    raise ValueError('Pretrained LM asset hash mismatch: ' + name)
            components = load_tinyllama(str(path), local_files_only=True, freeze_tinyllama=True,
                use_ddia=config.language_mode == 'ddia',
                torch_dtype=torch.bfloat16 if config.precision == 'bf16' else torch.float32)
            self.tokenizer = components.tokenizer
            self.language = components.model.model  # decoder only; no unused vocabulary logits/head
            hidden = components.model.config.hidden_size
            if config.language_mode == 'ddia':
                for layer in self.language.layers:
                    layer.self_attn = EmptySafeDDIAAttention(layer.self_attn.original_attention, components.model.config)
            if config.language_policy == 'last_layer':
                self.language.layers[-1].requires_grad_(True)
                self.language.norm.requires_grad_(True)
            elif config.language_policy == 'full':
                self.language.requires_grad_(True)
                # Text embedding lookup is fixed; current data has no semantic supervision.
                self.language.embed_tokens.requires_grad_(False)
            for parameter in self.language.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()  # stable Adam state and updates
            self.max_positions = components.model.config.max_position_embeddings
        self.visual_to_lm = nn.Sequential(nn.LayerNorm(config.connector_dim), nn.Linear(config.connector_dim, hidden))
        self.route_token = nn.Sequential(nn.Linear(82, 256), nn.GELU(), nn.Linear(256, hidden))
        self.time_token = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, hidden))
        self.direct_route = nn.Sequential(nn.Linear(82, 128), nn.GELU())
        self.decoder = MetricTrajectoryDecoder(hidden + 128)
        if self.language is None:
            self.route_token.requires_grad_(False)
        self.to(device)

    def encode_visual(self, sensors):
        outputs = []
        for t in range(sensors['rgb_front'].shape[1]):
            keys = sorted(sensors)
            args = tuple(sensors[k][:, t:t+1] for k in keys)
            def encode(*values):
                return self.visual_encoder(dict(zip(keys, values)))
            use_checkpoint = (self.training and self.config.encoder_policy != 'frozen'
                              and self.config.checkpoint_visual and torch.is_grad_enabled())
            feature = checkpoint(encode, *args, use_reentrant=False) if use_checkpoint else encode(*args)
            outputs.append(feature)
        return torch.cat(outputs, 1)

    def forward(self, inputs, *, selection_mode=None, compute_auxiliary=None, retain_visual_grad=False):
        feature_key = 'sensors' if self.config.input_mode == 'raw' else 'visual_feature'
        if set(inputs) != {feature_key, 'route', 'elapsed_s', 'instruction'}:
            raise ValueError('Model input allowlist rejects targets, teacher, ROS/GPS/IMU metadata')
        route, elapsed = inputs['route'], inputs['elapsed_s']
        if route.ndim != 4 or route.shape[-2:] != (41, 2) or elapsed.shape != route.shape[:2]:
            raise ValueError('Expected route[B,T,41,2] and receive elapsed_s[B,T]')
        if not bool(torch.isfinite(elapsed).all()) or bool((elapsed[:, 0] != 0).any()) or bool((torch.diff(elapsed) <= 0).any()):
            raise ValueError('Time must increase from zero within each receive-causal window')
        visual = self.encode_visual(inputs['sensors']) if feature_key == 'sensors' else inputs['visual_feature']
        if visual.shape[:2] != route.shape[:2]:
            raise ValueError('Visual/route temporal shape mismatch')
        if retain_visual_grad and visual.requires_grad:
            visual.retain_grad()
        mode = 'retained_query_correct_history' if self.config.mefa else 'retained_query_current'
        selection = selection_mode or ('gumbel' if self.training and self.config.ccdp else 'topk')
        if not self.config.ccdp:
            selection = 'topk'  # exact all-106-token ablation, even during preflight
        auxiliary = self.training if compute_auxiliary is None else compute_auxiliary
        connected = self.connector(visual.float(), mode=mode, route_context=self.selector_route(route),
            selection_mode=selection, compute_reconstruction=auxiliary and self.config.ccdp and self.config.reconstruction_weight > 0)
        tokens, valid = connected['language_tokens'], connected['selected_valid_mask'].bool()
        expected_count = self.config.query_count if self.config.ccdp else 106
        if self.config.selection_policy == 'fixed_budget' and not bool(valid.sum(-1).eq(expected_count).all()):
            raise RuntimeError('CCDP train/inference token budget mismatch')
        batch, time, queries, _ = tokens.shape
        route_flat = (route.float() / 40.).flatten(-2)
        dt = torch.diff(elapsed, prepend=elapsed[:, :1], dim=1)
        time_embedding = self.time_token(torch.stack((elapsed, dt), -1).float())
        projected = self.visual_to_lm(tokens) + time_embedding.unsqueeze(2)
        metric_token = self.route_token(route_flat) + time_embedding
        chronological = torch.cat((metric_token.unsqueeze(2), projected), 2)
        frame_valid = torch.cat((torch.ones(batch, time, 1, dtype=torch.bool, device=valid.device), valid), 2)
        text = inputs['instruction']
        if not isinstance(text, (list, tuple)) or len(text) != batch or not all(isinstance(s, str) for s in text):
            raise ValueError('instruction must be a batch of real text strings or empty strings')
        if self.language is not None:
            instruction_embeds, instruction_mask = self.embed_instructions(text, chronological)
            sequence = build_multimodal_sequence(instruction_embeds, instruction_mask,
                chronological.flatten(1, 2), frame_valid.flatten(1, 2))
            if bool((sequence.attention_mask.sum(1) > self.max_positions).any()):
                raise ValueError('Window exceeds pretrained LM context; reduce tokens/history')
            if self.config.language_mode == 'ddia':
                set_ddia_modality_ids(self.language, sequence.modality_ids)
            result = self.language(inputs_embeds=sequence.inputs_embeds,
                attention_mask=sequence.attention_mask, position_ids=sequence.position_ids, use_cache=False,
                return_dict=True).last_hidden_state
            hidden = result[:, sequence.visual_start_index:].reshape(batch, time, queries + 1, -1)
        else:
            if any(text):
                raise ValueError('LM bypass ablation cannot silently discard semantic instructions')
            hidden = chronological
        # Pool valid VISUAL tokens only, not count-1 (padding can occur inside each frame).
        pooled = (hidden[:, :, 1:].float() * valid.unsqueeze(-1)).sum(2) / valid.sum(2).clamp_min(1).unsqueeze(-1)
        context = torch.cat((pooled, self.direct_route(route_flat).float()), -1)
        output = self.decoder(context, route)
        output.update(pruning_loss=connected['pruning_loss'], reconstruction_loss=connected['reconstruction_loss'],
                      selected_count=valid.sum(-1), visual_feature=visual)
        return output

    def embed_instructions(self, texts, reference):
        # No BOS, EOS, or canned prompt is inserted for the instruction-free dataset.
        ids = [self.tokenizer.encode(text, add_special_tokens=False) if text.strip() else [] for text in texts]
        if any(len(row) > self.config.max_instruction_tokens for row in ids):
            raise ValueError('Instruction too long; refusing silent semantic truncation')
        count = max(map(len, ids))
        if count == 0:
            return reference.new_zeros(len(ids), 0, reference.shape[-1]), torch.zeros(len(ids), 0, dtype=torch.bool, device=reference.device)
        index = torch.full((len(ids), count), self.tokenizer.pad_token_id, dtype=torch.long, device=reference.device)
        mask = torch.zeros_like(index, dtype=torch.bool)
        for b, row in enumerate(ids):
            if row:
                index[b, :len(row)] = torch.tensor(row, device=index.device)
                mask[b, :len(row)] = True
        return self.language.embed_tokens(index).to(reference.dtype), mask

    def gradient_groups(self):
        groups = dict(qformer_self='connector.qformer.bert.encoder.layer.0.attention.',
            qformer_cross='connector.qformer.bert.encoder.layer.0.crossattention.',
            retained_query='connector.retained_query_projection.', connector_output='connector.qformer_to_language.',
            visual_to_lm='visual_to_lm.', direct_route='direct_route.', trajectory='decoder.residual_head.',
            speed='decoder.speed_head.', temporal_positions='time_token.')
        if self.config.ccdp:
            groups.update(selector='connector.ccdp.score_mlp.', route_selector='selector_route.')
        if self.config.mefa:
            groups['mefa_history'] = 'connector.temporal_encoder.'
        if self.language is not None:
            groups['metric_route_token'] = 'route_token.'
            if self.config.language_policy != 'frozen':
                groups['pretrained_lm_ddia' if self.config.language_mode == 'ddia' else 'pretrained_lm_causal'] = 'language.layers.{}.self_attn.'.format(len(self.language.layers) - 1)
        if self.visual_encoder is not None and self.config.encoder_policy != 'frozen':
            groups['visual_encoder_fusion'] = 'visual_encoder.encoder.encoder.'
            groups['visual_encoder_decoder'] = 'visual_encoder.encoder.decoder.'
            if self.config.encoder_policy == 'all_features':
                groups['rgb_backbone'] = 'visual_encoder.encoder.rgb_backbone.'
                groups['lidar_backbone'] = 'visual_encoder.encoder.lidar_backbone.'
                groups['rgb_projection'] = 'visual_encoder.encoder.rgb_patch_embed.proj.'
        return groups

    def describe(self):
        return dict(config=asdict(self.config), language_asset=self.language_manifest,
            visual_encoder=self.visual_encoder.describe() if self.visual_encoder else 'cached frozen ablation',
            total_parameters=sum(p.numel() for p in self.parameters()),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            trainable_names=[n for n, p in self.named_parameters() if p.requires_grad])
