"""
Paper-aligned Modified Q-Former MEFA connector for the VLDrive reproduction.

Scope
-----
This module implements the connector contract described by VLDrive:

    B_i       = [F_{i-Z}, ..., F_{i-1}]
    B_i^avg   = Avg(B_i)
    TE_i      = MLP(F_i || B_i^avg)
    F_i^v     = Q-Former(F_i^k, F_i + TE_i)

where:
- F_i is the current frame's N visual tokens;
- F_i^k is the subset retained by CCDP;
- retained visual tokens replace conventional learnable Q-Former queries;
- the full current-frame feature enhanced by temporal context is used as K/V.

This is a paper-aligned independent implementation, not an author-source-
identical reproduction. The VLDrive main paper does not disclose every
Q-Former hyperparameter. By default, the reference configuration follows the
LMDrive/BLIP-2 Q-Former family: BERT width 768, 12 layers, 12 heads, and
cross-attention every two layers. All such choices are configurable.

CARLA/LMDrive usage in this repository is for implementation-contract and
regression tests. Final performance optimization and domain adaptation belong
in the MORAI competition environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .ccdp import CCDPTokenSparsifier
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel


MODE_LEARNABLE_CURRENT = "learnable_query_current"
MODE_RETAINED_CURRENT = "retained_query_current"
MODE_RETAINED_ZERO = "retained_query_zero_history"
MODE_RETAINED_MEFA = "retained_query_correct_history"

SUPPORTED_MODES = (
    MODE_LEARNABLE_CURRENT,
    MODE_RETAINED_CURRENT,
    MODE_RETAINED_ZERO,
    MODE_RETAINED_MEFA,
)


@dataclass
class QFormerMEFAConfig:
    """Serializable connector configuration."""

    visual_dim: int = 256
    num_visual_tokens: int = 106
    language_dim: int = 2048

    query_count: int = 32
    target_retention: float = 0.30
    memory_size: int = 10

    route_context_dim: int = 0
    ego_context_dim: int = 0

    qformer_hidden_size: int = 768
    qformer_num_layers: int = 12
    qformer_num_attention_heads: int = 12
    qformer_intermediate_size: int = 3072
    qformer_cross_attention_freq: int = 2

    hidden_dropout_prob: float = 0.0
    attention_dropout_prob: float = 0.0
    initializer_range: float = 0.02

    # Legacy MORAI experiments opt into their original graph and mean loss.
    reconstruction_formulation: str = "eq8"
    reconstruction_reduction: str = "sum"
    reconstruction_layers: int = 4
    reconstruction_heads: int = 8
    reconstruction_dropout: float = 0.0
    use_reconstruction_position_embedding: bool = True

    lambda_pruning: float = 10.0
    lambda_reconstruction: float = 1.0

    def validate(self) -> None:
        if self.visual_dim <= 0:
            raise ValueError("visual_dim must be positive")
        if self.num_visual_tokens <= 0:
            raise ValueError("num_visual_tokens must be positive")
        if self.language_dim <= 0:
            raise ValueError("language_dim must be positive")
        if not 1 <= self.query_count <= self.num_visual_tokens:
            raise ValueError(
                "query_count must be in [1, num_visual_tokens]"
            )
        if not 0.0 < self.target_retention <= 1.0:
            raise ValueError("target_retention must be in (0,1]")
        if self.memory_size < 1:
            raise ValueError("memory_size must be >= 1")

        if self.route_context_dim < 0 or self.ego_context_dim < 0:
            raise ValueError("context dimensions cannot be negative")

        if self.qformer_hidden_size <= 0:
            raise ValueError("qformer_hidden_size must be positive")
        if self.qformer_num_layers < 1:
            raise ValueError("qformer_num_layers must be >= 1")
        if self.qformer_num_attention_heads < 1:
            raise ValueError("qformer_num_attention_heads must be >= 1")
        if (
            self.qformer_hidden_size
            % self.qformer_num_attention_heads
            != 0
        ):
            raise ValueError(
                "qformer_hidden_size must be divisible by "
                "qformer_num_attention_heads"
            )
        if self.qformer_intermediate_size < self.qformer_hidden_size:
            raise ValueError(
                "qformer_intermediate_size must be >= qformer_hidden_size"
            )
        if self.qformer_cross_attention_freq < 1:
            raise ValueError(
                "qformer_cross_attention_freq must be >= 1"
            )

        if self.reconstruction_formulation not in ("eq8", "legacy"):
            raise ValueError("reconstruction_formulation must be eq8 or legacy")
        if self.reconstruction_reduction not in ("sum", "mean"):
            raise ValueError("reconstruction_reduction must be sum or mean")
        if self.reconstruction_layers < 1:
            raise ValueError("reconstruction_layers must be >= 1")
        if self.reconstruction_heads < 1:
            raise ValueError("reconstruction_heads must be >= 1")
        if self.visual_dim % self.reconstruction_heads != 0:
            raise ValueError(
                "visual_dim must be divisible by reconstruction_heads"
            )

        for name, value in (
            ("hidden_dropout_prob", self.hidden_dropout_prob),
            ("attention_dropout_prob", self.attention_dropout_prob),
            (
                "reconstruction_dropout",
                self.reconstruction_dropout,
            ),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError("%s must be in [0,1)" % name)

        if self.initializer_range <= 0.0:
            raise ValueError("initializer_range must be positive")
        if self.lambda_pruning < 0.0:
            raise ValueError("lambda_pruning cannot be negative")
        if self.lambda_reconstruction < 0.0:
            raise ValueError(
                "lambda_reconstruction cannot be negative"
            )

        # query_count is used only by the learned-query/top-k ablations.
        # Dynamic CCDP has no fixed query-count contract.

    def to_dict(self) -> Dict[str, object]:
        self.validate()
        return asdict(self)


class CausalMEFATemporalEncoder(nn.Module):
    """
    VLDrive Eq. (5)-(7) temporal feature enhancement.

    For frame i:
        history = mean(F[max(0,i-Z):i])
        TE_i = MLP([F_i ; history])
        enhanced_i = F_i + TE_i

    The current frame is never included in its own history. Future frames are
    never read. Frame zero receives an all-zero history tensor.
    """

    def __init__(
        self,
        dim: int,
        memory_size: int,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError("dim must be positive")
        if memory_size < 1:
            raise ValueError("memory_size must be >= 1")

        self.dim = int(dim)
        self.memory_size = int(memory_size)

        # This width matches the already validated lightweight MEFA proxy:
        # 2C -> C -> C. The paper does not state the hidden width.
        self.temporal_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def causal_history_average(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: [B,T,N,C]

        Returns:
            history_average: [B,T,N,C]
        """
        if features.ndim != 4:
            raise ValueError(
                "features must have shape [B,T,N,C], got %s"
                % (tuple(features.shape),)
            )
        if features.shape[-1] != self.dim:
            raise ValueError(
                "expected feature dim %d, got %d"
                % (
                    self.dim,
                    features.shape[-1],
                )
            )

        time_steps = int(features.shape[1])
        history_average = torch.zeros_like(features)

        # Explicit loop keeps the causality contract obvious and auditable.
        for time_index in range(1, time_steps):
            start = max(
                0,
                time_index - self.memory_size,
            )
            history_average[:, time_index] = (
                features[:, start:time_index].mean(
                    dim=1
                )
            )

        return history_average

    def forward(
        self,
        features: torch.Tensor,
        history_mode: str = "correct",
        history_average_override: Optional[
            torch.Tensor
        ] = None,
    ) -> Dict[str, torch.Tensor]:
        if history_average_override is not None:
            if tuple(history_average_override.shape) != tuple(
                features.shape
            ):
                raise ValueError(
                    "history_average_override must have shape %s, got %s"
                    % (
                        tuple(features.shape),
                        tuple(
                            history_average_override.shape
                        ),
                    )
                )
            history_average = (
                history_average_override
            )

        elif history_mode == "correct":
            history_average = (
                self.causal_history_average(
                    features
                )
            )

        elif history_mode == "zero":
            history_average = torch.zeros_like(
                features
            )

        elif history_mode == "current_copy":
            history_average = features

        else:
            raise ValueError(
                "history_mode must be one of: "
                "correct, zero, current_copy"
            )

        temporal_input = torch.cat(
            [
                features,
                history_average,
            ],
            dim=-1,
        )

        temporal_encoding = self.temporal_mlp(
            temporal_input
        )
        enhanced_features = (
            features + temporal_encoding
        )

        return {
            "history_average": history_average,
            "temporal_encoding": temporal_encoding,
            "enhanced_features": enhanced_features,
        }


def build_qformer(
    config: QFormerMEFAConfig,
    local_pretrained_path: Optional[
        str
    ] = None,
) -> BertLMHeadModel:
    """
    Build the LMDrive/LAVIS Q-Former implementation without network access.

    When local_pretrained_path is None, Q-Former is randomly initialized from
    an explicit BERT-style configuration. If a local Hugging Face BERT
    directory is supplied, only local files are used.
    """
    config.validate()

    bert_config = BertConfig(
        vocab_size=30522,
        hidden_size=config.qformer_hidden_size,
        num_hidden_layers=config.qformer_num_layers,
        num_attention_heads=(
            config.qformer_num_attention_heads
        ),
        intermediate_size=(
            config.qformer_intermediate_size
        ),
        hidden_act="gelu",
        hidden_dropout_prob=(
            config.hidden_dropout_prob
        ),
        attention_probs_dropout_prob=(
            config.attention_dropout_prob
        ),
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=(
            config.initializer_range
        ),
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=False,
    )

    # Custom fields consumed by LMDrive's LAVIS Qformer.py.
    bert_config.encoder_width = config.visual_dim
    bert_config.add_cross_attention = True
    bert_config.cross_attention_freq = (
        config.qformer_cross_attention_freq
    )
    bert_config.query_length = config.query_count
    bert_config.chunk_size_feed_forward = 0

    if local_pretrained_path is None:
        qformer = BertLMHeadModel(
            bert_config
        )
    else:
        local_path = Path(
            local_pretrained_path
        ).expanduser().resolve()

        if not local_path.exists():
            raise FileNotFoundError(local_path)

        qformer = (
            BertLMHeadModel.from_pretrained(
                str(local_path),
                config=bert_config,
                local_files_only=True,
            )
        )

    # We use Qformer.bert only; the vocabulary prediction head is irrelevant.
    qformer.cls = None
    return qformer


class QFormerMEFAConnector(nn.Module):
    """
    CCDP + paper-aligned Modified Q-Former MEFA connector.

    Four modes are supported in one implementation:

    learnable_query_current
        Q = learned query tokens, K/V = current F_i

    retained_query_current
        Q = CCDP-retained F_i^k, K/V = current F_i

    retained_query_zero_history
        Q = CCDP-retained F_i^k,
        K/V = F_i + MLP([F_i ; 0])

    retained_query_correct_history
        Q = CCDP-retained F_i^k,
        K/V = F_i + MLP([F_i ; B_i^avg])
    """

    def __init__(
        self,
        config: Optional[
            QFormerMEFAConfig
        ] = None,
        qformer_local_pretrained_path: Optional[
            str
        ] = None,
    ) -> None:
        super().__init__()

        self.config = (
            config
            if config is not None
            else QFormerMEFAConfig()
        )
        self.config.validate()

        cfg = self.config

        # LMDrive applies ln_vision after its frozen visual encoder.
        self.visual_input_norm = nn.LayerNorm(
            cfg.visual_dim
        )

        self.ccdp = CCDPTokenSparsifier(
            dim=cfg.visual_dim,
            target_retention=(
                cfg.target_retention
            ),
            temperature=1.0,
            route_context_dim=(
                cfg.route_context_dim
            ),
            ego_context_dim=(
                cfg.ego_context_dim
            ),
        )

        self.temporal_encoder = (
            CausalMEFATemporalEncoder(
                dim=cfg.visual_dim,
                memory_size=cfg.memory_size,
            )
        )

        # Retained visual tokens are content-dependent Q-Former queries.
        self.retained_query_projection = nn.Sequential(
            nn.LayerNorm(cfg.visual_dim),
            nn.Linear(
                cfg.visual_dim,
                cfg.qformer_hidden_size,
            ),
        )

        # Matched baseline for conventional Q-Former queries.
        self.learnable_query_tokens = nn.Parameter(
            torch.empty(
                1,
                cfg.query_count,
                cfg.qformer_hidden_size,
            )
        )
        nn.init.normal_(
            self.learnable_query_tokens,
            mean=0.0,
            std=cfg.initializer_range,
        )

        self.qformer = build_qformer(
            cfg,
            local_pretrained_path=(
                qformer_local_pretrained_path
            ),
        )

        self.qformer_to_language = nn.Sequential(
            nn.LayerNorm(
                cfg.qformer_hidden_size
            ),
            nn.Linear(
                cfg.qformer_hidden_size,
                cfg.language_dim,
            ),
        )

        # Training-only CCDP reconstruction branch.
        align_hidden = max(
            cfg.visual_dim * 2,
            cfg.language_dim // 2,
        )
        self.reconstruction_align = nn.Sequential(
            nn.Linear(
                cfg.language_dim,
                align_hidden,
            ),
            nn.GELU(),
            nn.Linear(
                align_hidden,
                cfg.visual_dim,
            ),
        )

        self.mask_embedding = nn.Parameter(
            torch.empty(
                1,
                cfg.num_visual_tokens,
                cfg.visual_dim,
            )
        )
        nn.init.normal_(
            self.mask_embedding,
            mean=0.0,
            std=0.02,
        )

        reconstruction_layer = (
            nn.TransformerEncoderLayer(
                d_model=cfg.visual_dim,
                nhead=cfg.reconstruction_heads,
                dim_feedforward=(
                    cfg.visual_dim * 4
                ),
                dropout=(
                    cfg.reconstruction_dropout
                ),
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
        )
        self.reconstruction_transformer = (
            nn.TransformerEncoder(
                reconstruction_layer,
                num_layers=(
                    cfg.reconstruction_layers
                ),
            )
        )

        if (
            cfg
            .use_reconstruction_position_embedding
        ):
            self.reconstruction_position_embedding = (
                nn.Parameter(
                    torch.empty(
                        1,
                        cfg.num_visual_tokens,
                        cfg.visual_dim,
                    )
                )
            )
            nn.init.normal_(
                self.reconstruction_position_embedding,
                mean=0.0,
                std=0.02,
            )
        else:
            self.register_parameter(
                "reconstruction_position_embedding",
                None,
            )

    @property
    def visual_dim(self) -> int:
        return self.config.visual_dim

    @property
    def num_visual_tokens(self) -> int:
        return self.config.num_visual_tokens

    @property
    def language_dim(self) -> int:
        return self.config.language_dim

    @property
    def query_count(self) -> int:
        return self.config.query_count

    def cross_attention_layer_indices(
        self,
    ) -> List[int]:
        indices = []

        for index, layer in enumerate(
            self.qformer
            .bert
            .encoder
            .layer
        ):
            if bool(
                getattr(
                    layer,
                    "has_cross_attention",
                    False,
                )
            ):
                indices.append(index)

        return indices

    def _validate_features(
        self,
        features: torch.Tensor,
    ) -> Tuple[int, int]:
        if features.ndim != 4:
            raise ValueError(
                "features must have shape [B,T,N,C], got %s"
                % (tuple(features.shape),)
            )

        if (
            int(features.shape[2])
            != self.num_visual_tokens
        ):
            raise ValueError(
                "expected N=%d, got N=%d"
                % (
                    self.num_visual_tokens,
                    int(features.shape[2]),
                )
            )

        if (
            int(features.shape[3])
            != self.visual_dim
        ):
            raise ValueError(
                "expected C=%d, got C=%d"
                % (
                    self.visual_dim,
                    int(features.shape[3]),
                )
            )

        return (
            int(features.shape[0]),
            int(features.shape[1]),
        )

    @staticmethod
    def _flatten_frames(
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        return tensor.reshape(
            -1,
            tensor.shape[-2],
            tensor.shape[-1],
        )

    @staticmethod
    def _pack_by_mask(
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Physically gather selected query tokens in original token-index order.

        The gathered values are multiplied by the straight-through mask values,
        preserving a downstream gradient path to the CCDP selector when a
        Gumbel mask is used.
        """
        if features.ndim != 3:
            raise ValueError(
                "features must be [BT,N,C]"
            )
        if mask.ndim != 2:
            raise ValueError(
                "mask must be [BT,N]"
            )
        if tuple(features.shape[:2]) != tuple(
            mask.shape
        ):
            raise ValueError(
                "features/mask shape mismatch"
            )

        batch_frames = int(
            features.shape[0]
        )
        dim = int(features.shape[2])

        index_list: List[torch.Tensor] = []
        max_count = 1

        for frame_index in range(
            batch_frames
        ):
            indices = torch.nonzero(
                mask[frame_index] > 0.5,
                as_tuple=False,
            ).flatten()

            if indices.numel() == 0:
                raise RuntimeError(
                    "CCDP produced an empty token set"
                )

            index_list.append(indices)
            max_count = max(
                max_count,
                int(indices.numel()),
            )

        packed = features.new_zeros(
            batch_frames,
            max_count,
            dim,
        )
        valid = torch.zeros(
            batch_frames,
            max_count,
            dtype=torch.bool,
            device=features.device,
        )
        packed_indices = torch.full(
            (
                batch_frames,
                max_count,
            ),
            -1,
            dtype=torch.long,
            device=features.device,
        )

        for frame_index, indices in enumerate(
            index_list
        ):
            count = int(indices.numel())
            selected_mask = mask[
                frame_index,
                indices,
            ].unsqueeze(-1)

            packed[
                frame_index,
                :count,
            ] = (
                features[
                    frame_index,
                    indices,
                ]
                * selected_mask
            )
            valid[
                frame_index,
                :count,
            ] = True
            packed_indices[
                frame_index,
                :count,
            ] = indices

        return (
            packed,
            valid,
            packed_indices,
        )

    def _run_qformer(
        self,
        query_embeddings: torch.Tensor,
        query_valid_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_valid_mask: Optional[
            torch.Tensor
        ] = None,
        capture_cross_attention: bool = False,
    ) -> Tuple[
        torch.Tensor,
        List[Tuple[int, torch.Tensor]],
    ]:
        """
        Run LMDrive's BERT Q-Former.

        query_embeddings: [BT,K,H]
        query_valid_mask: [BT,K], True for real query slots
        memory: [BT,N,C]
        memory_valid_mask: [BT,N], True for valid K/V slots
        """
        if query_embeddings.ndim != 3:
            raise ValueError(
                "query_embeddings must be [BT,K,H]"
            )
        if query_valid_mask.ndim != 2:
            raise ValueError(
                "query_valid_mask must be [BT,K]"
            )
        if memory.ndim != 3:
            raise ValueError(
                "memory must be [BT,N,C]"
            )

        if tuple(
            query_embeddings.shape[:2]
        ) != tuple(query_valid_mask.shape):
            raise ValueError(
                "query/query_valid shape mismatch"
            )

        if (
            int(query_embeddings.shape[-1])
            != self.config.qformer_hidden_size
        ):
            raise ValueError(
                "wrong Q-Former query width"
            )

        if (
            int(memory.shape[-1])
            != self.visual_dim
        ):
            raise ValueError(
                "wrong Q-Former memory width"
            )

        if memory_valid_mask is None:
            memory_valid_mask = torch.ones(
                memory.shape[:2],
                dtype=torch.bool,
                device=memory.device,
            )

        if tuple(memory_valid_mask.shape) != tuple(
            memory.shape[:2]
        ):
            raise ValueError(
                "memory/memory_valid shape mismatch"
            )

        captured: List[
            Tuple[int, torch.Tensor]
        ] = []
        handles = []

        if capture_cross_attention:
            for layer_index, layer in enumerate(
                self.qformer
                .bert
                .encoder
                .layer
            ):
                if not bool(
                    getattr(
                        layer,
                        "has_cross_attention",
                        False,
                    )
                ):
                    continue

                def make_hook(index: int):
                    def hook(
                        module,
                        inputs,
                        output,
                    ):
                        if (
                            not isinstance(
                                output,
                                tuple,
                            )
                            or len(output) < 2
                        ):
                            raise RuntimeError(
                                "Q-Former cross-attention "
                                "did not return an attention map"
                            )

                        captured.append(
                            (
                                index,
                                output[1],
                            )
                        )

                    return hook

                handles.append(
                    layer.crossattention.register_forward_hook(
                        make_hook(layer_index)
                    )
                )

        try:
            qformer_output = (
                self.qformer.bert(
                    input_ids=None,
                    attention_mask=(
                        query_valid_mask.long()
                    ),
                    query_embeds=(
                        query_embeddings
                    ),
                    encoder_hidden_states=(
                        memory
                    ),
                    encoder_attention_mask=(
                        memory_valid_mask.long()
                    ),
                    use_cache=False,
                    output_attentions=(
                        capture_cross_attention
                    ),
                    output_hidden_states=False,
                    return_dict=True,
                    is_decoder=False,
                )
            )
        finally:
            for handle in handles:
                handle.remove()

        hidden = (
            qformer_output
            .last_hidden_state
        )

        if tuple(hidden.shape) != tuple(
            query_embeddings.shape
        ):
            raise RuntimeError(
                "unexpected Q-Former output shape %s; "
                "expected %s"
                % (
                    tuple(hidden.shape),
                    tuple(
                        query_embeddings.shape
                    ),
                )
            )

        hidden = (
            hidden
            * query_valid_mask
            .unsqueeze(-1)
            .to(hidden.dtype)
        )

        return hidden, captured

    def _reconstruct(
        self,
        normalized_features: torch.Tensor,
        language_tokens_flat: torch.Tensor,
        selected_indices_flat: torch.Tensor,
        selected_valid_flat: torch.Tensor,
        full_selection_mask_flat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Training-only reconstruction of the pruned visual tokens.

        The sparse Q-Former output is aligned C_t -> C and scattered back to
        each selected original token position. Dropped positions receive
        learnable mask embeddings. Loss is evaluated only on dropped positions.
        """
        batch_frames = int(
            normalized_features.shape[0]
        )
        num_tokens = int(
            normalized_features.shape[1]
        )
        visual_dim = int(
            normalized_features.shape[2]
        )

        aligned = self.reconstruction_align(
            language_tokens_flat
        )
        valid_float = (
            selected_valid_flat
            .unsqueeze(-1)
            .to(aligned.dtype)
        )
        aligned = aligned * valid_float

        safe_indices = (
            selected_indices_flat
            .clamp_min(0)
        )
        scatter_indices = (
            safe_indices
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                visual_dim,
            )
        )

        dense_selected = (
            normalized_features.new_zeros(
                batch_frames,
                num_tokens,
                visual_dim,
            )
        )
        dense_selected = dense_selected.scatter_add(
            dim=1,
            index=scatter_indices,
            src=aligned,
        )

        mask_3d = (
            full_selection_mask_flat
            .unsqueeze(-1)
        )
        mask_embeddings = (
            self.mask_embedding.expand(
                batch_frames,
                -1,
                -1,
            )
        )

        if self.config.reconstruction_formulation == "eq8":
            # Fig. 3: replace kept positions with aligned enhanced tokens,
            # retain original features at dropped positions, then apply Eq. 8.
            # Packing is discrete; detach only that placement decision. The
            # explicit M below retains the straight-through selector gradient.
            filled = dense_selected + normalized_features * (1.0 - mask_3d.detach())
            reconstruction_base = filled * mask_3d + mask_embeddings * (1.0 - mask_3d)
        else:
            reconstruction_base = dense_selected + mask_embeddings * (1.0 - mask_3d)

        reconstruction_input = (
            reconstruction_base
        )

        if (
            self
            .reconstruction_position_embedding
            is not None
        ):
            reconstruction_input = (
                reconstruction_input
                + self
                .reconstruction_position_embedding
                .expand(
                    batch_frames,
                    -1,
                    -1,
                )
            )

        reconstructed = (
            self.reconstruction_transformer(
                reconstruction_input
            )
        )

        # Manual squared-error reductions need the same FP32 protection as
        # torch's autocast-aware loss operators. A finite FP16 per-element
        # error can overflow when summed across tokens/channels.
        reconstructed_for_loss = (
            reconstructed.float()
            if reconstructed.dtype in (torch.float16, torch.bfloat16)
            else reconstructed
        )
        target = normalized_features.detach().to(reconstructed_for_loss.dtype)
        dropped = (
            1.0
            - full_selection_mask_flat.to(reconstructed_for_loss.dtype)
        )
        squared_error = (
            target - reconstructed_for_loss
        ).pow(2)

        numerator = (
            squared_error
            * dropped.unsqueeze(-1)
        ).sum(dim=(1, 2))

        # Eq. 13 sums dropped-token/channel errors within each frame;
        # batching averages frames. The legacy mean is an explicit ablation.
        if self.config.reconstruction_reduction == "sum":
            loss_per_frame = numerator
        else:
            denominator = (dropped.sum(dim=1) * visual_dim).clamp_min(1.0)
            loss_per_frame = numerator / denominator
        loss = loss_per_frame.mean()

        return {
            "aligned_sparse_tokens": aligned,
            "reconstruction_base_input": (
                reconstruction_base
            ),
            "reconstruction_input": (
                reconstruction_input
            ),
            "reconstructed_features": (
                reconstructed
            ),
            "reconstruction_loss_per_frame": (
                loss_per_frame
            ),
            "reconstruction_loss": loss,
            "dropped_mask": dropped,
        }

    def forward(
        self,
        features: torch.Tensor,
        mode: str = MODE_RETAINED_MEFA,
        route_context: Optional[
            torch.Tensor
        ] = None,
        ego_context: Optional[
            torch.Tensor
        ] = None,
        selection_mode: Optional[str] = None,
        hard_mask: Optional[
            torch.Tensor
        ] = None,
        history_mode: Optional[str] = None,
        history_average_override: Optional[
            torch.Tensor
        ] = None,
        compute_reconstruction: Optional[bool] = None,
        capture_cross_attention: bool = False,
    ) -> Dict[str, object]:
        # None delegates to CCDP: Gumbel in train, dynamic argmax in eval.
        # Reconstruction is a training-only default; ablations can opt out.
        if compute_reconstruction is None:
            compute_reconstruction = self.training and mode != MODE_LEARNABLE_CURRENT
        if mode not in SUPPORTED_MODES:
            raise ValueError(
                "mode must be one of %s, got %s"
                % (
                    SUPPORTED_MODES,
                    mode,
                )
            )

        batch_size, time_steps = (
            self._validate_features(
                features
            )
        )
        cfg = self.config

        normalized = self.visual_input_norm(
            features
        )
        normalized_flat = (
            self._flatten_frames(
                normalized
            )
        )

        zero_scalar = features.new_zeros(())

        selector: Optional[
            Dict[str, torch.Tensor]
        ]
        selection_mask: torch.Tensor
        selected_indices_flat: torch.Tensor
        selected_valid_flat: torch.Tensor
        initial_query_flat: torch.Tensor

        if mode == MODE_LEARNABLE_CURRENT:
            selector = None
            selection_mask = features.new_zeros(
                batch_size,
                time_steps,
                cfg.num_visual_tokens,
            )

            selected_indices_flat = torch.full(
                (
                    batch_size
                    * time_steps,
                    cfg.query_count,
                ),
                -1,
                dtype=torch.long,
                device=features.device,
            )
            selected_valid_flat = torch.ones(
                (
                    batch_size
                    * time_steps,
                    cfg.query_count,
                ),
                dtype=torch.bool,
                device=features.device,
            )

            initial_query_flat = (
                self.learnable_query_tokens
                .expand(
                    batch_size * time_steps,
                    -1,
                    -1,
                )
            )
            pruning_loss = zero_scalar

        else:
            selector = self.ccdp(
                normalized,
                route_context=route_context,
                ego_context=ego_context,
                hard=True,
                selection_mode=(
                    selection_mode
                ),
            )

            if hard_mask is None:
                selection_mask = (
                    selector["mask"]
                )
            else:
                # Diagnostic override only: pruning_loss below still describes
                # the selector's own sample. Do not train with this override.
                if tuple(
                    hard_mask.shape
                ) != tuple(
                    features.shape[:3]
                ):
                    raise ValueError(
                        "hard_mask must have shape %s, got %s"
                        % (
                            tuple(
                                features.shape[:3]
                            ),
                            tuple(
                                hard_mask.shape
                            ),
                        )
                    )

                if not bool(((hard_mask == 0) | (hard_mask == 1)).all()):
                    raise ValueError("hard_mask must contain only finite binary 0/1 values")

                if bool(
                    (
                        hard_mask.sum(dim=-1)
                        < 1
                    ).any()
                ):
                    raise ValueError(
                        "hard_mask contains an empty frame"
                    )

                selection_mask = hard_mask.to(
                    dtype=normalized.dtype,
                    device=normalized.device,
                )

            mask_flat = (
                selection_mask.reshape(
                    -1,
                    cfg.num_visual_tokens,
                )
            )

            (
                retained_visual_flat,
                selected_valid_flat,
                selected_indices_flat,
            ) = self._pack_by_mask(
                normalized_flat,
                mask_flat,
            )

            initial_query_flat = (
                self
                .retained_query_projection(
                    retained_visual_flat
                )
            )

            pruning_loss = (
                selector["pruning_loss"]
            )

        history_average = torch.zeros_like(
            normalized
        )
        temporal_encoding = torch.zeros_like(
            normalized
        )

        if mode in (
            MODE_LEARNABLE_CURRENT,
            MODE_RETAINED_CURRENT,
        ):
            enhanced_memory = normalized

        else:
            if history_mode is None:
                history_mode = (
                    "zero"
                    if mode
                    == MODE_RETAINED_ZERO
                    else "correct"
                )

            if mode == MODE_RETAINED_ZERO:
                # Keep the matched-capacity baseline honest even if a caller
                # accidentally requests another history mode.
                history_mode = "zero"
                history_average_override = None

            temporal = self.temporal_encoder(
                normalized,
                history_mode=history_mode,
                history_average_override=(
                    history_average_override
                ),
            )
            history_average = (
                temporal["history_average"]
            )
            temporal_encoding = (
                temporal["temporal_encoding"]
            )
            enhanced_memory = (
                temporal["enhanced_features"]
            )

        memory_flat = self._flatten_frames(
            enhanced_memory
        )
        memory_valid_flat = torch.ones(
            memory_flat.shape[:2],
            dtype=torch.bool,
            device=memory_flat.device,
        )

        qformer_hidden_flat, attention_maps = (
            self._run_qformer(
                query_embeddings=(
                    initial_query_flat
                ),
                query_valid_mask=(
                    selected_valid_flat
                ),
                memory=memory_flat,
                memory_valid_mask=(
                    memory_valid_flat
                ),
                capture_cross_attention=(
                    capture_cross_attention
                ),
            )
        )

        language_tokens_flat = (
            self.qformer_to_language(
                qformer_hidden_flat
            )
        )
        language_tokens_flat = (
            language_tokens_flat
            * selected_valid_flat
            .unsqueeze(-1)
            .to(
                language_tokens_flat.dtype
            )
        )

        max_queries = int(
            language_tokens_flat.shape[1]
        )

        result: Dict[str, object] = {
            "mode": mode,
            "normalized_features": normalized,
            "selector": selector,
            "selection_mask": selection_mask,
            "selected_indices": (
                selected_indices_flat.reshape(
                    batch_size,
                    time_steps,
                    max_queries,
                )
            ),
            "selected_valid_mask": (
                selected_valid_flat.reshape(
                    batch_size,
                    time_steps,
                    max_queries,
                )
            ),
            "selected_count_per_frame": (
                selected_valid_flat
                .sum(dim=-1)
                .reshape(
                    batch_size,
                    time_steps,
                )
            ),
            "initial_query_embeddings": (
                initial_query_flat.reshape(
                    batch_size,
                    time_steps,
                    max_queries,
                    cfg.qformer_hidden_size,
                )
            ),
            "history_average": history_average,
            "temporal_encoding": (
                temporal_encoding
            ),
            "enhanced_memory": enhanced_memory,
            "qformer_hidden": (
                qformer_hidden_flat.reshape(
                    batch_size,
                    time_steps,
                    max_queries,
                    cfg.qformer_hidden_size,
                )
            ),
            "language_tokens": (
                language_tokens_flat.reshape(
                    batch_size,
                    time_steps,
                    max_queries,
                    cfg.language_dim,
                )
            ),
            "cross_attention_maps": (
                attention_maps
            ),
            "pruning_loss": pruning_loss,
            "reconstruction_computed": False,
            "reconstruction_supported": (
                mode
                != MODE_LEARNABLE_CURRENT
            ),
            "reconstruction_loss": (
                zero_scalar
            ),
            "cycle_loss": (
                cfg.lambda_pruning
                * pruning_loss
            ),
        }

        if compute_reconstruction:
            if mode == MODE_LEARNABLE_CURRENT:
                raise ValueError(
                    "Token reconstruction requires CCDP-retained "
                    "queries with original visual-token indices"
                )

            reconstruction = self._reconstruct(
                normalized_features=(
                    normalized_flat
                ),
                language_tokens_flat=(
                    language_tokens_flat
                ),
                selected_indices_flat=(
                    selected_indices_flat
                ),
                selected_valid_flat=(
                    selected_valid_flat
                ),
                full_selection_mask_flat=(
                    selection_mask.reshape(
                        -1,
                        cfg.num_visual_tokens,
                    )
                ),
            )

            result.update(reconstruction)
            result[
                "reconstruction_computed"
            ] = True
            result["cycle_loss"] = (
                cfg.lambda_pruning
                * pruning_loss
                + cfg.lambda_reconstruction
                * reconstruction[
                    "reconstruction_loss"
                ]
            )

        return result

    def load_ccdp_frontend_checkpoint(
        self,
        checkpoint_or_path,
    ) -> Dict[str, object]:
        """
        Load only visual_input_norm and CCDP from existing reproduction
        checkpoints.

        Supported state prefixes:
            base.connector.*
            connector.*
            (direct module keys)
        """
        if isinstance(
            checkpoint_or_path,
            (str, Path),
        ):
            path = Path(
                checkpoint_or_path
            ).expanduser().resolve()

            if not path.exists():
                raise FileNotFoundError(path)

            checkpoint = torch.load(
                path,
                map_location="cpu",
            )
            source = str(path)
        elif isinstance(
            checkpoint_or_path,
            dict,
        ):
            checkpoint = checkpoint_or_path
            source = "<in-memory>"
        else:
            raise TypeError(
                "checkpoint_or_path must be a path or dict"
            )

        state = checkpoint.get(
            "model",
            checkpoint,
        )

        if not isinstance(state, dict):
            raise RuntimeError(
                "checkpoint model state is not a dict"
            )

        loaded: Dict[
            str,
            List[str],
        ] = {}

        for module_name, module in (
            (
                "visual_input_norm",
                self.visual_input_norm,
            ),
            (
                "ccdp",
                self.ccdp,
            ),
        ):
            target_state = (
                module.state_dict()
            )
            extracted = {}
            source_keys = []

            for target_key in target_state:
                candidates = (
                    "base.connector.%s.%s"
                    % (
                        module_name,
                        target_key,
                    ),
                    "connector.%s.%s"
                    % (
                        module_name,
                        target_key,
                    ),
                    "%s.%s"
                    % (
                        module_name,
                        target_key,
                    ),
                )

                matched = None

                for candidate in candidates:
                    if candidate in state:
                        matched = candidate
                        break

                if matched is None:
                    raise RuntimeError(
                        "checkpoint does not contain %s.%s"
                        % (
                            module_name,
                            target_key,
                        )
                    )

                if tuple(
                    state[matched].shape
                ) != tuple(
                    target_state[
                        target_key
                    ].shape
                ):
                    raise RuntimeError(
                        "shape mismatch for %s: checkpoint=%s model=%s"
                        % (
                            matched,
                            tuple(
                                state[
                                    matched
                                ].shape
                            ),
                            tuple(
                                target_state[
                                    target_key
                                ].shape
                            ),
                        )
                    )

                extracted[target_key] = (
                    state[matched]
                )
                source_keys.append(
                    matched
                )

            load_result = module.load_state_dict(
                extracted,
                strict=True,
            )

            if (
                load_result.missing_keys
                or load_result.unexpected_keys
            ):
                raise RuntimeError(
                    "frontend load mismatch: %s"
                    % load_result
                )

            loaded[module_name] = (
                source_keys
            )

        return {
            "source": source,
            "loaded_modules": loaded,
            "loaded_key_count": sum(
                len(keys)
                for keys in loaded.values()
            ),
        }
