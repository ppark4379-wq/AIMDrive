from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
from torch import nn

from lavis.common.registry import registry
from lavis.models.base_model import BaseModel

from .ddia_attention import set_ddia_modality_ids
from .multimodal_sequence import MultimodalSequence, build_multimodal_sequence
from .tinyllama_backbone import (
    DEFAULT_TINYLLAMA_CHECKPOINT,
    load_tinyllama,
    parameter_report,
    resolve_torch_dtype,
)
from .visual_token_interface import VisualTokenBatch, last_valid_indices


@dataclass
class VLDriveOutput:
    predicted_waypoints: torch.Tensor
    predicted_end_logits: torch.Tensor
    hidden_states: torch.Tensor
    sequence: MultimodalSequence
    debug: Optional[dict] = None


@registry.register_model("vldrive")
class VLDriveModel(BaseModel):
    """TinyLLaMA + DDIA driving head, consuming completed CCDP/MEFA tokens."""

    PRETRAINED_MODEL_CONFIG_DICT = {
        "tinyllama_ddia": "configs/models/vldrive/tinyllama_ddia.yaml",
    }

    def __init__(self, tinyllama_model_name_or_path=DEFAULT_TINYLLAMA_CHECKPOINT,
                 tinyllama_cache_dir=None, local_files_only=False, freeze_tinyllama=False,
                 use_ddia=True, visual_feature_dim=256, use_visual_projection=True,
                 max_instruction_length=64, debug_vldrive=False, precision="fp16",
                 torch_dtype=None):
        super().__init__()
        if torch_dtype is None:
            torch_dtype = resolve_torch_dtype(precision)
        components = load_tinyllama(
            tinyllama_model_name_or_path, tinyllama_cache_dir, local_files_only,
            freeze_tinyllama, use_ddia, torch_dtype
        )
        self.tokenizer = components.tokenizer
        self.tinyllama = components.model
        hidden_size = self.tinyllama.config.hidden_size
        language_dtype = self.tinyllama.get_input_embeddings().weight.dtype
        if visual_feature_dim != hidden_size:
            if not use_visual_projection:
                raise ValueError("visual_feature_dim differs from TinyLLaMA hidden size but projection is disabled")
            self.visual_to_language_projection = nn.Linear(visual_feature_dim, hidden_size).to(dtype=language_dtype)
        else:
            self.visual_to_language_projection = nn.Identity()
        # LMDrive-compatible MLP: every selected frame token predicts five (x,y) waypoints.
        self.waypoints_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 10)
        ).to(dtype=language_dtype)
        self.end_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 2)
        ).to(dtype=language_dtype)
        self.max_instruction_length = max_instruction_length
        self.use_ddia = use_ddia
        self.debug_vldrive = debug_vldrive
        self.precision = precision

    def _tokenize(self, instructions: Union[Sequence[str], dict], device):
        if isinstance(instructions, dict):
            return {key: value.to(device) for key, value in instructions.items()}
        if (not isinstance(instructions, (list, tuple)) or not instructions
                or any(not isinstance(text, str) or not text.strip() for text in instructions)):
            raise ValueError("VLDrive requires nonempty navigation instructions")
        encoded = self.tokenizer(list(instructions), padding=True, truncation=False, return_tensors="pt")
        if encoded["input_ids"].shape[1] > self.max_instruction_length:
            raise ValueError("Instruction exceeds max_instruction_length; refusing semantic truncation")
        return {key: value.to(device) for key, value in encoded.items()}

    def forward(self, instructions, visual_tokens, visual_attention_mask=None,
                waypoint_token_indices=None, return_debug=False, predict_per_frame=False):
        visual_batch = VisualTokenBatch(visual_tokens, visual_attention_mask).flattened()
        device = visual_batch.visual_tokens.device
        encoded = self._tokenize(instructions, device)
        if (encoded["attention_mask"].sum(dim=1) == 0).any():
            raise ValueError("DDIA requires at least one valid instruction token per sample")
        if encoded["attention_mask"].shape[0] != visual_tokens.shape[0]:
            raise ValueError("Instruction and visual batch sizes differ")
        if bool((encoded["attention_mask"].sum(1) > self.max_instruction_length).any()):
            raise ValueError("Instruction exceeds max_instruction_length")
        instruction_embeddings = self.tinyllama.get_input_embeddings()(encoded["input_ids"])
        language_dtype = self.tinyllama.get_input_embeddings().weight.dtype
        visual_embeddings = self.visual_to_language_projection(
            visual_batch.visual_tokens.to(dtype=language_dtype)
        )
        sequence = build_multimodal_sequence(
            instruction_embeddings, encoded["attention_mask"], visual_embeddings,
            visual_batch.visual_attention_mask
        )
        if bool((sequence.attention_mask.sum(1) > self.tinyllama.config.max_position_embeddings).any()):
            raise ValueError("Visual window exceeds TinyLlama context length")
        set_ddia_modality_ids(self.tinyllama, sequence.modality_ids)
        # Driving needs decoder states, not vocabulary logits or every layer's states.
        outputs = self.tinyllama.model(
            inputs_embeds=sequence.inputs_embeds,
            attention_mask=sequence.attention_mask,
            position_ids=sequence.position_ids,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state
        if predict_per_frame:
            if visual_tokens.ndim != 4 or waypoint_token_indices is not None:
                raise ValueError("Per-frame readout requires [B,T,K,D] and no explicit indices")
            batch, frames, retained = visual_tokens.shape[:3]
            mask = visual_batch.visual_attention_mask.reshape(batch, frames, retained)
            offsets = torch.arange(frames, device=device) * retained
            indices = last_valid_indices(mask) + offsets + sequence.visual_start_index
            waypoint_hidden = hidden_states[torch.arange(batch, device=device)[:, None], indices]
        else:
            if waypoint_token_indices is None:
                # Padding can occur BETWEEN frames. Count-1 is not a token index.
                waypoint_token_indices = sequence.visual_start_index + last_valid_indices(
                    visual_batch.visual_attention_mask)
            indices = waypoint_token_indices.to(device=device)
            if indices.dtype != torch.long:
                raise ValueError("Waypoint indices must be int64")
            if indices.ndim == 1 and indices.shape[0] == hidden_states.shape[0]:
                rows, columns = torch.arange(hidden_states.shape[0], device=device), indices
            elif indices.ndim == 2 and indices.shape[1] == 2 and indices.shape[0] > 0:
                rows, columns = indices[:, 0], indices[:, 1]
            else:
                raise ValueError("Waypoint indices must be [B] or [num_predictions,2]")
            if (bool((rows < 0).any()) or bool((rows >= hidden_states.shape[0]).any())
                    or bool((columns < 0).any()) or bool((columns >= hidden_states.shape[1]).any())):
                raise ValueError("Waypoint indices are outside the sequence")
            if not bool(sequence.modality_ids[rows, columns].eq(1).all()):
                raise ValueError("Waypoint readout must select valid visual tokens")
            waypoint_hidden = hidden_states[rows, columns]
        debug = None
        if self.debug_vldrive or return_debug:
            report = parameter_report(self)
            debug = {
                "instruction_tokens": encoded["attention_mask"].sum(dim=1).tolist(),
                "visual_tokens": visual_batch.visual_attention_mask.sum(dim=1).tolist(),
                "total_sequence_length": hidden_states.shape[1],
                "attention_mask_shape": tuple(sequence.attention_mask.shape),
                "inputs_embeds_shape": tuple(sequence.inputs_embeds.shape),
                "hidden_states_shape": tuple(hidden_states.shape),
                "trainable_percentage": 100.0 * report["trainable"] / report["total"],
            }
        predicted_waypoints = self.waypoints_predictor(waypoint_hidden)
        predicted_end_logits = self.end_predictor(waypoint_hidden)
        if debug is not None:
            debug["predicted_waypoint_shape"] = tuple(predicted_waypoints.shape)
        return VLDriveOutput(predicted_waypoints, predicted_end_logits, hidden_states, sequence, debug)

    @classmethod
    def from_config(cls, cfg):
        model = cls(
            tinyllama_model_name_or_path=cfg.get("tinyllama_model_name_or_path", DEFAULT_TINYLLAMA_CHECKPOINT),
            tinyllama_cache_dir=cfg.get("tinyllama_cache_dir"),
            local_files_only=cfg.get("local_files_only", False),
            freeze_tinyllama=cfg.get("freeze_tinyllama", False),
            use_ddia=cfg.get("use_ddia", True),
            visual_feature_dim=cfg.get("visual_feature_dim", 256),
            use_visual_projection=cfg.get("use_visual_projection", True),
            max_instruction_length=cfg.get("max_instruction_length", 64),
            debug_vldrive=cfg.get("debug_vldrive", False),
            precision=cfg.get("precision", "fp16"),
        )
        # Loading the pretrained language checkpoint above is sufficient unless
        # an additional driving checkpoint is explicitly requested.
        if cfg.get("load_finetuned", False) or cfg.get("load_pretrained", False):
            model.load_checkpoint_from_config(cfg)
        return model
