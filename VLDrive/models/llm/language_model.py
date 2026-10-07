from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
from torch import nn


from .ddia_attention import set_ddia_modality_ids
from .multimodal_sequence import MultimodalSequence, build_multimodal_sequence, build_frame_multimodal_sequence
from .tinyllama_backbone import (
    DEFAULT_TINYLLAMA_CHECKPOINT,
    load_tinyllama,
    parameter_report,
    resolve_torch_dtype,
)
from ..connector.visual_token_interface import VisualTokenBatch, last_valid_indices


@dataclass
class VLDriveOutput:
    predicted_waypoints: torch.Tensor
    predicted_end_logits: torch.Tensor
    hidden_states: torch.Tensor
    sequence: MultimodalSequence
    debug: Optional[dict] = None


class VLDriveModel(nn.Module):
    """TinyLLaMA + DDIA driving head, consuming completed CCDP/MEFA tokens."""

    PRETRAINED_MODEL_CONFIG_DICT = {
        "tinyllama_ddia": "configs/models/vldrive/tinyllama_ddia.yaml",
    }

    def __init__(self, tinyllama_model_name_or_path=DEFAULT_TINYLLAMA_CHECKPOINT,
                 tinyllama_cache_dir=None, local_files_only=False, freeze_tinyllama=False,
                 use_ddia=True, visual_feature_dim=256, use_visual_projection=True,
                 max_instruction_length=64, debug_vldrive=False, precision="fp16",
                 torch_dtype=None, path_navigation=False, path_width=256, path_heads=4,
                 path_fusion='cross_attention', situation_instruction=False, max_situation_tokens=128,
                 zone_instruction=False):
        super().__init__()
        if path_fusion not in ('cross_attention', 'mlp'):
            raise ValueError('path_fusion must be cross_attention or mlp')
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
        self.path_navigation = path_navigation
        self.path_fusion = path_fusion
        self.situation_instruction = situation_instruction
        self.zone_instruction = zone_instruction
        self.max_situation_tokens = max_situation_tokens
        if situation_instruction and not use_ddia:
            raise ValueError('Frame Situation instructions require frame-aware DDIA')
        if zone_instruction:
            if not use_ddia:
                raise ValueError('Frame Zone instructions require frame-aware DDIA')
            from .zone_embedding import ZoneEmbedding
            # A separate LLM module; do not mutate pretrained Situation weights
            # or shift the initialization of the existing frozen driving heads.
            with torch.random.fork_rng(devices=[]):
                self.zone_embedding = ZoneEmbedding(hidden_size).to(dtype=language_dtype)
        if path_navigation:
            if not use_ddia:
                raise ValueError('Frame path navigation requires frame-aware DDIA')
            # Do not shift the RNG stream used by existing heads/connector.
            # Their frozen initialization fingerprints must survive warm start.
            with torch.random.fork_rng(devices=[]):
                if path_fusion == 'cross_attention':
                    from instruction.global_path.tokenizer.path_tokenizer import PathTokenizer
                    from instruction.navigation.navigation_cross_attention import NavigationCrossAttention
                    self.path_tokenizer = PathTokenizer(path_width).to(dtype=language_dtype)
                    self.navigation_cross_attention = NavigationCrossAttention(hidden_size, path_width, path_heads).to(dtype=language_dtype)
                else:
                    from instruction.navigation.navigation_mlp import NavigationMLP
                    self.navigation_mlp = NavigationMLP(hidden_size, path_width).to(dtype=language_dtype)
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

    def _tokenize(self, instructions: Union[Sequence[str], dict], device, *, max_length=None):
        if isinstance(instructions, dict):
            return {key: value.to(device) for key, value in instructions.items()}
        if (not isinstance(instructions, (list, tuple)) or not instructions
                or any(not isinstance(text, str) or not text.strip() for text in instructions)):
            raise ValueError("VLDrive requires nonempty navigation instructions")
        encoded = self.tokenizer(list(instructions), padding=True, truncation=False, return_tensors="pt")
        limit = self.max_instruction_length if max_length is None else max_length
        if encoded["input_ids"].shape[1] > limit:
            raise ValueError("Instruction exceeds token limit; refusing semantic truncation")
        return {key: value.to(device) for key, value in encoded.items()}

    def forward(self, instructions, visual_tokens, visual_attention_mask=None,
                waypoint_token_indices=None, return_debug=False, predict_per_frame=False,
                path_features=None, situation_instructions=None, zone_ids=None):
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
        if self.path_navigation:
            if path_features is None or visual_tokens.ndim != 4:
                raise ValueError('Path navigation requires per-frame features and visual[B,T,K,D]')
            if path_features['path_valid'].shape != visual_tokens.shape[:2]:
                raise ValueError('Path and visual frames differ')
            if self.path_fusion == 'cross_attention':
                path_tokens = self.path_tokenizer(path_features)
                navigation = self.navigation_cross_attention(instruction_embeddings, path_tokens,
                    encoded['attention_mask'], path_features['path_valid'])
            else:
                navigation = self.navigation_mlp(instruction_embeddings, path_features, encoded['attention_mask'])
        else:
            if path_features is not None:
                raise ValueError('Path features supplied while path_navigation is disabled')

        frame_conditioning = self.path_navigation or self.situation_instruction or self.zone_instruction
        if frame_conditioning:
            if visual_tokens.ndim != 4:
                raise ValueError('Frame instructions require visual tokens [B,T,K,D]')
            b, t = visual_tokens.shape[:2]
            nav_mask = encoded['attention_mask'][:, None].expand(-1, t, -1)
            if not self.path_navigation:
                navigation = instruction_embeddings[:, None].expand(-1, t, -1, -1)
            frame_instructions, frame_mask = navigation, nav_mask

        if self.zone_instruction:
            if (not isinstance(zone_ids, torch.Tensor) or zone_ids.shape != (b, t)
                    or zone_ids.dtype not in (torch.int32, torch.int64)
                    or zone_ids.device != device or bool(((zone_ids < -1) | (zone_ids > 5)).any())):
                raise ValueError('Zone IDs must be integer [B,T] on the visual device, with -1 unknown or 0..5')
            zone_valid = zone_ids >= 0
            zone = self.zone_embedding(zone_ids.clamp_min(0)).unsqueeze(2)
            zone = zone.masked_fill(~zone_valid[..., None, None], 0)
            frame_instructions = torch.cat((frame_instructions, zone), dim=2)
            frame_mask = torch.cat((frame_mask, zone_valid.unsqueeze(-1)), dim=2)
        elif zone_ids is not None:
            raise ValueError('Zone IDs supplied while zone_instruction is disabled')

        if self.situation_instruction:
            if visual_tokens.ndim != 4:
                raise ValueError('Situation text requires per-frame visual tokens [B,T,K,D]')
            b, t = visual_tokens.shape[:2]
            if (not isinstance(situation_instructions, (list, tuple)) or len(situation_instructions) != b
                    or any(not isinstance(row, (list, tuple)) or len(row) != t for row in situation_instructions)):
                raise ValueError('Situation instructions must be batch-major [B][T] strings or None')
            # Both GT and predicted sentences use the same tokenizer/embedding.
            # The numeric navigation MLP is applied only to navigation tokens.
            from instruction.situation.text import prompt_text
            sentences = [sentence for row in situation_instructions for sentence in row]
            normalized = [prompt_text(sentence) for sentence in sentences]
            present = [i for i, text in enumerate(normalized) if text is not None]
            if present:
                situation = self._tokenize([normalized[i] for i in present], device,
                                           max_length=self.max_situation_tokens)
                embedded = self.tinyllama.get_input_embeddings()(situation['input_ids'])
                indices = torch.tensor(present, device=device, dtype=torch.long)
                # Absent frames have no active Situation tokens, including BOS.
                situation_embeddings = embedded.new_zeros((b*t, *embedded.shape[1:])).index_copy(0, indices, embedded)
                situation_mask = situation['attention_mask'].new_zeros((b*t, embedded.shape[1])).index_copy(
                    0, indices, situation['attention_mask'])
            else:
                # Avoid tokenization and extra sequence slots for all-empty batches.
                situation_embeddings = instruction_embeddings.new_zeros((b*t, 0, instruction_embeddings.shape[-1]))
                situation_mask = encoded['attention_mask'].new_zeros((b*t, 0))
            situation_embeddings = situation_embeddings.reshape(b, t, -1, instruction_embeddings.shape[-1])
            frame_instructions = torch.cat((frame_instructions, situation_embeddings), dim=2)
            frame_mask = torch.cat((frame_mask, situation_mask.reshape(b, t, -1)), dim=2)
        else:
            if situation_instructions is not None:
                raise ValueError('Situation text supplied while situation_instruction is disabled')

        if frame_conditioning:
            sequence = build_frame_multimodal_sequence(frame_instructions, frame_mask,
                visual_embeddings.reshape(*visual_tokens.shape[:3], -1),
                visual_batch.visual_attention_mask.reshape(visual_tokens.shape[:3]))
        else:
            sequence = build_multimodal_sequence(
                instruction_embeddings, encoded["attention_mask"], visual_embeddings,
                visual_batch.visual_attention_mask)
        if bool((sequence.attention_mask.sum(1) > self.tinyllama.config.max_position_embeddings).any()):
            raise ValueError("Visual window exceeds TinyLlama context length")
        set_ddia_modality_ids(self.tinyllama, sequence.modality_ids, sequence.frame_ids)
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
            indices = (sequence.frame_readout_indices if sequence.frame_readout_indices is not None else
                       last_valid_indices(mask) + offsets + sequence.visual_start_index)
            waypoint_hidden = hidden_states[torch.arange(batch, device=device)[:, None], indices]
        else:
            if waypoint_token_indices is None:
                # Padding can occur BETWEEN frames. Count-1 is not a token index.
                waypoint_token_indices = (sequence.frame_readout_indices[:, -1] if sequence.frame_readout_indices is not None else
                    sequence.visual_start_index + last_valid_indices(visual_batch.visual_attention_mask))
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
                "path_fusion": self.path_fusion if self.path_navigation else 'disabled',
                "situation_instruction": self.situation_instruction,
                "zone_instruction": self.zone_instruction,
            }
        predicted_waypoints = self.waypoints_predictor(waypoint_hidden)
        predicted_end_logits = self.end_predictor(waypoint_hidden)
        if debug is not None:
            debug["predicted_waypoint_shape"] = tuple(predicted_waypoints.shape)
        return VLDriveOutput(predicted_waypoints, predicted_end_logits, hidden_states, sequence, debug)
