"""LMDrive language-conditioned driving model (stage 2, instruction fine-tuning).

For a clip of T frames the model
  1. runs the frozen vision encoder on every frame -> [B*T, 106, 256] visual tokens,
  2. compresses each frame into a few query tokens with a Q-Former that also
     reads the navigation instruction,
  3. projects those tokens into the LLaMA embedding space and interleaves them
     with the instruction text (and an optional notice instruction),
  4. reads the LLaMA hidden state at the last token of every frame and predicts
     5 future waypoints plus an "instruction completed" flag for that frame.

Ported from lavis/models/drive_models/drive.py (Blip2VicunaDrive, BSD-3-Clause,
Copyright salesforce.com, inc. and the LMDrive authors).  Parameter names are
unchanged, so released LMDrive checkpoints load directly.
"""
import importlib.util
import logging
import math

import torch
import torch.nn as nn
from transformers import LlamaTokenizer

from .llama import LlamaForCausalLM
from .qformer import BertConfig, BertLMHeadModel
from .vision_encoder import create_vision_encoder

logger = logging.getLogger(__name__)

SENSOR_KEYS = ("rgb_front", "rgb_left", "rgb_right", "rgb_rear", "rgb_center", "lidar", "num_points", "velocity")


class LayerNorm(nn.LayerNorm):
    """LayerNorm computed in float32 and cast back, safe under fp16 autocast."""

    def forward(self, x):
        orig_type = x.dtype
        return super().forward(x.type(torch.float32)).type(orig_type)


def disabled_train(self, mode=True):
    """Replaces a frozen module's train() so that model.train() keeps it in eval mode."""
    return self


def _accelerate_available():
    # transformers accepts low_cpu_mem_usage=True only when accelerate is installed.
    return importlib.util.find_spec("accelerate") is not None


def _strip_module_prefix(state_dict):
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}


class LMDriveModel(nn.Module):
    def __init__(
        self,
        vision_encoder,
        vision_encoder_checkpoint="",
        llm_model_path="",
        qformer_bert_path="bert-base-uncased",
        num_query_tokens=4,
        max_text_length=64,
        use_frame_position_prompt=False,
        use_notice_prompt=False,
        freeze_vision_encoder=True,
        freeze_vision_encoder_decoder=True,
        use_gru_waypoint_decoder=False,
        use_lora=False,
        vision_encoder_chunks=2,
    ):
        super().__init__()
        self.use_extra_prompt = use_frame_position_prompt
        self.use_notice_prompt = use_notice_prompt
        self.freeze_decoder_of_visual_encoder = freeze_vision_encoder_decoder
        self.has_gru_decoder = use_gru_waypoint_decoder
        self.has_lora = use_lora
        self.split_section_num_for_visual_encoder = vision_encoder_chunks

        # 1. Vision encoder that emits [N, 106, 256] tokens.  A checkpoint replaces
        #    every weight, so the ImageNet backbone download is skipped in that case.
        self.visual_encoder = create_vision_encoder(
            vision_encoder,
            return_feature=True,
            pretrained_backbone=bool(vision_encoder.get("pretrained_backbone", True)) and not vision_encoder_checkpoint,
        )
        self.ln_vision = LayerNorm(self.visual_encoder.num_features)
        if vision_encoder_checkpoint:
            checkpoint = torch.load(vision_encoder_checkpoint, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.visual_encoder.load_state_dict(_strip_module_prefix(state_dict), strict=True)
        else:
            logger.warning("No vision_encoder_checkpoint given: the vision encoder is not pre-trained")

        if freeze_vision_encoder:
            for name, param in self.visual_encoder.named_parameters():
                if self.freeze_decoder_of_visual_encoder or "decoder" not in name:
                    param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logger.info("freeze vision encoder")

        # 2. Language model.
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model_path, use_fast=False, truncation_side="left")
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=_accelerate_available())
        self.llm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.llm_tokenizer.add_special_tokens({"bos_token": "</s>"})
        self.llm_tokenizer.add_special_tokens({"eos_token": "</s>"})
        self.llm_tokenizer.add_special_tokens({"unk_token": "</s>"})
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        hidden_size = self.llm_model.config.hidden_size

        # 3. Prediction heads on the LLM hidden state of each frame.
        if self.has_gru_decoder:
            self.waypoints_fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 64),
            )
            self.waypoints_predictor = nn.GRUCell(input_size=2, hidden_size=64)
            self.waypoints_output = nn.Linear(64, 2)
        else:
            self.waypoints_predictor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 10),
            )
        self.end_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

        # 4. Q-Former that turns 106 visual tokens into num_query_tokens tokens per frame.
        logger.info("Loading Q-Former")
        self.Qformer, self.query_tokens = self.init_qformer(
            qformer_bert_path, num_query_tokens, self.visual_encoder.num_features)
        self.Qformer.resize_token_embeddings(len(self.llm_tokenizer))
        self.Qformer.cls = None

        if self.has_lora:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm_model = get_peft_model(self.llm_model, lora_config)
            self.llm_model.print_trainable_parameters()
        else:
            for param in self.llm_model.parameters():
                param.requires_grad = False

        self.llm_proj = nn.Linear(self.Qformer.config.hidden_size, hidden_size)
        self.max_txt_len = max_text_length
        self.waypoints_loss = torch.nn.L1Loss()
        self.end_loss = torch.nn.CrossEntropyLoss()

    @staticmethod
    def init_qformer(bert_path, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained(bert_path)
        encoder_config.encoder_width = vision_width
        # insert a cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        qformer = BertLMHeadModel.from_pretrained(bert_path, config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return qformer, query_tokens

    @staticmethod
    def maybe_autocast(dtype=torch.float16):
        return torch.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available())

    def concat_text_image_input(self, input_embeds, input_atts, image_embeds, image_nums,
                                end_flag_pos_list, image_atts=None):
        """Place the frame tokens right after each sample's instruction tokens.

        attention mask: 1 for tokens that are attended, 0 for padding.
        """
        input_part_targets_len = []
        llm_inputs = []
        llm_attention_mask = []
        wp_target_index = []
        bs = image_embeds.size()[0]
        for i in range(bs):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            if image_atts is None:
                bs, t, n, dim = image_embeds.size()
                llm_inputs.append(torch.cat([
                    input_embeds[i][:this_input_ones],
                    image_embeds[i].view(t * n, -1),
                    input_embeds[i][this_input_ones:],
                ]))
                llm_attention_mask.append(torch.cat([
                    input_atts[i][:this_input_ones],
                    torch.ones((image_nums[i] * n), device=image_embeds.device, dtype=torch.long),
                    torch.zeros(((t - image_nums[i]) * n), device=image_embeds.device, dtype=torch.long),
                    input_atts[i][this_input_ones:],
                ]))
            else:
                llm_inputs.append(torch.cat([
                    input_embeds[i][:this_input_ones],
                    image_embeds[i],
                    input_embeds[i][this_input_ones:],
                ]))
                llm_attention_mask.append(torch.cat([
                    input_atts[i][:this_input_ones],
                    image_atts[i],
                    input_atts[i][this_input_ones:],
                ]))
            for j in end_flag_pos_list[i]:
                wp_target_index.append([i, j + this_input_ones])
        llm_inputs = torch.stack(llm_inputs, 0)
        llm_attention_mask = torch.stack(llm_attention_mask, 0)
        return llm_inputs, llm_attention_mask, input_part_targets_len, wp_target_index

    def concat_text_image_input_with_notice(self, input_embeds, input_atts, image_embeds, image_nums,
                                            end_flag_pos_list, notice_frame_id, notice_text, image_atts=None):
        """Like concat_text_image_input, but inserts the notice text after notice_frame_id frames.

        notice_frame_id < 0 means the clip has no notice.
        """
        if image_atts is not None:
            raise NotImplementedError("use_notice_prompt cannot be combined with use_frame_position_prompt")
        input_part_targets_len = []
        llm_inputs = []
        llm_attention_mask = []
        wp_target_index = []
        bs = image_embeds.size()[0]

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"
        text_input_tokens = self.llm_tokenizer(
            notice_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image_embeds.device)
        input_notice_atts = text_input_tokens.attention_mask
        notice_embeds = self.llm_model.get_input_embeddings()(text_input_tokens.input_ids)

        for i in range(bs):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            this_notice_input_ones = input_notice_atts[i].sum()
            bs, t, n, dim = image_embeds.size()
            if notice_frame_id[i] <= 0:
                llm_inputs.append(torch.cat([
                    input_embeds[i][:this_input_ones],
                    image_embeds[i].view(t * n, -1),
                    input_embeds[i][this_input_ones:],
                    notice_embeds[i][:],
                ]))
            else:
                llm_inputs.append(torch.cat([
                    input_embeds[i][:this_input_ones],
                    image_embeds[i, :notice_frame_id[i]].view(notice_frame_id[i] * n, -1),
                    notice_embeds[i][:this_notice_input_ones],
                    image_embeds[i, notice_frame_id[i]:].view((t - notice_frame_id[i]) * n, -1),
                    input_embeds[i][this_input_ones:],
                    notice_embeds[i][this_notice_input_ones:],
                ]))
            if notice_frame_id[i] < 0:
                llm_attention_mask.append(torch.cat([
                    input_atts[i][:this_input_ones],
                    torch.ones((image_nums[i] * n), device=image_embeds.device, dtype=torch.long),
                    torch.zeros(((t - image_nums[i]) * n), device=image_embeds.device, dtype=torch.long),
                    torch.zeros((input_notice_atts.size(1)), device=image_embeds.device, dtype=torch.long),
                    input_atts[i][this_input_ones:],
                ]))
            else:
                llm_attention_mask.append(torch.cat([
                    input_atts[i][:this_input_ones],
                    torch.ones((image_nums[i] * n), device=image_embeds.device, dtype=torch.long),
                    input_notice_atts[i][:this_notice_input_ones],
                    torch.zeros(((t - image_nums[i]) * n), device=image_embeds.device, dtype=torch.long),
                    input_atts[i][this_input_ones:],
                    input_notice_atts[i][this_notice_input_ones:],
                ]))
            for j in range(len(end_flag_pos_list[i])):
                # frames after the notice are shifted by the notice length
                if j < notice_frame_id[i] or notice_frame_id[i] < 0:
                    wp_target_index.append([i, end_flag_pos_list[i][j] + this_input_ones])
                else:
                    wp_target_index.append([i, end_flag_pos_list[i][j] + this_input_ones + this_notice_input_ones])
        llm_inputs = torch.stack(llm_inputs, 0)
        llm_attention_mask = torch.stack(llm_attention_mask, 0)
        return llm_inputs, llm_attention_mask, input_part_targets_len, wp_target_index

    @staticmethod
    def build_gt_waypoints(waypoints, valid_frames):
        return torch.cat([waypoints[i, :valid_frames[i]] for i in range(waypoints.size(0))], dim=0)

    @staticmethod
    def build_gt_end_flags(valid_frames):
        gt_end_flags = []
        for i in range(len(valid_frames)):
            gt_end_flags.extend([0] * (valid_frames[i] - 1))
            gt_end_flags.append(1)
        return torch.tensor(gt_end_flags, device=valid_frames.device).long()

    def prompt_wrap(self, image_embeds, text_before_img, text_after_img, valid_frames):
        """Wrap every frame as '<frame x,y>' [frame tokens] '</frame>' (use_frame_position_prompt)."""
        bs, t, n, dim = image_embeds.size()
        emb_list = []
        end_flag_pos_list = []
        for i in range(bs):
            before_texts = text_before_img[i].split("|")
            after_texts = text_after_img[i].split("|")
            temp_embeds = []
            temp_end_flag_pos_list = []
            for j in range(valid_frames[i]):
                p_before_tokens = self.llm_tokenizer(
                    before_texts[j], return_tensors="pt", add_special_tokens=False).to(image_embeds.device)
                p_after_tokens = self.llm_tokenizer(
                    after_texts[j], return_tensors="pt", add_special_tokens=False).to(image_embeds.device)
                p_before_embed = self.llm_model.get_input_embeddings()(p_before_tokens.input_ids)
                p_after_embed = self.llm_model.get_input_embeddings()(p_after_tokens.input_ids)
                p_embed = torch.cat([p_before_embed, image_embeds[i][j][None], p_after_embed], dim=1)
                temp_embeds.append(p_embed)
                temp_end_flag_pos_list.append(p_embed.size(1) - 1)
            end_flag_pos_list.append(temp_end_flag_pos_list)
            emb_list.append(torch.cat(temp_embeds, dim=1))
        emb_lens = [emb.shape[1] for emb in emb_list]
        pad_emb = self.llm_model.get_input_embeddings()(
            torch.tensor(self.llm_tokenizer.pad_token_id, device=image_embeds.device))
        wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
        wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.long, device=image_embeds.device)
        for i, emb in enumerate(emb_list):
            wrapped_embs[i, :emb_lens[i]] = emb
            wrapped_atts[i, :emb_lens[i]] = 1
        return wrapped_embs, wrapped_atts, end_flag_pos_list

    def split_data(self, samples):
        """Split the flattened frames into vision_encoder_chunks pieces to bound GPU memory."""
        total = samples["rgb_front"].size(0)
        split_size = max(1, math.ceil(total / self.split_section_num_for_visual_encoder))
        pieces = {key: torch.split(samples[key], split_size, dim=0) for key in SENSOR_KEYS}
        return [{key: pieces[key][i] for key in SENSOR_KEYS} for i in range(len(pieces["rgb_front"]))]

    def forward(self, samples, inference_mode=False, image_embeds=None):
        samples = dict(samples)  # the sensor tensors are reshaped below; keep the caller's batch intact
        if image_embeds is None:
            device = samples["rgb_front"].device
            bs = samples["rgb_front"].size(0)
            t = samples["rgb_front"].size(1)
            for key in SENSOR_KEYS:
                shape = samples[key].size()
                samples[key] = samples[key].view(bs * t, *shape[2:])

            if self.freeze_decoder_of_visual_encoder:
                with torch.no_grad():
                    with self.maybe_autocast():
                        image_embeds = torch.cat(
                            [self.visual_encoder(chunk) for chunk in self.split_data(samples)], dim=0)
            else:
                with self.maybe_autocast():
                    image_embeds = self.visual_encoder(samples)
        else:
            # pre-computed visual tokens [B, T, 106, C]
            device = image_embeds.device
            bs = image_embeds.size(0)
            t = image_embeds.size(1)
            image_embeds = image_embeds.view(bs * t, *image_embeds.size()[2:])

        image_embeds = self.ln_vision(image_embeds)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        text_qformer = self.llm_tokenizer(
            [text for text in samples["text_input"] for _ in range(t)],
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
        qformer_atts = torch.cat([query_atts, text_qformer.attention_mask], dim=1)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        query_output = self.Qformer.bert(
            text_qformer.input_ids,
            attention_mask=qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = self.llm_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
        image_embeds = image_embeds.view(bs, t, *image_embeds.size()[1:])

        if self.use_extra_prompt:
            image_embeds, image_atts, end_flag_pos_list = self.prompt_wrap(
                image_embeds, samples["text_before_img"], samples["text_after_img"], samples["valid_frames"])
        else:
            image_atts = None
            end_flag_pos_list = []
            n_length = image_embeds.size(2)  # tokens per frame
            for i in range(bs):
                end_flag_pos_list.append([n_length * (j + 1) - 1 for j in range(samples["valid_frames"][i])])

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"
        text_input_tokens = self.llm_tokenizer(
            samples["text_input"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(device)
        inputs_embeds = self.llm_model.get_input_embeddings()(text_input_tokens.input_ids)

        if self.use_notice_prompt:
            llm_inputs, llm_attention_mask, _, wp_target_index = self.concat_text_image_input_with_notice(
                inputs_embeds, text_input_tokens.attention_mask, image_embeds, samples["valid_frames"],
                end_flag_pos_list, samples["notice_frame_id"], samples["notice_text"], image_atts)
        else:
            llm_inputs, llm_attention_mask, _, wp_target_index = self.concat_text_image_input(
                inputs_embeds, text_input_tokens.attention_mask, image_embeds, samples["valid_frames"],
                end_flag_pos_list, image_atts)
        wp_target_index = torch.tensor(wp_target_index, device=device).long()

        with self.maybe_autocast():
            hidden_states = self.llm_model(
                inputs_embeds=llm_inputs,
                attention_mask=llm_attention_mask,
                return_dict=False,
            )

        if self.has_gru_decoder:
            output_wp = []
            _, n_tokens, _ = hidden_states.size()
            x = torch.zeros(size=(bs * n_tokens, 2), dtype=hidden_states.dtype).to(device)
            waypoints_feature = self.waypoints_fc(hidden_states.reshape(-1, self.llm_model.config.hidden_size))
            for _ in range(5):
                waypoints_feature = self.waypoints_predictor(x, waypoints_feature)
                dx = self.waypoints_output(waypoints_feature)
                x = dx + x
                output_wp.append(x)
            predicted_waypoints = torch.cat(output_wp, dim=1).view(bs, n_tokens, 10)
        else:
            predicted_waypoints = self.waypoints_predictor(hidden_states)
        predicted_waypoints = predicted_waypoints[wp_target_index[:, 0], wp_target_index[:, 1]]
        predicted_end_prob = self.end_predictor(hidden_states)
        predicted_end_prob = predicted_end_prob[wp_target_index[:, 0], wp_target_index[:, 1]]
        if inference_mode:
            return predicted_waypoints, predicted_end_prob

        gt_waypoints = self.build_gt_waypoints(samples["local_future_waypoints"], samples["valid_frames"])
        waypoints_loss = self.waypoints_loss(predicted_waypoints, gt_waypoints)
        gt_end_flags = self.build_gt_end_flags(samples["valid_frames"])
        end_loss = self.end_loss(predicted_end_prob, gt_end_flags)
        predicted_end = torch.argmax(predicted_end_prob, dim=1)
        end_acc = (predicted_end == gt_end_flags).float().mean().item()
        loss = waypoints_loss + end_loss * 0.2
        return {"loss": loss, "waypoints_loss": waypoints_loss, "end_loss": end_loss, "end_acc": end_acc}

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        """AdamW groups: no weight decay for biases and 1-D parameters."""
        parameter_group_vars = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.0
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if "visual_encoder" in name:
                group_name = "vit_layer_%s" % group_name
            if group_name not in parameter_group_vars:
                parameter_group_vars[group_name] = {"weight_decay": this_weight_decay, "params": [], "lr_scale": 1}
            parameter_group_vars[group_name]["params"].append(param)
        return list(parameter_group_vars.values())

    @classmethod
    def from_config(cls, cfg):
        vision_encoder = cfg.get("vision_encoder")
        if not isinstance(vision_encoder, dict):
            raise ValueError("model.vision_encoder must hold the vision encoder config; "
                             "set vision_encoder_config in the model YAML")
        return cls(
            vision_encoder=vision_encoder,
            vision_encoder_checkpoint=cfg.get("vision_encoder_checkpoint") or "",
            llm_model_path=cfg["llm_model_path"],
            qformer_bert_path=cfg.get("qformer_bert_path", "bert-base-uncased"),
            num_query_tokens=int(cfg.get("num_query_tokens", 4)),
            max_text_length=int(cfg.get("max_text_length", 64)),
            use_frame_position_prompt=bool(cfg.get("use_frame_position_prompt", False)),
            use_notice_prompt=bool(cfg.get("use_notice_prompt", False)),
            freeze_vision_encoder=bool(cfg.get("freeze_vision_encoder", True)),
            freeze_vision_encoder_decoder=bool(cfg.get("freeze_vision_encoder_decoder", True)),
            use_gru_waypoint_decoder=bool(cfg.get("use_gru_waypoint_decoder", False)),
            use_lora=bool(cfg.get("use_lora", False)),
            vision_encoder_chunks=int(cfg.get("vision_encoder_chunks", 2)),
        )
