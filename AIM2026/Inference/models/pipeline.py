"""Frozen live VE -> Situation text; cached-format features -> Joint model."""
from collections import deque
from contextlib import nullcontext
import io
import json

import numpy as np

from ..runtime.history import FeatureFrame, fit_context


class LiveModels:
    def __init__(self, config, sensors):
        import torch
        from PIL import Image
        from scipy.spatial.transform import Rotation
        from VLDrive.inference.checkpoint import load_joint, inspect_checkpoint
        from VLDrive.inference.situation import SituationPredictor
        from VLDrive.inference.dataset import sha256
        from data.src.loaders.morai_encoder_contract import checkpoint_encoder_contract, LiveInputAdapter
        from models.ve.frozen_vision_encoder import MoraiVisualEncoder
        from ..navigation.route import RouteNavigation, PoseProjector

        self.torch, self.Image = torch, Image
        self.device = torch.device(config['device'])
        torch.set_num_threads(config['cpu_threads'])
        state, _ = inspect_checkpoint(config['joint_checkpoint'])
        self.cache_meta = state['provenance']['datasets']['train']['cache_meta']
        if self.cache_meta.get('trajectory_coordinate_convention') != 'lmdrive_left_negative_forward':
            raise ValueError('Controller expects Joint left-negative-forward target coordinates')
        del state
        digest = sha256(config['vision_checkpoint'])
        blob = torch.load(config['vision_checkpoint'], map_location='cpu', mmap=True, weights_only=False)
        # The cache exporter canonicalizes checkpoint tuples to JSON arrays.
        self.contract = json.loads(json.dumps(checkpoint_encoder_contract(blob, digest), allow_nan=False))
        del blob
        if digest != self.cache_meta['checkpoint_sha256'] or self.contract != self.cache_meta['encoder_contract']:
            raise ValueError('Live VE weights/recipe differ from Joint training')
        if self.contract['velocity_input'] != 'causal_ego_topic_speed_mps_50ms':
            raise ValueError('This runtime supplies causally paired Competition Vehicle Status speed')
        self.joint, self.info, text_contract = load_joint(config['joint_checkpoint'], self.cache_meta,
                                                        device=self.device, asset_paths=config.get('asset_paths'))
        self.info['timing_scope'] = 'live_VE_and_joint_workers_reported_separately'
        self.vision = MoraiVisualEncoder(config['vision_checkpoint'], policy='frozen',
                                         device=self.device, expected_sha256=digest).eval()
        mount = sensors['LiDAR']
        self.adapter = LiveInputAdapter(self.contract, lidar_frame='sensor_flu', lidar_to_base=dict(
            translation_xyz=mount['pos'],
            rotation_xyzw=Rotation.from_euler('xyz', mount['rot_deg'], degrees=True).as_quat().tolist()))
        self.situation = SituationPredictor(config['situation_checkpoint'], self.cache_meta, text_contract,
                                           config['situation_renderer'], device=self.device)
        self.situation_frames = deque(maxlen=11)
        self.situation_reason = 'waiting_for_11_observations'
        self.navigation = RouteNavigation(config['navigation'], self.contract)
        self.projector = PoseProjector(self.contract['map_basis'], sensors['GPS']['pos'], sensors['IMU']['rot_deg'])
        self.camera_config = {name: sensors['Camera'][key] for name, key in (
            ('rgb_front', 'Cam_Front'), ('rgb_left', 'Cam_Left'),
            ('rgb_right', 'Cam_Right'), ('rgb_rear', 'Cam_Rear'))}
        self.max_gap_s = config['history']['max_gap_s']
        self.max_context = min(config['history']['max_context_tokens'],
                               self.joint.language_model.tinyllama.config.max_position_embeddings)
        self.joint.language_model.token_cache_size = 4096

    def stream(self):
        if self.device.type != 'cuda':
            return None
        stream = self.torch.cuda.Stream(device=self.device)
        stream.wait_stream(self.torch.cuda.default_stream(self.device))
        return stream

    def stream_context(self, stream):
        return self.torch.cuda.stream(stream) if stream is not None else nullcontext()

    def precision_context(self):
        return (self.torch.autocast(self.device.type, dtype=self.torch.bfloat16)
                if self.joint.config.precision == 'bf16' else nullcontext())

    def encode(self, frame):
        torch = self.torch
        images = {}
        for name, jpeg in frame.images.items():
            with self.Image.open(io.BytesIO(jpeg)) as image:
                geometry = self.camera_config[name]
                if image.size != (geometry['width'], geometry['height']):
                    raise ValueError('UDP camera size differs from config.yaml: ' + name)
                images[name] = np.asarray(image.convert('RGB')).copy()
        navigation = self.navigation.update(frame.pose)
        inputs = self.adapter.assemble(images, frame.lidar, velocity=frame.speed_mps,
                                       target_point=[0., 0.], rng_seed=frame.sequence)
        # The return-feature VE exits before its target-point-dependent head.
        inputs.pop('target_point')
        inputs = {k: v.unsqueeze(0).unsqueeze(0).to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            visual = self.vision(inputs)[0, 0].to(torch.float16).cpu().numpy()
        if not np.isfinite(visual).all():
            raise ValueError('Invalid VE output after training-cache FP16 conversion')
        if self.situation_frames:
            gap = frame.source_time - self.situation_frames[-1].source_time
            if gap <= 0 or gap > self.max_gap_s + 1e-7:
                self.situation_frames.clear()
        observed = FeatureFrame(frame.source_time, frame.received_at, frame.sequence,
                                frame.pose, visual, navigation)
        self.situation_frames.append(observed)
        text = None
        self.situation_reason = 'non_static_zone_situation_disabled'
        if navigation['zone_id'] == 1:
            self.situation_reason = 'waiting_for_11_observations'
            if len(self.situation_frames) == 11:
                rows = list(self.situation_frames)
                prediction = self.situation.predict_features(np.stack([r.visual for r in rows]),
                    [r.source_time for r in rows], [r.navigation['zone_id'] for r in rows])
                text, self.situation_reason = prediction['text'], prediction['reason']
        return FeatureFrame(frame.source_time, frame.received_at, frame.sequence,
                            frame.pose, visual, navigation, text)

    def predict(self, frames):
        from instruction.situation.text import prompt_text
        torch = self.torch
        language = self.joint.language_model
        instruction = frames[-1].navigation['instruction']
        if any(f.navigation['segment'] != frames[-1].navigation['segment'] for f in frames):
            raise ValueError('Mixed Navigation instruction segments')
        visual = torch.from_numpy(np.stack([f.visual for f in frames])).unsqueeze(0).to(self.device).float()
        with torch.inference_mode(), self.precision_context():
            # Cheap selector-only pass gives exact keep counts before expensive
            # Q-Former/LLM work. It has no time dependence or mutable state.
            normalized = self.joint.connector.visual_input_norm(visual)
            selected = self.joint.connector.ccdp(normalized, selection_mode=self.joint.config.inference_selection_mode)
            counts = selected['mask'].sum(-1)[0].int().cpu().tolist()
        nav_length = len(language.tokenizer(instruction)['input_ids'])
        lengths = [len(language.tokenizer(prompt_text(f.situation))['input_ids']) if f.situation else 0 for f in frames]
        frames, budget = fit_context(frames, nav_length, lengths, counts, self.max_context)
        visual = visual[:, -len(frames):]
        inputs = dict(visual_feature=visual, instruction=[instruction],
            zone_ids=torch.tensor([[f.navigation['zone_id'] for f in frames]], device=self.device),
            situation_instructions=[[f.situation for f in frames]],
            path_features={key: torch.tensor([[f.navigation['path_features'][key] for f in frames]],
                dtype=torch.bool if key == 'path_valid' else torch.float32, device=self.device)
                for key in ('ego_relation', 'path_context', 'path_valid')})
        with torch.inference_mode(), self.precision_context():
            result = self.joint.predict(inputs, compute_auxiliary=False,
                                       token_selection_mode=self.joint.config.inference_selection_mode)
        waypoints = result['trajectory'][0, -1].float().cpu().numpy()
        if waypoints.shape != (5, 2) or not np.isfinite(waypoints).all():
            raise ValueError('Invalid Joint latest-frame waypoint prediction')
        return dict(waypoints=waypoints, anchor=frames[-1], frames=len(frames),
                    padded_tokens_bound=budget, selected_counts=result['selected_count'][0].cpu().tolist())
