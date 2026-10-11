"""Causal Situation predictions, rendered with the Joint training templates."""
from collections import OrderedDict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from .dataset import sha256


class SituationPredictor:
    def __init__(self, checkpoint, cache_meta, text_contract, renderer_path, *, device='cuda', length=11):
        import torch
        from VLDrive.models.llm.situation_head import SituationHead
        from VLDrive.models.llm.zone_embedding import ZONE_NAMES, ZONE_SCHEMA
        self.torch, self.device, self.length = torch, torch.device(device), length
        self.checkpoint = Path(checkpoint).resolve()
        before = self.checkpoint.stat()
        digest = sha256(self.checkpoint)
        state = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        after = self.checkpoint.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Situation checkpoint changed during loading')
        if (state.get('format') != 'situation_head_v2_situation_only'
                or state.get('zone_schema') != ZONE_SCHEMA or state.get('zone_names') != list(ZONE_NAMES)):
            raise ValueError('Unexpected Situation checkpoint/Zone format')
        source = state['feature_source']
        if (source.get('feature_kind') != 'vision_encoder'
                or source.get('checkpoint_sha256') != cache_meta['checkpoint_sha256']
                or source.get('encoder_contract') != cache_meta['encoder_contract']):
            raise ValueError('Situation and Joint cache VE input contracts differ')
        if 'readout' in state['model_config']:
            raise ValueError('This Situation checkpoint requires its original readout implementation')
        self.vocab = state['vocabulary']
        if state['model_config']['class_counts'] != {k: len(v) for k, v in self.vocab.items()}:
            raise ValueError('Situation vocabulary/classifier dimensions differ')
        self.model = SituationHead(**state['model_config'])
        self.model.load_state_dict(state['model'], strict=True)
        self.model.requires_grad_(False).to(self.device).eval()
        renderer_path = Path(renderer_path).resolve()
        if sha256(renderer_path) != text_contract['renderer_sha256']:
            raise ValueError('Situation renderer differs from Joint training')
        spec = spec_from_file_location('_offline_joint_situation_renderer', renderer_path)
        self.renderer = module_from_spec(spec)
        spec.loader.exec_module(self.renderer)
        if sha256(self.renderer.CATALOG_PATH) != text_contract['catalog_sha256']:
            raise ValueError('Situation sentence catalog differs from Joint training')
        self.catalog = self.renderer.load_catalog()
        for identifier in self.vocab['instruction']:
            zone, name = identifier.split(':', 1)
            if name not in self.catalog.get(zone, {}):
                raise ValueError('Situation classifier label is absent from the Joint text catalog')
        self.info = dict(path=str(self.checkpoint), sha256=digest, epoch=state['epoch'],
                         sequence_length=length, frame_interval=1, active_zone_ids=[1],
                         catalog_sha256=text_contract['catalog_sha256'])
        self.cache = OrderedDict()

    def predict(self, dataset, identity, position):
        key = identity, position
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        result = self._predict(dataset, identity, position)
        self.cache[key] = result
        while len(self.cache) > 2048:
            self.cache.popitem(last=False)
        return result

    def _predict(self, dataset, identity, position):
        import numpy as np
        torch = self.torch
        current = dataset.episode(identity)['document']['frames'][position]
        if current['zone_id'] != 1:
            return dict(text=None, reason='non_static_zone_situation_disabled', fields={})
        frames, reason = dataset.window(identity, position, length=self.length, interval=1, require_path=False)
        if frames is None:
            return dict(text=None, reason=reason, fields={})
        if any(f['zone_id'] < 0 for f in frames):
            return dict(text=None, reason='unknown_zone_in_situation_history', fields={})
        times = [f['sensor_source_ns'] for f in frames]
        visual = np.stack([dataset.array(dataset.visual[(identity, t)]['visual_feature'], (106, 256)) for t in times])
        return self.predict_features(visual, [t / 1e9 for t in times], [f['zone_id'] for f in frames])

    def predict_features(self, visual, times, zones):
        """The same classifier/renderer for replay and live VE observations."""
        import numpy as np
        torch = self.torch
        visual = np.asarray(visual)
        if visual.shape != (self.length, 106, 256) or len(times) != self.length or len(zones) != self.length:
            raise ValueError('Situation input must contain the trained 11 observed frames')
        if zones[-1] != 1:
            return dict(text=None, reason='non_static_zone_situation_disabled', fields={})
        inputs = dict(visual_tokens=torch.from_numpy(visual).unsqueeze(0).to(self.device),
                      visual_mask=torch.ones(1, self.length, 106, dtype=torch.bool, device=self.device),
                      timestamps=torch.tensor([[t-times[-1] for t in times]], device=self.device),
                      zone_ids=torch.tensor([zones], dtype=torch.long, device=self.device))
        with torch.inference_mode():
            output = self.model(**inputs)
        if not torch.isfinite(output['situation_embedding']).all():
            raise ValueError('Nonfinite Situation embedding')
        fields = {}
        for name, vocab in self.vocab.items():
            if not vocab:
                fields[name] = dict(value=None, confidence=None)
                continue
            logits = output['logits'][name][0].float()
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite Situation classifier output')
            probabilities = logits.softmax(-1).cpu()
            index = int(probabilities.argmax())
            fields[name] = dict(value=vocab[index], confidence=float(probabilities[index]))
        identifier = fields['instruction']['value']
        zone, name = identifier.split(':', 1)
        template = self.catalog[zone][name]
        variables = {k: v['value'] for k, v in fields.items()}
        missing = [k for k in template['required_variables'] if not variables.get(k)]
        if missing:
            return dict(text=None, reason='missing_predicted_slots:' + ','.join(missing), fields=fields)
        text, _, _ = self.renderer.render_template(template, variables)
        return dict(text=' '.join(text.split()), reason=None, fields=fields,
                    embedding_shape=list(output['situation_embedding'].shape))

    def window_predictions(self, dataset, identity, position):
        frames, reason = dataset.window(identity, position)
        if frames is None:
            return None, reason
        start = position - (dataset.length - 1) * dataset.interval
        # Each historical Joint frame gets its own trailing Situation window.
        # A prediction at the newest time is never copied into older frames.
        return [self.predict(dataset, identity, i) for i in range(start, position + 1, dataset.interval)], None
