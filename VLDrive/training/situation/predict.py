"""Read frozen VE caches and decode Situation predictions without using GT inputs."""
from collections import OrderedDict
import io
import json
from pathlib import Path
import threading

import numpy as np
import torch
from PIL import Image

from ...instruction.situation.text import load_catalog, render_situation, text_contract
from ...data.src.loaders.situation_dataset import (
    ROOT, CACHE_SCHEMA, child, jsonl, labeled_episodes, load_config, project_path, sha256,
)
from ...models.llm.situation_head import SituationHead
from ...models.llm.zone_embedding import ZONE_NAMES, ZONE_SCHEMA
from .labels import FIELDS, situation_targets
from .trainer import CHECKPOINT_FORMAT


def local_path(value):
    """Accept a Windows path pasted into the WSL-hosted viewer."""
    text = str(value).strip().strip('"').replace('\\', '/')
    if len(text) > 2 and text[1:3] == ':/' and Path('/mnt').is_dir():
        text = '/mnt/' + text[0].lower() + '/' + text[3:]
    return Path(text).expanduser().resolve()


def decode_prediction(logits, vocab, catalog):
    """Only classifier outputs fill templates; no labels or input Zone fallback."""
    predictions = {}
    for name in FIELDS:
        if name not in logits or not vocab.get(name):
            predictions[name] = dict(value=None, confidence=None)
            continue
        scores = logits[name][0].float().softmax(-1).cpu()
        index = int(scores.argmax())
        predictions[name] = dict(value=vocab[name][index], confidence=float(scores[index]))
    identifier = predictions['instruction']['value']
    rendered = render_situation(identifier, {k: v['value'] for k, v in predictions.items()}, catalog)
    return dict(fields=predictions, **rendered)


class CachedSituationReview:
    def __init__(self, config_path, checkpoint_path, cache_path=None, device='cpu'):
        self.cfg = load_config(local_path(config_path))
        self.root = local_path(cache_path) if cache_path else project_path(self.cfg['cache_output'])
        self.checkpoint_path = local_path(checkpoint_path)
        self.device = torch.device(device)
        self.catalog = load_catalog()
        self.lock = threading.Lock()
        self.tokens, self.predictions, self.episode_pages = OrderedDict(), {}, {}
        self.image_cache = OrderedDict()
        self.meta = json.loads((self.root / 'cache_meta.json').read_text(encoding='utf-8'))
        if (self.meta.get('schema') != CACHE_SCHEMA or self.meta.get('feature_kind') != 'vision_encoder'
                or self.meta.get('zone_schema') != ZONE_SCHEMA
                or self.meta.get('checkpoint_sha256') != self.cfg['checkpoint_sha256']):
            raise ValueError('VE cache/checkpoint/Zone contract mismatch')
        for name, key in [('manifest.jsonl', 'manifest_sha256'), ('source.json', 'source_sha256')]:
            if sha256(self.root / name) != self.meta[key]:
                raise ValueError('Changed cache file: ' + name)
        before = self.checkpoint_path.stat()
        checkpoint_sha = sha256(self.checkpoint_path)
        blob = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
        after = self.checkpoint_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Checkpoint is being saved; retry when saving completes')
        if (blob.get('format') != CHECKPOINT_FORMAT or blob.get('zone_schema') != ZONE_SCHEMA
                or blob.get('zone_names') != list(ZONE_NAMES)):
            raise ValueError('Expected Situation Head checkpoint with the current Zone schema')
        source = blob.get('feature_source') or {}
        if (source.get('feature_kind') != 'vision_encoder'
                or source.get('checkpoint_sha256') != self.meta['checkpoint_sha256']
                or source.get('encoder_contract') != self.meta.get('encoder_contract')):
            raise ValueError('Situation Head was trained with a different VE/input contract')
        self.vocab = blob['vocabulary']
        if (set(self.vocab) != set(FIELDS) or blob['model_config']['class_counts'] !=
                {k: len(v) for k, v in self.vocab.items()} or blob['model_config']['visual_dim'] != 256):
            raise ValueError('Checkpoint classifier vocabulary or VE dimensions mismatch')
        self.model = SituationHead(**blob['model_config'])
        self.model.load_state_dict(blob['model'], strict=True)
        self.model.requires_grad_(False).to(self.device).eval()
        self.info = dict(checkpoint=str(self.checkpoint_path), checkpoint_sha256=checkpoint_sha,
                         epoch=blob['epoch'], selection=blob.get('selection_metric'), device=str(self.device),
                         cache=str(self.root), sequence_length=self.cfg['sequence_length'],
                         situation_text_contract=text_contract())
        self.training_labels = source.get('labels_sha256', {})
        del blob
        self.cached = {}
        for record in jsonl(self.root / 'manifest.jsonl'):
            key = (record['episode_id'], record['frame_index'])
            if key in self.cached:
                raise ValueError('Duplicate cached frame')
            self.cached[key] = record
        self.episodes = labeled_episodes(self.cfg['datasets'])
        for episode in self.episodes:
            if self.meta['labels_sha256'].get(episode['identity']) != episode['label_sha256']:
                raise ValueError('Cache and label file do not match: ' + episode['identity'])

    def summary(self):
        return dict(self.info, episodes=[dict(id=i, name=e['identity'], frames=len(e['rows']),
                    training_data=e['identity'] in self.training_labels,
                    labels_revised=(e['identity'] in self.training_labels and
                                    self.training_labels[e['identity']] != e['label_sha256']))
                    for i, e in enumerate(self.episodes)])

    def _episode(self, episode_id):
        if not 0 <= episode_id < len(self.episodes):
            raise ValueError('Unknown episode')
        return self.episodes[episode_id]

    def window(self, episode_id, index):
        episode = self._episode(episode_id)
        if not 0 <= index < len(episode['rows']):
            raise ValueError('Unknown frame')
        length = self.cfg['sequence_length']
        if index < length-1:
            return None, f'이력 부족: {index+1}/{length} 프레임'
        rows = episode['rows'][index-length+1:index+1]
        if any(r.get('zone') is None for r in rows):
            return None, '입력 이력에 미지정 Zone이 있음'
        if any(b['frame_id'] != a['frame_id']+1 or not 0 < b['sensor_time_ns']-a['sensor_time_ns']
               <= self.cfg['max_frame_gap_ms']*1e6 for a, b in zip(rows, rows[1:])):
            return None, '입력 프레임의 시간/번호가 연속적이지 않음'
        records = [self.cached.get((episode['identity'], r['frame_id'])) for r in rows]
        if any(r is None for r in records):
            return None, '입력 이력에 VE 캐시가 없는 프레임이 있음'
        if any(r['sensor_source_ns'] != row['sensor_time_ns'] for r, row in zip(records, rows)):
            raise ValueError('Token/frame timestamp mismatch')
        return (rows, records), None

    def inputs(self, episode_id, index):
        window, reason = self.window(episode_id, index)
        if window is None:
            return None, reason
        rows, records = window
        features = []
        for record in records:
            ref = record['visual_feature']
            key = ref['path']
            if key not in self.tokens:
                path = child(self.root, key)
                if sha256(path) != ref['sha256']:
                    raise ValueError('Changed VE token: ' + key)
                value = np.load(path, allow_pickle=False)
                if value.shape != (106, 256) or not np.isfinite(value).all():
                    raise ValueError('Invalid VE features')
                self.tokens[key] = torch.from_numpy(value.astype(np.float32))
                if len(self.tokens) > 256:
                    self.tokens.popitem(last=False)
            self.tokens.move_to_end(key)
            features.append(self.tokens[key])
        anchor_ns = rows[-1]['sensor_time_ns']
        inputs = dict(visual_tokens=torch.stack(features).unsqueeze(0),
                      visual_mask=torch.ones(1, len(rows), 106, dtype=torch.bool),
                      timestamps=torch.tensor([[(r['sensor_time_ns']-anchor_ns)/1e9 for r in rows]]),
                      zone_ids=torch.tensor([[r['zone_instruction'] for r in rows]], dtype=torch.long))
        return {k: v.to(self.device) for k, v in inputs.items()}, None

    def predict(self, episode_id, index):
        with self.lock:
            key = (episode_id, index)
            if key in self.predictions:
                return self.predictions[key]
            inputs, reason = self.inputs(episode_id, index)
            if inputs is None:
                result = dict(available=False, reason=reason)
            else:
                # GT is accessed only after forward and template decoding.
                with torch.inference_mode():
                    output = self.model(**inputs)
                if not torch.isfinite(output['situation_embedding']).all() or any(
                        not torch.isfinite(v).all() for v in output['logits'].values()):
                    raise ValueError('Nonfinite Situation prediction')
                result = dict(available=True, **decode_prediction(output['logits'], self.vocab, self.catalog),
                              fs_shape=list(output['situation_embedding'].shape))
                row = self._episode(episode_id)['rows'][index]
                gt = row['labeling']
                valid_gt = (row.get('frame_valid') and gt.get('instruction_consistency') == 'consistent'
                            and gt.get('zone_id') in ZONE_NAMES)
                targets = situation_targets(gt)[0] if valid_gt else {k: None for k in FIELDS}
                result['comparison'] = {k: dict(gt=targets[k], prediction=result['fields'][k]['value'],
                    correct=(result['fields'][k]['value'] == targets[k]) if targets[k] is not None else None,
                    unseen=targets[k] is not None and targets[k] not in self.vocab[k]) for k in FIELDS}
                matches = [r['correct'] for r in result['comparison'].values() if r['correct'] is not None]
                result['all_correct'] = all(matches) if matches else None
            self.predictions[key] = result
            return result

    def episode_page(self, episode_id):
        if episode_id in self.episode_pages:
            return self.episode_pages[episode_id]
        episode = self._episode(episode_id)
        rows = episode['rows']
        right = {}
        features_path = episode['path'] / 'situation_labels/features/frame_features.jsonl'
        # Preserve the original viewer's optional route-relative lateral trace.
        if features_path.is_file():
            from MoraiDrive.scenario_runner.tools import static_lane_compliance as lane
            features = jsonl(features_path)
            xy = [(f['index'], f.get('ego_world_x_m'), f.get('ego_world_y_m')) for f in features]
            xy = [(i, x, y) for i, x, y in xy if x is not None and y is not None]
            if xy and lane.DEFAULT_ROUTE.is_file():
                _, lateral = lane.route_frame(np.array([(x, y) for _, x, y in xy]),
                                               np.loadtxt(lane.DEFAULT_ROUTE)[:, :2], 0)
                right = {i: float(-lat) for (i, _, _), lat in zip(xy, lateral)}
        result = []
        for index, row in enumerate(rows):
            record = self.cached.get((episode['identity'], row['frame_id']), {})
            gt = row['labeling']
            _, reason = self.window(episode_id, index)
            result.append(dict(index=index, i=row['frame_id'], t=(row['sensor_time_ns']-rows[0]['sensor_time_ns'])/1e9,
                zone=row.get('zone'), zone_instruction=row.get('zone_instruction'),
                state=gt.get('instruction_id'), text=gt.get('instruction'), consistency=gt.get('instruction_consistency'),
                label=gt.get('label'), right=right.get(row['frame_id']), speed=record.get('velocity_mps'),
                reason=reason, available=reason is None, variables=gt.get('instruction_variables', {})))
        page = dict(rows=result, name=episode['identity'],
                    training_data=episode['identity'] in self.training_labels,
                    labels_revised=(episode['identity'] in self.training_labels and
                                    self.training_labels[episode['identity']] != episode['label_sha256']))
        self.episode_pages[episode_id] = page
        return page

    def image(self, episode_id, index, view):
        cameras = dict(front='camera_1', left='camera_2', right='camera_3', rear='camera_4')
        if view not in cameras:
            raise ValueError('Unknown camera')
        episode = self._episode(episode_id)
        if not 0 <= index < len(episode['rows']):
            raise ValueError('Unknown frame')
        key = (episode_id, index, view)
        # HTTP image requests can overlap; the cache is a small optional accelerator.
        with self.lock:
            if key in self.image_cache:
                self.image_cache.move_to_end(key)
                return self.image_cache[key]
        asset = episode['rows'][index]['assets'].get(cameras[view])
        if not asset:
            raise FileNotFoundError('Camera asset unavailable')
        path = child(episode['native'], asset['relative_path'])
        with Image.open(path) as source:
            image = source.convert('RGB')
            image.thumbnail((640, 480) if view == 'front' else (320, 240))
            stream = io.BytesIO()
            image.save(stream, format='JPEG', quality=80)
        data = stream.getvalue()
        with self.lock:
            self.image_cache[key] = data
            if len(self.image_cache) > 128:
                self.image_cache.popitem(last=False)
        return data
