"""Checkpoint contracts and sensor-token inference; no simulator transmission."""
from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import threading
import time

import numpy as np
import torch

from utils.common.runtime import deterministic_pointpillar_reductions, sha256
from utils.common.source_compatibility import source_matches, source_path

ROOT = Path(__file__).resolve().parents[1]
FEATURE_MODEL = 'memfuser_baseline_e1d3_return_feature'
HORIZONS = [.5, 1., 1.5, 2., 2.5]
CONVENTION = 'lmdrive_left_negative_forward'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def local_path(value):
    return (ROOT / value).resolve()


def seed_runtime(seed):
    """Called once at startup, never once per frame."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_sources(provenance):
    """Permit exact reviewed inference-compatible revisions, never whole files."""
    # Exact old/new hashes: do not exempt entire files from source verification.
    # 2026-09-22: Gumbel evaluation, isolated seeded RNG, evaluation CLI wiring.
    evaluation_changes = {
        'utils/trainer.py': (
            'f8443704f25aaba3fbe0759d810de88194aff864f08c4797bbbcc5952c7f73e8',
            'd2f088735f0232101716365d1cd1178d27b0c88b02e493b935333b76ca687e2c'),
        'train.py': (
            'bca46c2afdb3d0e1b89a1b802971b99f08458c59ff38d2ac744aaa04065c6638',
            '954443cf18999d36914d9d0b94ca374566a4fd28436aae53d8a78dbc6b3c031e'),
    }
    # Reviewed 9b60831 against its parent (the CRLF files match this run's
    # recorded hashes). These revisions do not change cached Stage-2 forward:
    # dataset child/verify_seal helpers are unchanged; its new admission checks
    # and equivalent target transform belong to SensorRecord training. The VE
    # training entry point only adds checkpoint metadata. Visualization retains
    # legacy axes by default. Cache recipe/model checks below remain mandatory.
    inference_compatible_changes = {
        'datasets/morai_sensor_record_dataset.py': (
            '9af9b8b1186a6b3158a880094faf0df8abb6cee27d8efd22d595d5e07b96caf5',
            'a626e2f87253570975fe726b6f7ecc19f0421c9036344f164c571d46e06c457e',
            'reviewed SensorRecord admission checks; cached inference helpers unchanged'),
        'utils/morai_transfer_training.py': (
            'cf0e8b89f74ef435dba10aeb318bb3efa48de432ad9b5b3837e72f39e55359e7',
            '775d3318afc8552571027dd9cc00952e50fc4c245db714386c990d153f186336',
            'VE training metadata export only; not executed by cached inference'),
        'utils/vision_encoder_visualization.py': (
            '5c9e02e8828266961264c8f7073845abae75199ea276eecbc4463837d15de8fd',
            'e2e4e1d0bad31a53ace0b453ef6ee23af2ee653f5aefa50bf893d10520ce389c',
            'explicit SensorRecord plot axes; default legacy VLDrive axes preserved'),
    }
    changes = []
    for name, expected in provenance['source'].items():
        raw = source_path(ROOT, name).read_bytes()
        if source_matches(ROOT, name, expected):
            if hashlib.sha256(raw).hexdigest() != expected:
                changes.append(name + ': approved utility-package relocation')
            continue
        actual = hashlib.sha256(raw).hexdigest()
        if actual == expected:
            continue
        if evaluation_changes.get(name) == (expected, actual):
            changes.append(name + ': Gumbel evaluation only; model and training loss unchanged')
            continue
        reviewed = inference_compatible_changes.get(name)
        if reviewed is not None and reviewed[:2] == (expected, actual):
            changes.append(name + ': ' + reviewed[2])
            continue
        if name == 'models/vldrive.py':
            original = raw.replace(
                b'def predict(self, inputs, *, compute_auxiliary=None, token_selection_mode=None):\n',
                b'def predict(self, inputs, *, compute_auxiliary=None):\r\n').replace(
                b'connected = self.connector(features.float(), selection_mode=token_selection_mode,\n'
                b'                                   compute_reconstruction=compute_auxiliary)\n',
                b'connected = self.connector(features.float(), compute_reconstruction=compute_auxiliary)\r\n')
            if hashlib.sha256(original).hexdigest() == expected:
                changes.append(name + ': optional runtime selection override only')
                continue
        raise ValueError('Training source changed outside the authorized inference API: ' + name)
    return changes


def load_contract(config):
    checkpoint = local_path(config['stage2_checkpoint'])
    run = checkpoint.parent
    training = read_json(run / 'resolved_config.json')
    provenance = read_json(run / 'provenance.json')
    description = read_json(run / 'model.json')
    changes = verify_sources(provenance)
    cache = local_path(config['visual_cache'])
    meta = read_json(cache / 'cache_meta.json')
    if any(value['cache_meta'] != meta for value in provenance['datasets'].values()):
        raise ValueError('Visual cache differs from stage-2 training provenance')
    for name, field in [('manifest.jsonl', 'manifest_sha256'),
                        ('features.jsonl', 'features_manifest_sha256'),
                        ('visual_source.json', 'visual_source_sha256')]:
        if sha256(cache / name) != meta[field]:
            raise ValueError('Changed cache provenance: ' + name)
    identity = read_json(cache / 'visual_source.json')
    if (identity['encoder_contract'] != meta['encoder_contract']
            or identity['checkpoint_sha256'] != meta['checkpoint_sha256']
            or identity['inference_precision'] != 'float32'
            or identity['storage_precision'] != 'float16'
            or meta['trajectory_coordinate_convention'] != CONVENTION):
        raise ValueError('Unsupported visual precision/coordinate contract')
    for name, digest in identity['sources_sha256'].items():
        if not source_matches(ROOT, name, digest):
            raise ValueError('Visual cache implementation changed: ' + name)
    if (training['sequence_length'] != 11 or description['config']['memory_size'] != 10
            or description['config']['input_mode'] != 'cached'
            or description['config']['visual_sha256'] != meta['checkpoint_sha256']):
        raise ValueError('Expected cached-input 11-frame model from this encoder')
    records = [json.loads(line) for line in (cache / 'features.jsonl').read_text().splitlines()]
    deltas = [(b['sensor_source_ns'] - a['sensor_source_ns']) / 1e9
              for a, b in zip(records, records[1:]) if a['episode_id'] == b['episode_id']]
    interval = float(np.median(deltas)) * training['frame_sampling_interval']
    if not np.isclose(1 / config['runtime']['target_hz'], interval, rtol=.01):
        raise ValueError(f'target_hz must preserve the training history interval: {interval}s')
    return dict(checkpoint=checkpoint, training=training, provenance=provenance,
                description=description, cache=cache, meta=meta, identity=identity,
                history_interval_s=interval, source_compatibility=changes,
                training_complete=(run / 'report.json').is_file())


def load_stage2(contract, device):
    from models.vldrive import VLDriveTrainingConfig, VLDriveTrainingModel
    from training.training import restore_model_checkpoint
    model_config = VLDriveTrainingConfig(**contract['description']['config'])
    for field in ('language_checkpoint', 'qformer_checkpoint'):
        if not Path(getattr(model_config, field)).is_dir():
            raise ValueError('Checkpoint asset path is unavailable. Use the original WSL training environment: '
                             + getattr(model_config, field))
    # Recreate any frozen initialization exactly, independently of inference RNG.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(contract['training']['seed'])
        model = VLDriveTrainingModel(model_config, device=device)
    for key, value in [('language_asset', model.language_manifest), ('qformer_asset', model.qformer_manifest)]:
        if contract['description'][key] != value:
            raise ValueError('Pretrained asset provenance mismatch: ' + key)
    checkpoint = contract['checkpoint']
    sidecar = checkpoint.with_suffix(checkpoint.suffix + '.sha256.json')
    if not sidecar.is_file():
        sidecar = checkpoint.with_suffix('.sha256.json')
    digest_before = read_json(sidecar)
    stat_before = checkpoint.stat()
    state = restore_model_checkpoint(checkpoint, model, provenance=contract['provenance'])
    stat_after = checkpoint.stat()
    if (digest_before != read_json(sidecar)
            or (stat_before.st_ino, stat_before.st_mtime_ns, stat_before.st_size)
            != (stat_after.st_ino, stat_after.st_mtime_ns, stat_after.st_size)):
        raise ValueError('Checkpoint was replaced during loading; retry after the epoch save finishes')
    if state['model_config'] != asdict(model_config):
        raise ValueError('Checkpoint model_config mismatch')
    information = {key: state[key] for key in ('kind', 'schema', 'updates', 'metrics',
                                               'trainable_hash', 'frozen_hash') if key in state}
    information.update(path=str(contract['checkpoint']), training_complete=contract['training_complete'],
                       source_compatibility=contract['source_compatibility'], file_digest=digest_before)
    del state
    model.eval()
    return model, information


def load_encoder(config, contract, device):
    from models.vision_encoder import create_vision_encoder
    path = local_path(config['stage1_checkpoint'])
    if sha256(path) != contract['meta']['checkpoint_sha256']:
        raise ValueError('Stage-1 checkpoint differs from the training cache')
    blob = torch.load(path, map_location='cpu', weights_only=True)
    if blob['morai_encoder_contract'] != contract['meta']['encoder_contract']:
        raise ValueError('Stage-1 preprocessing recipe mismatch')
    model = create_vision_encoder({'name': FEATURE_MODEL}, pretrained_backbone=False)
    model.load_state_dict(blob['state_dict'], strict=True)
    del blob
    model.requires_grad_(False).to(device).eval()
    return model


def synchronize(device):
    if torch.device(device).type == 'cuda':
        # Wait only for this thread's stream; another independent model may run.
        torch.cuda.current_stream(device).synchronize()


@torch.inference_mode()
def encode_frames(model, inputs):
    """Encode a bounded sensor batch; each output remains a separate real frame."""
    device = inputs['rgb_front'].device
    batch = inputs['rgb_front'].shape[0]
    if batch not in (1, 2):
        raise ValueError('Live VE batches must contain one or two real frames')
    synchronize(device)
    started = time.perf_counter()
    with deterministic_pointpillar_reductions():
        feature = model(inputs)
    synchronize(device)
    elapsed = (time.perf_counter() - started) * 1000
    if feature.shape != (batch, 106, 256) or not bool(torch.isfinite(feature).all()):
        raise ValueError('Vision encoder must return finite [B,106,256] features')
    # Cache writes float16 and the dataset reads it as float32. Reproduce that
    # quantization boundary; the encoder itself always computes in float32.
    feature = feature.to(torch.float16).float()
    if not bool(torch.isfinite(feature).all()):
        raise ValueError('Feature overflow at the cache float16 storage boundary')
    return feature, elapsed


@torch.inference_mode()
def encode_frame(model, inputs):
    if inputs['rgb_front'].shape[0] != 1:
        raise ValueError('encode_frame requires one frame')
    features, elapsed = encode_frames(model, inputs)
    return features[0], elapsed


class TokenHistory:
    """Thread-safe real frames; reject repeats, reset across discontinuities."""
    def __init__(self, interval_s=.2, tolerance_s=.06):
        if interval_s <= 0 or not 0 < tolerance_s < interval_s:
            raise ValueError('Invalid history timing')
        self.interval = interval_s
        self.tolerance = tolerance_s
        self.frames = deque(maxlen=11)
        self.lock = threading.Lock()
        self.resets = 0

    def clear(self):
        with self.lock:
            if self.frames:
                self.frames.clear()
                self.resets += 1

    def accepts(self, stamp, ids):
        with self.lock:
            if not self.frames:
                return True
            last = self.frames[-1]
            if stamp <= last['stamp'] or any(ids[k] <= last['ids'][k] for k in ids):
                return False
            return stamp - last['stamp'] >= self.interval - self.tolerance

    def expire(self, *, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.frames and now-self.frames[-1]['stamp'] > self.interval+self.tolerance:
                self.frames.clear()
                self.resets += 1

    def append(self, feature, *, stamp, ids, vision_ms=0., context=None):
        if feature.shape != (106, 256) or not bool(torch.isfinite(feature).all()):
            raise ValueError('History expects one finite [106,256] real feature')
        if not self.accepts(stamp, ids):
            return False
        with self.lock:
            if self.frames and stamp - self.frames[-1]['stamp'] > self.interval + self.tolerance:
                self.frames.clear()
                self.resets += 1
            owned=feature.detach().clone()
            ready=None
            if owned.is_cuda:
                ready=torch.cuda.Event()
                ready.record(torch.cuda.current_stream(owned.device))
            self.frames.append(dict(feature=owned, ready=ready, stamp=float(stamp),
                                    ids=dict(ids), vision_ms=float(vision_ms),
                                    context=context, generation=self.resets))
        return True

    def snapshot(self, *, materialize=True):
        with self.lock:
            rows = list(self.frames)
        if len(rows) < 11 or not materialize:
            return None, rows
        return self.materialize(rows), rows

    @staticmethod
    def materialize(rows):
        if len(rows)!=11:
            raise ValueError('A complete real 11-frame window is required')
        for row in rows:
            feature=row['feature']
            if feature.is_cuda:
                stream=torch.cuda.current_stream(feature.device)
                stream.wait_event(row['ready'])
                # The producer owns the allocation, but the consumer uses it.
                feature.record_stream(stream)
        return torch.stack([row['feature'] for row in rows]).unsqueeze(0)


@torch.inference_mode()
def predict_window(model, features, instruction, *, optimized=False):
    from utils.training.training_helper import autocast
    if model.training:
        raise ValueError('Inference requires model.eval()')
    if features.shape != (1, 11, 106, 256):
        raise ValueError('Stage-2 input must be [1,11,106,256]')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('navigation_instruction must be a nonempty string')
    synchronize(features.device)
    started = time.perf_counter()
    with autocast(model):
        if optimized:
            from inference.fast_window import predict_latest
            output = predict_latest(model, features, instruction)
        else:
            output = model.predict({'visual_feature': features, 'instruction': [instruction]},
                                   compute_auxiliary=False, token_selection_mode='gumbel')
    synchronize(features.device)
    elapsed = (time.perf_counter() - started) * 1000
    expected_shape = (1, 1 if optimized else 11, 5, 2)
    if output['trajectory'].shape != expected_shape or not bool(torch.isfinite(output['trajectory']).all()):
        raise ValueError(f'Expected finite {expected_shape} waypoint predictions')
    if not bool((output['selected_count'] >= 1).all()):
        raise ValueError('Every frame must retain at least one token')
    latest = output['trajectory'][0, -1].float().cpu()
    return dict(waypoints_raw=latest.tolist(),
                waypoints_lateral_left_longitudinal_forward=torch.stack([latest[:, 0], -latest[:, 1]], -1).tolist(),
                horizons_s=HORIZONS, coordinate_convention=CONVENTION, units='m',
                selected_tokens=int(output['selected_count'][0, -1]),
                selected_tokens_history=output['selected_count'][0].cpu().tolist(),
                vldrive_ms=elapsed)
