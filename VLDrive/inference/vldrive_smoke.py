"""Recorded-sensor A-D regression through the exact live inference functions."""
from datetime import datetime, timezone
import hashlib
import io
import json
import time

import numpy as np
from PIL import Image
import torch

from data.src.loaders.morai_sensor_record_dataset import child
from data.src.loaders.navigation_cache_dataset import NavigationCacheDataset
from data.src.loaders.navigation_sensor_record import NativeNavigationSensors
from inference.live_frame import CAMERA_KEYS, LiveFrameSource
from inference.vldrive_runtime import (TokenHistory, encode_frame, load_encoder,
    load_stage2, predict_window, read_json, seed_runtime)


def recorded_inputs(source, dataset, index, device):
    """Decode verified raw assets, then call LiveFrameSource's shared adapter."""
    scene = dataset.scenes[index]
    def read(name):
        raw = child(dataset.native, name).read_bytes()
        ref = dataset.assets[name]
        if len(raw) != ref['size_bytes'] or hashlib.sha256(raw).hexdigest() != ref['sha256']:
            raise ValueError('Raw sensor checksum mismatch: ' + name)
        return raw
    images = {}
    for key, role in zip(CAMERA_KEYS, ('camera_1', 'camera_2', 'camera_3', 'camera_4')):
        with Image.open(io.BytesIO(read(scene['camera_files'][role]))) as im:
            images[key] = np.asarray(im.convert('RGB')).copy()
    raw = read(scene['lidar_file'])
    if not raw or len(raw) % 16:
        raise ValueError('Invalid raw LiDAR file')
    sweep = np.frombuffer(raw, dtype='<f4').reshape(-1, 4).copy()
    return source.prepare_encoder_frame(images, sweep, device=device, rng_seed=index)


def run_offline(cfg, contract, output_dir, report):
    device = cfg['runtime']['device']
    tests = report['tests'] = {}
    report['stage'] = 'A_checkpoint_load'
    print('A: loading stage-2 checkpoint with frozen/trainable/provenance checks', flush=True)
    model, report['stage2'] = load_stage2(contract, device)
    tests['A'] = 'passed'
    instruction = cfg['navigation_instruction']
    dataset = NavigationCacheDataset(contract['cache'], contract['training']['instructions'],
        'validation', sequence_length=11, seed=contract['training']['seed'],
        frame_sampling_interval=contract['training']['frame_sampling_interval'])
    index = cfg['offline']['validation_window_index']
    admitted = read_json(contract['checkpoint'].parent / 'window_indices.json')['val']
    if index not in admitted:
        raise ValueError('Choose a window used by the training validation run')
    rows = dataset.windows[index]
    # Only visual references are read. Dataset.__getitem__/target are deliberately
    # not called: future labels must never enter this inference pipeline.
    cached = torch.stack([dataset._array(row['visual_feature'], (106, 256)) for row in rows]).unsqueeze(0).to(device)
    reconstruction_calls = []
    hook = model.connector.reconstruction_transformer.register_forward_pre_hook(
        lambda *args: reconstruction_calls.append(True))
    try:
        seed_runtime(cfg['seed'])
        # Paired offline comparisons replay the same RNG state intentionally;
        # the live runner never restores/reseeds between frames.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        def replay_rng():
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        report['stage'] = 'B_cached_validation'
        cached_result = predict_window(model, cached, instruction, optimized=cfg['runtime'].get('optimized_window', False))
        tests['B'] = dict(status='passed', input_shape=list(cached.shape), output_shape=[5, 2],
                          result=cached_result)
        print('B: cached validation -> [5,2] passed', flush=True)
        encoder = load_encoder(cfg, contract, device)
        identity = rows[0]['episode_id']
        root = contract['training']['dataset']
        from pathlib import Path
        parts = identity.split('/')
        candidates = [p.parent for p in Path(root).rglob('episode.json')
                      if p.parent.name == parts[-1] and p.parent.parent.name == parts[-2]]
        if len(candidates) != 1:
            raise ValueError('Expected exactly one recorded episode matching ' + identity)
        episode = candidates[0]
        native = NativeNavigationSensors(episode, contract['meta']['encoder_contract'], parts[0])
        if native.provenance != contract['identity']['source'][identity]:
            raise ValueError('Recorded source seal differs from the visual cache')
        source = LiveFrameSource({}, open_receivers=False, encoder_recipe=contract['meta']['encoder_contract'])
        history = TokenHistory(contract['history_interval_s'], cfg['runtime']['history_tolerance_s'])
        differences, latencies = [], []
        report['stage'] = 'C_recorded_frame_encoder_regression'
        for n, row in enumerate(rows):
            frame_index = row['frame_index']
            if native.times[frame_index] != row['sensor_source_ns']:
                raise ValueError('Raw/cache frame identity mismatch')
            inputs = recorded_inputs(source, native, frame_index, device)
            if n == 0:
                reference = native.inputs(frame_index)
                for key, value in inputs.items():
                    torch.testing.assert_close(value.cpu(), reference[key].unsqueeze(0), atol=0, rtol=0)
            feature, elapsed = encode_frame(encoder, inputs)
            expected = cached[0, n]
            diff = (feature - expected).abs()
            comparison = dict(frame_index=frame_index, max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                              rms=float((feature - expected).square().mean().sqrt()))
            differences.append(comparison)
            torch.testing.assert_close(feature, expected, atol=cfg['offline']['token_atol'],
                                       rtol=cfg['offline']['token_rtol'])
            assert history.append(feature, stamp=row['sensor_source_ns'] / 1e9,
                                  ids={'recorded': frame_index}, vision_ms=elapsed)
            latencies.append(elapsed)
            window, _ = history.snapshot()
            if n < 10 and window is not None:
                raise AssertionError('Inference history became ready before eleven real frames')
            if n == 0:
                tests['C'] = dict(status='passed', encoder_shape=[1, 106, 256],
                                  shared_preprocessing_exact=True, comparison=comparison)
                report['stage'] = 'D_recorded_sequence_waypoints'
            print(f'C/D: recorded real frame {n+1}/11 | token max error {comparison["max_abs"]:.6g}', flush=True)
        raw_window, _ = history.snapshot()
        replay_rng()
        raw_result = predict_window(model, raw_window, instruction, optimized=cfg['runtime'].get('optimized_window', False))
        a = torch.tensor(cached_result['waypoints_raw'])
        b = torch.tensor(raw_result['waypoints_raw'])
        torch.testing.assert_close(a, b, atol=cfg['offline']['waypoint_atol_m'], rtol=cfg['offline']['waypoint_rtol'])
        if cached_result['selected_tokens_history'] != raw_result['selected_tokens_history']:
            raise AssertionError('Paired cache/raw selection counts differ')
        replay_rng()
        repeat_result = predict_window(model, raw_window, instruction, optimized=cfg['runtime'].get('optimized_window', False))
        torch.testing.assert_close(torch.tensor(raw_result['waypoints_raw']),
                                   torch.tensor(repeat_result['waypoints_raw']), atol=0, rtol=0)
        if reconstruction_calls:
            raise AssertionError('Reconstruction transformer executed during inference')
        tests['D'] = dict(status='passed', input_shape=list(raw_window.shape), trajectory_shape=[1,11,5,2],
            latest_shape=[5,2], window_frame_indices=[r['frame_index'] for r in rows],
            timestamps_ns=[r['sensor_source_ns'] for r in rows], token_comparison=differences,
            waypoint_max_abs_m=float((a-b).abs().max()), same_rng_reproducible=True,
            reconstruction_forward_calls=0, minimum_selected_tokens=min(raw_result['selected_tokens_history']),
            no_future_targets_passed_to_model=True)
        record = dict(raw_result, timestamp=datetime.now(timezone.utc).isoformat(),
                      source='recorded_validation_sensors', frame=rows[-1]['frame_index'],
                      instruction=instruction, history_ready=True, vision_encoder_ms=latencies[-1],
                      total_inference_ms=latencies[-1]+raw_result['vldrive_ms'])
        (output_dir / 'waypoints.jsonl').write_text(json.dumps(record, allow_nan=False)+'\n', encoding='utf-8')
        report.update(status='offline_passed', outputs=1, latest=record,
            latency=dict(vision_encoder_ms=latencies, vision_median_ms=float(np.median(latencies[1:])),
                         vldrive_cached_ms=cached_result['vldrive_ms'], vldrive_raw_ms=raw_result['vldrive_ms'],
                         vldrive_repeat_ms=repeat_result['vldrive_ms']),
            live_test='not attempted by offline smoke; use live entrypoint separately')
        print('D: raw sensors -> 11 tokens -> 5 waypoints passed; reconstruction calls = 0', flush=True)
    finally:
        hook.remove()
