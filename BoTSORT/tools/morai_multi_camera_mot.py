#!/usr/bin/env python3
"""Three cameras -> one detector -> independent BoT-SORT trackers -> JSONL."""
import argparse
from collections import Counter, deque
import contextlib
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'yolov7'))
LOG = logging.getLogger('camera_mot')


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--transport', choices=('udp', 'ros'), help='UDP default; ROS for development only')
    p.add_argument('--cameras', nargs='+')
    p.add_argument('--replay', help='Directory from record_cameras.py; preserves original timestamps')
    p.add_argument('--run-dir', required=True, help='New directory for results and metadata')
    p.add_argument('--duration', type=float, default=60, help='Live duration in seconds; ignored for replay')
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--save-every', type=int, default=0, help='Save raw JPEG and overlay every N frames/camera')
    p.add_argument('--detector', choices=('yolov7', 'yolo26'), default='yolov7')
    p.add_argument('--weights', required=True)
    p.add_argument('--device', default='0')
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--yolov7-runtime', choices=('eager', 'cuda-graph', 'tensorrt'), default='eager',
                   help='YOLOv7 execution backend; configured camera shapes warmed before reception')
    p.add_argument('--engine-manifest', help='engines.json from build_morai_tensorrt.py; TensorRT only')
    p.add_argument('--postprocess-device', choices=('model', 'cpu'), default='model',
                   help='YOLOv7 NMS/coordinate scaling device; compare CPU postprocessing on the same input')
    p.add_argument('--conf-thres', type=float, default=0.05)
    p.add_argument('--iou-thres', type=float, default=0.7)
    p.add_argument('--classes', nargs='+', type=int)
    p.add_argument('--agnostic-nms', action='store_true')
    p.add_argument('--track-high-thresh', type=float, default=0.3)
    p.add_argument('--track-low-thresh', type=float, default=0.05)
    p.add_argument('--new-track-thresh', type=float, default=0.4)
    p.add_argument('--include-tentative-tracks', action='store_true',
                   help='Diagnostic: also publish unconfirmed IDs; raw detections are always retained')
    p.add_argument('--track-buffer', type=int, default=30, help='Upstream reference buffer; scaled by configured Hz, counted in updates')
    p.add_argument('--match-thresh', type=float, default=0.7)
    p.add_argument('--min-box-area', type=float, default=10)
    p.add_argument('--cmc-method', choices=('sparseOptFlow', 'orb', 'ecc', 'none'), default='sparseOptFlow')
    p.add_argument('--fuse-score', dest='mot20', action='store_true')
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--cudnn-benchmark', action='store_true', help='Compare CUDA convolution autotuning on fixed camera shapes')
    return p


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def run(args):
    import cv2
    import numpy as np
    import torch
    from camera_mot.config import load_config
    from camera_mot.pipeline import CameraPipeline, draw_tracks
    from camera_mot.sources import create_source
    from camera_mot.replay import ReplaySource
    from tools.morai_detectors import create_detector

    config_path = args.config or (str(Path(args.replay) / 'config.json') if args.replay
                                 else str(ROOT / 'configs/morai_three_cameras.yaml'))
    config = load_config(config_path, args.cameras)
    if args.transport:
        config['transport'] = args.transport
    output = Path(args.run_dir)
    output.mkdir(parents=True, exist_ok=False)
    session_id = uuid.uuid4().hex
    model_hash = hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest = {'schema_version': '1.0', 'session_id': session_id,
        'mode': 'replay' if args.replay else 'live', 'config': config,
        'config_sha256': config_hash, 'weights_sha256': model_hash, 'arguments': vars(args),
        'started_unix_ns': time.time_ns(), 'python': sys.version,
        'torch': torch.__version__, 'opencv': cv2.__version__, 'numpy': np.__version__,
        'input_scope': 'camera RGB JPEG only', 'ground_truth_used': False}
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'config.json', config)
    cv2.setNumThreads(1)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    detector = create_detector(args)
    detector.warmup(image_sizes=[v['image_size'] for v in config['cameras'].values()])
    class_map = dict(enumerate(detector.names)) if isinstance(detector.names, list) else detector.names
    class_hash = hashlib.sha256(json.dumps(class_map, sort_keys=True).encode()).hexdigest()
    manifest.update(class_map=class_map, class_map_sha256=class_hash)
    if hasattr(detector, 'runtime_info'):
        manifest['detector_runtime'] = detector.runtime_info()
    write_json(output / 'manifest.json', manifest)
    pipeline = CameraPipeline(detector, config['cameras'], args, session_id)
    source = ReplaySource(args.replay, config['cameras']) if args.replay else create_source(config)
    counts, errors, detected_frames, tracked_frames = Counter(), Counter(), Counter(), Counter()
    last_output = {}
    observed_transports = set()
    metrics = {key: {name: deque(maxlen=20000) for name in
        ('processing_ms', 'detection_ms', 'tracking_ms', 'receive_to_result_ms', 'source_dt_s')}
        for key in config['cameras']}
    max_empty = Counter()
    empty_streak = Counter()
    start = time.monotonic()
    next_status = start
    next_log = start
    failure = None
    consecutive_errors = 0
    try:
        with (output / 'tracks.jsonl').open('x', buffering=1) as sink, \
             (output / 'status.jsonl').open('x', buffering=1) as status, \
             (output / 'frames.jsonl').open('x', buffering=1) as frames:
            while not source.closed:
                if not args.replay and time.monotonic() - start >= args.duration:
                    break
                if args.max_frames and sum(counts.values()) >= args.max_frames:
                    break
                frame = source.get(timeout=0.05)
                if frame is not None:
                    observed_transports.add(frame.transport)
                    try:
                        record, image = pipeline.process(frame)
                    except ValueError as exc:
                        errors[frame.camera_id] += 1
                        consecutive_errors += 1
                        status.write(json.dumps({'message_type': 'frame_error', 'camera_id': frame.camera_id,
                            'received_frame_id': frame.received_frame_id, 'error': str(exc)}) + '\n')
                        LOG.error('%s frame rejected: %s', frame.camera_id, exc)
                        if consecutive_errors >= 5:
                            raise RuntimeError('Five consecutive invalid frames; stopping') from exc
                        continue
                    consecutive_errors = 0
                    now_ns = time.monotonic_ns()
                    record.update(result_monotonic_ns=now_ns,
                        receive_to_result_ms=(now_ns - frame.received_monotonic_ns) / 1e6,
                        weights_sha256=model_hash, class_map_sha256=class_hash,
                        config_sha256=config_hash, replay=bool(args.replay))
                    sink.write(json.dumps(record, allow_nan=False) + '\n')
                    key = frame.camera_id
                    counts[key] += 1
                    detected_frames[key] += bool(record['detections'])
                    tracked_frames[key] += bool(record['tracks'])
                    empty_streak[key] = 0 if record['tracks'] else empty_streak[key] + 1
                    max_empty[key] = max(max_empty[key], empty_streak[key])
                    last_output[key] = time.monotonic_ns()
                    for name in metrics[key]:
                        if record[name] is not None:
                            metrics[key][name].append(record[name])
                    if args.save_every and (counts[key] - 1) % args.save_every == 0:
                        folder = output / key
                        folder.mkdir(exist_ok=True)
                        basename = 'e%d_%08d' % (frame.stream_epoch, frame.received_frame_id)
                        relative = '%s/%s.jpg' % (key, basename)
                        (output / relative).write_bytes(frame.jpeg)
                        if not cv2.imwrite(str(folder / (basename + '_tracks.jpg')), draw_tracks(image, record)):
                            raise OSError('Failed to save overlay')
                        saved = asdict(frame)
                        saved.pop('jpeg')
                        saved['image_path'] = relative
                        frames.write(json.dumps(saved) + '\n')
                now = time.monotonic()
                if now >= next_status:
                    health = {}
                    receiver = source.stats()
                    for key in config['cameras']:
                        age = (time.monotonic_ns() - last_output[key]) / 1e6 if key in last_output else None
                        received_age = receiver[key].get('last_received_age_ms')
                        state = ('no_frames' if age is None else 'stale_output' if age > 1000 else 'active')
                        if not args.replay and received_age is not None and received_age > 1000:
                            state = 'stale_input'
                        health[key] = {'state': state, 'last_result_age_ms': age,
                                       'processed_frames': counts[key], **receiver[key]}
                    status.write(json.dumps({'message_type': 'status', 'session_id': session_id,
                        'elapsed_s': now - start, 'cameras': health}, allow_nan=False) + '\n')
                    if now >= next_log:
                        LOG.info('elapsed=%.1fs processed=%s', now - start, dict(counts))
                        next_log = now + 10
                    next_status = now + 1
    except KeyboardInterrupt:
        failure = 'interrupted'
    except Exception as exc:
        failure = '%s: %s' % (type(exc).__name__, exc)
        raise
    finally:
        elapsed = time.monotonic() - start
        source.close()
        report = {'session_id': session_id, 'duration_s': elapsed, 'mode': manifest['mode'],
            'transport': config.get('transport', 'udp'), 'observed_transports': sorted(observed_transports),
            'failure': failure,
            'receiver': source.stats(), 'ignored_frames': source.ignored_frames,
            'metric_window': 'last 20000 processed frames per camera; warmup excluded',
            'latency_scope': 'complete JPEG received to result ready; excludes simulator/transport and file write',
            'cameras': {}}
        for key in config['cameras']:
            report['cameras'][key] = {'processed_frames': counts[key], 'processed_fps': counts[key] / elapsed,
                'frame_errors': errors[key], 'frames_with_detections': detected_frames[key],
                'frames_with_tracks': tracked_frames[key], 'longest_empty_track_run_frames': max_empty[key],
                'metrics': {name: {'samples': len(values), 'p50': float(np.median(values)),
                    'p95': float(np.percentile(values, 95)), 'max': float(max(values))}
                    for name, values in metrics[key].items() if values}}
        if torch.cuda.is_available() and args.device != 'cpu':
            report['peak_torch_gpu_memory_mib'] = torch.cuda.max_memory_allocated() / 2**20
        if hasattr(detector, 'runtime_info'):
            report['detector_runtime'] = detector.runtime_info()
        write_json(output / 'summary.json', report)
        print(json.dumps(report, indent=2))
    return 0 if all(counts[key] for key in config['cameras']) and not failure and not sum(errors.values()) else 2


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.weights = str(Path(args.weights).expanduser().resolve())
    if not Path(args.weights).is_file():
        parser.error('weights file does not exist')
    if args.duration <= 0 or args.img_size <= 0 or args.cpu_threads < 1 or args.max_frames < 0 or args.save_every < 0:
        parser.error('Invalid duration, size, thread count or frame count')
    for name in ('conf_thres', 'iou_thres', 'track_high_thresh', 'track_low_thresh', 'new_track_thresh', 'match_thresh'):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(name + ' must be in [0, 1]')
    if args.track_low_thresh >= args.track_high_thresh or args.track_buffer <= 0:
        parser.error('Require track_low_thresh < track_high_thresh and positive track_buffer')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    with contextlib.redirect_stdout(sys.stderr):
        return run(args)


if __name__ == '__main__':
    sys.exit(main())
