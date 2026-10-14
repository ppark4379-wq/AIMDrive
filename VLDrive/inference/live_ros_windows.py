"""Exact E3 ROS perception on Windows CUDA; publishes planner input via WSL.

No simulator object GT, controls, waypoint planning, or guessed motion smoothing.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import signal
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
E3_SHA = '6824d3998ec4be85bf40f96de37a195d12ff00aa8181d6739ed75125a7bb5e5b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--port', type=int, default=12061)
    parser.add_argument('--seconds', type=float, default=0)
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args()
    from utils.common.runtime import configure_runtime
    configure_runtime()
    import cv2
    import torch
    from utils.evaluation.morai_transfer_evaluation import _load_model
    from data.src.loaders.morai_encoder_contract import checkpoint_encoder_contract
    from utils.evaluation.morai_ve_expert_adapter import decode_world_predictions
    from inference.sensors.morai_ros_perception import RosPerceptionSource
    from inference.sensors.morai_ros_perception import forward_perception
    from inference.sensors.ros_binary_transport import SensorClient
    from inference.ve_object_tracking import VeObjectTracker

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment')
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    print('[VE] Loading verified E3 checkpoint on ' + torch.cuda.get_device_name(0), flush=True)
    model, payload, digest = _load_model(args.checkpoint, E3_SHA, 'cuda')
    contract = checkpoint_encoder_contract(payload, digest)
    source = RosPerceptionSource(contract, lidar_to_base=dict(
        translation_xyz=contract['adapter']['lidar_translation_xyz'],
        rotation_xyzw=contract['adapter']['lidar_rotation_xyzw']))
    tracker = VeObjectTracker()
    generation = str(uuid.uuid4())
    client = SensorClient(source, port=args.port)
    stopped = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    start, log_at, health_at = time.perf_counter(), 0., 0.
    sequence, last_source = 0, None
    published = deque(maxlen=50)
    common = dict(kind='ve_result', generation=generation, checkpoint_sha256=digest,
                  map_basis=contract['map_basis'],
                  roi_flu_m=dict(x_min=-20., x_max=30., y_min=-25., y_max=25.))
    print('[VE] Waiting for ROS camera x4 + LiDAR + Ego. Output: /ve/Object_topic', flush=True)
    try:
        while not stopped[0]:
            now = time.perf_counter()
            if args.seconds and now - start >= args.seconds:
                break
            if args.stop_file and args.stop_file.exists():
                break
            if client.error:
                raise RuntimeError(client.error)
            begin = time.perf_counter_ns()
            inputs, meta = source.latest('cuda')
            built = time.perf_counter_ns()
            if inputs is None:
                if not meta.get('waiting_for_new_frame') and now - health_at >= .25:
                    reason = meta.get('invalid_reason', 'waiting_for_pair')
                    client.send(dict(common, ready=False, reason=reason))
                    health_at = now
                if now - log_at >= 3:
                    print('[VE] ' + str(meta.get('invalid_reason', 'waiting_for_new_frame'))
                          + ' streams=' + json.dumps(client.counts), flush=True)
                    log_at = now
                time.sleep(.005)
                continue
            if last_source is not None and meta['stamp_ns'] <= last_source:
                generation = str(uuid.uuid4())
                common['generation'] = generation
                published.clear()
            last_source = meta['stamp_ns']
            gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            gpu_start.record()
            with torch.inference_mode():
                detection, _ = forward_perception(model, inputs)
            gpu_end.record()
            boxes = decode_world_predictions(detection[0], meta['ego_pose'])['boxes']
            decoded = time.perf_counter_ns()
            # Buffer receipts use monotonic_ns. Never subtract a different clock.
            age_ms = (time.monotonic_ns() - meta['sensor_received_monotonic_ns']) / 1e6
            if age_ms > 350 or age_ms < 0:
                client.send(dict(common, ready=False, reason='input_expired_after_inference'))
                continue
            tracked = tracker.update(dict(generation=generation, sent_ns=decoded, boxes=boxes))['boxes']
            sequence += 1
            sent = time.perf_counter()
            published.append(sent)
            new_hz = (len(published) - 1) / (published[-1] - published[0]) if len(published) > 1 else 0.
            report = dict(common, ready=True, sequence=sequence, source_stamp_ns=meta['stamp_ns'],
                          publish_perf_ns=time.perf_counter_ns(),
                          source_stamps_ns=meta['source_stamps_ns'], ego_pose=meta['ego_pose'],
                          ego_speed_mps=meta['speed_mps'], boxes=tracked,
                          metrics=dict(new_hz=new_hz, gpu_ms=gpu_start.elapsed_time(gpu_end),
                                       preprocess_ms=(built-begin)/1e6, pipeline_ms=(decoded-begin)/1e6,
                                       input_receipt_age_ms=age_ms,
                                       sensor_skew_ms=meta['matched_sensor_skew_ns']/1e6))
            client.send(report)
            if sent - log_at >= 2:
                print('[VE] frame={} objects={} new={:.2f}Hz GPU={:.1f}ms pipeline={:.1f}ms age={:.1f}ms'.format(
                    sequence, len(tracked), new_hz, report['metrics']['gpu_ms'],
                    report['metrics']['pipeline_ms'], age_ms), flush=True)
                log_at = sent
    finally:
        try:
            client.send(dict(common, ready=False, reason='inference_stopped'))
        except OSError:
            pass
        client.close()
        source.close()
        print('[VE] Stopped. No vehicle control was published.', flush=True)


if __name__ == '__main__':
    main()
