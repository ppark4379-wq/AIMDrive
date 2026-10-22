#!/usr/bin/env python3
"""Publish sensor-only frozen VE footprints for the opt-in Expert input.

No simulator actor state, object GT, traffic-light GT, or vehicle-control topic
is read or written here. The Expert keeps its separate signal input. Objects
are class-unknown, have no persistent identity, and carry only BEV geometry.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKPOINTS = {
    'E3': {
        'path': '/mnt/d/MoraiDrive/dataset/_training_runs/morai_transfer/mixed_grid_fast_20260930_152712_134039/checkpoint-best-object.pth.tar',
        'sha256': '6824d3998ec4be85bf40f96de37a195d12ff00aa8181d6739ed75125a7bb5e5b'},
    'E4': {
        'path': '/mnt/d/MoraiDrive/dataset/_training_runs/morai_transfer/continue_e4e5_20261004_050644/checkpoint-epoch-001.pth.tar',
        'sha256': '6a1de2df2cf08f169d86656f28d489263be54a0e649579300e92a74524d2cd27'},
    'P1': {
        'path': '/mnt/d/MoraiDrive/dataset/_training_runs/morai_transfer/perception_g2_tunnel_20261014_114540/checkpoint-epoch-001.pth.tar',
        'sha256': '054769a5462cd9684f0de6ba80c7b7fc28b234051a7b0490020a45be050e59f7'},
}
HEALTH_SCHEMA = 'morai_ve_perception_health_v1'
MAP_BASIS = {
    'schema': 'morai_projected_grid_basis_v1',
    'position_frame': 'projected_grid',
    'source_orientation_frame': 'true_enu',
    'orientation_frame': 'projected_grid',
    'crs': '+proj=utm +zone=52 +datum=WGS84 +units=m +no_defs',
    'origin_xyz': [302595.0, 4124145.0, 0.0],
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def load_config(path, label=None):
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    if doc.get('schema') != 'morai_live_ve_objects_v1':
        raise ValueError('Unsupported live VE object configuration')
    selected_label = label or doc.get('checkpoint_label', 'E3')
    if selected_label not in CHECKPOINTS:
        raise ValueError('Unknown checkpoint label')
    selected = dict(CHECKPOINTS[selected_label])
    catalog = doc.get('checkpoints')
    # Existing E3/E4 configurations remain usable after a new opt-in label is
    # registered. Every supplied identity must still match the reviewed table,
    # and the selected checkpoint must be explicitly present in the config.
    if (not isinstance(catalog, dict) or selected_label not in catalog
            or any(key not in CHECKPOINTS or value != CHECKPOINTS[key]
                   for key, value in catalog.items())):
        raise ValueError('Checkpoint table differs from the reviewed checkpoint identities')
    if doc.get('map_basis') != MAP_BASIS:
        raise ValueError('Projected-grid basis differs from the checkpoint evidence')
    if doc.get('object_topic') != '/ve/Object_topic' or doc.get('health_topic') != '/ve/perception_health':
        raise ValueError('Only separate VE object/health topics may be published')
    max_age = float(doc.get('max_age_s', .35))
    if not math.isfinite(max_age) or not 0. < max_age <= .35:
        raise ValueError('Live source age must be positive and at most .35 seconds')
    if doc.get('slop_ns', 50000000) != 50000000:
        raise ValueError('Sensor pairing must preserve the 50ms source-time limit')
    lidar_id = doc.get('expected_lidar_frame_id', 'Lidar3D-5')
    if not isinstance(lidar_id, str) or not lidar_id.strip() or lidar_id != lidar_id.strip():
        raise ValueError('Declare a nonempty exact ROS LiDAR frame ID')
    if not Path(selected['path']).is_file():
        raise FileNotFoundError(selected['path'])
    doc.update(checkpoint_label=selected_label, checkpoint=selected,
               max_age_s=max_age, slop_ns=50000000,
               expected_lidar_frame_id=lidar_id)
    return doc


def validate_contract(contract, expected):
    from data.src.loaders.morai_encoder_contract import validate_encoder_contract
    contract = validate_encoder_contract(contract)
    if contract.get('map_basis') != expected['map_basis']:
        raise ValueError('Loaded model map basis mismatch')
    if contract.get('velocity_input') != 'causal_ego_topic_speed_mps_50ms':
        raise ValueError('Live source requires the trained Ego velocity contract')
    if contract.get('detection_output_axes') != 'right_forward_offsets_right_negative_forward_heading':
        raise ValueError('Detection axes mismatch')
    if contract.get('feature_shape') != [106, 256]:
        raise ValueError('Visual token contract mismatch')
    return contract


def make_health(config, generation, sequence, meta, *, ready, inference_ms=0.,
                object_count=0, invalid_reason=None, now_ns=None):
    now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
    return dict(schema=HEALTH_SCHEMA, ready=bool(ready),
                stamp_ns=int(meta.get('stamp_ns') or 0),
                sequence=int(sequence) & 0xffffffff,
                sensor_received_monotonic_ns=int(meta.get('sensor_received_monotonic_ns') or 0),
                published_monotonic_ns=now_ns,
                checkpoint_sha256=config['checkpoint']['sha256'],
                output_frame='projected_grid', map_basis=config['map_basis'],
                invalid_reason=None if ready else str(invalid_reason or meta.get('invalid_reason') or 'not_ready'),
                model_inference_ms=float(inference_ms), generation=str(generation),
                object_count=int(object_count), object_class_predicted=False,
                persistent_tracks_available=False, velocity_vector_available=False,
                scalar_velocity_only=True,
                object_id_policy='frame_local_sequence_not_actor_track',
                geometry_height_available=False,
                object_size_axes='heading_forward_length_lateral_width',
                lidar_frame_id=meta.get('lidar_frame_id'),
                source_stamps_ns=meta.get('source_stamps_ns', {}),
                matched_sensor_skew_ns=meta.get('matched_sensor_skew_ns'),
                source_ages_s=meta.get('ages_s', {}))


def make_object_message(boxes, meta, sequence, message_type, object_type, stamp_factory):
    """Dedicated VE wire: size=(forward length,lateral width,unknown height).

    Height=0 is unavailable, not a learned height. IDs enumerate this message
    only. No class, motion direction, GT ID, or static classification is inferred.
    """
    stamp_ns = int(meta['stamp_ns'])
    if stamp_ns <= 0:
        raise ValueError('Positive source timestamp required')
    message = message_type()
    message.header.stamp = stamp_factory(stamp_ns // 1000000000, stamp_ns % 1000000000)
    message.header.seq = int(sequence) & 0xffffffff
    message.header.frame_id = 'map'
    for index, box in enumerate(boxes, 1):
        x, y = map(float, box['center_world'])
        width, length, heading = (float(box[key]) for key in ('width_m', 'length_m', 'heading_deg'))
        if not all(math.isfinite(value) for value in (x, y, width, length, heading)) or min(width, length) <= 0.:
            raise ValueError('Invalid decoded VE footprint')
        obj = object_type()
        obj.unique_id, obj.type, obj.name = index, -1, 've_unknown'
        obj.position.x, obj.position.y, obj.position.z = x, y, 0.
        obj.heading = heading
        obj.size.x, obj.size.y, obj.size.z = length, width, 0.
        speed = box.get('speed_mps')
        speed = float(speed) if speed is not None else 0.
        # The opt-in Expert treats motion as unknown. This scalar is diagnostic
        # only, never a trusted world velocity or static-object decision.
        obj.velocity.x = speed * 3.6 if math.isfinite(speed) and speed >= 0. else 0.
        obj.velocity.y, obj.velocity.z = 0., 0.
        message.npc_list.append(obj)
    message.num_of_npcs = len(message.npc_list)
    message.num_of_obstacle, message.num_of_pedestrian = 0, 0
    return message


def can_publish(meta, now_ns, max_age_s):
    if not meta.get('ready'):
        return False, meta.get('invalid_reason') or 'source_not_ready'
    receipt = int(meta.get('sensor_received_monotonic_ns') or 0)
    age = (int(now_ns) - receipt) / 1e9
    if receipt <= 0 or not 0. <= age <= max_age_s:
        return False, 'source_stale_after_inference'
    if int(meta.get('stamp_ns') or 0) <= 0:
        return False, 'invalid_source_stamp'
    return True, None


def publish_object_message(publisher, message):
    """Return the Header.seq rospy actually put on the object wire."""
    publisher.publish(message)
    return int(message.header.seq) & 0xffffffff


def run(config, args):
    # All CUDA, ROS, sensor, and checkpoint deserialization stays after --check.
    from utils.common.runtime import configure_runtime
    configure_runtime()
    import numpy as np
    import torch
    import rospy
    from morai_msgs.msg import ObjectStatusList, ObjectStatus
    from std_msgs.msg import String
    from inference.sensors.morai_ros_perception import RosPerceptionSource, forward_perception
    from utils.evaluation.morai_transfer_evaluation import _load_model
    from data.src.loaders.morai_encoder_contract import checkpoint_encoder_contract
    from utils.evaluation.morai_ve_expert_adapter import decode_world_predictions
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    model, payload, checkpoint_hash = _load_model(Path(config['checkpoint']['path']), config['checkpoint']['sha256'], args.device)
    contract = validate_contract(checkpoint_encoder_contract(payload, checkpoint_hash), config)
    rospy.init_node('morai_live_ve_objects', anonymous=False, disable_signals=True)
    objects_pub = rospy.Publisher(config['object_topic'], ObjectStatusList, queue_size=1, latch=False)
    health_pub = rospy.Publisher(config['health_topic'], String, queue_size=1, latch=False)
    generation = str(uuid.uuid4())
    source = RosPerceptionSource(contract, lidar_frame='sensor_flu',
        lidar_to_base={'translation_xyz': contract['adapter']['lidar_translation_xyz'],
                       'rotation_xyzw': contract['adapter']['lidar_rotation_xyzw']},
        max_age_s=config['max_age_s'], slop_ns=config['slop_ns'],
        expected_lidar_frame_id=config['expected_lidar_frame_id'])
    stopped = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    deadline = time.monotonic() + args.seconds
    sequence, last_stamp, forwards, last_invalid_at = 0, None, 0, 0.
    meta = {}

    def health(**updates):
        report = make_health(config, generation, sequence, meta, **updates)
        health_pub.publish(String(data=json.dumps(report, allow_nan=False, sort_keys=True)))
        return report

    health(ready=False, invalid_reason='starting_sensor_source')
    try:
        source.start()
        while not stopped[0] and not rospy.is_shutdown() and time.monotonic() < deadline:
            try:
                inputs, meta = source.latest(args.device)
                valid, reason = can_publish(meta, time.monotonic_ns(), config['max_age_s'])
                if inputs is None and meta.get('ready') and meta.get('waiting_for_new_frame'):
                    time.sleep(.005)
                    continue
                if inputs is None or not valid:
                    if time.monotonic() - last_invalid_at >= .1:
                        health(ready=False, invalid_reason=reason or 'source_not_ready')
                        last_invalid_at = time.monotonic()
                    time.sleep(.01)
                    continue
                if int(meta['stamp_ns']) == last_stamp:
                    time.sleep(.005)
                    continue
                started = time.monotonic()
                with torch.no_grad():
                    detection, _traffic_light = forward_perception(model, inputs)
                raw = detection[0].detach().float().cpu().numpy()
                forwards += 1
                if raw.shape != (2500, 8) or not np.isfinite(raw).all():
                    raise ValueError('Nonfinite or incompatible detection output')
                decoded = decode_world_predictions(raw, meta['ego_pose'])
                inference_ms = (time.monotonic() - started) * 1000.
                valid, reason = can_publish(meta, time.monotonic_ns(), config['max_age_s'])
                if not valid:
                    health(ready=False, invalid_reason=reason, inference_ms=inference_ms)
                    last_stamp = int(meta['stamp_ns'])
                    continue
                sequence = (sequence + 1) & 0xffffffff
                message = make_object_message(decoded['boxes'], meta, sequence,
                    ObjectStatusList, ObjectStatus, rospy.Time)
                # Independent ROS transports can arrive in either order. The
                # Expert matches sequence/stamp and waits for BOTH messages.
                # rospy assigns Header.seq during synchronous serialization.
                # With no object subscribers it can return without doing so;
                # once Expert connects its own publisher counter replaces our
                # provisional value. Health must certify the actual wire seq.
                sequence = publish_object_message(objects_pub, message)
                report = health(ready=True, inference_ms=inference_ms, object_count=len(decoded['boxes']))
                last_stamp = int(meta['stamp_ns'])
                if forwards == 1 or forwards % 25 == 0:
                    print(json.dumps(dict(phase='ve_objects', checkpoint=config['checkpoint_label'],
                        sequence=sequence, objects=report['object_count'], stamp_ns=report['stamp_ns'],
                        model_inference_ms=round(inference_ms, 3), model_forwards=forwards,
                        training_updates=0, control_messages=0), allow_nan=False), flush=True)
            except Exception as error:
                health(ready=False, invalid_reason=type(error).__name__ + ': ' + str(error))
                raise
    finally:
        try:
            health(ready=False, invalid_reason='publisher_stopped')
            # Allow the invalid heartbeat transport a moment before exit.
            time.sleep(.1)
        finally:
            try:
                source.close()
            finally:
                try:
                    objects_pub.unregister()
                finally:
                    health_pub.unregister()
    print(json.dumps(dict(status='COMPLETE', model_forwards=forwards, training_updates=0,
                         control_messages=0, generation=generation)), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint-label', choices=sorted(CHECKPOINTS))
    parser.add_argument('--check', action='store_true', help='Read metadata only; no model, ROS, or GPU')
    parser.add_argument('--seconds', type=float, default=600.)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 0. < args.seconds <= 3600. or not 1 <= args.threads <= 16:
        parser.error('seconds must be in (0,3600], threads in [1,16]')
    config = load_config(args.config, args.checkpoint_label)
    if args.check:
        print(json.dumps(dict(status='METADATA_CHECKED', checkpoint=config['checkpoint'],
            checkpoint_label=config['checkpoint_label'], checkpoint_bytes=Path(config['checkpoint']['path']).stat().st_size,
            map_basis=config['map_basis'], topics=[config['object_topic'], config['health_topic']],
            model_loaded=False, model_forwards=0, training_updates=0, control_messages=0), allow_nan=False))
        return 0
    run(config, args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
