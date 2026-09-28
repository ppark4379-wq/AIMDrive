"""Live MORAI camera/LiDAR calibration preview, optionally using existing BoTSORT."""
from pathlib import Path
import sys
import time
import logging
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[1]
BOTSORT = ROOT.parent / 'BoTSORT'
if not BOTSORT.is_dir():
    BOTSORT = ROOT.parent.parent / 'BoTSORT'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BOTSORT))
sys.path.insert(0, str(BOTSORT / 'yolov7'))

import cv2
import numpy as np
import yaml
from tools.morai_udp import UdpCamera
from tools.morai_udp_mot import make_parser, track_records
from gps_jamming.lidar.runtime import load_config, make_assembler
from gps_jamming.interfaces.lidar_udp import LidarUDPReceiver, LidarUDPStream
from gps_jamming.interfaces.ego_udp import EgoUDPReceiver
from gps_jamming.pipeline import LidarPipeline, AssociationPipeline
from gps_jamming.types import CameraTrack
from gps_jamming.timing.synchronization import FrameSynchronizer, restore_hour_timestamp
from gps_jamming.visualization.preview_hold import PreviewHold
from gps_jamming.timing.gap_metrics import GapMetrics, DisplayIntervals
from gps_jamming.timing.debug_log import selection_lines, display_line
from gps_jamming.motion.bev_tracks import BEVTrackHistory
from gps_jamming.association.observations import (
    ObservationLog, frame_observation, track_observations, position_labels)


def main():
    parser = make_parser()
    parser.description = __doc__
    parser.set_defaults(port=None, bind_ip=None)
    for action in parser._actions:
        if action.dest == 'port':
            action.help = 'Camera destination port; defaults to camera.udp.port in config'
        elif action.dest == 'bind_ip':
            action.help = 'Local camera listen address; defaults to camera.udp.bind in config'
    # Only expose options consumed by this viewer, keeping detector/tracker defaults.
    for action in list(parser._actions):
        if action.dest in ('view', 'show_detections', 'output', 'save_images'):
            parser._handle_conflict_resolve(None, [(flag, action) for flag in action.option_strings[:]])
    for action in parser._actions:
        if action.dest == 'weights':
            action.required = False
    parser.add_argument('--projection-only', action='store_true')
    parser.add_argument('--config', default=str(ROOT / 'configs/perception.yaml'))
    parser.add_argument('--calibration', default=str(ROOT / 'configs/calibration.yaml'))
    parser.add_argument('--snapshot', default=str(ROOT / 'artifacts/association_preview.jpg'))
    parser.add_argument('--log-dir', default=str(ROOT / 'artifacts/association'))
    args = parser.parse_args()
    if not args.projection_only and (not args.weights or not Path(args.weights).is_file()):
        parser.error('Supply existing --weights, or use --projection-only for calibration preview')
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    camera_udp = config.get('camera', {}).get('udp', {})
    if not 2 <= config['camera'].get('frame_buffer_size', 8) <= 120:
        parser.error('camera.frame_buffer_size must be between 2 and 120')
    if args.port is None:
        args.port = camera_udp.get('port', 5050)
    if args.bind_ip is None:
        args.bind_ip = camera_udp.get('bind', '127.0.0.1')
    if args.port != 5050:
        parser.error('Single front-camera preview uses port 5050 only')
    with open(args.calibration, encoding='utf-8') as f:
        calibration = yaml.safe_load(f)
        fusion = AssociationPipeline(calibration['cameras']['front'], config['association'])
    lidar = LidarPipeline(config['lidar'])
    synchronizer = FrameSynchronizer(**config['timing'].get('synchronization', {}))
    hold_ms = config.get('visualization', {}).get('hold_last_pair_ms', 100)
    preview_hold = PreviewHold(hold_ms)
    hold_enabled = hold_ms > 0
    gap_metrics = GapMetrics()
    bev_history = BEVTrackHistory(**config.get('motion', {}))
    display_intervals = DisplayIntervals()
    gap_info = {'consecutive_failed_attempts': 0, 'recovery': None}
    title = 'Camera + LiDAR | calibration preview'
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1280, 720)
    waiting = np.zeros((720, 1280, 3), np.uint8)
    cv2.putText(waiting, f"Starting camera :{args.port} and LiDAR :{config['lidar']['udp']['port']} ...", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    cv2.imshow(title, waiting)
    cv2.waitKey(1)
    detector = tracker = None
    try:
        if not args.projection_only:
            from tools.morai_detectors import create_detector
            from tracker.mc_bot_sort import BoTSORT
            detector = create_detector(args)
            args.with_reid = False
            args.proximity_thresh, args.appearance_thresh = .5, .25
            args.name, args.ablation = 'association_preview', False
            tracker = BoTSORT(args, frame_rate=20)
            detector.warmup()
        with ExitStack() as resources:
            ego_config = config.get('ego', {})
            ego = (resources.enter_context(EgoUDPReceiver(**ego_config.get('udp', {})))
                   if ego_config.get('enabled', False) else None)
            observation_log = ObservationLog(args.log_dir)
            resources.callback(observation_log.close)
            print(f'ID/3D observations: {observation_log.path}', flush=True)
            display_log = ObservationLog(Path(args.log_dir) / 'display')
            resources.callback(display_log.close)
            def finish_display_log():
                event = display_intervals.finish(time.monotonic_ns())
                if event is not None:
                    display_log.write(event)
                    print(display_line(event), flush=True)
            resources.callback(finish_display_log)
            print(f'Display intervals: {display_log.path}', flush=True)
            print('Q/ESC: quit | S: snapshot | H: toggle hold. HUD describes the displayed pair; console describes selection attempts.', flush=True)
            receiver = resources.enter_context(LidarUDPReceiver(
                make_assembler(config), decoder=config['lidar']['decoder'], **config['lidar']['udp']))
            stream = resources.enter_context(LidarUDPStream(receiver, queue_size=2))
            camera = UdpCamera(args.bind_ip, args.port, args.sender_ip, args.index_base,
                               expected_image_size=fusion.calibration.image_size,
                               sender_port=camera_udp.get('source_port', 5000),
                               frame_buffer_size=config['camera'].get('frame_buffer_size', 8))
            resources.callback(camera.close)
            last_frame = None
            image = waiting
            records = []
            stamp = None
            last_camera = 0
            last_report = 0
            frames = 0
            camera_sequence = None
            sync_info = {'status': 'waiting_lidar', 'delta_ns': None}
            result = None
            live_stamp = None
            display_state = None
            observation_cache = None
            while True:
                stage_ms = {}
                camera_diagnostic = None
                observation = None
                new_camera_frame = False
                # Freeze the newest available LiDAR target before inference. Frames
                # arriving during inference are considered on the next iteration.
                pending = stream.read(timeout=0)
                pending += stream.drain()
                packet = None
                reference_ns = camera.latest_timestamp_ns()
                if pending and reference_ns is not None:
                    target = pending[-1]
                    if (target.complete and 0 <= time.monotonic_ns()-target.receive_end_monotonic_time <= synchronizer.max_age):
                        try:
                            target_ns = restore_hour_timestamp(target.raw_timestamp, reference_ns)
                        except ValueError:
                            target_ns = None
                        if target_ns is not None:
                            packet = camera.get_nearest(target_ns, after_stamp=live_stamp,
                                max_age_ms=config['camera'].get('frame_max_age_ms', 400),
                                max_delta_ms=synchronizer.tolerance/1e6)
                if packet is not None and packet.stamp != live_stamp:
                    decoded = cv2.imdecode(np.frombuffer(packet.jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if decoded is not None and (decoded.shape[1], decoded.shape[0]) == tuple(fusion.calibration.image_size):
                        image = decoded
                        stamp = packet.stamp
                        live_stamp = packet.stamp
                        last_camera = packet.received_at
                        frames += 1
                        camera_sequence = packet.sequence
                        new_camera_frame = True
                        if tracker is not None and (image.shape[1], image.shape[0]) == tuple(fusion.calibration.image_size):
                            stage_start = time.perf_counter()
                            detections = detector.detect(image)
                            stage_ms['detector'] = (time.perf_counter() - stage_start) * 1000
                            stage_start = time.perf_counter()
                            records = track_records(tracker.update(detections, image),
                                                    image.shape[1], image.shape[0], args.min_box_area)
                            stage_ms['tracker'] = (time.perf_counter() - stage_start) * 1000
                        else:
                            records = []
                        synchronizer.add_camera({
                            'timestamp_ns': stamp[0] * 10**9 + stamp[1],
                            'receive_ns': round(last_camera * 1e9), 'stamp': stamp,
                            'sequence': camera_sequence, 'image': image, 'records': records})
                        camera.mark_processed(packet.sequence)
                        if display_state is not None and display_state[-1].get('epoch') != synchronizer.epoch:
                            preview_hold.clear()
                    else:
                        camera.mark_processed(packet.sequence, valid=False)
                # Render the selected camera and LiDAR together, never current boxes
                # over an older image. MOT updates above still run in capture order.
                new_camera_frame = bool(pending)
                result = None
                if pending:
                    chosen, mapped, sync_info = synchronizer.select(pending[-1], time.monotonic_ns())
                    camera_diagnostic = camera.diagnose_selection(
                        mapped.timestamp if mapped is not None else None,
                        chosen['sequence'] if chosen is not None else None,
                        {c['sequence'] for c in synchronizer.cameras},
                        synchronizer.max_age, synchronizer.tolerance)
                    if sync_info['status'] in ('lidar_clock_discontinuity', 'invalid_lidar_timestamp'):
                        preview_hold.clear()
                    if chosen is not None:
                        image, records = chosen['image'], chosen['records']
                        stamp, camera_sequence = chosen['stamp'], chosen['sequence']
                        last_camera = chosen['receive_ns'] / 1e9
                    stage_start = time.perf_counter()
                    last_frame = (lidar.process(mapped) if mapped is not None and sync_info['status'] == 'matched'
                                  else mapped if mapped is not None else pending[-1])
                    if mapped is not None and sync_info['status'] == 'matched':
                        stage_ms['lidar'] = (time.perf_counter() - stage_start) * 1000
                else:
                    new_camera_frame = False
                    if display_state is not None:
                        image, records, stamp, last_camera, camera_sequence, last_frame, result, sync_info = display_state
                canvas = image.copy()
                tracks = [CameraTrack(r['track_id'], r['bbox_xyxy'], r['class_id'], r['score']) for r in records]
                age = ((time.monotonic_ns() - last_frame.receive_end_monotonic_time) / 1e9
                       if last_frame is not None else float('inf'))
                visible = matches = 0
                matched_track_ids = set()
                association_status = sync_info['status']
                actual_size = (image.shape[1], image.shape[0])
                resolution_ok = actual_size == tuple(fusion.calibration.image_size)
                if pending and last_frame is not None and resolution_ok and sync_info['status'] == 'matched':
                    stage_start = time.perf_counter()
                    result = fusion.process(last_frame, tracks, image_size=(image.shape[1], image.shape[0]))
                    stage_ms['association'] = (time.perf_counter() - stage_start) * 1000
                    association_status = 'timestamp_matched'
                if pending:
                    display_state = (image, records, stamp, last_camera, camera_sequence, last_frame, result, sync_info)
                if age > synchronizer.max_age / 1e9 or time.monotonic() - last_camera > synchronizer.max_age / 1e9:
                    result = None
                    association_status = 'stale_sensor'
                if result is not None:
                    assigned = {m.cluster_id for m in result.matches}
                    matched_track_ids = {m.track_id for m in result.matches}
                    boxes_by_id = {t.track_id: t.bbox for t in tracks}
                    matches = len(result.matches)
                    for projection in result.projections:
                        visible += len(projection.pixels)
                        if projection.cluster_id not in assigned:
                            continue
                        inside = np.zeros(len(projection.pixels), dtype=bool)
                        for match in result.matches:
                            if match.cluster_id == projection.cluster_id:
                                box = boxes_by_id.get(match.track_id)
                                if box is not None:
                                    inside |= np.all((projection.pixels >= box[:2]) &
                                                     (projection.pixels <= box[2:]), axis=1)
                        for u, v in projection.pixels[inside]:
                            cv2.circle(canvas, (int(u), int(v)), 2, (0, 255, 0), -1)
                observations = track_observations(tracks, result)
                if pending and result is not None:
                    bev_history.update(observations, result.timestamp, synchronizer.epoch)
                    observation_cache = (synchronizer.epoch, result.frame_id, observations)
                elif (result is not None and observation_cache is not None
                      and observation_cache[:2] == (synchronizer.epoch, result.frame_id)):
                    # Rendering the same pair must not drop its motion labels or
                    # add duplicate samples to the velocity estimator.
                    observations = observation_cache[2]
                ego_snapshot = ego.snapshot() if ego else {'status': 'disabled', 'state': None}
                by_id = {row['track_id']: row for row in observations}
                if new_camera_frame:
                    gap_info = gap_metrics.observe(last_frame.frame_id, synchronizer.epoch,
                        last_frame.timestamp if sync_info.get('epoch') is not None else None,
                        time.monotonic_ns(), result is not None, association_status)
                if new_camera_frame and stamp is None:
                    observation = {'schema_version': 4, 'log_trigger': 'lidar_selection_attempt',
                        'camera_timestamp': None, 'synchronized': False,
                        'association_status': association_status, 'gap_metrics': gap_info,
                        'lidar': {'frame_id': last_frame.frame_id, 'raw_timestamp': last_frame.raw_timestamp},
                        'tracks': []}
                if new_camera_frame and stamp is not None:
                    observation = frame_observation(
                        camera_sequence, stamp, last_camera, last_frame, tracks, result,
                        association_status, camera.stats())
                    observation['ego'] = ego_snapshot
                    observation['schema_version'] = 4
                    observation['gap_metrics'] = gap_info
                    observation['log_trigger'] = 'lidar_selection_attempt'
                    observation['synchronized'] = result is not None
                    observation['acquisition_delta_ns'] = sync_info['delta_ns']
                    observation['synchronization'] = sync_info
                    observation['camera_timestamp']['clock_domain'] = 'unix_observed'
                    observation['performance'] = {
                        'stage_wall_ms': stage_ms,
                        'selected_camera_receive_age_ms': (time.monotonic() - last_camera) * 1000,
                        'selected_lidar_receive_age_ms': (time.monotonic_ns() - last_frame.receive_end_monotonic_time) / 1e6 if last_frame else None,
                        'camera_overwritten_frames': camera.overwritten_frames,
                        'lidar_queue_dropped_frames': stream.dropped_frames,
                        'note': 'Receipt ages are not end-to-end latency. Stages refer to this loop, not necessarily selected camera.'}
                for t in tracks:
                    if t.track_id not in matched_track_ids:
                        continue
                    x1, y1, x2, y2 = map(int, t.bbox)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 160, 0), 2)
                    labels = position_labels(by_id[t.track_id])
                    top = max(18, min(y1 - 5 - (len(labels)-1)*20,
                                      canvas.shape[0] - len(labels) * 20))
                    panel_width = max(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .48, 1)[0][0]
                                      for label in labels)
                    label_x = max(0, min(x1, canvas.shape[1] - panel_width - 4))
                    cv2.rectangle(canvas, (label_x, top-15),
                                  (label_x+panel_width+4, top+(len(labels)-1)*20+4), (0, 0, 0), -1)
                    for index, label in enumerate(labels):
                        (width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .48, 1)
                        origin = (label_x+2, top + index * 20)
                        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 3)
                        cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
                # Cache the complete annotated pair, before the live status HUD.
                # A failed selection is still logged above; holding is display-only.
                held = None
                now_ns = time.monotonic_ns()
                if result is not None:
                    preview_hold.update(canvas, (synchronizer.epoch, last_frame.frame_id, camera_sequence),
                        now_ns, min(last_frame.receive_end_monotonic_time, round(last_camera*1e9)) + synchronizer.max_age,
                        {'lidar_frame': last_frame.frame_id, 'camera_frame': camera_sequence,
                         'delta_ms': sync_info['delta_ns']/1e6,
                         'cluster_count': len(last_frame.clusters),
                         'camera_receive_ns': round(last_camera*1e9),
                         'lidar_receive_ns': last_frame.receive_end_monotonic_time})
                else:
                    cached = preview_hold.get(now_ns) if hold_enabled else None
                    if cached is not None:
                        canvas, held = cached
                shown_delta = held['delta_ms'] if held else (
                    sync_info['delta_ns'] / 1e6 if sync_info['delta_ns'] is not None else None)
                shown_clusters = held['cluster_count'] if held else (len(last_frame.clusters) if last_frame else 0)
                delta_text = f'{shown_delta:.2f}' if shown_delta is not None else 'N/A'
                lines = [f'delta_ms={delta_text}', f'lidar clusters={shown_clusters}']
                for i, line in enumerate(lines):
                    y = 25 + 27*i
                    cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .58, (0, 0, 0), 4)
                    cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1)
                cv2.imshow(title, canvas)
                presented_ns = time.monotonic_ns()
                display_mode = 'hold' if held is not None else 'matched' if result is not None else 'unmatched'
                pair_key = ([held['camera_frame'], held['lidar_frame']] if held else
                            [camera_sequence, last_frame.frame_id if last_frame else None])
                event = display_intervals.present(pair_key, display_mode, presented_ns)
                if event is not None:
                    display_log.write(event)
                    print(display_line(event), flush=True)
                if observation is not None:
                    observation['schema_version'] = 5
                    observation['tracks'] = observations
                    observation['camera_selection_diagnostic'] = camera_diagnostic
                    observation['display'] = {'mode': display_mode, 'pair_key': pair_key,
                        'submitted_monotonic_ns': presented_ns,
                        'hold_limit_ms': hold_ms if hold_enabled else 0,
                        'held_ms': held['held_ms'] if held is not None else None}
                    observation_log.write(observation)
                    print('\n'.join(selection_lines(observation)), flush=True)
                key = cv2.waitKey(10) & 255
                if key == ord('h'):
                    hold_enabled = not hold_enabled if hold_ms > 0 else False
                    preview_hold.clear()
                if key in (27, ord('q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord('s') or time.monotonic() - last_report > 5:
                    output = Path(args.snapshot)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    ok, encoded = cv2.imencode('.jpg', canvas)
                    if not ok:
                        raise RuntimeError('Could not encode preview snapshot')
                    output.write_bytes(encoded.tobytes())
                    print(f"[STATUS] ego={ego_snapshot['status']} receiver={camera.stats()} stage_ms={stage_ms}", flush=True)
                    print(f"[EGO] { {k: v for k, v in ego_snapshot.items() if k != 'state'} }", flush=True)
                    last_report = time.monotonic()
                if args.max_frames and frames >= args.max_frames:
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
