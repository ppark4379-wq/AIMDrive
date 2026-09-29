"""Shared detector, independent camera trackers, original-image output."""
from copy import copy
import time
import uuid
import cv2
import numpy as np


class CameraPipeline:
    def __init__(self, detector, cameras, tracker_args, session_id=None):
        self.detector = detector
        self.cameras = cameras
        self.args = tracker_args
        self.session_id = session_id or uuid.uuid4().hex
        self.states = {}

    def _state(self, frame):
        from tracker.mc_bot_sort import BoTSORT
        state = self.states.get(frame.camera_id)
        if state is None or state['epoch'] != frame.stream_epoch:
            args = copy(self.args)
            args.with_reid = False
            args.proximity_thresh, args.appearance_thresh = 0.5, 0.25
            args.name, args.ablation = frame.camera_id, False
            state = {'epoch': frame.stream_epoch, 'count': 0, 'last_stamp_ns': None,
                'tracker': BoTSORT(args, frame_rate=self.cameras[frame.camera_id]['expected_hz'])}
            self.states[frame.camera_id] = state
        return state

    def process(self, frame):
        started = time.monotonic_ns()
        image = cv2.imdecode(np.frombuffer(frame.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('JPEG decode failed for ' + frame.camera_id)
        height, width = image.shape[:2]
        expected = self.cameras[frame.camera_id]['image_size']
        if [width, height] != expected:
            raise ValueError('Unexpected image size for %s: %s, expected %s' %
                (frame.camera_id, [width, height], expected))
        polygons = self.cameras[frame.camera_id].get('ego_mask_polygons', [])
        inference_image = image
        if polygons:
            # Opt-in, calibrated ego-body silhouette; retain untouched original for review.
            inference_image = image.copy()
            cv2.fillPoly(inference_image, [np.asarray(p, np.int32) for p in polygons], (114, 114, 114))
        state = self._state(frame)
        last = state['last_stamp_ns']
        if last is not None and frame.stamp_ns <= last:
            raise ValueError('Non-increasing sensor timestamp')
        detect_started = time.monotonic_ns()
        detections = np.asarray(self.detector.detect(inference_image), dtype=np.float32)
        if detections.ndim != 2 or detections.shape[1] != 6 or not np.isfinite(detections).all():
            raise ValueError('Detector must return finite N x 6 detections')
        detections = detections.copy()
        detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, width)
        detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, height)
        valid = ((detections[:, 2] > detections[:, 0]) &
                 (detections[:, 3] > detections[:, 1]) &
                 (detections[:, 4] >= 0) & (detections[:, 4] <= 1) &
                 (detections[:, 5] >= 0) & (detections[:, 5] == np.floor(detections[:, 5])))
        if not valid.all():
            raise ValueError('Invalid detector box, class or confidence')
        detected = time.monotonic_ns()
        tracks = state['tracker'].update(detections, inference_image)
        records = []
        for track in tracks:
            # Keep tentative tracks inside BoT-SORT for confirmation, but do not
            # publish their one-frame IDs as established identities by default.
            if not track.is_activated and not getattr(self.args, 'include_tentative_tracks', False):
                continue
            box = np.asarray(track.tlbr, dtype=float)
            if not np.isfinite(box).all():
                raise ValueError('Non-finite tracker box')
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            area = (box[2] - box[0]) * (box[3] - box[1])
            if box[2] <= box[0] or box[3] <= box[1] or area <= self.args.min_box_area:
                continue
            cls = int(track.cls)
            names = self.detector.names
            name = (names.get(cls, str(cls)) if isinstance(names, dict)
                    else names[cls] if 0 <= cls < len(names) else str(cls))
            records.append({'track_id': int(track.track_id), 'class_id': cls,
                'class_name': name, 'score': float(track.score),
                'bbox_xyxy': box.tolist(), 'confirmed': bool(track.is_activated)})
        finished = time.monotonic_ns()
        state['count'] += 1
        state['last_stamp_ns'] = frame.stamp_ns
        record = {'schema_version': '1.0', 'message_type': 'frame',
            'session_id': self.session_id, 'camera_id': frame.camera_id,
            'sensor_id': self.cameras[frame.camera_id].get('sensor_id'),
            'source_frame_id': frame.source_frame_id, 'stream_epoch': frame.stream_epoch,
            'frame_id': state['count'], 'received_frame_id': frame.received_frame_id,
            'timestamp': {'sec': frame.stamp[0], 'nanosec': frame.stamp[1]},
            'timestamp_domain': frame.timestamp_domain, 'transport': frame.transport,
            'received_monotonic_ns': frame.received_monotonic_ns,
            'image_size': [width, height], 'bbox_format': 'xyxy_pixels',
            'ego_mask_polygon_count': len(polygons),
            'tracks': records,
            'detections': [{'bbox_xyxy': row[:4].tolist(), 'score': float(row[4]),
                'class_id': int(row[5])} for row in detections],
            'source_dt_s': None if last is None else (frame.stamp_ns - last) / 1e9,
            'decode_ms': (detect_started - started) / 1e6,
            'detection_ms': (detected - detect_started) / 1e6,
            'tracking_ms': (finished - detected) / 1e6,
            'processing_ms': (finished - started) / 1e6}
        return record, image


def draw_tracks(image, record):
    output = image.copy()
    for track in record['tracks']:
        x1, y1, x2, y2 = [round(v) for v in track['bbox_xyxy']]
        color = (0, 255, 0) if track['confirmed'] else (0, 180, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = '%s:%s %s %.2f' % (record['camera_id'], track['track_id'],
                                   track['class_name'], track['score'])
        cv2.putText(output, label, (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(output, record['camera_id'], (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 255), 2)
    return output
