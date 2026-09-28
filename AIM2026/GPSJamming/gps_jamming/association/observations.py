"""Per-camera-frame audit records. No inferred synchronization or object velocity."""
import json
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from .positions import estimate_positions


def track_observations(tracks, result):
    """Preserve every cluster candidate; do not silently merge multiple clusters."""
    grouped = {}
    positions = estimate_positions(tracks, result)
    if result is not None:
        for match in result.matches:
            grouped.setdefault(match.track_id, []).append({
                'cluster_id': int(match.cluster_id),
                'centroid_xyz_m': match.centroid.tolist(),
                'coordinate_frame': match.coordinate_frame,
                'point_count': len(match.cluster_points),
                'inside_points': int(match.inside_points),
                'visible_points': int(match.visible_points),
                'overlap_ratio': float(match.overlap_ratio),
            })
    rows = []
    for track in tracks:
        candidates = grouped.get(track.track_id, [])
        rows.append({
            'track_id': int(track.track_id), 'class_id': int(track.class_id),
            'confidence': float(track.confidence), 'bbox_xyxy': track.bbox.tolist(),
            'status': ('unavailable' if result is None else 'unmatched' if not candidates
                       else 'matched' if len(candidates) == 1 else 'multiple_clusters'),
            'clusters': candidates,
            'position': positions.get(track.track_id),
        })
    return rows


def position_labels(observation):
    """Compact overlay; complete geometry and candidate details remain in JSONL."""
    labels = [f"ID {observation['track_id']} | {observation['status']}"]
    position = observation.get('position')
    if position is not None:
        labels.append(f"Cluster {position['cluster_id']} | XY range {position['planar_distance_m']:.2f}m")
        x, y = position['xyz_m'][:2]
        labels.append(f"Rel POS X {x:+.2f} Y {y:+.2f} m")
    else:
        ids = ','.join(str(c['cluster_id']) for c in observation['clusters']) or '--'
        labels.append(f"Cluster {ids} | XY range --")
        labels.append('Rel POS --')
    motion = observation.get('motion') or {}
    velocity = motion.get('velocity_xy_mps')
    if motion.get('status') == 'relative_only' and velocity is not None:
        vx, vy = velocity
        labels.append(f"Rel VEL X {vx:+.2f} Y {vy:+.2f} m/s")
    else:
        state = {'warming_up': 'warming up',
                 'ambiguous_or_degenerate': 'ambiguous',
                 'position_jump_reset': 'reset: position jump'}.get(motion.get('status'), 'unavailable')
        labels.append(f'Rel VEL -- ({state})')
    return labels


def frame_observation(sequence, camera_stamp, camera_received_at, lidar_frame,
                      tracks, result, status, receiver_stats):
    """Sensor stamps stay in their original domains until their clocks are verified."""
    lidar = None
    if lidar_frame is not None:
        lidar = {
            'frame_id': lidar_frame.frame_id, 'raw_timestamp': lidar_frame.raw_timestamp,
            'timestamp_unix_ns': lidar_frame.timestamp,
            'timestamp_status': lidar_frame.timestamp_status,
            'clock_epoch': lidar_frame.clock_epoch,
            'complete': lidar_frame.complete, 'issues': lidar_frame.issues,
            'coordinate_frame': lidar_frame.coordinate_frame,
            'receive_start_monotonic_ns': lidar_frame.receive_monotonic_time,
            'receive_end_monotonic_ns': lidar_frame.receive_end_monotonic_time,
        }
    return {
        'schema_version': 1, 'camera_id': 'front', 'camera_frame_id': sequence,
        'camera_timestamp': {'sec': int(camera_stamp[0]), 'nanosec': int(camera_stamp[1]),
                             'clock_domain': 'unverified_sensor_clock'},
        'camera_receive_monotonic_ns': round(camera_received_at * 1e9),
        'lidar': lidar, 'association_status': status,
        'synchronized': False, 'acquisition_delta_ns': None,
        'receiver': receiver_stats,
        'tracks': track_observations(tracks, result),
        'unmatched_cluster_ids': result.unmatched_cluster_ids if result is not None else None,
    }


class ObservationLog:
    """New JSONL file per run; one append per processed camera frame."""
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.path = directory / f'front_{name}_{uuid4().hex[:8]}.jsonl'
        self.handle = self.path.open('x', encoding='utf-8', buffering=1)

    def write(self, row):
        self.handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

    def close(self):
        self.handle.close()
