"""Replay saved JPEGs with the original camera IDs, timestamps and epochs."""
import json
from pathlib import Path
import time
from camera_mot.streams import CameraFrame


class ReplaySource:
    def __init__(self, path, cameras):
        self.root = Path(path).resolve()
        self.stream = (self.root / 'frames.jsonl').open()
        self.cameras = cameras
        self.closed = False
        self.ignored_frames = 0
        self.counts = {key: 0 for key in cameras}

    def get(self, timeout=0.1):
        for line in self.stream:
            row = json.loads(line)
            if row['camera_id'] not in self.cameras:
                self.ignored_frames += 1
                continue
            image_path = (self.root / row['image_path']).resolve()
            if self.root not in image_path.parents:
                raise ValueError('Replay image path must remain in capture directory')
            self.counts[row['camera_id']] += 1
            return CameraFrame(camera_id=row['camera_id'], source_frame_id=row['source_frame_id'],
                stamp=tuple(row['stamp']), received_monotonic_ns=time.monotonic_ns(),
                received_frame_id=row['received_frame_id'], stream_epoch=row['stream_epoch'],
                jpeg=image_path.read_bytes(), transport=row.get('transport', 'ros'),
                timestamp_domain=row.get('timestamp_domain', 'morai_ros_header_raw'))
        self.closed = True
        return None

    def stats(self):
        return {key: {'replayed_frames': count} for key, count in self.counts.items()}

    def close(self):
        self.stream.close()
        self.closed = True
