"""Recorded /Ego_topic speed; causal lookup, 50 ms maximum age, no fallback."""
from bisect import bisect_right
import importlib
import math
from pathlib import Path
import sys


class CausalEgoSpeed:
    def __init__(self, samples):
        unique = {}
        self.ambiguous_times = set()
        for sample in samples:
            stamp, speed, _ = sample
            previous = unique.get(stamp)
            if previous is not None and previous[1] != speed:
                self.ambiguous_times.add(stamp)
            # Identical retransmissions share a speed; keep the first record as provenance.
            unique.setdefault(stamp, sample)
        self.samples = sorted(unique.values(), key=lambda row: row[0])
        self.times = [row[0] for row in self.samples]
        if not self.times:
            raise ValueError('Missing Ego source timestamps')

    @classmethod
    def from_episode(cls, episode):
        # Decode the schema sealed with the recording, not an installed morai_msgs version.
        try:
            dynamic = importlib.import_module('genpy.dynamic')
        except ImportError:
            ros = Path('/opt/ros/noetic/lib/python3/dist-packages')
            if ros.is_dir() and str(ros) not in sys.path:
                sys.path.append(str(ros))
            try:
                dynamic = importlib.import_module('genpy.dynamic')
            except ImportError as error:
                raise RuntimeError('Ego decoding requires ROS genpy; use the WSL ROS environment') from error
        from MoraiDrive.data_collection.internal_sensor_record.async_capture.journal import records, verify_run
        raw = Path(episode) / 'raw'
        verified = verify_run(raw)
        definition = verified[0]['extra']['topic_state']['definitions']['morai_msgs/EgoVehicleStatus']
        message_type = dynamic.generate_dynamic('morai_msgs/EgoVehicleStatus', definition['definition'])[
            'morai_msgs/EgoVehicleStatus']
        if message_type._md5sum != definition['md5sum']:
            raise ValueError('Recorded Ego message schema mismatch')
        samples = []
        for meta, payload in records(raw, verified=verified):
            if meta.get('role') != 'ego':
                continue
            if (meta.get('ros_type') != 'morai_msgs/EgoVehicleStatus'
                    or meta.get('md5sum') != definition['md5sum']):
                raise ValueError('Ego record schema changed')
            msg = message_type().deserialize(payload)
            stamp = msg.header.stamp.to_nsec()
            if stamp != meta['source_timestamp_ns']:
                raise ValueError('Ego journal timestamp differs from message')
            samples.append((stamp, math.hypot(msg.velocity.x, msg.velocity.y), meta['record_id']))
        return cls(samples)

    def at(self, sensor_time_ns):
        index = bisect_right(self.times, sensor_time_ns) - 1
        if index < 0:
            return None, 'ego_no_preceding_sample'
        stamp, speed, record_id = self.samples[index]
        age = sensor_time_ns - stamp
        if age > 50_000_000:
            return None, 'ego_older_than_50ms'
        if stamp in self.ambiguous_times:
            return None, 'ego_conflicting_duplicate'
        if not math.isfinite(speed) or speed < 0:
            return None, 'ego_invalid_speed'
        return dict(velocity_mps=speed, ego_source_ns=stamp, ego_age_ns=age,
                    ego_record_id=record_id), None
