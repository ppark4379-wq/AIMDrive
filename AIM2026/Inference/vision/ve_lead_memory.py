"""One-second VE-only lead-body gap filling, after observed-only tracking."""
from copy import deepcopy
import math

from .detection_memory import DetectionMemory
from .lane_tracking import Lane


class VeLeadMemory:
    def __init__(self, lanes, hold_s=1.):
        if type(hold_s) not in (int, float) or not math.isfinite(hold_s) or not 0 <= hold_s <= 1.:
            raise ValueError('VE lead hold must be within 0..1 second')
        self.hold_s = hold_s
        self.lanes = {name: Lane(points) for name, points in (lanes or {}).items()}
        self.reset()

    def reset(self):
        self.tracks = {}
        self.generation = None
        self.last_time = None

    def _lane(self, xy, heading):
        choices = sorted(((*lane.project(xy), name) for name, lane in self.lanes.items()))
        if not choices or choices[0][0] > 1.2:
            return None
        if len(choices) > 1 and choices[1][0] - choices[0][0] < .4:
            return None  # Ambiguous fork/merge: do not enable memory here.
        distance, progress, name = choices[0]
        _, tangent = self.lanes[name].at(progress)
        if abs((heading - tangent + 180) % 360 - 180) > 15:
            return None
        return name, progress

    def _eligible(self, box, ego_lane):
        if ego_lane is None or not (5.5 <= box['length_m'] <= 18 and 2. <= box['width_m'] <= 3.8):
            return False
        lane = self._lane(box['center_world'], box['planning_heading_deg'])
        return lane is not None and lane[0] == ego_lane[0] and 0 < lane[1]-ego_lane[1] <= 45

    def apply(self, packet):
        now = packet.get('object_received_ns', packet['sensor_received_ns']) / 1e9
        generation = packet.get('generation')
        if generation != self.generation or (self.last_time is not None and now < self.last_time):
            self.reset()
        repeated = self.last_time is not None and now == self.last_time
        self.generation, self.last_time = generation, now
        pose = packet.get('ego_pose', {})
        ego_lane = self._lane((pose['x'], pose['y']), math.degrees(pose['yaw_grid_rad'])) if all(
            k in pose for k in ('x', 'y', 'yaw_grid_rad')) else None
        real = packet['boxes']
        observed = [b for b in real if b.get('detection_origin', 've') == 've'
                    and b.get('source') != 'lidar' and not b.get('detection_held')
                    and b.get('planning_speed_source') not in ('static_memory', 'static_map')
                    and isinstance(b.get('track_id'), int) and b['track_id'] >= 0]
        present = {b['track_id'] for b in real}
        held = []
        for identity, (seen, template) in list(self.tracks.items()):
            age = now-seen
            if identity in present or age >= self.hold_s or not self.hold_s:
                del self.tracks[identity]
                continue
            candidate = DetectionMemory._predict(template, age)
            if not self._eligible(candidate, ego_lane) or any(
                    DetectionMemory._overlaps(candidate, b) for b in real):
                del self.tracks[identity]
                continue
            candidate.update(detection_position_source='ve_same_lane_hold',
                             planning_speed_source='ve_lead_memory',
                             detection_last_observed_ns=round(seen*1e9))
            # Metadata from old LiDAR support must not look like current evidence.
            candidate.pop('lidar_center', None)
            candidate.pop('lidar_points', None)
            held.append(candidate)
        if not repeated and self.hold_s:
            for b in observed:
                if not self._eligible(b, ego_lane):
                    continue
                template = deepcopy(b)
                # Do not push a disappearing lead vehicle forward using VE's
                # provisional scalar speed. Unknown motion stays at its last
                # world position and remains explicitly unknown to the planner.
                known = b.get('motion_state') == 'stationary' or (
                    b.get('motion_direction_observed') and b.get('planning_speed_source') in
                    ('position_history', 've_supported_by_position', 'lane_position_history'))
                speed = b.get('observed_speed_mps') if known and b.get('motion_state') != 'stationary' else 0.
                if not known or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed < 0:
                    template.update(planning_speed_mps=None, motion_state='uncertain')
                else:
                    template['planning_speed_mps'] = speed
                self.tracks[b['track_id']] = (now, template)
        held = held[:max(0, 256-len(real))]
        return dict(packet, boxes=[*real, *held], ve_lead_held_count=len(held))
