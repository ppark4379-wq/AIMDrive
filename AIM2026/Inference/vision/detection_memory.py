"""Bridge short VE detection gaps without feeding predictions back into tracking."""
from copy import deepcopy
import math


class DetectionMemory:
    def __init__(self, hold_s=2.0):
        if not math.isfinite(hold_s) or hold_s < 0:
            raise ValueError('detection hold seconds must be finite and nonnegative')
        self.hold_s = hold_s
        self.tracks = {}
        self.generation = None
        self.last_time = None

    @staticmethod
    def _predict(box, age):
        predicted = deepcopy(box)
        speed = box.get('planning_speed_mps')
        heading = box.get('planning_heading_deg')
        if (box.get('motion_state') != 'stationary' and speed is not None
                and heading is not None and math.isfinite(speed) and math.isfinite(heading)):
            angle = math.radians(heading)
            x, y = box['center_world']
            predicted['center_world'] = [x + speed * math.cos(angle) * age,
                                         y + speed * math.sin(angle) * age]
        predicted.update(detection_held=True, detection_hold_age_s=age,
                         detection_position_source='last_ve_constant_velocity')
        return predicted

    @staticmethod
    def _overlaps(a, b):
        # Check both boxes' axes; a neighbouring lane must not cancel a held box.
        delta = [b['center_world'][i] - a['center_world'][i] for i in (0, 1)]
        angles = [math.radians(box['planning_heading_deg']) for box in (a, b)]
        for angle in angles:
            for axis in (angle, angle + math.pi / 2):
                separation = abs(delta[0] * math.cos(axis) + delta[1] * math.sin(axis))
                extent = sum((box['length_m'] * abs(math.cos(yaw - axis))
                              + box['width_m'] * abs(math.sin(yaw - axis))) / 2
                             for box, yaw in zip((a, b), angles))
                if separation > extent:
                    return False
        return True

    def apply(self, packet):
        now = packet.get('object_received_ns', packet['sensor_received_ns']) / 1e9
        generation = packet.get('generation')
        if generation != self.generation or (self.last_time is not None and now < self.last_time):
            self.tracks.clear()
        self.generation, self.last_time = generation, now
        real = packet['boxes']
        observed = {box['track_id']: box for box in real
                    if isinstance(box.get('track_id'), int) and box['track_id'] >= 0
                    and box.get('source') != 'lidar' and not box.get('detection_held')
                    and box.get('planning_speed_source') not in ('static_memory', 'static_map')}
        held = []
        for identity, (seen, template) in list(self.tracks.items()):
            age = now - seen
            if identity in observed or age >= self.hold_s:
                del self.tracks[identity]
                continue
            predicted = self._predict(template, age)
            if any(self._overlaps(predicted, box) for box in real):
                del self.tracks[identity]
                continue
            held.append(predicted)
        for identity, box in observed.items():
            self.tracks[identity] = (now, deepcopy(box))
        return dict(packet, boxes=[*real, *held], detection_held_count=len(held))
