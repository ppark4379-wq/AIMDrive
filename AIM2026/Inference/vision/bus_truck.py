"""Large-body LiDAR supplement between VE and the unchanged Expert contract.

Geometry identifies bus/truck-sized candidates, not semantic vehicle classes.
Only these candidates enter missing-detection memory; other VE boxes pass through.
"""
import math
import numpy as np

from AIM2026.UDP.Detection import Box
from .lidar_obstacles import LidarObstacleTracker
from .detection_memory import DetectionMemory
from .lane_tracking import LaneTracker


class LargeBodyDetector(LidarObstacleTracker):
    def accepts(self, cloud, shape):
        return (shape is not None and 5.5 <= shape['length'] <= 18.
                and shape['width'] <= 3.8
                and np.quantile(cloud[:, 2], .95) >= 2.2
                and np.ptp(np.quantile(cloud[:, 2], [.05, .95])) >= 1.2)


class BusTruckSupplement:
    def __init__(self, hold_s=2., lanes=None):
        self.detector = LargeBodyDetector()
        self.memory = LaneTracker(lanes) if lanes else DetectionMemory(hold_s)
        self.last_time = None
        self.observations = []

    @staticmethod
    def row(box, identity):
        return dict(box.to_wire(), track_id=identity,
                    planning_heading_deg=box.heading_deg,
                    planning_speed_mps=box.speed_mps,
                    planning_speed_source='large_body_observation',
                    motion_state=('uncertain' if box.speed_mps is None else
                                  'stationary' if box.speed_mps < .5 else 'moving'))

    @staticmethod
    def box(row):
        # Encode bounded positional uncertainty in the ordinary footprint.
        # No planner-only hazard fields or speed-uncertainty hooks are required.
        return Box(*row['center_world'], row['planning_heading_deg'],
                   min(30., row['width_m'] + 2*row.get('lateral_uncertainty_m', 0.)),
                   min(30., row['length_m'] + 2*row.get('longitudinal_uncertainty_m', 0.)),
                   row['planning_speed_mps'], detection_origin='lidar_supplement')

    def apply(self, boxes, points, pose, now):
        if not math.isfinite(now):
            raise ValueError('Large vehicle observation time must be finite')
        if self.last_time is not None and now <= self.last_time:
            # Do not refresh a missing object's lifetime with a repeated frame.
            if now == self.last_time:
                return list(boxes), {'large_vehicle_supplement': 0}
            self.detector = LargeBodyDetector()
        self.last_time = now
        self.detector.observe(points, pose, now)
        self.observations = self.detector.observations
        originals = [self.row(box, -1) for box in boxes]
        observed = []
        for item in self.observations:
            if not item['confirmed']:
                continue
            body = Box(*item['center_world'], item['heading_deg'], item['width_m'],
                       item['length_m'], item['speed_mps'])
            row = self.row(body, item['track_id'])
            # The observed large-body extent seeds memory even when VE sees
            # only a small fragment. Do not replace the real VE output.
            matches = [b for b in originals if DetectionMemory._overlaps(row, b)]
            if len(matches) > 1:
                continue
            if matches and row['planning_speed_mps'] is None:
                row['planning_speed_mps'] = matches[0]['planning_speed_mps']
                row['motion_state'] = matches[0]['motion_state']
            observed.append(row)
        packet = dict(generation='large-body', sensor_received_ns=round(now*1e9),
                      boxes=observed, ego_pose=dict(x=pose[0], y=pose[1]))
        recovered = self.memory.apply(packet)['boxes']
        additions = []
        for row in recovered:
            candidate = self.box(row)
            if any(DetectionMemory._overlaps(self.row(candidate, -1), other)
                   for other in originals):
                continue
            if any(DetectionMemory._overlaps(self.row(candidate, -1), self.row(other, -1))
                   for other in additions):
                continue
            additions.append(candidate)
        additions = additions[:max(0, 256-len(boxes))]
        return [*boxes, *additions], dict(large_vehicle_supplement=len(additions),
            large_vehicle_observed=len(observed),
            large_vehicle_held=sum(bool(row.get('detection_held')) for row in recovered))
