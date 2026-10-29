import unittest
from copy import deepcopy

from AIM2026.Inference.vision.detection_memory import DetectionMemory


def box(identity=1, x=0., y=0., **changes):
    return dict(dict(track_id=identity, center_world=[x, y], length_m=12., width_m=2.5,
                     planning_heading_deg=0., planning_speed_mps=10., motion_state='moving'), **changes)


def packet(seconds, boxes, generation='run'):
    return dict(generation=generation, sensor_received_ns=int(seconds * 1e9),
                object_received_ns=int(seconds * 1e9), boxes=boxes)


class DetectionMemoryTest(unittest.TestCase):
    def test_predicts_from_last_real_observation_and_expires_at_two_seconds(self):
        memory = DetectionMemory(2.)
        original = packet(10., [box()])
        saved = deepcopy(original)
        memory.apply(original)
        for seconds in (10.5, 11., 11.9):
            result = memory.apply(packet(seconds, []))
            held = result['boxes'][0]
            self.assertAlmostEqual(held['center_world'][0], (seconds - 10.) * 10.)
            self.assertTrue(held['detection_held'])
            self.assertEqual(held['track_id'], 1)
        self.assertEqual(memory.apply(packet(12., []))['boxes'], [])
        self.assertEqual(original, saved)

    def test_redetection_replaces_prediction_and_refreshes_expiry(self):
        memory = DetectionMemory()
        memory.apply(packet(0., [box()]))
        memory.apply(packet(1., []))
        actual = box(x=12.)
        self.assertEqual(memory.apply(packet(1.5, [actual]))['boxes'], [actual])
        self.assertEqual(memory.apply(packet(2.5, []))['boxes'][0]['center_world'], [22., 0.])

    def test_reassigned_id_does_not_leave_duplicate_ghost(self):
        memory = DetectionMemory()
        memory.apply(packet(0., [box()]))
        actual = box(identity=2, x=10.)
        self.assertEqual(memory.apply(packet(1., [actual]))['boxes'], [actual])
        self.assertEqual([b['track_id'] for b in memory.apply(packet(1.5, []))['boxes']], [2])

    def test_adjacent_lane_detection_does_not_erase_missing_bus(self):
        memory = DetectionMemory()
        memory.apply(packet(0., [box(), box(identity=2, y=3.5)]))
        result = memory.apply(packet(1., [box(identity=2, x=10., y=3.5)]))
        self.assertEqual(len(result['boxes']), 2)
        self.assertEqual(result['boxes'][1]['center_world'], [10., 0.])

    def test_stationary_and_unknown_speed_are_not_displaced(self):
        for changes in ({'motion_state': 'stationary'}, {'planning_speed_mps': None}):
            memory = DetectionMemory()
            memory.apply(packet(0., [box(**changes)]))
            self.assertEqual(memory.apply(packet(1., []))['boxes'][0]['center_world'], [0., 0.])

    def test_generation_and_clock_reset_clear_history(self):
        for later in (packet(1., [], generation='new'), packet(-1., [])):
            memory = DetectionMemory()
            memory.apply(packet(0., [box()]))
            self.assertEqual(memory.apply(later)['boxes'], [])

    def test_lidar_and_synthetic_boxes_are_not_remembered(self):
        memory = DetectionMemory()
        memory.apply(packet(0., [box(source='lidar'), box(identity=-1),
                                 box(identity=3, detection_held=True),
                                 box(identity=1000000, planning_speed_source='static_map')]))
        self.assertEqual(memory.apply(packet(1., []))['boxes'], [])

    def test_zero_disables_hold_and_invalid_duration_is_rejected(self):
        memory = DetectionMemory(0.)
        memory.apply(packet(0., [box()]))
        self.assertEqual(memory.apply(packet(.1, []))['boxes'], [])
        for value in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                DetectionMemory(value)


if __name__ == '__main__':
    unittest.main()
