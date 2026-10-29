import unittest
from unittest.mock import Mock, patch
import numpy as np

from AIM2026.Inference.vision.bus_truck import BusTruckSupplement
from AIM2026.UDP.Detection import Box


def cloud(length=10., top=3.):
    return np.array([[4., y, z] for y in np.arange(5., 5.+length+.01, .2)
                     for z in np.arange(.6, top+.01, .3)])


class BusTruckSupplementTests(unittest.TestCase):
    def test_runtime_runs_supplement_after_ve_and_supports_disabled_path(self):
        from AIM2026.Inference.vision import runtime as module
        runtime = object.__new__(module.VisionRuntime)
        runtime.inputs = Mock()
        runtime.device = 'cpu'
        runtime.contract = {'velocity_input': True}
        runtime.model = Mock()
        runtime.signal_head = runtime.phase_head = None
        runtime.left_arrow_rule = runtime.lidar_heading = runtime.lidar_centers = False
        runtime.big_vehicles = False
        runtime.fragment_merge = False
        runtime.lidar_tracker = BusTruckSupplement()
        runtime._body_points_3d = Mock(return_value=cloud())
        car = Box(20., 5., 0., 2., 4., 8.)
        with patch.object(module, 'forward_perception', return_value=(np.zeros((1, 1)), None, None)), \
             patch.object(module, 'decode_world_boxes', return_value=([car], {})), \
             patch.object(module, 'decode_native_signal', return_value=[{}]), \
             patch.object(module, '_RAW_DUMP', ''), patch.object(module, '_TOKEN_DUMP', ''), \
             patch.dict('os.environ', {}, clear=True):
            for t in (0., .1, .2):
                result = runtime.infer({}, None, 0., dict(x=0., y=0., yaw_grid_rad=0.), observation_time=t)
            self.assertIs(result.boxes[0], car)
            self.assertEqual(len(result.boxes), 2)
            self.assertEqual(result.supplement_indices, (1,))
            runtime.lidar_tracker = None
            result = runtime.infer({}, None, 0., dict(x=0., y=0., yaw_grid_rad=0.), observation_time=.3)
            self.assertEqual(result.boxes, (car,))
            self.assertEqual(result.supplement_indices, ())

    def test_missed_large_body_confirmed_then_held_then_expired(self):
        detector = BusTruckSupplement(hold_s=1.)
        for t in (0., .1, .2):
            boxes, stats = detector.apply([], cloud(), (0., 0., 0.), t)
        self.assertEqual(len(boxes), 1)
        self.assertNotIn('source', boxes[0].to_wire())
        self.assertEqual(Box.from_wire(boxes[0].to_wire()), boxes[0])
        held, stats = detector.apply([], None, (0., 0., 0.), .3)
        self.assertEqual(len(held), 1)
        self.assertEqual(stats['large_vehicle_held'], 1)
        self.assertEqual(detector.apply([], None, (0., 0., 0.), 1.3)[0], [])

    def test_cars_and_rails_are_not_supplemented(self):
        for points in (cloud(length=4.), cloud(top=1.2)):
            detector = BusTruckSupplement()
            for t in (0., .1, .2):
                result, _ = detector.apply([], points, (0., 0., 0.), t)
            self.assertEqual(result, [])

    def test_other_ve_objects_are_identical_and_never_held(self):
        detector = BusTruckSupplement()
        car = Box(20., 5., 0., 2., 4., 8.)
        result, _ = detector.apply([car], None, (0., 0., 0.), 0.)
        self.assertIs(result[0], car)
        self.assertEqual(detector.apply([], None, (0., 0., 0.), .1)[0], [])

    def test_ve_reappearance_deduplicates_held_large_body(self):
        detector = BusTruckSupplement()
        for t in (0., .1, .2):
            result, _ = detector.apply([], cloud(), (0., 0., 0.), t)
        detected = result[0]
        result, _ = detector.apply([detected], None, (0., 0., 0.), .3)
        self.assertEqual(result, [detected])

    def test_clock_reset_drops_predictions(self):
        detector = BusTruckSupplement()
        for t in (1., 1.1, 1.2):
            detector.apply([], cloud(), (0., 0., 0.), t)
        self.assertEqual(detector.apply([], None, (0., 0., 0.), .1)[0], [])


if __name__ == '__main__':
    unittest.main()
