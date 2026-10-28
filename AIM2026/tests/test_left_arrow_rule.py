import unittest

import cv2
import numpy as np

from AIM2026.Inference.vision.arrow_rule import left_arrow
from AIM2026.Inference.vision.runtime import _left_arrow_signal


def frame(*lamps):
    """1280x720 BGR image with lit lamps drawn in the upper area as (x, y, bgr)."""
    image = np.full((720, 1280, 3), 90, np.uint8)
    for x, y, colour in lamps:
        cv2.circle(image, (x, y), 7, colour, -1)
    return image


RED, GREEN, YELLOW = (40, 30, 230), (60, 220, 40), (30, 210, 230)


class LeftArrowRuleTest(unittest.TestCase):
    def test_red_and_green_lamps_of_one_housing_are_the_arrow(self):
        self.assertTrue(left_arrow(frame((650, 65, RED), (720, 65, GREEN)))['arrow'])

    def test_single_lamps_are_not_the_arrow(self):
        for lamps in ([(650, 65, RED)], [(780, 65, GREEN)], [(670, 65, YELLOW)], []):
            self.assertFalse(left_arrow(frame(*lamps))['arrow'])

    def test_red_and_green_of_two_heads_are_not_the_arrow(self):
        # a red head and a green head on different rows, or far apart on the same row
        self.assertFalse(left_arrow(frame((650, 65, RED), (720, 140, GREEN)))['arrow'])
        self.assertFalse(left_arrow(frame((400, 65, RED), (1000, 65, GREEN)))['arrow'])
        # green left of red is not the red + left-arrow housing
        self.assertFalse(left_arrow(frame((720, 65, RED), (650, 65, GREEN)))['arrow'])

    def test_lamps_below_the_search_area_are_ignored(self):
        self.assertFalse(left_arrow(frame((650, 400, RED), (720, 400, GREEN)))['arrow'])

    def test_signal_merge_decides_the_left_turn_from_the_image(self):
        jpeg = lambda *lamps: cv2.imencode('.jpg', frame(*lamps))[1].tobytes()
        head_arrow = dict(code=33, confidence=0.97)
        # arrow in the image: 33 whatever the head said
        self.assertEqual(_left_arrow_signal(dict(code=1, confidence=0.4), jpeg((650, 65, RED), (720, 65, GREEN)))['code'], 33)
        # a lamp fully in view without the arrow: a confident stop, also over the head's 33 (yellow, straight green)
        for lamps in ([(650, 65, RED)], [(670, 65, YELLOW)], [(780, 65, GREEN)]):
            out = _left_arrow_signal(head_arrow, jpeg(*lamps))
            self.assertEqual((out['code'], out['confidence']), (1, 1.0))
        # the lamp leaving the top of the image (vehicle past ~11 m) or no lamp: no reading, never a confident red
        self.assertIsNone(_left_arrow_signal(head_arrow, jpeg((650, 20, RED)))['code'])
        self.assertIsNone(_left_arrow_signal(head_arrow, jpeg())['code'])

    def test_undecodable_front_image_leaves_the_head_output(self):
        self.assertEqual(_left_arrow_signal(dict(code=33, confidence=0.95), b'not a jpeg')['code'], 33)


if __name__ == '__main__':
    unittest.main()
