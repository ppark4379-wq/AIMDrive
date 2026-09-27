import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('light_input_review', ROOT / 'tools/validation/audit_sensor_record_light_inputs.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class LightInputReviewTests(unittest.TestCase):
    def test_crop_bounds_match_actual_transform(self):
        # A smooth coordinate image makes crop position observable in the
        # actual tensor, not only in a copy of the geometry formula.
        width, height = 1280, 720
        ramp = np.zeros((height, width, 3), np.uint8)
        ramp[:,:,0] = np.rint(np.arange(width)[None,:] * 255 / (width-1))
        ramp[:,:,1] = np.rint(np.arange(height)[:,None] * 255 / (height-1))
        adapter = review.MoraiLMDriveVisionAdapter()
        tensors = adapter.camera_tensors(**{v+'_rgb':ramp for v in review.VIEWS})
        for view in review.INPUTS:
            with self.subTest(view=view):
                size = 224 if view == 'rgb_front' else 128
                rgb = review.rgb_from_model_tensor(tensors[view])
                self.assertEqual(rgb.shape, (size,size,3))
                x0,y0,x1,y1 = review.crop_rectangle(width,height,view)
                actual = [float(rgb[0,0,0])*(width-1)/255., float(rgb[0,0,1])*(height-1)/255.,
                          float(rgb[-1,-1,0])*(width-1)/255., float(rgb[-1,-1,1])*(height-1)/255.]
                self.assertLess(np.max(np.abs(np.array(actual)-[x0,y0,x1,y1])), 9)

    def test_signal_at_raw_top_is_outside_both_front_inputs(self):
        for view in ('rgb_front','rgb_center'):
            x0,y0,x1,y1 = review.crop_rectangle(1280,720,view)
            self.assertFalse(x0 <= 492 < x1 and y0 <= 16 < y1)

    def test_invalid_crop_view_rejected(self):
        with self.assertRaises(ValueError):
            review.crop_rectangle(1280,720,'unknown')


if __name__ == '__main__':
    unittest.main()
