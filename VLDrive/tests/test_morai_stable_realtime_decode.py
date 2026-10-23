"""Display-policy integration checks without models, sockets or CUDA."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from data.src.loaders.morai_encoder_contract import DETECTION_AXES
from inference.realtime import realtime_vision_encoder as live


def saturated_prediction():
    raw = np.zeros((2500, 8), dtype=np.float64)
    raw[:, 0] = -20.
    for cell, logit, speed in ((521, 38., 1.), (571, 65., 2.)):
        row, col = divmod(cell, 50)
        raw[cell] = [logit, -(col - 24.5) / 3.5, (10. - (29.5 - row)) / 3.5,
                     0., -1., 4. / 7., 2. / 4., speed / 8.]
    return raw


class StableRealtimeDecodeTests(unittest.TestCase):
    def test_default_preserves_legacy_and_opt_in_keeps_original_geometry_and_speed(self):
        raw = saturated_prediction()
        untouched = raw.copy()
        old = live.native_detection_boxes(raw, detection_axes=DETECTION_AXES)
        fixed = live.native_detection_boxes(raw, detection_axes=DETECTION_AXES,
                                             bbox_order="raw_logit")
        self.assertEqual([b["cell"] for b in old], [(10, 21)])
        self.assertEqual([b["cell"] for b in fixed], [(11, 21)])
        for key in ("objectness", "lateral", "forward", "heading", "half_extent"):
            np.testing.assert_array_equal(fixed[0][key], old[0][key])
        self.assertEqual(fixed[0]["speed"], 2.)
        np.testing.assert_array_equal(raw, untouched)

    def test_custom_threshold_changes_admission_without_changing_raw_priority(self):
        raw = saturated_prediction()
        # This isolated box has sigmoid(.7); threshold .8 must exclude it.
        row, col = divmod(900, 50)
        raw[900] = [math.log(.7 / .3), (-10. - (col - 24.5)) / 3.5,
                    (10. - (29.5 - row)) / 3.5, 0., -1., 4. / 7., .5, 0.]
        for threshold, expected in ((.2, [(11, 21), (18, 0)]), (.8, [(11, 21)])):
            boxes = live.native_detection_boxes(raw, threshold=threshold,
                        detection_axes=DETECTION_AXES, bbox_order="raw_logit")
            self.assertEqual([b["cell"] for b in boxes], expected)
            if threshold == .2:
                self.assertAlmostEqual(boxes[1]["objectness"], .7)

    def test_axis_confidence_and_policy_validation_still_apply(self):
        raw = saturated_prediction()
        for mode in live.BBOX_ORDERS:
            with self.assertRaisesRegex(ValueError, "axes"):
                live.native_detection_boxes(raw, bbox_order=mode)
            for threshold in (0., 1., float("nan")):
                with self.assertRaisesRegex(ValueError, "confidence"):
                    live.native_detection_boxes(raw, threshold=threshold,
                        detection_axes=DETECTION_AXES, bbox_order=mode)
        with self.assertRaisesRegex(ValueError, "bbox order"):
            live.native_detection_boxes(raw, detection_axes=DETECTION_AXES, bbox_order="typo")

    def test_renderer_uses_selected_decoder_and_restores_global_on_success_or_failure(self):
        raw = saturated_prediction()
        original = live.viz.decode_detection_map

        def render(detection, **kwargs):
            self.assertNotIn("bbox_order", kwargs)
            return live.viz.decode_detection_map(detection, threshold=kwargs["threshold"],
                                                 detection_axes=kwargs["detection_axes"])

        with patch.object(live.viz, "render_bev_detection", side_effect=render):
            result = live.render_native_detection(raw, bbox_order="raw_logit", threshold=.5,
                                                 detection_axes=DETECTION_AXES)
        self.assertEqual(result[0]["cell"], (11, 21))
        self.assertIs(live.viz.decode_detection_map, original)
        with patch.object(live.viz, "render_bev_detection", side_effect=RuntimeError("render failed")):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                live.render_native_detection(raw, bbox_order="raw_logit")
        self.assertIs(live.viz.decode_detection_map, original)

    def test_child_render_path_receives_config_mode_and_leaves_waypoint_light_calls_unchanged(self):
        raw = saturated_prediction()
        config = {"visualization": {"detection_threshold": .5, "show_camera": False}}
        meta = dict(speed=2., yaw_degrees=0., target_point=[0., 0.],
                    detection_output_axes=DETECTION_AXES,
                    waypoint_output_axes="right_negative_forward")
        snapshot = (raw, np.zeros((5, 2)), np.zeros(2), np.zeros((1, 256)), meta)
        calls = []
        for mode in live.BBOX_ORDERS:
            live.resolve_bbox_order(config, mode)
            with patch.object(live, "render_native_detection", return_value="boxes") as bbox, \
                 patch.object(live.viz, "render_waypoints", return_value="path") as wp, \
                 patch.object(live.viz, "render_traffic_light", return_value="light") as light:
                figures = live.build_figures(config, snapshot, ["bev_detection", "waypoints", "traffic_light"])
            self.assertEqual([value for _, value in figures], ["boxes", "path", "light"])
            self.assertEqual(bbox.call_args.kwargs["bbox_order"], mode)
            calls.append((wp.call_args, light.call_args))
        self.assertEqual(calls[0][0].kwargs, calls[1][0].kwargs)
        self.assertEqual(calls[0][1].kwargs, calls[1][1].kwargs)

    def test_config_default_override_and_invalid_policy_before_model_or_sensors(self):
        config = {}
        self.assertEqual(live.resolve_bbox_order(config), "legacy_sigmoid")
        config["visualization"]["bbox_order"] = "raw_logit"
        self.assertEqual(live.resolve_bbox_order(config), "raw_logit")
        self.assertEqual(live.resolve_bbox_order(config, "legacy_sigmoid"), "legacy_sigmoid")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"visualization": {"bbox_order": "typo"}}))
            with patch("sys.argv", ["live", "--config", str(path)]), \
                 patch.object(live, "load_encoder") as load, \
                 patch.object(live, "LiveFrameSource") as source, \
                 patch.object(live.torch, "set_num_threads") as threads:
                with self.assertRaisesRegex(ValueError, "bbox_order"):
                    live.main()
            load.assert_not_called()
            source.assert_not_called()
            threads.assert_not_called()


if __name__ == "__main__":
    unittest.main()
