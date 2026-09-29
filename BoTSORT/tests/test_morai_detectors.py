import unittest
from types import SimpleNamespace
import numpy as np
import torch
from tools.morai_detectors import ultralytics_detections, YOLOv7Detector, create_detector


class DetectorAdapterTests(unittest.TestCase):
    def test_coordinates_score_and_class_stay_aligned(self):
        class Boxes:
            xyxy = torch.tensor([[12.5, 30, 300, 420], [1, 2, 3, 4]])
            conf = torch.tensor([0.8, 0.3])
            cls = torch.tensor([5, 0])
            def __len__(self):
                return 2
        rows = ultralytics_detections(SimpleNamespace(boxes=Boxes()))
        np.testing.assert_allclose(rows, [[12.5,30,300,420,0.8,5],[1,2,3,4,0.3,0]])
        self.assertEqual(rows.dtype, np.float32)

    def test_empty_results_have_six_columns(self):
        for boxes in (None, []):
            rows = ultralytics_detections(SimpleNamespace(boxes=boxes))
            self.assertEqual(rows.shape, (0, 6))


class YoloTrainingTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA training regression")
    def test_ota_assignment_and_backward_on_cuda(self):
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "yolov7"))
        try:
            from utils.loss import ComputeLossOTA
        finally:
            sys.path.pop(0)
        model = torch.nn.Module()
        model.register_parameter("device_marker", torch.nn.Parameter(torch.zeros(1, device="cuda")))
        model.hyp = dict(cls_pw=1., obj_pw=1., fl_gamma=0., anchor_t=4., box=.05, obj=1., cls=.5)
        model.gr = 1.
        model.model = [SimpleNamespace(nl=3, na=3, nc=3,
            stride=torch.tensor([8., 16., 32.], device="cuda"),
            anchors=torch.tensor([[[1., 1.], [2., 2.], [3., 3.]]] * 3, device="cuda"))]
        predictions = [torch.zeros(2, 3, size, size, 8, device="cuda", requires_grad=True)
                       for size in (16, 8, 4)]
        targets = torch.tensor([[0., 1., .5, .5, .2, .3]], device="cuda")
        images = torch.zeros(2, 3, 128, 128, device="cuda")
        loss, _ = ComputeLossOTA(model)(predictions, targets, images)
        self.assertTrue(torch.isfinite(loss).all())
        loss.backward()
        for prediction in predictions:
            self.assertIsNotNone(prediction.grad)
            self.assertTrue(torch.isfinite(prediction.grad).all())


class YoloRuntimeTests(unittest.TestCase):
    def test_yolov7_options_reject_other_detector_before_loading(self):
        for options in ({'yolov7_runtime': 'cuda-graph'}, {'postprocess_device': 'cpu'},
                        {'yolov7_runtime': 'tensorrt'}, {'engine_manifest': 'engines.json'}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'only to'):
                create_detector(SimpleNamespace(detector='yolo26', **options))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA Graph runtime regression")
    def test_alternating_shapes_fresh_inputs_and_unprepared_shape(self):
        # A small CUDA model reproduces the YOLO head's shape-dependent grid
        # replacement without downloading weights or requiring simulator input.
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 2, 1)
                self.model = [SimpleNamespace(grid=[torch.empty(0, device='cuda')])]

            def forward(self, x):
                head = self.model[-1]
                if head.grid[0].shape != x.shape[2:]:
                    head.grid[0] = torch.arange(x.shape[2] * x.shape[3],
                        device=x.device, dtype=x.dtype).reshape(x.shape[2:]) / 100
                return (self.conv(x) + head.grid[0],)

        detector = YOLOv7Detector.__new__(YOLOv7Detector)
        detector.device = torch.device('cuda', torch.cuda.current_device())
        detector.model = Model().to(detector.device, memory_format=torch.channels_last).eval()
        detector.runtime = 'cuda-graph'
        detector.postprocess_device = 'model'
        detector._graphs = {}
        detector._eager_fallbacks = 0
        prepared = [(1, 3, 8, 12), (1, 3, 12, 16)]
        for shape in prepared:
            detector._capture(torch.zeros(shape, device=detector.device)
                              .contiguous(memory_format=torch.channels_last))
        with torch.no_grad():
            for index, shape in enumerate([*prepared, prepared[0], (1, 3, 8, 16), *prepared]):
                x = torch.full(shape, index / 10, device=detector.device)
                x = x.contiguous(memory_format=torch.channels_last)
                expected = detector.model(x)[0]
                actual = detector._forward(x).clone()
                torch.testing.assert_close(actual, expected)
        self.assertEqual(len(detector._graphs), 2)
        self.assertEqual(detector.runtime_info()['eager_fallback_frames'], 1)
        self.assertEqual(detector.runtime_info()['graph_shapes'], [list(s) for s in prepared])
        self.assertNotEqual(detector._graphs[prepared[0]][1].data_ptr(),
                            detector._graphs[prepared[1]][1].data_ptr())

    @unittest.skipUnless(torch.cuda.is_available(), "GPU/CPU postprocessing regression")
    def test_postprocess_preserves_classes_duplicates_and_empty_input(self):
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'yolov7'))
        try:
            detector = YOLOv7Detector.__new__(YOLOv7Detector)
            detector.args = SimpleNamespace(conf_thres=.05, iou_thres=.5, classes=None, agnostic_nms=False)
            detector._tensor = lambda image: torch.zeros((1, 3, 64, 64), device='cuda')
            image = np.zeros((64, 64, 3), np.uint8)
            prediction = torch.tensor([[[20, 20, 8, 8, .9, .9, .1],
                                        [20, 20, 8, 8, .8, .9, .1],
                                        [20, 20, 8, 8, .95, .1, .9]]],
                                      device='cuda', dtype=torch.float16)
            for classes, agnostic, count in [(None, False, 2), ([1], False, 1), (None, True, 1)]:
                detector.args.classes, detector.args.agnostic_nms = classes, agnostic
                detector._forward = lambda tensor: prediction.clone()
                detector.postprocess_device = 'model'
                expected = detector.detect(image)
                detector.postprocess_device = 'cpu'
                actual = detector.detect(image)
                self.assertEqual(len(actual), count)
                np.testing.assert_allclose(actual, expected, atol=1e-6)
            detector._forward = lambda tensor: torch.zeros((1, 0, 7), device='cuda', dtype=torch.float16)
            self.assertEqual(detector.detect(image).shape, (0, 6))
        finally:
            sys.path.pop(0)


class TensorRTArtifactTests(unittest.TestCase):
    def test_wrong_weights_and_changed_engine_are_rejected(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from tools.morai_tensorrt import load_engine_manifest, read_engine_bytes
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = root / 'weights.pt'
            weights.write_bytes(b'weights-a')
            engine = root / 'model.engine'
            engine.write_bytes(b'engine-a')
            metadata = {'schema_version': 1, 'precision': 'fp16', 'size': 64, 'stride': 32,
                'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(), 'names': ['box'],
                'engines': [{'shape': [1, 3, 32, 64], 'path': 'model.engine',
                    'sha256': hashlib.sha256(engine.read_bytes()).hexdigest()}]}
            manifest = root / 'engines.json'
            manifest.write_text(json.dumps(metadata))
            path, loaded = load_engine_manifest(manifest, weights, 64)
            self.assertEqual(read_engine_bytes(path, loaded['engines'][0]), b'engine-a')
            with self.assertRaisesRegex(ValueError, 'input size'):
                load_engine_manifest(manifest, weights, 960)
            engine.write_bytes(b'engine-b')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                read_engine_bytes(path, loaded['engines'][0])
            weights.write_bytes(b'weights-b')
            with self.assertRaisesRegex(ValueError, 'weights mismatch'):
                load_engine_manifest(manifest, weights, 64)
            weights.write_bytes(b'weights-a')
            metadata['engines'].append(metadata['engines'][0])
            manifest.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_engine_manifest(manifest, weights, 64)

    def test_engine_options_fail_before_loading_optional_dependencies(self):
        with self.assertRaisesRegex(ValueError, 'requires --engine-manifest'):
            create_detector(SimpleNamespace(detector='yolov7', yolov7_runtime='tensorrt'))
        with self.assertRaisesRegex(ValueError, 'applies only'):
            create_detector(SimpleNamespace(detector='yolov7', yolov7_runtime='eager',
                                            engine_manifest='engines.json'))
        with self.assertRaisesRegex(ValueError, 'requires --fp16'):
            create_detector(SimpleNamespace(detector='yolov7', yolov7_runtime='tensorrt',
                engine_manifest='engines.json', fp16=False, device='0'))

    def test_eager_factory_does_not_require_tensorrt(self):
        from unittest.mock import patch
        marker = object()
        with patch.dict('sys.modules', {'tensorrt': None}), \
                patch('tools.morai_detectors.YOLOv7Detector', return_value=marker):
            self.assertIs(create_detector(SimpleNamespace(detector='yolov7')), marker)


if __name__ == "__main__":
    unittest.main()
