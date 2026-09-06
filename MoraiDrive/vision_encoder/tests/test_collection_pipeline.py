"""Exercise the real collector writers, split builder, dataset and loader."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


MORAI = Path(__file__).resolve().parents[2]
READY = all(importlib.util.find_spec(name) for name in ("torch", "cv2", "skimage"))


@unittest.skipUnless(READY, "Install the vision encoder dependencies")
class CollectionPipelineTest(unittest.TestCase):
    def test_writer_split_and_training_batch(self):
        sys.path.insert(0, str(MORAI / "vision_encoder"))
        sys.path.insert(0, str(MORAI / "data_collection"))
        import numpy as np
        from schema import EgoState, ActorData, build_measurement_dict
        from writer.dataset_writer import DatasetWriter
        from writer.image_writer import ImageWriter
        from writer.lidar_writer import LidarWriter
        from timm.data import create_morai_dataset, create_morai_loader
        from timm.data.morai_routes import resolve_route_split

        with tempfile.TemporaryDirectory() as temp:
            for name, success in (("a", True), ("b", True), ("failed", False)):
                writer = DatasetWriter(temp, "job", name, append_timestamp=False)
                images = ImageWriter(writer.output_dir, ("front", "left", "right", "rear"))
                lidar = LidarWriter(writer.output_dir)
                for frame in range(6):
                    ego = EgoState(x=float(frame), y=0, theta=0, speed=1,
                                   x_command=frame + 25, y_command=0)
                    actors = {"7": ActorData(loc=[frame + 10, 0, 0], ori=[1, 0],
                                             box=[2, 1], vel=[1, 0, 0], tpe=0)}
                    measurement = build_measurement_dict(ego, actors=actors)
                    writer.write_frame(frame, measurement, {
                        "timestamp_ns": frame * 100_000_000,
                        "navigation": {"route_valid": frame != 1},
                    }, actors=measurement["actors_data"])
                    images.write(frame, {view: np.full((180, 240, 3), 120, np.uint8)
                                         for view in ("front", "left", "right", "rear")})
                    lidar.write(frame, np.tile([10, 2, 3, 180], (40001, 1)).astype(np.float32))
                writer.finalize(success=success)
            root = Path(temp) / "job"
            subprocess.run([sys.executable, str(MORAI / "data_collection/tools/build_dataset_splits.py"),
                            str(root)], check=True, capture_output=True)
            train, val = resolve_route_split(root, "dataset_index.txt", "dataset_index_test.txt")
            self.assertEqual(set(train + val), {"a", "b"})
            dataset = create_morai_dataset("morai", str(root), routes=train,
                                           with_lidar=True, multi_view=True, waypoint_interval=1)
            self.assertEqual([frame for _, frame in dataset.route_frames], [0, 2, 3, 4, 5])
            loader = create_morai_loader(dataset, (3, 224, 224), batch_size=2,
                                         multi_view_input_size=(3, 128, 128), num_workers=0)
            inputs, targets = next(iter(loader))
            self.assertEqual(tuple(inputs["rgb_front"].shape), (2, 3, 224, 224))
            self.assertEqual(tuple(inputs["rgb_center"].shape), (2, 3, 128, 128))
            self.assertEqual(tuple(inputs["lidar"].shape), (2, 40000, 4))
            self.assertTrue((inputs["lidar"][:, :, 3] == 180).all())
            self.assertEqual(tuple(targets[4].shape), (2, 2500, 8))
            self.assertTrue(targets[4][:, :, 0].max() > 0)
            detection = targets[4][0].numpy().reshape(50, 50, 8)
            row, col = np.unravel_index(detection[:, :, 0].argmax(), (50, 50))
            record = detection[row, col]
            # Same object ahead must be above ego in BOTH detection and waypoint BEV.
            np.testing.assert_allclose([col - 24.5 + record[1] * 3.5,
                                        29.5 - row + record[2] * 3.5], [0, 10], atol=1e-6)
            np.testing.assert_allclose(record[3:5], [0, -1], atol=1e-6)
            np.testing.assert_allclose(inputs["target_point"][0].numpy(), [0, -25], atol=1e-6)
            # Invalid frame 1 is excluded as an input, but original trajectory time is retained.
            np.testing.assert_allclose(targets[1][0].numpy(), [[0, -i] for i in range(1, 6)], atol=1e-6)
            self.assertTrue(targets[5].max() > 0)  # writer backfilled future_waypoints


if __name__ == "__main__":
    unittest.main()
