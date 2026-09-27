"""Train/live parity and output geometry, without sockets, CUDA or MORAI."""
import math
import json
import queue
import unittest

import cv2
import numpy as np
import torch

from data.src.loaders.morai_encoder_contract import (
    DETECTION_AXES, LiveInputAdapter, checkpoint_encoder_contract,
    sensor_record_encoder_contract, world_target_to_local,
)
from data.src.loaders.morai_transfer_adapter import MoraiLMDriveVisionAdapter, MoraiVisionAdapterConfig
from data.src.loaders.morai_transfer_targets import render_object_target
from inference.live_frame import LiveFrameSource
from inference.realtime_vision_encoder import FigureWriter
from utils.visualization.vision_encoder_visualization import decode_detection_map, _box_corners, forward_right


def adapter_config():
    return MoraiVisionAdapterConfig(use_previous_lidar_sweep=False,
                                   lidar_intensity_policy="reflectance_uint8_unit",
                                   lidar_translation_xyz=(1., 0., 1.48))


class Receiver:
    def __init__(self, value, age=0.):
        self.value, self.age = value, age

    def latest(self):
        return self.value, self.age


def replay_live_build(images, points, config, speed, target, seed, gps_age=0.):
    """Exercise live build including packet decode without opening sockets."""
    source = LiveFrameSource.__new__(LiveFrameSource)
    contract = sensor_record_encoder_contract([config], "gps")
    source.input_adapter = LiveInputAdapter(contract, lidar_frame="sensor_flu", lidar_to_base={
        "translation_xyz": config.lidar_translation_xyz,
        "rotation_xyzw": config.lidar_rotation_xyzw})
    source.contract = contract
    source.cameras = {}
    for name, rgb in images.items():
        ok, packet = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        assert ok
        source.cameras[name] = Receiver(packet.tobytes())
    source.lidar = Receiver(points)
    source.status = Receiver(dict(yaw=0., yaw_degrees=0., speed=speed))
    source.gps = Receiver(dict(latitude=37., longitude=126.), age=gps_age)
    source.velocity_source = "vehicle_status"
    source.velocity_mismatch = True
    source.route = np.array([[0., 0.], [100., 0.]])
    source.frames = 0
    source._ego_xy = lambda fix: (0., 0.)
    source._target_point = lambda *args: target
    batch, metadata = source.build("cpu", rng_seed=seed)
    return ({name: value[0] for name, value in batch.items()} if batch is not None else None), metadata


class RealtimeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_actual_live_build_matches_training_adapter_and_preserves_high_intensity(self):
        rng = np.random.RandomState(7)
        images = {"rgb_" + name: rng.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
                  for name in ("front", "left", "right", "rear")}
        points = np.array([[5, 2, 0, 200], [10, -2, 1, 255],
                           [101, 0, 0, 1], [0, 0, 0, 99]], dtype=np.float32)
        untouched = points.copy()
        config = adapter_config()
        trained = MoraiLMDriveVisionAdapter(config)
        expected = trained.camera_tensors(**{name + "_rgb": images["rgb_" + name]
                                            for name in ("front", "left", "right", "rear")})
        lidar, count = trained.lidar_to_raw_features(points, rng_seed=9)
        live, meta = replay_live_build(images, points, config, 5., np.array([-2., -10.]), 9)
        for name in expected:
            torch.testing.assert_close(live[name], expected[name], rtol=0, atol=0)
        np.testing.assert_array_equal(live["lidar"].numpy(), lidar)
        np.testing.assert_array_equal(points, untouched)
        self.assertEqual(int(live["num_points"]), count)
        self.assertTrue(np.isclose(lidar[:count, 3], 200 / 255).any())
        self.assertTrue(meta["velocity_contract_mismatch"])

    def test_figure_writer_preserves_output_axis_contract(self):
        writer = FigureWriter.__new__(FigureWriter)
        writer.period = 0.0
        writer.submitted = 0
        writer.dropped = 0
        writer._next = 0.0
        writer._channel = queue.Queue(maxsize=1)
        outputs = (
            torch.zeros((1, 2500, 8)),
            torch.zeros((1, 5, 2)),
            torch.zeros((1, 2)),
            torch.zeros((1, 1)),
            torch.zeros((1, 106, 256)),
        )
        meta = {
            "speed": 2.0,
            "yaw_degrees": 15.0,
            "target_point": [1.0, -10.0],
            "front_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "lidar_xy": np.zeros((1, 2)),
            "lidar_points": 1,
            "velocity_contract_mismatch": True,
            "waypoint_output_axes": "right_negative_forward",
            "detection_output_axes": DETECTION_AXES,
        }

        self.assertTrue(writer.submit(outputs, meta))
        snapshot_meta = writer._channel.get_nowait()[4]
        self.assertTrue(snapshot_meta["velocity_contract_mismatch"])
        self.assertEqual(snapshot_meta["waypoint_output_axes"], "right_negative_forward")
        self.assertEqual(snapshot_meta["detection_output_axes"], DETECTION_AXES)

    def test_rotated_world_targets_and_renderer_are_right_negative_forward(self):
        for yaw in (0., math.pi / 2, -1.2):
            # Ahead 10 m, left 2 m in the physical ego frame.
            c, s = math.cos(yaw), math.sin(yaw)
            target = world_target_to_local((10*c - 2*s, 10*s + 2*c), (0, 0), yaw)
            np.testing.assert_allclose(target, [-2, -10], atol=1e-6)
            forward, right = forward_right(target, waypoint_axes="right_negative_forward")
            np.testing.assert_allclose([forward[0], right[0]], [10, -2])
        _, legacy_right = forward_right([[-2, -10]], waypoint_axes="left_negative_forward")
        self.assertEqual(legacy_right[0], 2)

    def test_target_render_roundtrip_preserves_small_rotated_bbox(self):
        obj = dict(x=10., y=2., heading_deg=30., width=.8, length=1.2,
                   height=1., vel_x=0., vel_y=0., geometry_valid=True)
        target, valid, _ = render_object_target({"ego_raw": dict(x=0., y=0., yaw=0.)}, [obj])
        self.assertTrue(valid)
        index = int(np.argmax(target[:, 0]))
        predicted = target.copy()
        predicted[:, 0] = -100.
        predicted[index, 0] = 10.
        box = decode_detection_map(predicted, detection_axes=DETECTION_AXES)[0]
        np.testing.assert_allclose([box["lateral"], box["forward"]], [-2, 10], atol=1e-6)
        np.testing.assert_allclose(box["half_extent"], [.6, .4], atol=1e-6)
        np.testing.assert_allclose(box["heading"], [-.5, math.sqrt(.75)], atol=1e-6)
        corners = _box_corners(box)
        self.assertAlmostEqual(np.linalg.norm(corners[0] - corners[1]), .8, places=6)
        self.assertAlmostEqual(np.linalg.norm(corners[1] - corners[2]), 1.2, places=6)

    def test_calibration_is_explicit_and_cannot_be_applied_twice(self):
        contract = sensor_record_encoder_contract([adapter_config()], "gps")
        with self.assertRaisesRegex(ValueError, "explicit"):
            LiveInputAdapter(contract, lidar_frame="sensor_flu")
        with self.assertRaisesRegex(ValueError, "second"):
            LiveInputAdapter(contract, lidar_frame="base_flu", lidar_to_base={"anything": 1})
        cloud = np.array([[5., 2., 0., 200.]], dtype=np.float32)
        sensor = LiveInputAdapter(contract, lidar_frame="sensor_flu", lidar_to_base={
            "translation_xyz": (1, 0, 1.48), "rotation_xyzw": (0, 0, 0, 1)})
        base = LiveInputAdapter(contract, lidar_frame="base_flu")
        translated = cloud.copy(); translated[:, :3] += (1, 0, 1.48)
        a, _ = sensor.adapter.lidar_to_raw_features(cloud)
        b, _ = base.adapter.lidar_to_raw_features(translated)
        np.testing.assert_array_equal(a, b)

    def test_contract_sha_pin_and_known_legacy_constant_zero(self):
        config = adapter_config()
        contract = sensor_record_encoder_contract([config], "gps")
        sidecar = dict(schema="morai_encoder_contract_sidecar_v1", checkpoint_sha256="abc", contract=contract)
        self.assertEqual(checkpoint_encoder_contract({}, "abc", sidecar), contract)
        # JSON tuples become lists; semantically identical embedded metadata is valid.
        checkpoint_encoder_contract({'morai_encoder_contract': contract}, 'abc',
                                    json.loads(json.dumps(sidecar)))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            checkpoint_encoder_contract({}, "other", sidecar)
        with self.assertRaisesRegex(ValueError, "lacks"):
            checkpoint_encoder_contract({}, "abc")
        legacy = dict(morai_encoder_contract=dict(schema="morai_encoder_recipe_v1", adapter=contract["adapter"],
                     velocity_input="constant_zero", target_point_input="constant_zero"),
                      morai_sensor_contract=dict(waypoint_output_axes="right_negative_forward"))
        migrated = checkpoint_encoder_contract(legacy, "abc")
        self.assertEqual(migrated["velocity_input"], "constant_zero")
        self.assertEqual(migrated["target_point_input"], "constant_zero")
        adapted = LiveInputAdapter(migrated, lidar_frame='base_flu')
        images = {'rgb_' + name: np.zeros((16, 16, 3), np.uint8)
                  for name in ('front', 'left', 'right', 'rear')}
        inputs = adapted.assemble(images, np.array([[5., 0., 0., 200.]], np.float32),
                                  velocity=9., target_point=[3., -25.])
        self.assertEqual(inputs['velocity'].tolist(), [0.])
        self.assertEqual(inputs['target_point'].tolist(), [0., 0.])

    def test_gps_speed_mismatch_requires_explicit_diagnostic_flag(self):
        contract = sensor_record_encoder_contract([adapter_config()], "gps")
        with self.assertRaisesRegex(ValueError, "source-time GPS"):
            LiveFrameSource({}, encoder_contract=contract, lidar_frame="base_flu")

    def test_stale_gps_never_fabricates_zero_navigation_target(self):
        images = {'rgb_' + name: np.zeros((16, 16, 3), np.uint8)
                  for name in ('front', 'left', 'right', 'rear')}
        inputs, meta = replay_live_build(images, np.array([[5., 0., 0., 200.]], np.float32),
                                         adapter_config(), 3., np.array([0., -25.]), 0, gps_age=5.)
        self.assertIsNone(inputs)
        self.assertIn('gps/route (navigation target unavailable)', meta['missing'])


if __name__ == "__main__":
    unittest.main()
