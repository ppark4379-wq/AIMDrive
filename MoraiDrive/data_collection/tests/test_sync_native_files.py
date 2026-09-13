import os
import sys
import tempfile
import types
import unittest

import cv2
import numpy as np


DATA_COLLECTION = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MORAI_DRIVE = os.path.dirname(DATA_COLLECTION)
if DATA_COLLECTION not in sys.path:
    sys.path.insert(0, DATA_COLLECTION)
if MORAI_DRIVE not in sys.path:
    sys.path.insert(0, MORAI_DRIVE)

from sources.morai_sync_file_source import MoraiSyncFileSource
from sources.native_file_collector import (
    _measurement_for_record, run_sync_native_files,
)
from sources.sync_capture_manifest import SyncCaptureManifest
from grpc_ctrl.morai_state_reader import GrpcLabelSnapshot
from ros.sensor_types import GTObject, TrafficLightState
from sync_runtime.capture_journal import CaptureJournal
from sync_runtime.frame_record import CaptureFrameRecord, NativeFrameCaptureScheduler


def _record(frame):
    empty_ego = {
        "position": {"x": 1.0, "y": 2.0, "z": 0.0},
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "heading": 0.0,
    }
    return CaptureFrameRecord(
        schema_version=1,
        session_id="test",
        scenario_id="scenario",
        native_sync_frame=frame,
        request_frame=frame,
        response_frame=frame + 1,
        capture_requested=True,
        capture_stride=5,
        ego_state_for_frame=empty_ego,
        ego_state_source_frame=frame,
        next_ego_state=empty_ego,
        next_ego_state_source_frame=frame + 1,
        applied_control={"accel": 0.0, "brake": 1.0, "front_steer": 0.0},
        control_source_sequence=1,
        control_reused=False,
        control_age_frames=0,
        control_mode="fresh",
        expected_sensor_files={
            "camera_front": "CAMERA_1/{}_Intensity.png".format(frame),
            "camera_left": "CAMERA_2/{}_Intensity.png".format(frame),
            "camera_right": "CAMERA_3/{}_Intensity.png".format(frame),
            "camera_rear": "CAMERA_4/{}_Intensity.png".format(frame),
            "lidar": "LIDAR_5/{}_Intensity.bin".format(frame),
        },
    )


class SyncNativeFilesTest(unittest.TestCase):
    def test_native_measurement_uses_grpc_actor_labels(self):
        labels = GrpcLabelSnapshot(
            tick=10,
            simulator_timestamp_raw=200000,
            ego_x=1.0,
            ego_y=2.0,
            ego_heading_deg=0.0,
            objects=[GTObject(
                obj_id=7, obj_type="vehicle", x=11.0, y=2.0, z=0.0,
                vel_x=3.0, vel_y=0.0, heading_deg=0.0,
                width=2.0, length=4.0, height=1.5,
            )],
            traffic_lights=[TrafficLightState(tl_id="TL-A", state=1)],
        )
        args = types.SimpleNamespace(
            require_route_valid=False,
            route_valid_max_error=3.0,
            morai_to_lmdrive_steer_sign=1.0,
            control_max_steer_deg=40.0,
        )

        measurement, navigation, attached = _measurement_for_record(
            _record(10).to_dict(), None, args, labels
        )

        self.assertIn("7", measurement["actors_data"])
        self.assertEqual(measurement["actors_data"]["7"]["box"], [2.0, 1.0])
        self.assertEqual(attached["objects"][0].obj_id, 7)
        self.assertIsNone(attached["affected_light_state"])
        self.assertIsNone(navigation["affected_light_state_raw"])

    def test_capture_scheduler_bootstrap_then_native_stride(self):
        scheduler = NativeFrameCaptureScheduler(True, 5)
        self.assertFalse(scheduler.should_capture(103))
        scheduler.observe_bootstrap_response(104)
        captured = [frame for frame in range(104, 120) if scheduler.should_capture(frame)]
        self.assertEqual(captured, [104, 109, 114, 119])

    def test_journal_manifest_and_exact_native_file_load(self):
        with tempfile.TemporaryDirectory() as root:
            manifest_path = os.path.join(root, "capture_manifest.jsonl")
            journal = CaptureJournal(manifest_path, queue_size=4)
            self.assertTrue(journal.submit(_record(104)))
            self.assertTrue(journal.submit(_record(109)))
            journal.close()

            with SyncCaptureManifest(manifest_path) as manifest:
                records = manifest.poll()
            self.assertEqual([r["native_sync_frame"] for r in records], [104, 109])

            frame = 104
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            for directory in ("CAMERA_1", "CAMERA_2", "CAMERA_3", "CAMERA_4"):
                path = os.path.join(root, directory)
                os.makedirs(path)
                self.assertTrue(cv2.imwrite(
                    os.path.join(path, "{}_Intensity.png".format(frame)), image
                ))
            os.makedirs(os.path.join(root, "LIDAR_5"))
            np.asarray([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32).tofile(
                os.path.join(root, "LIDAR_5", "{}_Intensity.bin".format(frame))
            )
            payload = MoraiSyncFileSource(
                root, timeout_sec=0.2, poll_interval_sec=0.005, stable_checks=1
            ).load(records[0])
            self.assertEqual(payload.native_sync_frame, 104)
            self.assertEqual(set(payload.cameras), {"front", "left", "right", "rear"})
            self.assertEqual(payload.lidar_points.shape, (1, 4))

    def test_ten_sparse_frames_become_contiguous_dataset_samples(self):
        with tempfile.TemporaryDirectory() as root:
            session_id = "same-session-id"
            camera_base = os.path.join(root, "camera_base")
            lidar_base = os.path.join(root, "lidar_base")
            camera_root = os.path.join(camera_base, session_id)
            lidar_root = os.path.join(lidar_base, session_id)
            for directory in ("CAMERA_1", "CAMERA_2", "CAMERA_3", "CAMERA_4"):
                os.makedirs(os.path.join(camera_root, directory))
            os.makedirs(os.path.join(lidar_root, "LIDAR_5"))
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            manifest_path = os.path.join(root, "capture_manifest.jsonl")
            journal = CaptureJournal(manifest_path, queue_size=16)
            for frame in range(104, 159, 5):
                self.assertTrue(journal.submit(_record(frame)))
                for directory in ("CAMERA_1", "CAMERA_2", "CAMERA_3", "CAMERA_4"):
                    cv2.imwrite(
                        os.path.join(camera_root, directory, "{}_Intensity.png".format(frame)),
                        image,
                    )
                np.asarray([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32).tofile(
                    os.path.join(lidar_root, "LIDAR_5", "{}_Intensity.bin".format(frame))
                )
            journal.close()

            args = types.SimpleNamespace(
                map_dir="", route_definition="", global_path="", route_spec="",
                route_definition_key="full_loop", route_start_index=None,
                route="route", job="job", map_id="map", no_route_timestamp=True,
                global_info="", morai_sync_root="", morai_sync_base="",
                morai_camera_sync_base=camera_base,
                morai_lidar_sync_base=lidar_base,
                writer_queue_size=4, native_file_timeout=0.2,
                native_poll_interval=0.005, native_stable_checks=1,
                capture_manifest=manifest_path, manifest_poll_interval=0.005,
                duration=1.0, max_samples=4, require_route_valid=False,
                route_valid_max_error=3.0, morai_to_lmdrive_steer_sign=1.0,
                control_max_steer_deg=40.0,
            )
            dataset_config = types.SimpleNamespace(
                append_timestamp=False, image_ext="jpg", image_quality=90
            )
            first_output, first_count = run_sync_native_files(
                args, os.path.join(root, "dataset"), dataset_config, None
            )
            self.assertEqual(first_count, 4)

            # Restart against the same Runtime manifest. It must reuse the
            # route, skip frames through the checkpoint, and append 0004+.
            args.max_samples = 6
            output, count = run_sync_native_files(
                args, os.path.join(root, "dataset"), dataset_config, None
            )
            self.assertEqual(output, first_output)
            self.assertEqual(count, 10)
            self.assertEqual(
                sorted(os.listdir(os.path.join(output, "rgb_front"))),
                ["{:04d}.jpg".format(i) for i in range(10)],
            )
            self.assertEqual(
                sorted(os.listdir(os.path.join(output, "lidar"))),
                ["{:04d}.npy".format(i) for i in range(10)],
            )
            with open(
                os.path.join(output, "measurements_full", "0009.json"),
                encoding="utf-8",
            ) as source:
                measurement = __import__("json").load(source)
            self.assertEqual(measurement["native_sync_frame"], 149)
            self.assertFalse(measurement["object_available"])
            self.assertEqual(measurement["traffic_scope"], "unavailable")
            checkpoint_dir = os.path.join(root, "dataset", "job", ".collector_state")
            checkpoint_path = os.path.join(
                checkpoint_dir, os.listdir(checkpoint_dir)[0]
            )
            with open(checkpoint_path, encoding="utf-8") as source:
                checkpoint = __import__("json").load(source)
            self.assertEqual(checkpoint["last_consumed_native_frame"], 149)
            self.assertEqual(checkpoint["next_sample_index"], 10)
            with open(
                os.path.join(root, "dataset", "job", "dataset_index.txt"),
                encoding="utf-8",
            ) as source:
                self.assertEqual(source.read(), "data/route/ 10\n")

    def test_native_collector_writes_grpc_labels_to_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            session_id = "label-session"
            camera_base = os.path.join(root, "camera_base")
            lidar_base = os.path.join(root, "lidar_base")
            camera_root = os.path.join(camera_base, session_id)
            lidar_root = os.path.join(lidar_base, session_id)
            for directory in ("CAMERA_1", "CAMERA_2", "CAMERA_3", "CAMERA_4"):
                os.makedirs(os.path.join(camera_root, directory))
            os.makedirs(os.path.join(lidar_root, "LIDAR_5"))

            frame = 104
            manifest_path = os.path.join(root, "capture_manifest.jsonl")
            journal = CaptureJournal(manifest_path, queue_size=2)
            record = _record(frame)
            self.assertTrue(journal.submit(record))
            journal.close()
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            for directory in ("CAMERA_1", "CAMERA_2", "CAMERA_3", "CAMERA_4"):
                cv2.imwrite(
                    os.path.join(camera_root, directory, "104_Intensity.png"), image
                )
            np.asarray([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32).tofile(
                os.path.join(lidar_root, "LIDAR_5", "104_Intensity.bin")
            )

            labels = GrpcLabelSnapshot(
                tick=frame,
                simulator_timestamp_raw=208000,
                ego_x=1.0,
                ego_y=2.0,
                ego_heading_deg=0.0,
                objects=[GTObject(
                    obj_id=7, obj_type="vehicle", x=11.0, y=2.0, z=0.0,
                    vel_x=3.0, vel_y=0.0, heading_deg=0.0,
                    width=2.0, length=4.0, height=1.5,
                )],
                traffic_lights=[TrafficLightState(tl_id="TL-A", state=1)],
            )

            class FakeHistory:
                is_aligned = False
                closed = False

                def start(self):
                    pass

                def wait_for_record(self, candidate, _timeout):
                    self.is_aligned = True
                    return labels if int(candidate["native_sync_frame"]) == frame else None

                def close(self):
                    self.closed = True

            history = FakeHistory()
            args = types.SimpleNamespace(
                map_dir="", route_definition="", global_path="", route_spec="",
                route_definition_key="full_loop", route_start_index=None,
                route="route", job="job", map_id="map", no_route_timestamp=True,
                global_info="", morai_sync_root="", morai_sync_base="",
                morai_camera_sync_base=camera_base,
                morai_lidar_sync_base=lidar_base,
                writer_queue_size=2, native_file_timeout=0.2,
                native_poll_interval=0.005, native_stable_checks=1,
                capture_manifest=manifest_path, manifest_poll_interval=0.005,
                duration=1.0, max_samples=1, require_route_valid=False,
                route_valid_max_error=3.0, morai_to_lmdrive_steer_sign=1.0,
                control_max_steer_deg=40.0,
            )
            dataset_config = types.SimpleNamespace(
                append_timestamp=False, image_ext="jpg", image_quality=90
            )

            output, count = run_sync_native_files(
                args,
                os.path.join(root, "dataset"),
                dataset_config,
                None,
                grpc_config={"state_required_timeout_sec": 0.1},
                state_history=history,
            )

            self.assertEqual(count, 1)
            self.assertTrue(history.closed)
            with open(
                os.path.join(output, "measurements_full", "0000.json"),
                encoding="utf-8",
            ) as source:
                full = __import__("json").load(source)
            self.assertEqual(full["simulator_labels_source"], "grpc")
            self.assertEqual(full["grpc_label_tick_raw"], frame)
            self.assertTrue(full["gt_objects_available"])
            self.assertEqual(full["gt_objects_raw"][0]["obj_id"], 7)
            self.assertEqual(full["traffic_lights"][0]["tl_id"], "TL-A")
            self.assertIn("7", full["actors_data"])


if __name__ == "__main__":
    unittest.main()
