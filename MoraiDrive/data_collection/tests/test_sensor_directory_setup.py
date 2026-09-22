import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = (Path(__file__).resolve().parents[1] / "internal_sensor_record"
          / "tools/configure_local_machine.py")
spec = importlib.util.spec_from_file_location("configure_local_machine", SCRIPT)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)
ROLES = ("CAMERA_1", "CAMERA_2", "CAMERA_3", "CAMERA_4", "LIDAR_5", "GPS_6", "IMU_7")


class SensorDirectorySetupTests(unittest.TestCase):
    def test_dangling_links_are_preserved_without_recreating_old_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sensors = root / "SensorData"
            sensors.mkdir()
            old = root / "native_sessions/campaign_01_lap_003"
            for role in ROLES:
                (sensors / role).symlink_to(old / role, target_is_directory=True)
            created, deferred = setup.prepare_sensor_directories(sensors)
            self.assertEqual(created, [])
            self.assertEqual(set(deferred), {str(sensors / r) for r in ROLES})
            self.assertFalse(old.exists())
            for role in ROLES:
                self.assertTrue((sensors / role).is_symlink())
                self.assertEqual((sensors / role).resolve(), old / role)

    def test_fresh_setup_and_repeated_setup_preserve_files(self):
        with tempfile.TemporaryDirectory() as directory:
            sensors = Path(directory) / "SensorData"
            created, deferred = setup.prepare_sensor_directories(sensors)
            self.assertEqual(len(created), 7)
            self.assertEqual(deferred, [])
            sample = sensors / "CAMERA_1/image.bin"
            sample.write_bytes(b"existing capture")
            self.assertEqual(setup.prepare_sensor_directories(sensors), ([], []))
            self.assertEqual(sample.read_bytes(), b"existing capture")

    def test_live_links_preserve_existing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sensors, target = root / "SensorData", root / "active"
            sensors.mkdir()
            target.mkdir()
            (target / "image.bin").write_bytes(b"preserve")
            (sensors / "CAMERA_1").symlink_to(target, target_is_directory=True)
            created, deferred = setup.prepare_sensor_directories(sensors)
            self.assertEqual(len(created), 6)
            self.assertEqual(deferred, [])
            self.assertTrue((sensors / "CAMERA_1").is_symlink())
            self.assertEqual((target / "image.bin").read_bytes(), b"preserve")

    def test_regular_file_is_rejected_without_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            sensors = Path(directory)
            (sensors / "CAMERA_1").write_bytes(b"preserve")
            with self.assertRaisesRegex(RuntimeError, "not a directory"):
                setup.prepare_sensor_directories(sensors)
            self.assertEqual((sensors / "CAMERA_1").read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
