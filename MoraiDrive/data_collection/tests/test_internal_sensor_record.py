import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


DATA_COLLECTION = Path(__file__).resolve().parents[1]
INTERNAL = DATA_COLLECTION / "internal_sensor_record"
sys.path.insert(0, str(INTERNAL))
sys.path.insert(0, str(DATA_COLLECTION))

from async_capture.journal import RawWriter, records, verify_run
from async_capture.health_policy import (
    SimulationPauseGrace, is_coherent_simulation_pause,
)
from async_capture.clock_health import evaluate as evaluate_local_clocks
from async_capture.sensor_record import filename_candidates
from bridge.camera_ingress_filter import dropped_publish_stream
from bridge.bridge_health import (
    BridgeMetrics, IMPLEMENTATION, OBSERVED_STREAMS, REQUIRED_STREAMS,
    boot_id, check_bridge, implementation_hashes,
)
from async_capture.internal_sensor_record_contract import (
    _ros, sensor_set_contract_sha256,
)
from network_config import bridge_ip, runtime_environment
from native_capture.ground_truth import default_grpc_target
from sources.morai_sync_file_source import REQUIRED_SENSOR_DIRS


class InternalSensorRecordTest(unittest.TestCase):
    def test_bounded_wsl_clock_slew_is_recorded_without_reboot_loop(self):
        samples = [
            {"monotonic_ns": 0, "monotonic_raw_ns": 0, "wall_ns": 0},
            {
                "monotonic_ns": 1_009_908_000,
                "monotonic_raw_ns": 1_000_000_000,
                "wall_ns": 1_009_908_000,
            },
        ]
        report = evaluate_local_clocks(samples)
        self.assertTrue(report["passed"])
        self.assertEqual(report["warnings"], ["monotonic_rate_slew_recorded"])
        self.assertAlmostEqual(report["rate_error_ppm"], 9908.0)

        samples[-1]["monotonic_ns"] = 1_083_333_000
        samples[-1]["wall_ns"] = 1_083_333_000
        report = evaluate_local_clocks(samples)
        self.assertFalse(report["passed"])
        self.assertIn("monotonic_rate_differs_from_raw", report["failures"])

    def test_simulation_pause_grace_never_hides_partial_sensor_loss(self):
        coherent = [
            "ros_clock_not_advancing",
            "bridge_missing_stream:Camera-1",
            "bridge_missing_stream:/lidar3D",
        ]
        self.assertTrue(is_coherent_simulation_pause(coherent))
        self.assertFalse(is_coherent_simulation_pause(
            ["bridge_missing_stream:Camera-1"]
        ))
        self.assertFalse(is_coherent_simulation_pause(
            coherent + ["bridge_delayed_stream:Camera-2"]
        ))

        guard = SimulationPauseGrace(8)
        self.assertEqual(guard.observe(coherent, 10_000_000_000)["action"], "hold")
        recovered = guard.observe([], 13_500_000_000)
        self.assertEqual(recovered["action"], "recovered")
        self.assertEqual(recovered["incident"]["duration_ns"], 3_500_000_000)

        self.assertEqual(
            guard.observe(["bridge_missing_stream:Camera-1"], 14_000_000_000)["action"],
            "fail",
        )

    def test_simulation_pause_grace_remains_bounded(self):
        failures = ["missing_or_stale_ros_clock", "bridge_missing_stream:/imu"]
        guard = SimulationPauseGrace(8)
        self.assertEqual(guard.observe(failures, 1_000_000_000)["action"], "hold")
        self.assertEqual(guard.observe(failures, 9_000_000_000)["action"], "hold")
        decision = guard.observe(failures, 9_000_000_001)
        self.assertEqual(decision["action"], "fail")
        self.assertEqual(decision["reason"], "coherent_simulation_pause_exceeded_grace")

    def test_raw_journal_round_trip_and_corruption_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            writer = RawWriter(root, {"purpose": "unit_test"})
            self.assertTrue(writer.submit(
                {
                    "role": "ego",
                    "source_stamp_ns": 123,
                    "receive_monotonic_ns": 456,
                    "receive_wall_ns": 789,
                },
                b"payload",
            ))
            manifest = writer.close()
            self.assertEqual(manifest["status"], "completed")
            verified = verify_run(root)
            row, payload = list(records(root, verified=verified))[0]
            self.assertEqual(row["source_stamp_ns"], 123)
            self.assertEqual(row["receive_monotonic_ns"], 456)
            self.assertEqual(payload, b"payload")

            chunk = root / manifest["chunks"][0]["path"]
            with chunk.open("ab") as stream:
                stream.write(b"x")
            with self.assertRaises(ValueError):
                verify_run(root)

    def test_native_filename_keeps_both_twelve_hour_candidates(self):
        candidates = filename_candidates(
            "01-01-01-01-02-03-123456_Intensity.png", "camera_1"
        )
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[1] - candidates[0], 12 * 3600 * 10**9)

    def test_bridge_camera_drop_is_explicit_and_lidar_is_forwarded(self):
        camera = json.dumps({
            "op": "publish",
            "topic": "/image_jpeg/compressed",
            "msg": {"data": "large", "header": {"frame_id": "Camera-4"}},
        }).encode()
        lidar = json.dumps({
            "op": "publish", "topic": "/lidar3D", "msg": {"data": [1.0]},
        }).encode()
        self.assertEqual(
            dropped_publish_stream(camera, drop_camera=True),
            ("/image_jpeg/compressed", "Camera-4"),
        )
        self.assertIsNone(dropped_publish_stream(lidar, drop_camera=True))

        metrics = BridgeMetrics()
        metrics.observe({
            "receive_ns": 10, "end_ns": 11, "client_id": 7,
            "queue_ms": 0.0, "processing_ms": 0.01, "residual": 0,
            "errors": 0, "op": "publish", "topic": "/Ego_topic",
        })
        self.assertEqual(metrics.snapshot(11)["streams"]["/Ego_topic"]["samples"], 1)
        self.assertEqual(
            set(implementation_hashes()),
            {"rosbridge_ingress.py", "bridge_health.py", "production_bridge.py",
             "camera_ingress_filter.py"},
        )

    def test_bridge_observes_clock_without_making_it_a_sensor_stream(self):
        metrics = BridgeMetrics()
        metrics.observe({
            "receive_ns": 10, "end_ns": 11, "client_id": 9,
            "queue_ms": 0.0, "processing_ms": 0.01, "residual": 0,
            "errors": 0, "op": "publish", "topic": "/clock",
        })
        self.assertIn("/clock", OBSERVED_STREAMS)
        self.assertNotIn("/clock", REQUIRED_STREAMS)
        self.assertEqual(metrics.snapshot(11)["streams"]["/clock"]["samples"], 1)

    def test_bridge_preflight_separates_stale_from_integrity_failures(self):
        now = 10_000_000_000
        health = {
            "implementation": IMPLEMENTATION,
            "binary_json_patch": True,
            "boot_id": boot_id(),
            "code_sha256": implementation_hashes(),
            "monotonic_ns": now,
            "listening": True,
            "recent_errors": 0,
            "clients": [],
            "streams": {
                role: {
                    "samples": 2,
                    "receive_age_ms": 0,
                    "queue_delay_ms": {"max": 0},
                    "client_ids": [role],
                }
                for role in REQUIRED_STREAMS
            },
        }
        self.assertEqual(check_bridge(health, now_ns=now), [])

        health["monotonic_ns"] = now - 3_000_000_000
        failures = check_bridge(health, now_ns=now)
        self.assertIn("BRIDGE_HEALTH_STALE: stale health heartbeat", failures)
        self.assertFalse(any(
            failure.startswith("BRIDGE_NOT_PATCHED:") for failure in failures
        ))

        health["monotonic_ns"] = now
        health["code_sha256"] = {"stale.py": "bad"}
        self.assertTrue(any(
            "running implementation differs" in failure
            for failure in check_bridge(health, now_ns=now)
        ))

    def test_team_native_sensor_ids_match_active_sensor_set(self):
        self.assertEqual(
            REQUIRED_SENSOR_DIRS,
            ("CAMERA_1", "CAMERA_2", "CAMERA_3", "CAMERA_4", "LIDAR_5"),
        )

    def test_grpc_target_can_be_overridden_without_pinning_wsl_gateway(self):
        with patch.dict(
            "os.environ", {"MORAI_GRPC_TARGET": "192.0.2.10:7789"}, clear=False
        ):
            self.assertEqual(default_grpc_target(), "192.0.2.10:7789")

    def test_collector_keeps_observation_clock_out_of_team_scenario_kwargs(self):
        source = (INTERNAL / "tools/run_async_scenario_review.py").read_text()
        constructor = source.split("class RawScenario", 1)[1].split(
            "    def event", 1
        )[0]
        self.assertIn("sensor_record_coordination=None, clock=None", constructor)
        self.assertIn("self.clock = clock", constructor)
        self.assertIn(
            "super().__init__(grpc_client, map_loader, global_cfg, scenario_cfg)",
            constructor,
        )
        self.assertNotIn("super().__init__(*args, **kwargs)", constructor)

    def test_sensor_record_supervisor_surfaces_child_startup_failure(self):
        source = (INTERNAL / "tools/run_sensor_record_scenario.py").read_text()
        self.assertIn("Scenario ROS capture exited before readiness", source)
        self.assertIn("Supervisor tail:", source)
        self.assertIn("did not become ready within 65 seconds", source)

    def test_storage_root_maps_any_windows_drive_without_d_pinning(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(INTERNAL)
        environment["MORAI_STORAGE_ROOT"] = "/mnt/c/MoraiDriveData"
        result = subprocess.run(
            [sys.executable, "-c", (
                "from ssd_storage import SSD_ROOT,WINDOWS_ROOT,windows_path;"
                "print(SSD_ROOT);print(WINDOWS_ROOT);"
                "print(windows_path(SSD_ROOT/'sensor_record_auto_v1'/'run'))"
            )],
            env=environment, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=10, check=True,
        )
        self.assertEqual(
            result.stdout.splitlines(),
            ["/mnt/c/MoraiDriveData", r"C:\MoraiDriveData",
             r"C:\MoraiDriveData\sensor_record_auto_v1\run"],
        )

    def test_wsl_home_storage_is_rejected_before_collection(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(INTERNAL)
        environment["MORAI_STORAGE_ROOT"] = "/home/user/repository/data"
        result = subprocess.run(
            [sys.executable, "-c", "import ssd_storage"],
            env=environment, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Windows-backed /mnt/<drive>", result.stderr)

    def test_first_workstation_sensor_folders_are_preserved(self):
        source = (INTERNAL / "tools/prepare_sensor_record_session.ps1").read_text()
        self.assertIn("[string]$SensorDataRoot", source)
        self.assertIn("$sourceRoot=$SensorDataRoot", source)
        self.assertNotIn("$sourceRoot='C:\\MoraiLauncher_Win", source)
        self.assertIn("old_kind=$kind", source)
        self.assertIn("'.data_before_'", source)
        self.assertIn("source_files_deleted=$false", source)
        self.assertNotIn("Remove-Item", source)

    def test_bridge_ip_override_and_sensor_contract_are_machine_portable(self):
        with patch.dict("os.environ", {"MORAI_BRIDGE_IP": "192.0.2.25"}, clear=False):
            self.assertEqual(bridge_ip(), "192.0.2.25")
        first = {"rosConfig": {
            "RosBridgeIP": "172.18.40.28",
            "RosBridgeServerUrl": "ws://172.18.40.28:9090",
            "RosBridgePort": 9090,
        }}
        second = {"rosConfig": {
            "RosBridgeIP": "192.0.2.25",
            "RosBridgeServerUrl": "ws://192.0.2.25:9090",
            "RosBridgePort": 9090,
        }}
        self.assertEqual(
            sensor_set_contract_sha256(first), sensor_set_contract_sha256(second)
        )
        config = {"commType": 3, "rosConfig": {
            "RosMessageOnOff": True,
            "RosBridgeServerUrl": "ws://192.0.2.25:9090",
            "RosBridgeIP": "192.0.2.25", "RosBridgePort": 9090,
            "Topic": "/gps", "frameID": "GPS-6",
        }}
        self.assertTrue(_ros(
            config, topic="/gps", frame="GPS-6",
            expected_bridge_ip="192.0.2.25",
        )["matches"])

    def test_localhost_bridge_is_stable_and_ros_node_ip_remains_wsl_address(self):
        with patch.dict(
            "os.environ",
            {
                "MORAI_BRIDGE_IP": "127.0.0.1",
                "MORAI_ROS_IP": "172.30.144.83",
            },
            clear=False,
        ):
            environment = runtime_environment()
        self.assertEqual(environment["MORAI_BRIDGE_IP"], "127.0.0.1")
        self.assertEqual(environment["ROS_IP"], "172.30.144.83")

    def test_ignored_machine_config_can_pin_stable_localhost_bridge(self):
        with tempfile.TemporaryDirectory(dir=str(DATA_COLLECTION.parent)) as temporary:
            project = Path(temporary)
            config = project / ".runtime/data_collection/local_machine.json"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"bridge_ip": "127.0.0.1"}))
            with patch.dict(
                "os.environ",
                {
                    "MORAI_PROJECT_ROOT": str(project),
                    "MORAI_BRIDGE_IP": "",
                },
                clear=False,
            ):
                self.assertEqual(bridge_ip(), "127.0.0.1")

    def test_ignored_machine_config_selects_storage_without_committing_d(self):
        with tempfile.TemporaryDirectory(dir=str(DATA_COLLECTION.parent)) as temporary:
            project = Path(temporary)
            config = project / ".runtime/data_collection/local_machine.json"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"storage_root": "/mnt/e/TeamData"}))
            environment = dict(os.environ)
            environment.update(
                PYTHONPATH=str(INTERNAL), MORAI_PROJECT_ROOT=str(project)
            )
            environment.pop("MORAI_STORAGE_ROOT", None)
            environment.pop("MORAI_D_SSD_ROOT", None)
            result = subprocess.run(
                [sys.executable, "-c", "from ssd_storage import SSD_ROOT;print(SSD_ROOT)"],
                env=environment, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=10, check=True,
            )
            self.assertEqual(result.stdout.strip(), "/mnt/e/TeamData")

            environment["MORAI_STORAGE_ROOT"] = "/mnt/c/OverrideData"
            result = subprocess.run(
                [sys.executable, "-c", "from ssd_storage import SSD_ROOT;print(SSD_ROOT)"],
                env=environment, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=10, check=True,
            )
            self.assertEqual(result.stdout.strip(), "/mnt/c/OverrideData")

    def test_runtime_paths_translate_configured_launcher_on_any_drive(self):
        environment = dict(os.environ)
        environment.update(
            PYTHONPATH=str(INTERNAL),
            MORAI_LAUNCHER_ROOT="/mnt/f/Apps/MoraiLauncher_Win",
        )
        result = subprocess.run(
            [sys.executable, "-c", (
                "from runtime_paths import WINDOWS_LAUNCHER_ROOT,WINDOWS_SENSOR_DATA_ROOT;"
                "print(WINDOWS_LAUNCHER_ROOT);print(WINDOWS_SENSOR_DATA_ROOT)"
            )],
            env=environment, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=10, check=True,
        )
        self.assertEqual(result.stdout.splitlines(), [
            r"F:\Apps\MoraiLauncher_Win",
            r"F:\Apps\MoraiLauncher_Win\MoraiLauncher_Win_Data\SaveFile\SensorData",
        ])

    def test_runtime_log_directory_is_created_from_a_clean_checkout(self):
        with tempfile.TemporaryDirectory(dir=str(DATA_COLLECTION.parent)) as temporary:
            project = Path(temporary) / "clean_checkout"
            environment = dict(os.environ)
            environment.update(
                PYTHONPATH=str(INTERNAL), MORAI_PROJECT_ROOT=str(project)
            )
            result = subprocess.run(
                [sys.executable, "-c", (
                    "from runtime_paths import ensure_runtime_directory;"
                    "print(ensure_runtime_directory('sensor_record_preflight_logs'))"
                )],
                env=environment, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=10, check=True,
            )
            expected = (
                project / ".runtime/data_collection/sensor_record_preflight_logs"
            ).resolve()
            self.assertEqual(Path(result.stdout.strip()), expected)
            self.assertTrue(expected.is_dir())

    def test_windows_hosted_checkout_uses_its_repository_dataset_directly(self):
        environment = dict(os.environ)
        environment.update(
            PYTHONPATH=str(INTERNAL),
            MORAI_PROJECT_ROOT="/mnt/c/Repos/AIMDrive/MoraiDrive",
        )
        environment.pop("MORAI_STORAGE_ROOT", None)
        environment.pop("MORAI_D_SSD_ROOT", None)
        result = subprocess.run(
            [sys.executable, "-c", "from ssd_storage import SSD_ROOT;print(SSD_ROOT)"],
            env=environment, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=10, check=True,
        )
        self.assertEqual(
            result.stdout.strip(), "/mnt/c/Repos/AIMDrive/MoraiDrive/dataset"
        )

    def test_wsl_home_checkout_defaults_to_windows_c_spool(self):
        with tempfile.TemporaryDirectory(dir=str(DATA_COLLECTION.parent)) as temporary:
            environment = dict(os.environ)
            environment.update(PYTHONPATH=str(INTERNAL), MORAI_PROJECT_ROOT=temporary)
            environment.pop("MORAI_STORAGE_ROOT", None)
            environment.pop("MORAI_D_SSD_ROOT", None)
            result = subprocess.run(
                [sys.executable, "-c", "from ssd_storage import SSD_ROOT;print(SSD_ROOT)"],
                env=environment, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=10, check=True,
            )
            self.assertEqual(result.stdout.strip(), "/mnt/c/MoraiDriveData")

    def test_runtime_sources_do_not_pin_this_workstations_addresses(self):
        first_party = [
            INTERNAL / "native_capture/ground_truth.py",
            INTERNAL / "configs/encoder_mission_candidate_v1.json",
            INTERNAL / "tools/collect_async_raw_proven.py",
            INTERNAL / "tools/run_proven_three_lap_scenario.py",
        ]
        for path in first_party:
            source = path.read_text()
            self.assertNotIn("172.18.32.1", source)
            self.assertNotIn("172.18.40.28", source)

    def test_optional_audit_dependencies_are_shipped_and_storage_portable(self):
        required = [
            INTERNAL / "tools/exec_owned_rosbag.py",
            INTERNAL / "tools/read_simulator_working_directory.ps1",
            INTERNAL / "tools/audit_scenario_rosbag.py",
            INTERNAL / "evidence/native_geometry_v1/persisted_sensor_settings.json",
            INTERNAL / "evidence/native_geometry_v1/camera_projection_evidence.json",
            INTERNAL / "evidence/native_mission_integration_v1/catalog_review_v1/report.json",
        ]
        self.assertTrue(all(path.is_file() for path in required))
        working_directory_reader = required[1].read_text()
        self.assertIn("[string]$TempRoot", working_directory_reader)
        self.assertNotIn("D:\\MoraiDrive", working_directory_reader)
        bag_audit = required[2].read_text()
        self.assertIn("require_ssd(directory", bag_audit)
        self.assertNotIn("ROOT not in directory.parents", bag_audit)


if __name__ == "__main__":
    unittest.main()
