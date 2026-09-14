import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest


DATA_COLLECTION = Path(__file__).resolve().parents[1]
INTERNAL = DATA_COLLECTION / "internal_sensor_record"
sys.path.insert(0, str(INTERNAL / "tools"))

from configure_local_machine import (
    discover_catkin_workspace,
    discover_launcher_root,
    discover_sensor_set,
    update_sensor_bridge,
)
from render_sensor_record_readiness import render


class PortableSensorRecordEntrypointTest(unittest.TestCase):
    def test_sensor_set_bridge_patch_changes_only_transport_address(self):
        config = {
            "commType": 3,
            "rosConfig": {
                "RosBridgeIP": "172.18.0.1",
                "RosBridgePort": 9090,
                "RosBridgeServerUrl": "ws://172.18.0.1:9090",
                "Topic": "/unchanged",
                "frameID": "unchanged",
            },
        }
        def wrapped(key):
            return {key: copy.deepcopy(config)}

        data = {
            "cameraList": [wrapped("cc") for _ in range(4)],
            "Lidar3DList": [wrapped("lc")],
            "GPSList": [wrapped("gc")],
            "IMUList": [wrapped("ic")],
        }
        changed = update_sensor_bridge(data, "172.19.0.55")
        self.assertEqual(changed, 14)
        for key, config_key in (("cameraList", "cc"), ("Lidar3DList", "lc"),
                                ("GPSList", "gc"), ("IMUList", "ic")):
            for item in data[key]:
                ros = item[config_key]["rosConfig"]
                self.assertEqual(ros["RosBridgeIP"], "172.19.0.55")
                self.assertEqual(
                    ros["RosBridgeServerUrl"], "ws://172.19.0.55:9090"
                )
                self.assertEqual(ros["Topic"], "/unchanged")
                self.assertEqual(ros["frameID"], "unchanged")

    def test_all_shell_entrypoints_have_valid_syntax(self):
        paths = [
            DATA_COLLECTION / "tools/00_prepare_internal_sensor_record.sh",
            DATA_COLLECTION / "tools/01_internal_sensor_record_preflight.sh",
            DATA_COLLECTION / "tools/02_collect_internal_sensor_record_laps.sh",
            DATA_COLLECTION / "tools/03_validate_internal_sensor_record.sh",
            DATA_COLLECTION / "tools/internal_sensor_record_env.sh",
            INTERNAL / "tools/collect_scenario_laps.sh",
        ]
        for path in paths:
            with self.subTest(path=path.name):
                subprocess.run(["bash", "-n", str(path)], check=True)

    def test_entrypoints_share_isolated_environment(self):
        shared = (DATA_COLLECTION / "tools/internal_sensor_record_env.sh").read_text()
        preflight = (
            DATA_COLLECTION / "tools/01_internal_sensor_record_preflight.sh"
        ).read_text()
        campaign = (INTERNAL / "tools/collect_scenario_laps.sh").read_text()
        validator = (
            DATA_COLLECTION / "tools/03_validate_internal_sensor_record.py"
        ).read_text()
        validator_shell = (
            DATA_COLLECTION / "tools/03_validate_internal_sensor_record.sh"
        ).read_text()
        bridge_launcher = (
            INTERNAL / "tools/start_instrumented_bridge.py"
        ).read_text()
        self.assertIn("PYTHONNOUSERSITE=1", shared)
        self.assertIn("MORAI_ASYNC_DEPS", shared)
        self.assertIn("source \"$TOOLS/internal_sensor_record_env.sh\"", preflight)
        self.assertIn("source \"$PUBLIC_TOOLS/internal_sensor_record_env.sh\"", campaign)
        self.assertIn("morai_prepare_internal_sensor_record_environment sensor-set", preflight)
        self.assertIn("morai_prepare_internal_sensor_record_environment runtime", campaign)
        self.assertIn("morai_prepare_internal_sensor_record_environment runtime", validator_shell)
        self.assertIn('capture_python = "/usr/bin/python3"', validator)
        self.assertIn('PYTHONNOUSERSITE="1"', validator)
        self.assertIn("signal.signal(signal.SIGTERM, request_stop)", bridge_launcher)
        self.assertIn('kill -TERM "$BRIDGE_PID"', preflight)
        self.assertIn('kill -KILL "$BRIDGE_PID"', preflight)
        self.assertIn("preflight_local_storage.py", preflight)
        self.assertIn("preflight_local_storage.py", campaign)
        self.assertIn("preflight_local_storage.py", validator_shell)

    def test_readiness_renderer_explains_each_missing_operator_action(self):
        report = {
            "ready": False,
            "failures": [
                "missing_or_stale:ego",
                "SENSOR_MISSING: bridge Camera-1",
                "missing_or_stale:lidar",
                "missing_or_stale:clock",
            ],
            "streams": {
                "ego": {"messages": 0},
                "objects": {"messages": 4, "receive_age_s": 0.1},
                "lidar": {"messages": 0},
                "gps": {"messages": 4, "receive_age_s": 0.1},
                "imu": {"messages": 4, "receive_age_s": 0.1},
                "clock": {"messages": 0},
                "traffic_light": {"messages": 0},
                "bridge_camera_ingress": {"streams": {}},
            },
        }
        output = render(report)
        self.assertIn("차량 정보 /Ego_topic", output)
        self.assertIn("Simulation Time", output)
        self.assertIn("Camera 1~4", output)
        self.assertIn("LiDAR", output)
        self.assertNotIn(json.dumps(report), output)

    def test_sensor_set_discovery_is_not_tied_to_one_drive(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            launcher = Path(temporary)
            sensor = (
                launcher
                / "MoraiLauncher_Win_Data/SaveFile/Sensor/26.R1.H3/aim_sensorset.json"
            )
            sensor.parent.mkdir(parents=True)
            sensor.write_text("{}")
            self.assertEqual(discover_sensor_set(launcher), sensor.resolve())

    def test_stale_launcher_config_is_rediscovered(self):
        import tempfile
        from unittest import mock

        with tempfile.TemporaryDirectory() as temporary:
            mount = Path(temporary)
            launcher = mount / "e/MoraiLauncher_Win"
            (launcher / "MoraiLauncher_Win_Data/SaveFile/Sensor").mkdir(
                parents=True
            )
            stale = mount / "d/MoraiLauncher_Win"
            with mock.patch("configure_local_machine.Path.glob",
                            return_value=[launcher]):
                discovered = discover_launcher_root(configured=stale)
            self.assertEqual(discovered, launcher.resolve())

    def test_checkout_catkin_is_default_on_clean_machine(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            (project / "expert/src").mkdir(parents=True)
            (project / "expert/src/CMakeLists.txt").write_text("")
            self.assertEqual(
                discover_catkin_workspace(project),
                (project / "expert").resolve(),
            )

    def test_runtime_adapter_consumes_clock_before_team_constructor(self):
        source = (INTERNAL / "tools/run_async_scenario_review.py").read_text()
        constructor = source.split("class RawScenario", 1)[1].split(
            "    def event", 1
        )[0]
        self.assertIn(
            "def __init__(self, grpc_client, map_loader, global_cfg, scenario_cfg, *",
            constructor,
        )
        self.assertIn(
            "super().__init__(grpc_client, map_loader, global_cfg, scenario_cfg)",
            constructor,
        )
        self.assertNotIn("super().__init__(*args, **kwargs)", constructor)

    def test_process_lock_creates_a_clean_runtime_hierarchy(self):
        source = (INTERNAL / "process_guard.py").read_text()
        self.assertIn("path.parent.mkdir(parents=True, exist_ok=True)", source)

    def test_capture_has_operator_heartbeat(self):
        source = (INTERNAL / "tools/run_sensor_record_scenario.py").read_text()
        self.assertIn("def powershell_with_heartbeat", source)
        self.assertIn("[CAPTURE]", source)

    def test_scenario_grpc_config_does_not_pin_a_wsl_lease(self):
        global_config = (
            DATA_COLLECTION.parent / "scenario_runner/config/global.yaml"
        ).read_text()
        adapter = (
            DATA_COLLECTION.parent / "scenario_runner/adapters/morai_grpc.py"
        ).read_text()
        self.assertIn("host: auto", global_config)
        self.assertIn("MORAI_GRPC_TARGET", adapter)
        self.assertIn("_windows_host_gateway", adapter)


if __name__ == "__main__":
    unittest.main()
