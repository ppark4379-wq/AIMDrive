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
    catkin_build_invocation,
    discover_catkin_workspace,
    discover_launcher_root,
    discover_sensor_set,
    probe_windows_interop,
    update_sensor_bridge,
)
from preflight_local_clock import wait_for_stable_clock
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
        changed = update_sensor_bridge(data, "127.0.0.1")
        self.assertEqual(changed, 14)
        for key, config_key in (("cameraList", "cc"), ("Lidar3DList", "lc"),
                                ("GPSList", "gc"), ("IMUList", "ic")):
            for item in data[key]:
                ros = item[config_key]["rosConfig"]
                self.assertEqual(ros["RosBridgeIP"], "127.0.0.1")
                self.assertEqual(
                    ros["RosBridgeServerUrl"], "ws://127.0.0.1:9090"
                )
                self.assertEqual(ros["Topic"], "/unchanged")
                self.assertEqual(ros["frameID"], "unchanged")

    def test_campaign_default_name_uses_scenario_laps_and_timestamp(self):
        source = (INTERNAL / "tools/collect_scenario_laps.sh").read_text()
        self.assertIn(
            'CAMPAIGN_ID="${SCENARIO_FOLDER}_lap${LAPS}_$(date +%Y%m%d_%H%M%S)"',
            source,
        )
        self.assertIn('full_loop_normal) SCENARIO_FOLDER="basic_driving"', source)
        self.assertIn('full_loop_static) SCENARIO_FOLDER="static_obstacle"', source)
        self.assertLess(
            source.index('full_loop_static) SCENARIO_FOLDER="static_obstacle"'),
            source.index('CAMPAIGN_ID="${SCENARIO_FOLDER}_lap${LAPS}_'),
        )

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

    def test_shell_entrypoints_are_forced_to_lf_on_windows_checkouts(self):
        repository_root = DATA_COLLECTION.parents[1]
        attributes = (repository_root / ".gitattributes").read_text(
            encoding="utf-8"
        ).splitlines()
        self.assertIn("*.sh text eol=lf", attributes)
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
                self.assertNotIn(b"\r\n", path.read_bytes())

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
        self.assertIn('prepare_mode="sensor-set"', preflight)
        self.assertIn('prepare_mode="runtime"', preflight)
        self.assertIn(
            'morai_prepare_internal_sensor_record_environment "$prepare_mode"',
            preflight,
        )
        self.assertIn("--scenario-only", preflight)
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

    def test_preflight_retries_only_transient_bridge_startup_checks(self):
        preflight = (
            DATA_COLLECTION / "tools/01_internal_sensor_record_preflight.sh"
        ).read_text()
        self.assertIn("for localhost_attempt in 1 2 3; do", preflight)
        self.assertIn("if (( localhost_attempt < 3 )); then", preflight)
        self.assertIn("bridge_stale_attempts=0", preflight)
        self.assertIn("((bridge_stale_attempts += 1))", preflight)
        self.assertIn("if (( bridge_stale_attempts >= 3 )); then", preflight)
        self.assertLess(
            preflight.index("if grep -q 'BRIDGE_NOT_PATCHED'"),
            preflight.index("if grep -q 'BRIDGE_HEALTH_STALE'"),
        )

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
        output = render(report, bridge_ip="127.0.0.1")
        self.assertIn("차량 정보 /Ego_topic", output)
        self.assertIn("Simulation Time", output)
        self.assertIn("Simulator Network", output)
        self.assertIn("127.0.0.1:9090", output)
        self.assertIn("차량 정보·센서셋도 같은 주소", output)
        self.assertIn("Camera 1~4", output)
        self.assertIn("LiDAR", output)
        self.assertIn("MORAI→bridge 패킷 없음", output)
        self.assertNotIn(json.dumps(report), output)

    def test_clock_ingress_delivery_failure_requests_bridge_restart(self):
        report = {
            "ready": False,
            "failures": [
                "missing_or_stale:clock",
                "clock_ingress_not_delivered_to_ros",
            ],
            "streams": {
                "clock": {"messages": 0},
                "traffic_light": {"messages": 0},
                "bridge_clock_ingress": {
                    "samples": 20,
                    "receive_age_ms": 20.0,
                },
                "bridge_camera_ingress": {"streams": {}},
            },
        }
        output = render(report, bridge_ip="172.30.144.83")
        self.assertIn("MORAI→bridge 수신됨, ROS 전달 없음", output)
        self.assertIn("계측 bridge를 재시작", output)
        self.assertNotIn("Simulator Network 재연결 필요", output)

    def test_camera_ingress_rate_is_not_reported_as_native_capture_rate(self):
        streams = {
            role: {"messages": 4, "receive_age_s": 0.1}
            for role in ("ego", "objects", "lidar", "gps", "imu", "clock")
        }
        streams["traffic_light"] = {"messages": 0}
        streams["bridge_camera_ingress"] = {
            "purpose": "connection_health_only",
            "rate_scope": "per_frame_id_on_shared_ros_topic",
            "validates_native_sensor_record_cadence": False,
            "expected_native_sensor_record_hz": 10,
            "streams": {
                "Camera-%d" % index: {
                    "samples": 27,
                    "receive_rate_hz": 5.2,
                    "receive_age_ms": 100.0,
                    "dropped_before_ros": 27,
                }
                for index in range(1, 5)
            },
        }
        output = render({"ready": True, "failures": [], "streams": streams})
        self.assertIn("ROS 연결 확인: 5.2Hz/카메라", output)
        self.assertIn("20.8Hz (Camera 4대가 공유하는 ROS 진단 토픽 합계)", output)
        self.assertIn("Sensor Record 원본", output)
        self.assertIn("수집 후 03번 검증 결과로 확정", output)
        self.assertNotIn("수신 정상: 5.2Hz", output)

    def test_clock_preflight_waits_for_wsl_startup_stabilization(self):
        from unittest import mock

        failed = {
            "passed": False,
            "failures": ["monotonic_rate_differs_from_raw"],
            "rate_error_ppm": 9907.6,
            "wall_steps": [],
        }
        passed = {
            "passed": True,
            "failures": [],
            "rate_error_ppm": 2.0,
            "wall_steps": [],
        }
        with mock.patch(
            "preflight_local_clock.probe", side_effect=[failed, passed]
        ), mock.patch(
            "preflight_local_clock.time.clock_gettime",
            side_effect=[100.0, 102.0, 106.0],
        ), mock.patch("preflight_local_clock.time.sleep"):
            result = wait_for_stable_clock(
                wait_seconds=90.0, probe_seconds=2.0, retry_seconds=3.0
            )
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["stabilization"]["attempts"]), 2)
        self.assertEqual(
            result["stabilization"]["attempts"][0]["rate_error_ppm"],
            9907.6,
        )

        source = (INTERNAL / "tools/preflight_local_clock.py").read_text()
        campaign = (INTERNAL / "tools/collect_scenario_laps.sh").read_text()
        preflight = (
            DATA_COLLECTION / "tools/01_internal_sensor_record_preflight.sh"
        ).read_text()
        self.assertIn("def wait_for_stable_clock", source)
        self.assertIn("time.CLOCK_MONOTONIC_RAW", source)
        self.assertIn("MORAI_CLOCK_STABILIZATION_WAIT_SECONDS", campaign)
        self.assertIn("MORAI_CLOCK_STABILIZATION_WAIT_SECONDS", preflight)

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

    def test_catkin_build_ignores_active_conda_python(self):
        command, environment = catkin_build_invocation(
            "/opt/ros/noetic",
            "/tmp/portable-expert",
            {
                "PATH": "/tmp/conda/bin:/usr/bin",
                "PYTHONHOME": "/tmp/conda",
                "PYTHONPATH": "/tmp/conda/site-packages",
                "CONDA_PREFIX": "/tmp/conda",
            },
        )
        self.assertEqual(command[:2], ["bash", "-c"])
        self.assertIn("--force-cmake", command[2])
        self.assertIn("-DPYTHON_EXECUTABLE=/usr/bin/python3", command[2])
        self.assertNotIn("PYTHONHOME", environment)
        self.assertNotIn("PYTHONPATH", environment)
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        self.assertEqual(environment["CONDA_PREFIX"], "/tmp/conda")

    def test_machine_setup_persists_stable_localhost_bridge_default(self):
        source = (INTERNAL / "tools/configure_local_machine.py").read_text()
        self.assertIn('or os.environ.get("MORAI_BRIDGE_IP")', source)
        self.assertIn('or existing.get("bridge_ip")', source)
        self.assertIn('"bridge_ip": selected_bridge_ip', source)
        self.assertIn('--bridge-ip', source)

    def test_windows_interop_exec_format_failure_has_an_actionable_result(self):
        from unittest import mock

        with mock.patch(
            "configure_local_machine.subprocess.run",
            side_effect=OSError(8, "Exec format error"),
        ):
            ready, detail = probe_windows_interop(
                "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
            )
        self.assertFalse(ready)
        self.assertIn("Exec format error", detail)

    def test_windows_interop_probe_requires_its_exact_response(self):
        from unittest import mock

        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=b"morai-wsl-interop-ok", stderr=b"",
        )
        with mock.patch(
            "configure_local_machine.subprocess.run", return_value=completed
        ):
            ready, detail = probe_windows_interop("powershell.exe")
        self.assertTrue(ready)
        self.assertEqual(detail, "ok")

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

    def test_campaign_checks_and_executes_the_same_selected_runtime(self):
        campaign = (INTERNAL / "tools/collect_scenario_laps.sh").read_text()
        supervisor = (
            INTERNAL / "tools/run_sensor_record_scenario.py"
        ).read_text()
        self.assertIn('SCENARIO_RUNTIME="${MORAI_SCENARIO_RUNTIME:-current}"', campaign)
        self.assertIn('runtime=$SCENARIO_RUNTIME', campaign)
        self.assertIn('--scenario-runtime "$SCENARIO_RUNTIME"', campaign)
        self.assertIn("run_proven_three_lap_scenario.py", campaign)
        self.assertNotIn("run_async_scenario_review.py", campaign)
        self.assertIn("'--scenario-runtime', choices=('current',)", supervisor)
        self.assertIn("scenario_runtime=a.scenario_runtime", supervisor)

    def test_current_runtime_resolves_grpc_only_when_not_explicit(self):
        source = (
            INTERNAL / "tools/run_async_scenario_review.py"
        ).read_text()
        self.assertIn("p.add_argument('--grpc-target')", source)
        self.assertIn("if not a.grpc_target:\n        a.grpc_target = default_grpc_target()", source)
        self.assertNotIn(
            "p.add_argument('--grpc-target', default=default_grpc_target())",
            source,
        )

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
