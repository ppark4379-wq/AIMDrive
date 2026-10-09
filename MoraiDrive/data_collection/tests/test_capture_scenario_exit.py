"""Scenario startup failures must not leave the native UI recording alone."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

INTERNAL = Path(__file__).resolve().parents[1] / "internal_sensor_record"
sys.path[:0] = [str(INTERNAL), str(INTERNAL / "tools")]
from sensor_record_barrier import request_stop_for_finished_scenario
from configure_local_machine import run_import_check


class CaptureScenarioExitTest(unittest.TestCase):
    def test_running_scenario_does_not_request_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = request_stop_for_finished_scenario(Mock(poll=lambda: None), temporary)
            self.assertIsNone(result)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_exit_before_ui_armed_requests_stop_and_keeps_original_request(self):
        for code in (0, 1):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                result = request_stop_for_finished_scenario(Mock(poll=lambda: code), temporary)
                path = Path(temporary) / "stop.request"
                self.assertEqual(json.loads(path.read_text())["scenario_exit_code"], code)
                self.assertFalse(result["training_visible"])
                original = path.read_bytes()
                self.assertIsNone(request_stop_for_finished_scenario(Mock(poll=lambda: 2), temporary))
                self.assertEqual(path.read_bytes(), original)

    def test_ui_owner_receives_stop_while_supervisor_waits_for_powershell(self):
        spec = importlib.util.spec_from_file_location("capture_exit_supervisor",
            INTERNAL / "tools/run_sensor_record_scenario.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stop.request"
            def ui_owner(script, timeout):
                deadline = time.monotonic() + 3
                while not path.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(path.exists(), "UI must receive stop without waiting for capture timeout")
                return "stopped"
            with patch.object(module, "powershell", side_effect=ui_owner):
                result = module.powershell_with_heartbeat("mock UI", timeout=3000,
                    label="failed_lap", scenario_process=Mock(poll=lambda: 1),
                    coordination=temporary)
            self.assertEqual(result, "stopped")

    def test_prepare_checks_shapely_two_api_in_capture_interpreter(self):
        with patch("configure_local_machine.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 1, stderr="missing Shapely 2")) as run:
            ok, error = run_import_check("/usr/bin/python3", ("numpy", "shapely"), {})
        self.assertFalse(ok)
        self.assertIn("missing Shapely 2", error)
        self.assertIn("from shapely import union_all, polygons", run.call_args[0][0][-1])


if __name__ == "__main__":
    unittest.main()
