"""Regression tests using the installed, unmodified external expert initializer.

No simulator/ROS connection or expert process control loop is started.
Temporary runtime exports stay inside scenario_runner/.runtime.
"""
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from ..core.full_loop_drive_base import FullLoopDriveBase
from ..main_runner import PACKAGE_ROOT, load_configuration
from ..scenarios.segment_modes import events as mission_events
from ..scenarios.segment_modes import expert as mission_expert
from ..scenarios.segment_modes.route import RouteWindow
from ..scenarios.segment_modes.scenario import MissionScenario
from ..utils.map_loader import MGeoMapLoader
from .test_segment_modes import make_grpc


EXPERT_INITIALIZE = """
import numpy as np
from expert.src.autonomous_driving import AutonomousDriving
driver = AutonomousDriving()
profile = driver.path_manager.velocity_profile
assert len(profile) == len(driver.path)
assert np.isfinite(profile).all()
print('REAL_EXPERT_INIT_OK points=' + str(len(profile)))
"""


class MissionExpertTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()):
            cls.global_cfg, _, cls.base_cfg = load_configuration()
        cls.map_loader = MGeoMapLoader(cls.global_cfg["paths"]["mgeo_root"])

    def setUp(self):
        (PACKAGE_ROOT / ".runtime").mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(
            prefix="mission-expert-test-", dir=PACKAGE_ROOT / ".runtime",
        )
        self.addCleanup(self.tmp.cleanup)

    def make_scenario(self, name):
        global_cfg, cfg = copy.deepcopy(self.global_cfg), copy.deepcopy(self.base_cfg)
        global_cfg["paths"]["runtime_root"] = self.tmp.name
        cfg["mission_mode"] = name
        cfg["route_debug_enabled"] = False
        cfg["profile_state_path"] = str(Path(self.tmp.name) / "profile.json")
        scenario = MissionScenario(make_grpc(cfg), self.map_loader, global_cfg, cfg)
        with redirect_stdout(io.StringIO()):
            mission_events.prepare_mission(scenario)
            scenario.route_definition = scenario.load_route_definition()
            scenario.prepare_full_loop_route()
        return scenario

    def initialize_expert(self, controller):
        environment = os.environ.copy()
        environment.update({
            "PYTHONPATH": controller.expert_source + os.pathsep + environment.get("PYTHONPATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "GT_BEV_CONFIG_PATH": controller.runtime_config_path,
            "GT_BEV_PATH_CSV": controller.runtime_path_csv,
            "GT_BEV_MAP_DIR": controller.mgeo_root,
            "GT_BEV_DIRECT_USE_ENV_CONFIG": "1",
        })
        return subprocess.run(
            [sys.executable, "-B", "-c", EXPERT_INITIALIZE],
            env=environment, cwd=self.tmp.name, capture_output=True,
            text=True, timeout=45,
        )

    def test_three_reported_crashes_then_real_expert_initializes(self):
        for name, last_index in (
            ("full_loop_static", 270), ("full_loop_dynamic", 328),
            ("full_loop_roundabout", 307),
        ):
            with self.subTest(mode=name):
                scenario = self.make_scenario(name)
                with redirect_stdout(io.StringIO()):
                    FullLoopDriveBase.setup_expert_controller(scenario)
                controller = scenario.expert_controller
                before = json.loads(Path(controller.runtime_config_path).read_text())
                route = Path(controller.runtime_path_csv).read_bytes()
                failed = self.initialize_expert(controller)
                self.assertNotEqual(failed.returncode, 0)
                self.assertIn(
                    f"Open-path speed zone ends before it starts: {last_index} -> 0",
                    failed.stderr,
                )
                with redirect_stdout(io.StringIO()):
                    scenario.setup_expert_controller()
                controller = scenario.expert_controller
                after = json.loads(Path(controller.runtime_config_path).read_text())
                expected = copy.deepcopy(before)
                expected["planning"]["velocity_profile"]["speed_limit_zones"] = []
                self.assertEqual(after, expected)
                self.assertEqual(Path(controller.runtime_path_csv).read_bytes(), route)
                success = self.initialize_expert(controller)
                self.assertEqual(success.returncode, 0, success.stdout + success.stderr)
                self.assertIn(f"REAL_EXPERT_INIT_OK points={last_index + 1}", success.stdout)

    def test_normal_and_highway_exports_remain_byte_identical(self):
        for name in ("full_loop_normal", "full_loop_highway"):
            with self.subTest(mode=name):
                scenario = self.make_scenario(name)
                with redirect_stdout(io.StringIO()):
                    FullLoopDriveBase.setup_expert_controller(scenario)
                config_path = Path(scenario.expert_controller.runtime_config_path)
                expected = config_path.read_bytes()
                with redirect_stdout(io.StringIO()):
                    scenario.setup_expert_controller()
                self.assertEqual(config_path.read_bytes(), expected)
                result = self.initialize_expert(scenario.expert_controller)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_filter_runs_before_process_start(self):
        scenario = self.make_scenario("full_loop_static")
        with redirect_stdout(io.StringIO()):
            scenario.setup_expert_controller()
        self.assertIsNone(scenario.expert_controller.process)


class SpeedZoneSelectionTest(unittest.TestCase):
    def setUp(self):
        self.points = tuple((float(i), 0.0, 0.0) for i in range(10))
        self.window = RouteWindow(self.points, 0.0, 3, 6)
        self.loader = Mock()

    def select_zone(self, first, last):
        self.loader.get_link_points.side_effect = [
            [self.points[first]], [self.points[last]],
        ]
        zone = {"name": "test", "start_link_id": "start", "end_link_id": "end", "max_velocity": 20.0}
        return zone, mission_expert.select_speed_zones([zone], self.window, self.loader)

    def test_outside_zone_is_excluded(self):
        _, result = self.select_zone(7, 9)
        self.assertEqual(result, ([], ["test"]))

    def test_inside_zone_keeps_all_settings(self):
        zone, result = self.select_zone(4, 5)
        self.assertEqual(result, ([zone], []))

    def test_closed_route_wrapping_zone_outside_mission(self):
        _, result = self.select_zone(8, 1)
        self.assertEqual(result, ([], ["test"]))

    def test_partial_or_covering_zone_fails_before_launch(self):
        for bounds in ((2, 4), (5, 8), (1, 8), (8, 4)):
            with self.subTest(bounds=bounds), self.assertRaisesRegex(ValueError, "partially overlaps"):
                self.select_zone(*bounds)

    def test_other_modes_do_not_access_runtime_config(self):
        for name in ("full_loop", "full_loop_normal", "full_loop_highway"):
            mission_expert.prepare_speed_zones(SimpleNamespace(
                mission_mode=SimpleNamespace(name=name),
            ))


if __name__ == "__main__":
    unittest.main()
