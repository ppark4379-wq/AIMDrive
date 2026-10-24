"""Runtime signal topology export, without ROS or starting an Expert process."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ..adapters.expert_controller import ExpertController
from ..core.full_loop_drive_base import FullLoopDriveBase
from ..core import full_loop_drive_base


class ExpertSignalRouteExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        config = self.root / "expert/src/expert/src/config/config.json"
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({
            "map": {"route_signal_route_links": ["STALE"], "route_signal_targets": {"stale": True}},
            "planning": {"velocity_profile": {"max_velocity": 60}},
        }), encoding="utf-8")

    def controller(self, links=None, speed_cap=None):
        return ExpertController(
            expert_root=str(self.root / "expert"), runtime_root=str(self.root / "runtime"),
            map_name="test", mgeo_root=str(self.root / "map"),
            route_points=[(0, 0), (1, 0)], route_signal_route_links=links, speed_cap_kmh=speed_cap,
        )

    def test_defaults_follow_current_expert_speed_settings_without_changing_source(self):
        source = self.root / "expert/src/expert/src/config/config.json"
        for maximum, window in ((37, 23), (43, 31)):
            with self.subTest(maximum=maximum, window=window):
                original = json.loads(source.read_text())
                original["planning"]["velocity_profile"].update(
                    max_velocity=maximum, window_size=window,
                )
                source.write_text(json.dumps(original))
                before = source.read_bytes()
                controller = self.controller()
                runtime = json.loads(Path(controller.runtime_config_path).read_text())
                self.assertEqual(runtime["planning"], original["planning"])
                self.assertTrue(controller.speed_contract["inherited"])
                self.assertIsNone(controller.requested_max_velocity_kmh)
                self.assertIsNone(controller.speed_cap_kmh)
                self.assertEqual(source.read_bytes(), before)

    def test_runtime_speed_cap_limits_highway_preserves_lower_limits_and_source(self):
        source = self.root / "expert/src/expert/src/config/config.json"
        config = json.loads(source.read_text())
        config["planning"]["velocity_profile"].update(
            max_velocity=100, speed_limit_zones=[
                {"name": "highway", "max_velocity": 100},
                {"name": "slow", "max_velocity": 20}])
        source.write_text(json.dumps(config))
        before = source.read_bytes()
        controller = self.controller(speed_cap=60)
        runtime = json.loads(Path(controller.runtime_config_path).read_text())
        profile = runtime["planning"]["velocity_profile"]
        self.assertEqual(profile["max_velocity"], 60)
        self.assertEqual([z["max_velocity"] for z in profile["speed_limit_zones"]], [60, 20])
        self.assertEqual(source.read_bytes(), before)
        unbounded = self.controller()
        profile = json.loads(Path(unbounded.runtime_config_path).read_text())["planning"]["velocity_profile"]
        self.assertEqual(profile["max_velocity"], 100)
        self.assertEqual([z["max_velocity"] for z in profile["speed_limit_zones"]], [100, 20])

    def test_invalid_speed_cap_rejected(self):
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.controller(speed_cap=value)

    def test_exports_full_ordered_links_for_cropped_path_and_clears_stale_targets(self):
        links = ["BEFORE-CROP", "CROP-APPROACH", "LEFT-EXIT", "AFTER-CROP"]
        controller = self.controller(links)
        config = json.loads(Path(controller.runtime_config_path).read_text(encoding="utf-8"))
        self.assertEqual(config["map"]["route_signal_route_links"], links)
        self.assertNotIn("route_signal_targets", config["map"])
        self.assertIsNone(controller.process)
        links.append("MUTATED-CALLER")
        self.assertEqual(controller.route_signal_route_links[-1], "AFTER-CROP")

    def test_missing_contract_preserves_legacy_mode_without_stale_route_from_base_config(self):
        controller = self.controller()
        config = json.loads(Path(controller.runtime_config_path).read_text(encoding="utf-8"))
        self.assertNotIn("route_signal_route_links", config["map"])
        self.assertNotIn("route_signal_targets", config["map"])

    def test_invalid_explicit_route_contract_fails_before_runtime_export(self):
        for links in ([], "APPROACH", [None], [""], ["   "], [12]):
            with self.subTest(links=links), self.assertRaisesRegex(ValueError, "ordered list"):
                self.controller(links)
        self.assertFalse((self.root / "runtime/path.csv").exists())

    def test_drive_base_passes_original_full_links_instead_of_cropped_link_selection(self):
        context = SimpleNamespace(
            global_cfg={"paths": {"expert_root": "expert", "runtime_root": "runtime",
                                  "mgeo_root": "map"}},
            cfg={}, route_points=[(10, 0), (11, 0)], start_link="CROP", end_link="END",
            route_links=["CROP", "END"],
            route_definition={"route_links": ["START", "CROP", "LEFT-EXIT", "END", "TAIL"]},
        )
        with patch.object(full_loop_drive_base, "ExpertController") as export:
            FullLoopDriveBase.setup_expert_controller(context)
        self.assertEqual(export.call_args.kwargs["route_signal_route_links"],
                         context.route_definition["route_links"])


if __name__ == "__main__":
    unittest.main()
