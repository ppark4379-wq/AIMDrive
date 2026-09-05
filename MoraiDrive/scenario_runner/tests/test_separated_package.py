import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ..adapters.expert_controller import ExpertController
from ..main_runner import (
    PACKAGE_ROOT,
    SUPPORTED_SCENARIOS,
    load_configuration,
)
from ..tools.verify_separation import verify_package


class SeparatedPackageTest(unittest.TestCase):
    def test_supported_scenarios_are_exact(self):
        self.assertEqual(
            set(SUPPORTED_SCENARIOS),
            {
                "full_loop",
                "full_loop_after_static",
                "full_loop_highway_tunnel_test",
            },
        )

    def test_offline_separation_contract(self):
        global_cfg, _, scenario_cfg = load_configuration()
        for scenario_name in SUPPORTED_SCENARIOS:
            report = verify_package(
                PACKAGE_ROOT,
                global_cfg,
                scenario_cfg,
                scenario_name,
            )
            self.assertEqual(report["status"], "PASS")

    def test_expert_controller_does_not_modify_expert(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expert_root = root / "expert"
            config_dir = (
                expert_root / "src" / "expert" / "src" / "config"
            )
            config_dir.mkdir(parents=True)
            config_path = config_dir / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "map": {},
                        "planning": {
                            "velocity_profile": {
                                "max_velocity": 30.0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            before = hashlib.sha256(config_path.read_bytes()).hexdigest()

            controller = ExpertController(
                expert_root=expert_root,
                runtime_root=root / "scenario_runtime",
                map_name="R_KR_PR_K-city_2025",
                mgeo_root=PACKAGE_ROOT
                / "data"
                / "R_KR_PR_K-city_2025",
                route_points=[(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
            )

            after = hashlib.sha256(config_path.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertTrue(Path(controller.runtime_config_path).is_file())
            self.assertFalse((expert_root / ".runtime").exists())


if __name__ == "__main__":
    unittest.main()
