"""실제 경로·Expert 설정을 이용한 신호 mission의 오프라인 회귀 검증."""
import copy
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ..adapters.expert_controller import ExpertController
from ..main_runner import load_configuration
from ..tools.verify_traffic_light_expert import verify_traffic_light_expert


class TrafficLightExpertVerificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()):
            cls.global_cfg, _, cls.scenario_cfg = load_configuration()

    def test_actual_export_phase_matrix_and_gradual_stop_without_process(self):
        with patch.object(ExpertController, "start_process", side_effect=AssertionError("검증에서 프로세스를 시작할 수 없습니다.")):
            result = verify_traffic_light_expert(self.global_cfg, self.scenario_cfg)
        self.assertTrue(result["passed"])
        self.assertTrue(result["source_files_unchanged"])
        self.assertEqual((1031, 1348), (result["global_start_index"], result["global_end_index"]))
        self.assertEqual(318, result["mission_points"])
        self.assertEqual(389, result["expert_points"])
        self.assertAlmostEqual(155.2933490661302, result["mission_length_m"], places=6)
        self.assertGreaterEqual(result["expert_tail_m"], 35.0)
        targets = result["route_signal_targets"]["targets"]
        self.assertEqual([("C1256W000123", 92), ("C1256W000052", 214)],
                         [(row["signal_id"], row["path_index"]) for row in targets])
        self.assertEqual([], result["runtime_speed_zones"])
        self.assertEqual({32, 33, 36, 48}, {row["raw_signal_bits"] for row in result["phase_matrix"] if row["left_permitted"]})
        self.assertTrue(all(row["velocity_cap_mps"] == 0 for row in result["unavailable_signal_cases"]))
        self.assertGreater(result["red_approach"][0]["velocity_cap_mps"], 0.0)
        self.assertEqual(0, result["red_approach"][0]["local_path_index"])
        self.assertAlmostEqual(105.05766429524465,
                               result["red_approach"][0]["remaining_route_m"], places=6)
        self.assertEqual(0.0, result["red_approach"][-1]["velocity_cap_mps"])
        self.assertFalse(result["expert_process_started"])
        self.assertFalse(result["ros_connected"])
        self.assertFalse(result["morai_connected"])

    def test_call_restores_environment_modules_and_input_configs(self):
        global_before, scenario_before = copy.deepcopy(self.global_cfg), copy.deepcopy(self.scenario_cfg)
        path_before = list(sys.path)
        modules_before = {key: value for key, value in sys.modules.items()
                          if key == "expert" or key.startswith("expert.")}
        with patch.dict(os.environ, {"GT_BEV_CONFIG_PATH": "검증전_설정",
                                     "GT_BEV_PATH_CSV": "검증전_경로"}):
            environment_before = dict(os.environ)
            verify_traffic_light_expert(self.global_cfg, self.scenario_cfg)
            self.assertEqual(environment_before, dict(os.environ))
        self.assertEqual(path_before, sys.path)
        self.assertEqual(modules_before, {key: value for key, value in sys.modules.items()
                                         if key == "expert" or key.startswith("expert.")})
        self.assertEqual(global_before, self.global_cfg)
        self.assertEqual(scenario_before, self.scenario_cfg)

    def test_changed_route_geometry_is_rejected_before_signal_validation(self):
        cfg = copy.deepcopy(self.scenario_cfg)
        points = Path(cfg["global_path"]).read_text(encoding="utf-8-sig").splitlines()
        # 승인된 START/END는 유지하고 crop 내부의 원본 포인트 하나만 바꾼다.
        index = 1200
        point_rows = [i for i, row in enumerate(points) if row.strip() and not row.lstrip().startswith("#")]
        line = point_rows[index]
        coordinates = points[line].split()
        coordinates[0] = str(float(coordinates[0]) + 0.05)
        points[line] = " ".join(coordinates)
        with tempfile.TemporaryDirectory(prefix="morai-traffic-light-route-test-", dir="/tmp") as temporary:
            changed = Path(temporary) / "changed_route.txt"
            changed.write_text("\n".join(points) + "\n", encoding="utf-8")
            cfg["global_path"] = str(changed)
            with self.assertRaisesRegex(ValueError, "unique exact slice"):
                verify_traffic_light_expert(self.global_cfg, cfg)

    def test_missing_signal_mode_speed_zone_filter_is_rejected(self):
        from ..scenarios.segment_modes import expert as mission_expert
        without_mode = mission_expert.FILTERED_MISSIONS - {"full_loop_traffic_light"}
        with patch.object(mission_expert, "FILTERED_MISSIONS", without_mode):
            with self.assertRaisesRegex(ValueError, "speed-zone 필터"):
                verify_traffic_light_expert(self.global_cfg, self.scenario_cfg)


if __name__ == "__main__":
    unittest.main()
