"""Production Jamming inputs and bounded, honest nominal screening evidence."""

import copy
import math
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, patch

from .test_tunnel_static_layout import sample_map, static_models
from ..scenarios.full_loop_parts.tunnel_static_parts.layout import bbox_world, generate_candidate
from ..scenarios.full_loop_parts.tunnel_static_parts import screening


class TunnelStaticScreeningTests(unittest.TestCase):
    def setUp(self):
        self.loader = sample_map()
        self.loader.link_set = self.loader.links
        self.candidate = generate_candidate(self.loader, static_models(), {}, seed=42,
                                            count=1, actor_kind="obstacle")

    def test_missing_dependencies_report_unavailability(self):
        unavailable = {"status": "search_unavailable", "path_found": False,
                       "reasons": ["ImportError: dependency absent"]}
        with patch.object(screening, "_run_bounded", return_value=unavailable):
            result = screening.screen_candidate(self.candidate, self.loader)
        self.assertEqual(result["status"], "search_unavailable")
        self.assertFalse(result["path_found"])
        self.assertIn("dependency absent", result["reasons"][0])

    def test_worker_missing_expert_dependency_is_not_a_highway_fallback(self):
        with patch.object(screening, "_expert_search", side_effect=ImportError("missing Jamming")):
            result = screening._worker({})
        self.assertEqual(result["status"], "search_unavailable")
        self.assertFalse(result["path_found"])

    def test_invalid_candidate_does_not_invoke_worker(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["obstacles"][0]["position"]["x"] = float("nan")
        with patch.object(screening, "_run_bounded") as run:
            result = screening.screen_candidate(candidate, self.loader)
        run.assert_not_called()
        self.assertEqual(result["status"], "invalid_candidate")

    def test_synthetic_map_validates_payload_not_recorded_path_success(self):
        observed = bbox_world(self.candidate["obstacles"][0])
        observed.update(width_m=2.4, length_m=4.2, heading_deg=45.)
        ego = {"position_xyz": [57.8, -459.5, 28.3],
               "rotation_xyz_deg": [0., 0., -117.], "speed_norm_raw": .01,
               "raw_bbox_normalized": {"length_m": 4.635, "width_m": 2.161}}
        footprint = {"length_m": 4.635, "width_m": 2.161,
                     "center_forward_m": 1.53, "center_left_m": -.04}
        def inspect(payload, budget):
            self.assertEqual(payload["boxes"], [observed])
            self.assertEqual(payload["ego"]["position_xyz"], ego["position_xyz"])
            self.assertAlmostEqual(payload["ego"]["yaw_rad"], -117. * 3.141592653589793 / 180.)
            self.assertAlmostEqual(payload["body_policy"]["front_m"], 3.85)
            self.assertAlmostEqual(payload["body_policy"]["rear_m"], .85)
            self.assertAlmostEqual(payload["body_policy"]["width_m"], 2.241)
            self.assertGreater(len(payload["global_route"]), 384)
            self.assertEqual(payload["global_route"][0], (57.79791772423833, -459.4873225380194, 28.25457937406963))
            self.assertLessEqual(budget, 15.)
            return {"status": "no_path_found", "path_found": False, "reasons": []}
        with patch.object(screening, "_run_bounded", side_effect=inspect):
            result = screening.screen_candidate(self.candidate, self.loader,
                measured_geometry=[observed], actual_ego_state=ego, actual_ego_footprint=footprint)
        self.assertIn("observed conservative mesh/API/catalogue", result["geometry_basis"])
        self.assertFalse(result["physical_walls_verified"])
        self.assertFalse(result["live_avoidance_verified"])

    def test_deadline_kills_and_reaps_worker_without_claiming_impossibility(self):
        process = Mock()
        process.communicate.side_effect = [subprocess.TimeoutExpired("worker", .001), ("", "")]
        process.poll.return_value = -9
        with patch.object(screening.subprocess, "Popen", return_value=process):
            result = screening._run_bounded({}, .001)
        process.kill.assert_called_once()
        self.assertEqual(process.communicate.call_count, 2)
        self.assertEqual(result["status"], "search_timeout")
        self.assertTrue(result["worker_terminated"])
        self.assertFalse(result["path_found"])

    def test_station_override_cannot_redefine_production_start_or_return(self):
        with patch.object(screening, "_run_bounded") as run:
            result = screening.screen_candidate(self.candidate, self.loader, start_station_m=5.)
        run.assert_not_called()
        self.assertEqual(result["status"], "invalid_candidate")

    def test_recording_and_policy_hash_changes_invalidate_offline_cache(self):
        original = screening.screening_provenance()
        self.assertEqual(len(original["recorded_library_sha256"]), 11)
        self.assertTrue(screening.provenance_matches(original))
        changed = copy.deepcopy(original)
        name = next(iter(changed["recorded_library_sha256"]))
        changed["recorded_library_sha256"][name] = "0" * 64
        self.assertFalse(screening.provenance_matches(changed))

    def test_changed_advanced_pattern_source_invalidates_existing_screening(self):
        original=screening.screening_provenance()
        path=Path(screening.__file__).with_name("advanced_patterns.py")
        self.assertIn(screening._relative(path),original["planner_source_sha256"])
        actual_hash=screening._file_hash
        def changed_file(value):
            return "0"*64 if Path(value).resolve()==path.resolve() else actual_hash(value)
        with patch.object(screening,"_file_hash",side_effect=changed_file):
            self.assertFalse(screening.provenance_matches(original))
        changed = copy.deepcopy(original)
        changed["screening_policy"]["central_line_crossing_allowed"] = False
        self.assertFalse(screening.provenance_matches(changed))

    def test_millimetre_pose_noise_does_not_invent_repositioning_edge(self):
        try:
            import numpy as np
        except ImportError as error:
            self.skipTest(str(error))
        route = [(0., 0., 28.), (.5, 0., 28.), (1., 0., 28.), (1.5, 0., 28.), (2., 0., 28.)]
        ego = type("Ego", (), {"position": np.array([.0035, -.0049]), "yaw": math.radians(1.34)})()
        chosen = [np.array([1.5, 0.]), np.array([2., 0.]), np.array([2.5, 0.])]
        complete = screening._complete_approach(chosen, route, ego, np)
        self.assertEqual(complete[0].tolist(), [0., 0.])
        self.assertTrue(np.all(np.diff(complete[:, 0]) >= .5-1e-9))
        alignment = screening._initial_alignment(complete, ego, np)
        self.assertTrue(alignment["passed"], alignment)
        self.assertAlmostEqual(alignment["position_error_m"], .0049)
        ego.position = np.array([.0035, -.8])
        self.assertFalse(screening._initial_alignment(complete, ego, np)["passed"])
        ego.position = np.array([.0035, -.0049])
        ego.yaw = math.radians(30.)
        self.assertEqual(screening._initial_alignment(complete, ego, np)["reason"], "initial_path_heading_error")

    def test_intrinsic_3d_body_bounds_cover_roll_pitch_and_centre_drift(self):
        footprint = {"length_m": 4.635007858276367, "width_m": 2.160834789276123,
                     "height_m": 1.6440463066101074,
                     "center_actor_flu_m": [1.5268378093708894, -.03586219119335112, .4542921915623166],
                     "max_abs_roll_pitch_deg": 3.}
        bounds = screening._body_input(footprint)
        self.assertEqual(set(bounds), {"rear_m", "front_m", "width_m"})
        self.assertAlmostEqual(bounds["rear_m"], .8605208519931932)
        self.assertAlmostEqual(bounds["front_m"], 3.914196470734972)
        self.assertAlmostEqual(bounds["width_m"], 2.3661535397451785)
        center = footprint["center_actor_flu_m"]
        half = [footprint[key] * .5 for key in ("length_m", "width_m", "height_m")]
        for roll_deg in (-3., -2.266, -1.22, 0., 1.22, 2.266, 3.):
            for pitch_deg in (-3., -1.5, 0., 1.5, 3.):
                r, p = math.radians(roll_deg), math.radians(pitch_deg)
                for a in (-1., 1.):
                    for b in (-1., 1.):
                        for c in (-1., 1.):
                            x, y, z = [v + sign * h for v, sign, h in zip(center, (a, b, c), half)]
                            yr = y * math.cos(r) - z * math.sin(r)
                            zr = y * math.sin(r) + z * math.cos(r)
                            xp = x * math.cos(p) + zr * math.sin(p)
                            self.assertGreaterEqual(xp, -bounds["rear_m"] - 1e-9)
                            self.assertLessEqual(xp, bounds["front_m"] + 1e-9)
                            self.assertLessEqual(abs(yr), .5 * bounds["width_m"] + 1e-9)
        # The flat centre moves laterally with roll; the sealed policy covers
        # both observed attitudes without changing its road or body settings.
        lateral_centres = [center[1] * math.cos(math.radians(r)) - center[2] * math.sin(math.radians(r))
                           for r in (1.22, 2.266)]
        self.assertGreater(abs(lateral_centres[1]), abs(lateral_centres[0]))

    def test_partial_or_different_3d_attitude_policy_does_not_silently_flatten(self):
        body = {"length_m": 4.635, "width_m": 2.161, "center_forward_m": 1.53,
                "height_m": 1.644}
        with self.assertRaises(KeyError):
            screening._body_input(body)
        body.update(center_actor_flu_m=[1.53, -.04, .45], max_abs_roll_pitch_deg=4.)
        with self.assertRaisesRegex(ValueError, "shared 3 degree policy"):
            screening._body_input(body)
        body.update(max_abs_roll_pitch_deg=3., height_m=float("nan"))
        with self.assertRaises(ValueError):
            screening._body_input(body)

    def test_initial_observed_roll_outside_envelope_rejects_before_worker(self):
        body = {"length_m": 4.635, "width_m": 2.161, "height_m": 1.644,
                "center_actor_flu_m": [1.5268, -.0359, .4543], "max_abs_roll_pitch_deg": 3.}
        ego = {"position_xyz": [57.7979, -459.4873, 28.59],
               "rotation_xyz_deg": [3.01, 0., -117.21], "speed_norm_raw": 0.}
        with patch.object(screening, "_run_bounded") as run:
            result = screening.screen_candidate(self.candidate, self.loader,
                actual_ego_state=ego, actual_ego_footprint=body)
        self.assertEqual(result["status"], "invalid_candidate")
        self.assertIn("projection envelope", result["reasons"][0])
        run.assert_not_called()


class ActualMapJammingScreeningIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy  # noqa: F401
            screening._expert_search()
        except ImportError as error:
            raise unittest.SkipTest("Production Jamming dependencies unavailable: " + str(error))
        from ..utils.map_loader import MGeoMapLoader
        from ..scenarios.full_loop_parts.tunnel_static_parts.catalog import load_static_models, load_vehicle_models
        cls.loader = MGeoMapLoader(str(Path(screening.ROUTE_PATH).parent))
        cls.statics = {"WoodBox": load_static_models()["WoodBox"]}
        cls.vehicles = {"DEFAULT_SEDAN": load_vehicle_models()["DEFAULT_SEDAN"]}

    def test_four_seeded_actual_map_cases_use_recordings_and_full_mission_start(self):
        for kind in ("obstacle", "vehicle"):
            for count in (1, 2):
                with self.subTest(kind=kind, count=count):
                    candidate = generate_candidate(self.loader, self.statics, self.vehicles,
                        seed=20261007, count=count, actor_kind=kind)
                    result = screening.screen_candidate(candidate, self.loader, time_budget_sec=15.)
                    self.assertEqual(result["status"], "nominal_path_found", result)
                    self.assertTrue(result["geometry_checks"]["passed"])
                    self.assertTrue(result["selected_routes"])
                    self.assertTrue(all(name.startswith("jamming_") for name in result["selected_routes"]))
                    self.assertAlmostEqual(result["path"][0][0], 57.79791772423833, places=6)
                    self.assertAlmostEqual(result["path"][0][1], -459.4873225380194, places=6)
                    self.assertEqual(result["screening_policy"]["start_original_route_index"], 3723)
                    self.assertIn("road_boundary_source_sha256", result)
                    self.assertEqual(result["screening_policy"]["planning_tunnel_body_road_margin_m"], .35)
                    self.assertEqual(result["screening_policy"]["full_body_road_margin_m"], .15)
                    self.assertEqual(result["geometry_checks"]["road_margin_mode"], "planning")
                    self.assertTrue(result["geometry_checks"]["runtime_path_road_passed"])
                    self.assertEqual(result["geometry_checks"]["planning_road_margin_m"], .35)
                    self.assertEqual(result["geometry_checks"]["runtime_road_margin_m"], .15)
                    for key in ("expert_verified", "live_avoidance_verified", "physical_walls_verified"):
                        self.assertFalse(result[key])

    def test_observed_whole_road_blockage_cannot_pass_recorded_route_screen(self):
        candidate = generate_candidate(self.loader, self.statics, {}, 42, count=1, actor_kind="obstacle")
        measured = bbox_world(candidate["obstacles"][0])
        measured.update(width_m=12., length_m=5.)
        result = screening.screen_candidate(candidate, self.loader, time_budget_sec=15.,
                                            measured_geometry=[measured])
        self.assertFalse(result["path_found"], result)
        self.assertFalse(result["geometric_impossibility_proven"])

    def test_observed_millimetre_spawn_noise_is_separate_from_path_curvature(self):
        candidate = generate_candidate(self.loader, self.statics, {}, 42, count=1, actor_kind="obstacle")
        ego = {"position_xyz": [57.792972564697266, -459.48382568359375, 28.592784881591797],
               "rotation_xyz_deg": [-358.77703857421875, 359.795654296875, -115.86912536621094],
               "speed_norm_raw": .007483144094571008}
        body = {"length_m": 4.635007858276367, "width_m": 2.160834789276123,
                "center_forward_m": 1.5251853835431897, "center_left_m": -.04554748821086829,
                "height_m": 1.6440463066101074,
                "center_actor_flu_m": [1.5268378093708894, -.03586219119335112, .4542921915623166],
                "max_abs_roll_pitch_deg": 3.}
        observed = {"center_xyz": [7.777024648042925, -500.493594793497, 28.34362030029297],
                    "heading_deg": -151.73887211665578, "width_m": 1.2000108728269652,
                    "length_m": 1.200007126792002}
        result = screening.screen_candidate(candidate, self.loader, measured_geometry=[observed],
            actual_ego_state=ego, actual_ego_footprint=body)
        self.assertEqual(result["status"], "nominal_path_found", result)
        checks = result["geometry_checks"]
        self.assertTrue(checks["initial_alignment"]["passed"], checks)
        self.assertLess(checks["initial_alignment"]["position_error_m"], .01)
        self.assertLess(checks["initial_alignment"]["heading_error_deg"], 2.)
        self.assertLess(checks["max_curvature_inv_m"], .24)
        self.assertTrue(checks["actual_ego_footprint_passed"])
        self.assertAlmostEqual(result["road_boundary_policy"]["body"]["width_m"], 2.3661535397451785)
        self.assertAlmostEqual(result["path"][0][0], 57.79791772423833)
        self.assertAlmostEqual(result["path"][0][1], -459.4873225380194)
        displaced = copy.deepcopy(ego)
        displaced["position_xyz"][0] += .8
        rejected = screening.screen_candidate(candidate, self.loader, measured_geometry=[observed],
            actual_ego_state=displaced, actual_ego_footprint=body)
        self.assertFalse(rejected["path_found"], rejected)
        self.assertEqual(rejected["reasons"], ["initial_path_position_error"])

    def test_old_sedan_edge_pose_rejected_for_planning_without_loosening_runtime_gate(self):
        import numpy as np
        _, boundary_type, _, _, _ = screening._expert_search()
        body = {"rear_m": .8604676685031636, "front_m": 3.914249370522757,
                "width_m": 2.366195770601937}
        boundary = boundary_type.from_links(self.loader.link_set, body_policy=body,
                                            planning_margin_m=.35, runtime_margin_m=.15)
        # This old path pose passed the previous 15 cm nominal check. Its
        # interior reserve is now rejected before driving, while the actual
        # shifted/yawed failing pose still fails the unchanged runtime gate.
        old_planned = np.array([[-.492143332, -508.28191953]])
        old_heading = np.array([math.radians(-140.97320788721453)])
        self.assertTrue(boundary.mask_world(old_planned, old_heading).all())
        self.assertFalse(boundary.planning_mask_world(old_planned, old_heading).all())
        actual = np.array([[-.4902738332748413, -508.20849609375]])
        actual_heading = np.array([math.radians(-138.20196533203125)])
        self.assertFalse(boundary.mask_world(actual, actual_heading).all())
        self.assertAlmostEqual(boundary.policy["nominal_tunnel_width_m"], 7.)
        approach = np.array([[57.79791772423833, -459.4873225380194]])
        approach_heading = np.array([math.radians(-117.21164428345622)])
        self.assertEqual(boundary.planning_mask_world(approach, approach_heading).tolist(),
                         boundary.mask_world(approach, approach_heading).tolist())


if __name__ == "__main__":
    unittest.main()
