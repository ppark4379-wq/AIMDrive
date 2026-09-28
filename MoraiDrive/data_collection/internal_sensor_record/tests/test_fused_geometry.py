import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from morai_pretraining.fused_geometry import (
    DEFAULT_REGISTRY_POLICY, candidate_sizes, check_fused_object_geometry, containment,
    fit_center, model_key, score_observation, select_model_geometry,
)
from morai_pretraining.sensor_record_manifest import objects_for_entry


def pedestrian(heading=0.0):
    # gRPC reports the bind-pose box: 1.8 m across a walker that is 0.62 m long.
    return {"id": "5", "instance_key": "p", "kind": "pedestrian", "model_name": "Man5",
            "position": [0.0, 0.0, 0.0], "heading_deg": heading, "size_wlh_m": [1.8, 0.41, 1.8],
            "velocity_kmh": [0.0, 0.0, 0.0]}


ROS_MAN5 = {"kind": "pedestrian", "name": "Man5", "size_xyz_m": [0.392, 0.616, 1.795], "valid": True}


def walker_points(heading_deg=0.0):
    """Stride along heading (+-0.33 m, as recorded for Man5) and a 0.36 m wide body."""
    along = np.linspace(-.33, .33, 13)
    across = np.linspace(-.18, .18, 7)
    grid = np.array([(a, b) for a in along for b in across])
    h = math.radians(heading_deg)
    rotation = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])
    return grid @ rotation.T


class FusedGeometryTest(unittest.TestCase):
    def test_candidate_sizes_keep_axes_explicit(self):
        sizes = candidate_sizes(pedestrian(), ROS_MAN5)
        self.assertEqual(sizes["grpc"], (0.41, 1.8))
        self.assertEqual(sizes["ros"], (0.392, 0.616))
        self.assertEqual(sizes["ros_swap"], (0.616, 0.392))
        self.assertEqual(set(candidate_sizes(pedestrian(), None)), {"grpc"})

    def test_containment_respects_heading(self):
        points = walker_points(90.0)
        self.assertEqual(containment(points, (0, 0), 90.0, .616, .392, .1), 1.0)
        self.assertLess(containment(points, (0, 0), 90.0, .41, 1.8, .1), 1.0)

    def test_score_prefers_tight_box_that_contains_the_walker(self):
        points = walker_points(30.0)
        row = score_observation(points, np.full(len(points), 1.0), pedestrian(30.0), ROS_MAN5, None)
        self.assertEqual(row["scores"]["grpc+ros_swap"]["containment"], 1.0)
        self.assertLess(row["scores"]["grpc+grpc"]["containment"], 1.0)
        chosen = select_model_geometry([row] * 12)
        self.assertEqual((chosen["center_source"], chosen["size_source"]), ("grpc", "ros_swap"))

    def test_low_object_points_are_measured_from_the_local_road(self):
        # base_link is 0.30 m above the road: a 0.43 m box spans -0.30..0.13 m.
        box = np.array([(x, y) for x in np.linspace(-.3, .3, 7) for y in np.linspace(-.3, .3, 7)])
        road = np.array([(r * math.cos(a), r * math.sin(a)) for r in (2.6, 3.0, 3.4)
                         for a in np.linspace(0, 2 * math.pi, 30, endpoint=False)])
        cloud = np.vstack((box, road))
        height = np.r_[np.full(len(box), .10), np.full(len(road), -.30)]
        item = {"kind": "obstacle", "model_name": "CargoBox", "position": [0., 0., 0.],
                "heading_deg": 0., "size_wlh_m": [.75, .75, .43]}
        row = score_observation(cloud, height, item, None, None)
        self.assertEqual(row["points"], len(box))
        self.assertEqual(row["scores"]["grpc+grpc"]["containment"], 1.0)

    def test_background_beyond_the_candidate_shell_is_ignored(self):
        points = walker_points(0.0)
        curb = np.array([(x, 1.6) for x in np.linspace(-1., 1., 20)])  # 1.6 m to the side
        cloud = np.vstack((points, curb))
        row = score_observation(cloud, np.ones(len(cloud)), pedestrian(), ROS_MAN5, None)
        self.assertEqual(row["scores"]["grpc+ros_swap"]["containment"], 1.0)

    def test_points_of_a_neighbouring_actor_are_not_counted(self):
        points = walker_points(0.0)
        neighbour_points = points + [0.9, 0.0]  # a second walker 0.9 m ahead
        cloud = np.vstack((points, neighbour_points))
        neighbour = dict(pedestrian(), id="6", instance_key="q", position=[0.9, 0.0, 0.0],
                         size_wlh_m=[0.4, 0.6, 1.8])
        crowded = score_observation(cloud, np.ones(len(cloud)), pedestrian(), ROS_MAN5, None)
        alone = score_observation(cloud, np.ones(len(cloud)), pedestrian(), ROS_MAN5, None,
                                  others=[neighbour])
        self.assertLess(crowded["scores"]["grpc+ros_swap"]["containment"], 1.0)
        self.assertEqual(alone["scores"]["grpc+ros_swap"]["containment"], 1.0)

    def test_too_few_observations_are_not_evidence_but_contrary_evidence_is(self):
        good = {"scores": {"grpc+grpc": {"containment": 1.0, "area_m2": 1.0}}}
        bad = {"scores": {"grpc+grpc": {"containment": .6, "area_m2": 1.0}}}
        self.assertEqual(select_model_geometry([good] * 3)["reason"], "insufficient_lidar_observations")
        contrary = select_model_geometry([bad] * DEFAULT_REGISTRY_POLICY["min_observations"])
        self.assertFalse(contrary["approved"])
        self.assertEqual(contrary["reason"], "no_candidate_meets_lidar_containment")

    def registry(self, **entry):
        return {"models": {model_key(pedestrian()): dict(
            {"approved": True, "center_source": "grpc", "size_source": "ros_swap", "reason": None}, **entry)}}

    def test_registry_replaces_catalog_decision_and_size(self):
        v4 = {"reasons": ["model_absent_from_measured_catalog"], "api_box_geometry_candidate": False}
        result = check_fused_object_geometry(pedestrian(), v4, self.registry(), ROS_MAN5, [1, 1, 1])
        self.assertTrue(result["api_box_geometry_candidate"])
        self.assertEqual(result["training_size_wl_m"], [.392, .616])
        self.assertEqual(result["training_center_world_xy"], [0.0, 0.0])

    def test_without_registry_evidence_v4_decides_except_for_pedestrians(self):
        v4 = {"reasons": [], "api_box_geometry_candidate": True, "training_center_world_xy": [1., 2.]}
        vehicle = dict(pedestrian(), kind="npc", model_name="DEFAULT_SEDAN")
        for registry in ({"models": {}}, {"models": {model_key(vehicle): {
                "approved": False, "reason": "insufficient_lidar_observations"}}}):
            result = check_fused_object_geometry(vehicle, v4, registry, None, [1, 1, 1])
            self.assertTrue(result["api_box_geometry_candidate"])
            self.assertEqual(result["training_center_world_xy"], [1., 2.])
        # A catalog-approved pedestrian still carries its bind-pose gRPC box.
        masked = check_fused_object_geometry(pedestrian(), v4, {"models": {}}, ROS_MAN5, [1, 1, 1])
        self.assertFalse(masked["api_box_geometry_candidate"])
        self.assertEqual(masked["reasons"], ["pedestrian_requires_lidar_registry"])

    def test_contrary_lidar_masks_even_a_catalog_approved_model(self):
        v4 = {"reasons": [], "api_box_geometry_candidate": True, "training_center_world_xy": [1., 2.]}
        registry = self.registry(approved=False, reason="no_candidate_meets_lidar_containment")
        result = check_fused_object_geometry(pedestrian(), v4, registry, ROS_MAN5, [1, 1, 1])
        self.assertFalse(result["api_box_geometry_candidate"])
        self.assertNotIn("training_center_world_xy", result)

    def test_missing_sources_mask_instead_of_guessing(self):
        v4 = {"reasons": [], "api_box_geometry_candidate": True}
        no_ros = check_fused_object_geometry(pedestrian(), v4, self.registry(), None, [1, 1, 1])
        self.assertEqual(no_ros["reasons"], ["registry_ros_size_unavailable"])
        no_v4 = check_fused_object_geometry(pedestrian(), v4, self.registry(center_source="v4"), ROS_MAN5, [1, 1, 1])
        self.assertEqual(no_v4["reasons"], ["registry_v4_centre_unavailable"])
        obstacle = dict(pedestrian(), kind="obstacle")
        unknown_scale = check_fused_object_geometry(obstacle, v4, self.registry(), ROS_MAN5, None)
        self.assertEqual(unknown_scale["registry"]["reason"], "instance_scale_unknown")
        self.assertTrue(unknown_scale["api_box_geometry_candidate"])

    def test_fit_centre_is_pivot_plus_reviewed_offset(self):
        # Scooter_02: the box was reported at the pivot; the fit centre is 0.537 m ahead.
        item = dict(pedestrian(90.0), kind="obstacle", model_name="Scooter_02", position=[10., 20., 0.])
        x, y = fit_center(item, (0.0, 0.0), (0.537, 0.022))
        self.assertAlmostEqual(x, 10. - 0.022)
        self.assertAlmostEqual(y, 20. + 0.537)
        registry = self.registry(center_source="fit", size_source="grpc", center_offset_local_xy=[0.537, 0.022])
        registry["models"][model_key(item)] = registry["models"].pop(model_key(pedestrian()))
        v4 = {"reasons": ["model_absent_from_measured_catalog"], "api_box_geometry_candidate": False}
        result = check_fused_object_geometry(item, v4, registry, None, [1, 1, 1], (0.0, 0.0))
        self.assertTrue(result["api_box_geometry_candidate"])
        self.assertAlmostEqual(result["training_center_world_xy"][1], 20.537)
        no_pivot = check_fused_object_geometry(item, v4, registry, None, [1, 1, 1], None)
        self.assertEqual(no_pivot["reasons"], ["registry_fit_centre_unavailable"])

    def test_manifest_trains_on_the_selected_size(self):
        gt = {"object_gt": {"objects": [pedestrian()]}}
        geometry = {"objects": [{"instance_key": "p", "api_box_geometry_candidate": True,
                                 "training_size_wl_m": [.392, .616]}]}
        entry = objects_for_entry(gt, geometry)[0]
        self.assertEqual((entry["width"], entry["length"]), (.392, .616))
        del geometry["objects"][0]["training_size_wl_m"]
        entry = objects_for_entry(gt, geometry)[0]
        self.assertEqual((entry["width"], entry["length"]), (1.8, .41))


if __name__ == "__main__":
    unittest.main()
