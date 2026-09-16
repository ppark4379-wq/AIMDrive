from pathlib import Path
import sys
import unittest


INTERNAL_ROOT = Path(__file__).resolve().parents[1]
if str(INTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(INTERNAL_ROOT))

from morai_pretraining.bev_geometry import conservative_training_bev_relation


def actor(x, y, width=2.0, length=4.0):
    return {
        "position": [x, y, 0.0],
        "position_semantics": "grpc_bounding_box_world_center",
        "size_wlh_m": [width, length, 1.5],
    }


class ConservativeTrainingBevRelationTest(unittest.TestCase):
    def setUp(self):
        self.ego = {"x": 0.0, "y": 0.0, "yaw": 0.0}

    def test_far_visible_actor_is_proven_outside(self):
        result = conservative_training_bev_relation(actor(300.0, 0.0), self.ego)
        self.assertFalse(result["training_bev_relevant"])
        self.assertTrue(result["bev_exclusion_proven"])

    def test_inside_actor_remains_relevant(self):
        result = conservative_training_bev_relation(actor(10.0, 0.0), self.ego)
        self.assertTrue(result["training_bev_relevant"])
        self.assertFalse(result["bev_exclusion_proven"])

    def test_box_near_boundary_is_not_excluded_by_center_only(self):
        result = conservative_training_bev_relation(
            actor(33.0, 0.0, width=2.0, length=6.0), self.ego)
        self.assertTrue(result["training_bev_relevant"])

    def test_invalid_box_fails_closed_as_relevant(self):
        malformed = actor(300.0, 0.0)
        malformed["size_wlh_m"][0] = 0.0
        result = conservative_training_bev_relation(malformed, self.ego)
        self.assertTrue(result["training_bev_relevant"])
        self.assertEqual(result["bev_relation"], "indeterminate_fail_closed")

    def test_ego_yaw_is_applied(self):
        ego = {"x": 0.0, "y": 0.0, "yaw": 1.5707963267948966}
        result = conservative_training_bev_relation(actor(0.0, 10.0), ego)
        self.assertTrue(result["training_bev_relevant"])
        self.assertAlmostEqual(result["center_forward_m"], 10.0, places=6)


if __name__ == "__main__":
    unittest.main()
