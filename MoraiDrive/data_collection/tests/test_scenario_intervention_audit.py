from pathlib import Path
import sys
import unittest


INTERNAL = Path(__file__).resolve().parents[1] / "internal_sensor_record"
sys.path.insert(0, str(INTERNAL))
sys.path.insert(0, str(INTERNAL / "tools"))

from audit_scenario_interventions import (
    summarize_roundabout_observation,
    summarize_traffic_interventions,
)


class ScenarioInterventionAuditTest(unittest.TestCase):
    def test_delayed_exact_id_readback_is_accepted_and_disclosed(self):
        common = {
            "light_id": "C-test",
            "intervention": True,
            "mission_distribution_changed": True,
            "training_visible": False,
        }
        report = summarize_traffic_interventions(
            [
                dict(
                    common,
                    event="traffic_light_override",
                    result={"accepted": True, "verified": False},
                ),
                dict(common, event="traffic_light_override_verified"),
            ]
        )
        self.assertEqual(report["commands"], 1)
        self.assertEqual(report["verified"], 1)
        self.assertTrue(report["all_commands_verified"])
        self.assertTrue(report["all_events_disclosed"])

    def test_timeout_or_undisclosed_intervention_fails(self):
        report = summarize_traffic_interventions(
            [
                {
                    "event": "traffic_light_override",
                    "light_id": "C-test",
                    "result": {"accepted": True, "verified": False},
                },
                {
                    "event": "traffic_light_override_verification_timeout",
                    "light_id": "C-test",
                },
            ]
        )
        self.assertFalse(report["all_commands_verified"])
        self.assertFalse(report["all_events_disclosed"])

    def test_three_car_roundabout_lifecycle_uses_explicit_contract_events(self):
        actors = ["6", "10", "13"]
        events = [
            {
                "event": "actor_lifecycle_finished",
                "action": "roundabout_spawn",
                "success": True,
            },
            {
                "event": "roundabout_spawn_contract",
                "actor_ids": actors,
                "target_speed_mps": 4.0,
                "actor_motion_contracts": [
                    {
                        "actor_id": actor_id,
                        "speed_limit_acknowledged": True,
                        "initial_velocity_acknowledged": True,
                    }
                    for actor_id in actors
                ],
            },
            {
                "event": "actor_lifecycle_finished",
                "action": "roundabout_retire",
                "success": True,
                "actor_ids": list(reversed(actors)),
            },
        ]
        report = summarize_roundabout_observation(
            events,
            {"count": 3, "speed_mps": 4.0},
        )
        self.assertEqual(report["observed_counts"], [3])
        self.assertTrue(report["spawn_matches_config"])
        self.assertTrue(report["lifecycle_complete"])

    def test_roundabout_missing_retire_or_wrong_count_is_rejected(self):
        report = summarize_roundabout_observation(
            [{
                "event": "roundabout_spawn_contract",
                "actor_ids": ["6", "10"],
                "target_speed_mps": 4.0,
                "actor_motion_contracts": [],
            }],
            {"count": 3, "speed_mps": 4.0},
        )
        self.assertFalse(report["spawn_matches_config"])
        self.assertFalse(report["lifecycle_complete"])


if __name__ == "__main__":
    unittest.main()
