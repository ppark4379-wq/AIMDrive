from pathlib import Path
import sys
import unittest


INTERNAL = Path(__file__).resolve().parents[1] / "internal_sensor_record"
sys.path.insert(0, str(INTERNAL))
sys.path.insert(0, str(INTERNAL / "tools"))

from audit_scenario_interventions import (
    scenario_zone_contract,
    summarize_roundabout_observation,
    summarize_static_blockage_releases,
    summarize_traffic_interventions,
    zone_observation_checks,
)
from build_sensor_record_training_dataset import required_training_coverage


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

    def test_static_blockage_release_must_be_disclosed_and_verified(self):
        removed = [{"actor_id": "7", "destroy_accepted": True, "absent_after_readback": True}]
        released = {"event": "static_blockage_released", "actor_ids": ["7"], "removed": removed,
                    "intervention": True, "mission_distribution_changed": True,
                    "training_visible": False}
        lifecycle = {"event": "actor_lifecycle_finished", "action": "static_blockage_release",
                     "success": True}
        report = summarize_static_blockage_releases([released, lifecycle])
        self.assertEqual(report["releases"], 1)
        self.assertTrue(report["all_events_disclosed"] and report["all_removals_verified"])
        self.assertEqual(summarize_static_blockage_releases([])["releases"], 0)
        hidden = dict(released, training_visible=True)
        self.assertFalse(summarize_static_blockage_releases([hidden, lifecycle])["all_events_disclosed"])
        left = dict(released, removed=[dict(removed[0], absent_after_readback=False)])
        self.assertFalse(summarize_static_blockage_releases([left, lifecycle])["all_removals_verified"])
        self.assertFalse(summarize_static_blockage_releases([released])["all_removals_verified"])

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

    def test_random_count_roundabout_accepts_configured_range(self):
        actors = ["6", "10", "13", "15", "18"]
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
                "actor_ids": actors,
            },
        ]
        report = summarize_roundabout_observation(
            events,
            {"min_count": 3, "max_count": 5, "speed_mps": 4.0},
        )
        self.assertEqual(report["observed_counts"], [5])
        self.assertTrue(report["spawn_matches_config"])
        self.assertTrue(report["lifecycle_complete"])


    def test_six_scenarios_read_enabled_zones_from_main_runner(self):
        self.assertEqual(
            scenario_zone_contract("full_loop"),
            {
                "static": True,
                "dynamic": True,
                "roundabout": True,
                "highway": True,
                "tunnel": True,
            },
        )
        expected = {
            "full_loop_normal": set(),
            "full_loop_static": {"static"},
            "full_loop_dynamic": {"dynamic"},
            "full_loop_roundabout": {"roundabout"},
            "full_loop_roundabout_nonuniform": {"roundabout"},
            "full_loop_roundabout_speed_random": {"roundabout"},
            "full_loop_roundabout_brake": {"roundabout"},
            "full_loop_highway": {"highway"},
        }
        for scenario_name, enabled in expected.items():
            zones = scenario_zone_contract(scenario_name)
            self.assertEqual(
                {name for name, active in zones.items() if active},
                enabled,
            )

    def test_each_scenario_requires_only_its_selected_events(self):
        inactive_roundabout = {
            "spawn_events": 0,
            "spawn_lifecycle_successes": 0,
            "retire_lifecycle_successes": 0,
            "spawn_matches_config": False,
            "lifecycle_complete": False,
        }
        active_roundabout = dict(
            inactive_roundabout,
            spawn_events=1,
            spawn_lifecycle_successes=1,
            retire_lifecycle_successes=1,
            spawn_matches_config=True,
            lifecycle_complete=True,
        )
        cases = {
            "full_loop": dict(
                selected_static="Cone",
                selected_dynamic="Pedestrian",
                dynamic_phases=["waiting", "crossing", "cleared"],
                roundabout_observation=active_roundabout,
                highway_samples=2,
                tunnel_phases=["armed", "completed"],
            ),
            "full_loop_normal": dict(
                selected_static="OFF",
                selected_dynamic="OFF",
                dynamic_phases=["disabled"],
                roundabout_observation=inactive_roundabout,
                highway_samples=0,
                tunnel_phases=["disabled"],
            ),
            "full_loop_static": dict(
                selected_static="Cone",
                selected_dynamic="OFF",
                dynamic_phases=["disabled"],
                roundabout_observation=inactive_roundabout,
                highway_samples=0,
                tunnel_phases=["disabled"],
            ),
            "full_loop_dynamic": dict(
                selected_static="OFF",
                selected_dynamic="Pedestrian",
                dynamic_phases=["waiting", "crossing"],
                roundabout_observation=inactive_roundabout,
                highway_samples=0,
                tunnel_phases=["disabled"],
            ),
            "full_loop_roundabout": dict(
                selected_static="OFF",
                selected_dynamic="OFF",
                dynamic_phases=["disabled"],
                roundabout_observation=active_roundabout,
                highway_samples=0,
                tunnel_phases=["disabled"],
            ),
            "full_loop_roundabout_nonuniform": dict(
                selected_static="OFF",
                selected_dynamic="OFF",
                dynamic_phases=["disabled"],
                roundabout_observation=active_roundabout,
                highway_samples=0,
                tunnel_phases=["disabled"],
            ),
            "full_loop_roundabout_speed_random": dict(
                selected_static="OFF",
                selected_dynamic="OFF",
                dynamic_phases=["disabled"],
                roundabout_observation=active_roundabout,
                highway_samples=0,
                tunnel_phases=["disabled"],
            ),
            "full_loop_roundabout_brake": dict(
                selected_static="OFF",
                selected_dynamic="OFF",
                dynamic_phases=["disabled"],
                roundabout_observation=active_roundabout,
                highway_samples=0,
                tunnel_phases=["disabled"],
            ),
            "full_loop_highway": dict(
                selected_static="OFF",
                selected_dynamic="OFF",
                dynamic_phases=["disabled"],
                roundabout_observation=inactive_roundabout,
                highway_samples=2,
                tunnel_phases=["disabled"],
            ),
        }
        for scenario_name, observations in cases.items():
            with self.subTest(scenario_name=scenario_name):
                checks = zone_observation_checks(
                    scenario_name,
                    scenario_zone_contract(scenario_name),
                    static_models=["Cone"],
                    dynamic_models=["Pedestrian"],
                    **observations,
                )
                self.assertTrue(all(checks.values()), checks)
                if scenario_name == "full_loop":
                    self.assertEqual(
                        set(checks),
                        {
                            "selected_static_from_pool",
                            "selected_dynamic_from_pool",
                            "roundabout_observed",
                        },
                    )

    def test_disabled_zone_activity_is_rejected(self):
        checks = zone_observation_checks(
            "full_loop_normal",
            scenario_zone_contract("full_loop_normal"),
            selected_static="Cone",
            selected_dynamic="OFF",
            static_models=["Cone"],
            dynamic_models=["Pedestrian"],
            dynamic_phases=["disabled"],
            roundabout_observation={
                "spawn_events": 0,
                "spawn_lifecycle_successes": 0,
                "retire_lifecycle_successes": 0,
                "spawn_matches_config": False,
                "lifecycle_complete": False,
            },
            highway_samples=0,
            tunnel_phases=["disabled"],
        )
        self.assertFalse(checks["selected_static_from_pool"])



    def test_training_coverage_matches_selected_scenario(self):
        self.assertEqual(
            required_training_coverage("full_loop"),
            (
                "waypoint_frames",
                "object_positive_frames",
                "traffic_light_0",
                "traffic_light_1",
            ),
        )
        self.assertEqual(
            required_training_coverage("full_loop_normal"),
            ("waypoint_frames", "traffic_light_0", "traffic_light_1"),
        )
        for scenario_name in (
            "full_loop_static",
            "full_loop_dynamic",
            "full_loop_roundabout",
            "full_loop_roundabout_nonuniform",
            "full_loop_roundabout_speed_random",
            "full_loop_roundabout_brake",
            "full_loop_highway",
        ):
            self.assertEqual(
                required_training_coverage(scenario_name),
                ("waypoint_frames", "object_positive_frames"),
            )



if __name__ == "__main__":
    unittest.main()
