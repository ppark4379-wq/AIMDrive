import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ..adapters.morai_grpc import MoraiGrpcClient
from ..adapters.object_ghost_audit import ObjectGhostAudit
from ..adapters.expert_controller import ExpertController
from ..main_runner import (
    PACKAGE_ROOT,
    SUPPORTED_SCENARIOS,
    build_parser,
    load_configuration,
)
from ..tools.verify_separation import verify_package


class SeparatedPackageTest(unittest.TestCase):
    def test_parser_accepts_finite_run_limit(self):
        args = build_parser().parse_args(
            ["--scenario", "full_loop", "--max-runs", "50"]
        )
        self.assertEqual(args.max_runs, 50)
        self.assertFalse(args.once)

    def test_parser_accepts_highway_object_audit(self):
        args = build_parser().parse_args(
            [
                "--scenario",
                "full_loop_highway_tunnel_test",
                "--highway-object-audit",
            ]
        )
        self.assertTrue(args.highway_object_audit)

    def test_progress_watchdog_is_enabled_for_batch_runs(self):
        _, _, scenario_cfg = load_configuration()
        watchdog = scenario_cfg["progress_watchdog"]
        self.assertTrue(watchdog["enabled"])
        self.assertGreaterEqual(watchdog["timeout_sec"], 90.0)
        self.assertGreater(watchdog["minimum_progress_m"], 0.0)

    def test_supported_scenarios_are_exact(self):
        self.assertEqual(
            set(SUPPORTED_SCENARIOS),
            {
                "full_loop",
                "full_loop_after_static",
                "full_loop_highway_tunnel_test",
                "full_loop_normal",
                "full_loop_static",
                "full_loop_dynamic",
                "full_loop_roundabout",
                "full_loop_highway",
            },
        )

    def test_object_ghost_audit_matches_by_live_position(self):
        ros_npcs = [
            {
                "unique_id": 10,
                "name": "matched",
                "x": 1.0,
                "y": 2.0,
            },
            {
                "unique_id": 20,
                "name": "ros-only",
                "x": 50.0,
                "y": 50.0,
            },
        ]
        grpc_vehicles = [
            {"actor_id": "vehicle-10", "x": 1.5, "y": 2.0},
            {"actor_id": "grpc-only", "x": -20.0, "y": -20.0},
        ]
        matches, ros_only, grpc_only = (
            ObjectGhostAudit.compare_snapshots(
                ros_npcs,
                grpc_vehicles,
                match_distance_m=5.0,
            )
        )
        self.assertEqual(
            [item["ros_unique_id"] for item in matches], [10]
        )
        self.assertEqual([item["unique_id"] for item in ros_only], [20])
        self.assertEqual(
            [item["actor_id"] for item in grpc_only], ["grpc-only"]
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

    def test_native_highway_spawn_points_match_competition_highway_subset(self):
        global_cfg, _, scenario_cfg = load_configuration()
        highway_cfg = scenario_cfg['highway_npc']
        roundabout_cfg = scenario_cfg["roundabout_npc"]
        specs = highway_cfg['spawn_points']
        lifecycle = highway_cfg['spawn_point_lifecycle']
        self.assertEqual(len(specs), 3)
        self.assertGreaterEqual(lifecycle['create_attempts'], 2)
        self.assertGreaterEqual(lifecycle['enable_attempts'], 3)
        self.assertGreaterEqual(lifecycle['disable_attempts'], 2)
        self.assertGreaterEqual(lifecycle["max_runtime_replacements"], 1)
        self.assertGreater(lifecycle['cleanup_settle_sec'], 0.0)
        self.assertFalse(
            any(str(key).startswith('lead_') for key in highway_cfg),
            'Scripted highway lead settings must stay removed',
        )
        self.assertGreater(highway_cfg["background_warmup_distance_m"], 0.0)
        self.assertEqual(roundabout_cfg["count"], 6)
        self.assertEqual(roundabout_cfg["speed_mps"], 4.0)
        self.assertEqual(roundabout_cfg["route_laps"], 100)
        self.assertEqual(highway_cfg['reference_safety_distance_m'], 5.0)
        self.assertEqual(highway_cfg['reference_ttc_sec'], 1.0)
        self.assertEqual(scenario_cfg['highway_tunnel_test_start_path_index'], 2275)

        expected = {
            10: (10.0, 60.0, 'A2256W000029', 167, 'A2256W000432', 246),
            6: (10.0, 40.0, 'A2256W000406', 78, 'A2256W001105', 83),
            8: (10.0, 50.0, 'A2256W000175', 78, 'A2256W000422', 59),
        }
        for spec in specs:
            values = expected[int(spec['id'])]
            self.assertEqual(spec['min_spawn_period_sec'], values[0])
            self.assertEqual(spec['max_spawn_period_sec'], values[0])
            self.assertEqual(spec['min_desired_velocity_custom_kmh'], values[1])
            self.assertEqual(spec['max_desired_velocity_custom_kmh'], values[1])
            self.assertEqual(spec['start_link']['id'], values[2])
            self.assertEqual(spec['start_link']['point_idx'], values[3])
            self.assertEqual(spec['end_link']['id'], values[4])
            self.assertEqual(spec['end_link']['point_idx'], values[5])
            self.assertEqual(spec['maximum_spawn_vehicle'], 50)
            self.assertFalse(spec['is_lane_change'])
            self.assertFalse(spec['is_close_loop'])

        # Build a real protobuf request through the production adapter without
        # connecting to MORAI, so field-name/version mismatches fail offline.
        class CaptureAdapter:
            def __init__(self):
                self.param = None

            def create_vehicle_spawn_point(self, param):
                self.param = param
                return SimpleNamespace(
                    status=1,
                    description='',
                    custom_message='runtime-spawn-10',
                )

        capture = CaptureAdapter()
        client = MoraiGrpcClient(global_cfg)
        client.world = SimpleNamespace(_sim_adapter=capture)
        created_id = client.create_vehicle_spawn_point(specs[0])
        self.assertEqual(created_id, 'runtime-spawn-10')
        param = capture.param
        self.assertEqual(param.spawn_point_info.object_info.id.value, '')
        self.assertEqual(param.config.start_link_info.id.value, 'A2256W000029')
        self.assertEqual(param.config.start_link_info.waypoint_idx, 167)
        self.assertEqual(param.config.end_link_info.id.value, 'A2256W000432')
        self.assertEqual(param.config.end_link_info.waypoint_idx, 246)
        self.assertFalse(param.config.is_lane_change)
        self.assertEqual(param.config.maximum_spawn_vehicle, 50)

    def test_vehicle_ai_enable_retries_transient_failure(self):
        global_cfg, _, _ = load_configuration()

        class FlakyVehicle:
            def __init__(self):
                self.results = iter((False, False, True))
                self.calls = 0

            def set_ai(self, enabled):
                self.calls += 1
                return next(self.results)

        client = MoraiGrpcClient(global_cfg)
        vehicle = FlakyVehicle()
        with patch("scenario_runner.adapters.morai_grpc.time.sleep") as sleep:
            enabled = client.set_vehicle_ai(
                vehicle,
                True,
                attempts=5,
                retry_delay_sec=0.25,
            )

        self.assertTrue(enabled)
        self.assertEqual(vehicle.calls, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_spawn_point_enable_retries_transient_morai_failure(self):
        global_cfg, _, _ = load_configuration()

        class FlakyWorld:
            def __init__(self):
                self.results = iter((False, False, True))
                self.calls = 0

            def enable_spawn_point(self, spawn_point_id, enabled):
                self.calls += 1
                return next(self.results)

        client = MoraiGrpcClient(global_cfg)
        client.world = FlakyWorld()
        with patch('scenario_runner.adapters.morai_grpc.time.sleep') as sleep:
            enabled = client.enable_vehicle_spawn_point(
                'runtime-5',
                True,
                attempts=5,
                retry_delay_sec=0.5,
            )

        self.assertTrue(enabled)
        self.assertEqual(client.world.calls, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_failed_spawn_point_runtime_id_is_replaced(self):
        scenario_class = SUPPORTED_SCENARIOS["full_loop"]
        scenario = scenario_class.__new__(scenario_class)
        scenario.cfg = {
            "highway_npc": {
                "background_warmup_distance_m": 300.0,
                "spawn_point_activation_retry_period_sec": 1.0,
                "spawn_point_lifecycle": {
                    "enable_attempts": 5,
                    "enable_retry_delay_sec": 0.5,
                    "create_attempts": 3,
                    "create_settle_sec": 0.5,
                    "recreate_delay_sec": 1.0,
                    "max_runtime_replacements": 3,
                },
            }
        }
        scenario.highway_activation_s = 100.0
        scenario.highway_spawn_points_active = False
        scenario.highway_spawn_point_activation_started = False
        scenario.highway_spawn_point_next_retry_at = 0.0
        original = {
            "id": "runtime-5",
            "name": "competition_spawn_8",
            "enabled": False,
            "retired": False,
            "generation": 0,
            "activation_failures": 0,
            "spec": {"id": 8},
        }
        scenario.highway_spawn_points = [original]

        class FakeGrpc:
            def __init__(self):
                self.created = 0

            def enable_vehicle_spawn_point(
                self, spawn_point_id, enabled, attempts, retry_delay_sec
            ):
                return str(spawn_point_id) == "runtime-14"

            def create_vehicle_spawn_point(self, spec):
                self.created += 1
                return "runtime-14"

        scenario.grpc = FakeGrpc()
        with patch("scenario_runner.scenarios.full_loop_scenario.time.sleep"):
            active = scenario.activate_highway_spawn_points(
                elapsed=1.0,
                current_s=0.0,
            )

        self.assertTrue(active)
        self.assertTrue(original["retired"])
        self.assertEqual(scenario.grpc.created, 1)
        current = [
            point for point in scenario.highway_spawn_points
            if not point["retired"]
        ]
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["id"], "runtime-14")
        self.assertTrue(current[0]["enabled"])

    def test_ambiguous_vehicle_spawn_recovers_without_duplicate(self):
        global_cfg, _, _ = load_configuration()

        class FakeAdapter:
            def __init__(self, world):
                self.world = world

            def get_actor_state(self, actor_info):
                if not self.world.created:
                    return None
                actor_id = actor_info.id.value
                return SimpleNamespace(
                    actor_info=SimpleNamespace(
                        id=SimpleNamespace(value=actor_id),
                    )
                )

        class FakeWorld:
            def __init__(self):
                self.created = False
                self.spawn_calls = 0
                self._actors = {}
                self._sim_adapter = FakeAdapter(self)

            def spawn_vehicle(self, **kwargs):
                self.spawn_calls += 1
                self.created = True
                return None

        client = MoraiGrpcClient(global_cfg)
        client.world = FakeWorld()
        with patch("scenario_runner.adapters.morai_grpc.time.sleep"):
            actor = client.spawn_vehicle(
                transform=SimpleNamespace(),
                model_name="DEFAULT_SEDAN",
                label="test_vehicle",
                velocity=0.0,
                multi_ego=False,
                request_id="scenario_test_vehicle",
                recover_ambiguous=True,
            )
            same_actor = client.spawn_vehicle(
                transform=SimpleNamespace(),
                model_name="DEFAULT_SEDAN",
                label="test_vehicle",
                velocity=0.0,
                multi_ego=False,
                request_id="scenario_test_vehicle",
                recover_ambiguous=True,
            )

        self.assertIsNotNone(actor)
        self.assertIsNotNone(same_actor)
        self.assertEqual(actor.get_id(), "scenario_test_vehicle")
        self.assertEqual(client.world.spawn_calls, 1)

    def test_ambiguous_pedestrian_spawn_recovers_actual_actor(self):
        global_cfg, _, _ = load_configuration()

        class FakeAdapter:
            pass

        class FakeWorld:
            def __init__(self):
                self.spawn_calls = 0
                self._actors = {}
                self._sim_adapter = FakeAdapter()
                self.states = []

            def get_all_actors_state(self, **kwargs):
                return SimpleNamespace(states=list(self.states))

            def spawn_pedestrian(self, **kwargs):
                self.spawn_calls += 1
                transform = kwargs["transform"]
                self.states.append(
                    SimpleNamespace(
                        actor_info=SimpleNamespace(
                            id=SimpleNamespace(value="ambiguous-ped-1"),
                        ),
                        transform=SimpleNamespace(
                            location=SimpleNamespace(
                                x=transform.location.x,
                                y=transform.location.y,
                            )
                        ),
                    )
                )
                return None

        client = MoraiGrpcClient(global_cfg)
        client.world = FakeWorld()
        transform = SimpleNamespace(
            location=SimpleNamespace(x=-56.74, y=-11.61)
        )
        with patch("scenario_runner.adapters.morai_grpc.time.sleep"):
            actor = client.spawn_pedestrian(
                transform=transform,
                model_name="Man1",
                label="dynamic-test",
                velocity=9.0,
                active_dist=0.0,
                move_dist=15.0,
                start_action=True,
            )

        self.assertIsNotNone(actor)
        self.assertEqual(actor.get_id(), "ambiguous-ped-1")
        self.assertTrue(
            getattr(actor, "_scenario_runner_ambiguous_recovery", False)
        )
        self.assertEqual(client.world.spawn_calls, 1)

    def test_dynamic_pedestrian_cleanup_only_removes_start_area(self):
        global_cfg, _, _ = load_configuration()

        def state(actor_id, x, y):
            return SimpleNamespace(
                actor_info=SimpleNamespace(
                    id=SimpleNamespace(value=actor_id),
                ),
                transform=SimpleNamespace(
                    location=SimpleNamespace(x=x, y=y),
                ),
            )

        class FakeAdapter:
            def __init__(self, world):
                self.world = world

            def destroy_actor(self, actor_info):
                actor_id = actor_info.id.value
                self.world.states = [
                    item
                    for item in self.world.states
                    if item.actor_info.id.value != actor_id
                ]
                return SimpleNamespace(
                    status=1,
                    description="Success",
                )

        class FakeWorld:
            def __init__(self):
                self._actors = {}
                self.states = [
                    state("stale-at-start", -56.7, -11.6),
                    state("unrelated-far-away", -70.5, -11.9),
                ]
                self._sim_adapter = FakeAdapter(self)

            def get_all_actors_state(self, **kwargs):
                return SimpleNamespace(states=list(self.states))

        client = MoraiGrpcClient(global_cfg)
        client.world = FakeWorld()
        removed = client.destroy_pedestrians_near(
            -56.74,
            -11.61,
            max_dist_m=3.0,
        )

        self.assertEqual(removed, ["stale-at-start"])
        remaining_ids = {
            item.actor_info.id.value for item in client.world.states
        }
        self.assertEqual(remaining_ids, {"unrelated-far-away"})

    def test_dynamic_obstacle_spawns_active_with_checkpoint_b_to_a_heading(self):
        scenario_class = SUPPORTED_SCENARIOS["full_loop"]
        scenario = scenario_class.__new__(scenario_class)
        scenario.dynamic_obstacle_start = {
            "x": -56.74,
            "y": -11.61,
            "z": 28.56,
        }
        scenario.dynamic_obstacle_end = {
            "x": -70.57,
            "y": -11.93,
            "z": 28.43,
        }
        scenario.dynamic_obstacle_path_length_m = 13.8337
        scenario.dynamic_obstacle_yaw_deg = -178.7
        scenario.dynamic_obstacle_speed_mps = 2.5
        scenario.dynamic_obstacle_api_speed_kph = 9.0
        scenario.dynamic_obstacle_model = "Man1"
        scenario.dynamic_obstacle_models = ["Man1"]
        scenario.dynamic_obstacle_min_ego_distance_m = float("inf")
        scenario.dynamic_obstacle_direction_checked = False
        scenario.dynamic_obstacle_direction_respawned = False
        scenario.dynamic_obstacle_last_print_sec = -1
        scenario.profile_index = 0
        scenario.scenario_actors = []

        class Actor:
            def __init__(self):
                self.pause_calls = []

            def set_pause(self, value):
                self.pause_calls.append(value)
                return True

        actor = Actor()

        class FakeGrpc:
            def __init__(self):
                self.spawn_kwargs = None
                self.transform_args = None

            def destroy_pedestrians_near(self, *args, **kwargs):
                return []

            def make_transform(self, *args):
                self.transform_args = args
                return SimpleNamespace()

            def spawn_pedestrian(self, **kwargs):
                self.spawn_kwargs = kwargs
                return actor

        scenario.grpc = FakeGrpc()
        scenario._spawn_active_dynamic_obstacle()

        self.assertEqual(scenario.grpc.spawn_kwargs["velocity"], 9.0)
        self.assertTrue(scenario.grpc.spawn_kwargs["start_action"])
        self.assertAlmostEqual(scenario.grpc.transform_args[3], -178.7)
        self.assertEqual(scenario.dynamic_obstacle_phase, "crossing")

    def _make_crossing_scenario(self, actor, started_at=100.0):
        scenario_class = SUPPORTED_SCENARIOS["full_loop"]
        scenario = scenario_class.__new__(scenario_class)
        scenario.dynamic_obstacle_phase = "crossing"
        scenario.dynamic_obstacle = actor
        scenario.dynamic_obstacle_start = {
            "x": -56.74,
            "y": -11.61,
            "z": 28.56,
        }
        scenario.dynamic_obstacle_end = {
            "x": -70.57,
            "y": -11.93,
            "z": 28.43,
        }
        scenario.dynamic_obstacle_path_length_m = 13.8337
        scenario.dynamic_obstacle_min_ego_distance_m = float("inf")
        scenario.dynamic_obstacle_started_at = started_at
        scenario.dynamic_obstacle_last_print_sec = -1
        scenario.dynamic_obstacle_direction_checked = False
        scenario.dynamic_obstacle_model = "Man1"
        scenario.dynamic_obstacle_arrival_tolerance_m = 1.5
        scenario.dynamic_obstacle_max_crossing_sec = 30.0
        scenario.scenario_actors = [actor]
        return scenario

    def test_stationary_dynamic_obstacle_is_removed_from_b(self):
        class Actor:
            def __init__(self):
                self.destroyed = False

            def get_actor_state(self):
                return SimpleNamespace(
                    transform=SimpleNamespace(
                        location=SimpleNamespace(x=-56.74, y=-11.61)
                    )
                )

            def destroy(self):
                self.destroyed = True
                return True

        actor = Actor()
        scenario = self._make_crossing_scenario(actor)
        with patch(
            "scenario_runner.scenarios.full_loop_scenario.time.time",
            return_value=103.0,
        ):
            with self.assertRaisesRegex(RuntimeError, "did not leave spawn B"):
                scenario.update_dynamic_obstacle(
                    {"x": -50.0, "y": -11.0, "speed": 0.0}, 0.0
                )
        self.assertTrue(actor.destroyed)
        self.assertIsNone(scenario.dynamic_obstacle)

    def test_b_to_c_dynamic_obstacle_is_removed_without_respawn(self):
        class Actor:
            def __init__(self):
                self.destroyed = False

            def get_actor_state(self):
                return SimpleNamespace(
                    transform=SimpleNamespace(
                        location=SimpleNamespace(x=-55.74, y=-11.59)
                    )
                )

            def destroy(self):
                self.destroyed = True
                return True

        actor = Actor()
        scenario = self._make_crossing_scenario(actor)
        scenario._spawn_active_dynamic_obstacle = unittest.mock.Mock()
        with patch(
            "scenario_runner.scenarios.full_loop_scenario.time.time",
            return_value=100.5,
        ):
            with self.assertRaisesRegex(RuntimeError, "required B->A crossing"):
                scenario.update_dynamic_obstacle(
                    {"x": -50.0, "y": -11.0, "speed": 0.0}, 0.0
                )
        self.assertTrue(actor.destroyed)
        self.assertIsNone(scenario.dynamic_obstacle)
        scenario._spawn_active_dynamic_obstacle.assert_not_called()

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
