import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ..adapters import morai_grpc
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
    def test_strict_pedestrian_inventory_cannot_treat_missing_response_as_empty(self):
        client = MoraiGrpcClient.__new__(MoraiGrpcClient)
        client.world = unittest.mock.Mock()
        for response in (None, SimpleNamespace()):
            with self.subTest(response=response):
                client.world.get_all_actors_state.return_value = response
                with self.assertRaisesRegex(RuntimeError, "inventory response is unavailable"):
                    client.get_all_pedestrian_states(strict=True)
                self.assertEqual(client.get_all_pedestrian_states(), [])
        client.world.get_all_actors_state.return_value = SimpleNamespace(states=[])
        self.assertEqual(client.get_all_pedestrian_states(strict=True), [])

    def test_closed_console_cannot_prevent_expert_process_cleanup(self):
        controller = ExpertController.__new__(ExpertController)
        process = unittest.mock.Mock(pid=12345)
        process.poll.return_value = None
        controller.process = process
        controller._terminate_process_group = unittest.mock.Mock()
        controller._clear_owned_ros_registrations = unittest.mock.Mock()
        with patch("builtins.print", side_effect=BrokenPipeError("closed terminal")):
            controller.stop_process()
        controller._terminate_process_group.assert_called_once()
        process.wait.assert_called_once_with(timeout=10.0)
        controller._clear_owned_ros_registrations.assert_called_once_with(12345)
        self.assertIsNone(controller.process)

    def test_closed_console_cannot_prevent_kill_fallback(self):
        import signal
        controller = ExpertController.__new__(ExpertController)
        controller.process = unittest.mock.Mock(pid=12345)
        with patch("os.getpgid", side_effect=OSError("missing group"), create=True), patch(
            "builtins.print", side_effect=BrokenPipeError("closed terminal")
        ):
            controller._terminate_process_group(signal.SIGTERM)
        controller.process.terminate.assert_called_once_with()

    def test_grpc_auto_host_falls_back_to_windows_wsl_adapter(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout="\r\n172.19.0.1\r\n",
        )
        with patch("builtins.open", side_effect=OSError("no route")), patch(
            "scenario_runner.adapters.morai_grpc.subprocess.run",
            return_value=completed,
        ) as run:
            host = morai_grpc._windows_host_gateway()

        self.assertEqual(host, "172.19.0.1")
        command = run.call_args.args[0]
        self.assertIn("powershell.exe", command)
        self.assertIn("*WSL*", command[-1])

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

    def test_collection_scenario_abort_guards_are_disabled(self):
        _, _, scenario_cfg = load_configuration()
        watchdog = scenario_cfg["progress_watchdog"]
        self.assertFalse(watchdog["enabled"])
        self.assertEqual(scenario_cfg["timeout_sec"], 0.0)
        self.assertEqual(
            scenario_cfg["off_route_timeout_sec"], 1000000000000.0
        )
        self.assertEqual(scenario_cfg["route_setup_attempts"], 1)

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
                "full_loop_roundabout_nonuniform",
                "full_loop_roundabout_speed_random",
                "full_loop_roundabout_brake",
                "full_loop_highway",
                "full_loop_tunnel_normal",
                "full_loop_tunnel_static",
                "full_loop_tunnel_dynamic",
                "full_loop_tunnel_vehicle",
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
        self.assertEqual(highway_cfg["background_warmup_distance_m"], 100.0)
        self.assertEqual(highway_cfg["collection_maximum_total_vehicles"], 12)
        self.assertEqual(roundabout_cfg["min_count"], 3)
        self.assertEqual(roundabout_cfg["max_count"], 5)
        self.assertEqual(
            roundabout_cfg["models"],
            [
                "DEFAULT_SEDAN",
                "2016_Hyundai_Sonata",
                "2020_hyundai_sonata",
                "2015_Kia_K5",
                "2020_Kia_K5",
                "2017_Hyundai_Grandeur",
                "2014_Kia_K7",
                "2016_Hyundai_Genesis_DH",
                "2019_Toyota_Prius",
                "2021_Volkswagen_Golf_GTI",
                "2020_Kia_Stinger",
                "2016_Hyundai_Ioniq",
                "2023_Hyundai_Ioniq5",
                "2023_Kia_EV6",
                "2019_Hyundai_Kona(EV)",
                "2017_Kia_Niro(HEV)",
                "2020_Kia_Niro(EV)",
                "2016_Kia_Soul(EV)",
                "2020_Kia_Soul(EV)",
                "2021_Hyundai_G80",
                "2019_hyundai_genesis_g90",
                "2019_hyundai_genesis_g90_4WS",
                "DEFAULT_SUV",
                "2016_Hyundai_Santafe",
                "2017_Kia_Sorento",
                "2020_volvo_xc90",
                "2021_Hyundai_Genesis_GV80",
                "2021_Hyundai_GV70",
                "2019_Hyundai_Nexo",
                "DEFAULT_MPV",
                "DEFAULT_WAGON",
                "2019_kia_sedona",
                "2021_Kia_Carnival",
                "2020_chrysler_pacifica_hybrid",
                "2017_hyundai_solati_h350",
                "DR_KGTorres_EVX_2024",
            ],
        )
        self.assertEqual(roundabout_cfg["speed_mps"], 4.0)
        self.assertEqual(roundabout_cfg["route_laps"], 100)
        self.assertEqual(
            roundabout_cfg["nonuniform_position"]["min_spawn_gap_m"],
            10.0,
        )
        self.assertEqual(
            roundabout_cfg["nonuniform_position"]["tight_gap_max_m"],
            12.0,
        )
        self.assertEqual(
            roundabout_cfg["nonuniform_position"]["gap_weight_exponent"],
            2.5,
        )
        self.assertEqual(
            roundabout_cfg["speed_randomization"],
            {
                "min_speed_mps": 2.5,
                "max_speed_mps": 7.0,
                "slow_band_max_mps": 3.2,
                "fast_band_min_mps": 6.3,
                "interaction_gap_min_m": 35.0,
                "interaction_gap_max_m": 45.0,
                "emergency_gap_m": 6.0,
                "follow_gap_m": 9.0,
                "resume_gap_m": 12.0,
                "control_period_s": 0.2,
                "live_log_period_s": 0.5,
                "ego_gap_radius_m": 6.0,
            },
        )
        self.assertEqual(
            roundabout_cfg["brake_event"],
            {
                "base_speed_mps": 5.5,
                "trough_speed_mps": 0.5,
                "hold_duration_s": 6.0,
                "emergency_gap_m": 6.0,
                "follow_gap_m": 9.0,
                "resume_gap_m": 12.0,
                "control_period_s": 0.2,
                "live_log_period_s": 0.5,
                "ego_gap_radius_m": 6.0,
                "entry_settle_s": 1.0,
            },
        )
        self.assertEqual(highway_cfg['reference_safety_distance_m'], 5.0)
        self.assertEqual(highway_cfg['reference_ttc_sec'], 1.0)
        self.assertEqual(scenario_cfg['highway_tunnel_test_start_path_index'], 2275)

        expected = {
            10: (5.0, 60.0, 'A2256W000029', 167, 'A2256W000432', 246),
            6: (5.0, 40.0, 'A2256W000406', 78, 'A2256W001105', 83),
            8: (5.0, 50.0, 'A2256W000175', 78, 'A2256W000422', 59),
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
            self.assertEqual(spec['maximum_spawn_vehicle'], 4)
            self.assertTrue(spec['pause'])
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
        self.assertEqual(param.config.maximum_spawn_vehicle, 4)

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

    def test_strict_pedestrian_inventory_never_treats_query_failure_as_empty(self):
        global_cfg, _, _ = load_configuration()

        class BrokenWorld:
            def get_all_actors_state(self, **_kwargs):
                raise RuntimeError("MORAI actor query unavailable")

        client = MoraiGrpcClient(global_cfg)
        with self.assertRaisesRegex(RuntimeError, "before world start"):
            client.get_all_pedestrian_states(strict=True)

        client.world = BrokenWorld()
        self.assertEqual(client.get_all_pedestrian_states(), [])
        with self.assertRaisesRegex(RuntimeError, "pedestrian actor inventory"):
            client.get_all_pedestrian_states(strict=True)

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
        scenario = self._make_crossing_scenario(None)
        event = scenario.dynamic_obstacle_events[0]
        event["phase"] = "armed"
        event["actor"] = None
        scenario.dynamic_obstacle = None
        scenario.dynamic_obstacle_models = ["Man1"]
        scenario.profile_index = 0
        scenario.scenario_actors = []

        class Actor:
            def __init__(self):
                self.pause_calls = []

            def set_pause(self, value):
                self.pause_calls.append(value)
                return True

        actor = Actor()
        actor.get_id = lambda: "test-dynamic-owned"

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
        scenario.cfg = {"dynamic_obstacle": {"require_ego_response": False}}
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
        scenario.grpc = SimpleNamespace(get_all_pedestrian_states=unittest.mock.Mock(return_value=[]))
        scenario.dynamic_obstacle_owned_ids = set()
        scenario.dynamic_obstacle_event_index = 0
        scenario.dynamic_obstacle_events = [{
            "id": "legacy_right_to_left",
            "sequence": 1,
            "phase": "crossing",
            "actor": actor,
            "actor_id": "test-dynamic-owned",
            "model": "Man1",
            "allowed_models": ["Man1"],
            "allowed_speeds_mps": [2.5],
            "speed_mps": 2.5,
            "start": dict(scenario.dynamic_obstacle_start),
            "end": dict(scenario.dynamic_obstacle_end),
            "conflict_point": {"x": -60.28, "y": -11.69, "z": 28.4},
            "route_path_index": 936,
            "route_s": 90.0,
            "path_length_m": 13.8337,
            "start_to_conflict_m": 3.5,
            "yaw_deg": -178.7,
            "trigger_distance_m": 25.0,
            "arrival_tolerance_m": 1.5,
            "max_crossing_sec": 30.0,
            "trigger": {"ego_speed_mps": 5.0},
            "response_observed": True,
            "response_kind": "stop",
            "resumed_after_clear": False,
            "min_ego_distance_m": float("inf"),
            "min_ego_speed_mps": float("inf"),
            "max_lateral_error_m": 0.0,
            "failure_reason": None,
            "completion_reason": None,
            "started_scenario_time": started_at,
            "last_print_sec": -1,
        }]
        scenario.profile_state_path = lambda: "/path/that/does/not/exist"
        return scenario

    def test_dynamic_obstacle_stops_at_endpoint_without_pause_reset(self):
        class Actor:
            def __init__(self):
                self.pause_calls = []

            def get_id(self):
                return "test-dynamic-owned"

            def set_pause(self, value):
                self.pause_calls.append(value)
                return True

        actor = Actor()
        scenario = self._make_crossing_scenario(actor)
        scenario._scenario_now = lambda: 105.5

        class FakeGrpc:
            def __init__(self):
                self.control_args = None
                self.control_kwargs = None

            def control_pedestrian(self, *args, **kwargs):
                self.control_args = args
                self.control_kwargs = kwargs
                return True

        scenario.grpc = FakeGrpc()
        scenario._finish_dynamic_obstacle("reached_dest")

        controlled_actor, direction_x, direction_y, speed = (
            scenario.grpc.control_args
        )
        self.assertIs(controlled_actor, actor)
        self.assertLess(direction_x, 0.0)
        self.assertLess(direction_y, 0.0)
        self.assertAlmostEqual(speed, 0.0)
        self.assertEqual(scenario.grpc.control_kwargs, {"verbose": True})
        self.assertEqual(actor.pause_calls, [])
        self.assertEqual(scenario.dynamic_obstacle_phase, "cleared")

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
        scenario._scenario_now = lambda: 103.0
        with self.assertRaisesRegex(RuntimeError, "did not leave spawn"):
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
        scenario._scenario_now = lambda: 100.5
        with self.assertRaisesRegex(RuntimeError, "left approved crossing"):
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
