"""Offline integration tests: real routing/events with a mocked MORAI boundary."""
import copy
import io
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from .. import main_runner
from ..main_runner import load_configuration, build_parser
from ..scenarios.full_loop_scenario import FullLoopScenario
from ..scenarios.segment_modes.scenario import MissionScenario
from ..scenarios.segment_modes.modes import (
    MISSION_MODES, PRIMARY_SCENARIOS, automatic_repeat_enabled,
    get_scenario_zones,
)
from ..scenarios.full_loop_parts import (
    dynamic_obstacle, highway, roundabout, static_obstacle, tunnel,
)
from ..scenarios.segment_modes import events as mission_events
from ..scenarios.segment_modes import route as mission_route
from ..utils.map_loader import MGeoMapLoader
from ..utils.transform_utils import polyline_length


def transform(x, y, z, yaw):
    return SimpleNamespace(
        location=SimpleNamespace(x=x, y=y, z=z),
        rotation=SimpleNamespace(x=0.0, y=0.0, z=yaw),
    )


def vehicle(**kwargs):
    actor = Mock()
    actor.get_id.return_value = kwargs.get("label", "actor")
    return actor


def spawnpoint(spec):
    return "native-" + str(spec["id"])


def make_grpc(cfg):
    grpc = Mock()
    grpc.client_key = "mission-mode-test"
    grpc.make_transform.side_effect = transform
    grpc.reset_actors.return_value = True
    grpc.get_available_obstacle_models.return_value = cfg["static_obstacle_models"]
    grpc.get_available_pedestrian_models.return_value = cfg["dynamic_obstacle_models"]
    grpc.get_available_surround_vehicle_models.return_value = ["DEFAULT_SEDAN"]
    grpc.destroy_obstacles_near.return_value = []
    grpc.obstacle_ids_near.side_effect = [[], ["static-actor"]]
    grpc.spawn_obstacle.return_value.get_id.return_value = "static-actor"
    grpc.spawn_vehicle.side_effect = vehicle
    grpc.set_vehicle_route.return_value = True
    grpc.set_weather.return_value = True
    grpc.set_time.return_value = True
    grpc.create_vehicle_spawn_point.side_effect = spawnpoint
    grpc.enable_spawn_point.return_value = True
    return grpc


class MissionModesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()):
            cls.global_cfg, cls.zone_cfg, cls.base_cfg = load_configuration()
        cls.map_loader = MGeoMapLoader(cls.global_cfg["paths"]["mgeo_root"])
        # Load installed protobuf definitions exactly as the CLI constructor does.
        # No connect/start/ROS call is made by this SDK initialization.
        main_runner.MoraiGrpcClient(cls.global_cfg)
        cls.points = FullLoopScenario(
            None, cls.map_loader, cls.global_cfg, cls.base_cfg,
        ).load_global_path()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def make_scenario(self, name, seed=7):
        global_cfg = copy.deepcopy(self.global_cfg)
        global_cfg["paths"]["runtime_root"] = self.tmp.name
        cfg = copy.deepcopy(self.base_cfg)
        cfg["mission_mode"] = name
        cfg["profile_state_path"] = str(Path(self.tmp.name) / "base.json")
        cfg["route_debug_enabled"] = False
        cfg["random_seed"] = seed
        scenario = MissionScenario(
            make_grpc(cfg), self.map_loader, global_cfg, cfg,
        )
        scenario.configure_drive_with_retries = Mock()
        return scenario

    def setup_scenario(self, scenario):
        position = scenario.load_route_definition()["zones"]["static_obstacle"]["position"]
        state = SimpleNamespace(
            actor_info=SimpleNamespace(id=SimpleNamespace(value="static-actor")),
            transform=transform(position["x"], position["y"], position["z"], 0.0),
        )
        scenario.grpc.world._sim_adapter._actor_stub.GetAllActorsState.return_value = (
            SimpleNamespace(states=[state])
        )
        scenario.grpc.get_all_obstacle_states.return_value = [state]
        with redirect_stdout(io.StringIO()), patch.object(highway.time, "sleep"):
            scenario.setup()

    def test_static_disappearance_fails_exact_state_verification(self):
        scenario = self.make_scenario("full_loop_static")
        self.setup_scenario(scenario)
        scenario.grpc.get_all_obstacle_states.return_value = []
        with patch(
            "scenario_runner.scenarios.full_loop_parts.static_obstacle.time.sleep"
        ), self.assertRaisesRegex(RuntimeError, "disappeared"):
            scenario.verify_static_obstacle(1.0, 0.0)

    def test_six_primary_modes_plus_legacy_commands(self):
        self.assertEqual(len(PRIMARY_SCENARIOS), 6)
        for name in PRIMARY_SCENARIOS:
            self.assertEqual(build_parser().parse_args(["--scenario", name]).scenario, name)
            self.assertTrue(automatic_repeat_enabled(name, self.base_cfg))
            self.assertFalse(automatic_repeat_enabled(name, self.base_cfg, once=True))
        self.assertFalse(automatic_repeat_enabled("full_loop_highway_tunnel_test", self.base_cfg))
        self.assertTrue(automatic_repeat_enabled(
            "full_loop_highway_tunnel_test", self.base_cfg, highway_audit=True,
        ))
        self.assertIs(main_runner.SUPPORTED_SCENARIOS["full_loop"], FullLoopScenario)

    def test_collection_zone_contract_comes_from_primary_modes(self):
        self.assertEqual(
            asdict(get_scenario_zones("full_loop")),
            {
                "static": True,
                "dynamic": True,
                "roundabout": True,
                "highway": True,
                "tunnel": True,
            },
        )
        for name, mode in MISSION_MODES.items():
            self.assertEqual(get_scenario_zones(name), mode.zones)

    def test_mode_descriptions_are_immutable_and_single_zone(self):
        for name, mode in MISSION_MODES.items():
            count = sum(asdict(mode.zones).values())
            self.assertEqual(count, 0 if name == "full_loop_normal" else 1)
            self.assertFalse(mode.zones.tunnel)
            with self.assertRaises(FrozenInstanceError):
                mode.name = "changed"

    def test_approved_coordinates_and_source_slice_are_exact(self):
        approved = {
            "full_loop_static": (
                (483, (-79.9685367429629, -230.3942881319672, 28.401036010337744),
                 61.097480542846796, "A2256W000213"),
                (753, (-59.777648542643476, -101.53081497592018, 28.364247184277144),
                 90.33835303338373, "A2256W000202"),
            ),
            "full_loop_dynamic": (
                (753, (-59.777648542643476, -101.53081497592018, 28.364247184277144),
                 90.33835303338373, "A2256W000202"),
                (1081, (-60.78981215174727, 58.50851457093197, 28.36528072840477),
                 90.58330256087204, "A2256W000315"),
            ),
            "full_loop_roundabout": (
                (1668, (-99.27731127804267, 252.81623345968816, 28.359828711565655),
                 90.28600888072691, "A2256W000304"),
                (1975, (-25.935784360231036, 345.127016285181, 28.33574698713155),
                 3.4004013012027805, "A2256W000054"),
            ),
            "full_loop_highway": (
                (1975, (-25.935784360231036, 345.127016285181, 28.33574698713155),
                 3.4004013012027805, "A2256W000054"),
                (3638, (70.15920404473096, -419.0559893071876, 28.11111997351344),
                 -96.58181855513165, "A2256W000446"),
            ),
        }
        for name, (approved_start, approved_end) in approved.items():
            scenario = self.make_scenario(name)
            mission_events.prepare_mission(scenario)
            with redirect_stdout(io.StringIO()):
                scenario.route_definition = scenario.load_route_definition()
                scenario.prepare_full_loop_route()
            mode = MISSION_MODES[name]
            self.assertEqual(
                (mode.start.index, mode.start.xyz, mode.start.yaw_deg, mode.start.link_id),
                approved_start,
            )
            self.assertEqual(
                (mode.end.index, mode.end.xyz, mode.end.yaw_deg, mode.end.link_id),
                approved_end,
            )
            self.assertEqual(scenario.route_points, self.points[mode.start.index:mode.end.index + 1])
            self.assertEqual(scenario.start_tf.rotation.z, mode.start.yaw_deg)
            self.assertEqual((scenario.goal_x, scenario.goal_y, scenario.goal_z), mode.end.xyz)
            self.assertFalse(scenario.cfg["expert_is_closed_path"])
            self.assertEqual(scenario.start_link, mode.start.link_id)
            self.assertEqual(scenario.end_link, mode.end.link_id)
            self.assertAlmostEqual(scenario.path_s_at_index(mode.start.index), 0.0)
            self.assertAlmostEqual(scenario.path_s_at_index(mode.end.index), scenario.route_length_m)

    def test_bounded_missions_export_expert_only_lookahead_tail(self):
        for name, mode in MISSION_MODES.items():
            scenario = self.make_scenario(name)
            mission_events.prepare_mission(scenario)
            with redirect_stdout(io.StringIO()):
                scenario.route_definition = scenario.load_route_definition()
                scenario.prepare_full_loop_route()

            mission_points = list(scenario.route_points)
            expert_points = mission_route.build_expert_route_points(scenario)
            self.assertEqual(scenario.route_points, mission_points)
            if mode.start is None:
                self.assertIs(expert_points, scenario.route_points)
                continue

            self.assertEqual(expert_points[:len(mission_points)], mission_points)
            self.assertGreater(len(expert_points), len(mission_points))
            self.assertGreaterEqual(
                polyline_length(expert_points[len(mission_points) - 1:]),
                mission_route.EXPERT_LOOKAHEAD_TAIL_M,
            )
            self.assertEqual(
                expert_points[len(mission_points)],
                self.points[mode.end.index + 1],
            )

    def test_route_change_fails_before_any_world_mutation(self):
        scenario = self.make_scenario("full_loop_static")
        corrupted = list(self.points)
        corrupted[483] = (0.0, 0.0, 0.0)
        scenario.load_global_path = Mock(return_value=corrupted)
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            scenario.setup()
        scenario.grpc.start_world.assert_not_called()
        scenario.grpc.spawn_vehicle.assert_not_called()

    def test_each_setup_spawns_only_selected_zone(self):
        for name, mode in MISSION_MODES.items():
            with self.subTest(name=name):
                scenario = self.make_scenario(name)
                self.setup_scenario(scenario)
                grpc = scenario.grpc
                self.assertEqual(grpc.spawn_obstacle.call_count, int(mode.zones.static))
                self.assertEqual(grpc.spawn_vehicle.call_count, 0)
                self.assertEqual(
                    scenario.roundabout_lifecycle_state,
                    "armed" if mode.zones.roundabout else "disabled",
                )
                self.assertEqual(grpc.create_vehicle_spawn_point.call_count, 3 if mode.zones.highway else 0)
                grpc.spawn_pedestrian.assert_not_called()  # Dynamic spawn is approach-triggered.
                self.assertEqual(scenario.dynamic_obstacle_phase, "waiting" if mode.zones.dynamic else "disabled")
                self.assertEqual(scenario.tunnel_event_phase, "disabled")
                self.assertIn(scenario.scenario_hour, [11, 13, 15])
                scenario.configure_drive_with_retries.assert_called_once()
                grpc.reset_actors.assert_called_once()
                self.assertEqual(scenario.cfg["expert_traffic_light_control"], False)

    def test_normal_has_full_path_and_no_model_dependency_or_late_spawn(self):
        scenario = self.make_scenario("full_loop_normal")
        self.setup_scenario(scenario)
        self.assertEqual(scenario.route_points, self.points)
        scenario.grpc.get_available_obstacle_models.assert_not_called()
        scenario.grpc.get_available_pedestrian_models.assert_not_called()
        scenario.grpc.get_available_surround_vehicle_models.assert_not_called()
        scenario.grpc.get_time.return_value = scenario.scenario_hour
        for s in [0.0, 300.0, 460.0, 880.0, 1100.0, 1850.0, 2100.0]:
            scenario.on_expert_timeline_tick(0.0, {}, s, 2184.6 - s)
        scenario.grpc.spawn_vehicle.assert_not_called()
        scenario.grpc.spawn_obstacle.assert_not_called()
        scenario.grpc.spawn_pedestrian.assert_not_called()
        scenario.grpc.create_vehicle_spawn_point.assert_not_called()

    def test_static_location_models_and_singleton_recovery_call_are_preserved(self):
        scenario = self.make_scenario("full_loop_static")
        self.setup_scenario(scenario)
        self.assertIn(scenario.static_obstacle_model, self.base_cfg["static_obstacle_models"])
        self.assertNotIn(scenario.static_obstacle_model, ["Tree01", "Tree02", "Tree03"])
        call = scenario.grpc.spawn_obstacle.call_args.kwargs
        expected = scenario.route_definition["zones"]["static_obstacle"]["position"]
        self.assertEqual(vars(call["transform"].location), expected)
        self.assertEqual(scenario.grpc.obstacle_ids_near.call_count, 2)
        scenario.grpc.destroy_obstacles_near.assert_called_once()
        scenario.static_obstacle.set_physics.assert_called_once_with(False)
        scenario.static_obstacle.set_transform.assert_called_once()
        self.assertEqual(scenario.static_obstacle_expected["actor_id"], "static-actor")

    def test_static_spawn_lock_repairs_a_model_that_falls_before_physics_lock(self):
        scenario = SimpleNamespace(grpc=Mock())
        actor = Mock()
        actor.set_physics.return_value = True
        actor.set_transform.return_value = True
        expected = transform(-60.5, -135.0, 28.4, -83.0)
        dropped = SimpleNamespace(
            actor_info=SimpleNamespace(id=SimpleNamespace(value="static-2")),
            transform=transform(-60.5, -135.0, -12.8, -83.0),
        )
        stable = SimpleNamespace(
            actor_info=SimpleNamespace(id=SimpleNamespace(value="static-2")),
            transform=transform(-60.5, -135.0, 28.4, -83.0),
        )
        scenario.grpc.get_all_obstacle_states.side_effect = [
            [dropped], [stable], [stable],
        ]

        with patch.object(static_obstacle.time, "sleep"):
            state, diagnostics = static_obstacle.lock_static_obstacle_transform(
                scenario,
                actor,
                "static-2",
                expected,
                (-60.5, -135.0, 28.4),
            )

        self.assertIs(state, stable)
        self.assertEqual(actor.set_physics.call_count, 2)
        self.assertEqual(actor.set_transform.call_count, 2)
        self.assertEqual([row["state"] for row in diagnostics], [
            "moved", "stable", "stable",
        ])

    def test_static_spawn_lock_rejects_a_persistently_falling_model(self):
        scenario = SimpleNamespace(grpc=Mock())
        actor = Mock()
        actor.set_physics.return_value = True
        actor.set_transform.return_value = True
        expected = transform(-60.5, -135.0, 28.4, -83.0)
        dropped = SimpleNamespace(
            actor_info=SimpleNamespace(id=SimpleNamespace(value="static-2")),
            transform=transform(-60.5, -135.0, -12.8, -83.0),
        )
        scenario.grpc.get_all_obstacle_states.return_value = [dropped]

        with patch.object(static_obstacle.time, "sleep"):
            state, diagnostics = static_obstacle.lock_static_obstacle_transform(
                scenario,
                actor,
                "static-2",
                expected,
                (-60.5, -135.0, 28.4),
                attempts=3,
            )

        self.assertIsNone(state)
        self.assertEqual(actor.set_physics.call_count, 3)
        self.assertEqual(actor.set_transform.call_count, 3)
        self.assertEqual([row["state"] for row in diagnostics], [
            "moved", "moved", "moved",
        ])

    def test_static_editor_restores_fallen_height_and_preserves_manual_pose(self):
        actor = Mock()
        actor.set_physics.return_value = True
        actor.set_transform.return_value = True
        dropped = SimpleNamespace(
            actor_info=SimpleNamespace(id=SimpleNamespace(value="owned-1")),
            transform=transform(-61.2, -140.7, -25000.0, -71.5),
        )
        repaired = SimpleNamespace(
            actor_info=SimpleNamespace(id=SimpleNamespace(value="owned-1")),
            transform=transform(-61.2, -140.7, 28.4, -71.5),
        )
        foreign = SimpleNamespace(
            actor_info=SimpleNamespace(id=SimpleNamespace(value="foreign")),
            transform=transform(-55.0, -140.0, 28.4, 0.0),
        )
        grpc = Mock()
        grpc.make_transform.side_effect = transform
        grpc.get_all_obstacle_states.side_effect = [
            [dropped, foreign], [dropped, foreign],
            [repaired, foreign], [repaired, foreign],
        ]
        scenario = SimpleNamespace(
            cfg={"static_obstacle_editor_mode": True},
            grpc=grpc,
            static_obstacles=[actor],
            static_obstacle_expectations=[{
                "actor_id": "owned-1", "z": 28.4, "model": "CargoBox",
            }],
        )

        with redirect_stdout(io.StringIO()):
            static_obstacle.maintain_static_obstacles_for_editor(scenario)

        self.assertEqual(actor.set_physics.call_count, 2)
        self.assertEqual(actor.set_transform.call_count, 2)
        restored = actor.set_transform.call_args.args[0]
        self.assertEqual(
            (restored.location.x, restored.location.y, restored.location.z),
            (-61.2, -140.7, 28.4),
        )
        self.assertEqual(restored.rotation.z, -71.5)

    def test_static_editor_healthy_height_only_refreshes_physics_lock(self):
        actor = Mock()
        actor.set_physics.return_value = True
        healthy = SimpleNamespace(
            actor_info=SimpleNamespace(id=SimpleNamespace(value="owned-1")),
            transform=transform(-60.5, -140.0, 28.4, -83.0),
        )
        grpc = Mock()
        grpc.get_all_obstacle_states.return_value = [healthy]
        scenario = SimpleNamespace(
            cfg={"static_obstacle_editor_mode": True},
            grpc=grpc,
            static_obstacles=[actor],
            static_obstacle_expectations=[{
                "actor_id": "owned-1", "z": 28.4, "model": "CargoBox",
            }],
        )

        static_obstacle.maintain_static_obstacles_for_editor(scenario)

        actor.set_physics.assert_called_once_with(False)
        actor.set_transform.assert_not_called()

    def test_static_two_object_test_layout_spawns_and_tracks_both(self):
        scenario = self.make_scenario("full_loop_static")
        scenario.cfg["static_obstacle_count"] = 2
        scenario.cfg["static_obstacle_multi_randomization"] = {
            "enabled": False,
        }
        zone = scenario.load_route_definition()["zones"]["static_obstacle"]
        placements = zone["multi_positions"]

        actors = []
        states = []
        for index, placement in enumerate(placements, 1):
            actor_id = f"static-{index}"
            actor = Mock()
            actor.get_id.return_value = actor_id
            actor.set_physics.return_value = True
            actors.append(actor)
            position = placement["position"]
            states.append(SimpleNamespace(
                actor_info=SimpleNamespace(id=SimpleNamespace(value=actor_id)),
                transform=transform(
                    position["x"], position["y"], position["z"], 0.0
                ),
            ))

        scenario.grpc.spawn_obstacle.side_effect = actors
        scenario.grpc.obstacle_ids_near.side_effect = [
            [], ["static-1"], [], [], ["static-2"],
        ]
        scenario.grpc.get_all_obstacle_states.return_value = states
        with redirect_stdout(io.StringIO()), patch.object(highway.time, "sleep"):
            scenario.setup()

        self.assertEqual(scenario.grpc.spawn_obstacle.call_count, 2)
        self.assertEqual(len(scenario.static_obstacles), 2)
        self.assertEqual(
            [item["actor_id"] for item in scenario.static_obstacle_expectations],
            ["static-1", "static-2"],
        )
        self.assertEqual(len(scenario.static_obstacle_models_selected), 2)
        self.assertNotEqual(
            scenario.static_obstacle_models_selected[0],
            scenario.static_obstacle_models_selected[1],
        )
        self.assertEqual(scenario.scenario_actors[-2:], actors)

    def test_static_pair_randomization_is_bounded_and_reproducible(self):
        first = self.make_scenario("full_loop_static", seed=123)
        second = self.make_scenario("full_loop_static", seed=123)
        for scenario in (first, second):
            scenario.random = random.Random(123)
        spec = first.load_route_definition()["zones"]["static_obstacle"]

        a = static_obstacle.select_static_obstacle_placements(first, spec, 2)
        b = static_obstacle.select_static_obstacle_placements(second, spec, 2)
        self.assertEqual(a, b)
        bases = spec["multi_positions"]
        for sampled, base in zip(a, bases):
            self.assertLessEqual(
                abs(sampled["position"]["x"] - base["position"]["x"]),
                0.35,
            )
            self.assertLessEqual(
                abs(sampled["position"]["y"] - base["position"]["y"]),
                2.5,
            )
            self.assertLessEqual(
                abs(sampled["yaw_deg"] - base["yaw_deg"]), 3.0
            )
        spacing = a[1]["position"]["y"] - a[0]["position"]["y"]
        self.assertGreaterEqual(spacing, 9.0)
        self.assertLessEqual(spacing, 11.0)

    def test_static_editor_preset_replays_exact_models_positions_and_scales(self):
        scenario = self.make_scenario("full_loop_static")
        preset = Path(self.tmp.name) / "replay.yaml"
        placements = [
            ("CargoBox", -60.61, -145.0, [2.0, 3.0, 2.0]),
            # Deliberately within the 3 m stale-cleanup radius. Replaying the
            # second item must preserve the first run-owned obstacle.
            ("WoodBox", -60.45, -143.0, [1.5, 2.0, 1.5]),
        ]
        preset.write_text(yaml.safe_dump({
            "schema": "morai_static_obstacle_preset_v1",
            "name": "replay",
            "map_name": "R_KR_PR_K-city_2025",
            "scenario": "full_loop_static",
            "obstacles": [
                {
                    "model": model,
                    "position": {"x": x, "y": y, "z": 28.38},
                    "yaw_deg": -83.0,
                    "scale": scale,
                }
                for model, x, y, scale in placements
            ],
        }), encoding="utf-8")
        scenario.cfg["static_obstacle_preset"] = str(preset)

        actors, states = [], []
        for index, (_, x, y, _) in enumerate(placements, 1):
            actor = Mock()
            actor.get_id.return_value = "preset-{}".format(index)
            actor.set_physics.return_value = True
            actors.append(actor)
            states.append(SimpleNamespace(
                actor_info=SimpleNamespace(
                    id=SimpleNamespace(value="preset-{}".format(index))
                ),
                transform=transform(x, y, 28.38, -83.0),
            ))
        scenario.grpc.spawn_obstacle.side_effect = actors
        scenario.grpc.obstacle_ids_near.side_effect = [
            [], ["preset-1"], ["preset-1"], ["preset-1"],
            ["preset-1", "preset-2"],
        ]
        scenario.grpc.get_all_obstacle_states.return_value = states
        with redirect_stdout(io.StringIO()), patch.object(highway.time, "sleep"):
            scenario.setup()

        calls = scenario.grpc.spawn_obstacle.call_args_list
        self.assertEqual([call.kwargs["model_name"] for call in calls], ["CargoBox", "WoodBox"])
        self.assertEqual(vars(calls[0].kwargs["transform"].location), {
            "x": -60.61, "y": -145.0, "z": 28.38,
        })
        self.assertEqual(
            [calls[1].kwargs["scale"].x, calls[1].kwargs["scale"].y, calls[1].kwargs["scale"].z],
            [1.5, 2.0, 1.5],
        )
        self.assertEqual(scenario.grpc.destroy_obstacles_near.call_count, 1)

    def test_roundabout_three_equal_positions_and_existing_controls(self):
        scenario = self.make_scenario("full_loop_roundabout")
        self.setup_scenario(scenario)
        self.assertEqual(scenario.roundabout_lifecycle_state, "armed")
        scenario.grpc.spawn_vehicle.assert_not_called()

        zone = scenario.route_definition["zones"]["roundabout"]
        entry = next(
            checkpoint
            for checkpoint in scenario.route_definition["checkpoints"]
            if str(checkpoint["id"]) == str(zone["ego_entry_checkpoint"])
        )
        trigger_s = scenario.path_s_at_index(entry["path_index"]) - 100.0
        roundabout.update_roundabout_npcs(
            scenario,
            elapsed=1.0,
            ego_state={"x": 0.0, "y": 0.0},
            current_s=trigger_s,
        )
        self.assertEqual(len(scenario.roundabout_npcs), 3)
        self.assertEqual(scenario.roundabout_lifecycle_state, "spawned")
        for call in scenario.grpc.spawn_vehicle.call_args_list:
            self.assertEqual(call.kwargs["velocity"], 4.0)
            self.assertEqual(call.kwargs["model_name"], "DEFAULT_SEDAN")
        self.assertEqual(scenario.grpc.set_vehicle_route.call_count, 3)
        self.assertEqual(scenario.grpc.set_vehicle_ai.call_count, 3)
        self.assertEqual(scenario.grpc.set_vehicle_speed_limit.call_count, 3)
        self.assertEqual(scenario.grpc.set_vehicle_velocity.call_count, 3)

        original = self.make_scenario("full_loop_roundabout")
        original.route_definition = scenario.route_definition
        original.roundabout_npcs, original.scenario_actors = [], []
        with redirect_stdout(io.StringIO()):
            roundabout.spawn_roundabout_npcs_now(original)
        actual = scenario.grpc.spawn_vehicle.call_args_list
        baseline = original.grpc.spawn_vehicle.call_args_list
        self.assertEqual(
            [vars(call.kwargs["transform"].location) for call in actual],
            [vars(call.kwargs["transform"].location) for call in baseline],
        )
        self.assertEqual(
            [call.args[1] for call in scenario.grpc.set_vehicle_route.call_args_list],
            [call.args[1] for call in original.grpc.set_vehicle_route.call_args_list],
        )

    def test_roundabout_retires_owned_actors_30m_after_exit(self):
        scenario = self.make_scenario("full_loop_roundabout")
        self.setup_scenario(scenario)
        zone = scenario.route_definition["zones"]["roundabout"]
        checkpoints = {
            str(item["id"]): item
            for item in scenario.route_definition["checkpoints"]
        }
        entry_s = scenario.path_s_at_index(
            checkpoints[str(zone["ego_entry_checkpoint"])]["path_index"]
        )
        exit_s = scenario.path_s_at_index(
            checkpoints[str(zone["ego_exit_checkpoint"])]["path_index"]
        )
        roundabout.update_roundabout_npcs(
            scenario, 1.0, {"x": 0.0, "y": 0.0}, entry_s - 100.0
        )
        scenario.grpc.get_all_vehicle_states.return_value = [
            SimpleNamespace(
                actor_info=SimpleNamespace(
                    id=SimpleNamespace(value=item["actor"].get_id())
                ),
                transform=transform(1.0, 1.0, 0.0, 0.0),
            )
            for item in scenario.roundabout_npcs
        ]
        scenario.grpc.destroy_vehicle_by_id.return_value = True
        roundabout.update_roundabout_npcs(
            scenario, 2.0, {"x": 0.0, "y": 0.0}, exit_s + 30.01
        )
        self.assertEqual(scenario.roundabout_lifecycle_state, "retired")
        self.assertEqual(scenario.grpc.destroy_vehicle_by_id.call_count, 3)
        self.assertEqual(scenario.roundabout_npcs, [])

    def test_roundabout_stall_retires_only_owned_three_actors(self):
        scenario = self.make_scenario("full_loop_roundabout")
        scenario.event = Mock()
        self.setup_scenario(scenario)
        zone = scenario.route_definition["zones"]["roundabout"]
        checkpoints = {
            str(item["id"]): item
            for item in scenario.route_definition["checkpoints"]
        }
        entry_s = scenario.path_s_at_index(
            checkpoints[str(zone["ego_entry_checkpoint"])]["path_index"]
        )
        trigger_s = entry_s - 100.0
        roundabout.update_roundabout_npcs(
            scenario, 1.0, {"x": 0.0, "y": 0.0}, trigger_s
        )
        scenario.grpc.get_all_vehicle_states.return_value = [
            SimpleNamespace(
                actor_info=SimpleNamespace(
                    id=SimpleNamespace(value=item["actor"].get_id())
                ),
                transform=transform(1.0, 1.0, 0.0, 0.0),
            )
            for item in scenario.roundabout_npcs
        ]
        scenario.grpc.destroy_vehicle_by_id.return_value = True
        roundabout.update_roundabout_npcs(
            scenario, 46.0, {"x": 0.0, "y": 0.0}, trigger_s
        )
        self.assertEqual(scenario.roundabout_lifecycle_state, "retired")
        self.assertEqual(scenario.grpc.destroy_vehicle_by_id.call_count, 3)
        calls = [call for call in scenario.event.call_args_list]
        named = [(call.args[0], call.kwargs) for call in calls]
        self.assertTrue(any(
            name == "actor_lifecycle_finished"
            and fields.get("retire_reason") == "progress_stall"
            and fields.get("success") is True
            for name, fields in named
        ))
        for name, fields in named:
            if name in ("actor_lifecycle_started", "actor_lifecycle_finished"):
                self.assertNotIn("training_visible", fields)
            if name == "roundabout_spawn_contract":
                self.assertIs(fields.get("training_visible"), False)

    def test_dynamic_arm_coordinates_direction_trigger_unchanged(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        spec = scenario.route_definition["zones"]["dynamic_obstacle"]
        self.assertEqual(scenario.dynamic_obstacle_start, spec["start"])
        self.assertEqual(scenario.dynamic_obstacle_end, spec["end"])
        self.assertEqual(scenario.dynamic_obstacle_trigger_distance_m, spec["trigger_distance_m"])
        self.assertAlmostEqual(scenario.dynamic_obstacle_route_s, 90.0217769485483)
        with patch.object(dynamic_obstacle, "update_dynamic_obstacle") as update:
            scenario.update_dynamic_obstacle({"x": 1}, 70.0)
            update.assert_called_once_with(scenario, {"x": 1}, 70.0)
        # Actual spawn/recovery implementation is inherited, not replaced.
        self.assertIs(MissionScenario._spawn_active_dynamic_obstacle,
                      FullLoopScenario._spawn_active_dynamic_obstacle)

    def test_highway_specs_and_trigger_cleanup_offsets_preserved(self):
        scenario = self.make_scenario("full_loop_highway")
        self.setup_scenario(scenario)
        specs = [call.args[0] for call in scenario.grpc.create_vehicle_spawn_point.call_args_list]
        self.assertEqual(specs, self.base_cfg["highway_npc"]["spawn_points"])
        self.assertAlmostEqual(scenario.highway_activation_s,
                               polyline_length(self.points[:2196]) - polyline_length(self.points[:1976]))
        self.assertAlmostEqual(scenario.highway_exit_s,
                               polyline_length(self.points[:3530]) - polyline_length(self.points[:1976]))
        self.assertIs(MissionScenario._set_highway_spawn_points_enabled,
                      FullLoopScenario._set_highway_spawn_points_enabled)
        scenario.grpc.get_time.return_value = scenario.scenario_hour
        scenario.highway_spawn_point_activation_started = True
        with patch.object(highway, "activate_highway_spawn_points"), patch.object(
            scenario, "deactivate_highway_npcs",
        ) as cleanup:
            scenario.on_expert_timeline_tick(1.0, {}, scenario.highway_exit_s, 53.3)
            cleanup.assert_called_once_with(scenario.highway_exit_s)

    def test_highway_activates_100m_early_and_stops_distant_spawners(self):
        scenario = self.make_scenario("full_loop_highway")
        self.setup_scenario(scenario)
        activation_s = scenario.highway_activation_s - 100.0
        self.assertFalse(highway.activate_highway_spawn_points(scenario, 1.0, activation_s - 0.01))
        scenario.grpc.enable_vehicle_spawn_point.reset_mock()
        scenario.grpc.enable_vehicle_spawn_point.return_value = True
        self.assertTrue(highway.activate_highway_spawn_points(scenario, 1.0, activation_s))
        self.assertEqual(scenario.grpc.enable_vehicle_spawn_point.call_count, 3)
        self.assertTrue(all(item["enabled"] for item in scenario.highway_spawn_points))

        highway_zone = scenario.route_definition["zones"]["highway"]
        checkpoints = {
            str(item["id"]): item
            for item in scenario.route_definition["checkpoints"]
        }
        entry_s = scenario.path_s_at_index(
            checkpoints[str(highway_zone["ego_entry_checkpoint"])]["path_index"]
        )
        scenario.grpc.enable_vehicle_spawn_point.reset_mock()
        highway.update_highway_actor_lifecycle(
            scenario, 2.0, {"x": -10000.0, "y": -10000.0}, entry_s + 220.01
        )
        self.assertTrue(scenario.nearby_highway_stopped)
        self.assertEqual(scenario.grpc.enable_vehicle_spawn_point.call_count, 3)
        self.assertTrue(all(not item["enabled"] for item in scenario.highway_spawn_points))
        self.assertFalse(highway.activate_highway_spawn_points(scenario, 3.0, activation_s + 1.0))

    def test_weather_time_model_reroll_and_history_isolation(self):
        for name in MISSION_MODES:
            a, b = self.make_scenario(name, 1), self.make_scenario(name, 2)
            self.setup_scenario(a)
            self.setup_scenario(b)
            self.assertEqual([a.profile["weather"], b.profile["weather"]], ["SUNNY", "FOGGY"])
            self.assertNotEqual(a.scenario_hour, b.scenario_hour)
            if MISSION_MODES[name].zones.static:
                self.assertNotEqual(a.static_obstacle_model, b.static_obstacle_model)
            if MISSION_MODES[name].zones.dynamic:
                self.assertNotEqual(a.dynamic_obstacle_model, b.dynamic_obstacle_model)
            for item in [a, b]:
                self.assertIn(item.scenario_hour, [11, 13, 15])
                self.assertEqual(item.cfg["profile_state_path"],
                                 str(Path(self.tmp.name) / (name + "_profile.json")))
        self.assertFalse((Path(self.tmp.name) / "base.json").exists())

    def test_preflight_failure_prevents_spawning_and_expert_start(self):
        scenario = self.make_scenario("full_loop_static")
        scenario.grpc.reset_actors.return_value = False
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "preflight"):
            scenario.setup()
        scenario.grpc.spawn_obstacle.assert_not_called()
        scenario.configure_drive_with_retries.assert_not_called()

    def test_failed_static_spawn_retains_original_failure_and_cleanup(self):
        scenario = self.make_scenario("full_loop_static")
        scenario.grpc.spawn_obstacle.return_value = None
        scenario.grpc.obstacle_ids_near.side_effect = None
        scenario.grpc.obstacle_ids_near.return_value = []
        with patch("scenario_runner.core.full_loop_drive_base.SyncRuntimeBridge"):
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "every randomized"):
                scenario.run()
        scenario.configure_drive_with_retries.assert_not_called()
        self.assertGreaterEqual(scenario.grpc.reset_actors.call_count, 2)

    def test_real_goal_check_ends_at_each_selected_end_not_at_start(self):
        for name in MISSION_MODES:
            scenario = self.make_scenario(name)
            self.setup_scenario(scenario)
            controller = Mock()
            controller.is_running.return_value = True
            scenario.expert_controller = controller
            first, end = scenario.route_points[0], scenario.route_points[-1]
            base = dict(yaw_deg=0.0, speed=1.0, current_link=scenario.end_link)
            scenario.grpc.get_ego_motion_state.side_effect = [
                dict(base, x=first[0], y=first[1]), dict(base, x=end[0], y=end[1]),
            ]
            scenario.project_on_route_near_progress = Mock(
                side_effect=[(0.0, 0.0), (scenario.route_length_m, 0.0)],
            )
            scenario.on_expert_timeline_tick = Mock()
            with redirect_stdout(io.StringIO()), patch("scenario_runner.core.full_loop_drive_base.time.sleep"):
                scenario.run_expert_timeline()
            self.assertEqual(scenario.grpc.get_ego_motion_state.call_count, 2)
            controller.stop_process.assert_called_once()

    def test_plain_scenario_verification_fails_on_grpc_collision(self):
        scenario = self.make_scenario("full_loop_static")
        self.setup_scenario(scenario)
        controller = Mock()
        controller.is_running.return_value = True
        scenario.expert_controller = controller
        start = scenario.route_points[0]
        scenario.grpc.get_ego_motion_state.return_value = {
            "x": start[0], "y": start[1], "yaw_deg": 0.0,
            "speed": 0.0, "current_link": scenario.start_link,
            "collision_objects": ["static-actor"],
        }
        scenario.stop_drive_control = Mock()
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(
            RuntimeError, "expert collision"
        ):
            scenario.run_expert_timeline()
        scenario.stop_drive_control.assert_called_once()

    def test_interactive_editor_keeps_running_to_show_collision_and_trace(self):
        scenario = self.make_scenario("full_loop_static")
        self.setup_scenario(scenario)
        scenario.cfg["static_obstacle_editor_mode"] = True
        scenario.cfg["static_editor_max_trials"] = 1
        scenario._static_editor_drive_enabled = True
        scenario.publish_static_editor_event = Mock()
        controller = Mock()
        controller.is_running.return_value = True
        scenario.expert_controller = controller
        first, end = scenario.route_points[0], scenario.route_points[-1]
        scenario.grpc.get_ego_motion_state.side_effect = [
            dict(
                yaw_deg=0.0, speed=1.0,
                current_link=scenario.start_link,
                collision_objects=[], x=first[0], y=first[1],
            ),
            dict(
                yaw_deg=0.0, speed=1.0,
                current_link=scenario.end_link,
                collision_objects=["static-actor"], x=end[0], y=end[1],
            ),
        ]
        scenario.project_on_route_near_progress = Mock(
            side_effect=[(0.0, 0.0), (scenario.route_length_m, 0.0)],
        )
        scenario.on_expert_timeline_tick = Mock()
        scenario.stop_drive_control = Mock()
        with redirect_stdout(io.StringIO()), patch(
            "scenario_runner.core.full_loop_drive_base.time.sleep"
        ):
            scenario.run_expert_timeline()
        scenario.stop_drive_control.assert_not_called()
        scenario.grpc.stop_ego_control.assert_called_once()
        self.assertEqual(scenario.grpc.get_ego_motion_state.call_count, 2)
        finished = [
            call for call in scenario.publish_static_editor_event.call_args_list
            if call.args and call.args[0] == "trial_finished"
        ]
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0].kwargs["outcome"], "failed")
        self.assertEqual(finished[0].kwargs["collisions"], ["static-actor"])

    def test_interactive_editor_resets_only_ego_and_expert_for_next_trial(self):
        scenario = self.make_scenario("full_loop_static")
        self.setup_scenario(scenario)
        scenario.cfg["static_obstacle_editor_mode"] = True
        scenario._static_editor_drive_enabled = False
        scenario.publish_static_editor_event = Mock()
        scenario.configure_drive_with_retries = Mock()
        actor_reset_count = scenario.grpc.reset_actors.call_count

        scenario.finish_static_editor_trial(
            4,
            "passed",
            "goal_reached",
            [],
            reset_for_next=True,
        )

        scenario.grpc.stop_ego_control.assert_called_once()
        scenario.configure_drive_with_retries.assert_called_once()
        self.assertEqual(scenario.grpc.reset_actors.call_count, actor_reset_count)
        self.assertEqual(
            [call.args[0] for call in scenario.publish_static_editor_event.call_args_list],
            ["trial_finished", "trial_resetting", "trial_ready"],
        )

    def test_grpc_ego_state_preserves_collision_actor_ids(self):
        from scenario_runner.adapters.morai_grpc import (
            _morai_get_ego_motion_state,
        )

        state = SimpleNamespace(
            transform=transform(1.0, 2.0, 3.0, 90.0),
            velocity=SimpleNamespace(x=3.0, y=4.0, z=0.0),
            vehicle_state=SimpleNamespace(
                current_link_info=SimpleNamespace(
                    id=SimpleNamespace(value="LINK")
                ),
                front_wheel_angle=1.5,
                collision_objects=["barrier-1", ""],
            ),
        )
        client = SimpleNamespace(
            get_ego=lambda: SimpleNamespace(get_actor_state=lambda: state)
        )
        observed = _morai_get_ego_motion_state(client)
        self.assertEqual(observed["collision_objects"], ["barrier-1"])
        self.assertEqual(observed["speed"], 5.0)

    def test_main_recreates_and_repeats_each_mode_three_times(self):
        for name in MISSION_MODES:
            factory = Mock()
            client = Mock()
            with patch.dict(main_runner.SUPPORTED_SCENARIOS, {name: factory}), patch.object(
                main_runner, "load_configuration",
                return_value=(self.global_cfg, self.zone_cfg, copy.deepcopy(self.base_cfg)),
            ), patch.object(main_runner, "MoraiGrpcClient", return_value=client), patch.object(
                main_runner, "MGeoMapLoader",
            ), patch.object(main_runner.time, "sleep"), patch(
                "sys.argv", ["runner", "--scenario", name, "--max-runs", "3"],
            ), redirect_stdout(io.StringIO()):
                main_runner.main()
            self.assertEqual(factory.call_count, 3)
            self.assertEqual(factory.return_value.run.call_count, 3)
            self.assertEqual(factory.call_args.kwargs["scenario_cfg"]["mission_mode"], name)
            client.stop.assert_called_once()


class StaticEditorPresetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with redirect_stdout(io.StringIO()):
            cls.global_cfg, _, cls.base_cfg = load_configuration()
        cls.map_loader = MGeoMapLoader(cls.global_cfg["paths"]["mgeo_root"])
        cls.points = FullLoopScenario(
            None, cls.map_loader, cls.global_cfg, cls.base_cfg,
        ).load_global_path()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write_preset(self, map_name="R_KR_PR_K-city_2025"):
        preset = Path(self.tmp.name) / "two.yaml"
        preset.write_text(yaml.safe_dump({
            "schema": "morai_static_obstacle_preset_v1",
            "name": "two",
            "map_name": map_name,
            "scenario": "full_loop_static",
            "obstacles": [
                {
                    "model": "CargoBox",
                    "position": {"x": -60.61, "y": -142.18, "z": 28.374},
                    "yaw_deg": -83.2,
                    "scale": [2.0, 3.0, 2.0],
                    "route_path_index": 0,
                },
                {
                    "model": "WoodBox",
                    "position": {"x": -60.61, "y": -136.0, "z": 28.38},
                    "yaw_deg": -82.0,
                    "scale": [2.0, 3.0, 2.0],
                    "route_path_index": 0,
                },
            ],
        }), encoding="utf-8")
        return preset

    def test_recomputes_route_binding_instead_of_trusting_saved_index(self):
        loaded = static_obstacle.load_static_obstacle_preset(
            self.write_preset(), "R_KR_PR_K-city_2025", self.points
        )
        self.assertEqual(loaded["name"], "two")
        self.assertEqual(len(loaded["placements"]), 2)
        self.assertGreater(loaded["placements"][0]["route_path_index"], 600)
        self.assertLess(loaded["placements"][0]["route_distance_m"], 3.0)

    def test_rejects_other_map(self):
        with self.assertRaisesRegex(ValueError, "map mismatch"):
            static_obstacle.load_static_obstacle_preset(
                self.write_preset("OTHER"),
                "R_KR_PR_K-city_2025",
                self.points,
            )

    def test_editor_mode_is_async_and_allows_live_repositioning(self):
        factory = Mock()
        client = Mock()
        with patch.dict(
            main_runner.SUPPORTED_SCENARIOS, {"full_loop_static": factory}
        ), patch.object(
            main_runner,
            "load_configuration",
            return_value=(
                self.global_cfg,
                {"zone": "full_loop"},
                copy.deepcopy(self.base_cfg),
            ),
        ), patch.object(
            main_runner, "MoraiGrpcClient", return_value=client
        ), patch.object(
            main_runner, "MGeoMapLoader"
        ), patch(
            "sys.argv",
            [
                "runner", "--scenario", "full_loop_static", "--once",
                "--static-editor", "--static-obstacle-count", "2",
            ],
        ), redirect_stdout(io.StringIO()):
            main_runner.main()
        cfg = factory.call_args.kwargs["scenario_cfg"]
        self.assertTrue(cfg["static_obstacle_editor_mode"])
        self.assertFalse(cfg["sync_runtime"]["enabled"])
        self.assertEqual(cfg["static_obstacle_count"], 2)
        self.assertEqual(
            cfg["expert_ros_remaps"],
            [
                "/ctrl_cmd:=/static_scenario_editor/ctrl_cmd",
                "_static_editor_mode:=true",
            ],
        )

        scenario = SimpleNamespace(
            cfg={"static_obstacle_editor_mode": True},
            grpc=Mock(),
        )
        static_obstacle.verify_static_obstacle(scenario, 1.0, 1.0)
        scenario.grpc.get_all_obstacle_states.assert_not_called()
