"""Offline integration tests: real routing/events with a mocked MORAI boundary."""
import ast
import copy
import io
import json
import random
import tempfile
import time
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
from ..utils.transform_utils import interpolate_on_polyline, polyline_length


HIGHWAY_MERGE_STATIC_MODELS = [
    "DEFAULT_SEDAN",
    "2014_Kia_K7",
    "2015_Kia_K5",
    "2015_Toyota_Prius(HEV)",
    "2016_Hyundai_Genesis_DH",
    "2016_Hyundai_Ioniq",
    "2016_Hyundai_Sonata",
    "2017_Hyundai_Grandeur",
    "2019_Toyota_Prius",
    "2020_Kia_K5",
    "2020_Kia_Stinger",
    "2020_hyundai_sonata",
    "2021_Hyundai_G80",
]


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
    grpc.get_available_surround_vehicle_models.return_value = list(
        HIGHWAY_MERGE_STATIC_MODELS
    )
    grpc.destroy_obstacles_near.return_value = []
    grpc.obstacle_ids_near.side_effect = [[], ["static-actor"]]
    grpc.spawn_obstacle.return_value.get_id.return_value = "static-actor"
    grpc.spawn_vehicle.side_effect = vehicle
    grpc.set_vehicle_route.return_value = True
    grpc.set_weather.return_value = True
    grpc.set_time.return_value = True
    grpc.get_traffic_light_state_by_id.return_value = {
        "id": "C1256W000016",
        "color": 16,
    }
    grpc.set_traffic_light_color_verified.side_effect = (
        lambda light_id, color, **_: {
            "id": str(light_id),
            "target_color": int(color),
            "accepted": True,
            "verified": True,
            "before": {"id": str(light_id), "color": int(color)},
            "after": {"id": str(light_id), "color": int(color)},
            "error": None,
        }
    )
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
        # Existing tests exercise the strict approval contract. Report-only
        # standalone behavior is selected explicitly in dedicated tests.
        cfg["dynamic_obstacle"]["assessment_policy"] = "strict"
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
                grpc.spawn_vehicle.assert_not_called()
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

    def test_dynamic_placement_bounds_check_both_endpoints_not_only_conflict(self):
        zone = {"placement_bounds": dict(min_x=-74, max_x=-49.5, min_y=-49,
                                         max_y=16, endpoint_margin_m=1)}
        entry = dict(id="bounds-test", start=dict(x=-71, y=-40),
                     end=dict(x=-51, y=10))
        dynamic_obstacle._validate_placement_bounds(entry, zone)
        for endpoint in ("start", "end"):
            for field, outside in (("x", -74), ("x", -49.5), ("y", -49), ("y", 16)):
                bad = copy.deepcopy(entry)
                bad[endpoint][field] = outside
                with self.subTest(endpoint=endpoint, field=field, outside=outside):
                    with self.assertRaisesRegex(RuntimeError, "outside placement_bounds"):
                        dynamic_obstacle._validate_placement_bounds(bad, zone)
        # Wider driving windows/legacy files are not retroactively narrowed.
        dynamic_obstacle._validate_placement_bounds(bad, {})

    def test_dynamic_placement_bounds_reject_malformed_or_empty_inset(self):
        entry = dict(id="bounds-test", start=dict(x=-71, y=-40), end=dict(x=-51, y=10))
        good = dict(min_x=-74, max_x=-49.5, min_y=-49, max_y=16, endpoint_margin_m=1)
        for bounds in ([], {}, dict(good, min_x=float("nan")),
                       dict(good, max_y=float("inf")), dict(good, min_x=-40),
                       dict(good, endpoint_margin_m=-1), dict(good, endpoint_margin_m=20)):
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(RuntimeError, "placement_bounds"):
                    dynamic_obstacle._validate_placement_bounds(entry, {"placement_bounds": bounds})

    def test_dynamic_all_approved_and_pending_paths_fit_inset_without_changing_legacy_crossing(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        zone = scenario.route_definition["zones"]["dynamic_obstacle"]
        self.assertEqual((zone["start_path_index"], zone["end_path_index"]), (753, 1081))
        catalogue = scenario.dynamic_obstacle_catalog
        self.assertEqual(len(catalogue), 52)
        self.assertEqual(len({dynamic_obstacle._geometry_signature(entry) for entry in catalogue}), 52)
        for entry in catalogue:
            dynamic_obstacle._validate_placement_bounds(entry, zone)
            self.assertLess(entry["start_d_m"] * entry["end_d_m"], 0)
        by_id = {entry["id"]: entry for entry in catalogue}
        self.assertEqual(dynamic_obstacle._geometry_signature(by_id["legacy_right_to_left"]),
                         "a82493720f6ec0460a6236c9b4c4c7794975de4a031c383f21ad68a93384824b")
        self.assertEqual(dynamic_obstacle._geometry_signature(by_id["legacy_left_to_right"]),
                         "c1dfcfd25c017bcff158f7d338260a6ab6c730ac7f2a30282a93241cf8f7daab")
        self.assertAlmostEqual(by_id["general_south_east_to_west_diagonal"]["yaw_deg"], 118.49563861824498)
        self.assertAlmostEqual(by_id["general_north_west_to_east_diagonal"]["yaw_deg"], 23.833328283756764)
        # The conflict remains in the driving window, but a roadside endpoint
        # outside the inset must still be rejected by the real catalogue gate.
        raw = copy.deepcopy(by_id["legacy_right_to_left"])
        raw["start"]["x"] = -44.0
        with self.assertRaisesRegex(RuntimeError, "outside placement_bounds"):
            dynamic_obstacle._validate_catalog_entry(scenario, raw, zone, scenario.cfg["dynamic_obstacle"], set())

    def test_dynamic_signal_guard_targets_only_explicit_or_pre_signal_paths(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        guarded = []
        unguarded = []
        for entry in scenario.dynamic_obstacle_catalog:
            event = copy.deepcopy(entry)
            dynamic_obstacle._initialize_signal_guard(scenario, event)
            target = guarded if event["signal_guard"]["required"] else unguarded
            target.append(event)
        approved_guarded = [event for event in guarded if event["status"] == "approved"]
        front_guarded = [event for event in guarded if event["id"].startswith("front_crosswalk_")]
        self.assertEqual(len(guarded), 28)
        self.assertEqual(len(unguarded), 24)
        self.assertEqual(len(approved_guarded), 20)
        self.assertEqual(len(front_guarded), 6)
        between_guarded = [event for event in guarded if event["id"].startswith("between_crosswalks_")]
        self.assertEqual(len(between_guarded), 4)
        automatic_guarded = [event for event in approved_guarded
                             if event["signal_guard"]["mode"] == "auto"]
        self.assertEqual(len(automatic_guarded), 10)
        self.assertTrue(all("south" in event["id"] for event in automatic_guarded))
        self.assertTrue(all(
            0.0 <= event["signal_guard"]["stopline_route_s"] - event["route_s"] <= 10.0
            for event in automatic_guarded
        ))
        self.assertTrue(all(
            event["signal_guard"]["mode"] == "through_conflict"
            and event["signal_guard"]["release_route_s"] > event["route_s"]
            for event in front_guarded
        ))
        self.assertTrue(all(event["signal_guard"]["phase"] == "not_required"
                            for event in unguarded))

    def test_dynamic_signal_guard_verifies_green_before_spawn_window(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        entry = next(item for item in scenario.dynamic_obstacle_catalog
                     if item["id"] == "general_south_east_to_west_diagonal")
        event = copy.deepcopy(entry)
        event.update(sequence=1, phase="armed", model="Man1", speed_mps=2.5)
        dynamic_obstacle._initialize_signal_guard(scenario, event)
        light_id = event["signal_guard"]["light_id"]
        scenario.grpc.get_traffic_light_state_by_id.return_value = {
            "id": light_id,
            "color": 1,
        }
        scenario.grpc.set_traffic_light_color_verified.side_effect = None
        scenario.grpc.set_traffic_light_color_verified.return_value = {
            "id": light_id,
            "target_color": 16,
            "accepted": True,
            "verified": True,
            "before": {"id": light_id, "color": 1},
            "after": {"id": light_id, "color": 16},
            "error": None,
        }
        with patch.object(dynamic_obstacle.time, "monotonic", return_value=100.0):
            self.assertFalse(dynamic_obstacle._update_signal_guard(
                scenario, event, event["route_s"] - 40.0, crossing=False,
            ))
        scenario.grpc.get_traffic_light_state_by_id.return_value = {
            "id": light_id,
            "color": 16,
        }
        with patch.object(dynamic_obstacle.time, "monotonic", return_value=100.31):
            self.assertTrue(dynamic_obstacle._update_signal_guard(
                scenario, event, event["route_s"] - 40.0, crossing=False,
            ))
        scenario.grpc.set_traffic_light_color_verified.assert_called_once_with(
            light_id, 16, impulse=True, set_sibling=True,
        )
        self.assertTrue(event["signal_guard"]["verified"])
        self.assertTrue(event["signal_guard"]["intervention"])
        self.assertFalse(event["signal_guard"]["confounded"])

    def test_dynamic_signal_guard_unknown_never_spawns_pedestrian(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        entry = next(item for item in scenario.dynamic_obstacle_catalog
                     if item["id"] == "general_south_east_to_west_diagonal")
        event = copy.deepcopy(entry)
        event.update(
            sequence=1,
            phase="armed",
            model="Man1",
            speed_mps=2.5,
            actor=None,
            actor_id=None,
            trigger=None,
            response_observed=False,
            response_kind=None,
            resumed_after_clear=False,
            min_ego_distance_m=float("inf"),
            min_ego_speed_mps=float("inf"),
            max_lateral_error_m=0.0,
            failure_reason=None,
            completion_reason=None,
        )
        dynamic_obstacle._initialize_signal_guard(scenario, event)
        scenario.dynamic_obstacle_events = [event]
        scenario.dynamic_obstacle_event_index = 0
        scenario.grpc.get_traffic_light_state_by_id.return_value = None
        with patch.object(dynamic_obstacle.time, "monotonic", return_value=100.0), \
                patch.object(scenario, "_spawn_active_dynamic_obstacle") as spawn:
            scenario.update_dynamic_obstacle(
                {"x": -60.0, "y": -70.0, "yaw_deg": 90.0, "speed_mps": 5.0},
                event["route_s"] - 40.0,
            )
        spawn.assert_not_called()
        scenario.grpc.set_traffic_light_color_verified.assert_not_called()
        self.assertEqual(event["phase"], "armed")

    def test_dynamic_signal_guard_red_during_crossing_stays_confounded(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        entry = next(item for item in scenario.dynamic_obstacle_catalog
                     if item["id"] == "general_south_east_to_west_diagonal")
        event = copy.deepcopy(entry)
        event.update(sequence=1, phase="crossing", model="Man1", speed_mps=2.5)
        dynamic_obstacle._initialize_signal_guard(scenario, event)
        guard = event["signal_guard"]
        light_id = guard["light_id"]
        guard.update(
            verified=True,
            command_accepted=True,
            command_attempts=1,
            last_command_wall=100.0,
            last_query_wall=100.0,
            last_verified_wall=100.0,
            verified_since_wall=100.0,
            phase="green",
        )
        scenario.grpc.get_traffic_light_state_by_id.return_value = {
            "id": light_id,
            "color": 1,
        }
        scenario.grpc.set_traffic_light_color_verified.side_effect = None
        scenario.grpc.set_traffic_light_color_verified.return_value = {
            "id": light_id,
            "target_color": 16,
            "accepted": False,
            "verified": False,
            "before": {"id": light_id, "color": 1},
            "after": {"id": light_id, "color": 1},
            "error": "not applied",
        }
        with patch.object(dynamic_obstacle.time, "monotonic", return_value=100.6):
            self.assertTrue(dynamic_obstacle._update_signal_guard(
                scenario, event, event["route_s"], crossing=True,
            ))
        self.assertFalse(guard["confounded"])
        with patch.object(dynamic_obstacle.time, "monotonic", return_value=101.7):
            self.assertTrue(dynamic_obstacle._update_signal_guard(
                scenario, event, event["route_s"], crossing=True,
            ))
        self.assertTrue(guard["confounded"])
        self.assertTrue(event["response_signal_limited"])
        self.assertEqual(event["response_speed_samples"], [])
        self.assertEqual(guard["phase"], "confounded_red")
        self.assertEqual(guard["command_attempts"], 3)
        self.assertEqual(guard["red_observations"], 2)

    def test_dynamic_signal_guard_non_red_transition_is_corrected_not_failed(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        entry = next(item for item in scenario.dynamic_obstacle_catalog
                     if item["id"] == "general_south_east_to_west_diagonal")
        event = copy.deepcopy(entry)
        event.update(sequence=1, phase="crossing", model="Man1", speed_mps=2.5)
        dynamic_obstacle._initialize_signal_guard(scenario, event)
        guard = event["signal_guard"]
        light_id = guard["light_id"]
        guard.update(
            verified=True,
            command_accepted=True,
            command_attempts=1,
            last_command_wall=100.0,
            last_query_wall=100.0,
            last_verified_wall=100.0,
            verified_since_wall=100.0,
            phase="green",
        )
        scenario.grpc.get_traffic_light_state_by_id.return_value = {
            "id": light_id,
            "color": 4,
        }
        with patch.object(dynamic_obstacle.time, "monotonic", return_value=102.1):
            self.assertTrue(dynamic_obstacle._update_signal_guard(
                scenario, event, event["route_s"], crossing=True,
            ))
        self.assertFalse(guard["confounded"])
        self.assertFalse(event.get("response_signal_limited", False))
        self.assertEqual(guard["command_attempts"], 2)

    def test_dynamic_signal_guard_releases_after_ego_passes_stopline(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        entry = next(item for item in scenario.dynamic_obstacle_catalog
                     if item["id"] == "general_south_east_to_west_diagonal")
        event = copy.deepcopy(entry)
        event.update(sequence=1, phase="crossing", model="Man1", speed_mps=2.5)
        dynamic_obstacle._initialize_signal_guard(scenario, event)
        guard = event["signal_guard"]
        guard.update(verified=True, command_accepted=True, phase="green")
        self.assertTrue(dynamic_obstacle._update_signal_guard(
            scenario, event, guard["stopline_route_s"] + 0.01, crossing=True,
        ))
        self.assertEqual(guard["phase"], "released_after_stopline")
        self.assertFalse(guard["confounded"])
        scenario.grpc.get_traffic_light_state_by_id.assert_not_called()
        scenario.grpc.set_traffic_light_color_verified.assert_not_called()

    def test_front_crosswalk_signal_guard_holds_past_stopline_through_conflict(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        entry = next(item for item in scenario.dynamic_obstacle_catalog
                     if item["id"] == "front_crosswalk_straight_left_to_right")
        event = copy.deepcopy(entry)
        event.update(sequence=1, phase="crossing", model="Man1", speed_mps=2.5)
        dynamic_obstacle._initialize_signal_guard(scenario, event)
        guard = event["signal_guard"]
        guard.update(
            verified=True,
            command_accepted=True,
            phase="green",
            last_query_wall=100.0,
            last_command_wall=100.0,
        )
        self.assertLess(guard["stopline_route_s"], event["route_s"])
        self.assertGreater(guard["release_route_s"], event["route_s"])
        with patch.object(dynamic_obstacle.time, "monotonic", return_value=100.1):
            self.assertTrue(dynamic_obstacle._update_signal_guard(
                scenario, event, guard["stopline_route_s"] + 0.01, crossing=True,
            ))
        self.assertEqual(guard["phase"], "green")
        self.assertTrue(dynamic_obstacle._update_signal_guard(
            scenario, event, guard["release_route_s"] + 0.01, crossing=True,
        ))
        self.assertEqual(guard["phase"], "released_after_conflict")

    def test_dynamic_signal_guard_does_not_touch_other_twenty_paths(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        entry = next(item for item in scenario.dynamic_obstacle_catalog
                     if "north" in item["id"])
        event = copy.deepcopy(entry)
        event.update(sequence=1, phase="armed", model="Man1", speed_mps=2.5)
        dynamic_obstacle._initialize_signal_guard(scenario, event)
        self.assertFalse(event["signal_guard"]["required"])
        self.assertTrue(dynamic_obstacle._update_signal_guard(
            scenario, event, event["route_s"] - 10.0, crossing=False,
        ))
        scenario.grpc.get_traffic_light_state_by_id.assert_not_called()
        scenario.grpc.set_traffic_light_color_verified.assert_not_called()

    def test_signal_after_verified_npc_response_does_not_retroactively_invalidate_it(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        event.update(
            actor=None,
            phase="crossing",
            response_observed=True,
            response_kind="slowdown",
            response_signal_limited=False,
            min_ego_distance_m=10.0,
        )
        scenario.traffic_wait_state = {"stop_key": ("TL", 1.0)}
        dynamic_obstacle._observe_ego_response(
            scenario, event, {"speed_mps": 0.0}, event["route_s"],
        )
        self.assertFalse(event.get("response_signal_limited", False))
        dynamic_obstacle._finish_dynamic_obstacle(scenario, "reached_dest")
        self.assertEqual(event["phase"], "cleared")
        self.assertEqual(event.get("assessment_issues", []), [])

    def test_dynamic_arm_coordinates_direction_trigger_unchanged(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        spec = next(entry for entry in scenario.route_definition["zones"]["dynamic_obstacle"]["crossing_catalog"]
                    if entry["id"] == "legacy_right_to_left")
        legacy = next(entry for entry in scenario.dynamic_obstacle_catalog if entry["id"] == "legacy_right_to_left")
        dynamic_obstacle._apply_event_compatibility_state(scenario, dict(legacy, model="Man1", speed_mps=2.5))
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

    def test_dynamic_catalogue_rejects_shadowed_zone_settings(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        spec = scenario.route_definition["zones"]["dynamic_obstacle"]
        for key, value in (
            ("start", {"x": 1, "y": 2}), ("end", {"x": 3, "y": 4}),
            ("route_path_index", 936), ("speed_mps", 1.5), ("speed_kph", 5.4),
            ("trigger_distance_m", 30), ("arrival_tolerance_m", 0.5),
            ("max_crossing_sec", 15), ("move_distance_m", 20),
        ):
            with self.subTest(key=key):
                spec[key] = value
                try:
                    with self.assertRaisesRegex(RuntimeError, "Mixed.*" + key):
                        scenario.spawn_dynamic_obstacle()
                    scenario.grpc.spawn_pedestrian.assert_not_called()
                finally:
                    del spec[key]

    def test_dynamic_catalogue_rejects_empty_catalogue_and_old_entry_speed_keys(self):
        for catalogue in ([], None, {}):
            with self.subTest(catalogue=catalogue), self.assertRaisesRegex(RuntimeError, "non-empty list"):
                dynamic_obstacle._catalogue_entries({"crossing_catalog": catalogue})
        for entry in (
            {"id": "test", "speed_mps": 1.5},
            {"id": "test", "allowed_speeds_mps": [2.5], "speed_kph": 9.0},
            {"id": "test", "allowed_speeds_mps": [2.5], "move_distance_m": 20},
            {"id": "test"},
        ):
            with self.subTest(entry=entry), self.assertRaisesRegex(RuntimeError, "allowed_speeds_mps"):
                dynamic_obstacle._catalogue_entries({"crossing_catalog": [entry]})

    def test_dynamic_legacy_yaml_without_catalogue_still_applies_settings(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        zone = scenario.route_definition["zones"]["dynamic_obstacle"]
        original = zone.pop("crossing_catalog")[0]
        for key in ("route_link", "route_path_index", "start", "end",
                    "trigger_distance_m", "arrival_tolerance_m", "max_crossing_sec"):
            zone[key] = copy.deepcopy(original[key])
        zone["speed_mps"] = 1.5
        zone["start"]["y"] += 0.01
        zone["end"]["y"] += 0.01
        scenario.spawn_dynamic_obstacle()
        self.assertEqual(len(scenario.dynamic_obstacle_events), 1)
        event = scenario.dynamic_obstacle_events[0]
        self.assertEqual(event["start"], zone["start"])
        self.assertEqual(event["end"], zone["end"])
        self.assertEqual(event["speed_mps"], 1.5)
        zone["speed_kph"] = 9.0
        with self.assertRaisesRegex(RuntimeError, "speed_mps and speed_kph must agree"):
            scenario.spawn_dynamic_obstacle()
        del zone["speed_mps"]
        scenario.spawn_dynamic_obstacle()
        self.assertEqual(scenario.dynamic_obstacle_events[0]["speed_mps"], 2.5)

    def test_dynamic_catalogue_timing_has_one_effective_edit_location(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        zone = scenario.route_definition["zones"]["dynamic_obstacle"]
        raw = next(entry for entry in zone["crossing_catalog"] if entry["id"] == "legacy_right_to_left")
        raw["trigger_distance_m"] = 30.0
        scenario.spawn_dynamic_obstacle()
        legacy = next(entry for entry in scenario.dynamic_obstacle_catalog if entry["id"] == raw["id"])
        self.assertEqual(legacy["trigger_distance_m"], 30.0)
        # Moving the preserved legacy crossing also invalidates its approval.
        raw["start"]["y"] += 0.01
        with self.assertRaisesRegex(RuntimeError, "geometry changed since approval"):
            scenario.spawn_dynamic_obstacle()

    def test_dynamic_catalog_is_fail_closed_until_morai_approval(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)

        status = {
            item["id"]: item["status"]
            for item in scenario.dynamic_obstacle_catalog
        }
        self.assertEqual(status["legacy_right_to_left"], "approved")
        self.assertEqual(sum(value == "approved" for value in status.values()), 40)
        self.assertEqual(sum(value == "pending" for value in status.values()), 12)
        self.assertTrue(all(status[event["id"]] == "approved" for event in scenario.dynamic_obstacle_events))
        pending = [dict(entry, status="pending") for entry in scenario.dynamic_obstacle_catalog]
        self.assertEqual(dynamic_obstacle._schedule_candidates(pending, 3, 35), {})

    def test_dynamic_scene_is_reproducible_from_seed_not_previous_profile(self):
        first = self.make_scenario("full_loop_dynamic", seed=12345)
        second = self.make_scenario("full_loop_dynamic", seed=12345)
        self.setup_scenario(first)
        # The first setup writes weather/time/model history. The second scene
        # must still be identical because dynamic randomization ignores it.
        self.setup_scenario(second)

        signature = lambda scenario: [
            (
                event["id"],
                event["model"],
                event["speed_mps"],
                event["start"],
                event["end"],
            )
            for event in scenario.dynamic_obstacle_events
        ]
        self.assertEqual(signature(first), signature(second))

    def test_three_event_schedule_exists_only_with_safe_approved_spacing(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        self.assertEqual(scenario.cfg["dynamic_obstacle"]["minimum_event_spacing_m"], 50.0)
        catalogue = copy.deepcopy(scenario.dynamic_obstacle_catalog)
        for entry in catalogue:
            entry["status"] = "approved"

        schedules = dynamic_obstacle._schedule_candidates(
            catalogue,
            maximum_count=3,
            minimum_spacing_m=50.0,
        )
        self.assertNotIn(3, schedules)  # The inset mission box is narrower.
        self.assertIn(2, schedules)
        self.assertEqual(len(schedules[2]), 10)
        self.assertTrue(all(
            pair[1]["route_s"] - pair[0]["route_s"] >= 50.0
            for pair in schedules[2]
        ))
        self.assertTrue(all(
            not any(
                event["id"].startswith(("front_crosswalk_", "rear_crosswalk_", "between_crosswalks_"))
                for event in pair
            )
            for pair in schedules[2]
        ))
        # The 47m forward-diagonal pair missed the second ETA during review.
        self.assertTrue(all(
            pair[0]["id"] != "general_south_east_to_west_diagonal__0_2_forward"
            for pair in schedules[2]
        ))
        events = schedules[2][0]
        self.assertEqual(
            [event["route_s"] for event in events],
            sorted(event["route_s"] for event in events),
        )
        for current, following in zip(events[:-1], events[1:]):
            self.assertGreaterEqual(
                following["route_s"] - current["route_s"], 50.0
            )
        # The generic max-three implementation remains intact for a site
        # that really has sufficient distance; do not relax the 35m gate.
        spacious = [dict(catalogue[0], id=str(i), route_s=40.0 + i * 50.0)
                    for i in range(3)]
        self.assertIn(3, dynamic_obstacle._schedule_candidates(spacious, 3, 35.0))

    def test_left_start_gets_earlier_eta_based_trigger(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        by_id = {
            item["id"]: item for item in scenario.dynamic_obstacle_catalog
        }
        right = dict(by_id["legacy_right_to_left"], speed_mps=2.5)
        left = dict(by_id["legacy_left_to_right"], speed_mps=2.5)
        ego = {"speed": 5.0}

        right_timing = dynamic_obstacle._timing_snapshot(
            scenario, right, ego, current_s=50.0,
        )
        left_timing = dynamic_obstacle._timing_snapshot(
            scenario, left, ego, current_s=50.0,
        )

        self.assertGreater(left["start_to_conflict_m"], right["start_to_conflict_m"])
        self.assertGreater(
            left_timing["trigger_distance_m"],
            right_timing["trigger_distance_m"],
        )

    def test_dynamic_timing_uses_si_speed_from_real_grpc_adapter(self):
        from ..adapters.morai_grpc import _morai_get_ego_motion_state
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        state = SimpleNamespace(
            velocity=SimpleNamespace(x=0.0, y=28.8, z=0.0),
            transform=transform(0.0, 0.0, 28.4, 90.0),
            vehicle_state=SimpleNamespace(
                current_link_info=SimpleNamespace(id=SimpleNamespace(value="link")),
                front_wheel_angle=0.0, collision_objects=[],
            ),
        )
        client = Mock()
        client.get_ego.return_value.get_actor_state.return_value = state
        observation = _morai_get_ego_motion_state(client)
        timing = dynamic_obstacle._timing_snapshot(
            scenario, scenario.dynamic_obstacle_events[0], observation, 40.0,
        )
        self.assertAlmostEqual(timing["ego_speed_mps"], 8.0)
        self.assertAlmostEqual(timing["safe_stop_distance_m"], 8.0 + 64.0 / 7.0 + 3.0)

    def test_dynamic_trigger_waits_through_signal_stop_then_uses_safe_window(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        event.update(next(entry for entry in scenario.dynamic_obstacle_catalog if entry["id"] == "legacy_right_to_left"))
        current_s = event["route_s"] - 20.0

        scenario.traffic_wait_state = {"stop_key": ("TL", 1.0)}
        with patch.object(scenario, "_spawn_active_dynamic_obstacle") as spawn:
            scenario.update_dynamic_obstacle(
                {"x": 0.0, "y": 0.0, "speed": 0.2}, current_s,
            )
            spawn.assert_not_called()

            scenario.traffic_wait_state = None
            scenario.update_dynamic_obstacle(
                {"x": 0.0, "y": 0.0, "speed": 5.0}, current_s,
            )
            spawn.assert_called_once_with()
        self.assertGreaterEqual(
            event["trigger"]["remaining_m"],
            event["trigger"]["safe_stop_distance_m"],
        )

    def test_dynamic_missed_trigger_fails_instead_of_spawning_behind_ego(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]

        with self.assertRaisesRegex(RuntimeError, "missed_trigger"):
            scenario.update_dynamic_obstacle(
                {"x": 0.0, "y": 0.0, "speed": 5.0},
                event["route_s"] + 2.0,
            )
        self.assertEqual(event["phase"], "failed")

    def make_dynamic_spawn_retry_case(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        event.update(next(entry for entry in scenario.dynamic_obstacle_catalog
                          if entry["id"] == "legacy_right_to_left"))
        event.update(model="Man1", speed_mps=2.5, phase="armed")
        scenario.dynamic_obstacle_events = [event]
        scenario.dynamic_obstacle_models = ["Woman1", "Man1"]
        scenario.traffic_light_stops = []
        scenario.traffic_wait_state = None
        scenario.grpc.destroy_pedestrians_near.return_value = []
        scenario.grpc.get_all_pedestrian_states.return_value = []
        clock = [100.0]
        scenario._scenario_now = lambda: clock[0]
        ego = {"x": -60.0, "y": -30.0, "speed_mps": 5.0}
        actor = Mock()
        actor.get_id.return_value = "fallback-pedestrian"
        actor.destroy.return_value = True
        return scenario, event, ego, actor, clock

    def test_dynamic_spawn_failure_retries_approved_model_on_next_fresh_tick(self):
        scenario, event, ego, actor, clock = self.make_dynamic_spawn_retry_case()
        scenario.grpc.spawn_pedestrian.side_effect = [None, actor]
        scenario.update_dynamic_obstacle(ego, event["route_s"] - 20.0)
        self.assertEqual(scenario.grpc.spawn_pedestrian.call_count, 1)
        self.assertEqual(event["phase"], "armed")
        self.assertIsNone(event["actor"])

        clock[0] += 0.2
        scenario.update_dynamic_obstacle(ego, event["route_s"] - 19.0)
        self.assertEqual([call.kwargs["model_name"] for call in
                          scenario.grpc.spawn_pedestrian.call_args_list], ["Man1", "Woman1"])
        self.assertEqual(event["phase"], "crossing")
        self.assertEqual(event["model"], "Woman1")
        self.assertEqual(scenario.dynamic_obstacle_model, "Woman1")
        self.assertEqual(event["trigger"]["remaining_m"], 19.0)
        self.assertAlmostEqual(event["trigger"]["trigger_scenario_time"], 100.2)
        self.assertEqual([a["trigger"]["remaining_m"] for a in event["spawn_attempts"]], [20, 19])
        with open(scenario.profile_state_path(), encoding="utf-8") as source:
            report = json.load(source)
        self.assertEqual(report["dynamic_obstacle_model"], "Woman1")
        actual = report["dynamic_obstacle_events"][0]
        self.assertEqual(actual["planned_model"], "Man1")
        self.assertEqual(actual["model"], "Woman1")
        self.assertEqual([a["result"] for a in actual["spawn_attempts"]], ["failed", "spawned"])

    def test_dynamic_retry_excludes_unapproved_model_speed_pairs(self):
        scenario, event, ego, actor, _ = self.make_dynamic_spawn_retry_case()
        scenario.cfg["dynamic_obstacle"]["model_selection_policy"] = "approved_pairs"
        scenario.dynamic_obstacle_models = ["Woman1", "Boy", "Man1"]
        event.update(allowed_speeds_mps=[2.5, 3.0], approved_combinations=[
            {"model": "Man1", "speed_mps": 2.5},
            {"model": "Woman1", "speed_mps": 3.0},
        ])
        scenario.grpc.spawn_pedestrian.side_effect = [None, actor]
        with self.assertRaisesRegex(RuntimeError, "every eligible"):
            scenario.update_dynamic_obstacle(ego, event["route_s"] - 20)
        self.assertEqual(scenario.grpc.spawn_pedestrian.call_count, 1)
        self.assertEqual(event["phase"], "failed")

    def test_dynamic_retry_aborts_if_failure_delay_loses_safe_window(self):
        for remaining, expected in ((5.0, "safe_stop"), (-2.0, "missed_trigger")):
            with self.subTest(remaining=remaining):
                scenario, event, ego, actor, clock = self.make_dynamic_spawn_retry_case()
                scenario.grpc.spawn_pedestrian.side_effect = [None, actor]
                scenario.update_dynamic_obstacle(ego, event["route_s"] - 20)
                clock[0] += 4
                with self.assertRaisesRegex(RuntimeError, expected):
                    scenario.update_dynamic_obstacle(ego, event["route_s"] - remaining)
                self.assertEqual(scenario.grpc.spawn_pedestrian.call_count, 1)
                self.assertEqual(event["phase"], "failed")

    def test_dynamic_retry_uses_new_speed_for_stopping_distance(self):
        scenario, event, ego, actor, _ = self.make_dynamic_spawn_retry_case()
        scenario.grpc.spawn_pedestrian.side_effect = [None, actor]
        scenario.update_dynamic_obstacle(ego, event["route_s"] - 20)
        with self.assertRaisesRegex(RuntimeError, "safe_stop"):
            scenario.update_dynamic_obstacle(dict(ego, speed_mps=10.0), event["route_s"] - 19)
        self.assertEqual(scenario.grpc.spawn_pedestrian.call_count, 1)

    def test_dynamic_retry_waits_for_signal_then_rechecks_timing(self):
        scenario, event, ego, actor, _ = self.make_dynamic_spawn_retry_case()
        scenario.grpc.spawn_pedestrian.side_effect = [None, actor]
        scenario.update_dynamic_obstacle(ego, event["route_s"] - 20)
        scenario.traffic_wait_state = {"stop_key": ("TL", 1.0)}
        scenario.update_dynamic_obstacle(dict(ego, speed_mps=0.0), event["route_s"] - 19)
        self.assertEqual(scenario.grpc.spawn_pedestrian.call_count, 1)
        scenario.traffic_wait_state = None
        scenario.update_dynamic_obstacle(ego, event["route_s"] - 19)
        self.assertEqual(scenario.grpc.spawn_pedestrian.call_count, 2)

    def test_dynamic_retry_checks_eta_not_only_remaining_distance(self):
        scenario, event, ego, actor, _ = self.make_dynamic_spawn_retry_case()
        scenario.grpc.spawn_pedestrian.side_effect = [None, actor]
        scenario.update_dynamic_obstacle(ego, event["route_s"] - 20)
        # Slow Ego: NPC would finish too early, so wait without another RPC.
        scenario.update_dynamic_obstacle(dict(ego, speed_mps=2.0), event["route_s"] - 19)
        self.assertEqual(scenario.grpc.spawn_pedestrian.call_count, 1)
        # A long left-side approach can lose its NPC ETA window while there
        # is still enough braking distance. This must also stop the retry.
        event["start_to_conflict_m"] = 15.0
        with self.assertRaisesRegex(RuntimeError, "NPC would arrive after Ego"):
            scenario.update_dynamic_obstacle(ego, event["route_s"] - 19)
        self.assertEqual(scenario.grpc.spawn_pedestrian.call_count, 1)

    def test_dynamic_retry_is_bounded_and_deterministic(self):
        orders = []
        for inventory in (["Woman1", "Boy", "Man1"], ["Man1", "Woman1", "Boy"]):
            scenario, event, ego, _, _ = self.make_dynamic_spawn_retry_case()
            scenario.dynamic_obstacle_models = inventory
            scenario.grpc.spawn_pedestrian.return_value = None
            scenario.update_dynamic_obstacle(ego, event["route_s"] - 20)
            scenario.update_dynamic_obstacle(ego, event["route_s"] - 19)
            with self.assertRaisesRegex(RuntimeError, "every eligible"):
                scenario.update_dynamic_obstacle(ego, event["route_s"] - 18)
            with self.assertRaisesRegex(RuntimeError, "already failed"):
                scenario.update_dynamic_obstacle(ego, event["route_s"] - 17)
            orders.append([call.kwargs["model_name"] for call in
                           scenario.grpc.spawn_pedestrian.call_args_list])
        self.assertEqual(orders, [["Man1", "Boy", "Woman1"]] * 2)

    def test_dynamic_recovered_spawn_does_not_retry_another_model(self):
        scenario, event, ego, actor, _ = self.make_dynamic_spawn_retry_case()
        actor._scenario_runner_ambiguous_recovery = True
        scenario.grpc.spawn_pedestrian.return_value = actor
        scenario.update_dynamic_obstacle(ego, event["route_s"] - 20)
        self.assertEqual(scenario.grpc.spawn_pedestrian.call_count, 1)
        self.assertEqual(event["model"], "Man1")
        self.assertEqual(event["spawn_attempts"][0]["result"], "recovered")

    def test_dynamic_spawn_exception_still_stops_without_model_retry(self):
        scenario, event, ego, _, _ = self.make_dynamic_spawn_retry_case()
        scenario.grpc.spawn_pedestrian.side_effect = RuntimeError("RPC disconnected")
        with self.assertRaisesRegex(RuntimeError, "RPC disconnected"):
            scenario.update_dynamic_obstacle(ego, event["route_s"] - 20)
        self.assertEqual(scenario.grpc.spawn_pedestrian.call_count, 1)
        self.assertEqual(event["phase"], "failed")
        self.assertEqual(event["spawn_attempts"][0]["result"], "error")

    def test_dynamic_review_keeps_explicit_model_without_fallback(self):
        scenario, event, ego, actor, _ = self.make_dynamic_spawn_retry_case()
        scenario.cfg["dynamic_obstacle_review"] = [
            {"id": event["id"], "model": "Man1", "speed_mps": 2.5}
        ]
        # Review selection restricts allowed_models to exactly the requested one.
        event["allowed_models"] = ["Man1"]
        scenario.grpc.spawn_pedestrian.side_effect = [None, actor]
        with self.assertRaisesRegex(RuntimeError, "every eligible"):
            scenario.update_dynamic_obstacle(ego, event["route_s"] - 20)
        self.assertEqual(scenario.grpc.spawn_pedestrian.call_count, 1)

    def test_dynamic_collection_event_keeps_npc_and_log_sequences_distinct(self):
        rows = []

        def collection_writer(event_name, **fields):
            row = dict(event=event_name, sequence=len(rows), **fields)
            rows.append(row)

        scenario = SimpleNamespace(
            cfg={"dynamic_obstacle_review": None},
            event=collection_writer,
        )
        event = {
            "id": "crossing-test",
            "sequence": 2,
            "model": "Man1",
            "route_s": 90.0,
            "signal_guard": {
                "light_id": "C-test",
                "stopline_route_s": 67.0,
                "release_route_s": 95.0,
                "mode": "auto",
            },
        }

        dynamic_obstacle._emit_event(
            scenario, "dynamic_obstacle_started", event, actor_id="actor-1"
        )
        dynamic_obstacle._emit_signal_guard(
            scenario, "dynamic_signal_guard_command", event, verified=True
        )

        self.assertEqual([row["sequence"] for row in rows], [0, 1])
        self.assertEqual([row["dynamic_sequence"] for row in rows], [2, 2])

    def enable_collection_actor_observation(self, scenario, time_mode="simulation_free_run"):
        # Exercise the actual unchanged collection hook/reader without
        # importing ROS, starting recording, or constructing its output dir.
        source = (Path(__file__).resolve().parents[2] / "data_collection" /
                  "internal_sensor_record/tools/run_proven_three_lap_scenario.py")
        module = ast.parse(source.read_text(encoding="utf-8"))
        reader = next(node for node in module.body
                      if isinstance(node, ast.ClassDef) and node.name == "InventoryObservedActor")
        adapter = next(node for node in module.body
                       if isinstance(node, ast.ClassDef) and node.name == "RawScenario")
        adapter.body = [node for node in adapter.body if isinstance(node, ast.FunctionDef)
                        and node.name in {"_spawn_active_dynamic_obstacle", "observe_collision",
                                          "on_expert_timeline_tick"}]
        namespace = {"FullLoopScenario": MissionScenario, "time": time}
        exec(compile(ast.Module(body=[reader, adapter], type_ignores=[]), str(source), "exec"), namespace)
        scenario.__class__ = namespace["RawScenario"]
        scenario.time_mode = time_mode
        scenario.event = Mock()
        return namespace["InventoryObservedActor"]

    def set_collection_actor_state(self, scenario, actor, position):
        identity = SimpleNamespace(id=SimpleNamespace(value=actor.get_id()), object_type=2)
        actor.get_object_info.return_value = identity
        state = SimpleNamespace(actor_info=identity, transform=transform(
            position["x"], position["y"], position["z"], 0.0))
        scenario.grpc.get_all_pedestrian_states.return_value = [state]
        return state

    def test_dynamic_collection_reads_inventory_through_crossing_stop_and_removal(self):
        scenario, event, ego, actor, clock = self.make_dynamic_spawn_retry_case()
        reader = self.enable_collection_actor_observation(scenario)
        actor.get_actor_state.side_effect = RuntimeError("broken GetActorState RPC")
        scenario.grpc.spawn_pedestrian.return_value = actor
        scenario.grpc.control_pedestrian.return_value = True
        self.set_collection_actor_state(scenario, actor, event["start"])
        scenario.update_dynamic_obstacle(ego, event["route_s"] - 20)
        observed = event["actor"]
        self.assertIsInstance(observed, reader)
        self.assertIs(observed, scenario.dynamic_obstacle)
        self.assertEqual(scenario.scenario_actors, [observed])

        clock[0] += event["path_length_m"] / event["speed_mps"]
        self.set_collection_actor_state(scenario, actor, event["end"])
        scenario.update_dynamic_obstacle(dict(ego, speed_mps=0), event["route_s"] - 5)
        self.assertEqual(event["phase"], "cleared")
        scenario.grpc.control_pedestrian.assert_called_once()
        self.assertIs(scenario.grpc.control_pedestrian.call_args.args[0], observed)
        self.assertEqual(scenario.grpc.control_pedestrian.call_args.args[3], 0.0)

        def destroy():
            scenario.grpc.get_all_pedestrian_states.return_value = []
            return True

        actor.destroy.side_effect = destroy
        clock[0] += 1
        scenario.update_dynamic_obstacle(dict(ego, speed_mps=2), event["route_s"] + 3.1)
        actor.get_actor_state.assert_not_called()
        actor.destroy.assert_called_once_with()
        self.assertTrue(event["resumed_after_clear"])
        self.assertIsNone(event["actor"])
        self.assertIsNone(scenario.dynamic_obstacle)
        self.assertEqual(scenario.scenario_actors, [])
        self.assertEqual(scenario.dynamic_obstacle_event_index, 1)

    def test_dynamic_collection_retry_wraps_only_successful_spawn(self):
        scenario, event, ego, actor, clock = self.make_dynamic_spawn_retry_case()
        reader = self.enable_collection_actor_observation(scenario)
        actor.get_actor_state.side_effect = RuntimeError("broken GetActorState RPC")
        scenario.grpc.spawn_pedestrian.side_effect = [None, actor]
        self.set_collection_actor_state(scenario, actor, event["start"])
        scenario.update_dynamic_obstacle(ego, event["route_s"] - 20)
        self.assertEqual(event["phase"], "armed")
        self.assertIsNone(event["actor"])
        self.assertIsNone(scenario.dynamic_obstacle)
        clock[0] += .1
        scenario.update_dynamic_obstacle(ego, event["route_s"] - 19.5)
        self.assertEqual(event["model"], "Woman1")
        self.assertIsInstance(event["actor"], reader)
        self.assertEqual(scenario.scenario_actors, [event["actor"]])
        scenario.update_dynamic_obstacle(ego, event["route_s"] - 19)
        actor.get_actor_state.assert_not_called()
        wrapped = [call for call in scenario.event.call_args_list
                   if call.args[0] == "scenario_actor_observation_source"]
        self.assertEqual(len(wrapped), 1)

    def test_dynamic_collection_inventory_requires_unique_matching_actor(self):
        for mismatch in ("other_id", "other_type", "duplicate"):
            with self.subTest(mismatch=mismatch):
                scenario, event, ego, actor, _ = self.make_dynamic_spawn_retry_case()
                self.enable_collection_actor_observation(scenario)
                actor.get_actor_state.side_effect = RuntimeError("broken GetActorState RPC")
                scenario.grpc.spawn_pedestrian.return_value = actor
                state = self.set_collection_actor_state(scenario, actor, event["start"])
                scenario.update_dynamic_obstacle(ego, event["route_s"] - 20)
                wrong = copy.deepcopy(state)
                if mismatch == "other_id":
                    wrong.actor_info.id.value = "unrelated-pedestrian"
                    states = [wrong]
                elif mismatch == "other_type":
                    wrong.actor_info.object_type = 999
                    states = [wrong]
                else:
                    states = [state, wrong]
                scenario.grpc.get_all_pedestrian_states.return_value = states
                with self.assertRaisesRegex(RuntimeError, "Expected one fresh inventory actor"):
                    scenario.update_dynamic_obstacle(ego, event["route_s"] - 19)
                actor.get_actor_state.assert_not_called()
                self.assertEqual(scenario.grpc.spawn_pedestrian.call_count, 1)

    def test_dynamic_collection_one_two_three_events_keep_separate_wrapped_handles(self):
        for count, seed in ((1, 0), (2, 3), (3, 1)):
            with self.subTest(count=count):
                scenario = self.make_scenario("full_loop_dynamic", seed)
                self.setup_scenario(scenario)
                # Exercise all supported counts independently of whether the
                # current inset site has room for three real crossing events.
                legacy = scenario.dynamic_obstacle_catalog[0]
                catalogue = [dict(legacy, id="collection-test-%d" % i,
                                  route_s=40.0 + i * 50.0) for i in range(count)]
                scenario.cfg["dynamic_obstacle"]["max_count"] = count
                with patch.object(dynamic_obstacle.random, "Random") as factory:
                    def choose(choices):
                        return max(choices) if isinstance(choices[0], int) else choices[0]
                    factory.return_value.choice.side_effect = choose
                    scenario.dynamic_obstacle_events = dynamic_obstacle._choose_schedule(
                        scenario, catalogue, scenario.cfg["dynamic_obstacle"])
                scenario.dynamic_obstacle_event_index = 0
                dynamic_obstacle._apply_event_compatibility_state(scenario, scenario.dynamic_obstacle_events[0])
                self.assertEqual(len(scenario.dynamic_obstacle_events), count)
                reader = self.enable_collection_actor_observation(scenario)
                scenario.traffic_light_stops = []
                scenario.traffic_wait_state = None
                scenario.grpc.destroy_pedestrians_near.return_value = []
                scenario.grpc.control_pedestrian.return_value = True
                clock = [100.0]
                scenario._scenario_now = lambda: clock[0]
                handles = []
                for index, event in enumerate(scenario.dynamic_obstacle_events):
                    actor = Mock()
                    actor.get_id.return_value = "collection-actor-%d" % index
                    actor.get_actor_state.side_effect = RuntimeError("broken GetActorState RPC")
                    def destroy():
                        scenario.grpc.get_all_pedestrian_states.return_value = []
                        return True
                    actor.destroy.side_effect = destroy
                    scenario.grpc.spawn_pedestrian.return_value = actor
                    self.set_collection_actor_state(scenario, actor, event["start"])
                    speed = 5.0 if event["id"] == "legacy_right_to_left" else 3.0
                    ego = {"x": -60.0, "y": -100.0, "speed_mps": speed}
                    timing = dynamic_obstacle._timing_snapshot(scenario, event, ego, event["route_s"])
                    # Being inside the distance gate alone is insufficient:
                    # place the mock Ego inside the configured ETA window too.
                    remaining = min(timing["trigger_distance_m"] - .05,
                                    speed * (timing["npc_eta_sec"] + 1.5))
                    self.assertGreaterEqual(remaining, timing["safe_stop_distance_m"])
                    scenario.update_dynamic_obstacle(ego, event["route_s"] - remaining)
                    self.assertIsInstance(event["actor"], reader)
                    self.assertEqual(event["actor"].get_id(), actor.get_id())
                    self.assertEqual(scenario.scenario_actors, [event["actor"]])
                    handles.append(event["actor"])
                    clock[0] += event["path_length_m"] / event["speed_mps"]
                    self.set_collection_actor_state(scenario, actor, event["end"])
                    scenario.update_dynamic_obstacle(dict(ego, speed_mps=0), event["route_s"] - 5)
                    clock[0] += 1
                    scenario.update_dynamic_obstacle(dict(ego, speed_mps=2), event["route_s"] + 3.1)
                    actor.get_actor_state.assert_not_called()
                    actor.destroy.assert_called_once_with()
                    self.assertEqual(scenario.scenario_actors, [])
                self.assertEqual(len({id(actor) for actor in handles}), count)
                self.assertEqual(scenario.dynamic_obstacle_event_index, count)

    def test_dynamic_collection_realtime_retains_original_actor_reader(self):
        scenario, event, ego, actor, _ = self.make_dynamic_spawn_retry_case()
        reader = self.enable_collection_actor_observation(scenario, time_mode="realtime")
        scenario.grpc.spawn_pedestrian.return_value = actor
        actor.get_actor_state.return_value = SimpleNamespace(transform=transform(
            event["start"]["x"], event["start"]["y"], event["start"]["z"], 0.0))
        scenario.update_dynamic_obstacle(ego, event["route_s"] - 20)
        self.assertIs(event["actor"], actor)
        self.assertNotIsInstance(event["actor"], reader)
        scenario.update_dynamic_obstacle(ego, event["route_s"] - 19)
        actor.get_actor_state.assert_called_once_with()

    def test_dynamic_spawn_hook_cannot_replace_actor_identity(self):
        scenario, event, ego, actor, _ = self.make_dynamic_spawn_retry_case()
        scenario.grpc.spawn_pedestrian.return_value = actor
        original_spawn = scenario._spawn_active_dynamic_obstacle
        unrelated = Mock()
        unrelated.get_id.return_value = "wrong-pedestrian"
        def wrong_hook():
            original_spawn()
            scenario.dynamic_obstacle = unrelated
        scenario._spawn_active_dynamic_obstacle = wrong_hook
        with self.assertRaisesRegex(RuntimeError, "changed the pedestrian actor identity"):
            scenario.update_dynamic_obstacle(ego, event["route_s"] - 20)
        self.assertIs(event["actor"], actor)
        unrelated.get_actor_state.assert_not_called()
        unrelated.destroy.assert_not_called()

    def test_signal_stop_is_not_counted_as_dynamic_ego_response(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        event["phase"] = "crossing"
        event["trigger"] = {"ego_speed_mps": 5.0}

        scenario.traffic_wait_state = {"stop_key": ("TL", 1.0)}
        dynamic_obstacle._observe_ego_response(
            scenario,
            event,
            {"speed": 0.0},
            event["route_s"],
        )
        self.assertFalse(event["response_observed"])

        scenario.traffic_wait_state = None
        scenario.traffic_light_stops = [{
            "route_s": event["route_s"] + 5.0,
            "light_id": "TL",
        }]
        scenario.grpc.get_traffic_light_state_by_id.return_value = {
            "id": "TL",
            "color": 1,
        }
        dynamic_obstacle._observe_ego_response(
            scenario,
            event,
            {"speed": 3.5},
            event["route_s"],
        )
        self.assertFalse(event["response_observed"])

        scenario._dynamic_signal_cache = None
        scenario.grpc.get_traffic_light_state_by_id.return_value = {
            "id": "TL",
            "color": 16,
        }
        event["observed_along_m"] = 1.0
        dynamic_obstacle._observe_ego_response(scenario, event, {"speed": 5.0}, event["route_s"])
        dynamic_obstacle._observe_ego_response(
            scenario,
            event,
            {"speed": 3.5},
            event["route_s"],
        )
        self.assertTrue(event["response_observed"])
        self.assertEqual(event["response_kind"], "slowdown")

    def test_cleared_event_is_removed_before_next_event_is_armed(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        first = scenario.dynamic_obstacle_events[0]
        second = copy.deepcopy(first)
        second.update(
            id="next_crossing",
            sequence=2,
            phase="pending",
            route_s=first["route_s"] + 40.0,
            actor=None,
            actor_id=None,
        )
        actor = Mock()
        actor.get_id.return_value = "owned-dynamic-1"
        actor.destroy.return_value = True
        actor.get_actor_state.return_value = SimpleNamespace(
            transform=transform(first["end"]["x"], first["end"]["y"], 28.4, 0.0),
        )
        first.update(
            phase="cleared",
            actor=actor,
            actor_id="owned-dynamic-1",
            response_observed=True,
            cleared_scenario_time=scenario._scenario_now(),
        )
        scenario.dynamic_obstacle_events = [first, second]
        scenario.dynamic_obstacle_event_index = 0
        scenario.dynamic_obstacle = actor
        scenario.dynamic_obstacle_owned_ids = {"owned-dynamic-1"}
        scenario.scenario_actors.append(actor)
        scenario.grpc.get_all_pedestrian_states.return_value = []

        dynamic_obstacle.update_dynamic_obstacle(
            scenario,
            {"x": 0.0, "y": 0.0, "speed": 2.0},
            first["route_s"] + 3.1,
        )

        actor.destroy.assert_called_once_with()
        self.assertEqual(scenario.dynamic_obstacle_event_index, 1)
        self.assertEqual(second["phase"], "armed")
        self.assertEqual(scenario.dynamic_obstacle_phase, "waiting")

    def test_dynamic_actor_removal_failure_is_not_silently_accepted(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        actor = Mock()
        actor.get_id.return_value = "owned-dynamic-1"
        actor.destroy.return_value = False
        event.update(actor=actor, actor_id="owned-dynamic-1")
        scenario.dynamic_obstacle = actor
        scenario.scenario_actors.append(actor)
        scenario.grpc.get_all_pedestrian_states.return_value = [SimpleNamespace(
            actor_info=SimpleNamespace(id=SimpleNamespace(value="owned-dynamic-1")))]

        with self.assertRaisesRegex(RuntimeError, "Failed to remove"):
            dynamic_obstacle._destroy_event_actor(
                scenario, event, strict=True,
            )

    def test_dynamic_generated_offsets_have_individual_approvals_and_varied_angles(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        generated = [entry for entry in scenario.dynamic_obstacle_catalog if "__" in entry["id"]]
        self.assertEqual(len(generated), 34)
        self.assertEqual(sum(entry["status"] == "pending" for entry in generated), 12)
        self.assertEqual(sum(entry["status"] == "approved" for entry in generated), 22)
        self.assertGreater(len({round(entry["yaw_deg"], 2) for entry in generated}), 6)
        self.assertTrue(all(entry["start_d_m"] * entry["end_d_m"] < 0 for entry in generated))

    def test_dynamic_reverse_geometry_revision_preserves_all_other_paths(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        spec = scenario.route_definition["zones"]["dynamic_obstacle"]
        original = copy.deepcopy(spec["crossing_catalog"])
        anchor = next(entry for entry in original if entry["id"] == "general_south_east_to_west_diagonal")
        anchor.pop("reverse_candidate_geometry")
        anchor.pop("candidate_geometry_overrides", None)
        old = {entry["id"]: entry for entry in dynamic_obstacle._expand_catalogue(scenario, original, spec)}
        current = {entry["id"]: entry for entry in scenario.dynamic_obstacle_catalog}
        self.assertEqual(set(old), set(current))
        revised = []
        for crossing_id, entry in current.items():
            is_revised = crossing_id.startswith("general_south_") and crossing_id.endswith("_reverse")
            if is_revised:
                revised.append(entry)
                self.assertNotEqual(dynamic_obstacle._geometry_signature(entry),
                                    dynamic_obstacle._geometry_signature(old[crossing_id]))
                self.assertLess(abs(entry["yaw_deg"]), 11.0)
                self.assertGreater(entry["route_s"], 59.0)
                self.assertLess(entry["route_s"], 65.0)
                self.assertGreater(entry["end"]["x"], -49.2)
                self.assertLess(entry["end"]["x"], -48.8)
            else:
                self.assertEqual(dynamic_obstacle._geometry_signature(entry),
                                 dynamic_obstacle._geometry_signature(old[crossing_id]))
        self.assertEqual(len(revised), 9)

    def test_dynamic_reverse_geometry_rejects_missing_or_nonfinite_endpoints(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        spec = scenario.route_definition["zones"]["dynamic_obstacle"]
        for malformed in ({}, {"start": {"x": float("nan"), "y": -50},
                              "end": {"x": -45, "y": -50}}):
            catalogue = copy.deepcopy(spec["crossing_catalog"])
            anchor = next(entry for entry in catalogue if entry["id"] == "general_south_east_to_west_diagonal")
            anchor["reverse_candidate_geometry"] = malformed
            with self.assertRaisesRegex(RuntimeError, "finite start/end"):
                dynamic_obstacle._expand_catalogue(scenario, catalogue, spec)

    def test_dynamic_revised_reverse_approvals_are_bound_to_geometry(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        spec = scenario.route_definition["zones"]["dynamic_obstacle"]
        reverse = [entry for entry in scenario.dynamic_obstacle_catalog
                   if entry["id"].startswith("general_south_") and entry["id"].endswith("_reverse")]
        self.assertEqual(len(reverse), 9)
        self.assertEqual(sum(entry["status"] == "approved" for entry in reverse), 5)
        self.assertEqual(sum(entry["status"] == "pending" for entry in reverse), 4)
        for entry in reverse:
            if entry["status"] == "approved":
                self.assertEqual(entry["verification"]["geometry_sha256"], dynamic_obstacle._geometry_signature(entry))
                self.assertIn({"model": "Man1", "speed_mps": 2.5}, entry["approved_combinations"])
            edited = copy.deepcopy(entry)
            edited["end"]["x"] += 0.01
            if entry["status"] == "approved":
                with self.assertRaisesRegex(RuntimeError, "geometry changed"):
                    dynamic_obstacle._validate_catalog_entry(scenario, edited, spec, scenario.cfg["dynamic_obstacle"], set())
        catalogue = copy.deepcopy(spec["crossing_catalog"])
        anchor = next(entry for entry in catalogue if entry["id"] == "general_south_east_to_west_diagonal")
        anchor["approved_endpoint_variants"].pop("0_0_reverse")
        unreviewed = next(entry for entry in dynamic_obstacle._expand_catalogue(scenario, catalogue, spec)
                          if entry["id"].endswith("__0_0_reverse") and entry["id"].startswith("general_south_"))
        self.assertEqual(unreviewed["status"], "pending")
        self.assertEqual(unreviewed["approved_combinations"], [])

    def test_dynamic_run22_spawn_correction_changes_only_one_path(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        spec = scenario.route_definition["zones"]["dynamic_obstacle"]
        previous = copy.deepcopy(spec["crossing_catalog"])
        anchor = next(entry for entry in previous if entry["id"] == "general_south_east_to_west_diagonal")
        anchor["candidate_geometry_overrides"].pop("0_1_reverse")
        old = {entry["id"]: entry for entry in dynamic_obstacle._expand_catalogue(scenario, previous, spec)}
        current = {entry["id"]: entry for entry in dynamic_obstacle._expand_catalogue(scenario, spec["crossing_catalog"], spec)}
        target = "general_south_east_to_west_diagonal__0_1_reverse"
        self.assertEqual(set(old), set(current))
        self.assertEqual(len(current), 52)
        for crossing_id, entry in current.items():
            if crossing_id != target:
                for field in ("start", "end", "route_path_index", "conflict_point", "status",
                              "allowed_speeds_mps", "approved_combinations", "verification"):
                    self.assertEqual(entry.get(field), old[crossing_id].get(field), (crossing_id, field))
        entry = current[target]
        self.assertEqual(entry["start"], dict(old[target]["start"], y=old[target]["start"]["y"] + 0.5))
        # The relocated site's tangent changes the 2m offset by <1cm; the
        # explicit reviewed override, not a new implicit offset, is binding.
        for field in ("x", "y", "z"):
            self.assertLess(abs(entry["end"][field] - old[target]["end"][field]), 0.01)
        self.assertEqual(entry["route_path_index"], 876)
        self.assertEqual(entry["verification"]["geometry_sha256"], dynamic_obstacle._geometry_signature(entry))
        self.assertIn({"model": "Man1", "speed_mps": 2.5}, entry["approved_combinations"])
        self.assertEqual(dynamic_obstacle._model_speed_pairs(scenario, entry),
                         [(model, 2.5) for model in sorted(scenario.dynamic_obstacle_models)])

    def test_dynamic_run22_geometry_override_requires_new_approval(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        spec = scenario.route_definition["zones"]["dynamic_obstacle"]
        entry = copy.deepcopy(next(entry for entry in scenario.dynamic_obstacle_catalog
                                  if entry["id"] == "general_south_east_to_west_diagonal__0_1_reverse"))
        entry["status"] = "approved"
        entry["verification"]["geometry_sha256"] = "32aa6f7e03536ee06244f8ff16b565df87f4d8bba8c0b8cd831df979e7a0b279"
        with self.assertRaisesRegex(RuntimeError, "geometry changed"):
            dynamic_obstacle._validate_catalog_entry(scenario, entry, spec, scenario.cfg["dynamic_obstacle"], set())

    def test_dynamic_geometry_override_rejects_invalid_coordinates_or_variant(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        spec = scenario.route_definition["zones"]["dynamic_obstacle"]
        cases = [
            None, [], {"invalid_variant": {}}, {"1_1_forward": {}},
            {"0_1_reverse": {}}, {"0_1_reverse": {"start": None, "end": {}}},
            {"0_1_reverse": {"start": {"x": -71, "y": float("nan")}, "end": {"x": -45, "y": -52}}},
            {"0_1_reverse": {"start": {"x": -71, "y": -49.5}, "end": {"x": -70, "y": -52}}},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                catalogue = copy.deepcopy(spec["crossing_catalog"])
                anchor = next(entry for entry in catalogue if entry["id"] == "general_south_east_to_west_diagonal")
                anchor["candidate_geometry_overrides"] = overrides
                with self.assertRaisesRegex(RuntimeError, "candidate_geometry_overrides"):
                    dynamic_obstacle._expand_catalogue(scenario, catalogue, spec)

    def test_dynamic_run22_seed_remains_reproducible_after_geometry_correction(self):
        scenario = self.make_scenario("full_loop_dynamic", seed=14710863293134869393)
        self.setup_scenario(scenario)
        events = scenario.dynamic_obstacle_events
        self.assertEqual(len(events), 2)
        selected = [(event["id"], event["model"], event["speed_mps"]) for event in events]
        # Schedules are sorted by route_s. Moving a crossing can change a seed's
        # selected path, but must not change the selector or its reproducibility.
        self.assertTrue(all(event["status"] == "approved" for event in events))
        repeat = self.make_scenario("full_loop_dynamic", seed=14710863293134869393)
        self.setup_scenario(repeat)
        self.assertEqual(selected, [(event["id"], event["model"], event["speed_mps"])
                                    for event in repeat.dynamic_obstacle_events])

    def test_dynamic_yaml_pool_preserves_pair_evidence_and_pending_candidates(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        reviewed = [entry for entry in scenario.dynamic_obstacle_catalog
                    if entry["verification"].get("source") ==
                    "morai_26r1_20260921_green_box_relocation"]
        self.assertEqual(len(reviewed), 36)
        for entry in reviewed:
            with self.subTest(crossing=entry["id"]):
                self.assertIn({"model": "Man1", "speed_mps": 2.5}, entry["approved_combinations"])
                reviewed_speeds = {pair["speed_mps"] for pair in entry["approved_combinations"]}
                self.assertEqual(dynamic_obstacle._model_speed_pairs(scenario, entry),
                                 [(model, speed) for model in sorted(scenario.dynamic_obstacle_models)
                                  for speed in sorted(reviewed_speeds)])
                evidence = entry["verification"]
                self.assertTrue(evidence["morai_closed_loop"])
                self.assertTrue(evidence["collision_free"])
                self.assertTrue(evidence["run_seeds"])
                self.assertEqual(evidence["geometry_sha256"],
                                 dynamic_obstacle._geometry_signature(entry))
        # Removing an explicit approval must still exclude its candidate;
        # the model-pool policy cannot turn pending geometry into approval.
        pending = [dict(entry, status="pending") for entry in scenario.dynamic_obstacle_catalog]
        self.assertEqual(dynamic_obstacle._schedule_candidates(pending, 3, 35), {})

    def test_dynamic_non_crossing_and_nonfinite_paths_are_rejected(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        spec = scenario.route_definition["zones"]["dynamic_obstacle"]
        raw = copy.deepcopy(spec["crossing_catalog"][0])
        raw["end"]["x"] = raw["start"]["x"] + 1
        raw["end"]["y"] += 20
        with self.assertRaisesRegex(RuntimeError, "opposite safe sides"):
            dynamic_obstacle._validate_catalog_entry(scenario, raw, spec, scenario.cfg["dynamic_obstacle"], set())
        raw["start"]["x"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            dynamic_obstacle._validate_catalog_entry(scenario, raw, spec, scenario.cfg["dynamic_obstacle"], set())

    def test_dynamic_pair_approval_does_not_form_unreviewed_cross_product(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        scenario.cfg["dynamic_obstacle"]["model_selection_policy"] = "approved_pairs"
        entry = dict(scenario.dynamic_obstacle_catalog[0])
        entry.update(allowed_models=["Man1", "Woman1"], allowed_speeds_mps=[2.5, 3.0],
                     approved_combinations=[{"model": "Man1", "speed_mps": 2.5},
                                            {"model": "Woman1", "speed_mps": 3.0}])
        self.assertEqual(dynamic_obstacle._model_speed_pairs(scenario, entry), [("Man1", 2.5), ("Woman1", 3.0)])

    def test_dynamic_yaml_pool_applies_to_all_approved_routes_without_new_speeds(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        self.assertEqual(scenario.cfg["dynamic_obstacle"]["model_selection_policy"], "yaml_pool")
        self.assertEqual(set(scenario.dynamic_obstacle_models), set(scenario.cfg["dynamic_obstacle_models"]))
        approved = [entry for entry in scenario.dynamic_obstacle_catalog if entry["status"] == "approved"]
        self.assertEqual(len(approved), 40)
        for entry in approved:
            with self.subTest(crossing=entry["id"]):
                pairs = dynamic_obstacle._model_speed_pairs(scenario, entry)
                speeds = set(entry["allowed_speeds_mps"])
                if entry["approved_combinations"]:
                    speeds &= {float(item["speed_mps"]) for item in entry["approved_combinations"]}
                self.assertEqual(set(pairs), {(model, speed) for model in scenario.dynamic_obstacle_models
                                             for speed in speeds})
        # Missing simulator models are excluded even when historically reviewed.
        scenario.dynamic_obstacle_models = ["Dog1", "Dog1"]
        for entry in approved:
            self.assertEqual({model for model, _ in dynamic_obstacle._model_speed_pairs(scenario, entry)}, {"Dog1"})
        scenario.dynamic_obstacle_models = []
        with self.assertRaisesRegex(RuntimeError, "No approved dynamic crossing schedule"):
            dynamic_obstacle._choose_schedule(scenario, scenario.dynamic_obstacle_catalog, scenario.cfg["dynamic_obstacle"])

    def test_dynamic_yaml_pool_samples_each_npc_with_replacement(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        scenario.dynamic_obstacle_models = ["Man1", "Woman1", "Dog1"]
        # Test the generic 3-event selector without pretending three crossings
        # fit inside the newly restricted real-world placement region.
        catalogue = [dict(scenario.dynamic_obstacle_catalog[0], id=str(i), route_s=40.0 + i * 50.0)
                     for i in range(3)]
        for models in (("Man1", "Man1", "Man1"), ("Man1", "Woman1", "Dog1"),
                       ("Man1", "Man1", "Dog1")):
            with self.subTest(models=models):
                choices_seen = []
                def choose(choices):
                    choices_seen.append(choices)
                    if len(choices_seen) == 1:
                        self.assertIn(3, choices)
                        return 3
                    if len(choices_seen) == 2:
                        return choices[0]
                    # Every event must retain the entire model pool, regardless
                    # of the models selected for the preceding events.
                    self.assertEqual({model for model, _ in choices}, set(scenario.dynamic_obstacle_models))
                    return next(pair for pair in choices if pair[0] == models[len(choices_seen) - 3])
                with patch.object(dynamic_obstacle.random, "Random") as factory:
                    factory.return_value.choice.side_effect = choose
                    events = dynamic_obstacle._choose_schedule(scenario, catalogue,
                                                              scenario.cfg["dynamic_obstacle"])
                self.assertEqual(tuple(event["model"] for event in events), models)
                self.assertEqual(len(choices_seen), 5)
                self.assertTrue(all(event["model_selection_policy"] == "yaml_pool" for event in events))

    def test_dynamic_model_policy_defaults_to_strict_and_rejects_typos(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        entry = next(item for item in scenario.dynamic_obstacle_catalog if item["approved_combinations"])
        scenario.cfg["dynamic_obstacle"].pop("model_selection_policy")
        self.assertEqual(dynamic_obstacle._model_selection_policy(scenario), "approved_pairs")
        expected = {(item["model"], float(item["speed_mps"])) for item in entry["approved_combinations"]}
        self.assertEqual(set(dynamic_obstacle._model_speed_pairs(scenario, entry)), expected)
        scenario.cfg["dynamic_obstacle"]["model_selection_policy"] = "yaml_poo1"
        with self.assertRaisesRegex(RuntimeError, "model_selection_policy"):
            dynamic_obstacle._choose_schedule(scenario, scenario.dynamic_obstacle_catalog, scenario.cfg["dynamic_obstacle"])

    def test_dynamic_yaml_pool_does_not_broaden_explicit_review(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        entry = next(item for item in scenario.dynamic_obstacle_catalog
                     if item["verification"].get("source") == "morai_26r1_20260921_green_box_relocation")
        scenario.cfg["sync_runtime"]["enabled"] = False
        scenario.cfg["dynamic_obstacle_review"] = [{"id": entry["id"], "model": "Dog1", "speed_mps": 2.5}]
        events = dynamic_obstacle._choose_schedule(scenario, scenario.dynamic_obstacle_catalog, scenario.cfg["dynamic_obstacle"])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["model_selection_policy"], "review_exact")
        self.assertEqual(dynamic_obstacle._model_speed_pairs(scenario, events[0]), [("Dog1", 2.5)])

    def test_dynamic_yaml_pool_retry_records_actual_pair_evidence(self):
        scenario, event, ego, actor, _ = self.make_dynamic_spawn_retry_case()
        event.update(allowed_models=["Man1"], approved_combinations=[{"model": "Man1", "speed_mps": 2.5}])
        self.assertTrue(dynamic_obstacle._event_report(event)["has_pair_review_record"])
        scenario.grpc.spawn_pedestrian.side_effect = [None, actor]
        scenario.update_dynamic_obstacle(ego, event["route_s"] - 20)
        scenario.update_dynamic_obstacle(ego, event["route_s"] - 19)
        report = dynamic_obstacle._event_report(event)
        self.assertEqual(report["model_selection_policy"], "yaml_pool")
        self.assertEqual(report["planned_model"], "Man1")
        self.assertEqual(report["model"], "Woman1")
        self.assertFalse(report["has_pair_review_record"])
        self.assertEqual(event["approved_combinations"], [{"model": "Man1", "speed_mps": 2.5}])

    def test_dynamic_far_future_eta_does_not_spawn_early(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        scenario.traffic_light_stops = []
        with patch.object(scenario, "_spawn_active_dynamic_obstacle") as spawn:
            scenario.update_dynamic_obstacle({"x": 0, "y": 0, "speed_mps": 2.0}, event["route_s"] - 24)
            spawn.assert_not_called()

    def test_dynamic_short_crossing_waits_for_meaningful_eta_after_restart(self):
        scenario, event, ego, actor, _ = self.make_dynamic_spawn_retry_case()
        # Live trial 344: a 4.64s ETA lead let a short right-to-left crossing
        # finish before sufficient Ego response. Do not relax response rules.
        scenario.grpc.spawn_pedestrian.return_value = actor
        self.assertEqual(scenario.cfg["dynamic_obstacle"]["maximum_eta_delta_sec"], 3.0)
        scenario.update_dynamic_obstacle(dict(ego, speed_mps=4.124312), event["route_s"] - 24.954288)
        scenario.grpc.spawn_pedestrian.assert_not_called()
        scenario.update_dynamic_obstacle(dict(ego, speed_mps=5.0), event["route_s"] - 20.0)
        scenario.grpc.spawn_pedestrian.assert_called_once()
        self.assertLessEqual(event["trigger"]["eta_delta_sec"], 3.0)
        self.assertGreaterEqual(event["trigger"]["remaining_m"], event["trigger"]["safe_stop_distance_m"])

    def test_dynamic_corridor_observes_actual_adapter_yaw_field(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        x, y, _ = scenario.mission_route.full_points[900]
        ego = {"x": x, "y": y, "yaw_deg": 90.0, "speed_mps": 5.0}
        dynamic_obstacle._observe_dynamic_zone(scenario, ego, scenario.path_s_at_index(900))
        self.assertEqual(scenario.dynamic_validation["corridor_observations"], 1)
        self.assertGreater(scenario.dynamic_validation["max_ego_body_lateral_extent_m"], 0.9)
        ego["x"] += 1.0
        with self.assertRaisesRegex(RuntimeError, "lane corridor"):
            dynamic_obstacle._observe_dynamic_zone(scenario, ego, scenario.path_s_at_index(900))

    def test_dynamic_goal_cannot_pass_with_unfinished_events(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        with patch.object(scenario, "run_expert_timeline"), self.assertRaisesRegex(RuntimeError, "before every"):
            scenario.run_timeline()
        self.assertFalse(scenario.dynamic_validation["goal_reached"])

    def make_dynamic_collision_case(self, policy="record"):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        self.enable_collection_actor_observation(scenario)
        scenario.collision_policy = policy
        scenario.collision_samples = scenario.collision_events = 0
        scenario.first_collision_receive_ns = None
        scenario.contact_ids = ()
        scenario.collector = Mock()
        scenario.collector.poll.return_value = None
        scenario.rosbag = None
        scenario.last_mode_check = scenario.last_progress = time.monotonic()
        scenario.drive_start_wall = None
        scenario.seconds = 100
        # Isolate policy after completed crossings, still inside the zone.
        scenario.dynamic_obstacle_event_index = len(scenario.dynamic_obstacle_events)
        scenario.dynamic_obstacle_phase = "cleared"
        x, y, _ = scenario.mission_route.full_points[900]
        return scenario, {"x": x, "y": y, "yaw_deg": 90.0, "speed_mps": 5.0,
                          "collision_objects": ["contact-12"]}, scenario.path_s_at_index(900)

    def test_dynamic_collection_record_preserves_supervisor_policy_and_evidence(self):
        scenario, ego, route_s = self.make_dynamic_collision_case()
        # Actual unmodified collection observer/tick; only unrelated zone work
        # and real driving are mocked, not the dynamic collision branch.
        def base_tick(context, elapsed, state, current_s, remaining_s):
            context.update_dynamic_obstacle(state, current_s)
        with patch.object(FullLoopScenario, "on_expert_timeline_tick", autospec=True,
                          side_effect=base_tick):
            scenario.on_expert_timeline_tick(1.0, ego, route_s, 30.0)
            scenario.on_expert_timeline_tick(1.1, dict(ego, collision_objects=[]), route_s, 30.0)
        self.assertEqual(scenario.collision_events, 1)
        collision = next(call for call in scenario.event.call_args_list if call.args[0] == "collision")
        self.assertTrue(collision.kwargs["continued"])
        self.assertEqual(scenario.dynamic_validation["collision_objects"], ["contact-12"])
        with patch.object(scenario, "run_expert_timeline"):
            scenario.run_timeline()
        with patch.object(FullLoopScenario.__bases__[0], "cleanup"), \
                patch.object(scenario, "cleanup_spawned_actors"):
            scenario.cleanup()
        self.assertTrue(scenario.dynamic_validation["goal_reached"])
        self.assertTrue(scenario.dynamic_validation["cleanup_verified"])
        report = json.loads(Path(scenario.profile_state_path()).read_text())
        self.assertFalse(report["dynamic_validation"]["safety_validation_passed"])
        self.assertEqual(report["dynamic_validation"]["collision_objects"], ["contact-12"])

    def test_dynamic_collection_abort_still_raises_in_original_observer(self):
        scenario, ego, route_s = self.make_dynamic_collision_case("abort")
        with patch.object(FullLoopScenario, "on_expert_timeline_tick") as later_tick, \
                self.assertRaisesRegex(RuntimeError, "Expert collision"):
            scenario.on_expert_timeline_tick(1.0, ego, route_s, 30.0)
        later_tick.assert_not_called()
        self.assertFalse(scenario.dynamic_validation["safety_validation_passed"])

    def test_dynamic_review_and_plain_runner_cannot_use_record_to_bypass_collision(self):
        for review in (False, True):
            with self.subTest(review=review):
                scenario, ego, route_s = self.make_dynamic_collision_case()
                if review:
                    scenario.cfg["dynamic_obstacle_review"] = [{"id": "test"}]
                else:
                    scenario.__class__ = MissionScenario  # No collection observer.
                # Missing yaw must not hide a reported collision either.
                del ego["yaw_deg"]
                with self.assertRaisesRegex(RuntimeError, "Collision in dynamic zone"):
                    scenario.update_dynamic_obstacle(ego, route_s)
                self.assertFalse(scenario.dynamic_validation["safety_validation_passed"])

    def test_dynamic_record_does_not_disable_other_safety_guards(self):
        scenario, ego, route_s = self.make_dynamic_collision_case()
        ego["x"] += 1.0
        with self.assertRaisesRegex(RuntimeError, "lane corridor"):
            scenario.update_dynamic_obstacle(ego, route_s)

    def test_dynamic_safety_result_requires_clean_complete_run_and_cleanup(self):
        for failure in (None, "collision_elsewhere", "unfinished", "base_cleanup", "actor_cleanup"):
            with self.subTest(failure=failure):
                scenario, ego, route_s = self.make_dynamic_collision_case()
                scenario.update_dynamic_obstacle(dict(ego, collision_objects=[]), route_s)
                if failure == "collision_elsewhere":
                    scenario.collision_events = 1
                if failure == "unfinished":
                    scenario.dynamic_obstacle_event_index = 0
                with patch.object(scenario, "run_expert_timeline"):
                    if failure == "unfinished":
                        with self.assertRaisesRegex(RuntimeError, "before every"):
                            scenario.run_timeline()
                    else:
                        scenario.run_timeline()
                self.assertFalse(scenario.dynamic_validation["safety_validation_passed"])
                with patch.object(FullLoopScenario.__bases__[0], "cleanup") as base_cleanup, \
                        patch.object(scenario, "cleanup_spawned_actors") as actor_cleanup:
                    if failure in ("base_cleanup", "actor_cleanup"):
                        target = base_cleanup if failure == "base_cleanup" else actor_cleanup
                        target.side_effect = RuntimeError("cleanup probe")
                        with self.assertRaisesRegex(RuntimeError, "cleanup probe"):
                            scenario.cleanup()
                    else:
                        scenario.cleanup()
                self.assertEqual(scenario.dynamic_validation["safety_validation_passed"], failure is None)

    def test_dynamic_timing_room_rejects_impossible_short_recovery_gap(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        self.assertFalse(dynamic_obstacle._timing_room(scenario, event, 2.5, 10.0, scenario.cfg["dynamic_obstacle"]))
        self.assertTrue(dynamic_obstacle._timing_room(scenario, event, 2.5, 50.0, scenario.cfg["dynamic_obstacle"]))

    def test_dynamic_approved_geometry_edit_requires_new_review(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        spec = scenario.route_definition["zones"]["dynamic_obstacle"]
        raw = copy.deepcopy(next(entry for entry in spec["crossing_catalog"] if entry["id"] == "legacy_left_to_right"))
        raw["end"]["x"] += 0.01
        with self.assertRaisesRegex(RuntimeError, "geometry changed"):
            dynamic_obstacle._validate_catalog_entry(scenario, raw, spec, scenario.cfg["dynamic_obstacle"], set())

    def test_dynamic_unavailable_pairs_reduce_schedule_not_crash_selection(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        scenario.cfg["dynamic_obstacle"]["model_selection_policy"] = "approved_pairs"
        # The newly approved routes only have reviewed Man1/Woman1 pairs.
        # An available legacy model must still give a valid single event.
        scenario.dynamic_obstacle_models = ["Dog1"]
        events = dynamic_obstacle._choose_schedule(scenario, scenario.dynamic_obstacle_catalog, scenario.cfg["dynamic_obstacle"])
        self.assertEqual([(entry["id"], entry["model"]) for entry in events], [("legacy_right_to_left", "Dog1")])

    def test_dynamic_signal_query_failure_cannot_become_clear_signal(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        scenario.traffic_light_stops = [{"route_s": 20.0, "light_id": "TL"}]
        scenario._scenario_now = Mock(return_value=100.0)
        scenario.grpc.get_traffic_light_state_by_id.return_value = None
        self.assertIsNone(dynamic_obstacle._red_route_signal_ahead(scenario, 10.0))
        # Cached unknown must remain unknown, not bool(None) == False.
        scenario._scenario_now.return_value = 100.1
        self.assertIsNone(dynamic_obstacle._red_route_signal_ahead(scenario, 10.0))
        scenario.grpc.get_traffic_light_state_by_id.assert_called_once_with("TL")
        scenario._scenario_now.return_value = 100.3
        scenario.grpc.get_traffic_light_state_by_id.return_value = {"id": "TL", "color": 16}
        self.assertIs(dynamic_obstacle._red_route_signal_ahead(scenario, 10.0), False)

    def test_dynamic_signal_query_malformed_or_rpc_error_defers_without_abort(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        scenario.traffic_light_stops = [{"route_s": 20.0, "light_id": "TL"}]
        scenario._scenario_now = Mock(return_value=100.0)
        for response in (None, {"id": "OTHER", "color": 16}, {"id": "TL"},
                         {"id": "TL", "color": "invalid"}, RuntimeError("RPC handler failed")):
            with self.subTest(response=response):
                scenario._dynamic_signal_cache = None
                reader = scenario.grpc.get_traffic_light_state_by_id
                reader.side_effect = response if isinstance(response, Exception) else None
                reader.return_value = response
                self.assertIsNone(dynamic_obstacle._red_route_signal_ahead(scenario, 10.0))

    def test_dynamic_unknown_signal_defers_spawn_until_query_recovers(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        event.update(next(e for e in scenario.dynamic_obstacle_catalog if e["id"] == "legacy_right_to_left"))
        current_s = event["route_s"] - 20.0
        scenario.traffic_light_stops = [{"route_s": current_s + 5.0, "light_id": "TL"}]
        scenario._scenario_now = Mock(return_value=100.0)
        scenario.grpc.get_traffic_light_state_by_id.return_value = None
        ego = {"x": 0.0, "y": 0.0, "speed_mps": 5.0}
        with patch.object(scenario, "_spawn_active_dynamic_obstacle") as spawn:
            scenario.update_dynamic_obstacle(ego, current_s)
            spawn.assert_not_called()
            self.assertEqual(event["phase"], "armed")
            scenario._scenario_now.return_value = 100.3
            scenario.grpc.get_traffic_light_state_by_id.return_value = {"id": "TL", "color": 16}
            scenario.update_dynamic_obstacle(ego, current_s)
            spawn.assert_called_once_with()

    def test_dynamic_unknown_signal_cannot_be_counted_as_ego_response(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        event.update(phase="crossing", trigger={"ego_speed_mps": 5.0})
        scenario.traffic_light_stops = [{"route_s": event["route_s"] + 5.0, "light_id": "TL"}]
        scenario._scenario_now = Mock(return_value=100.0)
        scenario.grpc.get_traffic_light_state_by_id.return_value = None
        dynamic_obstacle._observe_ego_response(scenario, event, {"speed_mps": 0.0}, event["route_s"])
        self.assertFalse(event["response_observed"])
        scenario._scenario_now.return_value = 100.3
        scenario.grpc.get_traffic_light_state_by_id.return_value = {"id": "TL", "color": 16}
        dynamic_obstacle._observe_ego_response(scenario, event, {"speed_mps": 0.0}, event["route_s"])
        # A stop carried over from an unknown/red signal is not NPC response.
        self.assertFalse(event["response_observed"])

    def test_dynamic_post_crossing_signal_failure_recovers_without_resume_failure(self):
        # Reproduce the live failure: NPC finished and Ego is moving, but
        # querying the upcoming signal returns None once before recovering.
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        event.update(phase="cleared", response_observed=True, actor=None,
                     cleared_scenario_time=100.0, resumed_after_clear=False)
        scenario.dynamic_obstacle_events = [event]
        scenario.dynamic_obstacle_event_index = 0
        current_s = event["route_s"] - 5.0
        scenario.traffic_light_stops = [{"route_s": current_s + 5.0, "light_id": "TL"}]
        scenario._scenario_now = Mock(return_value=100.0)
        scenario.grpc.get_traffic_light_state_by_id.return_value = None
        scenario.update_dynamic_obstacle({"speed_mps": 3.0}, current_s)
        self.assertFalse(event["resumed_after_clear"])
        self.assertEqual(event["cleared_scenario_time"], 100.0)
        self.assertEqual(event["phase"], "cleared")
        # A recovered red still defers; only a known clear signal resumes.
        scenario._scenario_now.return_value = 100.3
        scenario.grpc.get_traffic_light_state_by_id.return_value = {"id": "TL", "color": 1}
        scenario.update_dynamic_obstacle({"speed_mps": 3.0}, current_s)
        self.assertFalse(event["resumed_after_clear"])
        scenario._scenario_now.return_value = 100.6
        scenario.grpc.get_traffic_light_state_by_id.return_value = {"id": "TL", "color": 16}
        scenario.update_dynamic_obstacle({"speed_mps": 3.0}, event["route_s"] + 5.0)
        self.assertTrue(event["resumed_after_clear"])
        self.assertEqual(scenario.dynamic_obstacle_event_index, 1)

    def test_dynamic_response_uses_recent_peak_not_only_spawn_speed(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        event.update(trigger={"ego_speed_mps": 2.0}, observed_along_m=1.0)
        scenario.traffic_light_stops = []
        scenario._scenario_now = Mock(return_value=100.0)
        for offset, speed in ((0, 2), (1, 6), (2, 4)):
            scenario._scenario_now.return_value = 100 + offset
            dynamic_obstacle._observe_ego_response(scenario, event, {"speed_mps": speed}, event["route_s"] - 10)
        self.assertTrue(event["response_observed"])
        self.assertEqual(event["response_kind"], "slowdown")

    def test_dynamic_stuck_npc_cannot_earn_response_and_old_peak_expires(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        event.update(trigger={"ego_speed_mps": 7.0}, observed_along_m=0.3)
        scenario.traffic_light_stops = []
        scenario._scenario_now = Mock(return_value=100.0)
        dynamic_obstacle._observe_ego_response(scenario, event, {"speed_mps": 4.0}, event["route_s"])
        self.assertFalse(event["response_observed"])
        event.update(observed_along_m=1.0, response_signal_limited=True)
        scenario._scenario_now.return_value = 106.0
        dynamic_obstacle._observe_ego_response(scenario, event, {"speed_mps": 4.0}, event["route_s"])
        self.assertFalse(event["response_observed"])

    def test_report_policy_does_not_abort_or_approve_unobserved_response(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        scenario.cfg["dynamic_obstacle"]["assessment_policy"] = "report"
        event = scenario.dynamic_obstacle_events[0]
        event.update(response_observed=False, response_signal_limited=True)
        scenario._finish_dynamic_obstacle("reached_dest")
        self.assertEqual(event["phase"], "cleared")
        self.assertEqual(event["assessment_issues"][0]["code"], "response_unverified_signal")
        scenario.dynamic_validation.update(goal_reached=True, corridor_observations=1)
        scenario.dynamic_obstacle_event_index = len(scenario.dynamic_obstacle_events)
        with patch.object(scenario, "cleanup_spawned_actors"), patch.object(scenario, "stop_expert_controller"):
            scenario.cleanup()
        self.assertTrue(scenario.dynamic_validation["cleanup_verified"])
        self.assertFalse(scenario.dynamic_validation["safety_validation_passed"])
        scenario.cfg["dynamic_obstacle_review"] = [{"id": "review"}]
        self.assertTrue(dynamic_obstacle._strict_assessment(scenario))
        scenario.cfg.pop("dynamic_obstacle_review")
        scenario.observe_collision = Mock()
        self.assertTrue(dynamic_obstacle._strict_assessment(scenario))

    def test_report_corridor_warning_retains_collision_abort(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        scenario.cfg["dynamic_obstacle"]["assessment_policy"] = "report"
        point = self.points[900]
        ego = {"x": point[0] + .2, "y": point[1], "yaw_deg": 80.0, "speed_mps": 3.0}
        dynamic_obstacle._observe_dynamic_zone(scenario, ego, scenario.path_s_at_index(900))
        self.assertEqual(scenario.dynamic_validation["assessment_issues"][0]["code"], "conservative_corridor_exceeded")
        with self.assertRaisesRegex(RuntimeError, "Collision in dynamic"):
            dynamic_obstacle._observe_dynamic_zone(scenario, dict(ego, collision_objects=["12"]), scenario.path_s_at_index(900))

    def test_report_unknown_signal_keeps_eta_gates_without_green_assumption(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        scenario.cfg["dynamic_obstacle"]["assessment_policy"] = "report"
        event = scenario.dynamic_obstacle_events[0]
        event.update(next(e for e in scenario.dynamic_obstacle_catalog if e["id"] == "legacy_right_to_left"))
        current_s = event["route_s"] - 20
        scenario.traffic_light_stops = [{"route_s": current_s + 5, "light_id": "TL"}]
        scenario.grpc.get_traffic_light_state_by_id.return_value = None
        with patch.object(scenario, "_spawn_active_dynamic_obstacle") as spawn:
            scenario.update_dynamic_obstacle({"speed_mps": 5.0}, current_s)
            spawn.assert_called_once_with()
            self.assertIsNone(dynamic_obstacle._red_route_signal_ahead(scenario, current_s))
            self.assertFalse(scenario.dynamic_validation["safety_validation_passed"])
            with self.assertRaisesRegex(RuntimeError, "unsafe_trigger_window"):
                scenario.update_dynamic_obstacle({"speed_mps": 5.0}, event["route_s"] - 2)

    def test_dynamic_state_fallback_requires_one_exact_pedestrian(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        actor = Mock()
        actor.get_actor_state.side_effect = RuntimeError("individual RPC failed")
        event.update(actor=actor, actor_id="owned")
        state = SimpleNamespace(actor_info=SimpleNamespace(id=SimpleNamespace(value="owned"), object_type=2))
        scenario.grpc.get_all_pedestrian_states.return_value = [state]
        self.assertIs(dynamic_obstacle._read_event_state(scenario, event), state)
        scenario.grpc.get_all_pedestrian_states.return_value = [state, state]
        self.assertIsNone(dynamic_obstacle._read_event_state(scenario, event))
        self.assertIsNone(dynamic_obstacle._read_event_state(scenario, event))
        with self.assertRaisesRegex(RuntimeError, "after 3 checks"):
            dynamic_obstacle._read_event_state(scenario, event)

    def test_dynamic_state_transient_failure_recovers_without_old_position(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        actor = Mock()
        state = SimpleNamespace(transform=transform(1, 2, 3, 0))
        actor.get_actor_state.side_effect = [None, state]
        event.update(actor=actor, actor_id="owned")
        scenario.grpc.get_all_pedestrian_states.side_effect = RuntimeError("inventory RPC failed")
        self.assertIsNone(dynamic_obstacle._read_event_state(scenario, event))
        self.assertIs(dynamic_obstacle._read_event_state(scenario, event), state)
        self.assertEqual(event["state_read_failures"], 0)

    def test_dynamic_delete_false_ack_needs_two_absence_checks(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        actor = Mock()
        actor.destroy.return_value = False
        event.update(actor=actor, actor_id="owned")
        scenario.scenario_actors.append(actor)
        scenario.grpc.get_all_pedestrian_states.return_value = []
        dynamic_obstacle._destroy_event_actor(scenario, event)
        self.assertIsNone(event["actor"])
        self.assertEqual(scenario.grpc.get_all_pedestrian_states.call_count, 2)
        actor.destroy.assert_called_once_with()

    def test_dynamic_cleanup_does_not_verify_ambiguous_spawn_reset_failure(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        event["spawn_attempts"] = [{"result": "error"}]
        scenario.grpc.reset_actors.return_value = False
        with self.assertRaisesRegex(RuntimeError, "actor reset failed"):
            scenario.cleanup()
        self.assertFalse(scenario.dynamic_validation["cleanup_verified"])

    def test_dynamic_cleanup_continues_after_console_and_actor_errors(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        bad, good = Mock(), Mock()
        bad.destroy.side_effect = RuntimeError("delete failed")
        scenario.scenario_actors = [good, bad]
        scenario.dynamic_obstacle_owned_ids = {"owned"}
        scenario.grpc.get_all_pedestrian_states.return_value = []
        with patch("builtins.print", side_effect=BrokenPipeError("closed pipe")):
            scenario.cleanup()
        good.destroy.assert_called_once_with()
        scenario.grpc.reset_actors.assert_called()
        self.assertTrue(scenario.dynamic_validation["cleanup_verified"])

    def test_dynamic_progress_write_failure_does_not_mask_cleanup_or_original_error(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        scenario.cfg["dynamic_obstacle"]["assessment_policy"] = "report"
        with patch.object(dynamic_obstacle.json, "load", side_effect=OSError("disk error")):
            dynamic_obstacle._persist_progress(scenario)
            self.assertTrue(scenario.dynamic_validation["recording_errors"])
            with self.assertRaisesRegex(RuntimeError, "original failure"):
                dynamic_obstacle._fail_event(scenario, event, "original failure")
        self.assertFalse(scenario.dynamic_validation["safety_validation_passed"])

    def test_dynamic_resume_signal_pause_is_bounded_and_not_an_ego_failure(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        event.update(phase="cleared", cleared_scenario_time=100.0)
        scenario._scenario_now = Mock(return_value=100.0)
        with patch.object(dynamic_obstacle, "_red_route_signal_ahead", return_value=None):
            dynamic_obstacle._advance_after_clear(scenario, event, {"speed_mps": 0}, event["route_s"] - 5)
            scenario._scenario_now.return_value = 161.0
            with self.assertRaisesRegex(RuntimeError, "signal wait exhausted: signal_unknown"):
                dynamic_obstacle._advance_after_clear(scenario, event, {"speed_mps": 0}, event["route_s"] - 5)

    def test_dynamic_resume_warning_grace_and_real_stall_are_distinct(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        scenario.cfg["dynamic_obstacle"]["assessment_policy"] = "report"
        event = scenario.dynamic_obstacle_events[0]
        event.update(phase="cleared", cleared_scenario_time=100.0)
        scenario.traffic_light_stops = []
        scenario._scenario_now = Mock(return_value=121.0)
        dynamic_obstacle._advance_after_clear(scenario, event, {"speed_mps": 0}, event["route_s"] - 5)
        self.assertEqual(event["phase"], "cleared")
        self.assertEqual(event["assessment_issues"][0]["code"], "resume_delayed")
        scenario._scenario_now.return_value = 141.0
        with self.assertRaisesRegex(RuntimeError, "did not resume"):
            dynamic_obstacle._advance_after_clear(scenario, event, {"speed_mps": 0}, event["route_s"] - 5)

    def test_dynamic_same_seed_produces_only_spaced_inset_event_schedules(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        counts = set()
        for seed in range(60):
            scenario.run_seed = seed
            events = dynamic_obstacle._choose_schedule(scenario, scenario.dynamic_obstacle_catalog, scenario.cfg["dynamic_obstacle"])
            counts.add(len(events))
            self.assertTrue(all(event["status"] == "approved" for event in events))
            self.assertTrue(all(b["route_s"] - a["route_s"] >= 35 for a, b in zip(events, events[1:])))
        self.assertEqual(counts, {1, 2})

    def test_dynamic_resumed_flag_cannot_hide_a_later_stall(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        event.update(phase="cleared", cleared_scenario_time=100.0)
        scenario.traffic_light_stops = []
        scenario._scenario_now = Mock(return_value=100.0)
        current_s = event["route_s"] - 5.0
        dynamic_obstacle._advance_after_clear(scenario, event, {"speed_mps": 2}, current_s)
        self.assertTrue(event["resumed_after_clear"])
        scenario._scenario_now.return_value = 121.0
        with self.assertRaisesRegex(RuntimeError, "did not resume"):
            dynamic_obstacle._advance_after_clear(scenario, event, {"speed_mps": 0}, current_s)

    def test_dynamic_slow_real_progress_is_not_a_resume_failure(self):
        scenario = self.make_scenario("full_loop_dynamic")
        self.setup_scenario(scenario)
        event = scenario.dynamic_obstacle_events[0]
        event.update(phase="cleared", cleared_scenario_time=100.0)
        scenario.traffic_light_stops = []
        scenario._scenario_now = Mock(return_value=100.0)
        for step in range(31):
            scenario._scenario_now.return_value = 100.0 + step
            dynamic_obstacle._advance_after_clear(scenario, event, {"speed_mps": 0.2}, event["route_s"] - 10 + step * 0.2)
        self.assertTrue(event["resumed_after_clear"])
        self.assertEqual(event["phase"], "cleared")
        self.assertLess(event["resume_stall_sec"], 4.0)

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

    def test_highway_merge_static_vehicle_is_independent_and_exact(self):
        scenario = self.make_scenario("full_loop_highway")
        original_highway_cfg = copy.deepcopy(scenario.cfg["highway_npc"])
        self.setup_scenario(scenario)

        # The existing native-SpawnPoint configuration and calls are exact.
        self.assertEqual(scenario.cfg["highway_npc"], original_highway_cfg)
        specs = [
            call.args[0]
            for call in scenario.grpc.create_vehicle_spawn_point.call_args_list
        ]
        self.assertEqual(specs, original_highway_cfg["spawn_points"])
        scenario.grpc.spawn_vehicle.assert_not_called()

        activation_s = scenario.highway_activation_s - float(
            original_highway_cfg["background_warmup_distance_m"]
        )
        self.assertFalse(
            highway.activate_highway_spawn_points(
                scenario, 1.0, activation_s - 0.01
            )
        )
        scenario.grpc.spawn_vehicle.assert_not_called()
        scenario.grpc.enable_vehicle_spawn_point.reset_mock()
        scenario.grpc.enable_vehicle_spawn_point.return_value = True
        self.assertTrue(
            highway.activate_highway_spawn_points(
                scenario, 1.0, activation_s
            )
        )
        self.assertEqual(
            scenario.grpc.enable_vehicle_spawn_point.call_count, 3
        )

        cfg = scenario.cfg["highway_merge_static_vehicle"]
        points = scenario.map_loader.get_link_points(cfg["route_link"])
        expected = interpolate_on_polyline(
            points, cfg["distance_from_link_start_m"]
        )
        call = scenario.grpc.spawn_vehicle.call_args.kwargs
        actual_tf = call["transform"]
        self.assertIn(call["model_name"], HIGHWAY_MERGE_STATIC_MODELS)
        self.assertEqual(
            call["model_name"],
            scenario.highway_merge_static_vehicle_info["model"],
        )
        self.assertEqual(call["label"], "highway_merge_static_vehicle")
        self.assertEqual(call["velocity"], 0.0)
        self.assertFalse(call["multi_ego"])
        self.assertTrue(call["recover_ambiguous"])
        self.assertTrue(call["request_id"].startswith(
            "highway_merge_static_vehicle_"
        ))
        self.assertAlmostEqual(actual_tf.location.x, expected[0], places=9)
        self.assertAlmostEqual(actual_tf.location.y, expected[1], places=9)
        self.assertAlmostEqual(actual_tf.location.z, expected[2], places=9)
        self.assertAlmostEqual(actual_tf.rotation.z, expected[3], places=9)
        self.assertAlmostEqual(expected[0], 62.56846034437952, places=9)
        self.assertAlmostEqual(expected[1], 186.49795763948603, places=9)
        self.assertAlmostEqual(expected[3], -89.6381350691572, places=9)

        actor = scenario.highway_merge_static_vehicle
        scenario.grpc.set_vehicle_ai.assert_not_called()
        scenario.grpc.set_vehicle_speed_limit.assert_called_once_with(
            actor, 0.0, enabled=True
        )
        scenario.grpc.set_vehicle_velocity.assert_called_once_with(actor, 0.0)
        route_call = scenario.grpc.set_vehicle_route.call_args
        route_index = scenario.route_links.index("A2256W000411")
        self.assertEqual(route_call.args[0], actor)
        self.assertEqual(route_call.args[1], scenario.route_links[route_index:])
        self.assertEqual(
            route_call.kwargs["route_waypoint_indices"],
            {"A2256W000411": scenario.highway_merge_static_vehicle_info["waypoint"]},
        )
        scenario.grpc.set_vehicle_physics.assert_not_called()
        actor.set_pause.assert_not_called()
        self.assertIn(actor, scenario.scenario_actors)
        self.assertEqual(
            scenario.highway_merge_static_vehicle_info["route_link"],
            "A2256W000411",
        )
        self.assertEqual(
            scenario.highway_merge_static_vehicle_info[
                "left_lane_change_dst_link"
            ],
            "A2256W000409",
        )

    def test_highway_merge_static_models_are_seeded_and_all_reachable(self):
        cfg = {
            "highway_merge_static_vehicle": {
                "models": list(HIGHWAY_MERGE_STATIC_MODELS),
            }
        }
        selected_by_seed = {}
        for seed in range(10000):
            context = SimpleNamespace(
                cfg=cfg,
                run_seed=seed,
                vehicle_models=lambda: list(HIGHWAY_MERGE_STATIC_MODELS),
            )
            candidates = highway._merge_static_vehicle_model_candidates(
                context
            )
            selected_by_seed.setdefault(candidates[0], seed)
            if len(selected_by_seed) == len(HIGHWAY_MERGE_STATIC_MODELS):
                break

        self.assertEqual(set(selected_by_seed), set(HIGHWAY_MERGE_STATIC_MODELS))
        for model, seed in selected_by_seed.items():
            context = SimpleNamespace(
                cfg=cfg,
                run_seed=seed,
                vehicle_models=lambda: list(HIGHWAY_MERGE_STATIC_MODELS),
            )
            first = highway._merge_static_vehicle_model_candidates(context)
            second = highway._merge_static_vehicle_model_candidates(context)
            self.assertEqual(first, second)
            self.assertEqual(first[0], model)

    def test_highway_merge_static_vehicle_retries_next_seeded_model(self):
        scenario = self.make_scenario("full_loop_highway", seed=91)
        self.setup_scenario(scenario)
        expected = highway._merge_static_vehicle_model_candidates(scenario)
        fallback_actor = vehicle(label="highway-merge-fallback")
        scenario.grpc.spawn_vehicle.side_effect = [None, fallback_actor]
        scenario.grpc.enable_vehicle_spawn_point.return_value = True
        activation_s = scenario.highway_activation_s - float(
            scenario.cfg["highway_npc"]["background_warmup_distance_m"]
        )

        with redirect_stdout(io.StringIO()):
            self.assertTrue(
                highway.activate_highway_spawn_points(
                    scenario, 1.0, activation_s
                )
            )

        calls = scenario.grpc.spawn_vehicle.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [call.kwargs["model_name"] for call in calls], expected[:2]
        )
        self.assertEqual(
            scenario.highway_merge_static_vehicle_info["model"], expected[1]
        )
        self.assertEqual(
            scenario.highway_merge_static_vehicle_info["spawn_attempts"],
            [
                {"attempt": 1, "model": expected[0], "spawned": False},
                {"attempt": 2, "model": expected[1], "spawned": True},
            ],
        )

    def test_highway_merge_static_vehicle_fails_closed_without_merge_or_stop(self):
        for mutation, message in (
            (
                lambda cfg: cfg["highway_merge_static_vehicle"].update(
                    route_link="A2256W000846"
                ),
                "legal left lane change",
            ),
            (
                lambda cfg: cfg["highway_merge_static_vehicle"].update(
                    distance_from_link_start_m=5.0
                ),
                "Unsafe highway merge static vehicle offset",
            ),
            (
                lambda cfg: cfg["highway_merge_static_vehicle"].update(
                    models=["ANOTHER_MODEL"]
                ),
                "models are unavailable",
            ),
        ):
            with self.subTest(message=message):
                scenario = self.make_scenario("full_loop_highway")
                mutation(scenario.cfg)
                self.setup_scenario(scenario)
                scenario.grpc.enable_vehicle_spawn_point.return_value = True
                activation_s = scenario.highway_activation_s - 100.0
                with redirect_stdout(io.StringIO()), patch.object(
                    highway.time, "sleep"
                ), self.assertRaisesRegex(RuntimeError, message):
                    highway.activate_highway_spawn_points(
                        scenario, 1.0, activation_s
                    )
                scenario.configure_drive_with_retries.assert_called_once()

        scenario = self.make_scenario("full_loop_highway")
        scenario.grpc.set_vehicle_speed_limit.return_value = False
        actor = vehicle(label="highway_merge_static_vehicle")
        scenario.grpc.spawn_vehicle.side_effect = None
        scenario.grpc.spawn_vehicle.return_value = actor
        self.setup_scenario(scenario)
        scenario.grpc.enable_vehicle_spawn_point.return_value = True
        activation_s = scenario.highway_activation_s - 100.0
        with redirect_stdout(io.StringIO()), patch.object(
            highway.time, "sleep"
        ), self.assertRaisesRegex(RuntimeError, "stationary contract"):
            highway.activate_highway_spawn_points(
                scenario, 1.0, activation_s
            )
        actor.destroy.assert_called()
        scenario.configure_drive_with_retries.assert_called_once()

    def test_highway_activates_100m_early_and_stops_distant_spawners(self):
        scenario = self.make_scenario("full_loop_highway")
        self.setup_scenario(scenario)
        activation_s = scenario.highway_activation_s - 100.0
        self.assertFalse(highway.activate_highway_spawn_points(scenario, 1.0, activation_s - 0.01))
        scenario.grpc.enable_vehicle_spawn_point.reset_mock()
        scenario.grpc.enable_vehicle_spawn_point.return_value = True
        self.assertTrue(highway.activate_highway_spawn_points(scenario, 1.0, activation_s))
        self.assertEqual(scenario.grpc.enable_vehicle_spawn_point.call_count, 3)
        self.assertEqual(scenario.grpc.spawn_vehicle.call_count, 1)
        self.assertTrue(all(item["enabled"] for item in scenario.highway_spawn_points))

        # Repeated timeline ticks do not create duplicate stationary vehicles.
        self.assertTrue(highway.activate_highway_spawn_points(scenario, 1.1, activation_s + 0.1))
        self.assertEqual(scenario.grpc.spawn_vehicle.call_count, 1)

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
                # Dynamic scenes intentionally allow consecutive repeats, and
                # newly reviewed paths may only have one approved model.
                # Different seeds do not guarantee different model names.
                for scene in (a, b):
                    for event in scene.dynamic_obstacle_events:
                        self.assertIn((event["model"], event["speed_mps"]),
                                      dynamic_obstacle._model_speed_pairs(scene, event))
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
