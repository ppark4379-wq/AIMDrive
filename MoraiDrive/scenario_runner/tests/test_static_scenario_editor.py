import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from ..tools.static_scenario_editor import (
    ExpertCommandRelay,
    LiveState,
    MoraiObstacleReader,
)

ROOT = Path(__file__).resolve().parents[2]


class StaticScenarioEditorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = LiveState(self.tmp.name)
        self.state.models = ["CargoBox", "WoodBox"]
        self.state.model_scales = {"CargoBox": [2.0, 3.0, 2.0]}
        self.state.default_scale = [1.0, 1.0, 1.0]
        self.state.global_path = [[float(index), 0.0] for index in range(21)]
        self.state.update_obstacles([{
            "actor_id": "obstacle-1",
            "model": "CargoBox",
            "x": 8.2,
            "y": 0.4,
            "z": 2.0,
            "yaw_deg": 15.0,
            "width_m": 2.0,
            "length_m": 3.0,
        }])
        self.mark_passed()

    def mark_passed(self, trial=1):
        self.state.on_trial_event({
            "event": "trial_finished",
            "trial": trial,
            "outcome": "passed",
            "reason": "goal_reached",
            "collisions": [],
        })
        self.state.on_trial_event({
            "event": "trial_ready",
            "trial": trial,
            "previous_outcome": "passed",
        })

    def test_save_uses_current_transform_and_nearest_route_index(self):
        result = self.state.save({
            "name": "layout_01",
            "obstacles": [{"actor_id": "obstacle-1", "model": "CargoBox"}],
        })
        output = Path(result["saved"])
        document = yaml.safe_load(output.read_text(encoding="utf-8"))
        self.assertEqual(document["schema"], "morai_static_obstacle_preset_v1")
        obstacle = document["obstacles"][0]
        self.assertEqual(obstacle["position"]["x"], 8.2)
        self.assertEqual(obstacle["route_path_index"], 8)
        self.assertEqual(obstacle["scale"], [2.0, 3.0, 2.0])

    def test_physical_box_rotation_is_separate_from_replay_actor_heading(self):
        vector = lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)
        actor = SimpleNamespace(
            actor_info=SimpleNamespace(id=SimpleNamespace(value="barrier-1")),
            transform=SimpleNamespace(
                location=vector(8.2, 0.4, 2.0),
                rotation=vector(0.0, 0.0, 15.0),
            ),
            bounding_box=SimpleNamespace(
                location=vector(8.5, 0.3, 2.5),
                rotation=vector(0.0, 0.0, 105.0),
                extent=vector(0.4, 1.8, 0.5),
            ),
        )
        payload = MoraiObstacleReader._obstacle_payload(actor, "TrafficBarrier")
        self.assertEqual(payload["yaw_deg"], 15.0)
        self.assertEqual(payload["footprint_yaw_deg"], 105.0)
        self.assertEqual(payload["width_m"], 0.8)
        self.assertEqual(payload["length_m"], 3.6)

        payload["model"] = "CargoBox"
        self.state.update_obstacles([payload])
        self.mark_passed()
        saved = self.state.save({
            "name": "physical_box_test",
            "obstacles": [{"actor_id": "barrier-1", "model": "CargoBox"}],
        })
        document = yaml.safe_load(Path(saved["saved"]).read_text(encoding="utf-8"))
        self.assertEqual(document["obstacles"][0]["yaw_deg"], 15.0)
        self.assertEqual(document["obstacles"][0]["position"]["x"], 8.2)

    def test_sensor_record_collection_forwards_static_preset(self):
        tools = ROOT / "data_collection" / "internal_sensor_record" / "tools"
        campaign = (tools / "collect_scenario_laps.sh").read_text(encoding="utf-8")
        capture = (tools / "run_sensor_record_scenario.py").read_text(encoding="utf-8")
        runner = (tools / "run_proven_three_lap_scenario.py").read_text(encoding="utf-8")
        for source in (campaign, capture, runner):
            self.assertIn("--static-preset", source)
        self.assertIn("cfg['static_obstacle_preset'] = a.static_preset", runner)
        self.assertIn("static_obstacle_preset_request", campaign)

    def test_existing_preset_requires_explicit_overwrite(self):
        request = {
            "name": "layout_01",
            "obstacles": [{"actor_id": "obstacle-1", "model": "CargoBox"}],
        }
        self.state.save(request)
        with self.assertRaises(FileExistsError):
            self.state.save(request)
        request["overwrite"] = True
        self.state.save(request)

    def test_only_the_exact_collision_free_completed_layout_can_be_saved(self):
        self.state.layout_verified = False
        self.state.trial_result = None
        with self.assertRaisesRegex(ValueError, "Run and pass"):
            self.state.save({
                "name": "not_run",
                "obstacles": [{"actor_id": "obstacle-1", "model": "CargoBox"}],
            })

        self.mark_passed(trial=2)
        self.assertTrue(self.state.snapshot()["layout_verified"])
        moved = dict(self.state.obstacles[0])
        moved["x"] += 0.5
        self.state.update_obstacles([moved])
        snapshot = self.state.snapshot()
        self.assertFalse(snapshot["layout_verified"])
        self.assertEqual(snapshot["trial_status"], "editing")

    def test_trial_finished_brakes_relay_and_preserves_pass_result(self):
        calls = []
        self.state.attach_drive_handler(calls.append)
        self.state.drive_enabled = True
        self.state.drive_session_recording = True
        self.state.on_trial_event({
            "event": "trial_finished",
            "trial": 3,
            "outcome": "passed",
            "reason": "goal_reached",
            "collisions": [],
        })
        snapshot = self.state.snapshot()
        self.assertEqual(calls, [False])
        self.assertFalse(snapshot["drive_enabled"])
        self.assertEqual(snapshot["trial_number"], 3)
        self.assertEqual(snapshot["trial_status"], "passed")
        self.assertTrue(snapshot["layout_verified"])

    def test_trial_reset_discards_stale_path_and_control_readiness(self):
        now = __import__("time").time()
        self.state.local_path = [[0.0, 0.0], [1.0, 0.0]]
        self.state.candidates = [{"selected": True, "valid": True}]
        self.state.expert_objects = [{"x": 1.0}]
        self.state.local_updated_at = now
        self.state.lattice_updated_at = now
        self.state.expert_objects_updated_at = now
        self.state.control_updated_at = now

        self.state.on_trial_event({"event": "trial_resetting", "trial": 1})
        snapshot = self.state.snapshot()
        self.assertEqual(snapshot["local_path"], [])
        self.assertEqual(snapshot["candidates"], [])
        self.assertEqual(snapshot["expert_objects"], [])
        self.assertIsNone(snapshot["local_age_sec"])
        self.assertIsNone(snapshot["control_age_sec"])

    def test_disappeared_or_unknown_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disappeared"):
            self.state.save({
                "name": "gone",
                "obstacles": [{"actor_id": "other", "model": "CargoBox"}],
            })
        with self.assertRaisesRegex(ValueError, "allowed replay model"):
            self.state.save({
                "name": "unknown",
                "obstacles": [{"actor_id": "obstacle-1", "model": "Tree01"}],
            })

    def test_drive_gate_requires_fresh_valid_plan_and_records_actual_trace(self):
        calls = []
        self.state.attach_drive_handler(calls.append)
        with self.assertRaisesRegex(ValueError, "local_path"):
            self.state.set_drive(True)

        now = __import__("time").time()
        self.state.local_path = [[0.0, 0.0], [1.0, 0.0]]
        self.state.local_updated_at = now
        self.state.lattice_updated_at = now
        self.state.control_updated_at = now
        self.state.candidates = [{"selected": True, "valid": True, "points": []}]
        self.state.ego = {"x": 0.0, "y": 0.0}
        self.state.update_collisions(["1"])
        snapshot = self.state.set_drive(True)
        self.assertTrue(snapshot["drive_enabled"])
        self.assertEqual(snapshot["drive_baseline_collision_objects"], ["1"])
        self.assertEqual(calls, [True])

        self.state.update_collisions(["1"])
        self.assertEqual(self.state.snapshot()["new_collision_objects"], [])
        self.state.update_collisions([])
        self.state.update_collisions(["1", "2"])
        self.assertEqual(
            self.state.snapshot()["new_collision_objects"], ["1", "2"]
        )

        ego_marker = SimpleNamespace(
            ns="ego_shape", action=0, DELETE=2, text="", points=[],
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.4, y=0.2),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        )
        self.state.on_lattice_paths(SimpleNamespace(markers=[ego_marker]))
        self.assertEqual(self.state.snapshot()["ego_trace"][-1], [0.4, 0.2])
        self.state.set_drive(False)
        self.assertEqual(calls, [True, False])

    def test_command_relay_brakes_while_stopped_and_forwards_while_enabled(self):
        class Control:
            def __init__(self):
                self.longlCmdType = 0
                self.accel = 0.0
                self.brake = 0.0
                self.front_steer = 0.0

        publisher = SimpleNamespace(messages=[])
        publisher.publish = publisher.messages.append
        mode_publisher = SimpleNamespace(messages=[])
        mode_publisher.publish = mode_publisher.messages.append
        relay = ExpertCommandRelay(
            self.state, publisher, Control, mode_publisher=mode_publisher
        )
        self.assertEqual(mode_publisher.messages, [False])
        command = Control()
        command.accel = 0.7
        relay.on_command(command)
        self.assertEqual(publisher.messages[-1].brake, 1.0)
        relay.set_enabled(True)
        self.assertEqual(mode_publisher.messages[-1], True)
        relay.on_command(command)
        self.assertIs(publisher.messages[-1], command)
        relay.set_enabled(False)
        self.assertEqual(mode_publisher.messages[-1], False)
        self.assertEqual(publisher.messages[-1].brake, 1.0)

    def test_reader_reports_ego_collision_ids(self):
        reader = MoraiObstacleReader.__new__(MoraiObstacleReader)
        reader.filter_type = lambda **kwargs: kwargs
        reader.stub = SimpleNamespace(GetAllActorsState=lambda request, timeout: SimpleNamespace(
            states=[SimpleNamespace(
                actor_info=SimpleNamespace(id=SimpleNamespace(value="Ego")),
                vehicle_state=SimpleNamespace(collision_objects=["1", ""]),
            )]
        ))
        self.assertEqual(reader.read_ego_collisions(), ["1", ""])

    def test_reader_refreshes_model_when_morai_reuses_actor_id(self):
        names = iter(("WoodBox", "Bicycle_mountain"))
        reader = MoraiObstacleReader.__new__(MoraiObstacleReader)
        reader.model_cache = {}
        reader.stub = SimpleNamespace(
            GetOptionName=lambda request, timeout: SimpleNamespace(
                value=next(names)
            )
        )
        actor_info = SimpleNamespace(id=SimpleNamespace(value="41"))
        self.assertEqual(reader._option_name(actor_info), "WoodBox")
        self.assertEqual(reader._option_name(actor_info), "Bicycle_mountain")
