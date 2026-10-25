"""Controller-free tests for collision event detection and JSONL output."""

import json
import tempfile
import unittest
from pathlib import Path

from ..validation import EgoObservation, EventMonitor, JsonlEventWriter, ValidationEvent


def observation(timestamp, collisions, run_id="run-1"):
    return EgoObservation(
        run_id=run_id,
        timestamp=timestamp,
        position_xyz=(1.0, 2.0, 0.0),
        speed_mps=3.0,
        collision_objects=collisions,
    )


class EventMonitorTests(unittest.TestCase):
    def test_collision_free_drive_emits_no_events(self):
        monitor = EventMonitor()
        self.assertEqual(monitor.observe(observation(0.0, ())), [])
        self.assertEqual(monitor.observe(observation(1.0, ())), [])

    def test_collision_starts_once_and_clears_once(self):
        monitor = EventMonitor()
        self.assertEqual(monitor.observe(observation(0.0, ())), [])
        self.assertEqual(
            monitor.observe(observation(1.0, ("npc-7",))),
            [ValidationEvent(1.0, "collision_started", "run-1", "npc-7")],
        )
        self.assertEqual(monitor.observe(observation(2.0, ("npc-7",))), [])
        self.assertEqual(
            monitor.observe(observation(3.0, ())),
            [ValidationEvent(3.0, "collision_cleared", "run-1", "npc-7")],
        )
        self.assertEqual(monitor.observe(observation(4.0, ())), [])

    def test_missing_observation_does_not_clear_collision(self):
        monitor = EventMonitor()
        monitor.observe(observation(1.0, ("npc-7",)))
        self.assertEqual(monitor.observe(None), [])
        self.assertEqual(monitor.observe(observation(2.0, None)), [])
        self.assertEqual(monitor.observe(observation(3.0, ("npc-7",))), [])
        self.assertEqual(
            monitor.observe(observation(4.0, ())),
            [ValidationEvent(4.0, "collision_cleared", "run-1", "npc-7")],
        )

    def test_actor_and_run_states_are_independent(self):
        monitor = EventMonitor()
        self.assertEqual(
            monitor.observe(observation(1.0, ("a", "b", "a"))),
            [ValidationEvent(1.0, "collision_started", "run-1", actor) for actor in ("a", "b")],
        )
        self.assertEqual(
            monitor.observe(observation(2.0, ("a", "c"))),
            [
                ValidationEvent(2.0, "collision_cleared", "run-1", "b"),
                ValidationEvent(2.0, "collision_started", "run-1", "c"),
            ],
        )
        self.assertEqual(monitor.observe(observation(3.0, (), run_id="run-2")), [])
        self.assertEqual(monitor.observe(observation(4.0, ("a", "c"))), [])

    def test_jsonl_events_can_be_read_back(self):
        monitor = EventMonitor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            writer = JsonlEventWriter(path)
            for sample in (observation(1.0, ("npc-7",)), observation(2.0, ("npc-7",)),
                           observation(3.0, ())):
                for event in monitor.observe(sample):
                    writer.write(event)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(rows, [
            {"timestamp": 1.0, "event_type": "collision_started", "run_id": "run-1", "actor_id": "npc-7"},
            {"timestamp": 3.0, "event_type": "collision_cleared", "run_id": "run-1", "actor_id": "npc-7"},
        ])


if __name__ == "__main__":
    unittest.main()
