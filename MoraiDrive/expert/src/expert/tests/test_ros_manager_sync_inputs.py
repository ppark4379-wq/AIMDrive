import unittest
from collections import deque

from expert.src.network.synchronized_inputs import (
    control_dt,
    select_objects,
    select_traffic_light,
)


class RosManagerSyncInputTest(unittest.TestCase):
    def setUp(self):
        self.object_history = deque(maxlen=200)
        self.traffic_light_history = deque(maxlen=200)

    def test_sync_object_selection_uses_matching_source_time(self):
        self.object_history.extend(
            [
                {"source_time": 9.8, "arrival_time": 1.0, "objects": ["old"]},
                {"source_time": 10.02, "arrival_time": 2.0, "objects": ["same"]},
            ]
        )

        objects, delta = select_objects(self.object_history, True, 10.0, 0.05)

        self.assertEqual(["same"], objects)
        self.assertAlmostEqual(0.02, delta)

    def test_sync_object_selection_does_not_reuse_out_of_window_frame(self):
        self.object_history.append(
            {"source_time": 9.8, "arrival_time": 1.0, "objects": ["stale"]}
        )

        objects, delta = select_objects(self.object_history, True, 10.0, 0.05)

        self.assertEqual([], objects)
        self.assertAlmostEqual(0.2, delta)

    def test_zero_stamp_object_falls_back_to_latest_arrival(self):
        self.object_history.append(
            {"source_time": None, "arrival_time": 1.0, "objects": ["latest"]}
        )

        objects, delta = select_objects(self.object_history, True, 10.0, 0.05)

        self.assertEqual(["latest"], objects)
        self.assertIsNone(delta)

    def test_sync_traffic_selection_carries_simulation_age(self):
        traffic = ["signal", 1, 2, 100.0]
        self.traffic_light_history.append(
            {
                "source_time": 10.0,
                "arrival_time": 100.0,
                "traffic_light": traffic,
            }
        )

        selected, age = select_traffic_light(
            self.traffic_light_history, True, 10.5, 1.0, monotonic_now=100.0
        )

        self.assertEqual(traffic + [10.0, 10.5], selected)
        self.assertEqual(0.5, age)

    def test_sync_traffic_selection_rejects_expired_simulation_state(self):
        self.traffic_light_history.append(
            {
                "source_time": 10.0,
                "arrival_time": 100.0,
                "traffic_light": ["signal", 1, 2, 100.0],
            }
        )

        selected, age = select_traffic_light(
            self.traffic_light_history, True, 11.1, 1.0, monotonic_now=100.0
        )

        self.assertEqual([], selected)
        self.assertAlmostEqual(1.1, age)

    def test_sync_control_dt_comes_from_ego_source_time(self):
        first_dt, previous = control_dt(True, 20.0, None, 30.0, 10.0)
        second_dt, previous = control_dt(True, 20.02, previous, 30.0, 10.0)

        self.assertEqual(0.1, first_dt)
        self.assertAlmostEqual(0.02, second_dt)

    def test_realtime_control_dt_stays_at_configured_rate(self):
        selected_dt, _ = control_dt(False, 20.0, None, 30.0, 10.0)
        self.assertAlmostEqual(1.0 / 30.0, selected_dt)


if __name__ == "__main__":
    unittest.main()
