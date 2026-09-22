"""Regression tests for bounded ROS control handoff between campaign laps."""

import unittest

from MoraiDrive.data_collection.internal_sensor_record.async_capture.scenario_simulation import (
    cleanup_unreachable_expert_publishers, wait_for_control_release,
)
from MoraiDrive.scenario_runner.adapters.expert_controller import ExpertController


def state(publishers):
    return (1, "", ([('/ctrl_cmd', list(publishers))],
                     [('/ctrl_cmd', ['/morai_internal_sensor_record_bridge'])], []))


class ControlHandoffTests(unittest.TestCase):
    def test_waits_for_stopped_expert_to_unregister(self):
        observations = [
            state(['/morai_standard_123_456']),
            state(['/morai_standard_123_456']),
            state([]),
        ]

        def read_state():
            return observations.pop(0)

        released = wait_for_control_release(
            read_state, '/ctrl_cmd', timeout=1.0, poll_seconds=0.0)
        self.assertEqual(dict(released[2][0]).get('/ctrl_cmd'), [])
        self.assertEqual(observations, [])

    def test_rejects_a_publisher_that_does_not_unregister(self):
        with self.assertRaisesRegex(RuntimeError, 'did not unregister'):
            wait_for_control_release(
                lambda: state(['/manual_controller']),
                '/ctrl_cmd', timeout=0.0, poll_seconds=0.0)

    def test_propagates_ros_master_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'master unavailable'):
            wait_for_control_release(
                lambda: (0, 'master unavailable', ([], [], [])),
                '/ctrl_cmd', timeout=0.0, poll_seconds=0.0)

    def test_cleanup_removes_only_unreachable_morai_expert(self):
        class Master:
            def getSystemState(self):
                return ([('/ctrl_cmd', [
                    '/morai_standard_321_111',
                    '/morai_standard_999_222',
                    '/manual_controller',
                ])], [], [])

        calls = []
        removed = cleanup_unreachable_expert_publishers(
            '/ctrl_cmd', master=Master(),
            ping=lambda node: node.endswith('_222'),
            cleanup=lambda master, nodes: calls.append(list(nodes)),
        )
        self.assertEqual(removed, ['/morai_standard_321_111'])
        self.assertEqual(calls, [['/morai_standard_321_111']])

    def test_stale_cleanup_targets_only_the_stopped_expert_pid(self):
        system_state = (
            [('/ctrl_cmd', [
                '/morai_standard_321_111',
                '/morai_standard_999_222',
                '/manual_controller',
            ])],
            [('/Ego_topic', ['/morai_standard_321_111'])],
            [],
        )
        self.assertEqual(
            ExpertController._owned_ros_nodes(system_state, 321),
            ['/morai_standard_321_111'],
        )


if __name__ == '__main__':
    unittest.main()
