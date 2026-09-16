import hashlib
import json
from pathlib import Path
import struct
import sys
import types
import unittest
from unittest.mock import patch


DATA_COLLECTION = Path(__file__).resolve().parents[1]
INTERNAL = DATA_COLLECTION / "internal_sensor_record"
sys.path.insert(0, str(DATA_COLLECTION))
sys.path.insert(0, str(INTERNAL))

from async_capture.live import RawSubscriptions


BRIDGE_PREFLIGHT = types.ModuleType("ros.bridge_preflight")
BRIDGE_PREFLIGHT.validate_bridge = lambda *args, **kwargs: []


class RejectUnexpectedWrite:
    def submit(self, metadata, payload):
        raise AssertionError("malformed callback must not reach the raw writer")


class HeaderMessage:
    _md5sum = "test-md5"
    _has_header = True
    _full_text = "Header header"

    def deserialize(self, payload):
        return self


class AnyMessage:
    def __init__(self, payload):
        self._buff = payload
        self._connection_header = {
            "type": "test_msgs/HeaderMessage",
            "md5sum": "test-md5",
            "callerid": "/test_publisher",
            "topic": "/Ego_topic",
            "latching": "0",
        }


class RawSubscriptionDiagnosticTest(unittest.TestCase):
    def subscriptions(self):
        return RawSubscriptions(RejectUnexpectedWrite(), {
            "capture": {
                "record_ros_cameras": False,
                "ros_camera_transport_policy": "bridge_dropped",
                "camera_frame_ids": [],
            },
        }, time_mode="simulation_free_run")

    def test_malformed_header_records_bounded_structured_context(self):
        payload = struct.pack("<IIII", 7, 11, 1_000_000_000, 0)
        subscriptions = self.subscriptions()
        with patch("roslib.message.get_message_class", return_value=HeaderMessage):
            subscriptions._receive(
                AnyMessage(payload), ("/Ego_topic", "ego"), 123, 456,
                {"monotonic_raw_ns": 120, "raw_bracket_ns": 6},
            )

        self.assertEqual(
            subscriptions.errors, ["ValueError: Malformed ROS header"])
        with patch.dict(
                sys.modules, {"ros.bridge_preflight": BRIDGE_PREFLIGHT}):
            diagnostic = subscriptions.readiness()["callback_errors"][0]
        self.assertEqual(diagnostic["topic"], "/Ego_topic")
        self.assertEqual(diagnostic["role"], "ego")
        self.assertEqual(diagnostic["ros_type"], "test_msgs/HeaderMessage")
        self.assertEqual(diagnostic["caller_id"], "/test_publisher")
        self.assertEqual(diagnostic["payload_bytes"], len(payload))
        self.assertEqual(
            diagnostic["payload_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(diagnostic["payload_prefix_hex"], payload.hex())
        self.assertEqual(diagnostic["header_preview"]["seq"], 7)
        self.assertEqual(diagnostic["header_preview"]["nsec"], 1_000_000_000)
        self.assertFalse(diagnostic["header_preview"]["nsec_in_range"])
        self.assertEqual(
            diagnostic["ros_deserialization"], {"status": "succeeded"})
        json.dumps(diagnostic)

    def test_callback_diagnostics_are_bounded_without_weakening_failures(self):
        payload = struct.pack("<IIII", 1, 2, 3, 70_000)
        subscriptions = self.subscriptions()
        with patch("roslib.message.get_message_class", return_value=HeaderMessage):
            for index in range(40):
                subscriptions._receive(
                    AnyMessage(payload), ("/local_path", "expert_local_path"),
                    index, index, None,
                )

        with patch.dict(
                sys.modules, {"ros.bridge_preflight": BRIDGE_PREFLIGHT}):
            state = subscriptions.readiness()
        self.assertEqual(len(subscriptions.errors), 40)
        self.assertEqual(len(state["callback_errors"]), 32)
        self.assertIn("ValueError: Malformed ROS header", state["failures"])
        self.assertFalse(state["ready"])

    def test_diagnostic_failure_cannot_mask_original_capture_failure(self):
        payload = struct.pack("<IIII", 1, 2, 1_000_000_000, 0)
        subscriptions = self.subscriptions()
        with patch("roslib.message.get_message_class", return_value=HeaderMessage):
            with patch.object(
                    subscriptions, "_callback_error_diagnostic",
                    side_effect=RuntimeError("diagnostic broke")):
                subscriptions._receive(
                    AnyMessage(payload), ("/imu", "imu"), 1, 2, None)

        self.assertEqual(
            subscriptions.errors, ["ValueError: Malformed ROS header"])
        diagnostic = subscriptions.callback_errors[0]
        self.assertEqual(diagnostic["topic"], "/imu")
        self.assertEqual(
            diagnostic["diagnostic_error"], "RuntimeError: diagnostic broke")


if __name__ == "__main__":
    unittest.main()
