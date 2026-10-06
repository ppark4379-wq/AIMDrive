"""Tunnel dynamic obstacle setup and lifecycle tests."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from ..scenarios.full_loop_parts.tunnel_dynamic import (
    arm_tunnel_dynamic,
    prepare_tunnel_dynamic,
    update_tunnel_dynamic,
)


class TunnelDynamicTest(unittest.TestCase):
    def test_enabled_crossing_requires_validated_handler(self):
        context = SimpleNamespace(cfg={"tunnel_dynamic": {"enabled": True}})
        with self.assertRaisesRegex(RuntimeError, "validated placement and lifecycle handler"):
            prepare_tunnel_dynamic(context)
        self.assertIsNone(context.tunnel_dynamic_handler)

    def test_validated_handler_receives_lifecycle_calls(self):
        context = SimpleNamespace(cfg={"tunnel_dynamic": {"enabled": True}})
        handler = Mock()
        self.assertTrue(prepare_tunnel_dynamic(context, handler))
        handler.validate.assert_called_once_with(context)
        arm_tunnel_dynamic(context)
        update_tunnel_dynamic(context, 1.0, {"speed": 5.0}, 35.0)
        handler.arm.assert_called_once_with(context)
        handler.update.assert_called_once_with(context, 1.0, {"speed": 5.0}, 35.0)

    def test_disabled_crossing_does_not_touch_handler(self):
        context = SimpleNamespace(cfg={"tunnel_dynamic": {"enabled": False}})
        handler = Mock()
        self.assertFalse(prepare_tunnel_dynamic(context, handler))
        self.assertIsNone(context.tunnel_dynamic_handler)
        handler.validate.assert_not_called()

    def test_clean_road_cannot_enable_crossing(self):
        context = SimpleNamespace(
            cfg={"tunnel_dynamic": {"enabled": True}}, clean_road_mode=True
        )
        with self.assertRaisesRegex(ValueError, "clean_road_mode"):
            prepare_tunnel_dynamic(context)


if __name__ == "__main__":
    unittest.main()
