"""The Expert must use the same ROS message build as the MORAI bridge."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ..adapters.expert_process import configure_ros_message_path


class ExpertMessagePathTest(unittest.TestCase):
    def test_bundled_messages_override_inherited_old_overlay(self):
        with tempfile.TemporaryDirectory() as root:
            message_path = Path(root).resolve() / "devel/lib/python3/dist-packages"
            package = message_path / "morai_msgs/msg"
            package.mkdir(parents=True)
            (package / "__init__.py").touch()
            inherited = ["/home/seoyeon/morai_ws/devel/lib/python3/dist-packages",
                         str(message_path), "/opt/ros/noetic/lib/python3/dist-packages"]
            with patch.object(sys, "path", inherited.copy()):
                self.assertEqual(configure_ros_message_path(root), message_path)
                configure_ros_message_path(root)
                self.assertEqual(sys.path[0], str(message_path))
                self.assertEqual(sys.path.count(str(message_path)), 1)
                self.assertIn(inherited[0], sys.path)

    def test_external_expert_without_message_build_keeps_environment(self):
        with tempfile.TemporaryDirectory() as root:
            inherited = sys.path.copy()
            with patch.object(sys, "path", inherited.copy()):
                self.assertIsNone(configure_ros_message_path(root))
                self.assertEqual(sys.path, inherited)


if __name__ == "__main__":
    unittest.main()
