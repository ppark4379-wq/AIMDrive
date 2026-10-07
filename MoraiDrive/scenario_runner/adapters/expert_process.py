#!/usr/bin/env python3
"""Import and execute the external AIMDrive expert from scenario_runner."""

import os
from pathlib import Path
import sys


def configure_ros_message_path(expert_root):
    """Use the message build shipped with this Expert, before inherited overlays."""
    message_path = Path(expert_root).resolve() / "devel/lib/python3/dist-packages"
    if not (message_path / "morai_msgs/msg/__init__.py").is_file():
        return None
    message_path_string = str(message_path)
    # An older ~/morai_ws overlay has a different EgoVehicleStatus MD5 and
    # silently prevents the Expert from receiving the bridge's /Ego_topic.
    if message_path_string in sys.path:
        sys.path.remove(message_path_string)
    sys.path.insert(0, message_path_string)
    return message_path


def main():
    expert_root = os.environ.get("MORAI_EXPERT_ROOT")
    if not expert_root:
        raise RuntimeError("MORAI_EXPERT_ROOT is not configured")

    configure_ros_message_path(expert_root)
    expert_source = os.path.join(os.path.abspath(expert_root), "src")
    if expert_source not in sys.path:
        sys.path.insert(0, expert_source)

    from expert.src.autonomous_driving import AutonomousDriving
    from expert.src.localization.path_manager import PathManager
    from expert.src.network.ros_manager import RosManager
    import morai_msgs.msg
    from morai_msgs.msg import CtrlCmd, EgoVehicleStatus

    root = Path(expert_root).resolve()
    modules = {
        "autonomous_driving": Path(
            sys.modules[AutonomousDriving.__module__].__file__
        ).resolve(),
        "path_manager": Path(
            sys.modules[PathManager.__module__].__file__
        ).resolve(),
        "ros_manager": Path(
            sys.modules[RosManager.__module__].__file__
        ).resolve(),
    }
    for name, module_path in modules.items():
        try:
            module_path.relative_to(root)
        except ValueError:
            raise RuntimeError(
                f"External Expert import escaped configured root: "
                f"{name}={module_path}, root={root}"
            )

    print(f"[Expert] mode={os.environ.get('MORAI_EXPERT_MODE', 'unknown')}", flush=True)
    print(f"[Expert] root={root}", flush=True)
    for name, module_path in modules.items():
        print(f"[Expert] {name} module={module_path}", flush=True)
    print(f"[Expert] morai_msgs={morai_msgs.msg.__file__}", flush=True)
    print(
        f"[Expert] ROS message MD5: EgoVehicleStatus={EgoVehicleStatus._md5sum} "
        f"CtrlCmd={CtrlCmd._md5sum}",
        flush=True,
    )

    RosManager(AutonomousDriving()).execute()


if __name__ == "__main__":
    main()
