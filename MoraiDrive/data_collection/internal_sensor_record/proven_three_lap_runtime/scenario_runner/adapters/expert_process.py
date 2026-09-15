#!/usr/bin/env python3
"""Import and execute the external AIMDrive expert from scenario_runner."""

import os
import sys


def main():
    expert_root = os.environ.get("MORAI_EXPERT_ROOT")
    if not expert_root:
        raise RuntimeError("MORAI_EXPERT_ROOT is not configured")

    expert_source = os.path.join(os.path.abspath(expert_root), "src")
    if expert_source not in sys.path:
        sys.path.insert(0, expert_source)

    from expert.src.autonomous_driving import AutonomousDriving
    from expert.src.network.ros_manager import RosManager

    RosManager(AutonomousDriving()).execute()


if __name__ == "__main__":
    main()
