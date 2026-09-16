#!/usr/bin/env python3
"""Import and execute the external AIMDrive expert from scenario_runner."""

import os
from pathlib import Path
import sys


def main():
    expert_root = os.environ.get("MORAI_EXPERT_ROOT")
    if not expert_root:
        raise RuntimeError("MORAI_EXPERT_ROOT is not configured")

    expert_source = os.path.join(os.path.abspath(expert_root), "src")
    if expert_source not in sys.path:
        sys.path.insert(0, expert_source)

    from expert.src.autonomous_driving import AutonomousDriving
    from expert.src.localization.path_manager import PathManager
    from expert.src.network.ros_manager import RosManager

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

    RosManager(AutonomousDriving()).execute()


if __name__ == "__main__":
    main()
