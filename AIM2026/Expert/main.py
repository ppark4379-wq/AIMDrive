#!/usr/bin/env python3
"""Planning/control entry. AIM2026 owns inference, coordinates and UDP.

Call on_control_tick with already adapted Expert interface objects.
The CLI delegates unified execution to AIM2026. Legacy ROS is explicit.
"""
import argparse
from pathlib import Path
import sys


EXPERT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERT_ROOT / 'src'))


def on_control_tick(driving, vehicle, objects, traffic, **timing):
    """Use the latest localized state; preserve planning then control order."""
    plan = driving.planning.run(vehicle, objects, traffic, **timing)
    return driving.control.run(vehicle, plan)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--transport', choices=('ros',), default=None)
    entry, runtime_arguments = parser.parse_known_args(argv)
    if entry.transport == 'ros':
        from expert.src.runtime.ros_runtime import main as run_ros
        return run_ros(runtime_arguments)

    sys.path.insert(0, str(EXPERT_ROOT.parents[1]))
    from AIM2026.Expert.runtime import main as run_aim
    return run_aim(runtime_arguments, control_tick=on_control_tick)


if __name__ == '__main__':
    raise SystemExit(main())
