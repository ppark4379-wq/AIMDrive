#!/usr/bin/env python3
"""Planning/control entry. AIM2026 owns inference, coordinates and UDP.

Call on_control_tick with already adapted Expert interface objects.
The CLI delegates unified UDP execution to AIM2026.
"""
from pathlib import Path
import sys


EXPERT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERT_ROOT.parents[1]))


def on_control_tick(driving, vehicle, objects, traffic, **timing):
    """Use the latest localized state; preserve planning then control order."""
    plan = driving.planning.run(vehicle, objects, traffic, **timing)
    return driving.control.run(vehicle, plan)


def main(argv=None):
    from AIM2026.Expert.runtime import main as run_aim
    return run_aim(argv, control_tick=on_control_tick)


if __name__ == '__main__':
    raise SystemExit(main())
