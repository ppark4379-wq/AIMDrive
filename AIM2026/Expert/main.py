#!/usr/bin/env python3
"""Planning/control entry. AIM2026 owns inference, coordinates and UDP.

Call on_control_tick with already adapted Expert interface objects.
The CLI delegates unified UDP execution to AIM2026.
"""
import os
from pathlib import Path
import sys

# One BLAS thread before NumPy loads: the planner's many small solves made OpenBLAS spin all cores (about 870 % CPU),
# which stalled this control loop and the VE inference process for up to 1.4 s. Inference gets its own settings back.
BLAS_THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS')
os.environ['AIM_EXPERT_BLAS_CAPPED'] = ','.join(name for name in BLAS_THREAD_VARS if name not in os.environ)
for _name in BLAS_THREAD_VARS:
    os.environ.setdefault(_name, '1')


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
