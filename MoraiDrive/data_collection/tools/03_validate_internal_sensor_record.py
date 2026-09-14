#!/usr/bin/env python3
"""Run the fail-closed offline validator for an internal Sensor Record campaign."""
import os
from pathlib import Path
import subprocess
import sys


TOOLS = Path(__file__).resolve().parent
DATA_COLLECTION = TOOLS.parent
PROJECT_ROOT = DATA_COLLECTION.parent
os.environ.setdefault("MORAI_PROJECT_ROOT", str(PROJECT_ROOT))
INTERNAL = DATA_COLLECTION / "internal_sensor_record"
if os.environ.get("MORAI_INTERNAL_ENV_READY") != "1":
    prepare = TOOLS / "00_prepare_internal_sensor_record.sh"
    result = subprocess.run([str(prepare)], env=dict(os.environ))
    if result.returncode:
        raise SystemExit(result.returncode)
sys.path.insert(0, str(INTERNAL))
from runtime_paths import ANALYSIS_PYTHON, ASYNC_DEPS, CATKIN_ROOT, ROS_ROOT

target = (
    INTERNAL / "tools" / "validate_sensor_record_campaign.py"
)
environment = dict(os.environ)
environment.update(
    MORAI_ANALYSIS_PYTHON=str(ANALYSIS_PYTHON),
    MORAI_ASYNC_DEPS=str(ASYNC_DEPS),
    PYTHONNOUSERSITE="1",
    PYTHONDONTWRITEBYTECODE="1",
    PYTHONPATH=":".join([
        str(ASYNC_DEPS),
        str(INTERNAL),
        str(PROJECT_ROOT),
        str(CATKIN_ROOT / "devel/lib/python3/dist-packages"),
        str(ROS_ROOT / "lib/python3/dist-packages"),
    ]),
)
capture_python = "/usr/bin/python3"
os.execve(
    capture_python,
    [capture_python, str(target)] + sys.argv[1:],
    environment,
)
