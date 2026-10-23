#!/usr/bin/env bash
set -euo pipefail
ve_full_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$ve_full_root:$ve_full_root/MoraiDrive/data_collection/internal_sensor_record${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 -B - "$ve_full_root" "$@" <<'PY'
import os,sys
from python_environment import build_capture_environment
repo=sys.argv[1]
os.execve('/usr/bin/python3',['/usr/bin/python3','-u','-B',
    repo+'/MoraiDrive/scenario_runner/tools/run_ve_gps_full_route.py',*sys.argv[2:]],build_capture_environment())
PY
