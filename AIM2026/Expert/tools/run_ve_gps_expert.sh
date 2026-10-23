#!/usr/bin/env bash
set -euo pipefail
ve_gps_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$ve_gps_repo:$ve_gps_repo/MoraiDrive/data_collection/internal_sensor_record${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 -B - "$ve_gps_repo" "$@" <<'PY'
import os,sys
from python_environment import build_capture_environment
repo=sys.argv[1]
os.execve('/usr/bin/python3',['/usr/bin/python3','-u','-B',
    repo+'/AIM2026/Expert/tools/run_ve_gps_expert.py',*sys.argv[2:]],build_capture_environment())
PY
