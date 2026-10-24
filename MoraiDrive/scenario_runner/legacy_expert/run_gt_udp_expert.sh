#!/usr/bin/env bash
set -euo pipefail
gt_udp_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gt_udp_repo="$(cd "$gt_udp_dir/../../.." && pwd)"
export PYTHONPATH="$gt_udp_repo:$gt_udp_repo/MoraiDrive/data_collection/internal_sensor_record${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 -B - "$gt_udp_dir" "$@" <<'PY'
import os,sys
from python_environment import build_capture_environment
env=build_capture_environment()
os.execve('/usr/bin/python3',['/usr/bin/python3','-u','-B',
    sys.argv[1]+'/run_gt_udp_expert.py',*sys.argv[2:]],env)
PY
