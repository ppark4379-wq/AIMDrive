#!/usr/bin/env bash
# Unified pedestrian launcher. Each completed run repeats; Ctrl+C stops.
# Examples:
#   bash run_scenario.sh --random-locations 2
#   bash run_scenario.sh --opposing-crosswalk 2 --roads 1 4 5
set -euo pipefail
scenario_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERT_PYTHON="$scenario_repo/.runtime/situation-windows/Scripts/python.exe"
if (($# == 0)); then
  echo 'Choose --random-locations 2|3|4, or --opposing-crosswalk 2|3 --roads 1 4 5'
  echo 'Optional: --once, --seed NUMBER, --expert-python PATH'
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export GRPC_POLL_STRATEGY=epoll1
export PYTHONPATH="$scenario_repo:$scenario_repo/MoraiDrive/data_collection/internal_sensor_record${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 -B - "$scenario_repo" \
  --repeat --validation-events --pedestrian-trace --expert-python "$EXPERT_PYTHON" "$@" <<'PY'
import os
import sys
from python_environment import build_capture_environment
repo = sys.argv[1]
os.execve('/usr/bin/python3', ['/usr/bin/python3', '-u', '-B',
    repo + '/MoraiDrive/scenario_runner/tools/run_aim_dynamic.py', *sys.argv[2:]],
    build_capture_environment())
PY
