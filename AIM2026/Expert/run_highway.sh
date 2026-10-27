#!/usr/bin/env bash
set -euo pipefail
# Highway traffic:
#   --highway-density competition [--rear-npcs few]   NPCs run independently of Ego, one speed per lane (config/highway_competition.yaml)
# Examples:
#   bash AIM2026/Expert/run_highway.sh --highway-density competition --runs 10 --seed 100 --run-timeout-s 600
#   bash AIM2026/Expert/run_highway.sh --highway-density competition --rear-npcs few --seed 100
# Repeat options (shared by run_highway / run_roundabout / run_static):
#   --runs N            run the scenario N times, one MORAI session per run
#   --seed S            first seed; run i uses S+i-1 (add --same-seed to keep S)
#   --stop-on-failure   stop at the first run that does not COMPLETE
#   --pause-s T         seconds between runs (default 10)
#   --run-timeout-s T   stop a run that takes longer than T seconds (default 1200; 0 = no limit)
# Failure Diagnosis is recorded for every run. Reports and index.html:
#   AIM2026/outputs/batches/<scenario>-<time>-x<N>/
# Runner, shared AIM2026 modules and the capture environment come from AIM2026/utils/scenario_runner.
source "$(dirname "${BASH_SOURCE[0]}")/../utils/scenario_runner/tools/project_env.sh"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 GRPC_POLL_STRATEGY=epoll1
exec /usr/bin/python3 -B - "$SCENARIO_AIM_ROOT" "$@" <<'PY'
import os
import sys
from scenario_runner.utils.environment import build_capture_environment
repo = sys.argv[1]
os.execve('/usr/bin/python3', ['/usr/bin/python3', '-u', '-B',
    repo + '/Expert/run_highway.py', *sys.argv[2:]], build_capture_environment())
PY
