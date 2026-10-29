#!/usr/bin/env bash
set -euo pipefail
# Tunnel scenes with the integrated Expert and Failure Diagnosis (virtual GPS inside the tunnel).
# Options:
#   --variant V         normal (empty tunnel; default) | brake | static | pedestrian | vehicle | all
#                       all = brake, static, pedestrian, vehicle in turn, one scene per run (4 runs by default)
#   --fixed-pedestrian, --pedestrian-reaction-s S, --pedestrian-margin-m M   pedestrian scene
#   --vehicle-behavior B                                                      vehicle scene
#   --no-virtual-gps    keep real GPS inside the tunnel
#   --seed S            scenario seed (with --runs N: first seed, run i uses S+i-1)
# Repeat options (shared by run_highway / run_roundabout / run_static / run_tunnel):
#   --runs N            run the scenario N times, one MORAI session per run
#   --same-seed         use --seed S for every run
#   --stop-on-failure   stop at the first run that does not COMPLETE
#   --pause-s T         seconds between runs (default 10)
#   --run-timeout-s T   stop a run that takes longer than T seconds (default 1200; 0 = no limit)
# Failure Diagnosis is recorded for every run. Reports and index.html:
#   AIM2026/outputs/batches/tunnel_<variant>-<time>-x<N>/   (see docs/repeat_runs.md)
# Runner, shared AIM2026 modules and the capture environment come from AIM2026/utils/scenario_runner.
source "$(dirname "${BASH_SOURCE[0]}")/../utils/scenario_runner/tools/project_env.sh"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 GRPC_POLL_STRATEGY=epoll1
exec /usr/bin/python3 -B - "$SCENARIO_AIM_ROOT" "$@" <<'PY'
import os
import sys
from scenario_runner.utils.environment import build_capture_environment
repo = sys.argv[1]
os.execve('/usr/bin/python3', ['/usr/bin/python3', '-u', '-B',
    repo + '/Expert/run_tunnel.py', *sys.argv[2:]], build_capture_environment())
PY
