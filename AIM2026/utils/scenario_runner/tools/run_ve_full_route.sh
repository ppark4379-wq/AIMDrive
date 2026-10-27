#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
ve_full_root="$SCENARIO_AIM_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
exec /usr/bin/python3 -B - "$ve_full_root" "$@" <<'PY'
import os,sys
from scenario_runner.project_paths import AIM_ROOT, REPO_ROOT, PACKAGE_ROOT, PACKAGE_PARENT, project_path
from scenario_runner.utils.environment import build_capture_environment
repo=sys.argv[1]
os.execve('/usr/bin/python3',['/usr/bin/python3','-u','-B',
    str(project_path('MoraiDrive/scenario_runner/tools/run_ve_gps_full_route.py')),*sys.argv[2:]],build_capture_environment())
PY
