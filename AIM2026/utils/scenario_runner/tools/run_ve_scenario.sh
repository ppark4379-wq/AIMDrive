#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
ve_scenario_repo="$SCENARIO_AIM_ROOT"
case "${1:-}" in
    roundabout) ve_scenario_mode=full_loop_roundabout ;;
    full) ve_scenario_mode=full_loop ;;
    traffic_light) ve_scenario_mode=full_loop_traffic_light ;;
    highway) ve_scenario_mode=full_loop_highway ;;
    *) echo "사용법: bash $0 {roundabout|highway|traffic_light|full} [--check] [--repeat] [--ve-checkpoint E3|E4|P1] [--ve-compare-gt] [--expert-speed-cap-kmh 60]" >&2; exit 2 ;;
esac
shift
ve_scenario_once=(--once)
ve_scenario_args=()
for ve_scenario_arg in "$@"; do
    if [[ "$ve_scenario_arg" == --repeat ]]; then
        ve_scenario_once=()
    else
        ve_scenario_args+=("$ve_scenario_arg")
    fi
done
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
exec /usr/bin/python3 -B - "$ve_scenario_repo" --scenario "$ve_scenario_mode" --no-sync-runtime --ve-objects "${ve_scenario_once[@]}" "${ve_scenario_args[@]}" <<'PY'
import os,sys
from scenario_runner.project_paths import AIM_ROOT, REPO_ROOT, PACKAGE_ROOT, PACKAGE_PARENT, project_path
from scenario_runner.utils.environment import build_capture_environment
repo=sys.argv[1]
env=build_capture_environment()
os.chdir(str(AIM_ROOT))
os.execve('/usr/bin/python3',['/usr/bin/python3','-u','-B','-m','scenario_runner.main_runner',*sys.argv[2:]],env)
PY
