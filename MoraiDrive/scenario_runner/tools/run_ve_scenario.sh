#!/usr/bin/env bash
set -euo pipefail
ve_scenario_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
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
export PYTHONPATH="$ve_scenario_repo:$ve_scenario_repo/MoraiDrive:$ve_scenario_repo/VLDrive:$ve_scenario_repo/MoraiDrive/data_collection/internal_sensor_record${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 -B - "$ve_scenario_repo" --scenario "$ve_scenario_mode" --no-sync-runtime --ve-objects "${ve_scenario_once[@]}" "${ve_scenario_args[@]}" <<'PY'
import os,sys
from python_environment import build_capture_environment
repo=sys.argv[1]
env=build_capture_environment()
env['PYTHONPATH']=os.pathsep.join([repo,repo+'/MoraiDrive',repo+'/VLDrive',repo+'/MoraiDrive/data_collection/internal_sensor_record',env.get('PYTHONPATH','')])
os.chdir(repo+'/MoraiDrive')
os.execve('/usr/bin/python3',['/usr/bin/python3','-u','-B','-m','scenario_runner.main_runner',*sys.argv[2:]],env)
PY
