#!/usr/bin/env bash
# Scenario runner with the GT-driven Expert (no VE object input), in the same dependency environment as
# run_ve_scenario.sh. Used with record_ve_manual.sh --alongside-gt-expert to compare VE against GT while the
# GT Expert drives (e.g. at a raised --expert-speed-cap-kmh).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
gt_scenario_repo="$SCENARIO_AIM_ROOT"
gt_scenario_once=(--once)
case "${1:-}" in
    # Highway validation repeats randomized laps until Ctrl+C by default.
    # Explicit --once or --max-runs N still limits the runner.
    highway) gt_scenario_mode=full_loop_highway; gt_scenario_once=() ;;
    roundabout) gt_scenario_mode=full_loop_roundabout ;;
    # Tunnel static: one approved layout per run; pass --max-runs N to repeat (no forced --once).
    tunnel_static) gt_scenario_mode=full_loop_tunnel_static; gt_scenario_once=() ;;
    # Checkpoint 10 -> highway exit -> tunnel, with the enabled tunnel sudden-brake vehicle.
    highway_tunnel) gt_scenario_mode=full_loop_highway_tunnel_test ;;
    *) echo "사용법: bash $0 {highway|roundabout|tunnel_static|highway_tunnel} [--check] [--expert-speed-cap-kmh N]
  highway는 Ctrl+C까지 반복. 1회는 --once, 횟수 제한은 --max-runs N.
  highway만 속도 캡 허용. tunnel_static은 --once 또는 --max-runs N을 직접 지정" >&2; exit 2 ;;
esac
shift
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
exec /usr/bin/python3 -B - "$gt_scenario_repo" --scenario "$gt_scenario_mode" --no-sync-runtime "${gt_scenario_once[@]}" "$@" <<'PY'
import os,sys
from scenario_runner.project_paths import AIM_ROOT, REPO_ROOT, PACKAGE_ROOT, PACKAGE_PARENT, project_path
from scenario_runner.utils.environment import build_capture_environment
repo=sys.argv[1]
env=build_capture_environment()
os.chdir(str(AIM_ROOT))
os.execve('/usr/bin/python3',['/usr/bin/python3','-u','-B','-m','scenario_runner.main_runner',*sys.argv[2:]],env)
PY
