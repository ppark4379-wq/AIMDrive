#!/usr/bin/env bash
set -euo pipefail
SCENARIO_LEGACY_FUNCTIONS_ONLY=1 source "$(dirname "${BASH_SOURCE[0]}")/ve_ros_env.sh"
ve_gps_repo="$SCENARIO_AIM_ROOT"
scenario_run_legacy_capture_python "" "MoraiDrive/scenario_runner/legacy_expert/run_ve_gps_expert.py" "$@"
