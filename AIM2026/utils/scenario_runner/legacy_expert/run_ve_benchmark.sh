#!/usr/bin/env bash
set -euo pipefail
SCENARIO_LEGACY_FUNCTIONS_ONLY=1 source "$(dirname "${BASH_SOURCE[0]}")/ve_ros_env.sh"
benchmark_repo="$SCENARIO_AIM_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
# Use exactly the ROS/native dependency environment used by the existing runner.
scenario_run_legacy_capture_python "AIM2026" -m scenario_runner.tools.ve_expert_benchmark "$@"
