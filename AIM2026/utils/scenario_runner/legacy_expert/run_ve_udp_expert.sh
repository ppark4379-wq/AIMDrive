#!/usr/bin/env bash
set -euo pipefail
SCENARIO_LEGACY_FUNCTIONS_ONLY=1 source "$(dirname "${BASH_SOURCE[0]}")/ve_ros_env.sh"
task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$task_dir/../../.." && pwd)"
if [[ " ${*} " == *" --check "* ]]; then
    exec /usr/bin/python3 -B "$task_dir/run_ve_udp_expert.py" "$@"
fi
scenario_run_legacy_ros_python "$task_dir/run_ve_udp_expert.py" "$@"
