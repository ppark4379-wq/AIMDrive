#!/usr/bin/env bash
set -euo pipefail
SCENARIO_LEGACY_FUNCTIONS_ONLY=1 source "$(dirname "${BASH_SOURCE[0]}")/ve_ros_env.sh"
d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$d/../../.." && pwd)"
scenario_run_legacy_ros_python "$d/record_ve_manual.py" "$@"
