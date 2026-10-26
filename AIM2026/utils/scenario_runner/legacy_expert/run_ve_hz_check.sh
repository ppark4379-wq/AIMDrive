#!/usr/bin/env bash
set -euo pipefail
SCENARIO_LEGACY_FUNCTIONS_ONLY=1 source "$(dirname "${BASH_SOURCE[0]}")/ve_ros_env.sh"
ve_audit_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ve_audit_repo="$(cd "$ve_audit_dir/../../.." && pwd)"
scenario_run_legacy_ros_python "$ve_audit_dir/run_ve_hz_check.py" "$@"
