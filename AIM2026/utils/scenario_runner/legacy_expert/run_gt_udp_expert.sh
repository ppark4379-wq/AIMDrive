#!/usr/bin/env bash
set -euo pipefail
SCENARIO_LEGACY_FUNCTIONS_ONLY=1 source "$(dirname "${BASH_SOURCE[0]}")/ve_ros_env.sh"
gt_udp_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gt_udp_repo="$(cd "$gt_udp_dir/../../.." && pwd)"
scenario_run_legacy_capture_python "" "$gt_udp_dir/run_gt_udp_expert.py" "$@"
