#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../tools/project_env.sh"
legacy_ros_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$legacy_ros_root/../../../.." && pwd)"
expert_root="$SCENARIO_AIM_ROOT/Expert"
if [[ -f "$expert_root/ros/devel/setup.bash" ]]; then
    source "$expert_root/ros/devel/setup.bash"
fi
exec "${MORAI_ROS_PYTHON:-/usr/bin/python3}" -u -B "$legacy_ros_root/runtime.py" "$@"
