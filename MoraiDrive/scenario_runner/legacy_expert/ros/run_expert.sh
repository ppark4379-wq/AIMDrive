#!/usr/bin/env bash
set -euo pipefail
legacy_ros_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$legacy_ros_root/../../../.." && pwd)"
expert_root="$repo_root/AIM2026/Expert"
if [[ -f "$expert_root/ros/devel/setup.bash" ]]; then
    source "$expert_root/ros/devel/setup.bash"
fi
exec "${MORAI_ROS_PYTHON:-/usr/bin/python3}" -u -B "$legacy_ros_root/runtime.py" "$@"
