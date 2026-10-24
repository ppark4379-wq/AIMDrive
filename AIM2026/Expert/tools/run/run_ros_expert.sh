#!/usr/bin/env bash
set -euo pipefail
expert_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "$expert_root/ros/devel/setup.bash" ]]; then
    source "$expert_root/ros/devel/setup.bash"
fi
exec "${MORAI_ROS_PYTHON:-/usr/bin/python3}" -u -B "$expert_root/main.py" --transport ros "$@"
