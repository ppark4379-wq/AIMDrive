#!/usr/bin/env bash
set -eo pipefail
viewer_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Pass setup its own arguments so viewer flags (e.g. --roi) are not parsed
# by catkin's environment setup script.
source /opt/ros/noetic/setup.bash --extend
export ROS_MASTER_URI=http://127.0.0.1:11412
export ROS_HOSTNAME=127.0.0.1
export PYTHONPATH="$viewer_repo/.runtime/ve_ros_deps:${PYTHONPATH:-}"
exec /usr/bin/python3 -u "$viewer_repo/AIM2026/Visualization/rviz.py" "$@"
