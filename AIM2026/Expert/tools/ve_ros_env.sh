#!/usr/bin/env bash
# Source only the ROS runtime needed by the read-only VE tools.
set -e
task_tools="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_repo="$(cd "$task_tools/../../.." && pwd)"
source /opt/ros/noetic/setup.bash
export PYTHONPATH="$task_repo/AIM2026/Expert/devel/lib/python3/dist-packages:$task_repo/VLDrive:${PYTHONPATH:-}"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
