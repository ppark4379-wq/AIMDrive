#!/usr/bin/env bash
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../tools/project_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/ve_ros_env.sh"
exec /usr/bin/python3 -u "$task_tools/ve_ros_cuda_relay.py" "$@"
