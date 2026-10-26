#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
morai_load_internal_sensor_record_environment
exec /usr/bin/python3 -u -B -m scenario_runner.tools.ros_preflight "$@"
