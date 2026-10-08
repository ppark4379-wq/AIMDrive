#!/usr/bin/env bash
set -Eeuo pipefail
TUNNEL_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TUNNEL_PREVIEW=0
for TUNNEL_ARGUMENT in "$@"; do
    if [[ "$TUNNEL_ARGUMENT" == "--preview" ]]; then TUNNEL_PREVIEW=1; fi
done
source "$TUNNEL_PROJECT_ROOT/data_collection/tools/internal_sensor_record_env.sh"
if (( TUNNEL_PREVIEW == 1 )); then
    morai_load_internal_sensor_record_environment
else
    # Reuse the dependency paths without requiring a live WSL/ROS address.
    MORAI_ASYNC_DEPS="$(morai_runtime_value ASYNC_DEPS)"
    MORAI_CATKIN_WS="$(morai_runtime_value CATKIN_ROOT)"
    MORAI_ROS_ROOT="$(morai_runtime_value ROS_ROOT)"
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="$MORAI_ASYNC_DEPS:$MORAI_INTERNAL_SENSOR_RECORD_ROOT:$MORAI_PROJECT_ROOT:$MORAI_CATKIN_WS/devel/lib/python3/dist-packages:$MORAI_ROS_ROOT/lib/python3/dist-packages"
fi
morai_require_capture_python
cd "$TUNNEL_PROJECT_ROOT"
exec /usr/bin/python3 -u -m scenario_runner.tools.tunnel_static_candidates "$@"
