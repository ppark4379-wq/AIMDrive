#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../project_env.sh"
TUNNEL_PROJECT_ROOT="$SCENARIO_AIM_ROOT"
TUNNEL_PREVIEW=0
for TUNNEL_ARGUMENT in "$@"; do
    if [[ "$TUNNEL_ARGUMENT" == "--preview" ]]; then TUNNEL_PREVIEW=1; fi
done
if (( TUNNEL_PREVIEW == 1 )); then
    morai_load_internal_sensor_record_environment
    scenario_export_project_paths
else
    # Reuse the dependency paths without requiring a live WSL/ROS address.
    MORAI_ASYNC_DEPS="$(morai_runtime_value ASYNC_DEPS)"
    MORAI_CATKIN_WS="$(morai_runtime_value CATKIN_ROOT)"
    MORAI_ROS_ROOT="$(morai_runtime_value ROS_ROOT)"
    export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="$MORAI_CATKIN_WS/devel/lib/python3/dist-packages:$MORAI_ROS_ROOT/lib/python3/dist-packages:$PYTHONPATH"
fi
morai_require_capture_python
cd "$TUNNEL_PROJECT_ROOT"
exec /usr/bin/python3 -u -m scenario_runner.tools.tunnel_static_candidates "$@"
