#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../project_env.sh"
export PYTHONDONTWRITEBYTECODE=1
TUNNEL_SEARCH_PROJECT_ROOT="$SCENARIO_AIM_ROOT"
morai_load_internal_sensor_record_environment
scenario_export_project_paths
morai_require_capture_python
cd "$TUNNEL_SEARCH_PROJECT_ROOT"
exec /usr/bin/python3 -u -m scenario_runner.tools.run_tunnel_static_search "$@"
