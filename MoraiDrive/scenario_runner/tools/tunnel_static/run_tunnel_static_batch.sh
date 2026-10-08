#!/usr/bin/env bash
set -Eeuo pipefail
TUNNEL_BATCH_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$TUNNEL_BATCH_PROJECT_ROOT/data_collection/tools/internal_sensor_record_env.sh"
morai_load_internal_sensor_record_environment
morai_require_capture_python
cd "$TUNNEL_BATCH_PROJECT_ROOT"
exec /usr/bin/python3 -u -m scenario_runner.tools.run_tunnel_static_batch "$@"
