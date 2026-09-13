#!/usr/bin/env bash
set -Eeuo pipefail

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_COLLECTION="$(cd "$TOOLS/.." && pwd)"
export MORAI_PROJECT_ROOT="${MORAI_PROJECT_ROOT:-$(cd "$DATA_COLLECTION/.." && pwd)}"
exec "$DATA_COLLECTION/internal_sensor_record/tools/collect_scenario_laps.sh" "$@"
