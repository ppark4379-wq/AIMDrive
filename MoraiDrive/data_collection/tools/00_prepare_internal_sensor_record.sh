#!/usr/bin/env bash
set -Eeuo pipefail

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$TOOLS/../.." && pwd)"
export MORAI_PROJECT_ROOT="${MORAI_PROJECT_ROOT:-$PROJECT_ROOT}"
export PYTHONNOUSERSITE=1

exec /usr/bin/python3 \
    "$TOOLS/../internal_sensor_record/tools/configure_local_machine.py" "$@"
