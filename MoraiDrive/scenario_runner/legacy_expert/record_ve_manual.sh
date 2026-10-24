#!/usr/bin/env bash
set -euo pipefail
d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$d/../../.." && pwd)"
source "$repo/MoraiDrive/data_collection/tools/internal_sensor_record_env.sh"
morai_load_internal_sensor_record_environment
exec /usr/bin/python3 -B "$d/record_ve_manual.py" "$@"
