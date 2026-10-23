#!/usr/bin/env bash
set -euo pipefail
ve_audit_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ve_audit_repo="$(cd "$ve_audit_dir/../../.." && pwd)"
source "$ve_audit_repo/MoraiDrive/data_collection/tools/internal_sensor_record_env.sh"
morai_load_internal_sensor_record_environment
exec /usr/bin/python3 -B "$ve_audit_dir/run_ve_hz_check.py" "$@"
