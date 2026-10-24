#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$task_dir/../../.." && pwd)"
if [[ " ${*} " == *" --check "* ]]; then
    exec /usr/bin/python3 -B "$task_dir/run_ve_udp_expert.py" "$@"
fi
source "$repo_dir/MoraiDrive/data_collection/tools/internal_sensor_record_env.sh"
morai_load_internal_sensor_record_environment
exec /usr/bin/python3 -B "$task_dir/run_ve_udp_expert.py" "$@"
