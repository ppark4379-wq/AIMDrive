#!/usr/bin/env bash
# Opt-in native crash capture for the scenario-owned Expert subprocess.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
debug_root="$project_root/expert/.runtime/native_debug"
if [[ ! -x "$debug_root/root/usr/bin/gdb" ]]; then
    printf 'Local gdb is missing: %s\n' "$debug_root/root/usr/bin/gdb" >&2
    exit 1
fi
: "${MORAI_DEBUG_PYTHON:?Launch with diagnostics/run_scenario_native_debug.py}"
mkdir -p "$debug_root/logs"
log_path="$debug_root/logs/expert-$(date +%Y%m%d-%H%M%S)-$$.log"
printf '[ExpertNativeDebug] log: %s\n' "$log_path"
export PYTHONUNBUFFERED=1
/lib64/ld-linux-x86-64.so.2 \
    --library-path "$debug_root/root/usr/lib/x86_64-linux-gnu" \
    "$debug_root/root/usr/bin/gdb" \
    --data-directory="$debug_root/root/usr/share/gdb" \
    -nx -batch --return-child-result \
    -x "$project_root/diagnostics/expert_crash.gdb" \
    --args "$MORAI_DEBUG_PYTHON" "$@" 2>&1 | tee "$log_path"
