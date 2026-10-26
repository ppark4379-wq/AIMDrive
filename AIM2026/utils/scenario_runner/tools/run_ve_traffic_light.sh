#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
ve_signal_repo="$SCENARIO_AIM_ROOT"
ve_signal_head="$(/usr/bin/python3 -B -c 'from scenario_runner.utils.perception import signal_head; print(signal_head())')"
if [[ ! -f "$ve_signal_head" ]]; then
    echo "신호 head 없음: $ve_signal_head (VE_SIGNAL_HEAD로 경로 지정)" >&2
    exit 2
fi
exec bash "$SCENARIO_RUNNER_ROOT/tools/run_ve_scenario.sh" traffic_light \
    --ve-checkpoint P1 --ve-signal-head "$ve_signal_head" "$@"
