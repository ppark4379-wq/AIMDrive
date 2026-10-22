#!/usr/bin/env bash
set -euo pipefail
ve_signal_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ve_signal_head="${VE_SIGNAL_HEAD:-$ve_signal_repo/MoraiDrive/.runtime/signal_state_p1_20261014/run_003/signal_head_final.pt}"
if [[ ! -f "$ve_signal_head" ]]; then
    echo "신호 head 없음: $ve_signal_head (VE_SIGNAL_HEAD로 경로 지정)" >&2
    exit 2
fi
exec bash "$ve_signal_repo/MoraiDrive/scenario_runner/tools/run_ve_scenario.sh" traffic_light \
    --ve-checkpoint P1 --ve-signal-head "$ve_signal_head" "$@"
