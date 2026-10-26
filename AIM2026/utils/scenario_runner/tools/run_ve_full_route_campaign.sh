#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
campaign_repo="$SCENARIO_AIM_ROOT"
echo '[캠페인] 실행 옵션의 구간·센서 설정으로 반복 · 정리 후 다음 회차'
campaign_output="$SCENARIO_RUNNER_ROOT/.runtime/ve_full_route_campaign/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$campaign_output"
if [[ "${1:-}" == --background ]]; then
    shift
    nohup setsid /usr/bin/python3 -u -B "$SCENARIO_RUNNER_ROOT/tools/ve_full_route_campaign.py" \
        --output "$campaign_output" "$@" >"$campaign_output/campaign.log" 2>&1 </dev/null &
    echo "[백그라운드 요청] PID=$!"
    echo "[진행] $campaign_output/status.json"
    echo "[로그] $campaign_output/campaign.log"
    echo "[결과] $campaign_output/report.json"
    sleep 2
    if [[ ! -f "$campaign_output/status.json" ]]; then
        cat "$campaign_output/campaign.log" >&2
        echo '[시작 실패] 상태 파일이 생성되지 않았습니다.' >&2
        exit 1
    fi
else
    exec /usr/bin/python3 -u -B "$SCENARIO_RUNNER_ROOT/tools/ve_full_route_campaign.py" \
        --output "$campaign_output" "$@"
fi
