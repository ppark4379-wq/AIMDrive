#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/project_env.sh"
echo '[시험] START → 고주로 종료 체크포인트 13 · 터널 제외 · 30회 기본'
echo '[인지] 장애물·신호=VE · 속도 상한 60km/h · 충돌/오류/180초 정체 후 다음 회차'
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_ve_full_route_campaign.sh" "$@" --stop-before-tunnel
