#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRESET="${1:-}"
RUNS="${2:-1}"

if [[ -z "$PRESET" ]]; then
  echo "사용법: $0 <편집기 프리셋 이름 또는 YAML 경로> [반복 횟수]" >&2
  exit 2
fi
if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[FAIL] 반복 횟수는 1 이상의 정수여야 합니다: $RUNS" >&2
  exit 2
fi

if [[ -r /opt/ros/noetic/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.bash
fi
if [[ -r "$ROOT/expert/devel/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/expert/devel/setup.bash"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/scenario_runner/.runtime/static_scenario_verification"
LOG_PATH="$LOG_DIR/${STAMP}.log"
mkdir -p "$LOG_DIR"

echo "[VERIFY] preset=$PRESET, runs=$RUNS"
echo "[VERIFY] Sensor Record/Synchronization Mode는 사용하지 않습니다."
echo "[VERIFY] 실제 Expert 폐루프 주행, MORAI collision_objects, 경로 이탈, 정체, 목표 도달을 검사합니다."
echo "[VERIFY] log=$LOG_PATH"

for ((run=1; run<=RUNS; run++)); do
  echo "[VERIFY] run $run/$RUNS" | tee -a "$LOG_PATH"
  if (
    cd "$ROOT"
    /usr/bin/python3 -m scenario_runner.main_runner \
      --zone full_loop \
      --scenario full_loop_static \
      --once \
      --no-sync-runtime \
      --static-preset "$PRESET"
  ) 2>&1 | tee -a "$LOG_PATH"; then
    echo "[PASS] run $run/$RUNS: 충돌 없이 정적 구간 목표에 도달했습니다." | tee -a "$LOG_PATH"
  else
    status=${PIPESTATUS[0]}
    echo "[FAIL] run $run/$RUNS: 실제 주행 검증 실패(exit=$status). log=$LOG_PATH" | tee -a "$LOG_PATH" >&2
    exit "$status"
  fi
done

echo "[PASS] static preset closed-loop verification: $RUNS/$RUNS, log=$LOG_PATH"
