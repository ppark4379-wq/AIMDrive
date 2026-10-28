#!/usr/bin/env bash
# Repeat one tunnel scene with a fresh seed, including after a failed run.
# Usage: bash tunnel_repeat.sh [pedestrian|vehicle|mixed|dog2|brake|normal] [first_seed]
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
variant="${1:-pedestrian}"
seed="${2:-2000}"
case "$variant" in
  pedestrian|vehicle|mixed|dog2|brake|normal) ;;
  *) echo "Usage: bash tunnel_repeat.sh [pedestrian|vehicle|mixed|dog2|brake|normal] [first_seed]" >&2; exit 2 ;;
esac
if [[ ! "$seed" =~ ^[0-9]{1,9}$ ]]; then
  echo "first_seed must be a non-negative integer of up to 9 digits." >&2
  exit 2
fi
seed=$((10#$seed))
expert_python="${AIM_EXPERT_PYTHON:-$repo/.runtime/situation-windows/Scripts/python.exe}"
if [[ ! -f "$expert_python" ]]; then
  echo "Expert Python not found: $expert_python" >&2
  exit 2
fi
export MORAI_GRPC_TARGET="${MORAI_GRPC_TARGET:-127.0.0.1:7789}"
trap 'exit 130' INT
trap 'exit 143' TERM
cd "$repo" || exit 1

while true; do
  echo "[Repeat] variant=$variant seed=$seed (Ctrl+C to stop)"
  bash "$repo/AIM2026/utils/scenario_runner/tools/run_tunnel_aim2026.sh" \
    --variant "$variant" --runs 1 --seed "$seed" --validation-events \
    --expert-python "$expert_python"
  result=$?
  if [[ "$result" -eq 130 || "$result" -eq 143 ]]; then
    exit "$result"
  fi
  echo "[Repeat] exit=$result; next seed in 3 seconds"
  seed=$((seed + 1))
  sleep 3
done
