#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/noetic/setup.bash
if [[ -f "$ROOT/expert/devel/setup.bash" ]]; then
  source "$ROOT/expert/devel/setup.bash"
fi

echo "[EDITOR] Expert 경로가 필요하면 별도 터미널에서 full_loop_static 시나리오를 실행하세요."

python_candidates=(
  "${MORAI_EDITOR_PYTHON:-}"
  "${MORAI_ANALYSIS_PYTHON:-}"
  "$(command -v python3 2>/dev/null || true)"
  "/usr/bin/python3"
)
editor_python=""
for candidate in "${python_candidates[@]}"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  if "$candidate" -c 'import rospy, grpc, yaml' >/dev/null 2>&1; then
    editor_python="$candidate"
    break
  fi
done
if [[ -z "$editor_python" ]]; then
  echo "[FAIL] rospy, grpcio, PyYAML을 함께 제공하는 Python을 찾지 못했습니다." >&2
  echo "       conda activate lmdrive 후 다시 실행하거나 MORAI_EDITOR_PYTHON을 지정하세요." >&2
  exit 1
fi
echo "[EDITOR] Python: $editor_python"
exec "$editor_python" -m scenario_runner.tools.static_scenario_editor "$@"
