#!/usr/bin/env bash
set -Eeuo pipefail

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_COLLECTION="$(cd "$TOOLS/.." && pwd)"
INTERNAL="$DATA_COLLECTION/internal_sensor_record"
PROJECT_ROOT="${MORAI_PROJECT_ROOT:-$(cd "$DATA_COLLECTION/.." && pwd)}"
CATKIN_ROOT="${MORAI_CATKIN_WS:-$(MORAI_PROJECT_ROOT="$PROJECT_ROOT" PYTHONPATH="$INTERNAL" /usr/bin/python3 -c 'from runtime_paths import CATKIN_ROOT; print(CATKIN_ROOT)')}"
ROS_ROOT="${MORAI_ROS_ROOT:-$(MORAI_PROJECT_ROOT="$PROJECT_ROOT" PYTHONPATH="$INTERNAL" /usr/bin/python3 -c 'from runtime_paths import ROS_ROOT; print(ROS_ROOT)')}"
CATKIN_SETUP="$CATKIN_ROOT/devel/setup.bash"
CATKIN_PYTHON="$CATKIN_ROOT/devel/lib/python3/dist-packages"
MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
ROS_HOST_IP="${MORAI_BRIDGE_IP:-}"
CONNECT_TIMEOUT="${MORAI_CONNECT_TIMEOUT_SECONDS:-600}"
BRIDGE_PID=""
STARTED_BRIDGE=0

cleanup() {
    if [[ "$STARTED_BRIDGE" == "1" ]] && [[ -n "$BRIDGE_PID" ]] \
            && kill -0 "$BRIDGE_PID" 2>/dev/null; then
        kill -INT "$BRIDGE_PID" 2>/dev/null || true
        wait "$BRIDGE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ ! "$CONNECT_TIMEOUT" =~ ^[0-9]+$ ]] || (( CONNECT_TIMEOUT < 30 )); then
    echo "[FAIL] MORAI_CONNECT_TIMEOUT_SECONDS는 30 이상의 정수여야 합니다." >&2
    exit 2
fi
if [[ ! -f "$ROS_ROOT/setup.bash" ]] || [[ ! -f "$CATKIN_SETUP" ]]; then
    echo "[FAIL] ROS1 Noetic 또는 catkin workspace setup을 찾을 수 없습니다." >&2
    exit 2
fi

set +u
source "$ROS_ROOT/setup.bash"
source "$CATKIN_SETUP"
set -u
export ROS_MASTER_URI="$MASTER_URI"
export MORAI_PROJECT_ROOT="$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$INTERNAL:$PROJECT_ROOT:$CATKIN_PYTHON:$ROS_ROOT/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "$ROS_HOST_IP" ]]; then
    ROS_HOST_IP="$(/usr/bin/python3 -c 'from network_config import bridge_ip; print(bridge_ip())')"
fi
export MORAI_BRIDGE_IP="$ROS_HOST_IP"
export ROS_IP="$ROS_HOST_IP"

echo "[CONFIG] ROS master: $ROS_MASTER_URI"
echo "[CONFIG] MORAI bridge: ${MORAI_BRIDGE_IP}:9090"
/usr/bin/python3 - <<'PY'
import json
from ssd_storage import ensure_storage_layout
print('[CONFIG] storage: '+json.dumps(ensure_storage_layout(),ensure_ascii=False))
PY

/usr/bin/python3 "$INTERNAL/tools/preflight_internal_sensor_record.py"

if ss -H -ltn 'sport = :9090' 2>/dev/null | grep -q .; then
    echo "[INFO] 9090 listener가 이미 있습니다. 실행 중인 bridge를 검증합니다."
else
    echo "[START] instrumented bridge: $MASTER_URI"
    /usr/bin/python3 "$INTERNAL/tools/start_instrumented_bridge.py" \
        --master "$MASTER_URI" --port 9090 --drop-camera-ingress &
    BRIDGE_PID=$!
    STARTED_BRIDGE=1
fi

deadline=$((SECONDS + 40))
until rosparam list >/dev/null 2>&1; do
    if [[ "$STARTED_BRIDGE" == "1" ]] && ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
        wait "$BRIDGE_PID" || true
        echo "[FAIL] bridge가 ROS master 준비 전에 종료됐습니다." >&2
        exit 2
    fi
    if (( SECONDS >= deadline )); then
        echo "[FAIL] ROS master가 40초 안에 준비되지 않았습니다: $MASTER_URI" >&2
        exit 2
    fi
    sleep 1
done

rosparam set /use_sim_time true
echo
echo "[WAIT] MORAI의 차량 정보, Camera 1~4, LiDAR, GPS, IMU를 ${ROS_HOST_IP}:9090에 연결하세요."
echo "       Simulation Time / Sensor Synchronization 10Hz / External Sync OFF"

deadline=$((SECONDS + CONNECT_TIMEOUT))
last_report=""
while (( SECONDS < deadline )); do
    if [[ "$STARTED_BRIDGE" == "1" ]] && ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
        wait "$BRIDGE_PID" || true
        echo "[FAIL] 연결 대기 중 bridge가 종료됐습니다." >&2
        exit 2
    fi
    if last_report=$(/usr/bin/python3 \
            "$INTERNAL/tools/probe_sensor_record_readiness.py" \
            --seconds 4 --raw-camera-transport bridge_dropped 2>&1); then
        echo
        echo "[PASS] 필수 stream, source 진행, Camera ingress 차단을 확인했습니다."
        echo "$last_report"
        echo
        echo "수집: $TOOLS/02_collect_internal_sensor_record_laps.sh 3 <campaign_id>"
        if [[ "$STARTED_BRIDGE" == "1" ]]; then
            echo "이 터미널은 bridge를 유지합니다. 수집 종료 후 Ctrl-C로 닫으세요."
            wait "$BRIDGE_PID"
        fi
        exit 0
    fi
    if [[ "$last_report" == *"BRIDGE_NOT_PATCHED:"* ]]; then
        echo "[FAIL] 9090에서 다른 버전의 bridge가 실행 중입니다." >&2
        echo "       기존 bridge를 종료한 뒤 이 01번 스크립트를 다시 실행하세요." >&2
        echo "$last_report" >&2
        exit 2
    fi
    echo "[WAIT] 아직 연결이 완전하지 않습니다: $last_report"
    sleep 2
done

echo "[FAIL] 제한 시간 안에 필수 stream이 준비되지 않았습니다." >&2
echo "$last_report" >&2
exit 2
