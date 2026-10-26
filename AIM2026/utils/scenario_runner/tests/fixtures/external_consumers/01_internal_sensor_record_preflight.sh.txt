#!/usr/bin/env bash
# Keep this entrypoint LF-only; Windows Git checkouts are supported.
set -Eeuo pipefail

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$TOOLS/internal_sensor_record_env.sh"

RUN_MODE="capture"
case "${1:-}" in
    "") ;;
    --scenario-only)
        RUN_MODE="scenario"
        shift
        ;;
    -h|--help)
        cat <<'EOF'
사용법:
  ./data_collection/tools/01_internal_sensor_record_preflight.sh
      전체 Sensor Record 수집용 센서 계약과 stream을 검사합니다.

  ./data_collection/tools/01_internal_sensor_record_preflight.sh --scenario-only
      데이터 저장 없이 Scenario Runner·Expert 검증에 필요한 ROS master와
      bridge만 유지합니다. Camera/LiDAR/GPS/IMU/clock 연결을 요구하지 않습니다.
EOF
        exit 0
        ;;
    *)
        echo "[FAIL] 알 수 없는 옵션: $1" >&2
        exit 2
        ;;
esac
if (( $# != 0 )); then
    echo "[FAIL] 불필요한 인자가 있습니다: $*" >&2
    exit 2
fi

if [[ "$RUN_MODE" == "scenario" ]]; then
    echo "[MODE] Scenario Runner·Expert 검증 전용 bridge 유지 모드"
    echo "       Sensor Record와 Camera/LiDAR/GPS/IMU/clock 연결은 검사하지 않습니다."
    prepare_mode="runtime"
else
    prepare_mode="sensor-set"
fi

echo "[1/5] 이 PC의 고정 MORAI 연결 주소·ROS/catkin·Python 환경을 준비합니다."
echo "      Git 추적 소스는 수정하지 않습니다. PC별 설정은 .runtime에만 저장합니다."
morai_prepare_internal_sensor_record_environment "$prepare_mode"
morai_load_internal_sensor_record_environment
morai_require_capture_python

INTERNAL="$MORAI_INTERNAL_SENSOR_RECORD_ROOT"
MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
MORAI_ENDPOINT_IP="$MORAI_BRIDGE_IP"
CONNECT_TIMEOUT="${MORAI_CONNECT_TIMEOUT_SECONDS:-600}"
BRIDGE_PID=""
STARTED_BRIDGE=0
STATUS_DIR="$MORAI_PROJECT_ROOT/.runtime/data_collection/preflight_status"
STATUS_FILE="$STATUS_DIR/latest.json"
STATUS_ERROR="$STATUS_DIR/latest.stderr"
BRIDGE_LOG="$STATUS_DIR/bridge.log"
mkdir -p "$STATUS_DIR"

cleanup() {
    if [[ "$STARTED_BRIDGE" == "1" ]] && [[ -n "$BRIDGE_PID" ]] \
            && kill -0 "$BRIDGE_PID" 2>/dev/null; then
        echo "[STOP] 이 터미널이 시작한 bridge와 roscore를 정리합니다."
        kill -TERM "$BRIDGE_PID" 2>/dev/null || true
        local deadline=$((SECONDS + 35))
        while kill -0 "$BRIDGE_PID" 2>/dev/null && (( SECONDS < deadline )); do
            sleep 0.2
        done
        if kill -0 "$BRIDGE_PID" 2>/dev/null; then
            kill -KILL "$BRIDGE_PID" 2>/dev/null || true
        fi
        wait "$BRIDGE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ ! "$CONNECT_TIMEOUT" =~ ^[0-9]+$ ]] \
        || (( CONNECT_TIMEOUT < 30 )); then
    echo "[FAIL] MORAI_CONNECT_TIMEOUT_SECONDS는 30 이상의 정수여야 합니다." >&2
    exit 2
fi
export MORAI_BRIDGE_IP="$MORAI_ENDPOINT_IP"

echo
echo "[2/5] 현재 PC 설정"
echo "      MORAI 연결 주소 : ${MORAI_BRIDGE_IP}:9090"
echo "      WSL ROS node 주소: ${ROS_IP} (내부 자동 탐지)"
echo "      ROS master       : $ROS_MASTER_URI"
echo "      원본 저장 경로   : $MORAI_STORAGE_ROOT"
if [[ "$RUN_MODE" == "capture" ]]; then
    echo "      센서셋 파일      : $MORAI_SENSOR_SET_PATH"
    /usr/bin/python3 "$INTERNAL/tools/preflight_local_storage.py" --reserve-gib 70

    clock_report="$MORAI_PROJECT_ROOT/.runtime/data_collection/clock_preflight.json"
    /usr/bin/python3 "$INTERNAL/tools/preflight_local_clock.py" \
        --wait-seconds "${MORAI_CLOCK_STABILIZATION_WAIT_SECONDS:-90}" \
        --output "$clock_report"
else
    echo "[SKIP] 데이터 저장 공간·WSL 수집 시계 검사 (시나리오 전용 모드)"
fi

echo
if [[ "$RUN_MODE" == "capture" ]]; then
    echo "[3/5] MORAI 26.R1.H3와 AIM 센서 geometry/rate/topic 계약을 검사합니다."
    if ! contract_report=$(/usr/bin/python3 \
            "$INTERNAL/tools/preflight_internal_sensor_record.py" 2>&1); then
        echo "$contract_report" >&2
        echo "[FAIL] 센서셋 내용이 AIM 수집 계약과 다릅니다." >&2
        echo "       주소 외 geometry/FOV/rate/topic 차이는 자동 수정하지 않습니다." >&2
        exit 2
    fi
    echo "[PASS] 센서셋 정적 계약 통과"
else
    echo "[3/5] [SKIP] Sensor Record 센서셋 계약 검사 (시나리오 전용 모드)"
fi

echo
echo "[4/5] ROS master와 계측 bridge를 준비합니다."
if ss -H -ltn 'sport = :9090' 2>/dev/null | grep -q .; then
    listener="$(ss -H -ltnp 'sport = :9090' 2>/dev/null | head -n 1 || true)"
    echo "[INFO] 9090 listener가 이미 있어 중복 실행하지 않습니다."
    [[ -n "$listener" ]] && echo "       $listener"
    echo "       아래 health 검사에서 우리 계측 bridge인지 확인합니다."
else
    echo "[START] 9090 계측 bridge를 시작합니다. 필요한 경우 roscore도 함께 시작합니다."
    echo "        bridge 상세 로그: $BRIDGE_LOG"
    /usr/bin/python3 "$INTERNAL/tools/start_instrumented_bridge.py" \
        --master "$MASTER_URI" --port 9090 --drop-camera-ingress \
        >"$BRIDGE_LOG" 2>&1 &
    BRIDGE_PID=$!
    STARTED_BRIDGE=1
fi

deadline=$((SECONDS + 40))
until rosparam list >/dev/null 2>&1; do
    if [[ "$STARTED_BRIDGE" == "1" ]] && ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
        wait "$BRIDGE_PID" || true
        echo "[FAIL] bridge가 ROS master 준비 전에 종료됐습니다." >&2
        tail -n 40 "$BRIDGE_LOG" >&2 || true
        exit 2
    fi
    if (( SECONDS >= deadline )); then
        echo "[FAIL] ROS master가 40초 안에 준비되지 않았습니다: $MASTER_URI" >&2
        exit 2
    fi
    sleep 1
done

if [[ "$RUN_MODE" == "capture" ]]; then
    rosparam set /use_sim_time true
    echo "[PASS] ROS master 준비 및 /use_sim_time=true"
else
    # With no Simulator Network /clock connection, use_sim_time=true freezes
    # rospy timers used by the Expert and editor. Scenario-only validation is
    # deliberately asynchronous and uses host/receive time.
    rosparam set /use_sim_time false
    echo "[PASS] ROS master 준비 및 /use_sim_time=false (비동기 시나리오 검증)"
fi

if [[ "$MORAI_ENDPOINT_IP" == 127.* ]]; then
    localhost_code=2
    for localhost_attempt in 1 2 3; do
        set +e
        "$MORAI_POWERSHELL_EXE" -NoProfile -NonInteractive -Command \
            '$client = New-Object System.Net.Sockets.TcpClient; try { $result = $client.BeginConnect("127.0.0.1", 9090, $null, $null); if (-not $result.AsyncWaitHandle.WaitOne(3000)) { exit 2 }; $client.EndConnect($result); exit 0 } catch { exit 2 } finally { $client.Dispose() }' \
            >/dev/null 2>&1
        localhost_code=$?
        set -e
        if (( localhost_code == 0 )); then
            break
        fi
        if (( localhost_attempt < 3 )); then
            echo "[WAIT] Windows→WSL localhost 연결을 재확인합니다. (${localhost_attempt}/3 실패)"
            sleep 2
        fi
    done
    if (( localhost_code != 0 )); then
        echo "[FAIL] 이 PC의 Windows→WSL localhost 전달로 127.0.0.1:9090에 접속할 수 없습니다." >&2
        echo "       다음 명령으로 이 PC만 WSL NAT 주소를 사용하도록 전환한 뒤 01번을 다시 실행하세요." >&2
        echo "       ./data_collection/tools/00_prepare_internal_sensor_record.sh --bridge-ip $ROS_IP --patch-sensor-set" >&2
        exit 2
    fi
    echo "[PASS] Windows→WSL 고정 주소 127.0.0.1:9090 접속 확인"
fi
echo
if [[ "$RUN_MODE" == "scenario" ]]; then
    echo "[5/5] MORAI 차량 제어 연결을 기다립니다."
    echo "      MORAI Vehicle Information을 ${MORAI_ENDPOINT_IP}:9090으로 연결하세요."
    echo "      Camera/LiDAR/GPS/IMU, Sensor Record, Synchronization Mode는 꺼도 됩니다."
    deadline=$((SECONDS + CONNECT_TIMEOUT))
    while ! /usr/bin/python3 - <<'PYCTRL'
import rosgraph

try:
    _, subscribers, _ = rosgraph.Master(
        "/aimdrive_scenario_preflight"
    ).getSystemState()
except Exception:
    raise SystemExit(2)

raise SystemExit(
    0 if dict(subscribers).get("/ctrl_cmd", []) else 1
)
PYCTRL
    do
        if [[ "$STARTED_BRIDGE" == "1" ]] \
                && ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
            wait "$BRIDGE_PID" || true
            echo "[FAIL] 차량 제어 연결 대기 중 bridge가 종료됐습니다." >&2
            tail -n 40 "$BRIDGE_LOG" >&2 || true
            exit 2
        fi
        if (( SECONDS >= deadline )); then
            echo "[FAIL] ${CONNECT_TIMEOUT}초 동안 /ctrl_cmd 구독자가 연결되지 않았습니다." >&2
            echo "       MORAI Vehicle Information을 ${MORAI_ENDPOINT_IP}:9090으로 다시 연결하세요." >&2
            exit 2
        fi
        sleep 1
    done
    echo "[PASS] MORAI 차량 제어 연결 준비: /ctrl_cmd subscriber 확인"
    echo
    echo "[5/5] 시나리오 검증용 bridge를 계속 유지합니다."
    echo "      Camera/LiDAR/GPS/IMU, Sensor Record, Synchronization Mode는 꺼도 됩니다."
    echo "      Expert 실제 주행에는 MORAI 차량 정보와 객체 정보 연결이 필요합니다."
    echo "      다른 터미널에서 다음 자동 검증을 실행하세요:"
    echo "        cd $MORAI_PROJECT_ROOT"
    echo "        ./scenario_runner/tools/run_static_scenario_batch.sh --attempts 3"
    echo
    echo "      이 터미널은 자동 검증이 끝날 때까지 닫지 마세요. Ctrl+C로 종료합니다."
    keepalive_report_at=$((SECONDS + 30))
    ctrl_cmd_missing_count=0
    while true; do
        if [[ "$STARTED_BRIDGE" == "1" ]]; then
            if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
                wait "$BRIDGE_PID" || true
                echo "[FAIL] 시나리오 검증용 bridge가 종료됐습니다." >&2
                tail -n 40 "$BRIDGE_LOG" >&2 || true
                exit 2
            fi
        elif ! ss -H -ltn 'sport = :9090' 2>/dev/null | grep -q .; then
            echo "[FAIL] 기존 9090 bridge listener가 종료됐습니다." >&2
            exit 2
        fi
        if /usr/bin/python3 - <<'PYCTRL'
import rosgraph

try:
    _, subscribers, _ = rosgraph.Master(
        "/aimdrive_scenario_keepalive"
    ).getSystemState()
except Exception:
    raise SystemExit(2)

raise SystemExit(
    0 if dict(subscribers).get("/ctrl_cmd", []) else 1
)
PYCTRL
        then
            ctrl_cmd_missing_count=0
        else
            ((ctrl_cmd_missing_count += 1))
            if (( ctrl_cmd_missing_count >= 5 )); then
                echo "[FAIL] MORAI /ctrl_cmd 구독 연결이 5초 이상 끊겼습니다." >&2
                echo "       Vehicle Information을 ${MORAI_ENDPOINT_IP}:9090으로 다시 연결한 뒤 01번을 재실행하세요." >&2
                exit 2
            fi
        fi
        if (( SECONDS >= keepalive_report_at )); then
            echo "[KEEPALIVE] ROS master·bridge·/ctrl_cmd 정상; 전체 센서 없이 시나리오 검증 대기 중"
            keepalive_report_at=$((SECONDS + 30))
        fi
        sleep 1
    done
fi

echo "[5/5] 실제 메시지를 확인합니다."
echo "      MORAI에서 차량 제어(/ctrl_cmd)·차량 정보·Simulator Network(/clock)와"
echo "      Camera 1~4·LiDAR·GPS·IMU를"
echo "      ${MORAI_ENDPOINT_IP}:9090으로 연결하세요."
echo "      Simulation Time / Sensor Synchronization 10Hz / External Sync OFF"
echo "      /clock은 센서셋과 별도인 Simulator Network 연결에서 송신됩니다."
echo "      127.0.0.1은 Windows→WSL localhost 전달을 사용하므로 WSL 재시작 뒤에도 주소가 유지됩니다."
echo "      차량 정보와 Simulator Network도 센서셋과 같은 주소를 사용해야 합니다."
echo "      차량 정보 수신과 별도로 /ctrl_cmd 제어 명령 수신 연결도 확인합니다."
echo "      센서셋 IP가 이번 실행에서 바뀌었다면 MORAI에서 센서셋을 다시 불러온 뒤 연결하세요."

deadline=$((SECONDS + CONNECT_TIMEOUT))
bridge_stale_attempts=0
while (( SECONDS < deadline )); do
    if [[ "$STARTED_BRIDGE" == "1" ]] && ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
        wait "$BRIDGE_PID" || true
        echo "[FAIL] 연결 대기 중 bridge가 종료됐습니다." >&2
        tail -n 40 "$BRIDGE_LOG" >&2 || true
        exit 2
    fi

    temporary="$STATUS_FILE.tmp.$$"
    set +e
    /usr/bin/python3 "$INTERNAL/tools/probe_sensor_record_readiness.py" \
        --seconds 4 --raw-camera-transport bridge_dropped \
        >"$temporary" 2>"$STATUS_ERROR"
    probe_code=$?
    set -e
    if [[ ! -s "$temporary" ]]; then
        echo "[FAIL] 센서 상태 검사기가 보고서를 만들지 못했습니다." >&2
        cat "$STATUS_ERROR" >&2 || true
        exit 2
    fi
    mv "$temporary" "$STATUS_FILE"
    /usr/bin/python3 "$INTERNAL/tools/render_sensor_record_readiness.py" \
        --bridge-ip "$MORAI_ENDPOINT_IP" "$STATUS_FILE"

    if (( probe_code == 0 )); then
        echo
        echo "수집 실행(다른 터미널):"
        echo "  cd $MORAI_PROJECT_ROOT"
        echo "  ./data_collection/tools/02_collect_internal_sensor_record_laps.sh 3"
        if [[ "$STARTED_BRIDGE" == "1" ]]; then
            echo
            echo "이 01번 터미널은 bridge를 유지하므로 수집이 끝날 때까지 닫지 마세요."
            wait "$BRIDGE_PID"
        fi
        exit 0
    fi
    if grep -q 'BRIDGE_NOT_PATCHED' "$STATUS_FILE"; then
        echo "[FAIL] bridge 호환성 또는 실행 상태 검사에 실패했습니다." >&2
        echo "       위 bridge 진단의 구체적인 원인을 확인하세요." >&2
        exit 2
    fi
    if grep -q 'BRIDGE_HEALTH_STALE' "$STATUS_FILE"; then
        ((bridge_stale_attempts += 1))
        if (( bridge_stale_attempts >= 3 )); then
            echo "[FAIL] bridge health heartbeat가 3회 연속 stale 상태입니다." >&2
            echo "       bridge 프로세스와 ROS 연결 상태를 확인한 뒤 01번을 다시 실행하세요." >&2
            exit 2
        fi
        echo "[WAIT] bridge health heartbeat를 재확인합니다. (${bridge_stale_attempts}/3 stale)"
    else
        bridge_stale_attempts=0
    fi
    sleep 2
done

echo "[FAIL] ${CONNECT_TIMEOUT}초 안에 필수 stream이 준비되지 않았습니다." >&2
echo "       마지막 원본 보고서: $STATUS_FILE" >&2
exit 2
