#!/usr/bin/env bash
# Keep this sourced helper LF-only; Windows Git checkouts are supported.
# Shared, deterministic environment for the three Sensor Record entrypoints.

MORAI_COLLECTION_TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MORAI_DATA_COLLECTION_ROOT="$(cd "$MORAI_COLLECTION_TOOLS/.." && pwd)"
MORAI_INTERNAL_SENSOR_RECORD_ROOT="$MORAI_DATA_COLLECTION_ROOT/internal_sensor_record"
export MORAI_PROJECT_ROOT="${MORAI_PROJECT_ROOT:-$(cd "$MORAI_DATA_COLLECTION_ROOT/.." && pwd)}"

morai_runtime_value() {
    local name="$1"
    MORAI_PROJECT_ROOT="$MORAI_PROJECT_ROOT" \
        PYTHONPATH="$MORAI_INTERNAL_SENSOR_RECORD_ROOT" \
        /usr/bin/python3 -c "from runtime_paths import $name; print($name)"
}

morai_load_internal_sensor_record_environment() {
    export MORAI_ROS_ROOT="${MORAI_ROS_ROOT:-$(morai_runtime_value ROS_ROOT)}"
    export MORAI_CATKIN_WS="${MORAI_CATKIN_WS:-$(morai_runtime_value CATKIN_ROOT)}"
    export MORAI_LAUNCHER_ROOT="${MORAI_LAUNCHER_ROOT:-$(morai_runtime_value LAUNCHER_ROOT)}"
    export MORAI_POWERSHELL_EXE="${MORAI_POWERSHELL_EXE:-$(morai_runtime_value POWERSHELL_EXE)}"
    export MORAI_ASYNC_DEPS="${MORAI_ASYNC_DEPS:-$(morai_runtime_value ASYNC_DEPS)}"
    export MORAI_ANALYSIS_PYTHON="${MORAI_ANALYSIS_PYTHON:-$(morai_runtime_value ANALYSIS_PYTHON)}"
    export MORAI_SENSOR_SET_PATH="${MORAI_SENSOR_SET_PATH:-$(morai_runtime_value SENSOR_SET_PATH)}"

    local ros_setup="$MORAI_ROS_ROOT/setup.bash"
    local catkin_setup="$MORAI_CATKIN_WS/devel/setup.bash"
    if [[ ! -f "$ros_setup" ]]; then
        echo "[FAIL] ROS1 Noetic setup을 찾지 못했습니다: $ros_setup" >&2
        echo "       ./data_collection/tools/00_prepare_internal_sensor_record.sh 를 먼저 실행하세요." >&2
        return 2
    fi
    if [[ ! -f "$catkin_setup" ]]; then
        echo "[FAIL] morai_msgs가 빌드된 catkin setup을 찾지 못했습니다: $catkin_setup" >&2
        echo "       PC별 catkin 경로를 00번 초기화에서 지정하세요." >&2
        return 2
    fi

    set +u
    source "$ros_setup"
    source "$catkin_setup"
    set -u

    export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="$MORAI_ASYNC_DEPS:$MORAI_INTERNAL_SENSOR_RECORD_ROOT:$MORAI_PROJECT_ROOT:$MORAI_CATKIN_WS/devel/lib/python3/dist-packages:$MORAI_ROS_ROOT/lib/python3/dist-packages"

    if [[ -z "${MORAI_BRIDGE_IP:-}" ]]; then
        MORAI_BRIDGE_IP="$(/usr/bin/python3 -c 'from network_config import bridge_ip; print(bridge_ip())')"
    fi
    export MORAI_BRIDGE_IP
    export ROS_IP="$MORAI_BRIDGE_IP"
    export MORAI_STORAGE_ROOT="${MORAI_STORAGE_ROOT:-$(/usr/bin/python3 -c 'from ssd_storage import SSD_ROOT; print(SSD_ROOT)')}"
    export MORAI_INTERNAL_ENV_READY=1
}

morai_prepare_internal_sensor_record_environment() {
    local mode="${1:-runtime}"
    local command=("$MORAI_COLLECTION_TOOLS/00_prepare_internal_sensor_record.sh")
    if [[ "$mode" == "sensor-set" ]]; then
        command+=(--patch-sensor-set)
    elif [[ "$mode" != "runtime" ]]; then
        echo "[FAIL] 알 수 없는 자동 준비 모드: $mode" >&2
        return 2
    fi
    "${command[@]}"
}

morai_require_capture_python() {
    if ! /usr/bin/python3 -c \
            'import grpc,google.protobuf,psutil,yaml,numpy,rospy,morai_msgs' \
            >/dev/null 2>&1; then
        echo "[FAIL] capture Python 의존성이 준비되지 않았습니다." >&2
        echo "       ./data_collection/tools/00_prepare_internal_sensor_record.sh 를 실행하세요." >&2
        return 2
    fi
}

morai_require_analysis_python() {
    if [[ ! -x "$MORAI_ANALYSIS_PYTHON" ]] || ! "$MORAI_ANALYSIS_PYTHON" -c \
            'import cv2,numpy,pyproj,scipy,PIL' >/dev/null 2>&1; then
        echo "[FAIL] 분석 Python 의존성이 준비되지 않았습니다: $MORAI_ANALYSIS_PYTHON" >&2
        echo "       ./data_collection/tools/00_prepare_internal_sensor_record.sh 를 실행하세요." >&2
        return 2
    fi
}
