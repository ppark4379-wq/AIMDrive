#!/usr/bin/env bash
# Shared path bootstrap; sourcing this file starts no ROS node or simulator.
SCENARIO_RUNNER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENARIO_AIM_ROOT="$(cd "$SCENARIO_RUNNER_ROOT/../.." && pwd)"
SCENARIO_AIM_PARENT="$(cd "$SCENARIO_AIM_ROOT/.." && pwd)"
SCENARIO_SDK_ROOT="$SCENARIO_RUNNER_ROOT/third_party/grpc_inha_univ/src"

scenario_export_project_paths() {
    export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
    local deps="${AIM2026_PYTHON_DEPS:-${MORAI_ASYNC_DEPS:-}}"
    export PYTHONPATH="${deps:+$deps:}$SCENARIO_AIM_PARENT:$SCENARIO_AIM_ROOT/utils:$SCENARIO_SDK_ROOT:$SCENARIO_SDK_ROOT/proto"
}

# Keep historical launcher function names while using runner-owned setup.
morai_runtime_value() {
    /usr/bin/python3 -B - "$1" <<'PY'
import sys
from scenario_runner.utils import environment
if sys.argv[1] not in ('ASYNC_DEPS', 'ANALYSIS_PYTHON', 'ROS_ROOT', 'CATKIN_ROOT'):
    raise SystemExit('Unsupported runner environment setting')
print(getattr(environment, sys.argv[1]))
PY
}

morai_load_internal_sensor_record_environment() {
    export MORAI_ROS_ROOT="${MORAI_ROS_ROOT:-/opt/ros/noetic}"
    export MORAI_CATKIN_WS="${MORAI_CATKIN_WS:-$SCENARIO_AIM_ROOT/Expert/ros}"
    if [[ ! -f "$MORAI_ROS_ROOT/setup.bash" ]]; then
        echo "[FAIL] ROS setup을 찾지 못했습니다: $MORAI_ROS_ROOT/setup.bash" >&2
        return 2
    fi
    # Do not let catkin's underlay traversal inspect an inherited old checkout.
    unset CMAKE_PREFIX_PATH ROS_PACKAGE_PATH
    set +u
    source "$MORAI_ROS_ROOT/setup.bash"
    set -u
    export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
    # Generated catkin setup files embed the workspace used at build time.
    # Select this workspace directly so a moved checkout never reads that tree.
    export CMAKE_PREFIX_PATH="$MORAI_CATKIN_WS/devel:$MORAI_ROS_ROOT"
    export ROS_PACKAGE_PATH="$MORAI_CATKIN_WS/src:$MORAI_ROS_ROOT/share"
    local native_library_path="$MORAI_CATKIN_WS/devel/lib:$MORAI_ROS_ROOT/lib"
    local library_entry
    local -a inherited_library_paths
    IFS=: read -r -a inherited_library_paths <<< "${LD_LIBRARY_PATH:-}"
    for library_entry in "${inherited_library_paths[@]}"; do
        case "$library_entry" in
            ""|*/MoraiDrive|*/MoraiDrive/*|*/VLDrive|*/VLDrive/*|*/_support|*/_support/*) continue ;;
        esac
        native_library_path+=":$library_entry"
    done
    export LD_LIBRARY_PATH="$native_library_path"
    scenario_export_project_paths
    export PYTHONPATH="$MORAI_CATKIN_WS/devel/lib/python3/dist-packages:$MORAI_ROS_ROOT/lib/python3/dist-packages:$PYTHONPATH"
    ROS_IP="$(/usr/bin/python3 -B -c 'from scenario_runner.utils.environment import ros_ip; print(ros_ip())')"
    export ROS_IP
}

morai_require_capture_python() {
    /usr/bin/python3 -B -c 'import grpc, yaml, numpy, scipy, rospy; import morai_msgs.msg'
}
scenario_export_project_paths
