#!/usr/bin/env bash
set -e
source "$(dirname "${BASH_SOURCE[0]}")/../tools/project_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/ve_ros_env.sh"
mkdir -p "$task_repo/artifacts/ve_live"
if [[ "${1:-}" == '--stop' ]]; then
    printf 'stop\n' > "$task_repo/artifacts/ve_live/roi.stop"
    echo 'Requested ROI viewer shutdown. Inference continues separately.'
    exit 0
fi
rm -f -- "$task_repo/artifacts/ve_live/roi.stop"
export PYTHONPATH="$task_repo/.runtime/ve_ros_deps:$PYTHONPATH"
if ! /usr/bin/python3 -c 'import pyproj' >/dev/null 2>&1; then
    if /usr/bin/python3 -m pip --version >/dev/null 2>&1; then
        /usr/bin/python3 -m pip install --disable-pip-version-check --target "$task_repo/.runtime/ve_ros_deps" 'pyproj==3.5.0'
    else
        # This ROS image has no pip. Install Linux cp38 wheels through Windows
        # pip into a private directory; never modify either system environment.
        /mnt/c/Users/dydwn/anaconda3/envs/vldrive/python.exe -m pip install --disable-pip-version-check \
            --target "$(wslpath -w "$task_repo/.runtime/ve_ros_deps")" \
            --platform manylinux2014_x86_64 --python-version 38 --implementation cp --abi cp38 \
            --only-binary=:all: 'pyproj==3.5.0'
    fi
fi
exec /usr/bin/python3 -u "$task_tools/ve_roi_viewer.py" --stop-file "$task_repo/artifacts/ve_live/roi.stop" "$@"
