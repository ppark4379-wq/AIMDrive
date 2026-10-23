#!/usr/bin/env bash
source /mnt/c/Users/dydwn/AIMDrive/MoraiDrive/expert/tools/ve_ros_env.sh
export PYTHONPATH="$task_repo/.runtime/ve_ros_deps:$PYTHONPATH"
/usr/bin/python3 "$task_repo/MoraiDrive/expert/src/expert/tests/test_ve_roi_viewer.py"
/usr/bin/python3 - <<'PY'
import os,xmlrpc.client
print(xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getSystemState('/ve_roi_setup'))
PY
