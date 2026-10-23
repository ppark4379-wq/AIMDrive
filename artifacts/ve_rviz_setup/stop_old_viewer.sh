#!/usr/bin/env bash
source /mnt/c/Users/dydwn/AIMDrive/AIM2026/Expert/tools/ve_ros_env.sh
/usr/bin/python3 - <<'PY'
import rosnode
print(rosnode.kill_nodes(['/ve_roi_viewer']))
PY
