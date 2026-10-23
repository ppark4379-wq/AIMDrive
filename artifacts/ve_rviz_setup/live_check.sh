#!/usr/bin/env bash
source /mnt/c/Users/dydwn/AIMDrive/MoraiDrive/expert/tools/ve_ros_env.sh
/usr/bin/python3 "$task_repo/artifacts/ve_rviz_setup/live_check.py"
/usr/bin/python3 "$task_repo/artifacts/ve_rviz_setup/capture_rviz.py"
