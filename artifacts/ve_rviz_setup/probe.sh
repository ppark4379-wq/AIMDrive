#!/usr/bin/env bash
source /mnt/c/Users/dydwn/AIMDrive/MoraiDrive/expert/tools/ve_ros_env.sh
command -v rviz
printenv DISPLAY WAYLAND_DISPLAY
/usr/bin/python3 - <<'PY'
import os, time, xmlrpc.client
import rospy, numpy, scipy, pyproj
import morai_msgs.msg, visualization_msgs.msg
master=xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI'])
print(master.getSystemState('/ve_setup_probe'))
print('DISPLAY',os.environ.get('DISPLAY'))
a,b=time.monotonic(),time.time()
time.sleep(2)
print('CLOCK',time.monotonic()-a,time.time()-b)
PY
