#!/usr/bin/env bash
source /mnt/c/Users/dydwn/AIMDrive/AIM2026/Expert/tools/ve_ros_env.sh
/usr/bin/python3 - <<'PY'
import json,time
from collections import Counter
import rospy
from sensor_msgs.msg import CompressedImage
rospy.init_node('ve_camera_status_check',anonymous=True,disable_rosout=True)
counts=Counter()
stamps={}
def callback(message):
    role=message.header.frame_id
    counts[role]+=1
    stamps[role]=message.header.stamp.to_nsec()
subscriber=rospy.Subscriber('/image_jpeg/compressed',CompressedImage,callback,queue_size=8,buff_size=32*1024*1024)
start=time.time()
while time.time()-start<5:
    time.sleep(.1)
print(json.dumps(dict(observed_seconds=time.time()-start,received=dict(counts),latest_source_stamps=stamps)))
subscriber.unregister()
rospy.signal_shutdown('read-only camera check complete')
PY
