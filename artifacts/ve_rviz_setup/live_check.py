import json
from pathlib import Path
import time
import rospy
from morai_msgs.msg import ObjectStatusList, CtrlCmd
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

rospy.init_node('ve_roi_live_check',anonymous=True)
result=dict(objects=0,diagnostics=0,markers=0,actual_control_messages=0,frames=[],samples=[])
active_markers={}
def objects(message):
    result['objects']+=1
    result['last_objects']=[dict(id=o.unique_id,speed_kmh=o.velocity.x,x=o.position.x,y=o.position.y)
                            for o in list(message.npc_list)+list(message.obstacle_list)]
def health(message):
    value=json.loads(message.data)
    if value.get('ready'):
        result['diagnostics']+=1
        result['last_diagnostics']=value
def markers(message):
    result['markers']+=1
    for m in message.markers:
        if m.action==3:active_markers.clear()
        elif m.action==2:active_markers.pop((m.ns,m.id),None)
        elif m.action==0:active_markers[(m.ns,m.id)]=m
    result['frames']=list(set(m.header.frame_id for m in active_markers.values()))
    current=[m for m in active_markers.values() if m.ns=='current_bbox']
    texts=[m.text for m in active_markers.values() if m.ns in ('object_speed','status','ego_label')]
    result['ego_bbox_seen']=any(m.ns=='ego_bbox' for m in active_markers.values())
    if len(result['samples']) < 20:
        result['samples'].append(dict(boxes=len(current),texts=texts))
def control(message):result['actual_control_messages']+=1
subscriptions=[rospy.Subscriber('/ve/Object_topic',ObjectStatusList,objects,queue_size=1),
               rospy.Subscriber('/ve/diagnostics',String,health,queue_size=1),
               rospy.Subscriber('/ve/roi_markers',MarkerArray,markers,queue_size=1),
               rospy.Subscriber('/ctrl_cmd',CtrlCmd,control,queue_size=1)]
start=time.time()
while time.time()-start<8:
    time.sleep(.1)
result['wall_seconds']=time.time()-start
result['local_clock_note']='WSL local duration is diagnostic only; use Windows-clock NEW/ROS matched rates in the status text.'
Path('/mnt/c/Users/dydwn/AIMDrive/artifacts/ve_rviz_setup/live_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('last_diagnostics','samples')}))
