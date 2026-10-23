"""Short, wall-timed, read-only ROS preflight before any scenario reset."""
import argparse
from collections import defaultdict
import json
import threading
import time

from bridge.bridge_health import check_bridge


def snapshot_observations(rows, lock):
    # Freeze observations before unregistering subscribers: unregister may let
    # another callback run, including a heartbeat newer than the cutoff.
    with lock:
        snapshot = {key: list(values) for key, values in rows.items()}
        end = time.monotonic_ns()
    return snapshot, end


def source_status(role, values, end):
    stamps = [value[1] for value in values]
    entry = dict(messages=len(values), source_unique=len(set(stamps)),
        receive_age_s=(end-values[-1][0])/1e9 if values else None,
        source_first_ns=stamps[0] if stamps else None,
        source_last_ns=stamps[-1] if stamps else None,
        source_reversals=sum(b<a for a,b in zip(stamps, stamps[1:])))
    failures = []
    if len(values)<2 or entry['receive_age_s']>1.5:
        failures.append('missing_or_stale:'+role)
    if len(values)>=2 and role in ('ego','objects','lidar','clock'):
        if entry['source_reversals']:
            failures.append('reversing_source:'+role)
        elif entry['source_unique']<2:
            failures.append('nonadvancing_source:'+role)
    return entry, failures


def control_status(state, topic='/ctrl_cmd'):
    # Ego/status publishing and simulator control subscribing are independent
    # rosbridge connections. Fresh sensors alone cannot prove drive readiness.
    code, message, topology = state
    row = dict(topic=topic, publishers=[], subscribers=[])
    if code != 1:
        row['error'] = str(message)
        return row, ['control_topology_unavailable']
    publishers, subscribers, _ = topology
    row.update(publishers=list(dict(publishers).get(topic, [])),
               subscribers=list(dict(subscribers).get(topic, [])))
    if not row['subscribers']:
        return row, ['missing_control_subscriber']
    # Publisher ownership is still checked by the scenario supervisor after
    # its bounded cleanup of a stopped expert between laps.
    return row, []


def clock_ingress_failure(clock_values, health, bridge_failures):
    if clock_values:
        return None
    # An absent/stale/incompatible heartbeat cannot locate a /clock outage.
    if not health or any(item.startswith('BRIDGE_') for item in bridge_failures):
        return 'clock_ingress_unavailable'
    clock = health.get('streams', {}).get('/clock', {})
    if clock.get('samples', 0)>=2 and clock.get('receive_age_ms', 1e9)<=500:
        return 'clock_ingress_not_delivered_to_ros'
    return 'morai_simulator_network_clock_not_sent'


def main():
    import rospy
    from morai_msgs.msg import EgoVehicleStatus, GPSMessage, ObjectStatusList, GetTrafficLightStatus
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CompressedImage, PointCloud2, Imu
    from std_msgs.msg import String

    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,default=4.)
    p.add_argument('--control-topic',default='/ctrl_cmd')
    p.add_argument('--raw-camera-transport',choices=('required','unrecorded','bridge_dropped','absent'),
                   default='required')
    a=p.parse_args()
    if not 2<=a.seconds<=10:p.error('Use 2..10 wall seconds')
    rows=defaultdict(list);lock=threading.Lock()
    rospy.init_node('native_sensor_preflight',anonymous=True,disable_signals=True)
    use_sim_time=rospy.get_param('/use_sim_time',False) is True
    def receive(role,message):
        now=time.monotonic_ns()
        if role=='bridge_health':
            with lock:rows[role].append((now,json.loads(message.data)))
            return
        if role=='clock':stamp=message.clock.to_nsec()
        else:stamp=message.header.stamp.to_nsec()
        if role=='camera':role+=':'+message.header.frame_id
        with lock:rows[role].append((now,stamp))
    subscribers=[rospy.Subscriber(topic,typ,lambda m,r=role:receive(r,m),queue_size=4,
                                 buff_size=4*1024*1024,tcp_nodelay=True)
        for role,topic,typ in [('ego','/Ego_topic',EgoVehicleStatus),('lidar','/lidar3D',PointCloud2),
                              ('gps','/gps',GPSMessage),('imu','/imu',Imu),('clock','/clock',Clock),
                              ('objects','/Object_topic',ObjectStatusList),
                              ('traffic_light','/GetTrafficLightStatus',GetTrafficLightStatus),
                              ('camera','/image_jpeg/compressed',CompressedImage),
                              ('bridge_health','/stage6e/bridge_health',String)]]
    started=time.monotonic()
    while time.monotonic()-started<a.seconds and not rospy.is_shutdown():time.sleep(.05)
    snapshot,end=snapshot_observations(rows,lock)
    for s in subscribers:s.unregister()
    camera_roles=['camera:Camera-'+str(i) for i in range(1,5)]
    # The local expert fork will hold its brake until fresh ObjectStatusList
    # receipts arrive, including valid empty lists. Traffic-light status is
    # reported when available but is not required before the start position.
    required=['ego','objects','lidar','gps','imu','clock']
    if a.raw_camera_transport in ('required','unrecorded'):required+=camera_roles
    report={};failures=[]
    if not use_sim_time:
        failures.append('use_sim_time_not_true')
    try:
        report['control'],control_failures=control_status(
            rospy.get_master().getSystemState(),topic=a.control_topic)
    except Exception as error:
        report['control']=dict(topic=a.control_topic,publishers=[],subscribers=[],
                               error=str(error))
        control_failures=['control_topology_unavailable']
    failures.extend(control_failures)
    for role in required:
        # Camera headers are not exposure clocks; only receipt/view presence is required here.
        report[role],source_failures=source_status(role,snapshot.get(role,[]),end)
        failures.extend(source_failures)
    traffic=snapshot.get('traffic_light',[])
    traffic_stamps=[value[1] for value in traffic]
    report['traffic_light']=dict(messages=len(traffic),required_at_start=False,
        receive_age_s=(end-traffic[-1][0])/1e9 if traffic else None,
        source_unique=len(set(traffic_stamps)),
        source_reversals=sum(b<a for a,b in zip(traffic_stamps,traffic_stamps[1:])))
    if a.raw_camera_transport in ('bridge_dropped','absent'):
        for role in camera_roles:
            values=snapshot.get(role,[])
            report[role]=dict(messages=len(values),expected_absent=True)
            if values:failures.append('unexpected_ros_camera_transport:'+role)
    if a.raw_camera_transport=='bridge_dropped':
        health_rows=snapshot.get('bridge_health',[])
        health=health_rows[-1][1] if health_rows else {}
        bridge_failures=check_bridge(health,now_ns=end,require_streams=True)
        bridge_clock=health.get('streams',{}).get('/clock',{})
        report['bridge_clock_ingress']=dict(bridge_clock)
        report['bridge_health']=dict(
            messages=len(health_rows),
            receive_age_s=(end-health_rows[-1][0])/1e9 if health_rows else None,
            heartbeat_age_s=(end-health['monotonic_ns'])/1e9 if 'monotonic_ns' in health else None,
            heartbeat_monotonic_ns=health.get('monotonic_ns'),
            evaluated_at_monotonic_ns=end,
            failures=[item for item in bridge_failures if item.startswith('BRIDGE_')],
        )
        clock_failure=clock_ingress_failure(snapshot.get('clock'),health,bridge_failures)
        if clock_failure:failures.append(clock_failure)
        report['bridge_camera_ingress']=dict(
            messages=len(health_rows),
            policy=health.get('camera_ingress_policy'),
            purpose='connection_health_only',
            rate_scope='per_frame_id_on_shared_ros_topic',
            validates_native_sensor_record_cadence=False,
            expected_native_sensor_record_hz=10,
            implementation=health.get('implementation'),
            code_sha256=health.get('code_sha256'),
            streams={role:health.get('streams',{}).get(role,{})
                     for role in ('Camera-1','Camera-2','Camera-3','Camera-4')},
        )
        failures.extend(bridge_failures)
        if health.get('camera_ingress_policy')!='drop_before_ros':
            failures.append('bridge_camera_ingress_policy_mismatch')
        for role in ('Camera-1','Camera-2','Camera-3','Camera-4'):
            stream=health.get('streams',{}).get(role,{})
            if (stream.get('samples',0)<2
                    or stream.get('dropped_before_ros')!=stream.get('samples')):
                failures.append('bridge_camera_ingress_not_fully_dropped:'+role)
    rospy.signal_shutdown('bounded preflight complete')
    print(json.dumps(dict(ready=not failures,failures=failures,streams=report,
        raw_camera_transport=a.raw_camera_transport,
                         use_sim_time=use_sim_time,
                         cross_sensor_clock_equality_assumed=False,physical_simultaneity_verified=False)),flush=True)
    if failures:raise SystemExit(2)


if __name__=='__main__':main()
