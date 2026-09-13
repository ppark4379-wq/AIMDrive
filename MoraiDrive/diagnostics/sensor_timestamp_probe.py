"""Read-only ROS header probe. Keeps bounded headers, never sensor payloads."""
import argparse
from collections import deque
import json
from pathlib import Path
import socket
import statistics
import struct
import threading
import time
import xmlrpc.client


def describe(values):
    if not values:
        return None
    return {"min":min(values),"median":statistics.median(values),"max":max(values)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration',type=float,default=15)
    parser.add_argument('--output',default='')
    parser.add_argument('--pause-test',action='store_true',help='Pause MORAI for 3 seconds, then resume; no mode/scene change')
    args=parser.parse_args()
    import rospy
    rospy.init_node('sensor_timestamp_probe',anonymous=True,disable_signals=True)
    topic_types=dict(rospy.get_published_topics())
    wanted_types={'sensor_msgs/CompressedImage','sensor_msgs/Image','sensor_msgs/PointCloud2',
                  'sensor_msgs/LaserScan','sensor_msgs/Imu','sensor_msgs/NavSatFix',
                  'morai_msgs/GPSMessage','morai_msgs/EgoVehicleStatus'}
    topics={name:kind for name,kind in topic_types.items() if kind in wanted_types or name=='/clock'}
    params={key:rospy.get_param(key,None) for key in ['/use_sim_time',
       '/morai_drive/sync_runtime_active','/morai_drive/scenario_ready']}
    lock=threading.Lock()
    records={name:deque(maxlen=1000) for name in topics}
    counts={name:0 for name in topics}
    errors={}
    clock=[None]
    started=time.time()
    def receive(message,topic):
        wall=time.time_ns()
        mono=time.monotonic_ns()
        raw=message._buff
        try:
            if topic=='/clock':
                sec,nsec=struct.unpack_from('<II',raw)
                item={'recv_wall_ns':wall,'recv_mono_ns':mono,'stamp_ns':sec*1000000000+nsec}
                with lock: clock[0]=item['stamp_ns']
            else:
                seq,sec,nsec,size=struct.unpack_from('<IIII',raw)
                if size>4096 or len(raw)<16+size or nsec>=1000000000:
                    raise ValueError('unexpected Header layout')
                frame_id=bytes(raw[16:16+size]).decode('utf-8',errors='replace')
                item={'recv_wall_ns':wall,'recv_mono_ns':mono,'seq':seq,
                      'stamp_ns':sec*1000000000+nsec,'frame_id':frame_id,
                      'serialized_bytes':len(raw),'callerid':message._connection_header.get('callerid'),
                      'latest_clock_ns':clock[0]}
            with lock:
                counts[topic]+=1
                records[topic].append(item)
        except Exception as error:
            with lock: errors[topic]=str(error)
    subscribers=[rospy.Subscriber(topic,rospy.AnyMsg,receive,callback_args=topic,
                    queue_size=1,buff_size=4*1024*1024) for topic in topics]
    print(json.dumps({'published_topics':topic_types,'observed_topics':topics,'params':params}),flush=True)
    events=[]
    grpc_samples=[]
    channel=None
    possibly_paused=False
    simulator=None
    simulation=None
    empty=None
    if args.pause_test:
        import sys
        import grpc
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'data_collection/third_party/grpc_inha_univ/src/proto'))
        from morai.common.type_pb2 import Empty
        from morai.simulator.simulator_pb2_grpc import SimulatorStub
        from morai.simulation.simulation_pb2_grpc import SimulationStub
        channel=grpc.insecure_channel('172.19.0.1:7789')
        simulator=SimulatorStub(channel)
        simulation=SimulationStub(channel)
        empty=Empty()
    def action(name):
        before=time.time_ns()
        reply=getattr(simulation,name)(empty,timeout=2)
        event={'action':name,'before_wall_ns':before,'after_wall_ns':time.time_ns(),'reply':str(reply)}
        events.append(event)
        print(json.dumps(event),flush=True)
        if reply.status!=1:
            raise RuntimeError('MORAI action failed: '+str(reply))
    try:
        begin=time.monotonic()
        end=begin+max(12 if args.pause_test else 1,min(args.duration,60))
        next_query=begin
        pause_done=False
        resume_done=False
        if simulator:
            events.append({'version':simulator.GetSimulatorVersion(empty,timeout=2).value,
                           'sync_mode':str(simulation.GetSynchronousMode(empty,timeout=2))})
        while time.monotonic()<end and not rospy.is_shutdown():
            current=time.monotonic()
            if simulation and not pause_done and current-begin>=4:
                possibly_paused=True
                action('Pause')
                pause_done=True
            if simulation and not resume_done and current-begin>=7:
                action('Resume')
                possibly_paused=False
                resume_done=True
            if simulator and current>=next_query:
                before=time.time_ns()
                value=simulator.GetTimestamp(empty,timeout=2).value
                grpc_samples.append({'before_wall_ns':before,'after_wall_ns':time.time_ns(),
                                     'timestamp_raw':value,'paused':possibly_paused})
                next_query=current+.5
            time.sleep(.05)
    finally:
        if possibly_paused:
            try:
                action('Resume')
            except Exception as error:
                print('RESUME FAILED: '+str(error),flush=True)
        if channel:
            channel.close()
        for subscriber in subscribers: subscriber.unregister()
    with lock:
        samples={name:list(items) for name,items in records.items()}
        count_copy=dict(counts)
    summaries={}
    for name,items in samples.items():
        stamps=[item['stamp_ns'] for item in items]
        summaries[name]={'count':count_copy[name],'retained':len(items),
          'first':items[0] if items else None,'last':items[-1] if items else None,
          'zero_stamps':sum(stamp==0 for stamp in stamps),
          'stamp_step_ms':describe([(b-a)/1e6 for a,b in zip(stamps,stamps[1:])]),
          'receive_interval_ms':describe([(b['recv_mono_ns']-a['recv_mono_ns'])/1e6 for a,b in zip(items,items[1:])]),
          'wall_minus_stamp_ms':describe([(item['recv_wall_ns']-item['stamp_ns'])/1e6 for item in items])}
    output=Path(args.output) if args.output else Path(__file__).parent / ('sensor_timestamp_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    output.write_text(json.dumps({'started_wall':started,'ended_wall':time.time(),
        'topics':topics,'params':params,'events':events,'grpc_samples':grpc_samples,'summary':summaries,'errors':errors,'samples':samples},indent=2),encoding='utf-8')
    print(json.dumps({'output':str(output),'summary':summaries,'errors':errors},indent=2),flush=True)
    rospy.signal_shutdown('bounded timestamp observation complete')

if __name__=='__main__':
    main()
