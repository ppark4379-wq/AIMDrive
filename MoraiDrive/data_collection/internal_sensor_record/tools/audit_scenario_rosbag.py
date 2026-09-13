"""Read every message of a closed scenario bag; preserve original bytes/times.

Bag iteration is sorted by ROS receipt indexing time, not monotonic arrival.
Use the companion raw journal to diagnose actual receive order and latency.
This audit does not certify camera exposure synchronization.
"""
import argparse
from array import array
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from async_capture.rosbag_recording import TOPICS, REQUIRED_STREAMS
from ssd_storage import require_ssd


def header(payload):
    if len(payload) < 16:
        raise ValueError('Truncated ROS header')
    seq, sec, nsec, length = struct.unpack_from('<IIII', payload)
    if nsec >= 1000000000 or length > 4096 or 16+length > len(payload):
        raise ValueError('Invalid ROS header')
    return seq, sec*1000000000+nsec, payload[16:16+length].decode('utf8'), 16+length


def stats(values):
    if not values:
        return {'count': 0}
    deltas = [b-a for a,b in zip(values, values[1:])]
    positive = sorted(x for x in deltas if x > 0)
    return dict(count=len(values), first_ns=values[0], last_ns=values[-1],
                span_s=(max(values)-min(values))/1e9,
                repeated_adjacent=sum(x==0 for x in deltas),
                regressions_in_bag_iteration=sum(x<0 for x in deltas),
                positive_step_p50_ms=positive[len(positive)//2]/1e6 if positive else None,
                positive_step_p99_ms=positive[min(len(positive)-1,int(len(positive)*.99))]/1e6 if positive else None,
                max_positive_gap_ms=max(positive)/1e6 if positive else None)


def distance(a,b,scale=1):
    return sum((getattr(a,k)/scale-getattr(b,k))**2 for k in ('x','y','z'))**.5


def audit(directory):
    import rosbag
    from PIL import Image
    directory=Path(directory).resolve()
    require_ssd(directory,reserve_bytes=0)
    original=directory/'sensors.bag'
    recording=json.loads((directory/'recording.json').read_text())
    if recording['status']!='recorded' or Path(str(original)+'.active').exists():
        raise ValueError('Only successfully finalized recordings can be audited')
    if (directory/'audit.json').exists() or (directory/'replay_original.bag').exists():
        raise FileExistsError('Do not overwrite a previous audit')
    counts=Counter(); sizes=Counter(); types={}; dimensions={}; decoded=Counter()
    stamps=defaultdict(lambda:array('q')); bag_times=defaultdict(lambda:array('q'))
    seqs=defaultdict(lambda:array('q')); egos=defaultdict(list); replay=[]; clock=array('q')
    coverage=Counter(); errors=[]
    replay_path=directory/'replay_original.bag'
    before=original.stat()
    with replay_path.open('xb') as output, rosbag.Bag(str(original),'r') as bag, rosbag.Bag(output,'w') as rb:
        expected=bag.get_message_count()
        for topic,raw,t,connection in bag.read_messages(raw=True, return_connection_header=True):
            datatype,payload,md5,position,cls=raw
            counts[topic]+=1; sizes[topic]+=len(payload); types[topic]=datatype
            key=topic; bag_ns=t.to_nsec()
            if cls._has_header:
                seq,stamp,frame,offset=header(payload)
                if topic=='/image_jpeg/compressed':
                    key=frame
                    counts[key]+=1
                stamps[key].append(stamp);seqs[key].append(seq)
            bag_times[key].append(bag_ns)
            if topic=='/image_jpeg/compressed':
                format_length=struct.unpack_from('<I',payload,offset)[0]
                cursor=offset+4+format_length
                length=struct.unpack_from('<I',payload,cursor)[0]
                image=payload[cursor+4:]
                if length!=len(image) or not image.startswith(b'\xff\xd8') or not image.endswith(b'\xff\xd9'):
                    raise ValueError('Invalid compressed JPEG payload')
                if counts[key]==1 or counts[key]%250==0:
                    with Image.open(io.BytesIO(image)) as im:
                        im.load(); dimensions[key]=list(im.size);decoded[key]+=1
            elif topic=='/lidar3D':
                msg=cls().deserialize(payload)
                if len(msg.data)!=msg.row_step*msg.height or msg.row_step<msg.point_step*msg.width:
                    raise ValueError('PointCloud2 layout mismatch')
            elif topic=='/clock':
                sec,nsec=struct.unpack('<II',payload)
                if nsec>=1000000000:raise ValueError('Invalid /clock nsecs')
                clock.append(sec*1000000000+nsec)
            elif topic=='/Ego_topic':
                msg=cls().deserialize(payload);egos[stamp].append((msg,bag_ns))
            elif topic=='/ReplayInfo_topic':
                msg=cls().deserialize(payload);replay.append((msg,bag_ns))
                # Exact original messages and original indexing time, including
                # duplicate states. No source-time rewriting or deduplication.
                rb.write(topic,raw,t,raw=True,connection_header=connection)
            elif topic=='/Object_topic':
                msg=cls().deserialize(payload)
                for kind in ('npc_list','pedestrian_list','obstacle_list'):
                    if getattr(msg,kind,[]):coverage[kind+'_nonempty_messages']+=1
            elif topic=='/GetTrafficLightStatus':
                msg=cls().deserialize(payload)
                coverage['traffic_status_'+str(msg.trafficLightStatus)]+=1
        if sum(counts[t] for t in types)!=expected:
            raise ValueError('Full traversal count differs from bag index')
    paired=0;missing_ego=0; max_position=0.;max_heading=0.;max_velocity=0.;max_dt=0.
    for msg,receipt in replay:
        actors=[x for x in msg.npc_list if x.type==-1]
        if len(actors)!=1:
            missing_ego+=1;continue
        candidates=egos.get(msg.header.stamp.to_nsec(),[])
        if not candidates:continue
        ego,ego_receipt=min(candidates,key=lambda v:abs(v[1]-receipt)); actor=actors[0];paired+=1
        max_position=max(max_position,distance(actor.position,ego.position))
        max_heading=max(max_heading,abs((actor.heading-ego.heading+180)%360-180))
        max_velocity=max(max_velocity,distance(actor.velocity,ego.velocity,3.6))
        max_dt=max(max_dt,abs(receipt-ego_receipt)/1e6)
    digest=hashlib.sha256()
    with original.open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):digest.update(data)
    after=original.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Original bag mutated during audit')
    missing=[k for k in REQUIRED_STREAMS if counts[k]<3]
    if missing:errors.append('Missing required streams: '+str(missing))
    log=(directory/'record.log').read_text(errors='replace')
    warnings=[line for line in log.splitlines() if any(s in line.lower() for s in ('buffer exceeded','dropping','no space','error','warn'))]
    if any(any(term in line.lower() for term in ('buffer exceeded','dropping','no space','error')) for line in warnings):
        errors.append('Recorder log reports message loss or recording errors')
    outcome=json.loads((directory.parent/'outcome.json').read_text())
    if not outcome.get('full_mission_completed'):errors.append('Scenario full lap incomplete')
    if missing_ego:errors.append('ReplayInfo lacks unique Ego entry')
    if paired<3 or max_position>.01 or max_heading>.01 or max_velocity>.01:
        errors.append('Replay/Ego comparison failed')
    with rosbag.Bag(str(replay_path),'r') as extracted:
        if extracted.get_message_count()!=len(replay):raise ValueError('Replay extraction count mismatch')
    report=dict(schema='morai_scenario_bag_audit_v1',status='recording_verified' if not errors else 'review_required',
        errors=errors,bag=str(original),bytes=before.st_size,sha256=digest.hexdigest(),fully_read_messages=expected,
        counts=dict(counts),message_types=types,payload_bytes=dict(sizes),
        requested_topics_without_messages=[t for t in TOPICS if not counts[t]],
        source_header_timing={k:stats(v) for k,v in stamps.items()},
        bag_ros_receipt_timing={k:stats(v) for k,v in bag_times.items()},clock_values=stats(clock),
        camera_dimensions=dimensions,camera_decode_sample_counts=dict(decoded),coverage=dict(coverage),
        sequence_stats={k:dict(count=len(v),nonincrease=sum(b<=a for a,b in zip(v,v[1:])),
            skipped_numbers=sum(max(0,b-a-1) for a,b in zip(v,v[1:]))) for k,v in seqs.items()},
        replay_ego_comparison=dict(exact_source_stamp_pairs=paired,missing_ego=missing_ego,
            max_position_difference_m=max_position,max_heading_difference_deg=max_heading,
            max_velocity_difference_mps=max_velocity,max_bag_receipt_difference_ms=max_dt),
        recorder_log_warnings=warnings,full_mission_completed=outcome.get('full_mission_completed',False),
        replay_extract=str(replay_path),replay_extract_recipe='Original messages and bag times; no deduplication or restamping',
        bag_iteration_order='Sorted ROS receipt index; does not certify monotonic receive order',
        receive_monotonic_evidence='../raw and ../quality',source_clock_identity_assumed=False,
        physical_simultaneity_verified=False,training_visible=False)
    with (directory/'audit.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('status','errors','bytes','counts','requested_topics_without_messages','replay_ego_comparison','recorder_log_warnings')},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    result=audit(parser.parse_args().directory)
    if result['errors']:raise SystemExit(2)
