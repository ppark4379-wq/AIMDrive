"""Shared VE UDP to ROS transport; never publishes vehicle control."""

from scenario_runner.project_paths import REPO_ROOT
import sys as _vl_sys
_vl_sys.path.insert(0, str(REPO_ROOT / 'VLDrive'))
import copy
from collections import deque
import json
import math
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

from inference.ve_object_tracking import VeObjectTracker
from inference.ve_objects_udp import PORT, MAX_PACKET_BYTES, decode_packet, make_object_message
from inference.realtime_ve_objects import CHECKPOINTS
from scenario_runner.utils.environment import build_analysis_environment
from scenario_runner.utils.environment import ANALYSIS_PYTHON
from scenario_runner.legacy_expert.run_ve_udp_expert import stop_child

REPO = REPO_ROOT


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def object_rows(message):
    rows = []
    for kind, objects in (('npc',message.npc_list),('obstacle',message.obstacle_list),('pedestrian',message.pedestrian_list)):
        for o in objects:
            values = [o.position.x,o.position.y,o.heading,o.size.x,o.size.y,o.velocity.x]
            if not all(math.isfinite(v) for v in values) or min(o.size.x,o.size.y) <= 0.:
                raise ValueError('Invalid GT/VE object geometry')
            rows.append(dict(center_world=[o.position.x,o.position.y], heading_deg=o.heading,
                length_m=o.size.x,width_m=o.size.y,speed_mps=abs(o.velocity.x)/3.6,
                track_id=str(o.unique_id) if kind != 'obstacle' else kind+':'+str(o.unique_id),name=o.name))
    return rows


class PerceptionSession:
    def __init__(self, output, label, observe_gt=True, camera_receiver_mode=None, signal_head=None):
        import rospy
        from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList, ObjectStatus
        self.ros, self.message_type, self.object_type = rospy, ObjectStatusList, ObjectStatus
        self.output, self.checkpoint = output, CHECKPOINTS[label]
        self.signal_head = signal_head
        self.signal_sink = None
        self.observe_gt = observe_gt
        self.camera_receiver_mode = camera_receiver_mode
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.packet, self.gt, self.ego = None, None, None
        self.gt_time = self.ego_time = 0.
        self.error, self.sender, self.thread = None, None, None
        self.sock, self.publisher, self.subs = None, None, []
        self.trace = self.log = None
        self.transport_log = None
        self.delivery_window = deque(maxlen=120)
        self.sequence, self.rejected = -1, 0
        self.generation, self.tracker = str(uuid.uuid4()), VeObjectTracker()

    def start(self):
        from morai_msgs.msg import ObjectStatusList, EgoVehicleStatus
        self.sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1',PORT))
        self.sock.settimeout(.1)
        self.publisher = self.ros.Publisher('/ve/Object_topic', ObjectStatusList, queue_size=1)
        self.subs = [self.ros.Subscriber('/Ego_topic',EgoVehicleStatus,self.on_ego,queue_size=1)]
        if self.observe_gt:
            self.subs.append(self.ros.Subscriber('/Object_topic',ObjectStatusList,self.on_gt,queue_size=1))
        config=json.loads((REPO/'VLDrive/configs/vision_encoder_realtime_best_20261003.json').read_text())
        config.update(checkpoint=self.checkpoint['path'],checkpoint_sha256=self.checkpoint['sha256'])
        if self.camera_receiver_mode is not None:
            config['runtime']['camera_receiver_mode'] = self.camera_receiver_mode
        if getattr(self, 'signal_head', None):
            config['signal_head'] = str(self.signal_head)
        write(self.output/'publisher_config.json', config)
        self.log=(self.output/'publisher.log').open('w',buffering=1)
        self.trace=(self.output/'adapter.jsonl').open('w',buffering=1)
        self.transport_log=(self.output/'transport.jsonl').open('w',buffering=1)
        self.sender=subprocess.Popen([str(ANALYSIS_PYTHON),'-u',str(REPO/'VLDrive/inference/ve_objects_udp.py'),
            '--config',str(self.output/'publisher_config.json'),'--generation',self.generation,'--seconds','0',
            '--audit-dir',str(self.output)],
            env=build_analysis_environment(ANALYSIS_PYTHON),stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        self.thread=threading.Thread(target=self.receive,daemon=True)
        self.thread.start()

    def on_gt(self, msg):
        try:
            rows=object_rows(msg)
            with self.lock:
                self.gt,self.gt_time=rows,time.monotonic()
        except (ValueError,AttributeError) as exc:
            self.error=str(exc)

    def on_ego(self,msg):
        with self.lock:
            self.ego=dict(x=float(msg.position.x),y=float(msg.position.y),yaw_deg=float(msg.heading),
                          speed_mps=math.hypot(msg.velocity.x,msg.velocity.y))
            self.ego_time=time.monotonic()

    def receive(self):
        while not self.stop.is_set():
            try:
                data,_=self.sock.recvfrom(MAX_PACKET_BYTES+1)
            except socket.timeout:
                continue
            except OSError:
                return
            received_ns = time.monotonic_ns()
            raw = {}
            try:
                raw = json.loads(data)
                if not isinstance(raw, dict):
                    raise ValueError('Not a packet object')
                packet=decode_packet(data,self.generation,self.checkpoint['sha256'],received_ns)
                if packet['sequence'] <= self.sequence:
                    self.transport_event(raw, received_ns, 'out_of_order_or_duplicate')
                    continue
                self.sequence=packet['sequence']
                with self.lock:
                    packet=self.tracker.update(packet)
                    self.packet=packet
                if getattr(self, 'signal_sink', None) is not None:
                    self.signal_sink.accept(packet)
                tracked_ns = time.monotonic_ns()
                stamp = self.ros.Time.now()
                self.publisher.publish(make_object_message(packet,self.message_type,self.object_type,stamp))
                published_ns = time.monotonic_ns()
                packet['delivery_audit'] = dict(received_ns=received_ns, tracked_ns=tracked_ns,
                    published_ns=published_ns, ros_stamp_ns=stamp.to_nsec())
                with self.lock:
                    self.delivery_window.append(packet)
                self.trace.write(json.dumps(packet,allow_nan=False)+'\n')
                self.transport_event(raw, received_ns, 'published')
            except (ValueError,TypeError,KeyError,OverflowError) as exc:
                self.transport_event(raw, received_ns, 'rejected', str(exc))
                if getattr(self, 'signal_sink', None) is not None:
                    self.signal_sink.invalidate()
                self.rejected+=1
                with self.lock:
                    self.packet=None
            except Exception as exc:
                self.error=repr(exc)
                return

    def transport_event(self, raw, received_ns, status, reason=None):
        identity = (isinstance(raw, dict) and raw.get('generation') == self.generation
                    and raw.get('checkpoint_sha256') == self.checkpoint['sha256'])
        self.transport_log.write(json.dumps(dict(received_ns=received_ns, status=status, reason=reason,
            datagram_sequence=raw.get('datagram_sequence') if identity else None,
            identity_matched=identity))+'\n')

    def performance_line(self):
        with self.lock:
            rows = list(self.delivery_window)
        if len(rows) < 2:
            return '계측 대기'
        end = rows[-1]['delivery_audit']['published_ns']
        rows = [r for r in rows if end-r['delivery_audit']['published_ns'] <= 5e9]
        if len(rows) < 2:
            return '계측 대기'
        start = rows[0]['delivery_audit']['published_ns']
        elapsed = (end-start)/1e9
        sensor = []
        for key in ('rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear', 'lidar'):
            a = rows[0].get('audit', {}).get('selected_samples', {}).get(key)
            b = rows[-1].get('audit', {}).get('selected_samples', {}).get(key)
            if a and b:
                sensor.append('{:.1f}'.format((b['sample_id']-a['sample_id'])/elapsed))
        sent_elapsed = (rows[-1]['sent_ns']-rows[0]['sent_ns'])/1e9
        return '입력(전/좌/우/후/LiDAR)={}Hz VE={:.1f}Hz ROS발행={:.1f}Hz 수신거절={}'.format(
            '/'.join(sensor), (len(rows)-1)/sent_elapsed, (len(rows)-1)/elapsed, self.rejected)

    def snapshot(self):
        now=time.monotonic()
        with self.lock:
            packet,gt,ego=copy.deepcopy((self.packet,self.gt,self.ego))
            gt_age,ego_age=now-self.gt_time,now-self.ego_time
        fresh=packet is not None and 0. <= now-packet['sensor_received_ns']/1e9 <= .5
        ve=[] if not fresh else [dict(b, speed_mps=b['planning_speed_mps']) for b in packet['boxes']]
        return dict(gt=gt or [],gt_fresh=gt is not None and gt_age<=.5,gt_age_s=gt_age,
                    ve=ve,ve_fresh=fresh,ego=ego,ego_age_s=ego_age,packet=packet)

    def require_ready(self, since=None):
        if self.error:
            raise RuntimeError(self.error)
        if self.sender.poll() is not None:
            raise RuntimeError('VE 프로세스 종료 (code={}): {}'.format(
                self.sender.returncode, self.output/'publisher.log'))
        data=self.snapshot()
        if ((getattr(self, 'observe_gt', True) and not data['gt_fresh'])
                or data['ego_age_s']>.5 or not data['ve_fresh']):
            raise RuntimeError('GT/VE/Ego input not fresh')
        packet=data['packet']
        if since is not None and packet['sensor_received_ns']/1e9 < since:
            raise RuntimeError('Waiting for post-placement sensors')
        if math.hypot(packet['ego_pose']['x']-data['ego']['x'],packet['ego_pose']['y']-data['ego']['y'])>1.5:
            raise RuntimeError('VE/ROS Ego pose mismatch')
        return data

    def await_ready(self,timeout=120.,since=None):
        deadline=time.monotonic()+timeout
        stable=None
        reason='waiting'
        while time.monotonic()<deadline:
            if self.error or self.sender.poll() is not None:
                return self.require_ready(since)
            try:
                data=self.require_ready(since)
                stable=time.monotonic() if stable is None else stable
                if time.monotonic()-stable>=.3:
                    return data
            except RuntimeError as exc:
                reason=str(exc)
                stable=None
            time.sleep(.05)
        raise RuntimeError('센서 준비 시간 초과: '+reason)

    def reset_tracks(self):
        if getattr(self, 'signal_sink', None) is not None:
            self.signal_sink.invalidate()
        with self.lock:
            self.tracker=VeObjectTracker()
            self.packet=None

    def close(self):
        self.stop.set()
        if self.sock is not None:
            self.sock.close()
        if self.thread is not None:
            self.thread.join(2.)
        stop_child(self.sender)
        for s in self.subs:
            s.unregister()
        if self.publisher is not None:
            self.publisher.unregister()
        for stream in (self.trace,self.log,self.transport_log):
            if stream is not None:
                stream.close()
