"""Optional official ROS1 bag recorder owned by one scenario episode.

Capture starts after world reset and before expert launch, and ends before
owned actors are deleted. Receipt metadata remains in the existing raw journal;
rosbag stores the original ROS messages with ROS receipt indexing time.
"""
from collections import Counter
import io
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import threading
import time
import xmlrpc.client

TOPICS = (
    '/ReplayInfo_topic', '/Ego_topic', '/image_jpeg/compressed', '/lidar3D',
    '/gps', '/imu', '/clock', '/tf', '/tf_static', '/Object_topic',
    '/GetTrafficLightStatus', '/InsnStatus', '/IntscnTL_topic',
    '/CollisionData', '/VehicleCollisionData', '/ctrl_cmd',
    '/global_path', '/local_path', '/expert_target_velocity',
    '/SyncModeInfo', '/stage6e/bridge_health',
    '/sim/process/state/msg/MoraiSimProcStatus',
)
REQUIRED_STREAMS = ('/ReplayInfo_topic', '/Ego_topic', '/lidar3D', '/gps', '/imu',
                    '/clock', 'Camera-1', 'Camera-2', 'Camera-3', 'Camera-4')
REQUIRED_TOPICS = frozenset(('/ReplayInfo_topic','/Ego_topic','/lidar3D','/gps','/imu',
                              '/clock','/image_jpeg/compressed'))


class _BoundedTransport(xmlrpc.client.Transport):
    def make_connection(self, host):
        connection=super().make_connection(host)
        connection.timeout=2
        return connection


def inbound_topics(response):
    if response[0]!=1:
        return set()
    return {row[4] for row in response[2]
            if len(row)>=6 and row[2]=='i' and row[5] is True}


def camera_frame_id(payload):
    if len(payload) < 16:
        raise ValueError('Truncated compressed image header')
    length = struct.unpack_from('<I', payload, 12)[0]
    if length > 1024 or 16+length > len(payload):
        raise ValueError('Invalid camera frame ID size')
    return payload[16:16+length].decode('utf-8')


def written_startup_streams(path, limit=32*1024*1024):
    """Inspect completed chunks in a bounded RAM snapshot of the active bag.

    rosbag.reindex() may truncate its file: it is deliberately run ONLY on
    BytesIO, never on the live file or the preserved original recording.
    """
    import rosbag
    path=Path(path)
    if not path.exists() or path.stat().st_size<4096:return Counter()
    with path.open('rb') as stream:payload=stream.read(limit)
    counts=Counter()
    snapshot=io.BytesIO(payload)
    snapshot.mode='rb'  # Noetic Bag checks file.mode even for an in-memory IOBase.
    with rosbag.Bag(snapshot,'r',allow_unindexed=True) as bag:
        for _ in bag.reindex():pass
        # Noetic reindex builds flat in-memory indexes. Its lazy loader would
        # otherwise reinterpret those as on-disk lists of per-chunk indexes.
        bag._connection_indexes_read=True
        for topic,raw,stamp in bag.read_messages(topics=sorted(REQUIRED_TOPICS),raw=True):
            key=camera_frame_id(raw[1]) if topic=='/image_jpeg/compressed' else topic
            counts[key]+=1
    if len(payload)==limit and not all(counts[key]>=3 for key in REQUIRED_STREAMS):
        raise RuntimeError('No complete startup sensor set within 32 MiB; preserve bag and reconnect before retry')
    return counts


class ScenarioRosbag:
    def __init__(self, directory, event):
        self.directory = Path(directory)
        self.path = self.directory/'sensors.bag'
        self.event = event
        self.process = None
        self.log = None
        self.subscribers = []
        self.lock = threading.Lock()
        self.counts = Counter()
        self.last_receive = {}
        self.error = None
        self.started = False
        self.closed = False
        self.result = None
        self.node_name = '/stage6e_scenario_bag_'+str(os.getpid())

    def recorder_connections(self, master, caller):
        lookup=master.lookupNode(self.node_name)
        if lookup[0]!=1:return set()
        with xmlrpc.client.ServerProxy(lookup[2],transport=_BoundedTransport()) as node:
            pid=node.getPid(caller)
            if pid[0]!=1 or pid[2]!=self.process.pid:
                raise RuntimeError('Unexpected rosbag node owner')
            return inbound_topics(node.getBusInfo(caller))

    def start(self, timeout=30):
        import rospy
        from std_msgs.msg import String
        self.directory.mkdir(exist_ok=False)
        begin = threading.Event()
        def begun(message):
            if message.data == str(self.path):
                begin.set()
        self.subscribers.append(rospy.Subscriber('/begin_write', String, begun, queue_size=10))
        def observe(message, topic):
            now = time.monotonic()
            key = topic
            try:
                if topic == '/image_jpeg/compressed':
                    payload = getattr(message, '_buff', None)
                    if payload is None:
                        buf = io.BytesIO()
                        message.serialize(buf)
                        payload = buf.getvalue()
                    key = camera_frame_id(payload)
                with self.lock:
                    self.counts[key] += 1
                    self.last_receive[key] = now
            except Exception as error:
                with self.lock:
                    self.error = repr(error)
        for topic in ('/ReplayInfo_topic', '/Ego_topic', '/image_jpeg/compressed', '/lidar3D', '/gps', '/imu', '/clock'):
            self.subscribers.append(rospy.Subscriber(topic, rospy.AnyMsg, observe,
                callback_args=topic, queue_size=20, buff_size=8*1024*1024, tcp_nodelay=True))
        self.log = (self.directory/'record.log').open('x')
        command = ['/opt/ros/noetic/lib/rosbag/record', '--publish', '--tcpnodelay',
                   '--buffsize=512', '--chunksize=768', '--min-space=50G',
                   '--output-name='+str(self.path), '__ns:=/', '__name:='+self.node_name[1:]] + list(TOPICS)
        wrapper = Path(__file__).resolve().parents[1]/'tools/exec_owned_rosbag.py'
        self.process = subprocess.Popen([sys.executable, str(wrapper), str(os.getpid())]+command,
            env=os.environ.copy(), stdout=self.log,
            stderr=subprocess.STDOUT, start_new_session=True)
        self.event('rosbag_started', pid=self.process.pid, path=str(self.path), topics=list(TOPICS),
                   compression='none', buffer_mib=512, bag_time='ROS receipt time; source headers unchanged')
        deadline = time.monotonic()+timeout
        connection_baseline = None
        next_connection_check = 0
        next_written_check = 0
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError('rosbag exited before readiness: '+str(self.process.returncode))
            with self.lock:
                if self.error:
                    raise RuntimeError(self.error)
                ready = all(self.counts[key] >= 3 and time.monotonic()-self.last_receive[key] < 2
                            for key in REQUIRED_STREAMS)
            if connection_baseline is None and time.monotonic()>=next_connection_check:
                next_connection_check=time.monotonic()+.5
                try:
                    connected=self.recorder_connections(rospy.get_master(),rospy.get_name())
                except (OSError,xmlrpc.client.Error):
                    connected=set()
                if REQUIRED_TOPICS<=connected:
                    with self.lock:connection_baseline=dict(self.counts)
                    self.event('rosbag_subscriber_connections_ready',node=self.node_name,
                               connected_topics=sorted(connected),stream_counts=connection_baseline)
            with self.lock:
                settled=connection_baseline is not None and all(
                    self.counts[key]-connection_baseline.get(key,0)>=3 for key in REQUIRED_STREAMS)
            if ready and settled and time.monotonic()>=next_written_check:
                next_written_check=time.monotonic()+.5
                written=written_startup_streams(Path(str(self.path)+'.active'))
                if all(written[key]>=3 for key in REQUIRED_STREAMS):
                    self.started = True
                    self.event('rosbag_ready_before_expert', stream_counts=dict(self.counts),
                               actual_bag_stream_counts=dict(written),startup_evidence='closed chunks in RAM snapshot')
                    return
            time.sleep(.05)
        with self.lock:
            missing = [key for key in REQUIRED_STREAMS if self.counts[key] < 3
                       or time.monotonic()-self.last_receive.get(key, 0) >= 2]
        raise TimeoutError('rosbag required streams missing: '+str(missing))

    def check(self):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError('rosbag stopped during scenario')
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            # Raw recorder covers the core sensors/bridge; additionally require
            # ReplayInfo to stay live so a sensor-only bag cannot look complete.
            if self.started and time.monotonic()-self.last_receive.get('/ReplayInfo_topic', 0) > 5:
                raise RuntimeError('ReplayInfo stopped for 5 wall seconds')

    def stop(self):
        if self.closed:
            return self.result
        self.closed = True
        errors = []
        if self.process is not None:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGINT)
                try:
                    self.process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    errors.append('SIGINT finalization timed out; partial bag retained')
                    os.killpg(self.process.pid, signal.SIGTERM)
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        errors.append('recorder still alive after SIGTERM')
            if self.process.returncode != 0:
                errors.append('recorder_exit_'+str(self.process.returncode))
        for subscriber in self.subscribers:
            subscriber.unregister()
        if self.log is not None:
            self.log.close()
        complete = self.path.is_file() and not Path(str(self.path)+'.active').exists()
        if not complete:
            errors.append('Bag not finalized')
        self.result = dict(schema='morai_scenario_rosbag_record_v1', status='recorded' if not errors else 'failed',
            path=str(self.path), topics=list(TOPICS), errors=errors, monitor_stream_counts=dict(self.counts),
            bytes=self.path.stat().st_size if self.path.exists() else None,
            training_visible=False, physical_simultaneity_verified=False,
            bag_time='ROS receipt time', source_headers_unchanged=True)
        with (self.directory/'recording.json').open('x') as stream:
            json.dump(self.result, stream, indent=2)
        self.event('rosbag_finalized', **self.result)
        if errors:
            raise RuntimeError('rosbag finalization failed: '+str(errors))
        return self.result
