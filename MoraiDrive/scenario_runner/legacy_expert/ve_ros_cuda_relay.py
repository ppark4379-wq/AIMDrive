#!/usr/bin/env python3
"""ROS sensors <-> Windows CUDA. Publish tracked VE objects, never controls."""
import argparse
from collections import defaultdict, deque
import json
from pathlib import Path
import signal
import socket
import threading
import time

import rospy
from morai_msgs.msg import EgoVehicleStatus, ObjectStatus, ObjectStatusList
from sensor_msgs.msg import CompressedImage, PointCloud2
from std_msgs.msg import String
from inference.sensors.ros_binary_transport import frames, send_frame
from inference.ve_objects_udp import make_object_message


def header(message):
    h = message.header
    return dict(seq=int(h.seq), stamp=dict(secs=int(h.stamp.secs), nsecs=int(h.stamp.nsecs)),
                frame_id=h.frame_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=12061)
    parser.add_argument('--stop-file', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    stopped = threading.Event()
    lock = threading.Condition()
    pending = defaultdict(lambda: deque(maxlen=1))
    pending['ego'] = deque(maxlen=64)
    counts, latest = defaultdict(int), {}
    result_count = [0]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', args.port))
    server.listen(1)
    server.settimeout(.5)
    # Refuse an existing producer instead of mixing two perception streams.
    import rosgraph
    master = rosgraph.Master('/ve_ros_cuda_relay_preflight')
    publications = dict(master.getSystemState()[0])
    if publications.get('/ve/Object_topic'):
        raise RuntimeError('/ve/Object_topic already has a publisher: ' + str(publications['/ve/Object_topic']))
    rospy.init_node('ve_ros_cuda_relay', anonymous=False, disable_signals=True)
    object_pub = rospy.Publisher('/ve/Object_topic', ObjectStatusList, queue_size=1)
    health_pub = rospy.Publisher('/ve/diagnostics', String, queue_size=1)

    def health(value):
        data = {k: v for k, v in value.items() if k != 'boxes'}
        health_pub.publish(String(data=json.dumps(data, allow_nan=False)))

    def push(topic, role, message, raw):
        with lock:
            pending[role].append((dict(topic=topic, role=role, msg=message), raw, time.monotonic_ns()))
            counts[role] += 1
            lock.notify()

    def camera(message):
        role = {'Camera-1': 'rgb_front', 'Camera-2': 'rgb_left',
                'Camera-3': 'rgb_right', 'Camera-4': 'rgb_rear'}.get(message.header.frame_id)
        if role:
            push('/image_jpeg/compressed', role, dict(header=header(message), format=message.format), bytes(message.data))

    def lidar(message):
        value = dict(header=header(message), height=int(message.height), width=int(message.width),
                     fields=[dict(name=f.name, offset=int(f.offset), datatype=int(f.datatype), count=int(f.count))
                             for f in message.fields], is_bigendian=bool(message.is_bigendian),
                     point_step=int(message.point_step), row_step=int(message.row_step), is_dense=bool(message.is_dense))
        push('/lidar3D', 'lidar', value, bytes(message.data))

    def ego(message):
        value = dict(header=header(message))
        for key in message.__slots__:
            if key == 'header':
                continue
            item = getattr(message, key)
            value[key] = dict(x=float(item.x), y=float(item.y), z=float(item.z)) if hasattr(item, 'x') else item
        push('/Ego_topic', 'ego', value, b'')

    subscribers = [
        rospy.Subscriber('/image_jpeg/compressed', CompressedImage, camera, queue_size=4,
                         buff_size=32*1024*1024, tcp_nodelay=True),
        rospy.Subscriber('/lidar3D', PointCloud2, lidar, queue_size=1,
                         buff_size=32*1024*1024, tcp_nodelay=True),
        rospy.Subscriber('/Ego_topic', EgoVehicleStatus, ego, queue_size=64, tcp_nodelay=True)]

    def receive(connection, disconnected):
        try:
            for packet, raw in frames(connection, disconnected):
                if packet.get('kind') != 've_result' or raw:
                    raise ValueError('Unexpected CUDA result frame')
                if packet.get('ready'):
                    stamp_ns = int(packet['source_stamp_ns'])
                    # Original sensor stamp is preserved, unlike the old audit harness.
                    stamp = rospy.Time(stamp_ns // 10**9, stamp_ns % 10**9)
                    message = make_object_message(packet, ObjectStatusList, ObjectStatus, stamp)
                    object_pub.publish(message)
                    packet['wire_seq'] = int(message.header.seq)
                    packet['object_count'] = message.num_of_npcs + message.num_of_obstacle
                    result_count[0] += 1
                    latest.update(sequence=packet['sequence'], metrics=packet['metrics'])
                health(packet)
        except (OSError, EOFError, ValueError, KeyError) as exc:
            if not stopped.is_set():
                print('[RELAY] CUDA connection ended: ' + str(exc), flush=True)
                health(dict(ready=False, reason='cuda_disconnected'))
        finally:
            disconnected.set()

    client, receiver, disconnected = None, None, threading.Event()
    last_report = 0.
    print('[RELAY] LISTENING port={} output=/ve/Object_topic'.format(args.port), flush=True)
    try:
        while not stopped.is_set() and not rospy.is_shutdown() and not args.stop_file.exists():
            if client is None:
                try:
                    client, _ = server.accept()
                    client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    client.settimeout(.5)
                    disconnected = threading.Event()
                    receiver = threading.Thread(target=receive, args=(client, disconnected), daemon=True)
                    receiver.start()
                except socket.timeout:
                    pass
            elif disconnected.is_set():
                client.close()
                receiver.join(timeout=2)
                client = None
            else:
                with lock:
                    if not any(pending.values()):
                        lock.wait(.02)
                    batch = []
                    for role in ('ego', 'rgb_front', 'rgb_left', 'rgb_right', 'rgb_rear', 'lidar'):
                        while pending[role]:
                            batch.append(pending[role].popleft())
                try:
                    for packet, raw, receipt in batch:
                        if time.monotonic_ns() - receipt <= 350_000_000:
                            send_frame(client, packet, raw)
                except OSError:
                    disconnected.set()
            if time.time() - last_report > 2:
                report = dict(status='RUNNING', sensors=dict(counts), results=result_count[0],
                              latest=dict(latest), control_publications=0)
                (args.report_dir/'relay_status.json').write_text(json.dumps(report, indent=2))
                last_report = time.time()
    finally:
        stopped.set()
        disconnected.set()
        if client:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.close()
        if receiver:
            receiver.join(timeout=2)
        for sub in subscribers:
            sub.unregister()
        health(dict(ready=False, reason='relay_stopped'))
        server.close()
        rospy.signal_shutdown('VE relay stopped')
        (args.report_dir/'relay_status.json').write_text(json.dumps(dict(
            status='STOPPED', sensors=dict(counts), results=result_count[0], control_publications=0), indent=2))


if __name__ == '__main__':
    main()
