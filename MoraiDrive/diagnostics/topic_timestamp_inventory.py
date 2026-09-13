"""Read-only ROS time-field inventory; bounded samples, no sensor decoding."""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import socket
import statistics
import struct
import threading
import time


def time_fields(cls, prefix='', depth=0):
    import roslib.message
    if cls is None or depth > 4:
        return []
    result = []
    for key, kind in zip(cls.__slots__, cls._slot_types):
        base = kind.split('[')[0]
        path = prefix + key + ('[]' if '[' in kind else '')
        if base == 'time':
            result.append(path)
        elif '/' in base:
            result.extend(time_fields(roslib.message.get_message_class(base), path + '.', depth + 1))
    return result


def extract_times(msg, prefix='', depth=0, context=''):
    if depth > 4:
        return
    header = getattr(msg, 'header', None)
    if header is not None:
        context = str(header.frame_id) + '->' + str(getattr(msg, 'child_frame_id', ''))
    for name, kind in zip(msg.__slots__, msg._slot_types):
        base = kind.split('[')[0]
        value = getattr(msg, name)
        path = prefix + name
        if base == 'time':
            yield path, int(value.to_nsec()), context
        elif '/' in base:
            if '[' in kind:
                for index, child in enumerate(value[:3]):
                    yield from extract_times(child, path + '[{}].'.format(index), depth + 1, context)
            else:
                yield from extract_times(value, path + '.', depth + 1, context)


def main():
    import rosgraph
    import roslib.message
    import rospy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=10)
    args = parser.parse_args()
    socket.setdefaulttimeout(5)
    rospy.init_node('topic_timestamp_inventory', anonymous=True, disable_signals=True)
    master = rosgraph.Master(rospy.get_name())
    publishers = dict(master.getSystemState()[0])
    inventory, classes = {}, {}
    for topic, kind in master.getPublishedTopics(''):
        try:
            cls = roslib.message.get_message_class(kind)
            fields = time_fields(cls)
            inventory[topic] = {'type': kind, 'time_fields': fields, 'publishers': publishers.get(topic, []),
                                'schema': list(zip(cls.__slots__, cls._slot_types)) if cls else None}
            if fields and cls:
                classes[topic] = cls
        except Exception as error:
            inventory[topic] = {'type': kind, 'error': str(error)}
    lock = threading.Lock()
    samples = defaultdict(lambda: deque(maxlen=256))
    counts, errors = Counter(), Counter()
    decoded_at, latest_clock = {}, [None]
    started_wall = time.time()

    def receive(msg, topic):
        wall, mono = time.time_ns(), time.monotonic_ns()
        caller = msg._connection_header.get('callerid', 'unknown')
        raw, cls, values = msg._buff, classes[topic], []
        try:
            if topic == '/clock':
                sec, nsec = struct.unpack_from('<II', raw)
                stamp = sec * 1000000000 + nsec
                values.append(('clock', stamp, ''))
                with lock:
                    latest_clock[0] = stamp
            elif cls._slot_types and cls._slot_types[0] == 'std_msgs/Header':
                _, sec, nsec, size = struct.unpack_from('<IIII', raw)
                if size > 4096 or len(raw) < 16 + size or nsec >= 1000000000:
                    raise ValueError('invalid Header')
                frame = bytes(raw[16:16 + size]).decode('utf-8', errors='replace')
                values.append(('header.stamp', sec * 1000000000 + nsec, frame))
            extra = any(f not in ('header.stamp', 'clock') for f in inventory[topic]['time_fields'])
            if extra:
                key = (topic, caller)
                with lock:
                    decode = mono - decoded_at.get(key, 0) >= 500000000
                    if decode:
                        decoded_at[key] = mono
                if decode and len(raw) <= 512 * 1024:
                    decoded = cls().deserialize(raw)
                    values.extend(v for v in extract_times(decoded) if v[0] not in ('header.stamp', 'clock'))
                elif decode:
                    with lock:
                        errors[topic + ': compound payload exceeds decode limit'] += 1
            with lock:
                counts[topic] += 1
                for field, stamp, context in values:
                    samples[(topic, caller, field, context)].append({
                        'stamp_ns': stamp, 'recv_wall_ns': wall, 'recv_mono_ns': mono,
                        'latest_clock_ns': latest_clock[0]})
        except Exception as error:
            with lock:
                errors[topic + ': ' + str(error)] += 1

    subscribers = [rospy.Subscriber(topic, rospy.AnyMsg, receive, callback_args=topic,
                    queue_size=1, buff_size=4 * 1024 * 1024) for topic in classes]
    print(json.dumps({'topics': len(inventory), 'observed_time_topics': len(classes)}), flush=True)
    deadline = time.monotonic() + min(30, max(1, args.duration))
    try:
        while time.monotonic() < deadline and not rospy.is_shutdown():
            time.sleep(.05)
    finally:
        for sub in subscribers:
            sub.unregister()
    with lock:
        copied = [(key, list(rows)) for key, rows in samples.items()]
        count_copy, error_copy = dict(counts), dict(errors)
    groups = []
    for (topic, caller, field, context), rows in copied:
        first, last = rows[0], rows[-1]
        offsets = [(r['stamp_ns'] - r['latest_clock_ns']) / 1e6 for r in rows if r['latest_clock_ns'] is not None]
        groups.append({'topic': topic, 'caller': caller, 'field': field, 'context': context,
                       'retained': len(rows), 'first_ns': first['stamp_ns'], 'last_ns': last['stamp_ns'],
                       'unique_stamps': len({r['stamp_ns'] for r in rows}),
                       'stamp_delta_sec': (last['stamp_ns'] - first['stamp_ns']) / 1e9,
                       'recv_delta_sec': (last['recv_mono_ns'] - first['recv_mono_ns']) / 1e9,
                       'clock_offset_ms': {'min': min(offsets), 'median': statistics.median(offsets), 'max': max(offsets)} if offsets else None,
                       'samples': rows})
    result = {'started_wall': started_wall, 'ended_wall': time.time(), 'inventory': inventory,
              'params': {p: rospy.get_param(p, None) for p in ['/use_sim_time', '/morai_drive/sync_runtime_active']},
              'counts': count_copy, 'errors': error_copy, 'groups': groups,
              'limits': '256 samples/group; nested arrays first 3 entries; nested decode <=2Hz and <=512KiB'}
    output = Path(__file__).parent / ('topic_timestamps_' + time.strftime('%Y%m%d_%H%M%S') + '.json')
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'counts': count_copy, 'errors': error_copy,
                      'groups': [{k: v for k, v in g.items() if k != 'samples'} for g in groups]}, indent=2), flush=True)
    rospy.signal_shutdown('bounded read-only time inventory complete')


if __name__ == '__main__':
    main()
