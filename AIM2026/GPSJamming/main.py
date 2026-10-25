#!/usr/bin/env python3
"""Run LiDAR wall detection and planning UDP output; ROS visualization is optional."""
import os
# Preserve the existing ROS launcher's single-threaded BLAS behavior before NumPy loads.
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
import yaml

from src.LidarDecoder import make_assembler
from src.Process import WallProcessor
from src.UDP import LidarUDPReceiver, LidarUDPStream, WallUDPSender, PORT

DEFAULT_CONFIG = Path(__file__).resolve().parent / 'config' / 'perception.yaml'


def load_config(path=DEFAULT_CONFIG):
    with open(path, encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not all(k in config for k in ('lidar', 'timing')):
        raise ValueError('Expected lidar and timing configuration')
    return config


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    # Import ROS only when explicitly requested; no ROS installation needed otherwise.
    ros = None
    if '--ros' in argv:
        import rospy as ros
        argv = ros.myargv(argv=[sys.argv[0]] + list(argv))[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--port', type=int)
    parser.add_argument('--boundary-host', default='127.0.0.1')
    parser.add_argument('--boundary-port', type=int, default=PORT)
    parser.add_argument('--ros', action='store_true', help='Publish ROS1/RViz debug topics')
    parser.add_argument('--duration', type=float, default=0, help='Run seconds; 0 until Ctrl+C')
    args = parser.parse_args(argv)
    if args.duration < 0:
        parser.error('--duration must be nonnegative')
    if ros:
        ros.init_node('gps_jamming_lidar')
        args.config = Path(ros.get_param('~config', str(args.config)))
        args.bind = ros.get_param('~bind', args.bind)
        args.boundary_host = ros.get_param('~boundary_udp_host', args.boundary_host)
        args.boundary_port = int(ros.get_param('~boundary_udp_port', args.boundary_port))
    config = load_config(args.config)
    udp = dict(config['lidar']['udp'])
    udp['bind'] = args.bind
    udp['port'] = int(args.port if args.port is not None else udp['port'])
    if ros:
        udp['port'] = int(ros.get_param('~port', udp['port']))
    for port in (udp['port'], args.boundary_port):
        if not 1 <= port <= 65535:
            parser.error('Ports must be in 1..65535')
    visualizer = None
    if ros:
        from visualization.ros1_publisher import ROS1Publisher
        visualizer = ROS1Publisher()
    processor = (WallProcessor(config, visualizer.now_ns, 'ros_publish_time')
                 if visualizer else WallProcessor(config))
    def log(message):
        if ros:
            ros.loginfo(message)
        else:
            print(message, flush=True)
    start = time.monotonic()
    next_status = start
    next_error = start
    with ExitStack() as resources:
        sender = WallUDPSender(args.boundary_host, args.boundary_port)
        resources.callback(sender.close)
        receiver = resources.enter_context(LidarUDPReceiver(make_assembler(config),
            decoder=config['lidar']['decoder'], **udp))
        stream = resources.enter_context(LidarUDPStream(receiver,
            config['lidar'].get('runtime', {}).get('frame_queue_size', 4)))
        log('LiDAR UDP %s:%s -> boundary UDP %s:%s%s' %
            (udp['bind'], udp['port'], args.boundary_host, args.boundary_port,
             ' + ROS1/RViz' if ros else ''))
        try:
            while not (ros and ros.is_shutdown()):
                if args.duration and time.monotonic() - start >= args.duration:
                    break
                for frame in stream.read():
                    if ros and ros.is_shutdown():
                        break
                    result = processor.process(frame)
                    if visualizer:
                        visualizer.publish_boundaries(result)
                    try:
                        sender.send(result.boundaries)
                    except (OSError, ValueError, TypeError, KeyError) as exc:
                        if time.monotonic() >= next_error:
                            print('Wall UDP send failed: %s' % exc, file=sys.stderr, flush=True)
                            next_error = time.monotonic() + 5
                    if visualizer:
                        visualizer.publish_clouds(result, receiver.invalid_packets, stream.dropped_frames)
                if time.monotonic() >= next_status:
                    log(json.dumps(dict(invalid_packets=receiver.invalid_packets,
                        dropped_frames=stream.dropped_frames, last_error=receiver.last_error)))
                    next_status = time.monotonic() + 5
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
