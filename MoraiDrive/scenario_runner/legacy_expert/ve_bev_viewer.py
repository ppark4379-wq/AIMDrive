#!/usr/bin/env python3
"""Observe VE object input and render exact footprints; never publish control."""
import argparse
import math
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

CONFIG = Path(__file__).resolve().parents[3] / 'AIM2026/Expert/ros/rviz/ve_udp_objects.rviz'
MAX_AGE = .75
MARKER_TOPIC = '/ve/bev_markers'


def build_markers(objects, odom, age, stamp):
    """Input dimensions are heading-forward length and lateral width in metres."""
    import rospy
    from geometry_msgs.msg import Point
    from visualization_msgs.msg import Marker, MarkerArray

    markers = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)

    def marker(namespace, index, kind, color):
        item = Marker()
        item.header.frame_id, item.header.stamp = 'map', stamp
        item.ns, item.id, item.type = namespace, index, kind
        item.action, item.pose.orientation.w = Marker.ADD, 1.
        item.color.r, item.color.g, item.color.b, item.color.a = color
        item.lifetime = rospy.Duration(MAX_AGE)
        markers.markers.append(item)
        return item

    frame_ok = objects is not None and objects.header.frame_id.lstrip('/') == 'map'
    fresh = frame_ok and age <= MAX_AGE
    origin = odom.pose.pose.position if odom is not None else None
    boxes = list(objects.obstacle_list) + list(objects.npc_list) if fresh else []
    for index, obj in enumerate(boxes):
        x, y = obj.position.x, obj.position.y
        length, width, yaw = obj.size.x, obj.size.y, math.radians(obj.heading)
        if not all(math.isfinite(v) for v in (x, y, length, width, yaw)) or min(length, width) <= 0:
            continue
        moving = obj.name.endswith(':moving')
        color = (.1, .85, 1., 1.) if moving else (1., .45, 0., 1.)
        outline = marker('ve_exact_bbox', index, Marker.LINE_STRIP, color)
        outline.scale.x = .05
        c, s = math.cos(yaw), math.sin(yaw)
        for dx, dy in ((.5, .5), (.5, -.5), (-.5, -.5), (-.5, .5), (.5, .5)):
            forward, lateral = dx * length, dy * width
            outline.points.append(Point(x=x + c * forward - s * lateral,
                                        y=y + s * forward + c * lateral, z=.2))
        label = marker('ve_bbox_labels', index, Marker.TEXT_VIEW_FACING, (1., .8, .5, 1.))
        label.pose.position = Point(x=x, y=y, z=1.)
        label.scale.z = .38
        distance = ' {:.1f}m'.format(math.hypot(x - origin.x, y - origin.y)) if origin is not None else ''
        state = obj.name.split(':')[-1] if obj.name.startswith('ve_tracked_unknown:') else 'unknown'
        label.text = 'ID {} {}{}  {:.2f}x{:.2f}m  {:.1f}km/h'.format(
            obj.unique_id, state, distance, length, width, obj.velocity.x)

    if objects is None:
        text = 'VE: WAITING FOR INPUT'
    elif not frame_ok:
        text = 'VE: UNEXPECTED FRAME ' + objects.header.frame_id
    elif not fresh:
        text = 'VE: INPUT STALE ({:.1f}s)'.format(age)
    else:
        text = 'VE INPUT: {} objects | age {:.2f}s'.format(len(boxes), age)
    status = marker('ve_input_status', 0, Marker.TEXT_VIEW_FACING,
                    (.3, 1., .6, 1.) if fresh else (1., .3, .3, 1.))
    status.pose.position = Point(x=origin.x if origin is not None else 0.,
                                 y=origin.y + 5. if origin is not None else 5., z=2.)
    status.scale.z, status.text = .6, text
    return markers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--markers-only', action='store_true', help='RViz 창 없이 시각화 토픽만 발행')
    parser.add_argument('--seconds', type=float, default=0., help='0이면 창을 닫거나 Ctrl+C까지 실행')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error('seconds must be finite and nonnegative')
    import rospy
    from morai_msgs.msg import ObjectStatusList
    from nav_msgs.msg import Odometry
    from visualization_msgs.msg import MarkerArray

    if not args.markers_only and (not CONFIG.is_file() or not shutil.which('rviz')):
        parser.error('RViz 또는 전용 설정 파일이 없습니다: ' + str(CONFIG))
    # Fail promptly rather than silently waiting forever for a ROS master.
    from run_ve_udp_expert import require_ros_master
    try:
        require_ros_master()
    except RuntimeError as exc:
        print('[FAIL] ' + str(exc), flush=True)
        return 2
    rospy.init_node('ve_bev_viewer', anonymous=True, disable_signals=True)
    stopped, lock = threading.Event(), threading.Lock()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    state = dict(objects=None, odom=None, received=None)

    def on_objects(message):
        with lock:
            state.update(objects=message, received=time.monotonic())

    def on_odom(message):
        with lock:
            state['odom'] = message

    subscribers = [rospy.Subscriber('/ve/Object_topic', ObjectStatusList, on_objects, queue_size=1),
                   rospy.Subscriber('/odom', Odometry, on_odom, queue_size=1)]
    publisher = rospy.Publisher(MARKER_TOPIC, MarkerArray, queue_size=1)
    viewer = None
    print('[BEV] 하늘색 bbox: 이동 추정 / 주황 bbox: 정지·미확정 / 노랑: 자차 / 파랑: Expert 경로', flush=True)
    print('[BEV] bbox 표기: 추적 ID · 이동 상태 · 중심 거리 · 길이×폭 · 계획에 전달한 속력.', flush=True)
    print('[BEV] 창을 닫거나 Ctrl+C로 보기만 종료합니다. 기존 VE·Expert 주행은 계속됩니다.', flush=True)
    try:
        if not args.markers_only:
            viewer = subprocess.Popen(['rviz', '-d', str(CONFIG)], start_new_session=True)
        started = time.monotonic()
        while not stopped.is_set() and not rospy.is_shutdown():
            if args.seconds > 0 and time.monotonic() - started >= args.seconds:
                break
            if viewer is not None and viewer.poll() is not None:
                return viewer.returncode
            with lock:
                snapshot = dict(state)
            age = time.monotonic() - snapshot['received'] if snapshot['received'] is not None else float('inf')
            publisher.publish(build_markers(snapshot['objects'], snapshot['odom'], age, rospy.Time.now()))
            stopped.wait(.1)
    finally:
        for subscriber in subscribers:
            subscriber.unregister()
        publisher.unregister()
        if viewer is not None and viewer.poll() is None:
            viewer.terminate()
            try:
                viewer.wait(timeout=3)
            except subprocess.TimeoutExpired:
                viewer.kill()
                viewer.wait()
        rospy.signal_shutdown('BEV viewer closed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
