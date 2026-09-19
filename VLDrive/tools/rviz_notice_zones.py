#!/usr/bin/env python3
"""Display snapped route zones and the exact current LLM notice in ROS1 RViz."""
import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.notice_policy import DEFAULT_ZONES, NoticeTracker, RouteNoticePolicy


def marker_specs(policy, state=None, ego=None):
    """Transport-free marker descriptions, also exercised by the boundary replay test."""
    active = set(state["zone_ids"]) if state else set()
    specs = [dict(id=0, ns="notice_route", kind="line", color=[.5, .5, .5], width=.12,
                  points=[(p[0], p[1], .04) for p in policy.points])]
    for number, zone in enumerate(policy.zones):
        start, end = zone["start"]["route_index"], zone["end"]["route_index"]
        selected = zone["id"] in active
        specs.append(dict(id=number, ns="notice_zones", kind="line", color=zone["color"],
                          alpha=1.0 if selected else .5, width=1.8 if selected else .65,
                          points=[(p[0], p[1], .12 + number * .03)
                                  for p in policy.points[start:end + 1]]))
        for key, index in (("start", start), ("end", end)):
            p = policy.points[index]
            specs.append(dict(id=number, ns="notice_" + key, kind="sphere", width=1.6,
                              color=zone["color"], position=(p[0], p[1], .5)))
        p = policy.points[start]
        specs.append(dict(id=number, ns="notice_labels", kind="text", width=1.2,
                          color=zone["color"], position=(p[0], p[1], 2.0 + number * .25),
                          text="{} [{}..{}]".format(zone["id"], start, end)))
    if ego is not None:
        text = "NOTICE: none"
        if state is None or not state.get("valid"):
            text = "NOTICE: position unavailable / outside route"
        elif state["notice_text"]:
            text = "NOTICE ACTIVE\n" + "\n".join(textwrap.wrap(state["notice_text"], 52))
        specs.append(dict(id=0, ns="notice_current", kind="text", width=1.1,
                          color=[1., 1., .2], position=(ego[0], ego[1], 5.0), text=text))
    return specs


def ros_markers(specs, frame, stamp, rospy, Marker, MarkerArray, Point):
    result = MarkerArray()
    for spec in specs:
        marker = Marker()
        marker.header.frame_id, marker.header.stamp = frame, stamp
        marker.ns, marker.id, marker.action = spec["ns"], spec["id"], Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.color.r, marker.color.g, marker.color.b = spec["color"]
        marker.color.a = spec.get("alpha", 1.0)
        if spec["kind"] == "line":
            marker.type, marker.scale.x = Marker.LINE_STRIP, spec["width"]
            marker.points = [Point(x=p[0], y=p[1], z=p[2]) for p in spec["points"]]
        else:
            marker.pose.position = Point(*spec["position"])
            if spec["kind"] == "text":
                marker.type, marker.scale.z, marker.text = Marker.TEXT_VIEW_FACING, spec["width"], spec["text"]
            else:
                marker.type = Marker.SPHERE
                marker.scale.x = marker.scale.y = marker.scale.z = spec["width"]
        result.markers.append(marker)
    return result


def main():
    import rospy
    from geometry_msgs.msg import Point
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from visualization_msgs.msg import Marker, MarkerArray
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zones", type=Path, default=DEFAULT_ZONES)
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--frame", default="map")
    parser.add_argument("--stale-seconds", type=float, default=2.0)
    args = parser.parse_args(rospy.myargv()[1:])
    if args.stale_seconds <= 0:
        parser.error("--stale-seconds must be positive")
    policy = RouteNoticePolicy(args.zones)
    tracker = NoticeTracker(policy)
    rospy.init_node("vldrive_notice_zones", anonymous=True)
    markers_pub = rospy.Publisher("/vldrive/notice_markers", MarkerArray, queue_size=1, latch=True)
    text_pub = rospy.Publisher("/vldrive/notice_text", String, queue_size=1, latch=True)
    state_pub = rospy.Publisher("/vldrive/notice_state", String, queue_size=1, latch=True)
    # Callback only captures the latest pose; state changes are serialized in the display loop.
    latest = [None]
    def receive(message):
        if message.header.frame_id.lstrip("/") != args.frame.lstrip("/"):
            rospy.logwarn_throttle(5., "Notice odometry frame differs from configured map frame")
            latest[0] = None
            return
        p = message.pose.pose.position
        latest[0] = (float(p.x), float(p.y), time.monotonic())
    rospy.Subscriber(args.odom_topic, Odometry, receive, queue_size=1)
    last_pose, state, ego = None, None, None
    try:
        while not rospy.is_shutdown():
            pose = latest[0]
            fresh = pose is not None and time.monotonic() - pose[2] <= args.stale_seconds
            if fresh and pose != last_pose:
                ego = pose[:2]
                try:
                    state = tracker.update(*ego)
                except ValueError:
                    state = dict(valid=False, notice_text="", zone_ids=[], entered=[], exited=list(tracker.active))
                    tracker.reset()
                if state["entered"] or state["exited"]:
                    rospy.loginfo("Notice entered=%s exited=%s text=%s", state["entered"],
                                  state["exited"], state["notice_text"])
                last_pose = pose
            elif not fresh:
                tracker.reset()
                state = dict(valid=False, notice_text="", zone_ids=[], entered=[], exited=[])
                last_pose = None
            text_pub.publish(String(data=state["notice_text"] if state else ""))
            state_pub.publish(String(data=json.dumps(state or {}, ensure_ascii=False)))
            specs = marker_specs(policy, state, ego)
            markers_pub.publish(ros_markers(specs, args.frame, rospy.Time.now(),
                                           rospy, Marker, MarkerArray, Point))
            # Continue clearing stale state even while simulation /clock is paused.
            rospy.rostime.wallsleep(.1)
    finally:
        marker = Marker()
        marker.action = Marker.DELETEALL
        markers_pub.publish(MarkerArray(markers=[marker]))
        text_pub.publish(String(data=""))


if __name__ == "__main__":
    main()
