#!/usr/bin/env python3
"""ROS1 integration check on a private master; never publishes to the driving master."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import xmlrpc.client

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.notice_policy import DEFAULT_ZONES, RouteNoticePolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zones", type=Path, default=DEFAULT_ZONES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    policy = RouteNoticePolicy(args.zones)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:{}".format(port)
    os.environ["ROS_HOSTNAME"] = "127.0.0.1"
    os.environ.pop("ROS_IP", None)
    children = []
    with tempfile.TemporaryFile() as logs:
        try:
            children.append(subprocess.Popen(["roscore", "-p", str(port)], stdout=logs, stderr=logs))
            master = xmlrpc.client.ServerProxy(os.environ["ROS_MASTER_URI"])
            deadline = time.monotonic() + 15
            while True:
                try:
                    if master.getPid("notice_test")[0] == 1:
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("Private ROS master failed to start")
                time.sleep(.1)
            import rospy
            from nav_msgs.msg import Odometry
            from std_msgs.msg import String
            from visualization_msgs.msg import MarkerArray
            rospy.init_node("notice_integration_probe", anonymous=True, disable_signals=True)
            received = {}
            rospy.Subscriber("/vldrive/notice_state", String,
                             lambda m: received.update(state=json.loads(m.data)), queue_size=1)
            rospy.Subscriber("/vldrive/notice_markers", MarkerArray,
                             lambda m: received.update(markers=m), queue_size=1)
            pub = rospy.Publisher("/odom", Odometry, queue_size=1)
            children.append(subprocess.Popen([sys.executable, str(ROOT / "tools/rviz_notice_zones.py"),
                "--zones", str(args.zones.resolve()), "--stale-seconds", "0.5"], stdout=logs, stderr=logs))
            deadline = time.monotonic() + 10
            while pub.get_num_connections() == 0:
                if time.monotonic() > deadline:
                    raise RuntimeError("Notice node failed to subscribe")
                time.sleep(.1)
            boundaries = {0}
            for zone in policy.zones:
                start, end = zone["start"]["route_index"], zone["end"]["route_index"]
                boundaries.update((max(0, start - 1), start, end,
                                   min(len(policy.points) - 1, policy.first_index_after(end))))
            checks = []
            for lap in (1, 2):
                for index in sorted(boundaries):
                    expected = policy.at_index(index)
                    msg = Odometry()
                    msg.header.frame_id = "map"
                    msg.pose.pose.position.x, msg.pose.pose.position.y = policy.points[index][:2]
                    deadline = time.monotonic() + 3
                    while True:
                        pub.publish(msg)
                        time.sleep(.04)
                        state = received.get("state", {})
                        markers = received.get("markers")
                        texts = ([m.text for m in markers.markers if m.ns == "notice_current"]
                                 if markers else [])
                        actual = " ".join(texts[0].splitlines()[1:]) if texts else None
                        actual_index = state.get("route_index", -1)
                        same_position = (0 <= actual_index < len(policy.points)
                                         and policy.points[actual_index][:2] == policy.points[index][:2])
                        if (same_position and state.get("zone_ids") == expected["zone_ids"]
                                and texts and (actual == expected["notice_text"]
                                               if expected["notice_text"] else texts[0] == "NOTICE: none")):
                            break
                        if time.monotonic() > deadline:
                            raise AssertionError("ROS notice mismatch at {}: {}".format(index, state))
                    checks.append(dict(lap=lap, index=index, notice_text=state["notice_text"]))
            time.sleep(.9)
            if received.get("state", {}).get("valid") is not False:
                raise AssertionError("Stale odometry did not clear notice")
            report = dict(status="passed", private_master=True, checked_boundary_states=len(checks),
                          stale_odometry_cleared=True, checks=checks)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({k: v for k, v in report.items() if k != "checks"}))
        except Exception:
            logs.seek(0)
            print(logs.read().decode("utf-8", errors="replace"), file=sys.stderr)
            raise
        finally:
            for child in reversed(children):
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    main()
