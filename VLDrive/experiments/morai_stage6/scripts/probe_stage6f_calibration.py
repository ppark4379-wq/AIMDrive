#!/usr/bin/env python3
"""Read-only ROS calibration snapshot; never publishes vehicle/sensor commands."""

import argparse
from collections import Counter
import io
import json
import math
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def primitive(value):
    if isinstance(value, (str, float, int, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [primitive(x) for x in value]
    if hasattr(value, "__slots__"):
        return {key: primitive(getattr(value, key)) for key in value.__slots__ if key != "data"}
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, default=10)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or not 1 <= args.duration <= 60:
        parser.error("Snapshot duration must be between 1 and 60 seconds")
    output = Path(args.output).expanduser().resolve()
    if ROOT not in output.parents:
        parser.error("Output must stay in this checkout")
    output.mkdir(parents=True, exist_ok=False)
    import rospy
    import roslib.message
    import numpy as np
    from PIL import Image
    from sensor_msgs import point_cloud2
    rospy.init_node("stage6f_calibration_probe", anonymous=True, disable_signals=True)
    topics = rospy.get_published_topics()
    lock = threading.Lock()
    counts, first, last, samples, images, tf = Counter(), {}, {}, {}, {}, {}
    saved = set()
    captures = {}
    motion = []
    errors = []

    def callback(msg, topic):
        receive = time.monotonic_ns()
        wall = time.time_ns()
        frame = getattr(getattr(msg, "header", None), "frame_id", "")
        key = topic + "|" + frame
        with lock:
            counts[key] += 1
            first.setdefault(key, receive)
            last[key] = receive
            if topic in ("/tf", "/tf_static"):
                for transform in msg.transforms:
                    pair = transform.header.frame_id + "->" + transform.child_frame_id
                    tf[pair] = dict(transform=primitive(transform), topic=topic,
                                    receive_monotonic_ns=receive)
                return
            if topic == "/stage6e/bridge_health":
                samples[key] = dict(receive_monotonic_ns=receive, health=json.loads(msg.data))
                return
            samples[key] = dict(receive_monotonic_ns=receive, receive_wall_ns=wall,
                                message=primitive(msg))
            if topic == "/Ego_topic":
                motion.append(samples[key])
            if key in saved:
                return
            saved.add(key)
            # Snapshot only one image per camera and one cloud. Subsequent
            # callbacks collect metadata/rates without decoding the payload.
            try:
                if topic == "/image_jpeg/compressed":
                    if frame not in ("Camera-1", "Camera-2", "Camera-3", "Camera-4"):
                        raise ValueError("Unexpected camera frame: " + frame)
                    path = output / (frame + ".jpg")
                    path.write_bytes(bytes(msg.data))
                    with Image.open(io.BytesIO(bytes(msg.data))) as im:
                        images[frame] = dict(path=path.name, size=list(im.size))
                    captures[frame] = dict(samples[key], path=path.name)
                elif topic == "/lidar3D":
                    cloud = np.array(list(point_cloud2.read_points(msg,
                        field_names=("x", "y", "z", "intensity"), skip_nans=False)), dtype=np.float32)
                    np.save(output / "lidar.npy", cloud, allow_pickle=False)
                    captures[frame] = dict(samples[key], path="lidar.npy")
            except Exception as exc:
                errors.append(dict(topic=topic, error=repr(exc)))

    subscribers = []
    for topic, msg_type in topics:
        if topic in ("/tf", "/tf_static", "/Ego_topic", "/gps", "/imu", "/stage6e/bridge_health",
                     "/image_jpeg/compressed", "/lidar3D") or msg_type == "sensor_msgs/CameraInfo":
            cls = roslib.message.get_message_class(msg_type)
            if cls is not None:
                subscribers.append(rospy.Subscriber(topic, cls, callback, callback_args=topic,
                    queue_size=5, buff_size=16 * 1024 * 1024))
    started = time.monotonic()
    while time.monotonic() - started < args.duration and not rospy.is_shutdown():
        time.sleep(.1)
    for sub in subscribers:
        sub.unregister()
    with lock:
        report = dict(topics=topics, duration_s=time.monotonic() - started, errors=errors,
            rates={key: dict(samples=n, callback_hz=(n-1) / ((last[key]-first[key])/1e9)
                            if last[key] > first[key] else None,
                            age_ms=(time.monotonic_ns()-last[key])/1e6) for key, n in counts.items()},
            latest=samples, transforms=tf, images=images, captures=captures, ego_motion=motion,
            clock="receive_monotonic_ns; header clocks are not cross-sensor synchronization")
        (output / "snapshot.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(rates=report["rates"], errors=errors, images=images,
                              transform_frames=sorted(tf), snapshot=str(output/"snapshot.json"))))
    rospy.signal_shutdown("snapshot complete")


if __name__ == "__main__":
    main()
