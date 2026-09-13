"""ROS health subscription and metadata-only recorder preflight/ongoing checks."""
import json
import threading
import time
import xmlrpc.client

from .bridge_health import HEALTH_TOPIC
from .collection_quality import evaluate_preflight


class TimeoutTransport(xmlrpc.client.Transport):
    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = 2
        return connection


class AcquisitionMonitor:
    def __init__(self, recorder, output):
        import os
        import rospy
        from std_msgs.msg import String
        self.rospy = rospy
        self.recorder = recorder
        self.health = {}
        self.caller = None
        self.lock = threading.Lock()
        self.instance_id = None
        self.last_decision = None
        self.log = (output / "bridge_health.jsonl").open("x", buffering=1)
        self.master = xmlrpc.client.ServerProxy(os.environ.get("ROS_MASTER_URI", "http://localhost:11311"),
                                               transport=TimeoutTransport())
        self.subscriber = rospy.Subscriber(HEALTH_TOPIC, String, self.on_health, queue_size=2)

    def on_health(self, msg):
        try:
            health = json.loads(msg.data)
            if not isinstance(health, dict):
                return
        except (ValueError, TypeError):
            return
        with self.lock:
            self.health = health
            self.caller = (getattr(msg, "_connection_header", {}) or {}).get("callerid")

    def sample(self, phase):
        now = time.monotonic_ns()
        with self.lock:
            health, caller = self.health, self.caller
        with self.recorder.lock:
            rows = [r for r in self.recorder.preview_rows if r["lidar"]["receive_monotonic_ns"] >= now-12_000_000_000]
            callbacks = {k: list(v) for k, v in self.recorder.callback_samples.items()}
        graph_error = None
        try:
            code, _, state = self.master.getSystemState(self.rospy.get_name())
            if code != 1:
                raise RuntimeError("ROS master graph unavailable")
            publishers = dict(state[0])
        except Exception as exc:
            publishers = None
            graph_error = str(exc)
        try:
            decision = evaluate_preflight(rows, callbacks, health, now_ns=now, publishers=publishers,
                                          health_callerid=caller, expected_instance=self.instance_id)
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
            decision = dict(state="INVALID_EGO_TEACHER", reasons=["Malformed health/sensor metadata: " + str(exc)], callbacks={})
        if graph_error:
            decision["state"] = "BRIDGE_NOT_PATCHED"
            decision["reasons"].append("ROS publisher graph unavailable: " + graph_error)
        if self.instance_id is None and decision["state"] == "READY":
            self.instance_id = health["instance_id"]
        if not self.last_decision or decision["state"] != self.last_decision["state"]:
            print(decision["state"], "; ".join(decision["reasons"]), flush=True)
        self.last_decision = decision
        self.log.write(json.dumps(dict(monotonic_ns=now, wall_ns=time.time_ns(), phase=phase,
                                       bridge=health, decision=decision), separators=(",", ":"))+"\n")
        return decision

    def wait_ready(self, timeout_s):
        start = time.monotonic()
        while not self.rospy.is_shutdown():
            decision = self.sample("preflight")
            elapsed = time.monotonic()-start
            if decision["state"] == "READY":
                return
            # Sequential MORAI reconnects can temporarily contain two client IDs
            # or a header gap in the rolling window. Keep recording disabled
            # until a completely clean moving window passes; sustained invalid
            # state times out. Once recording starts, any invalid state aborts.
            if decision["state"] == "BRIDGE_NOT_PATCHED" and elapsed>5:
                raise RuntimeError(decision["state"] + ": " + "; ".join(decision["reasons"]))
            if elapsed >= timeout_s:
                raise RuntimeError(decision["state"] + ": preflight timed out; main recording never started")
            time.sleep(1)
        raise RuntimeError("ROS shutdown during preflight")

    def close(self):
        self.subscriber.unregister()
        self.log.close()
