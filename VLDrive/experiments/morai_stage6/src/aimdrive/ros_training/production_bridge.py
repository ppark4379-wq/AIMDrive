"""Instrument installed Autobahn rosbridge without changing FIFO or ROS files."""
from collections import deque
from functools import wraps
import os
import socket
import threading
import time
import uuid

from .bridge_health import BridgeMetrics, HEALTH_TOPIC, IMPLEMENTATION, boot_id, implementation_hashes
from .rosbridge_ingress import install_binary_json_fix


def install(ws, protocol_class, port):
    metrics = BridgeMetrics()
    local = threading.local()
    clients = {}
    clients_lock = threading.Lock()
    original_init = ws.IncomingQueue.__init__
    original_push = ws.IncomingQueue.push
    original_finish = ws.IncomingQueue.finish
    original_incoming = protocol_class.incoming
    original_deserialize = protocol_class.deserialize
    original_open = ws.RosbridgeWebSocket.onOpen
    original_close = ws.RosbridgeWebSocket.onClose

    def init(self, protocol):
        original_init(self, protocol)
        self._stage6e_arrivals = deque()
        protocol._stage6e_queue = self

    def push(self, message):
        with self.cond:
            if self._finished:
                return
            self._stage6e_arrivals.append(time.monotonic_ns())
            original_push(self, message)

    def finish(self):
        with self.cond:
            original_finish(self)
            self._stage6e_arrivals.clear()

    def incoming(self, message):
        q = self._stage6e_queue
        start = time.monotonic_ns()
        with q.cond:
            receive = q._stage6e_arrivals.popleft() if q._stage6e_arrivals else start
        row = dict(receive_ns=receive, client_id=self.client_id, queue_ms=(start-receive)/1e6)
        local.row = row
        try:
            return original_incoming(self, message)
        except Exception:
            row["errors"] = row.get("errors", 0)+1
            raise
        finally:
            row.update(end_ns=time.monotonic_ns(), processing_ms=(time.monotonic_ns()-start)/1e6,
                       residual=len(self.buffer))
            metrics.observe(row)
            local.row = None

    def deserialize(self, message, cid=None):
        row = getattr(local, "row", None)
        try:
            value = original_deserialize(self, message, cid)
        except Exception:
            if row is not None:
                row["errors"] = row.get("errors", 0)+1
            raise
        if row is not None and isinstance(value, dict):
            msg = value.get("msg")
            row.update(op=value.get("op"), topic=value.get("topic"))
            if isinstance(msg, dict):
                row["frame_id"] = (msg.get("header") or {}).get("frame_id")
        return value

    def on_open(self):
        original_open(self)
        with clients_lock:
            clients[str(self.protocol.client_id)] = self

    def on_close(self, *args):
        with clients_lock:
            if hasattr(self, "protocol"):
                clients.pop(str(self.protocol.client_id), None)
        return original_close(self, *args)

    ws.IncomingQueue.__init__ = init
    ws.IncomingQueue.push = push
    ws.IncomingQueue.finish = finish
    protocol_class.incoming = incoming
    protocol_class.deserialize = deserialize
    ws.RosbridgeWebSocket.onOpen = on_open
    ws.RosbridgeWebSocket.onClose = on_close
    install_binary_json_fix(ws.RosbridgeWebSocket)
    original_message = ws.RosbridgeWebSocket.onMessage

    @wraps(original_message)
    def on_message(self, message, binary):
        self._stage6e_binary = bool(binary)
        try:
            return original_message(self, message, binary)
        except Exception:
            metrics.observe(dict(end_ns=time.monotonic_ns(), errors=1))
            raise
    ws.RosbridgeWebSocket.onMessage = on_message

    identity = dict(implementation=IMPLEMENTATION, code_sha256=implementation_hashes(),
                    boot_id=boot_id(), hostname=socket.gethostname(), pid=os.getpid(), port=port,
                    instance_id=str(uuid.uuid4()), master_uri=os.environ.get("ROS_MASTER_URI"))

    def health_loop():
        import json
        import psutil
        import rospy
        from std_msgs.msg import String
        while not rospy.core.is_initialized():
            time.sleep(.1)
        pub = rospy.Publisher(HEALTH_TOPIC, String, queue_size=1, latch=False)
        while not rospy.is_shutdown():
            now = time.monotonic_ns()
            with clients_lock:
                active = list(clients.items())
            states = []
            for cid, client in active:
                q = client.incoming_queue
                with q.cond:
                    states.append(dict(client_id=cid, peer=str(client.transport.getPeer()),
                                       queue_depth=len(q.queue),
                                       oldest_pending_ms=max(0, (now-q._stage6e_arrivals[0])/1e6) if q._stage6e_arrivals else 0,
                                       websocket_binary=getattr(client, "_stage6e_binary", None)))
            try:
                listening = any(c.status == "LISTEN" and c.laddr.port == port
                                for c in psutil.Process().connections(kind="tcp"))
                report = dict(identity, monotonic_ns=now, wall_ns=time.time_ns(), node=rospy.get_name(),
                              binary_json_patch=bool(getattr(ws.RosbridgeWebSocket.onMessage, "_stage6e_binary_json_fix", False)),
                              listening=listening, clients=states, **metrics.snapshot(now))
                pub.publish(String(data=json.dumps(report, separators=(",", ":"))))
            except Exception as exc:
                rospy.logerr_throttle(5, "Stage6E bridge health unavailable: %s", str(exc))
            time.sleep(.5)  # Wall/monotonic timer, unaffected by /use_sim_time.
    threading.Thread(target=health_loop, name="stage6e_bridge_health", daemon=True).start()
    return metrics
