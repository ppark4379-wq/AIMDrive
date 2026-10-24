"""Optional receipt audit. Does not select objects or change vehicle control."""
import json
import time


class ObjectInputAudit:
    def __init__(self, path):
        self.stream = open(path, 'w', buffering=1)

    def record(self, message, received_ns, objects):
        header = message.header
        self.stream.write(json.dumps(dict(received_ns=received_ns,
            completed_ns=time.monotonic_ns(), ros_sequence=int(header.seq),
            ros_stamp_ns=int(header.stamp.to_nsec()), objects=objects)) + '\n')

    def close(self):
        self.stream.close()
