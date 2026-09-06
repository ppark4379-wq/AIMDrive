#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level entry point for the MoraiDrive expert node."""

import atexit
import datetime
import os
import sys
import threading


WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
CATKIN_SOURCE = os.path.join(WORKSPACE_ROOT, "src")
if CATKIN_SOURCE not in sys.path:
    sys.path.insert(0, CATKIN_SOURCE)

# A direct expert run must use the data bundled in this workspace unless the
# caller explicitly opts into externally supplied paths.
if os.environ.get("GT_BEV_DIRECT_USE_ENV_CONFIG", "0") != "1":
    os.environ.pop("GT_BEV_CONFIG_PATH", None)
    os.environ.pop("GT_BEV_MAP_DIR", None)

from expert.src.autonomous_driving import AutonomousDriving
from expert.src.network.ros_manager import RosManager


class _TeeStream(object):
    """Mirror stdout/stderr to the terminal and one UTF-8 run log."""

    def __init__(self, terminal_stream, log_stream):
        self.terminal_stream = terminal_stream
        self.log_stream = log_stream
        self.lock = threading.Lock()

    def write(self, value):
        with self.lock:
            self.terminal_stream.write(value)
            if not self.log_stream.closed:
                self.log_stream.write(value)
        return len(value)

    def flush(self):
        with self.lock:
            self.terminal_stream.flush()
            if not self.log_stream.closed:
                self.log_stream.flush()

    def isatty(self):
        return self.terminal_stream.isatty()

    @property
    def encoding(self):
        return getattr(self.terminal_stream, "encoding", "utf-8")


def _setup_debug_output():
    output_dir = os.environ.get(
        "GT_BEV_DEBUG_OUTPUT_DIR",
        os.path.join(WORKSPACE_ROOT, ".runtime", "debug_output"),
    )
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        output_dir,
        "driving_debug_{}_pid{}.log".format(timestamp, os.getpid()),
    )
    log_stream = open(output_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = _TeeStream(sys.stdout, log_stream)
    sys.stderr = _TeeStream(sys.stderr, log_stream)

    def close_log():
        sys.stdout.flush()
        sys.stderr.flush()
        if not log_stream.closed:
            log_stream.close()

    atexit.register(close_log)
    print("[DEBUG] run log: {}".format(output_path))


def main():
    _setup_debug_output()
    RosManager(AutonomousDriving()).execute()


if __name__ == "__main__":
    main()
