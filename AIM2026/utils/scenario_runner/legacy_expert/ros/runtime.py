#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level entry point for the MoraiDrive expert node."""

# Resolve this relocated package before importing its shared dependencies.
import sys as _runner_sys
from pathlib import Path as _RunnerPath
for _runner_ancestor in _RunnerPath(__file__).resolve().parents:
    if (_runner_ancestor / "project_paths.py").is_file():
        _runner_sys.path.insert(0, str(_runner_ancestor.parent))
        break
from scenario_runner.project_paths import project_path, activate_project_paths
activate_project_paths()


import atexit
import datetime
import os
import sys
import threading


from pathlib import Path

WORKSPACE_ROOT = str(project_path("AIM2026") / "Expert")
CATKIN_SOURCE = str(Path(WORKSPACE_ROOT).parents[1])
if CATKIN_SOURCE not in sys.path:
    sys.path.insert(0, CATKIN_SOURCE)

# A direct expert run must use the data bundled in this workspace unless the
# caller explicitly opts into externally supplied paths.
if os.environ.get("GT_BEV_DIRECT_USE_ENV_CONFIG", "0") != "1":
    os.environ.pop("GT_BEV_CONFIG_PATH", None)
    os.environ.pop("GT_BEV_MAP_DIR", None)

from AIM2026.Expert.src.autonomous_driving import AutonomousDriving
from scenario_runner.legacy_expert.ros.ros_manager import RosManager


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
    from AIM2026.Expert.src.planning.roundabout import diagnostics as roundabout_debug
    roundabout_debug.configure(os.path.splitext(output_path)[0] + '_roundabout.jsonl')


def main(argv=None):
    if argv is not None:
        # Keep ROS topic/node remapping arguments for rospy.init_node.
        sys.argv = [sys.argv[0], *argv]
    _setup_debug_output()
    RosManager(AutonomousDriving()).execute()


if __name__ == "__main__":
    main()
