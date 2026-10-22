"""Evaluation-only record of the simulator's own light colors beside the ego trace.

Written next to ``ego_trace_path`` (``signal_truth.jsonl``) so a long wait at a light can be
compared with what the VE heads reported. Read-only gRPC queries, never fed to the Expert.
"""
import json
import os
import time

LIGHT_IDS = ("C1256W000077", "C1256W000091", "C1256W000016", "C1256W000052")
PERIOD_S = 2.0


def initialize(context):
    context.signal_truth_stream = None
    context.signal_truth_last = float("-inf")


def update(context, elapsed, ego_state):
    path = context.cfg.get("ego_trace_path")
    if not path or float(elapsed) - context.signal_truth_last < PERIOD_S:
        return
    context.signal_truth_last = float(elapsed)
    if context.signal_truth_stream is None:
        directory = os.path.dirname(os.path.abspath(path))
        context.signal_truth_stream = open(os.path.join(directory, "signal_truth.jsonl"), "w",
                                           encoding="utf-8", buffering=1)
    colors = {}
    for light in LIGHT_IDS:
        try:
            state = context.grpc.get_traffic_light_state_by_id(light)
        except Exception:
            state = None
        colors[light] = None if state is None else state["color"]
    pose = ego_state or {}
    context.signal_truth_stream.write(json.dumps({
        "elapsed_s": round(float(elapsed), 2), "unix": time.time(), "colors": colors,
        "x": pose.get("x"), "y": pose.get("y")}) + "\n")
