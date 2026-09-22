#!/usr/bin/env python3
"""ROS-independent timestamp selection for Expert planning inputs."""

import time


def has_matching_object(history, ego_source_time, slop_sec):
    return any(
        sample["source_time"] is not None
        and abs(sample["source_time"] - ego_source_time) <= slop_sec
        for sample in history
    )


def select_objects(history, sync_mode, ego_source_time, slop_sec):
    if not history:
        return [], None
    if not sync_mode or ego_source_time is None:
        return list(history[-1]["objects"]), None

    stamped = [sample for sample in history if sample["source_time"] is not None]
    if not stamped:
        return list(history[-1]["objects"]), None
    selected = min(
        stamped,
        key=lambda sample: abs(sample["source_time"] - ego_source_time),
    )
    delta = abs(selected["source_time"] - ego_source_time)
    if delta > slop_sec:
        return [], delta
    return list(selected["objects"]), delta


def select_traffic_light(
    history,
    sync_mode,
    ego_source_time,
    max_age_sec,
    monotonic_now=None,
):
    if not history:
        return [], None
    now = time.monotonic() if monotonic_now is None else float(monotonic_now)
    if not sync_mode or ego_source_time is None:
        sample = history[-1]
        return list(sample["traffic_light"]), now - sample["arrival_time"]

    candidates = [
        sample
        for sample in history
        if sample["source_time"] is not None
        and sample["source_time"] <= ego_source_time
    ]
    if candidates:
        selected = max(candidates, key=lambda sample: sample["source_time"])
        age = ego_source_time - selected["source_time"]
        if age <= max_age_sec:
            traffic_light = list(selected["traffic_light"])
            traffic_light.extend((selected["source_time"], ego_source_time))
            return traffic_light, age
        return [], age

    selected = history[-1]
    age = now - selected["arrival_time"]
    if age <= max_age_sec:
        return list(selected["traffic_light"]), age
    return [], age


def control_dt(
    sync_mode,
    ego_source_time,
    previous_ego_source_time,
    realtime_sampling_rate,
    sync_fallback_sampling_rate,
):
    if not sync_mode:
        return 1.0 / realtime_sampling_rate, previous_ego_source_time

    fallback_dt = 1.0 / sync_fallback_sampling_rate
    if ego_source_time is None or previous_ego_source_time is None:
        return fallback_dt, ego_source_time
    measured_dt = ego_source_time - previous_ego_source_time
    if measured_dt <= 0.0 or measured_dt > 0.5:
        return fallback_dt, ego_source_time
    return measured_dt, ego_source_time
