#!/usr/bin/env python3
"""Observe MORAI ROS SyncMode frame identity without modifying timestamps.

This diagnostic is intentionally isolated from Scenario Runner, Expert and Data
Collector.  It owns a temporary ROS SyncMode session, advances ticks with a
stationary brake command, and records the original message headers from four
cameras and LiDAR.  ``--capture-stride`` can pulse ``sensor_capture`` so the
MORAI publication-gate semantics can be measured independently of timestamps.

Important invariants:
  * sensor headers are read-only;
  * no message_filters or ApproximateTimeSynchronizer is used;
  * no timestamp tolerance, nearest-match or arrival-time frame matching is used;
  * monotonic observation times are used only to determine callback-vs-service
    ordering.  They are never treated as sensor/source timestamps or frame keys.

Run this probe only when no other process owns MORAI SyncMode.
"""

import argparse
import collections
import datetime
import json
import os
import sys
import threading
import time
from fractions import Fraction


REQUIRED_SENSORS = (
    "camera_front",
    "camera_left",
    "camera_right",
    "camera_rear",
    "lidar",
)


def _message_type(message):
    ros_type = getattr(message, "_type", None)
    if ros_type:
        return str(ros_type)
    return "{}.{}".format(type(message).__module__, type(message).__name__)


def _header_metadata(message):
    header = getattr(message, "header", None)
    if header is None:
        return {
            "present": False,
            "seq": None,
            "stamp_secs": None,
            "stamp_nsecs": None,
            "frame_id": None,
        }
    stamp = getattr(header, "stamp", None)
    return {
        "present": True,
        "seq": int(getattr(header, "seq", 0)),
        "stamp_secs": int(getattr(stamp, "secs", 0)) if stamp is not None else None,
        "stamp_nsecs": int(getattr(stamp, "nsecs", 0)) if stamp is not None else None,
        "frame_id": str(getattr(header, "frame_id", "")),
    }


def _identity_candidates(message):
    """Return top-level scalar fields whose names suggest native frame identity."""
    candidates = {}
    for name in getattr(message, "__slots__", ()):  # generated ROS messages
        lowered = str(name).lower()
        if not any(token in lowered for token in ("frame", "seq", "tick")):
            continue
        try:
            value = getattr(message, name)
        except Exception:
            continue
        if isinstance(value, (bool, int, float, str)):
            candidates[str(name)] = value
    return candidates


def _vector_metadata(value):
    if value is None:
        return None
    return {
        "x": float(getattr(value, "x", 0.0)),
        "y": float(getattr(value, "y", 0.0)),
        "z": float(getattr(value, "z", 0.0)),
    }


def _vehicle_status_metadata(status):
    return {
        "message_type": _message_type(status),
        "header": _header_metadata(status),
        "identity_candidates": _identity_candidates(status),
        "unique_id": int(getattr(status, "unique_id", 0)),
        "position": _vector_metadata(getattr(status, "position", None)),
        "velocity": _vector_metadata(getattr(status, "velocity", None)),
        "acceleration": _vector_metadata(getattr(status, "acceleration", None)),
        "heading": float(getattr(status, "heading", 0.0)),
        "accel": float(getattr(status, "accel", 0.0)),
        "brake": float(getattr(status, "brake", 0.0)),
        "front_steer_angle": float(getattr(status, "front_steer_angle", 0.0)),
        "rear_steer_angle": float(getattr(status, "rear_steer_angle", 0.0)),
    }


class EventRecorder:
    def __init__(self):
        self._lock = threading.Lock()
        self._next_order = 0
        self._callback_counts = collections.Counter()
        self._record_sensors = True
        self._current_tick_index = None
        self._current_phase = "before_first_tick"
        self.events = []
        self.ticks = []
        self._ticks_by_index = {}

    def _order_locked(self):
        self._next_order += 1
        return self._next_order

    def _append_locked(self, event):
        event["arrival_order"] = self._order_locked()
        self.events.append(event)
        return event["arrival_order"]

    def record_general(self, event_type, **values):
        observed_ns = time.monotonic_ns()
        with self._lock:
            event = {
                "event": event_type,
                "observed_monotonic_ns": observed_ns,
            }
            event.update(values)
            return self._append_locked(event)

    def sensor_callback(self, sensor_name, topic, message):
        # Capture callback-entry time before taking the recorder lock. This value
        # is diagnostic ordering metadata only and is never used as frame identity.
        observed_ns = time.monotonic_ns()
        header = _header_metadata(message)
        message_type = _message_type(message)
        candidates = _identity_candidates(message)
        with self._lock:
            if not self._record_sensors:
                return
            self._callback_counts[sensor_name] += 1
            event = {
                "event": "sensor_callback",
                "sensor": sensor_name,
                "topic": topic,
                "message_type": message_type,
                "header": header,
                "identity_candidates": candidates,
                "callback_count": int(self._callback_counts[sensor_name]),
                "observed_monotonic_ns": observed_ns,
                # This is an observation window, not a production frame key.
                "observed_tick_index": self._current_tick_index,
                "observed_phase": self._current_phase,
            }
            self._append_locked(event)

    def sync_info_callback(self, message):
        observed_ns = time.monotonic_ns()
        with self._lock:
            if not self._record_sensors:
                return
            self._callback_counts["sync_mode_info"] += 1
            self._append_locked({
                "event": "sync_mode_info",
                "message_type": _message_type(message),
                "callback_count": int(self._callback_counts["sync_mode_info"]),
                "observed_monotonic_ns": observed_ns,
                "observed_tick_index": self._current_tick_index,
                "observed_phase": self._current_phase,
                "master_id": str(getattr(message, "master_id", "")),
                "status": bool(getattr(message, "status", False)),
                "frame": int(getattr(message, "frame", 0)),
                "can_send_tick": bool(getattr(message, "can_send_tick", False)),
            })

    def begin_tick(self, tick_index, request_frame, analyzed, sensor_capture):
        observed_ns = time.monotonic_ns()
        with self._lock:
            self._current_tick_index = int(tick_index)
            self._current_phase = "sync_control_rpc"
            tick = {
                "tick_index": int(tick_index),
                "analyzed": bool(analyzed),
                "request_frame": int(request_frame),
                "sensor_capture": bool(sensor_capture),
                "cycle_start_monotonic_ns": observed_ns,
                "cycle_start_order": None,
                "control_return_monotonic_ns": None,
                "control_return_order": None,
                "wait_call_monotonic_ns": None,
                "wait_call_order": None,
                "wait_return_monotonic_ns": None,
                "wait_return_order": None,
                "response_frame": None,
                "tick_status": None,
                "pause_status": None,
                "vehicle_status": None,
            }
            order = self._append_locked({
                "event": "tick_cycle_start",
                "tick_index": int(tick_index),
                "analyzed": bool(analyzed),
                "request_frame": int(request_frame),
                "sensor_capture": bool(sensor_capture),
                "observed_monotonic_ns": observed_ns,
            })
            tick["cycle_start_order"] = order
            self.ticks.append(tick)
            self._ticks_by_index[int(tick_index)] = tick

    def mark_control_return(self, tick_index, result):
        observed_ns = time.monotonic_ns()
        with self._lock:
            tick = self._ticks_by_index[int(tick_index)]
            order = self._append_locked({
                "event": "sync_control_return",
                "tick_index": int(tick_index),
                "result": bool(getattr(result, "result", False)),
                "observed_monotonic_ns": observed_ns,
            })
            tick["control_return_monotonic_ns"] = observed_ns
            tick["control_return_order"] = order
            self._current_phase = "before_wait_call"

    def mark_wait_call(self, tick_index):
        observed_ns = time.monotonic_ns()
        with self._lock:
            tick = self._ticks_by_index[int(tick_index)]
            self._current_phase = "wait_for_tick_rpc"
            order = self._append_locked({
                "event": "wait_for_tick_call",
                "tick_index": int(tick_index),
                "request_frame": int(tick["request_frame"]),
                "observed_monotonic_ns": observed_ns,
            })
            tick["wait_call_monotonic_ns"] = observed_ns
            tick["wait_call_order"] = order

    def finish_tick(self, tick_index, response, return_observed_ns):
        with self._lock:
            tick = self._ticks_by_index[int(tick_index)]
            order = self._append_locked({
                "event": "wait_for_tick_return",
                "tick_index": int(tick_index),
                "request_frame": int(tick["request_frame"]),
                "response_frame": int(getattr(response, "frame", 0)),
                "tick_status": bool(getattr(response, "tick_status", False)),
                "pause_status": bool(getattr(response, "pause_status", False)),
                "observed_monotonic_ns": int(return_observed_ns),
            })
            tick["wait_return_monotonic_ns"] = int(return_observed_ns)
            tick["wait_return_order"] = order
            tick["response_frame"] = int(getattr(response, "frame", 0))
            tick["tick_status"] = bool(getattr(response, "tick_status", False))
            tick["pause_status"] = bool(getattr(response, "pause_status", False))
            tick["vehicle_status"] = _vehicle_status_metadata(response.vehicle_status)
            self._current_phase = "after_wait_return"

    def freeze_sensor_recording(self):
        with self._lock:
            self._record_sensors = False


def _exact_affine_mapping(pairs):
    """Find an exact sync_frame = a * sensor_seq + b mapping, if one exists."""
    unique_pairs = []
    seen = set()
    for sensor_seq, sync_frame in pairs:
        pair = (int(sensor_seq), int(sync_frame))
        if pair not in seen:
            unique_pairs.append(pair)
            seen.add(pair)
    if len(unique_pairs) < 3:
        return None

    first = unique_pairs[0]
    second = next((pair for pair in unique_pairs[1:] if pair[0] != first[0]), None)
    if second is None:
        return None
    slope = Fraction(second[1] - first[1], second[0] - first[0])
    intercept = Fraction(first[1]) - slope * first[0]
    if all(Fraction(frame) == slope * seq + intercept for seq, frame in unique_pairs):
        return {
            "sync_frame_per_sensor_seq": str(slope),
            "intercept": str(intercept),
            "formula": "sync_frame = ({}) * header.seq + ({})".format(
                slope, intercept
            ),
            "evidence_pairs": len(unique_pairs),
        }
    return None


def _ratio(numerator, denominator):
    if not denominator:
        return None
    return float(numerator) / float(denominator)


def _capture_gate_analysis(ticks, events_by_tick, required_sensors):
    """Analyze capture pulses using service-call observation windows only.

    An observation window starts immediately before ``SyncModeCtrlCmd`` and is
    associated with that request frame. Callback order is used only to test the
    publication gate and WaitForTick completion boundary; it is not a sensor
    frame identity.
    """
    capture_ticks = [tick for tick in ticks if tick["sensor_capture"]]
    non_capture_ticks = [tick for tick in ticks if not tick["sensor_capture"]]
    sensor_results = {}

    for sensor in required_sensors:
        true_counts = []
        false_counts = []
        true_before = 0
        true_after = 0
        false_before = 0
        false_after = 0
        true_ticks_with_after = 0
        false_ticks_with_after = 0

        for tick in ticks:
            events = events_by_tick[tick["tick_index"]][sensor]
            before = sum(
                1 for event in events
                if event["observed_monotonic_ns"] <= tick["wait_return_monotonic_ns"]
            )
            after = len(events) - before
            if tick["sensor_capture"]:
                true_counts.append(len(events))
                true_before += before
                true_after += after
                true_ticks_with_after += int(after > 0)
            else:
                false_counts.append(len(events))
                false_before += before
                false_after += after
                false_ticks_with_after += int(after > 0)

        true_exactly_one = sum(count == 1 for count in true_counts)
        false_with_messages = sum(count > 0 for count in false_counts)
        if sum(false_counts) > 0:
            gate_behavior = "publishes_on_sensor_capture_false"
        elif true_counts and all(count == 1 for count in true_counts):
            gate_behavior = "exactly_one_on_true_only"
        elif sum(true_counts) > 0:
            gate_behavior = "true_only_but_not_exactly_one_per_tick"
        else:
            gate_behavior = "no_callbacks_observed"
        sensor_results[sensor] = {
            "gate_behavior": gate_behavior,
            "capture_true_tick_count": len(true_counts),
            "capture_false_tick_count": len(false_counts),
            "callbacks_on_capture_true_ticks": sum(true_counts),
            "callbacks_on_capture_false_ticks": sum(false_counts),
            "capture_true_callback_histogram": {
                str(count): occurrences
                for count, occurrences in sorted(collections.Counter(true_counts).items())
            },
            "capture_false_callback_histogram": {
                str(count): occurrences
                for count, occurrences in sorted(collections.Counter(false_counts).items())
            },
            "capture_true_ticks_exactly_one": true_exactly_one,
            "capture_true_ticks_exactly_one_ratio": _ratio(
                true_exactly_one, len(true_counts)
            ),
            "capture_false_ticks_with_messages": false_with_messages,
            "capture_false_ticks_with_messages_ratio": _ratio(
                false_with_messages, len(false_counts)
            ),
            "callbacks_before_wait_return": {
                "capture_true": true_before,
                "capture_false": false_before,
            },
            "callbacks_after_wait_return": {
                "capture_true": true_after,
                "capture_false": false_after,
            },
            "ticks_with_callbacks_after_wait_return": {
                "capture_true": true_ticks_with_after,
                "capture_false": false_ticks_with_after,
            },
        }

    all_required_exactly_one_true = 0
    all_required_exactly_one_before_return = 0
    true_ticks_all_callbacks_before_return = 0
    for tick in capture_ticks:
        sensor_lists = events_by_tick[tick["tick_index"]]
        if all(len(sensor_lists[sensor]) == 1 for sensor in required_sensors):
            all_required_exactly_one_true += 1
            if all(
                sensor_lists[sensor][0]["observed_monotonic_ns"]
                <= tick["wait_return_monotonic_ns"]
                for sensor in required_sensors
            ):
                all_required_exactly_one_before_return += 1
        all_events = [
            event
            for sensor in required_sensors
            for event in sensor_lists[sensor]
        ]
        if all_events and all(
            event["observed_monotonic_ns"] <= tick["wait_return_monotonic_ns"]
            for event in all_events
        ):
            true_ticks_all_callbacks_before_return += 1

    no_false_publication = all(
        result["callbacks_on_capture_false_ticks"] == 0
        for result in sensor_results.values()
    )
    all_sensors_publish_on_false = bool(non_capture_ticks) and all(
        result["callbacks_on_capture_false_ticks"] > 0
        for result in sensor_results.values()
    )
    perfect_true_capture = bool(capture_ticks) and (
        all_required_exactly_one_true == len(capture_ticks)
    )

    if not non_capture_ticks:
        verdict = {
            "case": None,
            "title": "capture gate not tested: no sensor_capture=false ticks",
            "reason": "Use --capture-stride greater than 1 to create a false control group.",
        }
    elif (
        no_false_publication
        and perfect_true_capture
        and all_required_exactly_one_before_return == len(capture_ticks)
    ):
        verdict = {
            "case": "A",
            "title": "sensor_capture acts as an exact ROS sensor publication gate",
            "reason": (
                "No required sensor callback was observed on false ticks, and every "
                "true tick produced exactly one callback from every required sensor."
            ),
        }
    elif all_sensors_publish_on_false:
        verdict = {
            "case": "B",
            "title": "sensor_capture does not gate ROS sensor publication",
            "reason": (
                "Every required sensor topic produced callbacks in sensor_capture=false "
                "observation windows."
            ),
        }
    else:
        verdict = {
            "case": "C",
            "title": "sensor_capture behavior is mixed or incomplete by sensor",
            "reason": (
                "Only some required sensors published on false ticks, or true ticks did "
                "not consistently produce exactly one callback from every sensor."
            ),
        }

    return {
        "capture_true_ticks": len(capture_ticks),
        "capture_false_ticks": len(non_capture_ticks),
        "sensor_stats": sensor_results,
        "all_required_exactly_one_on_capture_true_ticks": (
            all_required_exactly_one_true
        ),
        "all_required_exactly_one_on_capture_true_ticks_ratio": _ratio(
            all_required_exactly_one_true, len(capture_ticks)
        ),
        "all_required_exactly_one_before_wait_return": (
            all_required_exactly_one_before_return
        ),
        "all_required_exactly_one_before_wait_return_ratio": _ratio(
            all_required_exactly_one_before_return, len(capture_ticks)
        ),
        "capture_true_ticks_all_observed_callbacks_before_wait_return": (
            true_ticks_all_callbacks_before_return
        ),
        "capture_true_ticks_all_observed_callbacks_before_wait_return_ratio": _ratio(
            true_ticks_all_callbacks_before_return, len(capture_ticks)
        ),
        "verdict": verdict,
        "observation_boundary": (
            "callback arrival is grouped only by the active SyncModeCtrlCmd/"
            "WaitForTick call window; it is not a frame identity"
        ),
    }


def analyze(recorder, required_sensors):
    ticks = [tick for tick in recorder.ticks if tick["analyzed"]]
    tick_by_index = {tick["tick_index"]: tick for tick in ticks}
    events_by_tick = {
        tick["tick_index"]: {name: [] for name in required_sensors}
        for tick in ticks
    }
    optional_by_tick = {
        tick["tick_index"]: {"objects": [], "traffic_light": []}
        for tick in ticks
    }

    for event in recorder.events:
        if event.get("event") != "sensor_callback":
            continue
        tick_index = event.get("observed_tick_index")
        if tick_index not in tick_by_index:
            continue
        sensor = event.get("sensor")
        if sensor in events_by_tick[tick_index]:
            events_by_tick[tick_index][sensor].append(event)
        elif sensor in optional_by_tick[tick_index]:
            optional_by_tick[tick_index][sensor].append(event)

    sensor_stats = {}
    deterministic_mappings = {}
    for sensor in required_sensors:
        sensor_events = []
        offsets = []
        mapping_pairs = []
        missing_ticks = 0
        duplicate_ticks = 0
        delayed_after_return = 0
        callbacks_before_return = 0
        callbacks_after_return = 0
        headerless = 0
        frame_ids = collections.Counter()
        message_types = collections.Counter()

        for tick in ticks:
            events = events_by_tick[tick["tick_index"]][sensor]
            if not events:
                missing_ticks += 1
            elif len(events) > 1:
                duplicate_ticks += 1
            for event in events:
                sensor_events.append(event)
                message_types[event["message_type"]] += 1
                header = event["header"]
                if not header["present"]:
                    headerless += 1
                    continue
                frame_ids[str(header["frame_id"])] += 1
                seq = header["seq"]
                if seq is not None and tick["response_frame"] is not None:
                    offset = int(seq) - int(tick["response_frame"])
                    offsets.append(offset)
                    mapping_pairs.append((int(seq), int(tick["response_frame"])))
                if event["observed_monotonic_ns"] <= tick["wait_return_monotonic_ns"]:
                    callbacks_before_return += 1
                else:
                    callbacks_after_return += 1
                    delayed_after_return += 1

        ordered_sequences = [
            event["header"]["seq"]
            for event in sorted(sensor_events, key=lambda item: item["arrival_order"])
            if event["header"]["seq"] is not None
        ]
        out_of_order = sum(
            1 for previous, current in zip(ordered_sequences, ordered_sequences[1:])
            if current < previous
        )
        repeated_sequence = sum(
            1 for previous, current in zip(ordered_sequences, ordered_sequences[1:])
            if current == previous
        )
        offset_counts = collections.Counter(offsets)
        constant_offset = (
            next(iter(offset_counts)) if len(offset_counts) == 1 and offsets else None
        )
        affine = _exact_affine_mapping(mapping_pairs)
        if constant_offset is not None:
            deterministic_mappings[sensor] = {
                "kind": "constant_offset",
                "sensor_seq_minus_sync_frame": int(constant_offset),
                "formula": "sync_frame = header.seq - ({})".format(constant_offset),
            }
        elif affine is not None:
            deterministic_mappings[sensor] = dict({"kind": "affine"}, **affine)

        sensor_stats[sensor] = {
            "message_types": dict(message_types),
            "callback_messages": len(sensor_events),
            "messages_with_header": len(sensor_events) - headerless,
            "headerless_messages": headerless,
            "header_frame_ids": dict(frame_ids),
            "header_seq_exact_match_count": int(offset_counts.get(0, 0)),
            "header_seq_mismatch_count": len(offsets) - int(offset_counts.get(0, 0)),
            "seq_minus_sync_frame_offsets": {
                str(key): int(value) for key, value in sorted(offset_counts.items())
            },
            "constant_offset": constant_offset,
            "offset_changes": max(0, len(offset_counts) - 1),
            "affine_mapping": affine,
            "missing_tick_windows": missing_ticks,
            "duplicate_tick_windows": duplicate_ticks,
            "callbacks_before_wait_return": callbacks_before_return,
            "callbacks_after_wait_return": callbacks_after_return,
            "delayed_arrival_count": delayed_after_return,
            "out_of_order_seq_count": out_of_order,
            "repeated_adjacent_seq_count": repeated_sequence,
        }

    complete_capture_windows = 0
    equal_sensor_seq_windows = 0
    unequal_sensor_seq_windows = 0
    wait_order_counts = collections.Counter()
    tick_window_stats = []
    for tick in ticks:
        sensor_lists = events_by_tick[tick["tick_index"]]
        counts = {sensor: len(events) for sensor, events in sensor_lists.items()}
        exactly_one = all(counts[sensor] == 1 for sensor in required_sensors)
        raw_sequences = {}
        if exactly_one:
            complete_capture_windows += 1
            raw_sequences = {
                sensor: sensor_lists[sensor][0]["header"]["seq"]
                for sensor in required_sensors
            }
            if None not in raw_sequences.values() and len(set(raw_sequences.values())) == 1:
                equal_sensor_seq_windows += 1
            else:
                unequal_sensor_seq_windows += 1

        all_events = [
            event
            for sensor in required_sensors
            for event in sensor_lists[sensor]
        ]
        if any(count == 0 for count in counts.values()):
            order_class = "missing_sensor"
        elif all(
            event["observed_monotonic_ns"] <= tick["wait_return_monotonic_ns"]
            for event in all_events
        ):
            order_class = "all_sensor_callbacks_before_wait_return"
        elif all(
            event["observed_monotonic_ns"] > tick["wait_return_monotonic_ns"]
            for event in all_events
        ):
            order_class = "wait_return_before_all_sensor_callbacks"
        else:
            order_class = "mixed_before_and_after_wait_return"
        wait_order_counts[order_class] += 1
        tick_window_stats.append({
            "tick_index": tick["tick_index"],
            "request_frame": tick["request_frame"],
            "response_frame": tick["response_frame"],
            "sensor_capture": tick["sensor_capture"],
            "sensor_counts": counts,
            "sensor_sequences": raw_sequences,
            "wait_vs_sensor_order": order_class,
        })

    capture_gate = _capture_gate_analysis(
        ticks, events_by_tick, required_sensors
    )

    all_have_observations = all(
        sensor_stats[name]["messages_with_header"] > 0 for name in required_sensors
    )
    all_direct = all_have_observations and all(
        set(sensor_stats[name]["seq_minus_sync_frame_offsets"].keys()) == {"0"}
        for name in required_sensors
    )
    all_deterministic = all_have_observations and all(
        name in deterministic_mappings for name in required_sensors
    )

    if all_direct:
        verdict = {
            "case": 1,
            "title": "header.seq directly matches MORAI Sync frame",
            "native_frame_key": "header.seq",
            "reason": (
                "Every observed required sensor callback in its tick observation "
                "window had header.seq == WaitForTickResponse.frame."
            ),
        }
    elif all_deterministic:
        verdict = {
            "case": 2,
            "title": "deterministic sequence mapping observed",
            "native_frame_key": "per-sensor header.seq mapping",
            "mappings": deterministic_mappings,
            "reason": (
                "Every required sensor had one exact mapping across all observed "
                "tick windows. Confirm the mapping in a second run before using it."
            ),
        }
    else:
        verdict = {
            "case": 3,
            "title": "no native frame identity proven from sensor headers",
            "native_frame_key": None,
            "reason": (
                "At least one required sensor lacked a stable direct or exact "
                "deterministic header.seq mapping. Arrival order and timestamps "
                "must not be promoted to frame identity."
            ),
            "repository_alternatives": [
                "morai_msgs/SyncModeInfo.frame exists, but it identifies simulator state rather than individual sensor payloads.",
                "morai_msgs/WaitForTickResponse.frame identifies the completed tick and includes vehicle_status.",
                "morai_msgs/SyncModeCtrlCmd.frame identifies the command target frame.",
                "The checked morai_msgs copy has SaveSensorData, but no sensor topic message with an explicit Sync frame field.",
            ],
        }

    sync_info_events = [
        event for event in recorder.events if event.get("event") == "sync_mode_info"
    ]
    optional_stats = {}
    for sensor in ("objects", "traffic_light"):
        events = [
            event
            for by_sensor in optional_by_tick.values()
            for event in by_sensor[sensor]
        ]
        optional_stats[sensor] = {
            "callback_messages": len(events),
            "message_types": dict(collections.Counter(
                event["message_type"] for event in events
            )),
            "headers_present": sum(1 for event in events if event["header"]["present"]),
            "header_seq_values": sorted(set(
                event["header"]["seq"] for event in events
                if event["header"]["seq"] is not None
            )),
            "top_level_identity_candidates": sorted(set(
                name
                for event in events
                for name in event.get("identity_candidates", {})
            )),
        }

    return {
        "contract": {
            "timestamp_headers_modified": False,
            "approximate_time_synchronizer_used": False,
            "timestamp_tolerance_used": False,
            "arrival_time_used_as_source_timestamp": False,
            "arrival_order_usage": (
                "diagnostic callback-vs-service ordering and tick observation windows only"
            ),
        },
        "analyzed_ticks": len(ticks),
        "first_response_frame": ticks[0]["response_frame"] if ticks else None,
        "last_response_frame": ticks[-1]["response_frame"] if ticks else None,
        "sensor_stats": sensor_stats,
        "cross_sensor_seq": {
            "complete_one_message_windows": complete_capture_windows,
            "all_sensor_seq_equal_windows": equal_sensor_seq_windows,
            "sensor_seq_unequal_windows": unequal_sensor_seq_windows,
        },
        "wait_for_tick_publish_order": dict(wait_order_counts),
        "sensor_capture_analysis": capture_gate,
        "tick_windows": tick_window_stats,
        "sync_mode_info": {
            "callback_messages": len(sync_info_events),
            "frames": [event["frame"] for event in sync_info_events],
            "can_send_tick_values": sorted(set(
                event["can_send_tick"] for event in sync_info_events
            )),
        },
        "optional_topics": optional_stats,
        "verdict": verdict,
    }


def _format_header(header):
    if not header["present"]:
        return "header=<absent>"
    return "seq={} stamp={}.{:09d} frame_id={!r}".format(
        header["seq"],
        header["stamp_secs"],
        header["stamp_nsecs"],
        header["frame_id"],
    )


def build_tick_report(recorder, summary, required_sensors):
    tick_by_index = {
        tick["tick_index"]: tick for tick in recorder.ticks if tick["analyzed"]
    }
    events_by_tick = {
        index: {name: [] for name in required_sensors + ("objects", "traffic_light")}
        for index in tick_by_index
    }
    for event in recorder.events:
        if event.get("event") != "sensor_callback":
            continue
        index = event.get("observed_tick_index")
        sensor = event.get("sensor")
        if index in events_by_tick and sensor in events_by_tick[index]:
            events_by_tick[index][sensor].append(event)

    lines = []
    for tick_index in sorted(tick_by_index):
        tick = tick_by_index[tick_index]
        lines.append("SYNC_FRAME={}".format(tick["response_frame"]))
        lines.append("request_frame={}".format(tick["request_frame"]))
        lines.append("sensor_capture={}".format(tick["sensor_capture"]))
        lines.append("wait_return_order={}".format(tick["wait_return_order"]))
        lines.append("tick_status={} pause_status={}".format(
            tick["tick_status"], tick["pause_status"]
        ))
        vehicle = tick["vehicle_status"]
        lines.append("vehicle_status: type={} {} position={} velocity={} heading={}".format(
            vehicle["message_type"],
            _format_header(vehicle["header"]),
            vehicle["position"],
            vehicle["velocity"],
            vehicle["heading"],
        ))
        callback_counts = {
            sensor: len(events_by_tick[tick_index][sensor])
            for sensor in required_sensors
        }
        callbacks_before_return = {
            sensor: sum(
                1 for event in events_by_tick[tick_index][sensor]
                if event["observed_monotonic_ns"] <= tick["wait_return_monotonic_ns"]
            )
            for sensor in required_sensors
        }
        callbacks_after_return = {
            sensor: callback_counts[sensor] - callbacks_before_return[sensor]
            for sensor in required_sensors
        }
        lines.append("callback_counts={}".format(callback_counts))
        lines.append("callbacks_before_wait_return={}".format(
            callbacks_before_return
        ))
        lines.append("callbacks_after_wait_return={}".format(
            callbacks_after_return
        ))
        for sensor in required_sensors + ("objects", "traffic_light"):
            events = events_by_tick[tick_index][sensor]
            if not events:
                lines.append("{}: MISSING_IN_OBSERVATION_WINDOW".format(sensor))
                continue
            for duplicate_index, event in enumerate(events, 1):
                suffix = "" if len(events) == 1 else "#{}".format(duplicate_index)
                relation = (
                    "before_wait_return"
                    if event["observed_monotonic_ns"] <= tick["wait_return_monotonic_ns"]
                    else "after_wait_return"
                )
                lines.append(
                    "{}{}: type={} {} arrival_order={} callback_count={} phase={} {}".format(
                        sensor,
                        suffix,
                        event["message_type"],
                        _format_header(event["header"]),
                        event["arrival_order"],
                        event["callback_count"],
                        event["observed_phase"],
                        relation,
                    )
                )
        lines.append("")

    lines.append("=== SUMMARY ===")
    lines.append("analyzed_ticks={}".format(summary["analyzed_ticks"]))
    lines.append(
        "sensor | messages | exact | mismatch | offsets | missing | duplicate | delayed | out_of_order"
    )
    for sensor in required_sensors:
        stats = summary["sensor_stats"][sensor]
        lines.append(
            "{} | {} | {} | {} | {} | {} | {} | {} | {}".format(
                sensor,
                stats["callback_messages"],
                stats["header_seq_exact_match_count"],
                stats["header_seq_mismatch_count"],
                stats["seq_minus_sync_frame_offsets"],
                stats["missing_tick_windows"],
                stats["duplicate_tick_windows"],
                stats["delayed_arrival_count"],
                stats["out_of_order_seq_count"],
            )
        )
    lines.append("cross_sensor_seq={}".format(summary["cross_sensor_seq"]))
    lines.append("wait_for_tick_publish_order={}".format(
        summary["wait_for_tick_publish_order"]
    ))
    capture = summary["sensor_capture_analysis"]
    lines.append("")
    lines.append("=== SENSOR_CAPTURE GATE ===")
    lines.append("capture_true_ticks={} capture_false_ticks={}".format(
        capture["capture_true_ticks"], capture["capture_false_ticks"]
    ))
    lines.append(
        "sensor | behavior | true_callbacks | false_callbacks | true_exactly_one | "
        "true_exactly_one_ratio | false_ticks_with_messages | false_message_ratio | "
        "before_wait(true/false) | after_wait(true/false)"
    )
    for sensor in required_sensors:
        stats = capture["sensor_stats"][sensor]
        lines.append(
            "{} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {}/{}".format(
                sensor,
                stats["gate_behavior"],
                stats["callbacks_on_capture_true_ticks"],
                stats["callbacks_on_capture_false_ticks"],
                stats["capture_true_ticks_exactly_one"],
                stats["capture_true_ticks_exactly_one_ratio"],
                stats["capture_false_ticks_with_messages"],
                stats["capture_false_ticks_with_messages_ratio"],
                stats["callbacks_before_wait_return"]["capture_true"],
                stats["callbacks_before_wait_return"]["capture_false"],
                stats["callbacks_after_wait_return"]["capture_true"],
                stats["callbacks_after_wait_return"]["capture_false"],
            )
        )
    lines.append(
        "all_required_exactly_one_on_true={}/{} ratio={}".format(
            capture["all_required_exactly_one_on_capture_true_ticks"],
            capture["capture_true_ticks"],
            capture["all_required_exactly_one_on_capture_true_ticks_ratio"],
        )
    )
    lines.append(
        "all_required_exactly_one_before_wait_return={}/{} ratio={}".format(
            capture["all_required_exactly_one_before_wait_return"],
            capture["capture_true_ticks"],
            capture["all_required_exactly_one_before_wait_return_ratio"],
        )
    )
    lines.append("SENSOR_CAPTURE_VERDICT=CASE {}: {}".format(
        capture["verdict"]["case"], capture["verdict"]["title"]
    ))
    lines.append("sensor_capture_reason={}".format(
        capture["verdict"]["reason"]
    ))
    lines.append("FRAME_IDENTITY_VERDICT=CASE {}: {}".format(
        summary["verdict"]["case"], summary["verdict"]["title"]
    ))
    lines.append("frame_identity_reason={}".format(summary["verdict"]["reason"]))
    return "\n".join(lines) + "\n"


def write_results(output_dir, recorder, summary, tick_report, arguments):
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "events.jsonl")
    with open(raw_path, "w", encoding="utf-8") as output:
        for event in sorted(recorder.events, key=lambda item: item["arrival_order"]):
            output.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    summary_document = {
        "arguments": vars(arguments),
        "summary": summary,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as output:
        json.dump(summary_document, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")

    report_path = os.path.join(output_dir, "tick_report.txt")
    with open(report_path, "w", encoding="utf-8") as output:
        output.write(tick_report)
    return raw_path, summary_path, report_path


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Probe MORAI native Sync frame identity and sensor_capture publication "
            "gating without timestamp matching."
        )
    )
    parser.add_argument("--ticks", type=int, default=120,
                        help="number of analyzed Sync ticks (minimum 100)")
    parser.add_argument("--warmup-ticks", type=int, default=10,
                        help="Sync ticks excluded from statistics")
    parser.add_argument("--time-step-ms", type=int, default=100)
    parser.add_argument(
        "--capture-stride",
        type=int,
        default=1,
        help=(
            "set sensor_capture=true when (request_frame + 1) is divisible by "
            "this value; 1 preserves the original all-true behavior"
        ),
    )
    parser.add_argument("--settle-sec", type=float, default=1.0,
                        help="observe callbacks after the final WaitForTick return")
    parser.add_argument("--service-timeout-sec", type=float, default=10.0)
    parser.add_argument("--user-id", default="MoraiSyncFrameProbe")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--camera-front", default="/camera_front")
    parser.add_argument("--camera-left", default="/camera_left")
    parser.add_argument("--camera-right", default="/camera_right")
    parser.add_argument("--camera-rear", default="/camera_rear")
    parser.add_argument("--lidar", default="/lidar3D")
    parser.add_argument("--objects", default="/Object_topic")
    parser.add_argument("--traffic-light", default="/GetTrafficLightStatus")
    parser.add_argument("--sync-info", default="/SyncModeInfo")
    parser.add_argument("--print-ticks", action="store_true",
                        help="also print the full per-tick report to stdout")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.ticks < 100:
        parser.error("--ticks must be at least 100 for this diagnostic")
    if args.warmup_ticks < 0:
        parser.error("--warmup-ticks cannot be negative")
    if args.time_step_ms <= 0:
        parser.error("--time-step-ms must be positive")
    if args.capture_stride <= 0:
        parser.error("--capture-stride must be positive")
    if args.settle_sec < 0:
        parser.error("--settle-sec cannot be negative")

    try:
        import rospy
        from sensor_msgs.msg import CompressedImage, PointCloud2
        from morai_msgs.msg import (
            CtrlCmd,
            GetTrafficLightStatus,
            ObjectStatusList,
            SyncModeCmd,
            SyncModeCtrlCmd,
            SyncModeInfo,
            WaitForTick,
        )
        from morai_msgs.srv import (
            MoraiSyncModeCmdSrv,
            MoraiSyncModeCtrlCmdSrv,
            MoraiWaitForTickSrv,
        )
    except ImportError as error:
        raise RuntimeError(
            "ROS/morai_msgs import failed. Run: source /opt/ros/noetic/setup.bash "
            "and source expert/devel/setup.bash"
        ) from error

    if not args.output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            "/tmp", "morai_sync_frame_probe_{}".format(timestamp)
        )
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    rospy.init_node("morai_sync_frame_probe", anonymous=True, disable_signals=True)
    recorder = EventRecorder()

    sensor_specs = {
        "camera_front": (args.camera_front, CompressedImage),
        "camera_left": (args.camera_left, CompressedImage),
        "camera_right": (args.camera_right, CompressedImage),
        "camera_rear": (args.camera_rear, CompressedImage),
        "lidar": (args.lidar, PointCloud2),
        "objects": (args.objects, ObjectStatusList),
        "traffic_light": (args.traffic_light, GetTrafficLightStatus),
    }
    subscribers = []
    for sensor_name, (topic, message_class) in sensor_specs.items():
        subscribers.append(rospy.Subscriber(
            topic,
            message_class,
            lambda message, name=sensor_name, source_topic=topic: recorder.sensor_callback(
                name, source_topic, message
            ),
            queue_size=100,
        ))
    subscribers.append(rospy.Subscriber(
        args.sync_info,
        SyncModeInfo,
        recorder.sync_info_callback,
        queue_size=100,
    ))

    service_types = {
        "/SyncModeCmd": MoraiSyncModeCmdSrv,
        "/SyncModeCtrlCmd": MoraiSyncModeCtrlCmdSrv,
        "/SyncModeWaitForTick": MoraiWaitForTickSrv,
    }
    for service_name in service_types:
        rospy.wait_for_service(service_name, timeout=args.service_timeout_sec)
    mode_service = rospy.ServiceProxy(
        "/SyncModeCmd", MoraiSyncModeCmdSrv, persistent=True
    )
    control_service = rospy.ServiceProxy(
        "/SyncModeCtrlCmd", MoraiSyncModeCtrlCmdSrv, persistent=True
    )
    tick_service = rospy.ServiceProxy(
        "/SyncModeWaitForTick", MoraiWaitForTickSrv, persistent=True
    )

    neutral = CtrlCmd()
    neutral.longlCmdType = 1
    neutral.accel = 0.0
    neutral.brake = 1.0
    neutral.front_steer = 0.0
    neutral.rear_steer = 0.0

    started = False
    user_id = str(args.user_id)
    current_frame = None
    run_error = None
    try:
        recorder.record_general("sync_start_call", user_id=user_id,
                                time_step_ms=int(args.time_step_ms))
        start_response = mode_service(SyncModeCmd(
            user_id=user_id,
            start_sync_mode=True,
            time_step=int(args.time_step_ms),
        )).response
        recorder.record_general(
            "sync_start_return",
            result=bool(start_response.result),
            user_id=str(start_response.user_id),
            frame=int(start_response.frame),
            time_step_ms=int(start_response.time_step),
        )
        if not start_response.result or int(start_response.time_step) <= 0:
            raise RuntimeError(
                "MORAI SyncMode start failed: result={} frame={} time_step={}".format(
                    start_response.result,
                    start_response.frame,
                    start_response.time_step,
                )
            )
        started = True
        user_id = str(start_response.user_id or user_id)
        current_frame = int(start_response.frame)
        total_ticks = int(args.warmup_ticks) + int(args.ticks)
        print(
            "[probe] SyncMode started user_id={} frame={} time_step={}ms; "
            "warmup={} analyzed={} capture_stride={}".format(
                user_id,
                current_frame,
                start_response.time_step,
                args.warmup_ticks,
                args.ticks,
                args.capture_stride,
            )
        )

        for tick_index in range(total_ticks):
            if rospy.is_shutdown():
                raise KeyboardInterrupt()
            analyzed = tick_index >= int(args.warmup_ticks)
            sensor_capture = (
                (int(current_frame) + 1) % int(args.capture_stride) == 0
            )
            recorder.begin_tick(
                tick_index, current_frame, analyzed, sensor_capture
            )
            control_response = control_service(SyncModeCtrlCmd(
                frame=current_frame,
                command=neutral,
                sensor_capture=sensor_capture,
            )).response
            recorder.mark_control_return(tick_index, control_response)
            if not control_response.result:
                raise RuntimeError(
                    "/SyncModeCtrlCmd failed at frame={}".format(current_frame)
                )

            recorder.mark_wait_call(tick_index)
            tick_response = tick_service(WaitForTick(
                user_id=user_id,
                frame=current_frame,
            )).response
            return_observed_ns = time.monotonic_ns()
            recorder.finish_tick(tick_index, tick_response, return_observed_ns)
            if not tick_response.tick_status:
                raise RuntimeError(
                    "/SyncModeWaitForTick failed at frame={}".format(current_frame)
                )
            next_frame = int(tick_response.frame)
            if next_frame <= current_frame:
                raise RuntimeError(
                    "Sync frame did not advance: before={} after={}".format(
                        current_frame, next_frame
                    )
                )
            current_frame = next_frame
            if analyzed and ((tick_index - args.warmup_ticks + 1) % 10 == 0):
                print("[probe] analyzed {}/{} ticks; sync_frame={}".format(
                    tick_index - args.warmup_ticks + 1, args.ticks, current_frame
                ))

        if args.settle_sec > 0:
            print("[probe] observing delayed callbacks for {:.3f}s".format(args.settle_sec))
            time.sleep(args.settle_sec)
    except BaseException as error:  # preserve partial diagnostics on Ctrl+C/errors
        run_error = error
        recorder.record_general(
            "probe_error",
            error_type=type(error).__name__,
            description=str(error),
        )
    finally:
        recorder.freeze_sensor_recording()
        if started:
            try:
                stop_response = mode_service(SyncModeCmd(
                    user_id=user_id,
                    start_sync_mode=False,
                    time_step=int(args.time_step_ms),
                )).response
                recorder.record_general(
                    "sync_stop_return",
                    result=bool(stop_response.result),
                    frame=int(stop_response.frame),
                )
            except BaseException as stop_error:
                recorder.record_general(
                    "sync_stop_error",
                    error_type=type(stop_error).__name__,
                    description=str(stop_error),
                )
                if run_error is None:
                    run_error = stop_error

        summary = analyze(recorder, REQUIRED_SENSORS)
        tick_report = build_tick_report(recorder, summary, REQUIRED_SENSORS)
        raw_path, summary_path, report_path = write_results(
            args.output_dir, recorder, summary, tick_report, args
        )
        print(tick_report.split("=== SUMMARY ===", 1)[-1].strip())
        print("[probe] raw events: {}".format(raw_path))
        print("[probe] summary: {}".format(summary_path))
        print("[probe] tick report: {}".format(report_path))
        if args.print_ticks:
            print("\n=== PER-TICK REPORT ===")
            print(tick_report)

    # Keep subscriber objects alive through result generation.
    del subscribers
    if run_error is not None:
        raise run_error
    return 0


if __name__ == "__main__":
    sys.exit(main())
