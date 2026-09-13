#!/usr/bin/env python3
"""
MORAI Stage 6E-1 Run-Oriented ROS Training Dataset Recorder.

Transport boundary
------------------
Training:
  ROS1 -> this recorder -> canonical dataset

Competition inference:
  UDP -> future UDP adapter -> same canonical model input contract

Synchronization clock
---------------------
The recorder intentionally uses PC `time.monotonic_ns()` receive time, not
MORAI `header.stamp`, for causal synchronization.

Observed reason:
- Camera frame-specific header clocks were monotonic.
- LiDAR header timestamps repeatedly moved backward during normal reception.
- GPS `rostopic hz` also showed a negative interval.
- PC receive monotonic time remained forward-moving.

Both timestamps are stored. Only monotonic receive time defines training
alignment.

Training-only /Ego_topic is recorded as privileged teacher data.
It is NOT a deployment/runtime model input.

This recorder still does NOT:
- create Global Path inputs,
- create final labels online,
- infer final trajectory/target speed,
- use /Ego_topic as a competition-time dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import rospy

from sensor_msgs.msg import (
    CompressedImage,
    Imu,
    PointCloud2,
)

try:
    from morai_msgs.msg import GPSMessage, EgoVehicleStatus
except ModuleNotFoundError as exc:
    raise SystemExit(
        "\nCould not import morai_msgs/GPSMessage or EgoVehicleStatus.\n"
        "Source both workspaces before running:\n\n"
        "  source /opt/ros/noetic/setup.bash\n"
        "  source ~/catkin_ws/devel/setup.bash\n\n"
        f"Original error: {exc}"
    )

from aimdrive.data.stage6e_runs import (
    append_jsonl,
    next_run_id,
    parse_tags,
    summarize_manifest,
)
from aimdrive.ros_training.teacher_integrity import validate_manifest

from aimdrive.ros_training.offline_contract import (
    BufferedSample,
    CURRENT_CAMERA_FRAME_MAP,
    MANDATORY_CAMERA_ROLES,
    decode_current_lidar_xyzi,
    latest_not_after_receive,
    quaternion_is_valid,
    quaternion_norm,
)


def ros_stamp_ns(header) -> Optional[int]:
    try:
        return int(
            header.stamp.to_nsec()
        )
    except Exception:
        return None


def provenance(msg):
    return {"seq": int(msg.header.seq),
            "callerid": (getattr(msg, "_connection_header", {}) or {}).get("callerid")}


def point_fields(msg: PointCloud2):
    return [
        {
            "name": field.name,
            "offset": field.offset,
            "datatype": field.datatype,
            "count": field.count,
        }
        for field in msg.fields
    ]


def percentile(values, fraction):
    if not values:
        return None

    ordered = sorted(float(x) for x in values)

    if len(ordered) == 1:
        return ordered[0]

    position = (
        len(ordered) - 1
    ) * fraction

    low = int(position)
    high = min(
        low + 1,
        len(ordered) - 1,
    )

    alpha = position - low

    return (
        ordered[low] * (1.0 - alpha)
        + ordered[high] * alpha
    )


def age_stats(values):
    if not values:
        return None

    values = [
        float(x)
        for x in values
    ]

    return {
        "count": len(values),
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
    }


class MoraiRosTrainingRecorder:
    def __init__(
        self,
        output_dir: Path,
        *,
        require_camera4: bool,
        require_gps: bool,
        max_buffer_samples: int,
        writer_queue_size: int,
        max_receive_age_ms: float = 250.0,
        preflight_enabled: bool = False,
    ) -> None:
        self.output_dir = output_dir
        if max_buffer_samples < 1 or writer_queue_size < 1 or max_receive_age_ms <= 0:
            raise ValueError("Buffer sizes and max receive age must be positive")
        self.max_receive_age_ms = float(max_receive_age_ms)
        self.writer_error = None
        self.recording_enabled = not preflight_enabled
        self.preview_rows = deque(maxlen=1024)
        self.callback_samples = defaultdict(lambda: deque(maxlen=2048))
        self.require_camera4 = bool(
            require_camera4
        )
        self.require_gps = bool(
            require_gps
        )
        self.max_buffer_samples = int(
            max_buffer_samples
        )

        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.buffers = {
            "front": deque(
                maxlen=max_buffer_samples
            ),
            "left": deque(
                maxlen=max_buffer_samples
            ),
            "right": deque(
                maxlen=max_buffer_samples
            ),
            "rear": deque(
                maxlen=max_buffer_samples
            ),
            "gps": deque(
                maxlen=max_buffer_samples
            ),
            "imu": deque(
                maxlen=max_buffer_samples
            ),
            "ego_teacher": deque(
                maxlen=max_buffer_samples
            ),
        }

        self.writer_queue = queue.Queue(
            maxsize=writer_queue_size
        )

        self.stats = {
            "camera_messages": defaultdict(int),
            "gps_messages": 0,
            "imu_messages": 0,
            "ego_teacher_messages": 0,
            "lidar_messages": 0,
            "frames_enqueued": 0,
            "frames_written": 0,
            "frames_skipped_missing": 0,
            "missing_sensor_count": defaultdict(int),
            "stale_samples_rejected": defaultdict(int),
            "frames_discarded_writer_error": 0,
            "writer_queue_drops": 0,
            "unknown_camera_frame_ids": defaultdict(int),
            "age_ms": defaultdict(list),
            "invalid_imu_quaternion_frames": 0,
        }

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for role in (
            "front",
            "left",
            "right",
            "rear",
        ):
            (
                self.output_dir
                / "camera"
                / role
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        (
            self.output_dir
            / "lidar"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.manifest_path = (
            self.output_dir
            / "manifest.jsonl"
        )
        self.manifest_file = open(
            self.manifest_path,
            "x",
            encoding="utf-8",
            buffering=1,
        )

        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            name="morai_dataset_writer",
            daemon=True,
        )
        self.writer_thread.start()

        self.subscribers = [
            rospy.Subscriber(
                "/image_jpeg/compressed",
                CompressedImage,
                self.camera_callback,
                queue_size=100,
                buff_size=128 * 1024 * 1024,
            ),
            rospy.Subscriber(
                "/gps",
                GPSMessage,
                self.gps_callback,
                queue_size=100,
            ),
            rospy.Subscriber(
                "/imu",
                Imu,
                self.imu_callback,
                queue_size=200,
            ),
            rospy.Subscriber(
                "/Ego_topic",
                EgoVehicleStatus,
                self.ego_teacher_callback,
                queue_size=200,
            ),
            rospy.Subscriber(
                "/lidar3D",
                PointCloud2,
                self.lidar_callback,
                queue_size=20,
                buff_size=128 * 1024 * 1024,
            ),
        ]

    def _append(
        self,
        source: str,
        sample: BufferedSample,
    ) -> None:
        self.buffers[source].append(
            sample
        )
        self.callback_samples[source].append({
            "receive_monotonic_ns": sample.receive_monotonic_ns,
            "receive_wall_ns": sample.receive_wall_ns,
            "source_stamp_ns": sample.source_stamp_ns,
            "seq": sample.seq, "callerid": sample.callerid,
        })

    def camera_callback(
        self,
        msg: CompressedImage,
    ) -> None:
        recv_mono = time.monotonic_ns()
        recv_wall = time.time_ns()

        role = CURRENT_CAMERA_FRAME_MAP.get(
            msg.header.frame_id
        )

        if role is None:
            with self.lock:
                self.stats[
                    "unknown_camera_frame_ids"
                ][msg.header.frame_id] += 1
            return

        payload = {
            "format": str(msg.format),
            "data": bytes(msg.data),
        }

        sample = BufferedSample(
            source_stamp_ns=ros_stamp_ns(
                msg.header
            ),
            receive_monotonic_ns=recv_mono,
            receive_wall_ns=recv_wall,
            frame_id=str(
                msg.header.frame_id
            ),
            payload=payload,
            **provenance(msg),
        )

        with self.lock:
            self._append(
                role,
                sample,
            )
            self.stats[
                "camera_messages"
            ][role] += 1

    def gps_callback(
        self,
        msg: GPSMessage,
    ) -> None:
        recv_mono = time.monotonic_ns()
        recv_wall = time.time_ns()

        payload = {
            "latitude": float(
                msg.latitude
            ),
            "longitude": float(
                msg.longitude
            ),
            "altitude": float(
                msg.altitude
            ),
            "eastOffset": float(
                msg.eastOffset
            ),
            "northOffset": float(
                msg.northOffset
            ),
            "status": int(
                msg.status
            ),
        }

        sample = BufferedSample(
            source_stamp_ns=ros_stamp_ns(
                msg.header
            ),
            receive_monotonic_ns=recv_mono,
            receive_wall_ns=recv_wall,
            frame_id=str(
                msg.header.frame_id
            ),
            payload=payload,
            **provenance(msg),
        )

        with self.lock:
            self._append(
                "gps",
                sample,
            )
            self.stats[
                "gps_messages"
            ] += 1

    def imu_callback(
        self,
        msg: Imu,
    ) -> None:
        recv_mono = time.monotonic_ns()
        recv_wall = time.time_ns()

        q = msg.orientation
        q_norm = quaternion_norm(
            q.x,
            q.y,
            q.z,
            q.w,
        )
        q_valid = quaternion_is_valid(
            q.x,
            q.y,
            q.z,
            q.w,
        )

        payload = {
            "orientation_xyzw": [
                float(q.x),
                float(q.y),
                float(q.z),
                float(q.w),
            ],
            "orientation_norm": float(
                q_norm
            ),
            "orientation_valid": bool(
                q_valid
            ),
            "orientation_covariance": [
                float(x)
                for x in msg.orientation_covariance
            ],
            "angular_velocity_xyz": [
                float(
                    msg.angular_velocity.x
                ),
                float(
                    msg.angular_velocity.y
                ),
                float(
                    msg.angular_velocity.z
                ),
            ],
            "angular_velocity_covariance": [
                float(x)
                for x in msg.angular_velocity_covariance
            ],
            "linear_acceleration_xyz": [
                float(
                    msg.linear_acceleration.x
                ),
                float(
                    msg.linear_acceleration.y
                ),
                float(
                    msg.linear_acceleration.z
                ),
            ],
            "linear_acceleration_covariance": [
                float(x)
                for x in msg.linear_acceleration_covariance
            ],
        }

        sample = BufferedSample(
            source_stamp_ns=ros_stamp_ns(
                msg.header
            ),
            receive_monotonic_ns=recv_mono,
            receive_wall_ns=recv_wall,
            frame_id=str(
                msg.header.frame_id
            ),
            payload=payload,
            **provenance(msg),
        )

        with self.lock:
            self._append(
                "imu",
                sample,
            )
            self.stats[
                "imu_messages"
            ] += 1

    def ego_teacher_callback(
        self,
        msg: EgoVehicleStatus,
    ) -> None:
        """
        Training-only privileged teacher signal.

        IMPORTANT:
        - This data may be used for labels, supervision, validation, and
          offline diagnostics.
        - It must not become a required competition-time model input.
        """
        recv_mono = time.monotonic_ns()
        recv_wall = time.time_ns()

        payload = {
            "teacher_only": True,
            "runtime_model_input": False,
            "unique_id": int(
                msg.unique_id
            ),
            "position_xyz": [
                float(msg.position.x),
                float(msg.position.y),
                float(msg.position.z),
            ],
            "velocity_xyz": [
                float(msg.velocity.x),
                float(msg.velocity.y),
                float(msg.velocity.z),
            ],
            "speed_xy_mps": float(
                (
                    msg.velocity.x * msg.velocity.x
                    + msg.velocity.y * msg.velocity.y
                ) ** 0.5
            ),
            "acceleration_xyz": [
                float(msg.acceleration.x),
                float(msg.acceleration.y),
                float(msg.acceleration.z),
            ],
            "angular_velocity_xyz": [
                float(msg.angular_velocity.x),
                float(msg.angular_velocity.y),
                float(msg.angular_velocity.z),
            ],
            "heading_deg": float(
                msg.heading
            ),
            "accel_command": float(
                msg.accel
            ),
            "brake_command": float(
                msg.brake
            ),
            "front_steer_angle": float(
                msg.front_steer_angle
            ),
            "rear_steer_angle": float(
                msg.rear_steer_angle
            ),
        }

        sample = BufferedSample(
            source_stamp_ns=ros_stamp_ns(
                msg.header
            ),
            receive_monotonic_ns=recv_mono,
            receive_wall_ns=recv_wall,
            frame_id=str(
                msg.header.frame_id
            ),
            payload=payload,
            **provenance(msg),
        )

        with self.lock:
            self._append(
                "ego_teacher",
                sample,
            )
            self.stats[
                "ego_teacher_messages"
            ] += 1

    def _select_snapshot(
        self,
        anchor_receive_monotonic_ns: int,
    ):
        required = list(
            MANDATORY_CAMERA_ROLES
        ) + [
            "imu",
            "ego_teacher",
        ]

        if self.require_camera4:
            required.append(
                "rear"
            )

        if self.require_gps:
            required.append(
                "gps"
            )

        selected = {}

        for source in (
            "front",
            "left",
            "right",
            "rear",
            "gps",
            "imu",
            "ego_teacher",
        ):
            sample = latest_not_after_receive(
                self.buffers[source],
                anchor_receive_monotonic_ns,
            )

            if sample is not None:
                if anchor_receive_monotonic_ns - sample.receive_monotonic_ns > self.max_receive_age_ms * 1e6:
                    self.stats["stale_samples_rejected"][source] += 1
                    continue
                selected[source] = sample

        missing = [
            source
            for source in required
            if source not in selected
        ]

        return selected, missing

    def lidar_callback(
        self,
        msg: PointCloud2,
    ) -> None:
        recv_mono = time.monotonic_ns()
        recv_wall = time.time_ns()

        lidar_payload = {
            **provenance(msg),
            "source_stamp_ns": (
                ros_stamp_ns(
                    msg.header
                )
            ),
            "receive_monotonic_ns": (
                recv_mono
            ),
            "receive_wall_ns": (
                recv_wall
            ),
            "frame_id": str(
                msg.header.frame_id
            ),
            "width": int(
                msg.width
            ),
            "height": int(
                msg.height
            ),
            "point_step": int(
                msg.point_step
            ),
            "row_step": int(
                msg.row_step
            ),
            "is_bigendian": bool(
                msg.is_bigendian
            ),
            "is_dense": bool(
                msg.is_dense
            ),
            "fields": point_fields(
                msg
            ),
            "data": bytes(
                msg.data
            ),
        }

        with self.lock:
            if self.stop_event.is_set() or self.writer_error is not None:
                return
            self.stats[
                "lidar_messages"
            ] += 1
            self.callback_samples["lidar"].append({k: lidar_payload[k] for k in
                                                   ("receive_monotonic_ns", "receive_wall_ns", "source_stamp_ns", "seq", "callerid")})

            selected, missing = (
                self._select_snapshot(
                    recv_mono
                )
            )

            preview = {"lidar": {k: v for k, v in lidar_payload.items() if k not in ("data", "fields")}, "cameras": {}}
            for role, sample in selected.items():
                value = {"receive_monotonic_ns": sample.receive_monotonic_ns,
                         "receive_wall_ns": sample.receive_wall_ns, "source_stamp_ns": sample.source_stamp_ns,
                         "seq": sample.seq, "callerid": sample.callerid}
                if role in ("front", "left", "right", "rear"):
                    preview["cameras"][role] = value
                else:
                    value.update(sample.payload)
                    preview[role] = value
            self.preview_rows.append(preview)
            if not self.recording_enabled:
                return

            if missing:
                for role in missing:
                    self.stats["missing_sensor_count"][role] += 1
                self.stats[
                    "frames_skipped_missing"
                ] += 1
                return

            frame_index = (
                self.stats[
                    "frames_enqueued"
                ]
            )

            snapshot = {
                "enqueued_monotonic_ns": time.monotonic_ns(),
                "frame_index": (
                    frame_index
                ),
                "lidar": (
                    lidar_payload
                ),
                "selected": (
                    selected
                ),
            }

            try:
                self.writer_queue.put_nowait(
                    snapshot
                )
                self.stats[
                    "frames_enqueued"
                ] += 1
            except queue.Full:
                self.stats[
                    "writer_queue_drops"
                ] += 1

    def _camera_extension(
        self,
        compressed_format: str,
    ) -> str:
        lowered = (
            compressed_format
            .lower()
        )

        if "png" in lowered:
            return ".png"

        return ".jpg"

    def _sample_metadata(
        self,
        source: str,
        sample: BufferedSample,
        anchor_mono: int,
    ):
        age_ms = (
            anchor_mono
            - sample.receive_monotonic_ns
        ) / 1e6

        self.stats[
            "age_ms"
        ][source].append(
            float(age_ms)
        )

        return {
            "seq": sample.seq,
            "callerid": sample.callerid,
            "frame_id": (
                sample.frame_id
            ),
            "source_stamp_ns": (
                sample.source_stamp_ns
            ),
            "receive_monotonic_ns": (
                sample.receive_monotonic_ns
            ),
            "receive_wall_ns": (
                sample.receive_wall_ns
            ),
            "age_ms": float(
                age_ms
            ),
        }

    def _write_snapshot(
        self,
        snapshot,
    ) -> None:
        write_started_ns = time.monotonic_ns()
        index = int(
            snapshot[
                "frame_index"
            ]
        )
        stem = f"{index:08d}"

        lidar = snapshot[
            "lidar"
        ]
        selected = snapshot[
            "selected"
        ]

        anchor_mono = int(
            lidar[
                "receive_monotonic_ns"
            ]
        )

        xyzi = decode_current_lidar_xyzi(
            lidar["data"],
            width=lidar["width"],
            height=lidar["height"],
            point_step=lidar["point_step"],
            row_step=lidar["row_step"],
            is_bigendian=lidar[
                "is_bigendian"
            ],
            fields=lidar["fields"],
        )

        lidar_rel = (
            Path("lidar")
            / f"{stem}.npy"
        )

        np.save(
            self.output_dir
            / lidar_rel,
            xyzi,
            allow_pickle=False,
        )

        record = {
            "writer_queue_wait_ms": (write_started_ns - snapshot["enqueued_monotonic_ns"]) / 1e6,
            "schema_version": (
                "morai-stage6e-run-v1"
            ),
            "frame_index": (
                index
            ),
            "sync_clock": (
                "receive_monotonic_ns"
            ),
            "anchor_source": (
                "lidar"
            ),
            "lidar": {
                "seq": lidar["seq"],
                "callerid": lidar["callerid"],
                "path": str(
                    lidar_rel
                ),
                "shape": list(
                    xyzi.shape
                ),
                "dtype": str(
                    xyzi.dtype
                ),
                "frame_id": (
                    lidar[
                        "frame_id"
                    ]
                ),
                "source_stamp_ns": (
                    lidar[
                        "source_stamp_ns"
                    ]
                ),
                "receive_monotonic_ns": (
                    anchor_mono
                ),
                "receive_wall_ns": (
                    lidar[
                        "receive_wall_ns"
                    ]
                ),
                "fields": (
                    lidar[
                        "fields"
                    ]
                ),
                "point_step": (
                    lidar[
                        "point_step"
                    ]
                ),
            },
            "cameras": {},
            "gps": None,
            "imu": None,
            "ego_teacher": None,
            "labels": None,
            "global_path": None,
        }

        for role in (
            "front",
            "left",
            "right",
            "rear",
        ):
            sample = selected.get(
                role
            )

            if sample is None:
                continue

            extension = (
                self._camera_extension(
                    sample.payload[
                        "format"
                    ]
                )
            )

            relative_path = (
                Path("camera")
                / role
                / f"{stem}{extension}"
            )

            absolute_path = (
                self.output_dir
                / relative_path
            )

            absolute_path.write_bytes(
                sample.payload[
                    "data"
                ]
            )

            camera_record = (
                self._sample_metadata(
                    role,
                    sample,
                    anchor_mono,
                )
            )

            camera_record.update(
                {
                    "path": str(
                        relative_path
                    ),
                    "compressed_format": (
                        sample.payload[
                            "format"
                        ]
                    ),
                }
            )

            record[
                "cameras"
            ][role] = camera_record

        gps = selected.get(
            "gps"
        )

        if gps is not None:
            gps_record = (
                self._sample_metadata(
                    "gps",
                    gps,
                    anchor_mono,
                )
            )

            gps_record.update(
                gps.payload
            )

            record["gps"] = gps_record

        imu = selected["imu"]

        imu_record = (
            self._sample_metadata(
                "imu",
                imu,
                anchor_mono,
            )
        )

        imu_record.update(
            imu.payload
        )

        record["imu"] = (
            imu_record
        )

        if not imu.payload[
            "orientation_valid"
        ]:
            self.stats[
                "invalid_imu_quaternion_frames"
            ] += 1

        ego_teacher = selected[
            "ego_teacher"
        ]

        ego_teacher_record = (
            self._sample_metadata(
                "ego_teacher",
                ego_teacher,
                anchor_mono,
            )
        )

        ego_teacher_record.update(
            ego_teacher.payload
        )

        record[
            "ego_teacher"
        ] = ego_teacher_record

        record["write_before_manifest_ms"] = (time.monotonic_ns() - write_started_ns) / 1e6
        self.manifest_file.write(
            json.dumps(
                record,
                separators=(
                    ",",
                    ":",
                ),
            )
            + "\n"
        )

        self.stats[
            "frames_written"
        ] += 1

    def _writer_loop(
        self,
    ) -> None:
        while (
            not self.stop_event.is_set()
            or not self.writer_queue.empty()
        ):
            try:
                snapshot = (
                    self.writer_queue.get(
                        timeout=0.2
                    )
                )
            except queue.Empty:
                continue

            try:
                if self.writer_error is None:
                    self._write_snapshot(snapshot)
                else:
                    self.stats["frames_discarded_writer_error"] += 1
            except Exception as exc:
                # Keep draining: otherwise queue.join() hangs forever after a
                # decode/disk error kills the only writer thread.
                self.writer_error = repr(exc)
                self.stats["frames_discarded_writer_error"] += 1
                rospy.logerr("Dataset writer failed: %s", self.writer_error)
            finally:
                self.writer_queue.task_done()

    def stop(
        self,
    ) -> None:
        # Synchronize with any in-flight LiDAR callback before draining.
        with self.lock:
            self.stop_event.set()
        for subscriber in (
            self.subscribers
        ):
            subscriber.unregister()

        self.stop_event.set()
        self.writer_queue.join()
        self.writer_thread.join(
            timeout=5.0
        )

        self.manifest_file.flush()
        self.manifest_file.close()

        serializable = {
            "missing_sensor_count": dict(self.stats["missing_sensor_count"]),
            "ego_teacher_messages": self.stats["ego_teacher_messages"],
            "writer_error": self.writer_error,
            "frames_discarded_writer_error": self.stats["frames_discarded_writer_error"],
            "stale_samples_rejected": dict(self.stats["stale_samples_rejected"]),
            "max_receive_age_ms": self.max_receive_age_ms,
            "camera_messages": dict(
                self.stats[
                    "camera_messages"
                ]
            ),
            "gps_messages": (
                self.stats[
                    "gps_messages"
                ]
            ),
            "imu_messages": (
                self.stats[
                    "imu_messages"
                ]
            ),
            "lidar_messages": (
                self.stats[
                    "lidar_messages"
                ]
            ),
            "frames_enqueued": (
                self.stats[
                    "frames_enqueued"
                ]
            ),
            "frames_written": (
                self.stats[
                    "frames_written"
                ]
            ),
            "frames_skipped_missing": (
                self.stats[
                    "frames_skipped_missing"
                ]
            ),
            "writer_queue_drops": (
                self.stats[
                    "writer_queue_drops"
                ]
            ),
            "unknown_camera_frame_ids": dict(
                self.stats[
                    "unknown_camera_frame_ids"
                ]
            ),
            "invalid_imu_quaternion_frames": (
                self.stats[
                    "invalid_imu_quaternion_frames"
                ]
            ),
            "age_ms": {
                source: age_stats(
                    values
                )
                for source, values
                in self.stats[
                    "age_ms"
                ].items()
            },
        }

        (
            self.output_dir
            / "stats.json"
        ).write_text(
            json.dumps(
                serializable,
                indent=2,
            ),
            encoding="utf-8",
        )

        session = {
            "schema_version": (
                "morai-stage6e-run-v1"
            ),
            "purpose": (
                "ROS1 training data acquisition only"
            ),
            "deployment_transport": (
                "UDP"
            ),
            "sync_clock": (
                "PC time.monotonic_ns receive time"
            ),
            "anchor_source": (
                "/lidar3D arrival"
            ),
            "topics": {
                "ego_teacher": "/Ego_topic",
                "camera": (
                    "/image_jpeg/compressed"
                ),
                "lidar": (
                    "/lidar3D"
                ),
                "gps": (
                    "/gps"
                ),
                "imu": (
                    "/imu"
                ),
            },
            "camera_frame_mapping_current": (
                CURRENT_CAMERA_FRAME_MAP
            ),
            "camera4_note": (
                "Current Camera-4 image is rear-facing; "
                "fourth camera remains configurable by the competition notice."
            ),
            "gps_contract": {
                "type": (
                    "morai_msgs/GPSMessage"
                ),
                "fields": [
                    "latitude",
                    "longitude",
                    "altitude",
                    "eastOffset",
                    "northOffset",
                    "status",
                ],
            },
            "imu_contract": {
                "type": (
                    "sensor_msgs/Imu"
                ),
                "orientation_policy": (
                    "Stored raw. Invalid zero-norm quaternion is flagged "
                    "and never converted to yaw."
                ),
            },
            "lidar_contract": {
                "type": (
                    "sensor_msgs/PointCloud2"
                ),
                "frame_id_observed": (
                    "Lidar3D-5"
                ),
                "canonical_array": (
                    "float32 Nx4 [x,y,z,intensity]"
                ),
            },
            "global_path": (
                "NOT RECORDED YET: official/provider format not yet connected"
            ),
            "labels": (
                "NOT RECORDED YET"
            ),
        }

        (
            self.output_dir
            / "session.json"
        ).write_text(
            json.dumps(
                session,
                indent=2,
            ),
            encoding="utf-8",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-receive-age-ms", type=float, default=250.0,
                        help="Maximum callback-receive age; does not certify source freshness")
    parser.add_argument("--preflight-timeout", type=float, default=180,
                        help="Maximum wait for all sensors and independently consistent motion; duration starts after READY")

    parser.add_argument(
        "--runs-root",
        default=str(REPO_ROOT / "data/stage6e_runs"),
    )
    parser.add_argument(
        "--run-id",
        default="auto",
        help=(
            "'auto' creates run_0001, run_0002, ... "
            "without frame-level split leakage."
        ),
    )
    parser.add_argument(
        "--scenario",
        default="unspecified",
        help=(
            "User-defined data-management label only; "
            "it does not imply an unpublished competition mission."
        ),
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help=(
            "Repeatable/comma-separated user data tag, "
            "e.g. --tag curve --tag deviation,recovery"
        ),
    )
    parser.add_argument(
        "--notes",
        default="",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help=(
            "Wall-clock acquisition duration. "
            "0 = until Ctrl-C. MORAI may run below real-time."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--require-camera4",
        action="store_true",
    )
    parser.add_argument(
        "--require-gps",
        action="store_true",
        help=(
            "Require fresh GPS for every raw frame. GPS remains diagnostic; "
            "moving GPS/Ego consistency is checked before training approval."
        ),
    )
    parser.add_argument(
        "--max-buffer-samples",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--writer-queue-size",
        type=int,
        default=64,
    )

    args = parser.parse_args()
    # Production Stage6E always requires the complete validated four-camera setup.
    args.require_camera4 = True
    args.require_gps = True
    if args.preflight_timeout <= 0 or args.duration < 0:
        parser.error("Preflight timeout must be positive and duration nonnegative")
    if args.max_buffer_samples < 1 or args.writer_queue_size < 1 or args.max_receive_age_ms <= 0:
        parser.error("Buffer sizes and max receive age must be positive")

    runs_root = Path(
        args.runs_root
    ).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if runs_root != repo_root and repo_root not in runs_root.parents:
        parser.error("--runs-root must stay inside this repository")
    os.environ.setdefault("ROS_LOG_DIR", str(repo_root / "diagnostics" / "ros_logs"))
    runs_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.run_id == "auto":
        run_id = next_run_id(
            runs_root
        )
    else:
        run_id = str(
            args.run_id
        ).strip()

    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise SystemExit(
            "--run-id cannot be empty"
        )

    output = (
        runs_root
        / run_id
    )

    if output.exists():
        raise SystemExit(
            "Run directory already exists: {}\n"
            "Use a new run ID; Stage 6E never appends to an existing run."
            .format(output)
        )

    tags = parse_tags(
        args.tag
    )

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    start_wall_ns = time.time_ns()
    start_mono_ns = time.monotonic_ns()

    metadata = {
        "schema_version": (
            "morai-stage6e-run-meta-v1"
        ),
        "run_id": run_id,
        "scenario": str(
            args.scenario
        ),
        "tags": tags,
        "notes": str(
            args.notes
        ),
        "status": "preflight",
        "quality_status": "warning",
        "reasons": ["preflight_pending"],
        "start_wall_ns": int(
            start_wall_ns
        ),
        "collection_transport": (
            "ROS1"
        ),
        "competition_transport": (
            "UDP"
        ),
        "map": (
            "R-KR_PG_K-City_2025"
        ),
        "sync_clock": (
            "receive_monotonic_ns"
        ),
        "trajectory_teacher_timebase": (
            "/Ego_topic source_stamp_ns"
        ),
        "training_teacher": {
            "source": "/Ego_topic",
            "runtime_model_input": False,
        },
        "gps": {
            "record_if_available": True,
            "required_for_recording": bool(
                args.require_gps
            ),
            "current_ros_status": (
                "measured per run; never assume GPS is permanently frozen"
            ),
        },
        "global_path": {
            "provider_path": (
                "data/competition/global_path.txt"
            ),
            "recorded_online": False,
            "postprocess_with_ego_teacher_localization": True,
        },
    }

    (
        output
        / "run_meta.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    rospy.init_node(
        "morai_stage6e_run_recorder",
        anonymous=True,
        disable_signals=True,
    )

    recorder = MoraiRosTrainingRecorder(
        output,
        require_camera4=(
            args.require_camera4
        ),
        require_gps=(
            args.require_gps
        ),
        max_buffer_samples=(
            args.max_buffer_samples
        ),
        writer_queue_size=(
            args.writer_queue_size
        ),
        max_receive_age_ms=args.max_receive_age_ms,
        preflight_enabled=True,
    )

    from aimdrive.ros_training.acquisition_monitor import AcquisitionMonitor
    monitor = AcquisitionMonitor(recorder, output)
    acquisition_failure = None
    try:
        monitor.wait_ready(args.preflight_timeout)
    except (RuntimeError, KeyboardInterrupt) as exc:
        acquisition_failure = str(exc) or "preflight_interrupted"
    if acquisition_failure is None:
        with recorder.lock:
            recorder.recording_enabled = True
        start_mono_ns = time.monotonic_ns()
        metadata.update(status="recording", recording_start_monotonic_ns=start_mono_ns,
                        preflight=monitor.last_decision, bridge_instance_id=monitor.instance_id)
        (output / "run_meta.json").write_text(json.dumps(metadata, indent=2))
        monitor.sample("recording")

    print("=" * 108)
    print(
        "MORAI Stage 6E-1 — RUN DATASET RECORDER"
    )
    print("=" * 108)
    print("run id          :", run_id)
    print("output          :", output)
    print("scenario        :", args.scenario)
    print("tags            :", tags)
    print(
        "sync clock      : receive_monotonic_ns"
    )
    print(
        "teacher time    : /Ego_topic source_stamp_ns"
    )
    print(
        "anchor          : /lidar3D callback arrival"
    )
    print(
        "require Camera4 :",
        args.require_camera4,
    )
    print(
        "require GPS     :",
        args.require_gps,
    )
    print(
        "GPS note        : diagnostic motion reference; not a model input or trajectory teacher."
    )
    print(
        "READY: recording duration starts now. " if acquisition_failure is None else "Recording blocked: " + acquisition_failure,
        "Use varied but safe route-following / curve / recovery behavior."
    )
    print("=" * 108)

    start = time.monotonic()
    next_health = start + 1

    try:
        while acquisition_failure is None and not rospy.is_shutdown():
            if recorder.writer_error is not None:
                break
            elapsed = (
                time.monotonic()
                - start
            )
            if time.monotonic() >= next_health:
                decision = monitor.sample("recording")
                next_health = time.monotonic() + 1
                if decision["state"] in ("BRIDGE_NOT_PATCHED", "SENSOR_MISSING", "INVALID_EGO_TEACHER"):
                    acquisition_failure = decision["state"] + ": " + "; ".join(decision["reasons"])
                    break

            if (
                args.duration > 0
                and elapsed >= args.duration
            ):
                break

            if (
                args.max_frames > 0
                and recorder.stats[
                    "frames_written"
                ] >= args.max_frames
            ):
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        pass

    finally:
        recorder.stop()
        monitor.close()

    end_wall_ns = time.time_ns()
    end_mono_ns = time.monotonic_ns()

    quality = summarize_manifest(
        output
        / "manifest.jsonl"
    )
    with (output / "manifest.jsonl").open() as handle:
        integrity = validate_manifest([json.loads(line) for line in handle if line.strip()],
                                      max_age_ms=args.max_receive_age_ms)
    (output / "teacher_validation.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    metadata.update(
        {
            "acquisition_failure": acquisition_failure,
            "status": ("failed" if recorder.writer_error or not quality.get("frames") else
                       "recorded" if integrity["status"] == "pass" else "quarantined"),
            "teacher_validation": {"status": integrity["status"], "failures": integrity["failures"]},
            "writer_error": recorder.writer_error,
            "end_wall_ns": int(
                end_wall_ns
            ),
            "wall_duration_s": float(
                (
                    end_mono_ns
                    - start_mono_ns
                )
                / 1e9
            ),
            "quality_summary": quality,
            "postprocessed": False,
        }
    )

    (
        output
        / "run_meta.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    from aimdrive.data.stage6e_validation import validate_run
    quality_report = validate_run(output, check_artifacts=False)
    metadata.update(quality_status=quality_report["quality_status"], reasons=quality_report["reasons"])
    if quality_report["quality_status"] != "valid":
        metadata["status"] = "quarantined"
    (output / "quality_report.json").write_text(json.dumps(quality_report, indent=2))
    (output / "run_meta.json").write_text(json.dumps(metadata, indent=2))

    registry_record = {
        "quality_status": metadata["quality_status"],
        "reasons": metadata["reasons"],
        "teacher_validation_status": integrity["status"],
        "schema_version": (
            "morai-stage6e-run-index-v1"
        ),
        "run_id": run_id,
        "scenario": str(
            args.scenario
        ),
        "tags": tags,
        "relative_path": run_id,
        "frames": int(
            quality.get(
                "frames",
                0,
            )
        ),
        "wall_duration_s": (
            metadata[
                "wall_duration_s"
            ]
        ),
        "postprocessed": False,
    }

    append_jsonl(
        runs_root
        / "runs.jsonl",
        registry_record,
    )

    print()
    print("=" * 108)
    print("Teacher validation:", integrity["status"], integrity["failures"])
    print("RUN QUALITY SUMMARY")
    print("=" * 108)
    print(
        json.dumps(
            quality,
            indent=2,
        )
    )
    print()
    print(
        "run meta :",
        output
        / "run_meta.json"
    )
    print(
        "registry :",
        runs_root
        / "runs.jsonl"
    )
    print(
        "manifest :",
        output
        / "manifest.jsonl"
    )
    print()
    if metadata["status"] != "recorded":
        raise SystemExit("Raw data preserved; teacher validation did not pass. Do not use this run for training.")
    print(
        "NEXT: postprocess this run with the lmdrive environment."
    )
    print("=" * 108)


if __name__ == "__main__":
    main()
