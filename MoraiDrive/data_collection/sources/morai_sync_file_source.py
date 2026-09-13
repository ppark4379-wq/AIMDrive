"""Load camera/LiDAR payloads saved by MORAI for an exact native frame N."""

from dataclasses import dataclass
import os
import time
from typing import Dict, Optional

import cv2
import numpy as np


# Active AIM sensor-set IDs (aim_sensorset.json, MORAI 26.R1.H3).
REQUIRED_SENSOR_DIRS = (
    "CAMERA_1", "CAMERA_2", "CAMERA_3", "CAMERA_4", "LIDAR_5"
)
MANIFEST_TO_VIEW = {
    "camera_front": "front",
    "camera_left": "left",
    "camera_right": "right",
    "camera_rear": "rear",
}


@dataclass(frozen=True)
class NativeImagePayload:
    encoded: Optional[bytes]
    encoding: str = "png"
    image: object = None


@dataclass(frozen=True)
class NativeFramePayload:
    native_sync_frame: int
    cameras: Dict[str, NativeImagePayload]
    lidar_points: np.ndarray


class IncompleteNativeFrame(RuntimeError):
    def __init__(self, native_sync_frame, missing_files, timeout_reason):
        super().__init__(timeout_reason)
        self.native_sync_frame = int(native_sync_frame)
        self.missing_files = list(missing_files)
        self.timeout_reason = str(timeout_reason)


def _looks_like_sensor_session(path):
    # Missing sensor directories are handled per frame and recorded in
    # incomplete_frames.jsonl. Discovery must not hide such a session.
    return os.path.isdir(path) and any(
        os.path.isdir(os.path.join(path, name)) for name in REQUIRED_SENSOR_DIRS
    )


def discover_sync_session_root(explicit_root="", base_root=""):
    """Resolve an exact MORAI save session; never scan outside a supplied base."""
    if explicit_root:
        root = os.path.abspath(os.path.expanduser(explicit_root))
        if not os.path.isdir(root):
            raise RuntimeError("MORAI sync session root does not exist: {}".format(root))
        return root

    if not base_root:
        raise RuntimeError("Specify --morai-sync-root or --morai-sync-base")
    base = os.path.abspath(os.path.expanduser(base_root))
    if _looks_like_sensor_session(base):
        return base
    if not os.path.isdir(base):
        raise RuntimeError("MORAI sync session base does not exist: {}".format(base))
    candidates = []
    for name in os.listdir(base):
        candidate = os.path.join(base, name)
        if _looks_like_sensor_session(candidate):
            candidates.append(candidate)
    if not candidates:
        raise RuntimeError("No MORAI synchronous sensor session under {}".format(base))
    return max(candidates, key=os.path.getmtime)


def discover_split_sync_session_roots(camera_base, lidar_base):
    """Select the newest session directory name present under both bases."""
    camera_base = os.path.abspath(os.path.expanduser(camera_base))
    lidar_base = os.path.abspath(os.path.expanduser(lidar_base))
    for base, label in ((camera_base, "camera"), (lidar_base, "lidar")):
        if not os.path.isdir(base):
            raise RuntimeError("MORAI {} sync base does not exist: {}".format(label, base))

    camera_sessions = {
        name for name in os.listdir(camera_base)
        if os.path.isdir(os.path.join(camera_base, name))
    }
    lidar_sessions = {
        name for name in os.listdir(lidar_base)
        if os.path.isdir(os.path.join(lidar_base, name))
    }
    common = camera_sessions & lidar_sessions
    if not common:
        raise RuntimeError(
            "No matching MORAI session id under camera/lidar sync bases"
        )
    session_id = max(
        common,
        key=lambda name: max(
            os.path.getmtime(os.path.join(camera_base, name)),
            os.path.getmtime(os.path.join(lidar_base, name)),
        ),
    )
    return (
        os.path.join(camera_base, session_id),
        os.path.join(lidar_base, session_id),
        session_id,
    )


class MoraiSyncFileSource:
    def __init__(self, session_root, lidar_session_root=None, timeout_sec=5.0,
                 poll_interval_sec=0.05, stable_checks=2):
        self.session_root = os.path.abspath(session_root)
        self.lidar_session_root = os.path.abspath(
            lidar_session_root or session_root
        )
        self.timeout_sec = max(0.0, float(timeout_sec))
        self.poll_interval_sec = max(0.005, float(poll_interval_sec))
        self.stable_checks = max(1, int(stable_checks))

    def _resolve_expected(self, record):
        resolved = {}
        for name, relative in record["expected_sensor_files"].items():
            relative = str(relative)
            if os.path.isabs(relative):
                raise ValueError("Expected sensor path must be relative: {}".format(relative))
            root = self.lidar_session_root if name == "lidar" else self.session_root
            path = os.path.abspath(os.path.join(root, relative))
            if os.path.commonpath((root, path)) != root:
                raise ValueError("Expected sensor path escapes session root: {}".format(relative))
            resolved[str(name)] = path
        required = set(MANIFEST_TO_VIEW) | {"lidar"}
        missing_keys = required - set(resolved)
        if missing_keys:
            raise ValueError("Manifest missing expected file keys: {}".format(sorted(missing_keys)))
        return resolved

    def load(self, record):
        native_frame = int(record["native_sync_frame"])
        expected = self._resolve_expected(record)
        deadline = time.monotonic() + self.timeout_sec
        prior_sizes = None
        stable_count = 0
        while True:
            missing = [
                os.path.relpath(
                    path,
                    self.lidar_session_root if name == "lidar" else self.session_root,
                )
                for name, path in expected.items()
                if not os.path.isfile(path) or os.path.getsize(path) <= 0
            ]
            if not missing:
                sizes = tuple(os.path.getsize(expected[name]) for name in sorted(expected))
                if sizes == prior_sizes:
                    stable_count += 1
                else:
                    prior_sizes = sizes
                    stable_count = 1
                if stable_count >= self.stable_checks:
                    try:
                        return self._decode(native_frame, expected)
                    except (OSError, ValueError, RuntimeError, cv2.error) as error:
                        decode_error = str(error)
                else:
                    decode_error = "files are not size-stable yet"
            else:
                decode_error = "required files are missing"
            if time.monotonic() >= deadline:
                raise IncompleteNativeFrame(
                    native_frame,
                    missing,
                    "native frame payload timeout: {}".format(decode_error),
                )
            time.sleep(self.poll_interval_sec)

    def _decode(self, native_frame, expected):
        cameras = {}
        for manifest_name, view_name in MANIFEST_TO_VIEW.items():
            with open(expected[manifest_name], "rb") as source:
                encoded = source.read()
            decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError("cannot decode {}".format(expected[manifest_name]))
            # Reuse the image validated here. Decoding again in the writer
            # bypasses this loader's retry/IncompleteNativeFrame handling.
            cameras[view_name] = NativeImagePayload(
                encoded=None,
                image=cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB),
            )

        lidar = np.fromfile(expected["lidar"], dtype=np.float32)
        if lidar.size == 0 or lidar.size % 4:
            raise RuntimeError(
                "invalid float32 XYZI payload: {} values in {}".format(
                    lidar.size, expected["lidar"]
                )
            )
        return NativeFramePayload(
            native_sync_frame=native_frame,
            cameras=cameras,
            lidar_points=lidar.reshape((-1, 4)),
        )
