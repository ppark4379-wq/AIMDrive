"""Frame-exact metadata emitted for MORAI synchronous sensor captures."""

from dataclasses import asdict, dataclass
from typing import Any, Dict


class NativeFrameCaptureScheduler:
    """Sparse native-frame schedule initialized by the bootstrap response."""

    def __init__(self, enabled: bool, stride: int):
        self.enabled = bool(enabled)
        self.stride = int(stride)
        if self.stride <= 0:
            raise ValueError("capture stride must be > 0")
        self.anchor_frame = None

    def observe_bootstrap_response(self, response_frame: int):
        if self.anchor_frame is None:
            self.anchor_frame = int(response_frame)

    def should_capture(self, request_frame: int) -> bool:
        return bool(
            self.enabled
            and self.anchor_frame is not None
            and (int(request_frame) - self.anchor_frame) % self.stride == 0
        )


def _vector3(value: Any) -> Dict[str, float]:
    return {
        "x": float(getattr(value, "x", 0.0)),
        "y": float(getattr(value, "y", 0.0)),
        "z": float(getattr(value, "z", 0.0)),
    }


def ego_state_dict(status: Any) -> Dict[str, Any]:
    """Serialize only stable EgoVehicleStatus fields needed by the dataset."""
    return {
        "unique_id": int(getattr(status, "unique_id", 0)),
        "position": _vector3(getattr(status, "position", None)),
        "velocity": _vector3(getattr(status, "velocity", None)),
        "acceleration": _vector3(getattr(status, "acceleration", None)),
        "angular_velocity": _vector3(
            getattr(status, "angular_velocity", None)
        ),
        "heading": float(getattr(status, "heading", 0.0)),
        "accel": float(getattr(status, "accel", 0.0)),
        "brake": float(getattr(status, "brake", 0.0)),
        "front_steer_angle": float(
            getattr(status, "front_steer_angle", 0.0)
        ),
        "rear_steer_angle": float(
            getattr(status, "rear_steer_angle", 0.0)
        ),
    }


def control_dict(command: Any) -> Dict[str, Any]:
    return {
        "longlCmdType": int(getattr(command, "longlCmdType", 1)),
        "accel": float(getattr(command, "accel", 0.0)),
        "brake": float(getattr(command, "brake", 0.0)),
        "front_steer": float(getattr(command, "front_steer", 0.0)),
        "rear_steer": float(getattr(command, "rear_steer", 0.0)),
        "velocity": float(getattr(command, "velocity", 0.0)),
        "acceleration": float(getattr(command, "acceleration", 0.0)),
    }


@dataclass(frozen=True)
class CaptureFrameRecord:
    schema_version: int
    session_id: str
    scenario_id: str
    native_sync_frame: int
    request_frame: int
    response_frame: int
    capture_requested: bool
    capture_stride: int
    ego_state_for_frame: Dict[str, Any]
    ego_state_source_frame: int
    next_ego_state: Dict[str, Any]
    next_ego_state_source_frame: int
    applied_control: Dict[str, Any]
    control_source_sequence: int
    control_reused: bool
    control_age_frames: int
    control_mode: str
    expected_sensor_files: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
