"""Current competition sensor constraints and phase-specific transport policy.

The later sensor announcement fixes competition-time sensor ingress to UDP.
The team has confirmed that ROS1 may be used during training/data collection.

Training/data collection: ROS1 or UDP
Competition inference:   UDP

No UDP packet byte layout is assumed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

TRAINING_PHASE = "training"
DEPLOYMENT_PHASE = "deployment"
TRANSPORT_ROS1 = "ROS1"
TRANSPORT_UDP = "UDP"


@dataclass(frozen=True)
class TransportPolicy:
    training_allowed: Tuple[str, ...]
    deployment_required: str

    def validate(self) -> None:
        if not self.training_allowed:
            raise ValueError("training_allowed cannot be empty")
        if self.deployment_required not in self.training_allowed:
            raise ValueError(
                "deployment transport should also be available for parity tests"
            )

    def is_allowed(self, phase: str, transport: str) -> bool:
        if phase == TRAINING_PHASE:
            return transport in self.training_allowed
        if phase == DEPLOYMENT_PHASE:
            return transport == self.deployment_required
        raise ValueError(f"unknown phase: {phase!r}")

    def require(self, phase: str, transport: str) -> None:
        if not self.is_allowed(phase, transport):
            raise ValueError(
                f"transport {transport!r} is not allowed in phase {phase!r}"
            )


@dataclass(frozen=True)
class RateSpec:
    max_hz: float
    recommended_hz: Optional[float] = None

    def validate(self) -> None:
        if self.max_hz <= 0:
            raise ValueError("max_hz must be positive")
        if self.recommended_hz is not None:
            if not 0 < self.recommended_hz <= self.max_hz:
                raise ValueError("recommended_hz must be in (0,max_hz]")


@dataclass(frozen=True)
class CameraSpec:
    name: str
    position_m: Tuple[float, float, float]
    rpy_deg: Tuple[float, float, float]
    max_resolution: Tuple[int, int]
    fov_deg: float
    fixed: bool = True

    def validate(self) -> None:
        if len(self.position_m) != 3 or len(self.rpy_deg) != 3:
            raise ValueError("camera position/rpy must each have 3 values")
        if min(self.max_resolution) <= 0:
            raise ValueError("camera resolution must be positive")
        if not 0 < self.fov_deg < 180:
            raise ValueError("camera FOV must be in (0,180)")

    @property
    def yaw_signed_deg(self) -> float:
        return ((float(self.rpy_deg[2]) + 180.0) % 360.0) - 180.0


@dataclass(frozen=True)
class CompetitionSensorSpec:
    transport_policy: TransportPolicy
    gps_max_count: int
    gps_rate: RateSpec
    imu_max_count: int
    imu_rate: RateSpec
    lidar_max_count: int
    lidar_models: Tuple[str, ...]
    lidar_intensity_type: str
    lidar_rate: RateSpec
    camera_max_count: int
    camera_rate: RateSpec
    fixed_cameras: Tuple[CameraSpec, ...]
    gps_imu_noise_may_be_injected: bool

    @property
    def training_networks(self) -> Tuple[str, ...]:
        return self.transport_policy.training_allowed

    @property
    def deployment_network(self) -> str:
        return self.transport_policy.deployment_required

    def fixed_camera_map(self) -> Dict[str, CameraSpec]:
        return {camera.name: camera for camera in self.fixed_cameras}

    def validate(self) -> None:
        self.transport_policy.validate()
        if self.deployment_network != TRANSPORT_UDP:
            raise ValueError("competition inference sensor ingress must be UDP")
        if self.gps_max_count != 1 or self.imu_max_count != 1:
            raise ValueError("GPS and IMU maximum count must each be 1")
        if self.lidar_max_count != 1 or self.lidar_models != ("VLP16",):
            raise ValueError("current notice allows one VLP16 only")
        if self.camera_max_count != 4 or len(self.fixed_cameras) != 3:
            raise ValueError("current notice allows max 4 cameras with 3 fixed")
        for rate in (self.gps_rate, self.imu_rate, self.lidar_rate, self.camera_rate):
            rate.validate()
        for camera in self.fixed_cameras:
            camera.validate()


CURRENT_SENSOR_SPEC = CompetitionSensorSpec(
    transport_policy=TransportPolicy(
        training_allowed=(TRANSPORT_ROS1, TRANSPORT_UDP),
        deployment_required=TRANSPORT_UDP,
    ),
    gps_max_count=1,
    gps_rate=RateSpec(max_hz=30.0),
    imu_max_count=1,
    imu_rate=RateSpec(max_hz=50.0),
    lidar_max_count=1,
    lidar_models=("VLP16",),
    lidar_intensity_type="Intensity",
    lidar_rate=RateSpec(max_hz=15.0, recommended_hz=10.0),
    camera_max_count=4,
    camera_rate=RateSpec(max_hz=30.0),
    fixed_cameras=(
        CameraSpec(
            name="front",
            position_m=(1.9, 0.0, 1.2),
            rpy_deg=(0.0, 2.0, 0.0),
            max_resolution=(1280, 720),
            fov_deg=90.0,
        ),
        CameraSpec(
            name="left",
            position_m=(1.15, 0.65, 1.2),
            rpy_deg=(0.0, 10.0, 70.0),
            max_resolution=(640, 480),
            fov_deg=130.0,
        ),
        CameraSpec(
            name="right",
            position_m=(1.15, -0.65, 1.2),
            rpy_deg=(0.0, 10.0, 290.0),
            max_resolution=(640, 480),
            fov_deg=130.0,
        ),
    ),
    gps_imu_noise_may_be_injected=True,
)
CURRENT_SENSOR_SPEC.validate()
