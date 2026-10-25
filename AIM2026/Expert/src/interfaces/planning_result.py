"""Planning-to-control contract, retaining the original path/result objects."""
from dataclasses import dataclass, field


@dataclass
class DrivingPlan:
    path: object
    target_velocity: float
    section_result: object
    pedestrian_released: bool = False
    force_brake: bool = False
    log_context: dict = field(default_factory=dict)
