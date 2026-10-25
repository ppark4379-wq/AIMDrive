"""Detect collision edges from externally supplied Ego observations.

The caller supplies timestamps in seconds from one consistent clock per run.
Missing collision_objects means unknown, while an empty collection means an
observed collision-free state. Other observation fields are reserved for later
validation rules and are not used to infer collision state.
"""

from dataclasses import dataclass
import math
from typing import Dict, FrozenSet, List, Optional, Tuple


@dataclass(frozen=True)
class EgoObservation:
    run_id: str
    timestamp: float
    position_xyz: Optional[Tuple[float, float, float]] = None
    speed_mps: Optional[float] = None
    collision_objects: Optional[Tuple[str, ...]] = None
    yaw_deg: Optional[float] = None
    time_basis: Optional[str] = None
    clock_domain: Optional[str] = None
    lap_index: Optional[int] = None


@dataclass(frozen=True)
class ValidationEvent:
    timestamp: float
    event_type: str
    run_id: str
    actor_id: str


class EventMonitor:
    def __init__(self):
        self._collisions_by_run: Dict[str, FrozenSet[str]] = {}

    def observe(self, observation: Optional[EgoObservation]) -> List[ValidationEvent]:
        if observation is None:
            return []
        if not isinstance(observation.run_id, str) or not observation.run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        if not math.isfinite(observation.timestamp):
            raise ValueError("timestamp must be finite seconds")
        if observation.collision_objects is None:
            return []

        current = set()
        for actor_id in observation.collision_objects:
            if not isinstance(actor_id, str) or not actor_id.strip():
                raise ValueError("collision actor IDs must be nonempty strings")
            current.add(actor_id)

        previous = self._collisions_by_run.get(observation.run_id, frozenset())
        events = [
            ValidationEvent(observation.timestamp, "collision_cleared", observation.run_id, actor_id)
            for actor_id in sorted(previous - current)
        ]
        events.extend(
            ValidationEvent(observation.timestamp, "collision_started", observation.run_id, actor_id)
            for actor_id in sorted(current - previous)
        )
        self._collisions_by_run[observation.run_id] = frozenset(current)
        return events
