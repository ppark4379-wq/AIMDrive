"""Non-blocking latest-value mailbox for Expert control commands."""

import copy
import threading
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ControlDecision:
    """Command selected for one MORAI sync frame."""

    command: Any
    source_sequence: int
    mode: str
    age_frames: int
    warning_threshold_reached: bool = False
    failsafe_activated: bool = False
    recovered_from_failsafe: bool = False


class LatestControlBuffer:
    """Keep only the newest command and never block the tick loop.

    Freshness is measured only by callback sequence and applied sync frames.
    ROS timestamps and wall-clock arrival times are deliberately not used.
    """

    def __init__(
        self,
        failsafe_factory: Callable[[], Any],
        warn_after_frames: int = 5,
        fail_safe_after_frames: int = 25,
    ):
        self._failsafe_factory = failsafe_factory
        self.warn_after_frames = int(warn_after_frames)
        self.fail_safe_after_frames = int(fail_safe_after_frames)
        if self.warn_after_frames < 0:
            raise ValueError("warn_after_frames must be >= 0")
        if self.fail_safe_after_frames <= 0:
            raise ValueError("fail_safe_after_frames must be > 0")
        if self.warn_after_frames > self.fail_safe_after_frames:
            raise ValueError(
                "warn_after_frames must be <= fail_safe_after_frames"
            )

        self._lock = threading.Lock()
        self._latest_command = None
        self._latest_sequence = 0
        self._last_applied_sequence = 0
        self._last_expert_command = None
        self._control_age_frames = 0
        self._last_mode = None

    def update(self, command: Any) -> int:
        """Replace the mailbox value and return its local source sequence."""
        with self._lock:
            self._latest_command = copy.deepcopy(command)
            self._latest_sequence += 1
            return self._latest_sequence

    @property
    def source_sequence(self) -> int:
        with self._lock:
            return self._latest_sequence

    def select(self) -> ControlDecision:
        """Select immediately: fresh, previous-command hold, or fail-safe."""
        with self._lock:
            previous_mode = self._last_mode

            if self._latest_sequence > self._last_applied_sequence:
                command = copy.deepcopy(self._latest_command)
                self._last_expert_command = copy.deepcopy(command)
                self._last_applied_sequence = self._latest_sequence
                self._control_age_frames = 0
                mode = "fresh"
            elif self._last_expert_command is None:
                # Waiting for the first Expert command must still be safe and
                # must not stop simulation progression.
                self._control_age_frames += 1
                command = self._failsafe_factory()
                mode = "failsafe"
            else:
                self._control_age_frames += 1
                if self._control_age_frames >= self.fail_safe_after_frames:
                    command = self._failsafe_factory()
                    mode = "failsafe"
                else:
                    command = copy.deepcopy(self._last_expert_command)
                    mode = "reused"

            warning_threshold_reached = (
                self._last_expert_command is not None
                and self.warn_after_frames > 0
                and self._control_age_frames == self.warn_after_frames
            )
            failsafe_activated = (
                mode == "failsafe" and previous_mode != "failsafe"
            )
            recovered_from_failsafe = (
                mode == "fresh" and previous_mode == "failsafe"
            )
            self._last_mode = mode

            return ControlDecision(
                command=command,
                source_sequence=self._last_applied_sequence,
                mode=mode,
                age_frames=self._control_age_frames,
                warning_threshold_reached=warning_threshold_reached,
                failsafe_activated=failsafe_activated,
                recovered_from_failsafe=recovered_from_failsafe,
            )

