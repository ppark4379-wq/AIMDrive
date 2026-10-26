"""Frame-based control decisions and native-frame capture metadata."""
from copy import deepcopy
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class Decision:
    command: object
    source_sequence: int
    mode: str
    age_frames: int
    warning_threshold_reached: bool
    failsafe_activated: bool
    recovered_from_failsafe: bool


class ControlMailbox:
    """A callback replaces one value; one tick consumes or holds that value."""

    def __init__(self, failsafe, warn_after_frames=5, fail_safe_after_frames=25):
        self.warn, self.limit = int(warn_after_frames), int(fail_safe_after_frames)
        if self.warn < 0:
            raise ValueError("warn_after_frames must be >= 0")
        if self.limit <= 0:
            raise ValueError("fail_safe_after_frames must be > 0")
        if self.warn > self.limit:
            raise ValueError("warn_after_frames must be <= fail_safe_after_frames")
        self.failsafe, self.lock = failsafe, Lock()
        self.pending, self.applied = None, None
        self.sequence = self.consumed = self.age = 0
        self.previous_mode = None

    def update(self, command):
        with self.lock:
            self.pending = deepcopy(command)
            self.sequence += 1
            return self.sequence

    @property
    def source_sequence(self):
        with self.lock:
            return self.sequence

    def select(self):
        with self.lock:
            if self.sequence != self.consumed:
                self.applied = deepcopy(self.pending)
                self.consumed, self.age = self.sequence, 0
                mode = "fresh"
            else:
                self.age += 1
                mode = "failsafe" if self.applied is None or self.age >= self.limit else "reused"
            command = self.failsafe() if mode == "failsafe" else deepcopy(self.applied)
            result = Decision(command, self.consumed, mode, self.age,
                              self.applied is not None and self.warn > 0 and self.age == self.warn,
                              mode == "failsafe" and self.previous_mode != "failsafe",
                              mode == "fresh" and self.previous_mode == "failsafe")
            self.previous_mode = mode
            return result


class CaptureSchedule:
    def __init__(self, enabled, stride):
        self.enabled, self.stride, self.anchor = bool(enabled), int(stride), None
        if self.stride <= 0:
            raise ValueError("capture stride must be > 0")

    def observe_bootstrap_response(self, frame):
        if self.anchor is None:
            self.anchor = int(frame)

    def should_capture(self, frame):
        return bool(self.enabled and self.anchor is not None
                    and (int(frame)-self.anchor) % self.stride == 0)


def ego_snapshot(status):
    vectors = {name:{axis:float(getattr(getattr(status,name,None),axis,0.))
                     for axis in ("x","y","z")}
               for name in ("position","velocity","acceleration","angular_velocity")}
    return {"unique_id":int(getattr(status,"unique_id",0)), **vectors,
            **{name:float(getattr(status,name,0.)) for name in
               ("heading","accel","brake","front_steer_angle","rear_steer_angle")}}


def control_snapshot(command):
    return {"longlCmdType":int(getattr(command,"longlCmdType",1)),
            **{name:float(getattr(command,name,0.)) for name in
               ("accel","brake","front_steer","rear_steer","velocity","acceleration")}}


def capture_record(session, scenario, stride, previous_ego, step, decision, patterns):
    """A capture belongs to request F, using Ego F, with response/Ego F+1."""
    return dict(schema_version=1, session_id=session, scenario_id=scenario,
                native_sync_frame=step.request_frame, request_frame=step.request_frame,
                response_frame=step.response_frame, capture_requested=True, capture_stride=stride,
                ego_state_for_frame=ego_snapshot(previous_ego), ego_state_source_frame=step.request_frame,
                next_ego_state=ego_snapshot(step.vehicle_status), next_ego_state_source_frame=step.response_frame,
                applied_control=control_snapshot(decision.command), control_source_sequence=decision.source_sequence,
                control_reused=decision.mode == "reused", control_age_frames=decision.age_frames,
                control_mode=decision.mode,
                expected_sensor_files={str(name):str(pattern).format(frame=int(step.request_frame))
                                       for name,pattern in patterns.items()})

