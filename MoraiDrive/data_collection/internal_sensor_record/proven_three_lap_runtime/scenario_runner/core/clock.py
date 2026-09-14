"""Scenario elapsed time; RPC and process deadlines still use host monotonic."""
import time


class HostScenarioClock:
    domain = 'host_monotonic'

    def now(self):
        return time.monotonic()


class NativeScenarioClock:
    domain = 'native_frame'

    def __init__(self, session_id, start_frame, step_ns):
        if not isinstance(session_id, str) or not session_id:
            raise ValueError('Native scenario requires a session identity')
        if type(start_frame) is not int or start_frame < 0 or type(step_ns) is not int or step_ns <= 0:
            raise ValueError('Invalid native clock origin/step')
        self.session_id = session_id
        self.origin_frame = self.frame = start_frame
        self.step_ns = step_ns

    def advance(self, session_id, frame, step_ns):
        if session_id != self.session_id or step_ns != self.step_ns:
            raise ValueError('Scenario clock session/step changed; a new episode is required')
        if type(frame) is not int or frame not in (self.frame, self.frame + 1):
            raise ValueError('Scenario clock reset or skipped a native tick')
        self.frame = frame
        return self.now()

    def now(self):
        return (self.frame - self.origin_frame) * self.step_ns / 1e9

    def snapshot(self):
        return dict(domain=self.domain, session_id=self.session_id, native_frame=self.frame,
                    origin_frame=self.origin_frame, step_ns=self.step_ns,
                    elapsed_ns=(self.frame - self.origin_frame) * self.step_ns)
