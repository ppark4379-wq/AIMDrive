"""Instruction-bounded LMDrive-style feature memory, independent of MEFA Z."""
from collections import deque
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class FeatureFrame:
    source_time: float
    received_at: float
    sequence: int
    pose: dict
    visual: object
    navigation: dict
    situation: object = None


class InstructionHistory:
    def __init__(self, *, capacity=400, stride=2, max_gap_s=.15):
        if type(capacity) is not int or type(stride) is not int or not 1 <= stride <= capacity:
            raise ValueError('Invalid history capacity/stride')
        self.frames = deque(maxlen=capacity)
        self.stride, self.max_gap = stride, max_gap_s
        self.lock = threading.Lock()
        self.generation = 0
        self.reason = 'initial'
        self.counter = 0

    def reset(self, reason):
        with self.lock:
            self._reset(reason)

    def _reset(self, reason):
        self.frames.clear()
        self.generation += 1
        self.reason = reason
        self.counter = 0

    def append(self, frame):
        with self.lock:
            if self.frames:
                previous = self.frames[-1][1]
                gap = frame.source_time - previous.source_time
                if gap <= 0 or gap > self.max_gap + 1e-7:
                    self._reset('sensor_time_gap_or_reset')
                elif frame.navigation['segment'] != previous.navigation['segment']:
                    self._reset('navigation_segment_changed')
            self.frames.append((self.counter, frame))
            self.counter += 1

    def snapshot(self):
        with self.lock:
            selected = [f for i, f in self.frames if i % self.stride == 0]
            if self.frames and (not selected or selected[-1].sequence != self.frames[-1][1].sequence):
                selected.append(self.frames[-1][1])
            return self.generation, selected


def fit_context(frames, navigation_tokens, situation_lengths, visual_lengths, limit):
    """Drop oldest whole observations, never truncate a sentence or latest frame.

    Bound padded as well as active sequence length because DDIA evaluates the
    rectangular tensor. Zone contributes exactly one token per frame.
    """
    if len(frames) != len(situation_lengths) or len(frames) != len(visual_lengths) or not frames:
        raise ValueError('Invalid context-budget inputs')
    for start in range(len(frames)):
        count = len(frames) - start
        padded = count * (navigation_tokens + 1 + max(situation_lengths[start:]) + max(visual_lengths[start:]))
        if padded <= limit:
            return frames[start:], padded
    raise ValueError('A single observation exceeds the LLM context limit')
