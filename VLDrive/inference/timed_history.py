"""Keep a bounded dense buffer; sample the trained two-second window by time."""
from bisect import bisect_left
from collections import deque
import math
import time

import torch

from inference.vldrive_runtime import TokenHistory


class TimedTokenHistory(TokenHistory):
    def __init__(self, interval_s=.2, tolerance_s=.06, *, capture_hz=10., reset_gap_s=1.):
        super().__init__(interval_s, tolerance_s)
        if not math.isfinite(capture_hz) or not 1 / interval_s <= capture_hz <= 20:
            raise ValueError('capture_hz must be between history Hz and 20')
        if not math.isfinite(reset_gap_s) or reset_gap_s <= interval_s + tolerance_s:
            raise ValueError('reset_gap_s must exceed one history interval plus tolerance')
        if tolerance_s >= interval_s / 2:
            raise ValueError('history tolerance must be below half the trained interval')
        self.history_interval = self.interval
        self.history_tolerance = self.tolerance
        # Capture cadence is independent of the trained temporal spacing.
        self.interval = 1. / capture_hz
        self.tolerance = min(tolerance_s, self.interval * .3)
        self.reset_gap_s = reset_gap_s
        self.frames = deque(maxlen=math.ceil((10 * interval_s + reset_gap_s + tolerance_s)
                                            / (self.interval - self.tolerance)) + 2)

    def append(self, feature, *, stamp, ids, vision_ms=0., context=None):
        if feature.shape != (106, 256) or not bool(torch.isfinite(feature).all()):
            raise ValueError('History expects one finite [106,256] real feature')
        if not math.isfinite(stamp) or not self.accepts(stamp, ids):
            return False
        with self.lock:
            if self.frames and stamp - self.frames[-1]['stamp'] > self.reset_gap_s:
                self.frames.clear()
                self.resets += 1
            owned = feature.detach().clone()
            ready = None
            if owned.is_cuda:
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(owned.device))
            self.frames.append(dict(feature=owned, ready=ready, stamp=float(stamp),
                                    ids=dict(ids), vision_ms=float(vision_ms),
                                    context=context, generation=self.resets))
        return True

    def snapshot(self, *, materialize=True):
        with self.lock:
            frames = list(self.frames)
        if not frames:
            return None, []
        stamps = [row['stamp'] for row in frames]
        anchor = stamps[-1]
        selected, upper = [], len(frames)
        for offset in range(11):
            target = anchor - offset * self.history_interval
            insertion = bisect_left(stamps, target, 0, upper)
            candidates = [i for i in (insertion - 1, insertion) if 0 <= i < upper]
            if not candidates:
                break
            chosen = min(candidates, key=lambda i: abs(stamps[i] - target))
            if abs(stamps[chosen] - target) > self.history_tolerance + 1e-8:
                break
            selected.append(frames[chosen])
            upper = chosen  # Never duplicate a frame or reverse time.
        rows = selected[::-1]
        if len(rows) != 11 or not materialize:
            return None, rows
        return self.materialize(rows), rows

    def expire(self, *, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.frames and now - self.frames[-1]['stamp'] > self.reset_gap_s:
                self.frames.clear()
                self.resets += 1
