"""Explicit source-to-Unix mapping; receipt time is never used as an anchor."""


class TimestampMapper:
    def __init__(self, ns_per_tick=None, source_anchor=None, unix_anchor_ns=None,
                 max_forward_jump_ns=5_000_000_000):
        if ns_per_tick is not None and (not isinstance(ns_per_tick, int) or ns_per_tick <= 0):
            raise ValueError("ns_per_tick must be a positive integer or null")
        if (source_anchor is None) != (unix_anchor_ns is None):
            raise ValueError("Both clock anchors must be supplied together")
        if max_forward_jump_ns <= 0:
            raise ValueError("max_forward_jump_ns must be positive")
        self.scale = ns_per_tick
        self.source_anchor = source_anchor
        self.unix_anchor = unix_anchor_ns
        self.max_jump = max_forward_jump_ns
        self.last_raw = None
        self.epoch = 0

    def map(self, raw):
        jump = self.last_raw is not None and (
            raw < self.last_raw or
            (self.scale is not None and (raw - self.last_raw) * self.scale > self.max_jump))
        self.last_raw = raw
        if jump:
            self.epoch += 1
            self.source_anchor = self.unix_anchor = None
        if self.scale is None or self.source_anchor is None:
            return None, "clock_jump" if jump else "unmapped", self.epoch
        return (self.unix_anchor + (raw - self.source_anchor) * self.scale,
                "mapped", self.epoch)
