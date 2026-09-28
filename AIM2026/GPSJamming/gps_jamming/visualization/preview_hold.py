"""Short display-only retention of a complete image/boxes/cloud composition."""
class PreviewHold:
    def __init__(self, hold_ms=350):
        if not 0 <= hold_ms <= 1000:
            raise ValueError('hold_ms must be in [0, 1000]')
        self.duration_ns = int(hold_ms * 1e6)
        self.saved = None

    def clear(self):
        self.saved = None

    def update(self, canvas, key, now_ns, expires_ns, metadata):
        # Rendering the same measurement again must not extend its lifetime.
        if self.saved is None or self.saved[0] != key:
            self.saved = (key, canvas.copy(), now_ns, expires_ns, dict(metadata))

    def get(self, now_ns):
        if self.saved is None or self.duration_ns == 0:
            return None
        _, canvas, saved_ns, expires_ns, metadata = self.saved
        if not 0 <= now_ns - saved_ns <= self.duration_ns or now_ns > expires_ns:
            return None
        return canvas.copy(), {**metadata, 'held_ms': (now_ns - saved_ns) / 1e6}
