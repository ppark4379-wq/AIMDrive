"""Small, flushed JSON logs for the independent, read-only Situation dashboard."""
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time


class SituationTelemetry:
    def __init__(self, directory, epochs, batches_per_epoch, lr):
        self.directory = Path(directory)
        self.epochs = epochs
        self.total = epochs * batches_per_epoch if batches_per_epoch is not None else None
        self.started = self.previous = time.monotonic()
        self.durations = deque(maxlen=100)
        self.step = 0
        self.last = dict(update=0, epoch=0, total_epochs=epochs, total_updates=self.total,
                         lr=lr, eta_s=None, elapsed_s=0)
        self.stream = (self.directory / 'training.jsonl').open('x', encoding='utf-8')
        self.status('starting')

    def status(self, state, error=None):
        payload = dict(self.last, schema='situation-training-live-v1', state=state,
                       timestamp=datetime.now(timezone.utc).isoformat(), error=error)
        pending = self.directory / 'situation_status.pending'
        pending.write_text(json.dumps(payload, allow_nan=False) + '\n', encoding='utf-8')
        pending.replace(self.directory / 'situation_status.json')

    def record(self, epoch, loss, fields, lr):
        now = time.monotonic()
        self.durations.append(now - self.previous)
        self.previous = now
        self.step += 1
        eta = max(0, self.total - self.step) * statistics.median(self.durations) if self.total else None
        self.last = dict(schema='situation-training-live-v1', update=self.step, epoch=epoch,
                         total_epochs=self.epochs, total_updates=self.total, lr=lr,
                         elapsed_s=now-self.started, eta_s=eta, loss=loss, fields=fields,
                         timestamp=datetime.now(timezone.utc).isoformat(), skipped=not bool(fields))
        self.stream.write(json.dumps(self.last, allow_nan=False) + '\n')
        self.stream.flush()
        # The JSONL heartbeat is every batch; sidecar writes are only for state transitions.

    def close(self):
        self.stream.close()
