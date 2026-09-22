"""Short-lived read-only file handles; incomplete tail bytes are retried."""
import json
import math
import os
from pathlib import Path


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


class JsonlTail:
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.rows = []
        self.invalid_lines = 0
        self.identity = None
        self.prefix = b''
        self.mtime = None
        self.error = None
        self.generation = 0

    def poll(self):
        self.error = None
        try:
            with self.path.open('rb') as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                head = stream.read(min(stat.st_size, 128))
                replaced = self.identity is not None and (identity != self.identity or stat.st_size < self.offset
                    or (self.prefix and not head.startswith(self.prefix)))
                if replaced:
                    self.offset = 0; self.rows.clear(); self.invalid_lines = 0; self.generation += 1
                self.identity, self.prefix, self.mtime = identity, head, stat.st_mtime
                stream.seek(self.offset)
                # Bound each refresh during the initial scan of a very large run.
                content = stream.read(8 * 1024 * 1024)
            boundary = content.rfind(b'\n')
            if boundary < 0:
                return 0
            self.offset += boundary + 1
            added = 0
            for line in content[:boundary+1].splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or type(row.get('update')) is not int or row['update'] < 0:
                        raise ValueError('not an update record')
                    self.rows.append(row); added += 1
                except (ValueError, UnicodeError):
                    self.invalid_lines += 1
            return added
        except FileNotFoundError:
            self.error = 'Waiting for training.jsonl'
        except OSError as exc:
            self.error = str(exc)
        return 0

    def read_all(self):
        while True:
            before = self.offset
            self.poll()
            if self.offset == before: break
        return self.rows


class RunMetadata:
    """Read run sidecars only when changed; never open checkpoint files."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.cache = {}

    def read(self, name):
        path = self.directory / name
        try:
            stat = path.stat(); stamp = (stat.st_mtime_ns, stat.st_size)
            if name not in self.cache or self.cache[name][0] != stamp:
                self.cache[name] = (stamp, json.loads(path.read_text(encoding='utf-8-sig')))
            return self.cache[name][1]
        except (OSError, ValueError):
            return self.cache.get(name, (None, {}))[1]

    def context(self):
        settings = self.read('resolved_config.json')
        settings = settings.get('base', settings)
        indices = self.read('window_indices.json').get('train')
        epochs = number(settings.get('epochs'))
        per_epoch = len(indices) if isinstance(indices, list) and indices else None
        total = number(self.read('report.json').get('schedule_updates'))
        if total is None and epochs and per_epoch:
            total = epochs * per_epoch
        model = settings.get('model', {})
        description = self.read('model.json')
        return dict(total_updates=total, total_epochs=epochs, updates_per_epoch=per_epoch,
                    target_retention=number(model.get('target_retention')),
                    num_visual_tokens=number(description.get('connector', {}).get('num_visual_tokens')),
                    system=self.read('provenance.json').get('backend', {}))
