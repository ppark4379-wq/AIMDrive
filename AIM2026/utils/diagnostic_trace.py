"""Bounded asynchronous evidence writer; recording failures never control driving."""
import json
from pathlib import Path
import queue
import threading


class DiagnosticTrace:
    def __init__(self, directory, name):
        self.directory, self.name = Path(directory), name
        self.queue = queue.Queue(maxsize=16)
        self.closed = threading.Event()
        self.stats = dict(enqueued=0, written=0, dropped=0, errors=0)
        self.thread = threading.Thread(target=self._run, daemon=True, name=name)
        self.thread.start()

    def submit(self, row, files=()):
        try:
            self.queue.put_nowait((row, files))
            self.stats['enqueued'] += 1
        except queue.Full:
            self.stats['dropped'] += 1

    def _run(self):
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory/(self.name+'.jsonl')).open('w', encoding='utf-8', buffering=1) as out:
                while not self.closed.is_set() or not self.queue.empty():
                    try:
                        row, files = self.queue.get(timeout=.1)
                    except queue.Empty:
                        continue
                    try:
                        for name, value in files:
                            path = self.directory/name
                            if name.endswith('.npy'):
                                import numpy as np
                                np.save(path, value, allow_pickle=False)
                            else:
                                path.write_bytes(value)
                        out.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                        self.stats['written'] += 1
                    except Exception as exc:
                        self.stats['errors'] += 1
                        self.stats['last_error'] = str(exc)
        except Exception as exc:
            self.stats['errors'] += 1
            self.stats['last_error'] = str(exc)

    def close(self):
        self.closed.set()
        self.thread.join(timeout=2)
        self.stats['incomplete_shutdown'] = self.thread.is_alive()
        try:
            (self.directory/(self.name+'_status.json')).write_text(
                json.dumps(self.stats, indent=2), encoding='utf-8')
        except OSError:
            pass
