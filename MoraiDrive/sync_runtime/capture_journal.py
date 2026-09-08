"""Non-blocking append-only capture manifest journal."""

import json
import os
import queue
import threading
from typing import Optional


class CaptureJournal:
    """Write JSONL on a worker; submit() never waits for disk or queue space."""

    def __init__(self, path: str, queue_size: int = 1024):
        self.path = os.path.abspath(path)
        self._queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop = object()
        self._error: Optional[BaseException] = None
        self._closed = False
        self.submitted = 0
        self.written = 0
        self.dropped = 0
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._thread = threading.Thread(
            target=self._run, name="sync-capture-journal", daemon=True
        )
        self._thread.start()

    def submit(self, record) -> bool:
        if self._closed or self._error is not None:
            self.dropped += 1
            return False
        try:
            self._queue.put_nowait(record.to_dict())
        except queue.Full:
            self.dropped += 1
            return False
        self.submitted += 1
        return True

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._error is not None:
            raise RuntimeError("Capture journal writer failed") from self._error
        # Shutdown may wait; the tick critical path never calls close().
        self._queue.put(self._stop)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("Capture journal writer failed") from self._error

    def _run(self):
        try:
            with open(self.path, "a", encoding="utf-8", buffering=1) as output:
                while True:
                    item = self._queue.get()
                    try:
                        if item is self._stop:
                            return
                        output.write(
                            json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        )
                        output.flush()
                        self.written += 1
                    finally:
                        self._queue.task_done()
        except BaseException as error:
            self._error = error
