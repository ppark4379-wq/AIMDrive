"""Append frame metadata off the control thread, with bounded backpressure."""
from copy import deepcopy
import json
from pathlib import Path
from queue import Queue
from threading import Thread


class ManifestWriter:
    def __init__(self, path, queue_size=1024):
        self.path = str(Path(path).absolute())
        self.queue, self.sentinel = Queue(maxsize=max(1,int(queue_size))), object()
        self.closed, self.error = False, None
        self.submitted = self.written = self.dropped = 0
        Path(self.path).parent.mkdir(parents=True,exist_ok=True)
        self.thread = Thread(target=self._write,name="scenario-capture-manifest",daemon=True)
        self.thread.start()

    def submit(self, record):
        if self.closed or self.error is not None:
            self.dropped += 1
            return False
        from queue import Full
        try:
            self.queue.put_nowait(deepcopy(record))
        except Full:
            self.dropped += 1
            return False
        self.submitted += 1
        return True

    def _write(self):
        try:
            with open(self.path,"a",encoding="utf-8",buffering=1) as output:
                while True:
                    row = self.queue.get()
                    try:
                        if row is self.sentinel:
                            return
                        output.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
                        self.written += 1
                    finally:
                        self.queue.task_done()
        except BaseException as error:
            self.error = error

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.error is None:
            self.queue.put(self.sentinel)
            self.thread.join()
        if self.error is not None:
            raise RuntimeError("Capture journal writer failed") from self.error

