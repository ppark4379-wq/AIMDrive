"""Mirror a child's existing log without placing its output behind a pipe.

The child keeps writing directly to its durable log file. A daemon reader can
therefore be slow or lose its terminal without blocking control or cleanup.
"""

import codecs
from pathlib import Path
import re
import sys
import threading


_HELPER_SETUP_PREFIXES = (
    "[KCityRunner] loading yaml: ",
    "[KCityRunner] MORAI gRPC SDK: ",
    "[KCityRunner] external AIMDrive expert: ",
    "[gRPC] connected to ",
)
_ERROR_WORD = re.compile(r"traceback|exception|error|warning|failed|failure|refused|blocked", re.IGNORECASE)


def show_helper_console_line(line):
    """Hide only repeated setup successes; always retain diagnostic lines."""
    return bool(_ERROR_WORD.search(line) or not line.startswith(_HELPER_SETUP_PREFIXES))


class FileConsoleRelay:
    """Tail only this invocation's appended bytes, with bounded shutdown.

    Console errors are stored in ``error`` and never propagate into the child
    monitor. An unavailable or blocked terminal must not prevent Ego shutdown.
    Neither the source file nor any of its bytes are changed by this reader.
    """

    def __init__(self, path, *, start_offset=0, console=None, line_filter=None,
                 poll_interval=.05):
        if start_offset < 0 or not 0 < poll_interval <= 1:
            raise ValueError("Log relay offset and polling interval are invalid")
        self.path = Path(path)
        self.start_offset = start_offset
        self.console = sys.stdout if console is None else console
        self.line_filter = line_filter
        self.poll_interval = poll_interval
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("A log relay may only be started once")
        self._thread = threading.Thread(target=self._run, name="scenario-log-relay", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout=1.):
        """Drain a finished log, or detach safely from a blocked console."""
        if not 0 <= timeout <= 2:
            raise ValueError("Log relay shutdown must be bounded to two seconds")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True

    def _emit(self, line):
        if self.line_filter is None or self.line_filter(line):
            self.console.write(line)
            self.console.flush()

    def _run(self):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending = ""
        final_drain = False
        try:
            with self.path.open("rb") as source:
                source.seek(self.start_offset)
                while True:
                    chunk = source.read(65536)
                    if chunk:
                        pending += decoder.decode(chunk)
                        while "\n" in pending:
                            line, pending = pending.split("\n", 1)
                            self._emit(line + "\n")
                        # Bound memory for programs emitting an enormous line.
                        if len(pending) >= 65536:
                            self._emit(pending)
                            pending = ""
                        continue
                    if self._stop.is_set():
                        # The EOF above may precede the parent's final write.
                        # Once stop is observed, re-read before accepting EOF.
                        if not final_drain:
                            final_drain = True
                            continue
                        pending += decoder.decode(b"", final=True)
                        if pending:
                            self._emit(pending)
                        return
                    self._stop.wait(self.poll_interval)
        except Exception as error:
            self.error = str(error) or type(error).__name__
