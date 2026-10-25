"""Small, controller-neutral JSONL bridge for a separate driving process.

Host monotonic nanoseconds are comparable only for processes on the same host.
Records are never used to command the vehicle.
"""

import json
import hashlib
import os
from pathlib import Path
import platform
import time


class PipelineWriter:
    def __init__(self, path, run_id, code_paths=()):
        self.path = Path(path)
        self.run_id = str(run_id)
        self.code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.controller_code_sha256 = {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                       for path in code_paths if Path(path).is_file()}
        self.sequence = 0
        self.clock_host = platform.system().lower()
        self.error = None
        self.stream = self.path.open("w", encoding="utf-8", buffering=65536)

    def write(self, stage, **fields):
        if self.error is not None:
            return False
        row = dict(schema="aimdrive_pipeline_v1", run_id=self.run_id,
                   sequence=self.sequence, stage=stage,
                   logger_code_sha256=self.code_sha256,
                   controller_code_sha256=self.controller_code_sha256,
                   timestamp_ns=time.monotonic_ns(), clock_domain="host_monotonic_ns",
                   clock_host=self.clock_host, wall_time_ns=time.time_ns(),
                   **fields)
        try:
            self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            if stage == "control":
                self.stream.flush()
            self.sequence += 1
            return True
        except (OSError, TypeError, ValueError) as error:
            self.error = str(error)
            self._report_error()
            return False

    def _report_error(self):
        try:
            self.path.with_name("pipeline_error.json").write_text(
                json.dumps({"run_id": self.run_id, "error": self.error}), encoding="utf-8")
        except OSError:
            pass

    def close(self):
        try:
            self.stream.flush()
            os.fsync(self.stream.fileno())
        except OSError as error:
            self.error = str(error)
            self._report_error()
        finally:
            try:
                self.stream.close()
            except OSError as error:
                self.error = str(error)
                self._report_error()


def read_window(path, run_id, until_ns, window_sec=30.0, *, until_wall_ns=None):
    """Read complete rows from a live append-only file; report partial tails."""
    path = Path(path)
    if not path.is_file():
        return [], "missing"
    try:
        data = path.read_bytes()
    except OSError as error:
        return [], str(error)
    end = data.rfind(b"\n") + 1
    issue = "truncated final line" if end != len(data) else None
    error_file = path.with_name("pipeline_error.json")
    if error_file.is_file():
        try:
            recorded = json.loads(error_file.read_text(encoding="utf-8"))
            if recorded.get("run_id") == run_id:
                issue = "writer: " + str(recorded.get("error"))
        except (OSError, ValueError, TypeError):
            issue = "writer error status unreadable"
    rows = []
    floor = until_ns - int(window_sec * 1e9)
    wall_floor = (until_wall_ns - int(window_sec * 1e9)
                  if type(until_wall_ns) is int else None)
    cross_host = False
    newest_wall = None
    for number, line in enumerate(data[:end].splitlines(), 1):
        try:
            row = json.loads(line)
            stamp = row["timestamp_ns"]
            if (row.get("schema") != "aimdrive_pipeline_v1" or
                    row.get("run_id") != run_id or
                    row.get("clock_domain") != "host_monotonic_ns" or
                    type(stamp) is not int):
                raise ValueError("identity or clock mismatch")
            remote = row.get("clock_host") not in (None, platform.system().lower())
            if remote:
                cross_host = True
                wall = row.get("wall_time_ns")
                if wall_floor is None or type(wall) is not int:
                    raise ValueError("cross-host window requires wall timestamps")
                newest_wall = wall if newest_wall is None else max(newest_wall, wall)
                in_window = wall_floor <= wall <= until_wall_ns
            else:
                in_window = floor <= stamp <= until_ns
            if in_window:
                rows.append(row)
        except (ValueError, KeyError, TypeError) as error:
            issue = f"line {number}: {error}"
    if cross_host:
        # File modification time is supplied by the shared filesystem. Reject
        # cross-host alignment unless it agrees with the producer's wall clock.
        file_wall_ns = path.stat().st_mtime_ns
        if (newest_wall is None or
                abs(file_wall_ns - newest_wall) > 2_000_000_000 or
                abs(file_wall_ns - until_wall_ns) > 2_000_000_000):
            return [], "cross-host wall clock not verified against file mtime"
        issue = issue or "cross-host wall window; monotonic clocks not aligned"
    return rows, issue


def read_ve_audit_window(path, run_id, generation, until_ns, window_sec=30.0):
    """Preserve publisher-side timing only when identity and host clock agree."""
    path = Path(path)
    if not path.is_file():
        return [], "missing"
    try:
        data = path.read_bytes()
    except OSError as error:
        return [], str(error)
    end = data.rfind(b"\n") + 1
    issue = "truncated final line" if end != len(data) else None
    rows = []
    floor = until_ns - int(window_sec * 1e9)
    for number, line in enumerate(data[:end].splitlines(), 1):
        try:
            row = json.loads(line)
            stamp = next((row[key] for key in ("sent_ns", "monotonic_ns", "begin_ns")
                          if type(row.get(key)) is int), None)
            if (row.get("run_id") != run_id or row.get("generation") != generation or
                    row.get("clock_domain") != "host_monotonic_ns" or stamp is None):
                raise ValueError("identity or clock mismatch")
            if floor <= stamp <= until_ns:
                rows.append(row)
        except (ValueError, KeyError, TypeError) as error:
            issue = f"line {number}: {error}"
    return rows, issue
