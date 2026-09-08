"""Incremental reader for Sync Runtime's append-only JSONL manifest."""

import json
import os
import time


class SyncCaptureManifest:
    def __init__(self, path: str, poll_interval_sec: float = 0.05):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.poll_interval_sec = max(0.005, float(poll_interval_sec))
        self._stream = None
        self._buffer = ""

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def poll(self):
        if self._stream is None:
            if not os.path.isfile(self.path):
                return []
            self._stream = open(self.path, "r", encoding="utf-8")

        chunk = self._stream.read()
        if not chunk:
            return []
        self._buffer += chunk
        lines = self._buffer.split("\n")
        self._buffer = lines.pop()
        records = []
        for line in lines:
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("capture_requested", False):
                continue
            native_frame = int(record["native_sync_frame"])
            if int(record["request_frame"]) != native_frame:
                raise ValueError(
                    "Manifest request/native frame mismatch: {}".format(record)
                )
            if int(record["response_frame"]) != native_frame + 1:
                raise ValueError(
                    "Manifest response frame is not native frame + 1: {}".format(
                        record
                    )
                )
            if int(record["ego_state_source_frame"]) != native_frame:
                raise ValueError("Manifest Ego/source frame mismatch: {}".format(record))
            records.append(record)
        return records

    def wait_for_records(self, stop_at_monotonic=None):
        while stop_at_monotonic is None or time.monotonic() < stop_at_monotonic:
            records = self.poll()
            if records:
                return records
            time.sleep(self.poll_interval_sec)
        return []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
