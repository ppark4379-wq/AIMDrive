"""Persist exact gRPC labels before the dataset writer can apply backpressure."""

from dataclasses import asdict
import hashlib
import json
import os
import sqlite3
import threading
import time
import zlib

from ros.sensor_types import GTObject, TrafficLightState
from sources.sync_capture_manifest import SyncCaptureManifest

from .morai_state_reader import GrpcLabelSnapshot, snapshot_matches_ego


def label_spool_path(spool_dir, manifest_path):
    """Return a stable local spool path for one Runtime manifest."""
    manifest_path = os.path.abspath(os.path.expanduser(manifest_path))
    session_name = os.path.basename(os.path.dirname(manifest_path)) or "session"
    safe_name = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in session_name
    )
    digest = hashlib.sha256(manifest_path.encode("utf-8")).hexdigest()[:12]
    return os.path.join(
        os.path.abspath(os.path.expanduser(spool_dir)),
        "{}-{}.sqlite3".format(safe_name, digest),
    )


def _encode_snapshot(snapshot):
    payload = {
        "schema_version": 1,
        "tick": int(snapshot.tick),
        "simulator_timestamp_raw": int(snapshot.simulator_timestamp_raw),
        "ego_x": float(snapshot.ego_x),
        "ego_y": float(snapshot.ego_y),
        "ego_heading_deg": float(snapshot.ego_heading_deg),
        "objects": [asdict(item) for item in snapshot.objects],
        "traffic_lights": [asdict(item) for item in snapshot.traffic_lights],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return zlib.compress(encoded, level=1)


def _decode_snapshot(payload):
    value = json.loads(zlib.decompress(payload).decode("utf-8"))
    return GrpcLabelSnapshot(
        tick=int(value["tick"]),
        simulator_timestamp_raw=int(value["simulator_timestamp_raw"]),
        ego_x=float(value["ego_x"]),
        ego_y=float(value["ego_y"]),
        ego_heading_deg=float(value["ego_heading_deg"]),
        objects=[GTObject(**item) for item in value.get("objects", [])],
        traffic_lights=[
            TrafficLightState(**item) for item in value.get("traffic_lights", [])
        ],
    )


class GrpcLabelStore:
    """Small local SQLite store keyed by the exact native simulator tick."""

    def __init__(self, path):
        self.path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()
        self._database = sqlite3.connect(
            self.path, timeout=30.0, check_same_thread=False
        )
        self._database.execute("PRAGMA journal_mode=WAL")
        self._database.execute("PRAGMA synchronous=NORMAL")
        self._database.execute(
            "CREATE TABLE IF NOT EXISTS labels ("
            "tick INTEGER PRIMARY KEY, payload BLOB NOT NULL)"
        )
        self._database.commit()

    def put(self, snapshot):
        payload = _encode_snapshot(snapshot)
        with self._lock:
            cursor = self._database.execute(
                "INSERT OR IGNORE INTO labels(tick, payload) VALUES (?, ?)",
                (int(snapshot.tick), sqlite3.Binary(payload)),
            )
            self._database.commit()
            return cursor.rowcount > 0

    def get(self, tick):
        with self._lock:
            row = self._database.execute(
                "SELECT payload FROM labels WHERE tick = ?", (int(tick),)
            ).fetchone()
        return _decode_snapshot(row[0]) if row is not None else None

    def contains(self, tick):
        with self._lock:
            return self._database.execute(
                "SELECT 1 FROM labels WHERE tick = ?", (int(tick),)
            ).fetchone() is not None

    def has_any(self):
        with self._lock:
            return self._database.execute(
                "SELECT 1 FROM labels LIMIT 1"
            ).fetchone() is not None

    def close(self):
        with self._lock:
            if self._database is not None:
                self._database.commit()
                self._database.close()
                self._database = None


class GrpcLabelCaptureSpool:
    """Tail capture records and preserve their labels independently of writers."""

    def __init__(self, history, manifest_path, spool_path,
                 manifest_poll_interval_sec=0.05,
                 required_timeout_sec=2.0,
                 ego_match_distance_m=0.5):
        self.history = history
        self.manifest_path = os.path.abspath(os.path.expanduser(manifest_path))
        self.spool_path = os.path.abspath(os.path.expanduser(spool_path))
        self.manifest_poll_interval_sec = max(
            0.005, float(manifest_poll_interval_sec)
        )
        self.required_timeout_sec = max(0.01, float(required_timeout_sec))
        self.ego_match_distance_m = max(0.01, float(ego_match_distance_m))
        self._store = GrpcLabelStore(self.spool_path)
        self._condition = threading.Condition()
        self._unavailable = set()
        self._stop = threading.Event()
        self._thread = None
        self._started = False
        self._error = None

    @property
    def is_aligned(self):
        return self._store.has_any() or self.history.is_aligned

    def start(self):
        if self._started:
            return
        self.history.start()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="morai-grpc-label-spool",
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def _remember(self, snapshot):
        self._store.put(snapshot)
        with self._condition:
            self._condition.notify_all()

    def _mark_unavailable(self, tick):
        with self._condition:
            self._unavailable.add(int(tick))
            self._condition.notify_all()

    def _capture_loop(self):
        manifest = SyncCaptureManifest(
            self.manifest_path,
            poll_interval_sec=self.manifest_poll_interval_sec,
        )
        try:
            while not self._stop.is_set():
                stop_at = time.monotonic() + self.manifest_poll_interval_sec
                records = manifest.wait_for_records(stop_at)
                for record in records:
                    if self._stop.is_set():
                        return
                    tick = int(record["native_sync_frame"])
                    if self._store.contains(tick):
                        continue
                    snapshot = self.history.wait_for_record(
                        record, self.required_timeout_sec
                    )
                    if snapshot is None:
                        self._mark_unavailable(tick)
                    else:
                        self._remember(snapshot)
        except BaseException as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()
            print("[collect/native] ERROR: gRPC label spool stopped: {}".format(error))
        finally:
            manifest.close()

    def _raise_if_failed(self):
        with self._condition:
            error = self._error
        if error is not None:
            raise RuntimeError("gRPC label spool failed") from error

    def _stored_match(self, record):
        snapshot = self._store.get(int(record["native_sync_frame"]))
        if snapshot is None:
            return None
        ego = record["ego_state_for_frame"]
        position = ego["position"]
        if not snapshot_matches_ego(
            snapshot,
            position["x"],
            position["y"],
            ego["heading"],
            max_distance_m=self.ego_match_distance_m,
        ):
            return None
        return snapshot

    def wait_for_record(self, record, timeout_sec):
        tick = int(record["native_sync_frame"])
        deadline = time.monotonic() + max(0.01, float(timeout_sec))
        while time.monotonic() < deadline:
            self._raise_if_failed()
            snapshot = self._stored_match(record)
            if snapshot is not None:
                return snapshot

            # The consumer may reach a fresh frame just before the independent
            # manifest tailer. Capture it here as well, then persist it.
            remaining = deadline - time.monotonic()
            snapshot = self.history.wait_for_record(
                record, min(0.05, max(0.01, remaining))
            )
            if snapshot is not None:
                self._remember(snapshot)
                return snapshot

            with self._condition:
                if tick in self._unavailable:
                    return None
                remaining = deadline - time.monotonic()
                if remaining > 0.0:
                    self._condition.wait(timeout=min(0.05, remaining))
        self._raise_if_failed()
        return self._stored_match(record)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.required_timeout_sec + 0.5))
        self.history.close()
        self._store.close()
