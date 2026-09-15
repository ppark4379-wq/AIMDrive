"""Fail-closed barriers between consecutive native Sensor Record laps."""

import json
from pathlib import Path
import time


def prepare_coordination_directory(coordination, episode):
    """Create the fresh per-episode handshake directory before UI ownership."""
    episode = Path(episode).resolve()
    coordination = Path(coordination).resolve()
    if coordination.parent != episode:
        raise ValueError("Sensor Record coordination must be inside its episode")
    coordination.mkdir(parents=False, exist_ok=False)
    return coordination


def require_stop_acknowledgement(coordination):
    path = Path(coordination) / "recording_stopped.json"
    if not path.is_file():
        raise RuntimeError("Sensor Record stop acknowledgement is missing")
    try:
        acknowledgement = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "Sensor Record stop acknowledgement is invalid: " + repr(error)
        ) from error
    if acknowledgement.get("error"):
        raise RuntimeError(
            "Sensor Record stop acknowledgement reported an error: "
            + str(acknowledgement["error"])
        )
    if acknowledgement.get("status") != "stopped":
        raise RuntimeError("Sensor Record stop acknowledgement has invalid status")
    if acknowledgement.get("record_on") is not False:
        raise RuntimeError(
            "Sensor Record stop acknowledgement does not prove the UI is OFF"
        )
    if "stopwatch_ticks" not in acknowledgement:
        raise RuntimeError("Sensor Record stop acknowledgement lacks timing evidence")
    return acknowledgement


def native_metadata_snapshot(root):
    root = Path(root)
    entries = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        entries.append((
            path.relative_to(root).as_posix(),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        ))
    return tuple(sorted(entries))


def wait_for_native_quiescence(
    root,
    *,
    stable_seconds=3.0,
    timeout_seconds=120.0,
    poll_seconds=0.5,
    snapshot=native_metadata_snapshot,
    monotonic=time.monotonic,
    sleep=time.sleep,
):
    """Require a non-empty native tree whose sizes/timestamps stop changing."""
    if stable_seconds < 0 or timeout_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("Invalid native quiescence timing")
    started = monotonic()
    deadline = started + timeout_seconds
    previous = None
    stable_since = None
    observations = 0
    while True:
        current = snapshot(Path(root))
        now = monotonic()
        observations += 1
        if current and current == previous:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= stable_seconds:
                return {
                    "schema": "morai_native_storage_quiescence_v1",
                    "passed": True,
                    "file_count": len(current),
                    "total_bytes": sum(row[1] for row in current),
                    "stable_seconds": now - stable_since,
                    "elapsed_seconds": now - started,
                    "observations": observations,
                }
        else:
            previous = current
            stable_since = now if current else None
        if now >= deadline:
            raise TimeoutError(
                "Native Sensor Record files did not become stable within "
                f"{timeout_seconds:.1f}s (files={len(current)})"
            )
        sleep(poll_seconds)
