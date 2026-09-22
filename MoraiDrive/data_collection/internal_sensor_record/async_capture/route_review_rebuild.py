"""Preserve old route-derived evidence before explicitly rebuilding it."""
import json
import time
from pathlib import Path
from async_capture.journal import write_json
from async_capture.route_sources import resolve_route_sources

ROUTE_DERIVED = ("route_signal_audit_v1", "review_closeout", "admission_v1")


def archive_route_review(run):
    run = Path(run).resolve()
    episode = run/"episode_000"
    if episode.is_symlink() or not episode.is_dir():
        raise ValueError("Expected a real episode directory")
    if (run/"ADMISSION_RECOVERY.json").exists():
        raise ValueError("Existing admission recovery needs explicit reconciliation before route rebuild")
    contract = json.loads((run/"sensor_contract.json").read_text(encoding="utf-8"))
    sources = resolve_route_sources(episode, contract)
    paths = [episode/name for name in ROUTE_DERIVED if (episode/name).exists()]
    for path in paths:
        if path.is_symlink() or not path.is_dir() or path.resolve().parent != episode:
            raise ValueError("Unsafe derived-output path: " + str(path))
    if not paths:
        return None
    history = episode/"route_review_history"
    if history.is_symlink():
        raise ValueError("History must not be a link")
    history.mkdir(exist_ok=True)
    archive = history/str(time.time_ns())
    archive.mkdir()
    record = dict(schema="morai_route_review_rebuild_v1", status="archiving",
                  archived=[], source_sha256=sources["hashes"],
                  raw_modified=False, native_modified=False)
    write_json(archive/"PLAN.json", record)
    for path in paths:
        path.rename(archive/path.name)
        record["archived"].append(path.name)
        write_json(archive/("MOVED_"+path.name+".json"), dict(record))
    record["status"] = "archived_for_rebuild"
    write_json(archive/"REBUILD.json", record)
    return archive
