"""Durable completion receipts for multi-file native collection frames."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from .atomic_io import write_json, atomic_path

COMMIT_DIR = ".frame_commits"
SESSION_FILE = ".collection_session.json"
FRAME_DIRS = ("rgb_front", "rgb_left", "rgb_right", "rgb_rear", "lidar",
              "measurements", "measurements_full", "actors_data", "affordances",
              "3d_bbs", "gnss", "imu")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def commit_frame(root, index, measurement, paths):
    root = Path(root).resolve()
    files = {}
    for path in paths:
        path = Path(path).resolve()
        relative = str(path.relative_to(root))
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError("Missing/empty frame payload: {}".format(path))
        files[relative] = {"size": path.stat().st_size, "sha256": digest(path)}
    _validate_inventory(files, index)
    receipt = {"schema_version": 1, "sample_index": index,
               "native_sync_frame": measurement["native_sync_frame"],
               "session_id": measurement["runtime_session_id"], "files": files}
    write_json(str(root / COMMIT_DIR / ("{:04d}.json".format(index))), receipt)
    return receipt


def _validate_inventory(names, index):
    sample = "{:04d}".format(index)
    required = ["lidar/"+sample+".npy", "measurements/"+sample+".json",
                "measurements_full/"+sample+".json", "actors_data/"+sample+".json",
                "affordances/"+sample+".npy", "3d_bbs/"+sample+".npy"]
    if any(name not in names for name in required) or any(
        not any(Path(name).parent.as_posix() == "rgb_"+view and Path(name).stem == sample
                for name in names) for view in ("front", "left", "right", "rear")
    ):
        raise RuntimeError("Incomplete file inventory for sample {}".format(sample))


def read_commits(root):
    root = Path(root).resolve()
    records = []
    for file in sorted((p for p in (root / COMMIT_DIR).glob("*.json") if p.stem.isdigit()),
                       key=lambda p: int(p.stem)):
        with file.open(encoding="utf-8") as stream:
            receipt = json.load(stream)
        if int(file.stem) != len(records) or receipt["sample_index"] != len(records):
            raise RuntimeError("Non-contiguous frame completion receipts: {}".format(file))
        names = receipt["files"]
        _validate_inventory(names, len(records))
        for name, expected in names.items():
            path = (root / name).resolve()
            if root not in path.parents:
                raise RuntimeError("Frame receipt path escapes route")
            if (not path.is_file() or path.stat().st_size != expected["size"]
                    or digest(path) != expected["sha256"]):
                raise RuntimeError("Committed frame payload is missing or changed: {}".format(path))
        if records and receipt["native_sync_frame"] <= records[-1]["native_sync_frame"]:
            raise RuntimeError("Non-increasing native frames in completion receipts")
        records.append(receipt)
    return records


def quarantine_uncommitted(root, count):
    """Retain crash leftovers outside training folders; never delete user data."""
    root = Path(root).resolve()
    destination = root / ".incomplete_frames" / uuid.uuid4().hex
    for name in FRAME_DIRS:
        directory = root / name
        if not directory.is_dir():
            continue
        for file in directory.iterdir():
            if file.is_file() and (file.name.startswith(".pending-") or
                                   (file.stem.isdigit() and int(file.stem) >= count)):
                if root not in file.resolve().parents:
                    raise RuntimeError("Incomplete payload path escapes route")
                target = destination / name / file.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file), str(target))


def rewrite_provenance(root, receipts):
    with atomic_path(str(Path(root) / "capture_provenance.jsonl")) as temporary:
        with open(temporary, "w", encoding="utf-8") as stream:
            for row in receipts:
                stream.write(json.dumps({"sample_index": "{:04d}".format(row["sample_index"]),
                                         "native_sync_frame": row["native_sync_frame"],
                                         "session_id": row["session_id"]}) + "\n")


def resume_state(root, manifest_path):
    root = Path(root).resolve()
    metadata = root / SESSION_FILE
    if not metadata.exists():
        return None
    session = json.loads(metadata.read_text(encoding="utf-8"))
    if session["manifest_path"] != os.path.abspath(manifest_path):
        raise RuntimeError("Collection session manifest mismatch")
    if not (root / COMMIT_DIR).is_dir():
        raise RuntimeError("Collection completion directory is missing")
    receipts = read_commits(root)
    if any(row["session_id"] != session["runtime_session_id"] for row in receipts):
        raise RuntimeError("Completion receipt session mismatch")
    quarantine_uncommitted(root, len(receipts))
    rewrite_provenance(root, receipts)
    return {"schema_version": 2, "manifest_path": session["manifest_path"],
            "runtime_session_id": session["runtime_session_id"], "output_dir": str(root),
            "last_consumed_native_frame": receipts[-1]["native_sync_frame"] if receipts else -1,
            "next_sample_index": len(receipts)}
