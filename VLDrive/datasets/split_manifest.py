"""Pure-Python helpers for deterministic MORAI vision-encoder splits.

This module deliberately has no torch/numpy dependency so a collected dataset
can be audited and split before the training environment is installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SCHEMA_VERSION = "morai-vision-split-v1"
DEFAULT_FUTURE_OFFSETS = (5, 10, 15, 20, 25)


def _range_dict(start: int, stop: int) -> Dict[str, int]:
    """Return an inclusive range description, allowing an empty range."""
    return {"start": int(start), "stop": int(stop)}


def expand_range(spec: Dict[str, int]) -> List[int]:
    start = int(spec["start"])
    stop = int(spec["stop"])
    if stop < start:
        return []
    return list(range(start, stop + 1))


def read_dataset_index(dataset_root: Path) -> List[Tuple[str, int]]:
    dataset_root = Path(dataset_root).resolve()
    index_path = dataset_root / "dataset_index.txt"
    if not index_path.is_file():
        raise FileNotFoundError("dataset_index.txt not found: {}".format(index_path))

    entries: List[Tuple[str, int]] = []
    for line_number, raw in enumerate(index_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError("Invalid dataset_index line {}: {!r}".format(line_number, raw))
        relative, count_text = parts
        relative = relative.replace("\\", "/").strip("/")
        if relative.startswith("data/"):
            relative = relative[len("data/") :]
        if any(part.startswith("_invalid") for part in Path(relative).parts):
            continue
        count = int(count_text)
        route = dataset_root / "data" / relative
        if not route.is_dir():
            raise FileNotFoundError("Indexed route does not exist: {}".format(route))
        entries.append((relative, count))

    if not entries:
        raise ValueError("No usable routes in {}".format(index_path))
    return entries


def build_chronological_split(
    dataset_root: Path,
    *,
    train_fraction: float = 0.8,
    purge_frames: int = 25,
    future_offsets: Sequence[int] = DEFAULT_FUTURE_OFFSETS,
) -> Dict[str, object]:
    """Build front-train / purged-boundary / rear-validation ranges.

    The last ``max(future_offsets)`` frames cannot be anchors because the
    expert future trajectory would be incomplete.  The purge is expressed in
    anchor indices; training targets may end inside the purge but can never
    reach a validation input frame.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if purge_frames < max(future_offsets):
        raise ValueError("purge_frames must be at least the largest future offset")

    routes = []
    for relative, frame_count in read_dataset_index(Path(dataset_root)):
        max_future = max(int(x) for x in future_offsets)
        anchor_count = frame_count - max_future
        available_after_purge = anchor_count - purge_frames
        if available_after_purge < 2:
            raise ValueError("Route is too short for train/purge/validation: {}".format(relative))
        train_count = int(available_after_purge * train_fraction)
        train_count = min(max(train_count, 1), available_after_purge - 1)
        purge_start = train_count
        purge_stop = purge_start + purge_frames - 1
        val_start = purge_stop + 1
        val_stop = anchor_count - 1
        routes.append(
            {
                "path": relative,
                "frame_count": frame_count,
                "last_anchor": anchor_count - 1,
                "train": _range_dict(0, train_count - 1),
                "purge": _range_dict(purge_start, purge_stop),
                "validation": _range_dict(val_start, val_stop),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "policy": "front_train_purge_rear_validation",
        "train_fraction": float(train_fraction),
        "purge_frames": int(purge_frames),
        "future_offsets": [int(x) for x in future_offsets],
        "routes": routes,
    }


def load_split_manifest(path: Path) -> Dict[str, object]:
    path = Path(path).resolve()
    from utils.config.config_files import read_document, document_bytes
    manifest = read_document(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported MORAI split schema: {}".format(manifest.get("schema_version")))
    offsets = [int(x) for x in manifest.get("future_offsets", [])]
    if offsets != list(DEFAULT_FUTURE_OFFSETS):
        raise ValueError("Expected future offsets {}, got {}".format(DEFAULT_FUTURE_OFFSETS, offsets))
    if int(manifest.get("purge_frames", 0)) < max(offsets):
        raise ValueError("Split purge is shorter than the future horizon")
    return manifest


def split_entries(manifest: Dict[str, object], split: str) -> Iterable[Tuple[str, int]]:
    key = "validation" if split in ("val", "validation") else split
    if key not in ("train", "validation"):
        raise ValueError("split must be train or validation")
    for route in manifest["routes"]:
        for frame in expand_range(route[key]):
            yield str(route["path"]), frame

