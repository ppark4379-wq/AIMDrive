"""Content fingerprints for inputs, targets and preprocessing metadata."""
import hashlib
import json
from torch.utils.data import Subset
from .validation import artifact_path, read_rows
from lavis.common.morai_runtime import sha256

def dataset_description(dataset):
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    description = base.describe()
    artifacts = {}
    for run in description["runs"]:
        root = base.root / run["run_id"]
        # Also fingerprint configuration/attestation manifests. Raw preprocessing
        # depends on feature-manifest options, not just final sample paths.
        paths = {"run_meta.json", "manifest.jsonl", "manifest_ready.jsonl",
                 "manifest_features.jsonl", "manifest_training.jsonl", "bridge_health.jsonl"}
        for row in read_rows(root / "manifest_training.jsonl"):
            paths.add(row["model_inputs"]["route"]["path"])
            paths.add(row["training_targets"]["future_trajectory"]["path"])
            if base.input_mode == "cached":
                paths.add(row["model_inputs"]["visual_feature"]["path"])
            else:
                paths.add(row["sensor_refs"]["lidar"]["path"])
                paths.update(ref["path"] for ref in row["sensor_refs"]["cameras"].values())
        digest = hashlib.sha256()
        size = 0
        for relative in sorted(paths):
            path = artifact_path(root, relative)
            digest.update(json.dumps([relative, sha256(path)]).encode())
            size += path.stat().st_size
        artifacts[run["run_id"]] = dict(sha256=digest.hexdigest(), files=len(paths), bytes=size)
    return dict(dataset=description, indices=list(dataset.indices) if isinstance(dataset, Subset) else None,
                model_artifacts=artifacts)
