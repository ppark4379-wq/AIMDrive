#!/usr/bin/env python3
"""Build immutable traffic-label sidecar/report under VLDrive, never raw data."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.traffic_labels import build_traffic_label_sidecar, sha256_file, sidecar_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    outputs = [args.output.resolve(), args.report.resolve()]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Sidecar and report must use distinct paths")
    for path in outputs:
        if ROOT not in path.parents:
            raise ValueError("Outputs must stay inside VLDrive; raw data is immutable")
        if path.exists():
            raise FileExistsError("Choose a new versioned output: {}".format(path))
    artifact = build_traffic_label_sidecar(args.dataset_root, args.split_file)
    report = sidecar_summary(artifact, split="train")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    with outputs[0].open("x", encoding="utf-8") as stream:
        json.dump(artifact, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    report["sidecar_sha256"] = sha256_file(outputs[0])
    with outputs[1].open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "newly_masked_frame_ids"}, indent=2))


if __name__ == "__main__":
    main()
