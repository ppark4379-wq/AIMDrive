#!/usr/bin/env python3
"""Validate full artifacts, expose the training schema and update run registry."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from aimdrive.data.stage6e_runs import append_jsonl
from aimdrive.data.stage6e_schema import publish_training_manifest
from aimdrive.data.stage6e_validation import validate_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--allow-legacy-offline",action="store_true")
    args=parser.parse_args()
    run=Path(args.run).expanduser().resolve()
    if ROOT not in run.parents:
        parser.error("Run must remain inside this checkout")
    quality=validate_run(run,require_features=True,allow_legacy=args.allow_legacy_offline)
    if quality["quality_status"]=="rejected":
        raise SystemExit("FAIL: "+str(quality["reasons"]))
    if not (run/"manifest_training.jsonl").exists():
        publish_training_manifest(run,allow_legacy=args.allow_legacy_offline)
    (run/"quality_report.json").write_text(json.dumps(quality,indent=2))
    meta=json.loads((run/"run_meta.json").read_text())
    meta.update(quality_status=quality["quality_status"], reasons=quality["reasons"], warnings=quality["warnings"],
                status="ready" if quality["quality_status"]=="valid" else "offline_regression_only",
                training_manifest="manifest_training.jsonl", feature_cache="manifest_features.jsonl")
    (run/"run_meta.json").write_text(json.dumps(meta,indent=2))
    if quality["quality_status"]=="valid":
        append_jsonl(run.parent/"runs.jsonl",dict(schema_version="stage6e-registry-v2",run_id=run.name,
                                               event="ready",quality_status="valid",frames=quality["feature_frames"],
                                               manifest=str(Path(run.name)/"manifest_training.jsonl"),wall_ns=time.time_ns()))
    print("PASS",meta["status"],quality["feature_frames"],"frames",run/"manifest_training.jsonl")


if __name__=="__main__":
    main()
