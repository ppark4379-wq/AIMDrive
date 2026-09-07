#!/usr/bin/env python3
"""Validate one run; raw files are never altered. Legacy inspection is explicit."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from aimdrive.data.stage6e_validation import validate_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--require-features", action="store_true")
    parser.add_argument("--allow-legacy-offline", action="store_true")
    parser.add_argument("--output", help="New report file inside this repository")
    args = parser.parse_args()
    run = Path(args.run).expanduser().resolve()
    if ROOT not in run.parents:
        parser.error("Run must be inside this repository")
    report = validate_run(run, require_features=args.require_features, allow_legacy=args.allow_legacy_offline)
    if args.output:
        output = Path(args.output).resolve()
        if ROOT not in output.parents:
            parser.error("Output must be inside this repository")
        with output.open("x") as handle:
            json.dump(report, handle, indent=2)
    print("FAIL" if report["quality_status"]=="rejected" else "PASS", report["quality_status"])
    print(json.dumps({k: report[k] for k in ("reasons", "warnings", "frames", "ready_frames", "feature_frames")}, indent=2))
    return 2 if report["quality_status"]=="rejected" else 0


if __name__ == "__main__":
    raise SystemExit(main())
