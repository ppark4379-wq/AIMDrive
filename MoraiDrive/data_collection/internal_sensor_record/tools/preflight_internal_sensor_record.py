#!/usr/bin/env python3
"""Fail closed when MORAI build or AIM sensor-set contract has changed."""
import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from async_capture.internal_sensor_record_contract import (
    DEFAULT_ASSEMBLY, DEFAULT_SENSOR_SET, capture_preflight,
)
from async_capture.journal import write_json


DEFAULT_EVIDENCE = ROOT / "evidence/internal_sensor_record_contract_v1.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, default=DEFAULT_ASSEMBLY)
    parser.add_argument("--sensor-set", type=Path, default=DEFAULT_SENSOR_SET)
    parser.add_argument("--static-evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = capture_preflight(args.assembly, args.sensor_set, args.static_evidence)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
