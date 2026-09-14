#!/usr/bin/env python3
"""Validate the configured Windows spool before ROS or MORAI is mutated."""

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reserve-gib", type=float, default=0.0,
        help="Free-space reserve required after starting this operation",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.reserve_gib <= 1024:
        parser.error("--reserve-gib must be in 0..1024")

    from ssd_storage import SSD_ROOT, ensure_storage_layout, require_ssd

    layout = ensure_storage_layout()
    health = require_ssd(
        SSD_ROOT,
        check_volume=True,
        reserve_bytes=int(args.reserve_gib * 1024**3),
    )
    report = {
        "ready": True,
        "storage_root": str(SSD_ROOT),
        "dataset_view": layout["dataset_view"],
        "dataset_view_kind": layout["dataset_view_kind"],
        "windows_root": health["windows_root"],
        "free_bytes": health["free_bytes"],
        "reserve_bytes": health["reserve_bytes"],
        "mount": health["mount"],
        "volume": health["volume"],
        "tracked_source_modified": False,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print("[PASS] 로컬 Sensor Record 저장소")
        print("       Windows 경로 : %s" % report["windows_root"])
        print("       레포에서 보기: %s (%s)" % (
            report["dataset_view"], report["dataset_view_kind"]
        ))
        print("       남은 공간    : %.1f GiB (보존 기준 %.1f GiB)" % (
            report["free_bytes"] / float(1024**3), args.reserve_gib
        ))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print("[FAIL] 로컬 Sensor Record 저장소 검사 실패: " + str(error),
              file=sys.stderr)
        print("       00번에 --storage-root /mnt/<드라이브>/<폴더>를 지정해 다시 준비하세요.",
              file=sys.stderr)
        if os.environ.get("MORAI_STORAGE_DEBUG") == "1":
            raise
        raise SystemExit(2)
