#!/usr/bin/env python3
"""Fail before a campaign if WSL receive-monotonic time is grossly unstable."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from async_capture.clock_health import probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = probe()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if report["passed"]:
        print("[PASS] WSL monotonic clock 검사: rate error %.3f ppm" %
              float(report["rate_error_ppm"]))
        return
    print("[FAIL] WSL 시계가 수집용 receive-monotonic 기준으로 불안정합니다.",
          file=sys.stderr)
    print("       failures=%s, rate_error=%.1f ppm" %
          (report["failures"], float(report["rate_error_ppm"])), file=sys.stderr)
    print("       Windows PowerShell에서 `wsl --shutdown` 후 WSL과 01번을 다시 시작하세요.",
          file=sys.stderr)
    print("       시스템 시간 서비스를 이 스크립트가 임의로 변경하지는 않습니다.",
          file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
