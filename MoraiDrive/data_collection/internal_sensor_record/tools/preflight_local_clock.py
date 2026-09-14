#!/usr/bin/env python3
"""Fail before a campaign if WSL receive-monotonic time is grossly unstable."""

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from async_capture.clock_health import probe


def summarized(report):
    return {
        "passed": bool(report["passed"]),
        "failures": list(report["failures"]),
        "rate_error_ppm": float(report["rate_error_ppm"]),
        "wall_step_count": len(report.get("wall_steps", [])),
    }


def wait_for_stable_clock(wait_seconds=90.0, probe_seconds=2.0,
                          retry_seconds=3.0):
    """Allow normal WSL startup clock discipline to settle before failing."""
    started = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    attempts = []
    last_notice = -10.0
    while True:
        report = probe(seconds=probe_seconds)
        elapsed = time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - started
        attempts.append(dict(elapsed_seconds=elapsed, **summarized(report)))
        if report["passed"]:
            report["stabilization"] = {
                "wait_limit_seconds": wait_seconds,
                "elapsed_seconds": elapsed,
                "attempts": attempts,
            }
            return report
        if elapsed >= wait_seconds:
            report["stabilization"] = {
                "wait_limit_seconds": wait_seconds,
                "elapsed_seconds": elapsed,
                "attempts": attempts,
            }
            return report
        if elapsed - last_notice >= 10.0:
            print(
                "[WAIT] WSL 시작 직후 시계 보정이 진행 중입니다: "
                "rate error %.1f ppm, %.0f/%.0fs" % (
                    float(report["rate_error_ppm"]), elapsed, wait_seconds
                ),
                file=sys.stderr,
                flush=True,
            )
            last_notice = elapsed
        remaining = max(0.0, wait_seconds - elapsed)
        time.sleep(min(retry_seconds, remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wait-seconds", type=float, default=90.0)
    parser.add_argument("--probe-seconds", type=float, default=2.0)
    parser.add_argument("--retry-seconds", type=float, default=3.0)
    args = parser.parse_args()
    if not 0.0 <= args.wait_seconds <= 600.0:
        parser.error("--wait-seconds must be in 0..600")
    if not 0.1 <= args.probe_seconds <= 10.0:
        parser.error("--probe-seconds must be in 0.1..10")
    if not 0.1 <= args.retry_seconds <= 30.0:
        parser.error("--retry-seconds must be in 0.1..30")
    report = wait_for_stable_clock(
        wait_seconds=args.wait_seconds,
        probe_seconds=args.probe_seconds,
        retry_seconds=args.retry_seconds,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if report["passed"]:
        print(
            "[PASS] WSL receive clock 검사: rate error %.3f ppm "
            "(%.1fs, %d회 측정)" % (
                float(report["rate_error_ppm"]),
                float(report["stabilization"]["elapsed_seconds"]),
                len(report["stabilization"]["attempts"]),
            )
        )
        return
    print("[FAIL] WSL 시계가 수집용 receive-monotonic 기준으로 불안정합니다.",
          file=sys.stderr)
    print("       failures=%s, rate_error=%.1f ppm" %
          (report["failures"], float(report["rate_error_ppm"])), file=sys.stderr)
    print("       %.0f초 동안 자동 재측정했지만 안정화되지 않았습니다." %
          float(args.wait_seconds), file=sys.stderr)
    print("       Windows PowerShell에서 `wsl --shutdown`을 한 번 실행한 뒤 WSL과 01번을 다시 시작하세요.",
          file=sys.stderr)
    print("       재시작 후에는 WSL IP가 바뀔 수 있습니다. 01번에 새로 표시된 IP:9090으로",
          file=sys.stderr)
    print("       MORAI 차량 정보·Simulator Network(/clock)·센서를 다시 연결해야 합니다.",
          file=sys.stderr)
    print("       시스템 시간 서비스를 이 스크립트가 임의로 변경하지는 않습니다.",
          file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
