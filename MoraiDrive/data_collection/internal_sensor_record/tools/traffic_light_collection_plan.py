#!/usr/bin/env python3
"""Create, check and snapshot balanced traffic-light collection plans."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic_light_collection_plan import (create_plan_file, distribution, load_plan,
                                          plan_runs, seal_plan)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--repeats", type=int, default=3)
    check = subparsers.add_parser("check")
    check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--laps", type=int, required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--plan", type=Path, required=True)
    seal.add_argument("--laps", type=int, required=True)
    seal.add_argument("--output", type=Path, required=True)
    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--plan", type=Path, required=True)
    resolve.add_argument("--lap", type=int, required=True)
    resolve.add_argument("--field", choices=("weather", "time_hour", "json"), required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            plan = create_plan_file(args.output, args.repeats)
            path = args.output
        elif args.command == "seal":
            sealed = seal_plan(args.plan, args.output, args.laps)
            print(json.dumps(dict(status="PASS", output=str(args.output.resolve()),
                                  laps=len(sealed["plan"]["runs"]),
                                  sha256=sealed["sha256"], bytes=sealed["bytes"])))
            return 0
        else:
            plan = load_plan(args.plan, args.laps if args.command == "check" else None)
            path = args.plan
        if args.command == "resolve":
            row = plan_runs(plan).get(args.lap)
            if row is None:
                raise ValueError("Requested lap is not present in the plan")
            print(json.dumps(row, ensure_ascii=False, sort_keys=True)
                  if args.field == "json" else row[args.field])
        else:
            print(json.dumps(dict(status="PASS", plan=str(path.resolve()),
                                  laps=len(plan["runs"]), distribution=distribution(plan),
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                             ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError) as error:
        print("[FAIL] {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
