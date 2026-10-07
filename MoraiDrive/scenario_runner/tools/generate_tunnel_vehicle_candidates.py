"""Export four slow-vehicle trial candidates without connecting to MORAI."""
import argparse
import json
from pathlib import Path

from .generate_tunnel_dynamic_candidates import SCENARIO_RUNNER_ROOT, load_catalog_inputs
from ..scenarios.full_loop_parts.tunnel_vehicle_catalog import generate_tunnel_vehicle_candidates


def build_catalog(scenario_runner_root=SCENARIO_RUNNER_ROOT):
    metadata, points, zone = load_catalog_inputs(scenario_runner_root)
    return {
        **metadata,
        "candidates": generate_tunnel_vehicle_candidates(
            points, metadata["tunnel_entry_s_m"], metadata["tunnel_exit_s_m"],
            zone["candidate_catalog"], zone["vehicle_candidate_catalog"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    content = json.dumps(build_catalog(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(args.output)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
