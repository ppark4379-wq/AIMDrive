"""Export whole-tunnel vehicle behavior plans without connecting to MORAI."""

import argparse
import json
import yaml
from pathlib import Path

from .generate_tunnel_dynamic_candidates import (
    SCENARIO_RUNNER_ROOT,
    load_catalog_inputs,
)
from ..scenarios.full_loop_parts.tunnel_vehicle_catalog import build_behavior_catalog
from ..utils.map_loader import MGeoMapLoader


def build_catalog(scenario_runner_root=SCENARIO_RUNNER_ROOT):
    metadata, points, zone = load_catalog_inputs(scenario_runner_root)
    cfg = yaml.safe_load(
        (scenario_runner_root / "config/full_loop.yaml").read_text(encoding="utf-8-sig")
    )["scenarios"]["full_loop"]
    loader = MGeoMapLoader(str(scenario_runner_root / "data/R_KR_PR_K-city_2025"))
    return {
        **metadata,
        **build_behavior_catalog(
            loader,
            points,
            metadata["tunnel_entry_s_m"],
            metadata["tunnel_exit_s_m"],
            zone,
            cfg["tunnel_vehicle_dynamic"],
        ),
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
