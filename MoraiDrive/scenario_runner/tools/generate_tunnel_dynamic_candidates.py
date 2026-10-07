"""Print tunnel pedestrian candidate paths without starting MORAI."""

import argparse
import json
from pathlib import Path

import yaml

from ..scenarios.full_loop_parts.tunnel_dynamic_catalog import generate_tunnel_candidates
from ..utils.transform_utils import polyline_length


SCENARIO_RUNNER_ROOT = Path(__file__).resolve().parents[1]


def load_catalog_inputs(scenario_runner_root=SCENARIO_RUNNER_ROOT):
    config = yaml.safe_load((scenario_runner_root / "config/full_loop.yaml")
                            .read_text(encoding="utf-8-sig"))["scenarios"]["full_loop"]
    route = yaml.safe_load((scenario_runner_root / config["route_definition_path"])
                           .read_text(encoding="utf-8-sig"))[config["route_definition_key"]]
    points = []
    for line in (scenario_runner_root / config["global_path"]).read_text(
            encoding="utf-8-sig").splitlines():
        fields = line.strip().split()
        if fields and not fields[0].startswith("#"):
            points.append(tuple(float(value) for value in fields[:3]))
    zone = route["zones"]["tunnel"]
    checkpoints = {str(item["id"]): int(item["path_index"])
                   for item in route["checkpoints"]}
    entry_index = checkpoints[str(zone["entry_checkpoint"])]
    exit_index = checkpoints[str(zone["exit_checkpoint"])]
    if not 0 <= entry_index < exit_index < len(points):
        raise ValueError("Tunnel checkpoint indices are invalid")
    entry_s = polyline_length(points[:entry_index + 1])
    exit_s = polyline_length(points[:exit_index + 1])
    metadata = {
        "route": str(config["global_path"]),
        "tunnel_entry_s_m": entry_s,
        "tunnel_exit_s_m": exit_s,
    }
    return metadata, points, zone


def build_catalog(scenario_runner_root=SCENARIO_RUNNER_ROOT):
    metadata, points, zone = load_catalog_inputs(scenario_runner_root)
    return {
        **metadata,
        "candidates": generate_tunnel_candidates(
            points, metadata["tunnel_entry_s_m"], metadata["tunnel_exit_s_m"],
            zone["candidate_catalog"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON to this file")
    args = parser.parse_args()
    content = json.dumps(build_catalog(), indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(args.output)
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
