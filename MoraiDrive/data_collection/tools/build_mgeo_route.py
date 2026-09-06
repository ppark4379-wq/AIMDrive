#!/usr/bin/env python3
"""Build a directed MGeo route from ordered competition checkpoints."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mgeo_navigation import MGeoMap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--link-set", required=True)
    parser.add_argument("--route-spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.route_spec, encoding="utf-8") as f:
        spec = json.load(f)
    route = MGeoMap(args.link_set).build_checkpoint_route(
        spec["checkpoints"], close_loop=bool(spec.get("close_loop"))
    )
    output = dict(spec)
    output.update(route)
    output["map_path"] = os.path.abspath(os.path.dirname(args.link_set))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"route: {output['route_id']}")
    print(f"checkpoints: {len(output['checkpoint_matches'])}")
    print(f"links: {len(output['nav_link_ids'])}")
    print(f"waypoints: {len(output['nav_waypoints'])}")
    print(f"written: {args.output}")
    for match in output["checkpoint_matches"]:
        print(f"  {match['name']:>9} -> {match['link_id']} "
              f"distance={match['distance_m']:.3f}m "
              f"heading_error={match['heading_error_deg']:.2f}deg")


if __name__ == "__main__":
    main()
