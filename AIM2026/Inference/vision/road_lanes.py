"""Approved highway map geometry; no NPC configuration or actor state.

The merge-adjacent chain follows the straight mainline, not the NPC exit route. These map-link chains are
independent of traffic population, spawn positions, vehicle routes and speeds.
"""
import json
from pathlib import Path

LANES_PATH = Path(__file__).resolve().parents[2] / 'Expert/src/config/highway_lanes.json'


def map_lanes(links):
    definitions = json.loads(LANES_PATH.read_text(encoding='utf-8'))
    return {name: [point for identity in route for point in links[identity]['points']]
            for name, route in definitions.items()}
