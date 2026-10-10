"""Captured-source controls for topology, with source-drift rejection."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
import yaml

from MoraiDrive.data_collection.mgeo_navigation import load_official_route
from MoraiDrive.data_collection.situation_labeling.derived_topology import apply_derived_roundabout_topology

COLLECTION = Path(__file__).resolve().parents[1]
SCENARIO = COLLECTION.parent / "scenario_runner"
CONFIG = COLLECTION / "situation_labeling/config/derived_roundabout_topology.yaml"
MAP = SCENARIO / "data/R_KR_PR_K-city_2025"
ROUTE = Path(__file__).resolve().parent / "fixtures/roundabout_captured_route.yaml"


def captured_route():
    route = load_official_route(str(ROUTE), str(MAP / "2026_molit_comp_global_path.txt"))
    route["map_root"] = str(MAP)
    route["zones"] = yaml.safe_load(ROUTE.read_text())["full_loop"]["zones"]
    return route


def test_captured_expert_constant_restores_anchor_without_changing_source():
    route = captured_route()
    before = deepcopy(route)
    original_sha = hashlib.sha256(Path(route["route_definition_source"]).read_bytes()).hexdigest()
    result, receipt = apply_derived_roundabout_topology(route, CONFIG)
    assert route == before
    assert "merge_point_xy" not in route["zones"]["roundabout"]
    assert result["nav_waypoints"] == before["nav_waypoints"]
    assert result["zones"]["roundabout"]["merge_point_xy"] == [-93.1035536401546, 327.685278548199]
    assert receipt["anchor_path_distance_m"] < .001
    assert receipt["anchor_ring_distance_m"] < .4
    assert receipt["physical_semantic_certified"] is False
    assert hashlib.sha256(Path(route["route_definition_source"]).read_bytes()).hexdigest() == original_sha


@pytest.mark.parametrize("key", ["route_definition", "global_path", "link_set", "node_set"])
def test_rejects_different_capture_sources(tmp_path, key):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["source_sha256"][key] = "0" * 64
    modified = tmp_path / "config.yaml"
    modified.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="source changed"):
        apply_derived_roundabout_topology(captured_route(), modified)


def test_rejects_modified_archived_parameter_text(tmp_path):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["parameter_source"]["text"] += "\nROUNDABOUT_ENTRY_MERGE_POINT = (0, 0)\n"
    modified = tmp_path / "config.yaml"
    modified.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="snapshot changed"):
        apply_derived_roundabout_topology(captured_route(), modified)


def test_rejects_conflicting_existing_anchor():
    route = captured_route()
    route["zones"]["roundabout"]["merge_point_xy"] = [0, 0]
    with pytest.raises(ValueError, match="conflicting"):
        apply_derived_roundabout_topology(route, CONFIG)
