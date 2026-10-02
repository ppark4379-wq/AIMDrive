"""Persisted rule inputs must reproduce the same instruction decisions."""
from MoraiDrive.data_collection.situation_labeling.feature_definitions import (
    project_feature_rows, validate_feature_rows,
)
from MoraiDrive.data_collection.situation_labeling.instruction_features import enrich
from MoraiDrive.data_collection.situation_labeling.mapper import attach_rule_instructions
from MoraiDrive.data_collection.situation_labeling.rules import evaluate_frames


def _decisions(frames, objects):
    rules = evaluate_frames(frames, objects)
    attach_rule_instructions(rules)
    return [(row["status"], row["instruction_state"], row.get("label_candidate"),
             row.get("instruction_id"), row.get("instruction")) for row in rules]


def test_dynamic_projection_keeps_rule_decisions_without_diagnostic_features():
    frames = [
        {"index": i, "lidar_source_ns": 1_000_000_000 + i * 100_000_000,
         "zone_id": "dynamic_obstacle", "valid": True, "invalid_reasons": [],
         "object_inventory_usable": True, "ego_world_x_m": float(i),
         "ego_world_y_m": 0.0, "ego_path_s_m": float(i), "ego_speed_mps": 1.0}
        for i in range(2)
    ]
    people = [
        {"index": i, "actor_id": "ped", "kind": "pedestrian",
         "valid": True, "invalid_reasons": [],
         "world_x_m": 8.0 + i, "world_y_m": d, "forward_m": 8.0,
         "path_d_m": d, "delta_s_m": 8.0,
         "actor_width_m": 0.6, "actor_length_m": 0.6,
         "v_d_mps": 0.0, "speed_mps": 0.0}
        for i, d in enumerate((0.2, 3.0))
    ]
    enrich(frames, people, {"nav_waypoints": [(0, 0), (100, 0)]})
    before = _decisions(frames, people)
    persisted_frames, persisted_objects = project_feature_rows(frames, people)
    validate_feature_rows(persisted_frames, persisted_objects)
    assert "pedestrian_clearance_m" not in persisted_frames[0]
    assert "ego_decelerating" not in persisted_frames[0]
    assert "pedestrian_clearance_m" not in persisted_objects[0]
    assert "forward_m" not in persisted_objects[0]
    assert "delta_s_m" not in persisted_objects[0]
    assert "dynamic_in_yield_family" in persisted_frames[0]
    assert _decisions(persisted_frames, persisted_objects) == before


def test_highway_projection_keeps_lane_change_instruction():
    frame = {
        "index": 0, "lidar_source_ns": 1, "zone_id": "highway",
        "valid": True, "invalid_reasons": [], "object_inventory_usable": True,
        "ego_world_x_m": 0.0, "ego_world_y_m": 0.0, "ego_speed_mps": 5.0,
        "ego_stopped": False, "ego_decelerating": True, "ego_hard_braking": False,
        "lane_change_active": True, "lane_change_direction": "left",
        "lane_change_aborting": False, "highway_nearby_geometry_unverified": False,
        "target_lane_collision_safe": None, "target_lane_initially_safe": True,
    }
    before = _decisions([frame], [])
    persisted_frames, persisted_objects = project_feature_rows([frame], [])
    validate_feature_rows(persisted_frames, persisted_objects)
    assert "target_lane_collision_safe" not in persisted_frames[0]
    assert "target_lane_initially_safe" not in persisted_frames[0]
    assert _decisions(persisted_frames, persisted_objects) == before