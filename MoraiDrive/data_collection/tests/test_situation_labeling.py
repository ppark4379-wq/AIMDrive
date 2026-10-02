"""Tests for route geometry, actor joins, and fail-closed evidence handling."""

import json
import pytest

from MoraiDrive.data_collection.situation_labeling.build_labels import (
    _match_actor, _zone_intervals, build_feature_tables, build_features)
from MoraiDrive.data_collection.situation_labeling.features import RouteProjector, numeric_features


def test_straight_route_signs_and_velocity():
    route = RouteProjector([(0, 0), (100, 0)])
    actor = {"position": [10, 2, 0], "velocity_kmh": [7.2, 0, 0],
             "size_wlh_m": [2, 4, 1.5]}
    geometry = {"center_forward_m": 10, "center_left_m": 2}
    feature = numeric_features(actor, geometry, route)
    assert feature["path_s_m"] == pytest.approx(10)
    assert feature["path_d_m"] == pytest.approx(2)
    assert feature["path_segment_index"] == 0
    assert feature["actor_width_m"] == pytest.approx(2)
    assert feature["actor_length_m"] == pytest.approx(4)
    assert feature["v_s_mps"] == pytest.approx(2)
    assert feature["v_d_mps"] == pytest.approx(0)
    actor["position"][1] = -2
    actor["velocity_kmh"] = [0, 7.2, 0]
    feature = numeric_features(actor, geometry, route)
    assert feature["path_d_m"] == pytest.approx(-2)
    assert feature["v_s_mps"] == pytest.approx(0)
    assert feature["v_d_mps"] == pytest.approx(2)


def test_curved_route_uses_local_tangent_and_cumulative_distance():
    route = RouteProjector([(0, 0), (10, 0), (10, 10)])
    actor = {"position": [8, 5, 0], "velocity_kmh": [0, 3.6, 0],
             "size_wlh_m": [2, 4, 1.5]}
    geometry = {"center_forward_m": 4, "center_left_m": 1}
    feature = numeric_features(actor, geometry, route)
    assert feature["path_segment_index"] == 1
    assert feature["path_s_m"] == pytest.approx(15)
    assert feature["path_d_m"] == pytest.approx(2)
    assert feature["v_s_mps"] == pytest.approx(1)
    assert feature["v_d_mps"] == pytest.approx(0)


def _write_frames(episode, folder, rows):
    path = episode / folder
    path.mkdir()
    (path / "frames.json").write_text(json.dumps(rows), encoding="utf-8")


def test_invalid_admission_and_ambiguous_actor_remain_visible(tmp_path):
    actor = {"id": "1", "instance_key": "ped:1", "kind": "pedestrian",
             "position": [10, 2, 0], "velocity_kmh": [0, 3.6, 0],
             "size_wlh_m": [2, 4, 1.5]}
    geometry = {"actor_id": "1", "instance_key": "ped:1", "kind": "pedestrian",
                "center_forward_m": 10, "center_left_m": 2,
                "training_bev_relevant": True, "api_box_geometry_candidate": True,
                "reasons": []}
    _write_frames(tmp_path, "gt_timeline_v1", [{"index": 0, "lidar_source_ns": 10, "ego_xy": [5, 0],
        "object_gt": {"objects": [actor], "source": "grpc_full_inventory",
                      "complete_inventory": True, "transport_timing_candidate": True,
                      "model_identity_candidate": True, "reasons": [], "identity_reasons": []}}])
    _write_frames(tmp_path, "object_geometry_audit_v2", [{"index": 0,
        "lidar_source_ns": 10, "objects": [geometry, dict(geometry)]}])
    _write_frames(tmp_path, "admission_v2", [{"index": 0,
        "lidar_source_time_ns": 10, "dataset_build_candidate": False,
        "object_gt_transport_timing_candidate": True,
        "object_model_identity_candidate": True, "object_training_label_valid": False}])
    route = {"nav_waypoints": [(0, 0), (100, 0)]}
    records, report = build_features(tmp_path, route)
    assert report["total_objects"] == 1
    assert records[0]["valid"] is False
    assert set(records[0]["invalid_reasons"]) == {
        "frame_not_admitted_for_dataset_build", "ambiguous_actor_match"}
    assert records[0]["object_training_label_valid"] is False
    assert records[0]["path_s_m"] is None
    _write = tmp_path / "object_geometry_audit_v2" / "frames.json"
    _write.write_text(json.dumps([{"index": 0, "lidar_source_ns": 10,
                                    "objects": [geometry]}]), encoding="utf-8")
    records, _ = build_features(tmp_path, route)
    assert records[0]["path_s_m"] == pytest.approx(10)
    assert records[0]["ego_path_s_m"] == pytest.approx(5)
    assert records[0]["delta_s_m"] == pytest.approx(5)
    assert records[0]["valid"] is False
    admission_path = tmp_path / "admission_v2" / "frames.json"
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    admission[0]["dataset_build_candidate"] = True
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    records, _ = build_features(tmp_path, route)
    assert records[0]["valid"] is True
    assert records[0]["delta_s_m"] == pytest.approx(5)
    gt_path = tmp_path / "gt_timeline_v1" / "frames.json"
    gt_rows = json.loads(gt_path.read_text(encoding="utf-8"))
    gt_rows[0]["ego_xy"] = [15, 0]
    gt_path.write_text(json.dumps(gt_rows), encoding="utf-8")
    records, _ = build_features(tmp_path, route)
    assert records[0]["valid"] is True
    assert records[0]["delta_s_m"] == pytest.approx(-5)
    gt_rows[0]["ego_xy"] = None
    gt_path.write_text(json.dumps(gt_rows), encoding="utf-8")
    records, _ = build_features(tmp_path, route)
    assert records[0]["valid"] is False
    assert records[0]["invalid_reasons"] == ["missing_ego_xy"]
    assert records[0]["ego_path_s_m"] is None
    assert records[0]["delta_s_m"] is None
    assert records[0]["path_s_m"] == pytest.approx(10)


def test_timestamp_mismatch_fails_closed(tmp_path):
    _write_frames(tmp_path, "gt_timeline_v1", [{"index": 0, "lidar_source_ns": 10, "ego_xy": [5, 0],
        "object_gt": {"objects": []}}])
    _write_frames(tmp_path, "object_geometry_audit_v2", [{"index": 0,
        "lidar_source_ns": 11, "objects": []}])
    _write_frames(tmp_path, "admission_v2", [{"index": 0,
        "lidar_source_time_ns": 10}])
    with pytest.raises(ValueError, match="source timestamp mismatch"):
        build_features(tmp_path, {"nav_waypoints": [(0, 0), (1, 0)]})


def test_actor_id_fallback_requires_both_missing_instance_keys():
    gt = {"id": "1", "kind": "pedestrian"}
    geometry = {"actor_id": "1", "kind": "pedestrian"}
    assert _match_actor(gt, [geometry]) == (geometry, None)
    assert _match_actor(gt, [dict(geometry, instance_key="other")]) == (
        None, "missing_geometry_actor")


def test_invalid_geometry_keeps_numeric_provenance(tmp_path):
    actor = {"id": "1", "instance_key": "ped:1", "kind": "pedestrian",
             "position": [10, 2, 0], "velocity_kmh": [0, 3.6, 0],
             "size_wlh_m": [2, 4, 1.5]}
    geometry = {"actor_id": "1", "instance_key": "ped:1", "kind": "pedestrian",
                "center_forward_m": 10, "center_left_m": 2,
                "training_bev_relevant": True, "api_box_geometry_candidate": False,
                "reasons": ["invalid_observed_dimensions"]}
    _write_frames(tmp_path, "gt_timeline_v1", [{"index": 0, "lidar_source_ns": 10, "ego_xy": [5, 0],
        "object_gt": {"objects": [actor], "source": "grpc_full_inventory",
                      "complete_inventory": True, "transport_timing_candidate": True,
                      "model_identity_candidate": True}}])
    _write_frames(tmp_path, "object_geometry_audit_v2", [{"index": 0,
        "lidar_source_ns": 10, "objects": [geometry]}])
    _write_frames(tmp_path, "admission_v2", [{"index": 0,
        "lidar_source_time_ns": 10, "dataset_build_candidate": True,
        "object_gt_transport_timing_candidate": True,
        "object_model_identity_candidate": True}])
    records, _ = build_features(tmp_path, {"nav_waypoints": [(0, 0), (100, 0)]})
    assert records[0]["valid"] is False
    assert records[0]["invalid_reasons"] == ["invalid_observed_dimensions"]
    assert records[0]["path_s_m"] == pytest.approx(10)


def test_v2_keeps_empty_frames_and_uses_only_explicit_zones(tmp_path):
    gt = [
        {"index": 0, "lidar_source_ns": 1_000_000_000,
         "ego_xy": [1, 0], "object_gt": {"objects": []}},
        {"index": 1, "lidar_source_ns": 1_100_000_000,
         "ego_xy": [2, 0], "object_gt": {"objects": []}},
        {"index": 2, "lidar_source_ns": 1_200_000_000,
         "ego_xy": None, "object_gt": {"objects": []}},
    ]
    _write_frames(tmp_path, "gt_timeline_v1", gt)
    _write_frames(tmp_path, "object_geometry_audit_v2", [
        {"index": row["index"], "lidar_source_ns": row["lidar_source_ns"],
         "objects": []} for row in gt])
    _write_frames(tmp_path, "admission_v2", [
        {"index": row["index"], "lidar_source_time_ns": row["lidar_source_ns"],
         "dataset_build_candidate": True} for row in gt])
    route = {"nav_waypoints": [(0, 0), (10, 0), (20, 0)],
             "zones": {"dynamic_obstacle": {"start_path_index": 0,
                                            "end_path_index": 1}}}
    frames, objects, report = build_feature_tables(tmp_path, route)
    assert len(frames) == 3
    assert objects == []
    assert report["total_frames"] == 3
    assert report["total_objects"] == 0
    assert frames[0]["ego_path_segment_index"] == 0
    assert frames[0]["zone_id"] == "dynamic_obstacle"
    assert frames[0]["ego_speed_mps"] is None
    assert frames[1]["ego_speed_mps"] == pytest.approx(10)
    assert frames[1]["ego_speed_source"] == "backward_gt_ego_xy_difference"
    assert frames[2]["valid"] is False
    assert frames[2]["invalid_reasons"] == ["missing_ego_xy"]


def test_zone_checkpoint_must_match_official_path(tmp_path):
    _write_frames(tmp_path, "gt_timeline_v1", [])
    _write_frames(tmp_path, "object_geometry_audit_v2", [])
    _write_frames(tmp_path, "admission_v2", [])
    route = {"nav_waypoints": [(0, 0), (10, 0)],
             "checkpoints": [{"id": 8, "path_index": 1, "x": 11, "y": 0}],
             "zones": {"roundabout": {"ego_entry_checkpoint": 8,
                                       "ego_exit_checkpoint": 8}}}
    with pytest.raises(ValueError, match="checkpoint coordinates differ"):
        build_feature_tables(tmp_path, route)


def test_static_zone_uses_checkpoints_three_to_four():
    route = {"nav_waypoints": [(float(x), 0) for x in range(6)],
             "checkpoints": [
                 {"id": 3, "path_index": 1, "x": 1, "y": 0},
                 {"id": 4, "path_index": 4, "x": 4, "y": 0}],
             "zones": {"static_obstacle": {"route_path_index": 3}}}
    assert _zone_intervals(route) == [("static_obstacle", 1, 4)]
    route["zones"]["static_obstacle"]["route_path_index"] = 5
    with pytest.raises(ValueError, match="anchor lies outside"):
        _zone_intervals(route)


def test_official_roundabout_map_builds_lead_gap_and_instruction():
    """The real MGeo link and merge anchor produce a causal lead state."""
    import math
    import yaml
    from pathlib import Path
    from MoraiDrive.data_collection.mgeo_navigation import load_official_route
    from MoraiDrive.data_collection.situation_labeling.instruction_features import enrich
    from MoraiDrive.data_collection.situation_labeling.rules import evaluate_frames

    scenario = Path(__file__).resolve().parents[2] / "scenario_runner"
    definition = scenario / "config" / "full_loop_links.yaml"
    data = scenario / "data" / "R_KR_PR_K-city_2025"
    route = load_official_route(str(definition),
                                str(data / "2026_molit_comp_global_path.txt"))
    route["zones"] = yaml.safe_load(definition.read_text(encoding="utf-8-sig"))[
        "full_loop"]["zones"]
    links = json.loads((data / "link_set.json").read_text(encoding="utf-8"))
    points = next(link["points"] for link in links
                  if link["idx"] == "A2256W000134")
    before, ego, lead_xy = points[4], points[5], points[18]
    heading = math.degrees(math.atan2(
        points[19][1] - points[17][1], points[19][0] - points[17][0]))
    projector = RouteProjector(route["nav_waypoints"])
    frames = [
        {"index": index, "lidar_source_ns": 1_000_000_000 + index * 100_000_000,
         "zone_id": "roundabout", "valid": True, "object_inventory_usable": True,
         "ego_world_x_m": xy[0], "ego_world_y_m": xy[1],
         "ego_path_s_m": projector.project(*xy[:2])[0],
         "ego_speed_mps": speed}
        for index, (xy, speed) in enumerate(((before, 5.2), (ego, 5.0)))
    ]
    vehicle = {
        "index": 1, "actor_id": "lead", "instance_key": "npc:lead",
        "kind": "npc", "valid": True, "invalid_reasons": [],
        "world_x_m": lead_xy[0], "world_y_m": lead_xy[1],
        "actor_heading_deg": heading, "actor_width_m": 2.0,
        "actor_length_m": 4.5, "speed_mps": 4.0,
    }
    enrich(frames, [vehicle], route)
    current = frames[1]
    assert current["current_link_id"] == "A2256W000134"
    assert current["in_circulating_lane"] is True
    assert current["lead_actor_id"] == "lead"
    assert current["lead_bumper_gap_verified"] is True
    assert current["lead_gap_insufficient"] is True
    assert vehicle["is_lead_vehicle"] is True
    assert current["collision_risk"] is False
    assert evaluate_frames(frames, [vehicle])[1]["instruction_state"] == "follow_decelerate"


def test_dynamic_clearance_transition_and_unmapped_roundabout_stay_conservative():
    from MoraiDrive.data_collection.situation_labeling.instruction_features import enrich
    from MoraiDrive.data_collection.situation_labeling.rules import evaluate_frames

    route = {"nav_waypoints": [(0, 0), (100, 0)]}
    frames = [
        {"index": i, "lidar_source_ns": 1_000_000_000 + i * 100_000_000,
         "zone_id": "dynamic_obstacle", "valid": True,
         "object_inventory_usable": True, "ego_world_x_m": i,
         "ego_world_y_m": 0, "ego_path_s_m": i, "ego_speed_mps": 1.0}
        for i in range(2)
    ]
    people = [
        {"index": i, "actor_id": "ped", "instance_key": "ped:1",
         "kind": "pedestrian", "valid": True, "invalid_reasons": [],
         "world_x_m": 8 + i, "world_y_m": d,
         "path_d_m": d, "delta_s_m": 8,
         "actor_width_m": 0.6, "actor_length_m": 0.6,
         "v_d_mps": 0.0, "speed_mps": 0.0}
        for i, d in enumerate((0.2, 3.0))
    ]
    enrich(frames, people, route)
    assert frames[0]["pedestrian_on_conflict_path"] is True
    assert frames[1]["all_relevant_pedestrians_clear"] is True
    assert frames[1]["dynamic_resume_transition_start"] is True
    assert evaluate_frames(frames, people)[1]["instruction_state"] == "resume"

    vehicle_frame = {
        "index": 0, "lidar_source_ns": 1, "zone_id": "roundabout",
        "valid": True, "object_inventory_usable": True,
        "ego_world_x_m": 0, "ego_world_y_m": 0,
        "ego_path_s_m": 0, "ego_speed_mps": 0,
    }
    vehicle = {
        "index": 0, "actor_id": "npc", "kind": "npc",
        "valid": True, "invalid_reasons": [],
        "world_x_m": 10, "world_y_m": 0,
        "actor_width_m": 2, "actor_length_m": 4,
    }
    enrich([vehicle_frame], [vehicle], route)
    assert vehicle_frame["collision_risk"] is None
    assert evaluate_frames([vehicle_frame], [vehicle])[0]["status"] == "unknown"
