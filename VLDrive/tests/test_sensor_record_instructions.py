"""Navigation labels must retain native scene identity, gaps, and route provenance."""
import csv
import json
from types import SimpleNamespace

import pytest

from tools.build_instructions import (
    COMMAND_TEXT, RouteLookaheadPolicy, SENSOR_RECORD_SCHEMA,
    main, parse_arguments, sensor_record_episode, sha256, verified_source,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def capture(tmp_path):
    root = tmp_path / "raw"
    episode = root / "campaign/lap_001/episode_000"
    map_root = tmp_path / "map"
    map_root.mkdir()
    points = [[i * 10, 0] for i in range(5)]
    write_json(map_root / "link_set.json", [
        {"idx": "follow", "related_signal": "straight", "points": points[:2]},
        {"idx": "turn", "related_signal": "right_unprotected", "points": points[2:]},
    ])
    write_json(map_root / "node_set.json", [])
    path = map_root / "2026_molit_comp_global_path.txt"
    path.write_text("".join("{} {}\n".format(*point) for point in points))
    definition = tmp_path / "route.yaml"
    definition.write_text("full_loop:\n  route_links: [follow, turn]\n")
    sources = {"global_path": path, "link_set": map_root / "link_set.json",
               "node_set": map_root / "node_set.json", "route_definition": definition}
    hashes = {key: sha256(value) for key, value in sources.items()}
    write_json(episode.parent / "sensor_contract.json", {"capture": {"traffic_light_ground_truth": {
        "map_root": str(map_root), "route_key": "full_loop", "source_sha256": hashes}}})
    write_json(episode / "route_signal_audit_v1/report.json", {
        "route_key": "full_loop", "global_path_sha256": hashes["global_path"],
        "route_definition_sha256": hashes["route_definition"], "route_source_sha256": hashes})
    expert_path = episode / "runtime/expert/path.csv"
    expert_path.parent.mkdir(parents=True)
    with expert_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x", "y"])
        writer.writerows(points)
    write_json(episode / "episode.json", {
        "global_path_sha256": hashes["global_path"],
        "expert_inputs": {"runtime/expert/path.csv": sha256(expert_path)},
        "scenario_config": {"route_definition_path": str(definition)}})
    signals, gt, associations = [], [], []
    for index, xy in enumerate(points):
        stamp = 1_000_000_000 + index * 100_000_000
        signal = {"index": index, "sensor_source_ns": stamp, "valid": False,
                  "reason": "no_upcoming_route_stopline"}
        if index != 2:  # Missing localization must not be interpolated from later frames.
            signal.update(route_index=index, route_link_id="follow" if index < 2 else "turn",
                          route_link_index=0 if index < 2 else 1,
                          lane_error_m=0.0, heading_error_deg=0.0)
        signals.append(signal)
        gt.append({"index": index, "lidar_source_ns": stamp,
                   "native_filename_time_ns": stamp + 3000, "ego_xy": xy})
        associations.append({"index": index, "scene_time_hypothesis_ns": stamp,
                             "native_filename_time_ns": stamp + 3000,
                             "camera_files": {"camera_1": "camera/{}.png".format(index)},
                             "lidar_file": "lidar/{}.bin".format(index)})
    for relative, value in [("route_signal_audit_v1/frames.json", signals),
                            ("gt_timeline_v1/frames.json", gt),
                            ("native_associations/associations.json", associations)]:
        write_json(episode / relative, value)
    args = SimpleNamespace(map_root=None, route_definition=None, lookahead_m=10.0,
                           curve_lookahead_m=20.0, curve_threshold_deg=20.0,
                           max_route_error_m=10.0, max_heading_error_deg=60.0,
                           max_sensor_gap_ms=150.0)
    return root, episode, args


def test_native_frames_signal_invalid_and_missing_route(capture):
    _, episode, args = capture
    records, report = sensor_record_episode(episode, args)
    assert [(r["start_frame"], r["stop_frame"], r["command"]) for r in records] == [
        (0, 0, 4), (1, 1, 2), (3, 4, 2)]
    assert all(row["schema_version"] == SENSOR_RECORD_SCHEMA for row in records)
    assert report["counts"] == {"labeled_frames": 4, "missing_audited_route_position": 1}
    assert report["frames"][2]["instruction_id"] is None
    assert report["frames"][3]["lidar_file"] == "lidar/3.bin"
    assert records[2]["start_sensor_source_ns"] == 1_300_000_000


@pytest.mark.parametrize("fault,match", [
    ("timestamp", "timestamp mismatch"), ("frame_index", "frame index mismatch"),
    ("link", "route/link association"), ("map", "SHA256-matching"),
    ("expert_path", "expert path hash mismatch"),
])
def test_corrupt_provenance_or_alignment_is_rejected(capture, fault, match):
    _, episode, args = capture
    if fault in ("timestamp", "frame_index"):
        path = episode / "native_associations/associations.json"
        rows = json.loads(path.read_text())
        rows[1]["scene_time_hypothesis_ns" if fault == "timestamp" else "index"] = -1
        write_json(path, rows)
    elif fault == "link":
        path = episode / "route_signal_audit_v1/frames.json"
        rows = json.loads(path.read_text())
        rows[1]["route_link_id"] = "wrong_branch"
        write_json(path, rows)
    elif fault == "map":
        (episode.parents[3] / "map/link_set.json").write_text("[]")
    else:
        (episode / "runtime/expert/path.csv").write_text("x,y\n0,0\n")
    with pytest.raises(ValueError, match=match):
        sensor_record_episode(episode, args)


def test_avoidance_is_not_lane_change_and_bad_localization_is_excluded(capture):
    _, episode, args = capture
    path = episode / "gt_timeline_v1/frames.json"
    rows = json.loads(path.read_text())
    rows[0]["ego_xy"] = [0, 3.5]  # Lateral obstacle avoidance, still following route.
    rows[1]["ego_xy"] = [10, 50]
    rows[0]["future_waypoints"] = [[float("nan"), 999]]
    write_json(path, rows)
    records, report = sensor_record_episode(episode, args)
    assert records[0]["command"] == 4
    assert report["frames"][1]["exclusion_reason"] == "route_distance_exceeds_navigation_gate"


def test_closed_route_announces_turn_across_start_seam():
    policy = RouteLookaheadPolicy({
        "nav_waypoints": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
        "nav_road_options": [4, 2, 4, 4, 4], "closed_route": True,
    }, lookahead_m=20)
    measurement = {"gps_x": 0, "gps_y": 5, "navigation": {"route_index": 3}}
    assert policy.instruction_for(measurement) == (2, COMMAND_TEXT[2])


def test_closed_route_curve_lookahead_crosses_seam():
    policy = RouteLookaheadPolicy({
        "nav_waypoints": [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
        "nav_road_options": [4] * 5, "closed_route": True,
    }, curve_lookahead_m=10)
    measurement = {"gps_x": 0, "gps_y": 5, "navigation": {"route_index": 3}}
    assert policy.instruction_for(measurement) == (4, "Follow the road as it curves to the left.")


def test_source_allows_only_line_ending_equivalence(tmp_path):
    path = tmp_path / "source.txt"
    path.write_bytes(b"a\nb\n")
    expected = sha256(path)
    path.write_bytes(b"a\r\nb\r\n")
    assert verified_source([path], expected, "test") == path
    path.write_bytes(b"a\r\nc\r\n")
    with pytest.raises(ValueError, match="SHA256-matching"):
        verified_source([path], expected, "test")


def test_cli_root_and_episode_repeat_run_and_overwrite_protection(capture, tmp_path, monkeypatch):
    root, episode, _ = capture
    output = tmp_path / "out/navigation.jsonl"
    argv = ["build", "--dataset-root", str(root), "--output", str(output), "--lookahead-m", "10"]
    monkeypatch.setattr("sys.argv", argv)
    main()
    first = output.read_bytes()
    report_path = output.with_suffix(".report.json")
    first_report = report_path.read_bytes()
    assert json.loads(first_report)["legacy_stage2_loader_compatible"] is False
    monkeypatch.setattr("sys.argv", ["build", "--dataset-root", str(episode), "--output", str(output),
                                     "--lookahead-m", "10"])
    main()
    assert output.read_bytes() == first
    assert report_path.read_bytes() == first_report
    monkeypatch.setattr("sys.argv", argv + ["--lookahead-m", "5"])
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        main()
    assert output.read_bytes() == first
    assert report_path.read_bytes() == first_report


def test_temporal_gap_splits_identical_instructions(capture):
    _, episode, args = capture
    for relative, time_field in [("route_signal_audit_v1/frames.json", "sensor_source_ns"),
                                  ("gt_timeline_v1/frames.json", "lidar_source_ns"),
                                  ("native_associations/associations.json", "scene_time_hypothesis_ns")]:
        path = episode / relative
        rows = json.loads(path.read_text())
        rows[4][time_field] += 1_000_000_000
        write_json(path, rows)
    records, _ = sensor_record_episode(episode, args)
    assert [(r["start_frame"], r["stop_frame"]) for r in records][-2:] == [(3, 3), (4, 4)]


def config_document():
    from tools.build_instructions import VLDRIVE_ROOT
    return __import__("utils.config.config_files", fromlist=["read_document"]).read_document(VLDRIVE_ROOT / "configs/vldrive_realtime.json#presets/navigation_instructions")


def test_config_paths_are_independent_of_working_directory(tmp_path, monkeypatch):
    from tools.build_instructions import VLDRIVE_ROOT
    path = tmp_path / "config.json"
    write_json(path, config_document())
    monkeypatch.chdir(tmp_path)
    args = parse_arguments(["--config", str(path)])
    assert args.dataset_root == VLDRIVE_ROOT / "data/raw/basic_drive"
    assert args.output == VLDRIVE_ROOT / "data/processed/instructions/basic_drive_navigation.jsonl"
    assert args.lookahead_m == 30
    assert args.max_sensor_gap_ms == 150


@pytest.mark.parametrize("key,value", [
    ("lookahead_m", True), ("lookahead_m", 0), ("lookahead_m", "30"),
    ("lookahead_m", float("nan")), ("curve_threshold_deg", 180),
    ("max_heading_error_deg", 181), ("input_format", "typo"),
    ("policy", "typo"), ("dataset_root", None), ("max_sensor_gap_ms", -1),
    ("lookahed_m", 30),
])
def test_bad_config_fails_before_generation(tmp_path, key, value):
    document = config_document()
    document[key] = value
    path = tmp_path / "config.json"
    write_json(path, document)
    with pytest.raises(SystemExit) as exc:
        parse_arguments(["--config", str(path)])
    assert exc.value.code == 2


def test_config_rejects_cli_overrides_and_missing_keys(tmp_path):
    path = tmp_path / "config.json"
    document = config_document()
    write_json(path, document)
    with pytest.raises(SystemExit):
        parse_arguments(["--config", str(path), "--lookahead-m", "5"])
    del document["lookahead_m"]
    write_json(path, document)
    with pytest.raises(SystemExit):
        parse_arguments(["--config", str(path)])


def test_config_controls_generation_and_records_effective_values(capture, tmp_path, monkeypatch):
    root, _, _ = capture
    document = config_document()
    document.update(dataset_root=str(root), output=str(tmp_path / "result.jsonl"),
                    lookahead_m=5, max_sensor_gap_ms=50)
    path = tmp_path / "config.json"
    write_json(path, document)
    monkeypatch.setattr("sys.argv", ["build", "--config", str(path)])
    main()
    output = tmp_path / "result.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    # 5 m (not 30 m) lookahead: frame 1 still follows the road.
    # 50 ms gap threshold: successive 100 ms frames form separate ranges.
    assert [(r["start_frame"], r["stop_frame"], r["command"]) for r in rows] == [
        (0, 0, 4), (1, 1, 4), (3, 3, 2), (4, 4, 2)]
    report = json.loads(output.with_suffix(".report.json").read_text())
    assert report["configuration"]["sha256"] == sha256(path)
    assert report["configuration"]["effective_parameters"]["lookahead_m"] == 5
    assert report["max_contiguous_sensor_gap_ns"] == 50_000_000
    main()  # The config and provenance also reproduce exactly.
