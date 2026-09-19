import json
from pathlib import Path

import pytest

from tools.build_notice_instructions import build_dataset_notices, compress_notices, snap_zones
from tools.rviz_notice_zones import marker_specs
from utils.notice_policy import (NoticeRanges, NoticeTracker, RouteNoticePolicy,
                                 compose_instruction_batch, current_condition_suffix, nearest_index)


def synthetic_policy(tmp_path):
    route = tmp_path / "route.txt"
    route.write_text("".join("{} 0 0\n".format(i) for i in range(20)))
    source = tmp_path / "points.json"
    source.write_text(json.dumps(dict(map_id="test", global_path="route.txt", zones=[
        dict(id="signal", label="signal", type="signal", text="Watch traffic lights.",
             start=[3.2, .1], end=[12.2, .1], color=[1, 1, 0]),
        dict(id="static", label="static", type="static", text="Watch stationary obstacles.",
             start=[6.1, .1], end=[12.2, .1], color=[1, 0, 0])])), encoding="utf-8")
    config = tmp_path / "zones.json"
    snap_zones(source, config)
    return RouteNoticePolicy(config)


def test_snap_uses_exact_vertices_and_preserves_requested_coordinates(tmp_path):
    policy = synthetic_policy(tmp_path)
    start = policy.zones[0]["start"]
    assert start["requested_xy"] == [3.2, .1]
    assert start["route_index"] == 3
    assert start["xyz"] == [3., 0., 0.]
    assert start["snap_distance_m"] < .3


def test_nearby_vertex_does_not_get_stuck_at_previous_hint():
    assert nearest_index([(0., 0.), (.01, 0.)], .01, 0., hint=0) == (1, 0.)


def test_real_route_duplicate_endpoints_have_same_notice():
    policy = RouteNoticePolicy()
    assert policy.points[424] == policy.points[425]
    assert policy.at_index(424)["zone_ids"] == policy.at_index(425)["zone_ids"] == ["signal_01"]
    assert policy.at_index(426)["zone_ids"] == []
    assert policy.first_index_after(424) == 426


def test_coordinate_outlier_is_rejected_not_silently_snapped(tmp_path):
    policy = synthetic_policy(tmp_path)
    source = tmp_path / "points.json"
    raw = json.loads(source.read_text())
    raw["zones"][1]["end"] = [-59.2, -902.58]
    source.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="confirm coordinates"):
        snap_zones(source, policy.config_path)


def test_entry_overlap_exit_and_second_lap_use_identical_live_and_index_notices(tmp_path):
    policy = synthetic_policy(tmp_path)
    tracker = NoticeTracker(policy)
    events = []
    for _ in range(2):
        for index, point in enumerate(policy.points):
            state = tracker.update(*point[:2], navigation="Follow the road.")
            assert state["notice_text"] == policy.at_index(index)["notice_text"]
            assert state["llm_text"] == compose_instruction_batch(
                ["Follow the road."], [state["notice_text"]])[0]
            if state["entered"] or state["exited"]:
                events.append((index, state["entered"], state["exited"]))
    lap = [(3, ["signal"], []), (6, ["static"], []), (13, [], ["signal", "static"])]
    assert events == lap * 2
    assert policy.at_index(12)["zone_ids"] == ["signal", "static"]
    assert tracker.update(1000, 1000)["valid"] is False
    assert tracker.update(1000, 1000)["notice_text"] == ""


def test_rviz_displays_exact_llm_notice_and_route_vertices(tmp_path):
    policy = synthetic_policy(tmp_path)
    state = policy.at_index(6)
    specs = marker_specs(policy, state, (6., 0.))
    active = next(s for s in specs if s["ns"] == "notice_current")
    assert " ".join(active["text"].splitlines()[1:]) == state["notice_text"]
    strip = next(s for s in specs if s["ns"] == "notice_zones" and s["id"] == 1)
    assert [(p[0], p[1]) for p in strip["points"]] == [(float(i), 0.) for i in range(6, 13)]
    assert "none" in marker_specs(policy, policy.at_index(13), (13., 0.))[-1]["text"]


def test_prefix_composition_no_notice_and_boundary_loss_suffix():
    assert compose_instruction_batch(["nav"], [""]) == ["nav"]
    assert compose_instruction_batch(["nav"], ["notice"]) == ["nav\nnotice"]
    with pytest.raises(ValueError):
        compose_instruction_batch([""], ["notice"])
    with pytest.raises(ValueError):
        compose_instruction_batch(["nav"], ["one", "two"])
    assert current_condition_suffix([("nav", ""), ("nav", "notice"), ("nav", "notice")]) == 1
    assert current_condition_suffix([("nav", "a"), ("nav", "b"), ("nav", "a")]) == 2


def test_modified_route_rejects_stale_snapped_zones(tmp_path):
    policy = synthetic_policy(tmp_path)
    policy.route_path.write_text("0 0 0\n1 0 0\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        RouteNoticePolicy(policy.config_path)


def test_offline_labels_use_current_position_and_cover_inactive_frames(tmp_path):
    policy = synthetic_policy(tmp_path)
    dataset = tmp_path / "dataset"
    measurements = dataset / "data/lap/measurements_full"
    measurements.mkdir(parents=True)
    (dataset / "dataset_index.txt").write_text("data/lap/ 20\n")
    (measurements.parent / "route.json").write_text("{}")
    for frame in range(20):
        (measurements / ("%04d.json" % frame)).write_text(json.dumps(dict(
            frame_id=frame, gps_x=frame, gps_y=0, future_waypoints="poison",
            objects="poison", navigation=dict(route_valid=True))))
    rows = build_dataset_notices(dataset, policy)
    assert [(r["start_frame"], r["stop_frame"]) for r in rows] == [(0, 2), (3, 5), (6, 12), (13, 19)]
    path = tmp_path / "notices.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ranges = NoticeRanges(path)
    for frame in range(20):
        assert ranges.for_frame("lap", frame)["text"] == policy.at_index(frame)["notice_text"]
    with pytest.raises(ValueError, match="Missing notice coverage"):
        ranges.for_frame("other_run", 0)
