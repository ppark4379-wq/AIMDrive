import copy
import json
import math

import pytest

from tools.build_instructions import (
    COMMAND_TEXT, RouteLookaheadPolicy, SCHEMA, main,
)


def route(points, options):
    return {"nav_waypoints": points, "nav_road_options": options}


def ego(index, x, y=0.0):
    return {"gps_x": x, "gps_y": y,
            "navigation": {"route_index": index, "route_valid": True}}


def test_turn_is_announced_by_route_arc_distance_not_future_ego():
    policy = RouteLookaheadPolicy(route([(i, 0) for i in range(101)],
                                       [4] * 50 + [2] * 11 + [4] * 40), 30)
    assert policy.command_for(ego(19, 19.9)) == 4
    measurement = ego(20, 20.0)
    assert policy.command_for(measurement) == 2
    # Poison all future-derived fields: only current position/route progress count.
    poisoned = copy.deepcopy(measurement)
    poisoned.update(future_waypoints=[[99999, -99999]], next_ego_raw={"x": -1e20},
                    future_trajectory="not even a trajectory", steer=1000)
    assert policy.instruction_for(poisoned) == policy.instruction_for(measurement)


def test_active_turn_wins_until_exit_even_if_following_turn_is_close():
    policy = RouteLookaheadPolicy(route([(i, 0) for i in range(101)],
                                       [4] * 20 + [1] * 21 + [4] * 9 + [2] * 11 + [4] * 40))
    assert policy.command_for(ego(30, 30)) == 1
    assert policy.command_for(ego(40, 40)) == 1
    assert policy.command_for(ego(41, 41)) == 2
    assert policy.command_for(ego(61, 61)) == 4


def test_route_distance_not_euclidean_shortcut_at_hairpin():
    points = [(0, 0), (0, 40), (1, 40), (1, 0)]
    policy = RouteLookaheadPolicy(route(points, [4, 4, 4, 1]), lookahead_m=30)
    # The turn lies 1 m away spatially but 81 m along the prescribed route.
    assert policy.command_for(ego(0, 0)) == 4


def test_curved_road_with_straight_metadata_gets_geometric_curve_text():
    radius = 15.0
    points = [(radius * math.sin(i / 40), radius * (math.cos(i / 40) - 1))
              for i in range(61)]
    policy = RouteLookaheadPolicy(route(points, [3] * len(points)))
    command, text = policy.instruction_for(ego(0, 0))
    assert command == 4
    assert text == "Follow the road as it curves to the right."
    left = RouteLookaheadPolicy(route([(x, -y) for x, y in points], [3] * len(points)))
    assert left.instruction_for(ego(0, 0))[1] == "Follow the road as it curves to the left."


def test_explicit_maneuver_overrides_road_geometry():
    points = [(15 * math.sin(i / 40), 15 * (math.cos(i / 40) - 1)) for i in range(61)]
    policy = RouteLookaheadPolicy(route(points, [4] * 30 + [1] * 31))
    assert policy.instruction_for(ego(0, 0)) == (1, COMMAND_TEXT[1])


def test_straight_metadata_is_route_follow_without_curve_or_turn():
    policy = RouteLookaheadPolicy(route([(0, 0), (20, 0), (40, 0)], [3, 3, 3]))
    assert policy.instruction_for(ego(0, 0)) == (4, COMMAND_TEXT[4])


def test_invalid_localization_and_nonfinite_parameters_are_rejected():
    data = route([(0, 0), (20, 0)], [4, 4])
    with pytest.raises(ValueError, match="finite and positive"):
        RouteLookaheadPolicy(data, float("nan"))
    policy = RouteLookaheadPolicy(data)
    measurement = ego(0, 0)
    measurement["navigation"]["route_valid"] = False
    with pytest.raises(ValueError, match="invalid localization"):
        policy.command_for(measurement)


def test_default_generator_preserves_v1_and_refuses_overwriting_different_policy(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    measurements = root / "data" / "route" / "measurements_full"
    measurements.mkdir(parents=True)
    (root / "dataset_index.txt").write_text("data/route/ 2\n")
    (measurements.parent / "route.json").write_text(json.dumps(
        route([(0, 0), (20, 0)], [4, 2])))
    for frame in range(2):
        measurement = ego(frame, frame * 20)
        measurement.update(frame_id=frame, native_sync_frame=100 + frame * 5,
                           gt_command=[4, 2][frame])
        (measurements / ("%04d.json" % frame)).write_text(json.dumps(measurement))
    output = tmp_path / "instructions.jsonl"
    arguments = ["build", "--dataset-root", str(root), "--output", str(output)]
    monkeypatch.setattr("sys.argv", arguments)
    main()
    content = output.read_text()
    records = [json.loads(line) for line in content.splitlines()]
    assert len(records) == 2
    assert all(row["schema_version"] == SCHEMA for row in records)
    assert "instruction_policy" not in records[0]
    main()
    assert output.read_text() == content
    monkeypatch.setattr("sys.argv", arguments + ["--policy", "route_lookahead"])
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        main()
    assert output.read_text() == content
