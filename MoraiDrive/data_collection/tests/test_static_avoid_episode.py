from MoraiDrive.data_collection.situation_labeling.instruction_features import (
    attach_static_avoid_episode,
)


def _frame(index, zone_id, expert_label, valid=True):
    return {"index": index, "zone_id": zone_id, "valid": valid,
            "expert_label": expert_label}


def test_single_shot_start_then_sustained_active_until_label_leaves_follow():
    frames = [
        _frame(0, "static_obstacle", "approach"),
        _frame(1, "static_obstacle", "follow"),
        _frame(2, "static_obstacle", "follow"),
        _frame(3, "static_obstacle", "follow"),
        _frame(4, "static_obstacle", "rejoined"),
        _frame(5, "static_obstacle", "rejoined"),
    ]
    attach_static_avoid_episode(frames)
    assert [f["static_avoid_transition_start"] for f in frames] == [
        False, True, False, False, False, False]
    assert [f["static_avoid_episode_active"] for f in frames] == [
        False, True, True, True, False, False]


def test_reentering_follow_after_rejoined_is_a_new_transition():
    frames = [
        _frame(0, "static_obstacle", "follow"),
        _frame(1, "static_obstacle", "rejoined"),
        _frame(2, "static_obstacle", "follow"),
    ]
    attach_static_avoid_episode(frames)
    assert frames[0]["static_avoid_transition_start"] is True
    assert frames[1]["static_avoid_episode_active"] is False
    assert frames[2]["static_avoid_transition_start"] is True
    assert frames[2]["static_avoid_episode_active"] is True


def test_other_zones_and_invalid_frames_never_get_an_active_episode():
    frames = [
        _frame(0, "static_obstacle", "follow"),
        _frame(1, "roundabout", "follow"),
        _frame(2, "static_obstacle", "follow", valid=False),
    ]
    attach_static_avoid_episode(frames)
    assert frames[1]["static_avoid_episode_active"] is False
    assert frames[1]["static_avoid_transition_start"] is False
    assert frames[2]["static_avoid_episode_active"] is False


def test_noncontiguous_index_resets_the_episode():
    frames = [
        _frame(0, "static_obstacle", "follow"),
        _frame(1, "static_obstacle", "follow"),
        _frame(5, "static_obstacle", "follow"),  # gap in index
    ]
    attach_static_avoid_episode(frames)
    assert frames[1]["static_avoid_transition_start"] is False
    assert frames[1]["static_avoid_episode_active"] is True
    # Treated as a fresh episode after the gap, not a continuation.
    assert frames[2]["static_avoid_transition_start"] is True


def test_initial_hold_covers_five_frames_from_the_start_by_default():
    # Start at index 12: "avoid" holds through index 16 (12-16, 5 frames),
    # then index 17 onward is "avoid_continue" territory.
    frames = [_frame(i, "static_obstacle", "approach") for i in range(12)]
    frames += [_frame(i, "static_obstacle", "follow") for i in range(12, 22)]
    attach_static_avoid_episode(frames)
    by_index = {f["index"]: f for f in frames}
    assert by_index[11]["static_avoid_initial_hold"] is False
    for i in range(12, 17):
        assert by_index[i]["static_avoid_initial_hold"] is True, i
    for i in range(17, 22):
        assert by_index[i]["static_avoid_initial_hold"] is False, i
    # The episode itself stays active throughout, independent of the hold.
    for i in range(12, 22):
        assert by_index[i]["static_avoid_episode_active"] is True


def test_initial_hold_window_is_configurable():
    frames = [_frame(i, "static_obstacle", "follow") for i in range(5)]
    attach_static_avoid_episode(frames, hold_frames=2)
    assert [f["static_avoid_initial_hold"] for f in frames] == [
        True, True, False, False, False]


def test_initial_hold_ends_early_if_the_episode_itself_ends_first():
    frames = [
        _frame(0, "static_obstacle", "follow"),
        _frame(1, "static_obstacle", "follow"),
        _frame(2, "static_obstacle", "rejoined"),
    ]
    attach_static_avoid_episode(frames, hold_frames=5)
    assert frames[0]["static_avoid_initial_hold"] is True
    assert frames[1]["static_avoid_initial_hold"] is True
    assert frames[2]["static_avoid_initial_hold"] is False
