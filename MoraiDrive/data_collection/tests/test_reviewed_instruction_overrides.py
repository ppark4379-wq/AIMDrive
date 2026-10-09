from MoraiDrive.data_collection.situation_labeling.reviewed_overrides import (
    LAP085, apply_reviewed_overrides,
)


def test_manual_visual_correction_cannot_certify_missing_future(tmp_path):
    row = dict(index=158, zone_id="static_obstacle", status="matched", label="follow",
               instruction_consistency="no_future", instruction_state="avoid",
               instruction_id="follow_avoid", variables={"avoid_direction": "left"})
    assert apply_reviewed_overrides([row], tmp_path / LAP085 / "episode_000") == 1
    assert row["instruction_state"] == "return_to_path"
    assert row["instruction_consistency"] == "no_future"


def test_manual_visual_correction_cannot_restore_unknown_evidence(tmp_path):
    row = dict(index=158, zone_id="static_obstacle", status="unknown",
               unknown_reasons=["object_inventory_unusable"], instruction=None,
               variables={})
    assert apply_reviewed_overrides([row], tmp_path / LAP085 / "episode_000") == 0
    assert row["instruction"] is None and row["status"] == "unknown"
