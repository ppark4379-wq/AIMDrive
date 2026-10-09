import tempfile
import hashlib
from pathlib import Path

from MoraiDrive.data_collection.situation_labeling.expert_roundabout_phase import (
    build_roundabout_observations,
)


def _frames(indices, zone="roundabout"):
    return [{"index": i, "lidar_source_ns": i * 100_000_000, "zone_id": zone}
            for i in indices]


def _write_log(tmp_path, lines):
    path = tmp_path / "supervisor.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_forward_fill_assigns_phase_only_to_roundabout_frames():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        log = _write_log(tmp_path, [
            "[INFO] [1000.0, 0.0]: [ExpertInput] tick",
            "[CTRL] dump",
            "[ROUNDABOUT] cruise target=16.3km/h conflict_id=None ttc=infs "
            "front_id=None gap=None hold_gap=18.10m entering=False merged=False "
            "ceiling=16.3km/h adjusted=False",
            "[CTRL] dump",
            "[ROUNDABOUT] yield target=25.9km/h conflict_id=1 ttc=4.20s "
            "front_id=3 gap=41.77 hold_gap=29.28m entering=False merged=False "
            "ceiling=25.9km/h adjusted=False",
            "[CTRL] dump",
            "[INFO] [1000.6, 0.6]: [ExpertInput] tick",
        ])
        frames = _frames(range(6))
        frames[4]["zone_id"] = "static_obstacle"  # must stay untouched
        observations = build_roundabout_observations(frames, log)
        # Frames before the first transition get no observation.
        assert 0 not in observations
        assert 1 not in observations
        semantic_fields = {
            "mode": "roundabout_entry", "roundabout_phase": "cruise",
            "roundabout_entering": False, "roundabout_committed": False,
            "roundabout_front_id": None, "roundabout_conflict_id": None,
            "planner_target_speed_mps": 16.3 / 3.6,
            "source": "existing_roundabout_text_log",
        }
        assert {key: observations[2][key] for key in semantic_fields} == semantic_fields
        source = observations[2]["transition_source"]
        assert source == {"path": str(log), "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                          "line_no": 3}
        receipt = observations[2]["legacy_transition_observation"]
        assert receipt["transition_source"] == source
        assert receipt["frame_index"] == 2 and receipt["lidar_source_ns"] == 200_000_000
        assert receipt["numeric_target_current_verified"] is False
        assert receipt["physical_simultaneity_verified"] is False
        assert receipt["uses_future_ego"] is False
        assert receipt["transition_anchor_lower_s"] == 0.0
        assert receipt["transition_anchor_upper_s"] == 0.6
        # Forward-fill carries "cruise" until the "yield" transition.
        assert observations[3]["roundabout_phase"] == "cruise"
        assert observations[5]["roundabout_phase"] == "yield"
        assert observations[5]["roundabout_conflict_id"] == "1"
        assert observations[5]["roundabout_front_id"] == "3"
        # A different zone's frame is never written, even mid-sequence.
        assert 4 not in observations


def test_no_transitions_returns_empty_mapping():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        log = _write_log(tmp_path, ["[INFO] [1000.0, 0.0]: [ExpertInput] tick"])
        assert build_roundabout_observations(_frames(range(3)), log) == {}


def test_transition_after_capture_end_cannot_replace_the_last_recorded_phase(tmp_path):
    log=_write_log(tmp_path,[
        "[INFO] [1.000, 1.000]: [ExpertInput]",
        "[CTRL] one",
        "[ROUNDABOUT] cruise target=10.0km/h conflict_id=None ttc=infs front_id=None gap=None hold_gap=5m entering=False merged=False",
        "[CTRL] two",
        "[ROUNDABOUT] emergency target=0.0km/h conflict_id=1 ttc=0.0s front_id=1 gap=1 hold_gap=5m entering=True merged=False",
        "[INFO] [3.000, 3.000]: [ExpertInput]"])
    frames=[{"index":i,"zone_id":"roundabout","lidar_source_ns":1_000_000_000+i*500_000_000}
            for i in range(3)]
    observations=build_roundabout_observations(frames,log)
    assert observations[2]["roundabout_phase"]=="cruise"
    assert all(raw["roundabout_phase"]!="emergency" for raw in observations.values())
