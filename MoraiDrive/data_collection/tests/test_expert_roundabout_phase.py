import tempfile
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
        assert observations[2] == {
            "mode": "roundabout_entry", "roundabout_phase": "cruise",
            "roundabout_entering": False, "roundabout_committed": False,
            "roundabout_front_id": None, "roundabout_conflict_id": None,
            "planner_target_speed_mps": 16.3 / 3.6,
            "source": "existing_roundabout_text_log",
        }
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
