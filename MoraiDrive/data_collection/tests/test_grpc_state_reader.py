import json
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grpc_ctrl.morai_state_reader import (  # noqa: E402
    GrpcLabelSnapshot,
    GrpcStateHistory,
    MoraiGrpcStateReader,
    host_candidates,
    label_snapshot_from_actor_states,
    snapshot_matches_ego,
)
from grpc_ctrl.label_spool import (  # noqa: E402
    GrpcLabelCaptureSpool,
    GrpcLabelStore,
)


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def actor(actor_id, object_type, x, y, yaw, velocity_x=0.0,
          global_velocity=None,
          extent=(1.0, 2.0, 0.5), traffic_id="", traffic_color=-2):
    result = ns(
        actor_info=ns(id=ns(value=actor_id), object_type=object_type),
        transform=ns(
            location=ns(x=x, y=y, z=1.0),
            rotation=ns(x=0.0, y=0.0, z=yaw),
        ),
        velocity=ns(x=velocity_x, y=0.0, z=0.0),
        bounding_box=ns(
            extent=ns(x=extent[0], y=extent[1], z=extent[2]),
            location=ns(x=x + 100.0, y=y + 100.0, z=0.0),
            rotation=ns(x=0.0, y=0.0, z=yaw + 100.0),
        ),
        vehicle_state=ns(
            tl_id=ns(value=traffic_id), tl_color=traffic_color
        ),
    )
    if global_velocity is not None:
        result.global_velocity = ns(
            x=global_velocity[0], y=global_velocity[1], z=global_velocity[2]
        )
    return result


def test_actor_states_match_ros_dataset_conventions():
    states = ns(
        states=[
            actor("ped", 2, 3.0, 4.0, 0.0),
            actor("9", 3, 5.0, 6.0, 0.0),
            actor("Ego", 1, 1.0, 2.0, 90.0,
                  traffic_id="TL-A", traffic_color=1),
            actor("7", 1, 10.0, 20.0, 90.0, velocity_x=36.0,
                  global_velocity=(6.0, 8.0, 0.0),
                  extent=(1.1, 2.4, 0.8)),
        ],
        sync_mode_tick_count=123,
        timestamp=2460000,
    )

    snapshot = label_snapshot_from_actor_states(states)

    assert snapshot.tick == 123
    assert [obj.obj_type for obj in snapshot.objects] == [
        "vehicle", "pedestrian", "static"
    ]
    vehicle = snapshot.objects[0]
    assert vehicle.obj_id == 7
    assert vehicle.x == 10.0 and vehicle.y == 20.0
    assert vehicle.heading_deg == 90.0
    assert vehicle.width == 2.2 and vehicle.length == 4.8
    assert vehicle.height == 1.6
    assert vehicle.vel_x == 6.0
    assert vehicle.vel_y == 8.0
    assert snapshot.traffic_lights[0].tl_id == "TL-A"
    assert snapshot.traffic_lights[0].state == 1
    assert snapshot.objects[1].obj_id == 2045086689  # stable CRC32("ped")


def test_history_requires_exact_native_tick_and_matching_ego_pose():
    history = GrpcStateHistory(object())
    snapshot = GrpcLabelSnapshot(
        tick=10,
        simulator_timestamp_raw=200,
        ego_x=1.0,
        ego_y=2.0,
        ego_heading_deg=90.0,
        objects=[],
        traffic_lights=[],
    )
    history._store(snapshot)
    record = {
        "native_sync_frame": 10,
        "ego_state_for_frame": {
            "position": {"x": 1.0, "y": 2.0},
            "heading": 90.0,
        },
    }

    selected = history.wait_for_record(record, 0.02)

    assert selected is snapshot
    assert history.is_aligned


def test_history_does_not_infer_tick_offset_from_stationary_ego():
    history = GrpcStateHistory(object())
    history._store(GrpcLabelSnapshot(
        tick=13,
        simulator_timestamp_raw=200,
        ego_x=1.0,
        ego_y=2.0,
        ego_heading_deg=90.0,
        objects=[],
        traffic_lights=[],
    ))
    record = {
        "native_sync_frame": 10,
        "ego_state_for_frame": {
            "position": {"x": 1.0, "y": 2.0},
            "heading": 90.0,
        },
    }

    assert history.wait_for_record(record, 0.02) is None
    assert not history.is_aligned


def test_old_actor_proto_projects_local_longitudinal_velocity():
    state = actor("7", 1, 10.0, 20.0, 90.0, velocity_x=36.0)
    snapshot = label_snapshot_from_actor_states(ns(
        states=[actor("Ego", 1, 1.0, 2.0, 0.0), state],
        sync_mode_tick_count=123,
        timestamp=2460000,
    ))

    assert abs(snapshot.objects[0].vel_x) < 1e-9
    assert math.isclose(snapshot.objects[0].vel_y, 10.0)


def _empty_snapshot(tick):
    return GrpcLabelSnapshot(
        tick=tick,
        simulator_timestamp_raw=tick * 2,
        ego_x=1.0,
        ego_y=2.0,
        ego_heading_deg=90.0,
        objects=[],
        traffic_lights=[],
    )


def test_required_fetch_retries_stale_response_until_exact_tick():
    reader = object.__new__(MoraiGrpcStateReader)
    reader.retry_interval_sec = 0.001
    reader.last_error = None
    responses = iter([_empty_snapshot(9), _empty_snapshot(10)])
    reader.fetch = lambda: next(responses)

    assert reader.fetch_required(0.1, expected_tick=10).tick == 10


def test_required_fetch_rejects_response_past_expected_tick():
    reader = object.__new__(MoraiGrpcStateReader)
    reader.retry_interval_sec = 0.001
    reader.last_error = None
    reader.fetch = lambda: _empty_snapshot(11)

    try:
        reader.fetch_required(0.1, expected_tick=10)
    except RuntimeError as error:
        assert "passed required tick" in str(error)
    else:
        raise AssertionError("future gRPC tick must not bind to an older frame")


def test_snapshot_pose_guard_rejects_a_different_sync_frame():
    snapshot = GrpcLabelSnapshot(
        tick=10,
        simulator_timestamp_raw=20,
        ego_x=1.0,
        ego_y=2.0,
        ego_heading_deg=359.0,
        objects=[],
        traffic_lights=[],
    )

    assert snapshot_matches_ego(snapshot, 1.1, 2.1, 1.0)
    assert not snapshot_matches_ego(snapshot, 2.0, 2.0, 1.0)


def test_explicit_host_has_no_implicit_fallback():
    assert host_candidates("10.0.0.7") == ["10.0.0.7"]


def _capture_record(frame):
    ego = {
        "position": {"x": 1.0, "y": 2.0, "z": 0.0},
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "heading": 90.0,
    }
    return {
        "schema_version": 1,
        "session_id": "spool-test",
        "scenario_id": "scenario",
        "native_sync_frame": frame,
        "request_frame": frame,
        "response_frame": frame + 1,
        "capture_requested": True,
        "capture_stride": 5,
        "ego_state_for_frame": ego,
        "ego_state_source_frame": frame,
        "next_ego_state": ego,
        "next_ego_state_source_frame": frame + 1,
        "applied_control": {
            "accel": 0.0, "brake": 0.0, "front_steer": 0.0
        },
        "control_source_sequence": 1,
        "control_reused": False,
        "control_age_frames": 0,
        "control_mode": "fresh",
        "expected_sensor_files": {},
    }


def test_local_label_store_survives_memory_history_eviction():
    with tempfile.TemporaryDirectory() as root:
        path = str(Path(root) / "labels.sqlite3")
        original = _empty_snapshot(25)
        store = GrpcLabelStore(path)
        assert store.put(original)
        store.close()

        reopened = GrpcLabelStore(path)
        restored = reopened.get(25)
        reopened.close()
        assert restored == original


def test_capture_spool_tails_manifest_independently_of_dataset_consumer():
    with tempfile.TemporaryDirectory() as root:
        manifest_path = str(Path(root) / "capture_manifest.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as manifest:
            manifest.write(json.dumps(_capture_record(30)) + "\n")
        snapshot = _empty_snapshot(30)

        class FakeHistory:
            is_aligned = False

            def start(self):
                pass

            def wait_for_record(self, record, _timeout):
                if int(record["native_sync_frame"]) == 30:
                    self.is_aligned = True
                    return snapshot
                return None

            def close(self):
                pass

        spool = GrpcLabelCaptureSpool(
            FakeHistory(),
            manifest_path,
            str(Path(root) / "labels.sqlite3"),
            manifest_poll_interval_sec=0.005,
            required_timeout_sec=0.05,
        )
        spool.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not spool.is_aligned:
            time.sleep(0.005)

        # Simulate the main dataset consumer arriving later, after the live
        # history would normally have evicted this tick.
        spool.history.wait_for_record = lambda _record, _timeout: None
        restored = spool.wait_for_record(_capture_record(30), 0.1)
        spool.close()
        assert restored == snapshot
