import json
import time
from types import SimpleNamespace

from AIM2026.UDP.Detection import Box
from AIM2026.utils.perception_evidence import GroundTruthRecorder, detection_evidence


def test_publisher_records_sequence_provenance_and_source_timestamps():
    import numpy as np
    from unittest.mock import Mock
    from AIM2026.Inference.main import VePublisher, CAMERAS
    from AIM2026.UDP.Common import ReceivedSample
    chosen = {name: ReceivedSample(b'jpeg', 10, 1, None) for name in CAMERAS}
    chosen.update(lidar=ReceivedSample(np.zeros((3,4)), 10, 1, None, source_stamp_ns=123),
        gps=ReceivedSample({}, 10, 1, None, dict(position_mode='gps')),
        imu=ReceivedSample({}, 9.99, 1, None),
        vehicle_status=ReceivedSample(dict(speed=5), 10, 1, None),
        lidar_observation_time=10.)
    udp = SimpleNamespace(recent=lambda: dict(lidar=[chosen['lidar']]), send_detection=Mock(),
                          detection=SimpleNamespace(sequence=42, last_ready=True))
    ve, extra = Box(1,2,0,2,4,5), Box(3,4,0,3,12,None)
    result = SimpleNamespace(boxes=(ve,extra), supplement_indices=(1,), signal=None,
        native_signal=None, inference_ms=1, raw_head=np.zeros((2500,8)), counts={})
    runtime = SimpleNamespace(infer=lambda *args, **kwargs: result)
    publisher = VePublisher(udp, runtime, SimpleNamespace(project=lambda *args:
        dict(x=0,y=0,yaw_raw_rad=0,yaw_grid_rad=0)))
    publisher._inputs = lambda *args: (chosen, 'test')
    publisher.collision_trace = Mock()
    assert publisher.step() == 'ready'
    row, files = publisher.collision_trace.submit.call_args.args
    assert row['sequence'] == 42
    assert row['input_source_stamp_ns']['lidar'] == 123
    assert row['ve_boxes'] == [ve.to_wire()]
    assert row['supplement_boxes'] == [extra.to_wire()]
    assert row['delivered'] and len(files) == 6
    assert row['object_received_ns'] == 10_000_000_000
    assert row['lidar_observation_time'] == 10.


def test_box_provenance_preserves_originals_and_empty_frames():
    ve = Box(1, 2, 0, 2, 4, 5)
    extra = Box(3, 4, 0, 3, 12, None)
    result = SimpleNamespace(boxes=(ve, extra), supplement_indices=(1,), lidar_candidates=({'confirmed': True},))
    evidence = detection_evidence(result)
    assert evidence['ve_boxes'] == [ve.to_wire()]
    assert evidence['supplement_boxes'] == [extra.to_wire()]
    assert evidence['final_boxes'] == [ve.to_wire(), extra.to_wire()]
    assert result.boxes == (ve, extra)
    empty = detection_evidence(SimpleNamespace(boxes=()))
    assert empty['final_boxes'] == empty['ve_boxes'] == empty['supplement_boxes'] == []


def test_gt_failure_is_recorded_and_does_not_escape_thread(tmp_path, monkeypatch):
    import sys
    from AIM2026.utils.perception_evidence import ROOT
    sys.path.insert(0, str(ROOT/'MoraiDrive/data_collection/third_party/grpc_inha_univ/src/proto'))
    from morai.actor import actor_pb2_grpc
    def fail(*args, **kwargs):
        raise RuntimeError('test simulator unavailable')
    monkeypatch.setattr(actor_pb2_grpc, 'ActorStub', lambda channel: SimpleNamespace(GetAllActorsState=fail))
    recorder = GroundTruthRecorder(tmp_path, {}, 'test', hz=100)
    deadline = time.monotonic()+2
    while recorder.writer.stats['enqueued'] == 0 and time.monotonic()<deadline:
        time.sleep(.01)
    recorder.close()
    rows = [json.loads(line) for line in (tmp_path/'gt_actors.jsonl').read_text().splitlines()]
    assert rows and rows[0]['error'] == 'test simulator unavailable'
    assert rows[0]['received_ns'] >= rows[0]['request_ns']
    assert not recorder.thread.is_alive()
    status = json.loads((tmp_path/'gt_actors_status.json').read_text())
    assert status['written'] == len(rows)
    assert status['errors'] == status['dropped'] == 0
