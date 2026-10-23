"""Socket-free checks of shared receipts, DR provenance and perception timing."""
from collections import Counter, deque
import ctypes
import json
import threading
import time
import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from AIM2026.UDP.Common import ReceivedSample, _Receiver
from AIM2026.UDP.GPS import GpsReceiver
from AIM2026.UDP.IMU import ImuReceiver, ImuPacket
from AIM2026.UDP.VehicleStatus import VehicleStatusReceiver, CompetitionVehicleStatusPacket
from AIM2026.UDP.Manager import UDPManager
from AIM2026.UDP.Pose import PoseSender, decode_pose
from AIM2026.UDP.Detection import DetectionSender, EgoPose, decode, DR_POSITION
from AIM2026.Inference.main import VePublisher, CAMERAS
from AIM2026.Inference.sensors.position import position_at_frame, position_identity


def receiver():
    value = object.__new__(_Receiver)
    value._history = deque(maxlen=200)
    value._lock, value._counts_lock = threading.Lock(), threading.Lock()
    value.audit_counts = Counter()
    value._latest = None
    value.samples = value.packets = value.clock_epoch = 0
    value._running = True
    value._rx_overflow_option = None
    value._recv_bytes = 512
    return value


def test_exact_receipt_is_captured_before_decode(monkeypatch):
    value = receiver()
    value._socket = SimpleNamespace(recvfrom=lambda _: (b'packet', ('127.0.0.1', 1)))
    receipt = 123_456_789_012
    monkeypatch.setattr('AIM2026.UDP.Common.time.monotonic_ns', lambda: receipt)
    def parse(payload):
        assert payload == b'packet'
        value._running = False
        monkeypatch.setattr('AIM2026.UDP.Common.time.monotonic_ns', lambda: receipt + 90_000_000)
        return dict(sim_time=1., signed_speed=36.)
    value._consume = parse
    value._run()
    sample, = value.recent()
    assert sample.received_ns == receipt
    assert sample.received_at == receipt / 1e9
    assert sample.source_stamp_ns == 1_000_000_000


def test_history_deduplicates_source_resets_epoch_and_retains_zero_stamp():
    value = receiver()
    value._store(dict(sim_time=-1.), received_ns=1)
    assert value.samples == 0
    assert value.audit_counts['negative_source_time'] == 1
    for stamp in (1., 1., .5, 0., 0.):
        value._store(dict(sim_time=stamp), received_ns=1_000_000_000+value.samples)
    assert value.samples == 4
    assert [r.source_time for r in value.recent()] == [.5, 0., 0.]
    assert value.clock_epoch == 1
    assert value.audit_counts['duplicate_source_time'] == 1
    for index in range(250):
        value._store(dict(sim_time=0.), received_ns=2_000_000_000+index)
    assert len(value.recent()) == 200


def test_parsers_preserve_raw_integer_source_time_separately_from_legacy_rounding():
    for parser, packet in ((ImuReceiver, ImuPacket()),
                           (VehicleStatusReceiver, CompetitionVehicleStatusPacket())):
        packet.sec, packet.nsec = 1_700_000_000, 123456789
        packet.header = b'#IMUData$' if parser is ImuReceiver else b'#MoraiInfo$'
        packet.tail = b'\r\n'
        if parser is ImuReceiver:
            packet.ori_w = 1.
        decoded = object.__new__(parser)._consume(bytes(packet))
        assert decoded['source_stamp_ns'] == 1_700_000_000_123456789
        value = receiver()
        value._store(decoded, received_ns=9_000_000_000)
        assert value.recent()[0].source_stamp_ns == round(decoded['sim_time'] * 1e9)


def test_gps_keeps_valid_raw_nmea_and_original_positive_quality_contract():
    value = object.__new__(GpsReceiver)
    value._fix = dict(latitude=None, longitude=None, altitude=None)
    valid = b'$GPGGA,123519,3723.000,N,12637.000,E,6,08,0.9,10.0,M,0,M,,'
    assert value._consume(valid)['raw_nmea'] == valid.decode()
    assert value._consume(valid.replace(b',6,08', b',0,08')) is None
    assert value._consume(valid+b'*00') is None
    assert value._consume(b'$GPRMC,123519,V,3723.000,N,12637.000,E') is None


def test_manager_consumers_use_localization_without_replacing_or_double_closing_raw_gps():
    manager = object.__new__(UDPManager)
    raw = Mock()
    raw.snapshot.return_value = ('raw', 0., 1, 1.)
    raw.recent.return_value = ('raw',)
    raw.diagnostics.return_value = dict(packets=2, samples=2, counters={})
    local = Mock()
    local.snapshot.return_value = (None, float('inf'), 2, 2.)
    local.recent.return_value = ('DR history',)
    manager.receivers = {'gps': raw}
    manager.localization, manager.pose = local, None
    manager.cmd = manager.detection = None
    assert manager.snapshot()['gps'][0] is None  # No stale raw-GPS fallback.
    assert manager.recent()['gps'] == ('DR history',)
    assert manager.receivers['gps'] is raw
    manager.close()
    local.close.assert_called_once_with()
    raw.close.assert_called_once_with()


def test_manager_starts_one_service_over_existing_receivers(monkeypatch):
    manager_module = importlib.import_module('AIM2026.UDP.Manager')
    service_module = importlib.import_module('AIM2026.GPSJamming.gps_jamming.localization.service')
    geometry = dict(pos=[0.,0.,0.], rot_deg=[0.,0.,0.])
    config = dict(Camera={key:dict(geometry, width=640, height=480, fov=90., dest_port=5000+index)
                          for index,key in enumerate(('Cam_Front','Cam_Left','Cam_Right','Cam_Rear'))},
                  GPS=dict(geometry,dest_port=5004), IMU=dict(geometry,dest_port=5005),
                  LiDAR=dict(geometry,dest_port=5006), Ego=dict(vehicle_status_port=5007),
                  Localization=dict(enabled=True))
    factories = {}
    for name in ('CameraReceiver','GpsReceiver','ImuReceiver','LidarReceiver','VehicleStatusReceiver'):
        factories[name] = Mock(side_effect=lambda *args, **kwargs: Mock())
        monkeypatch.setattr(manager_module, name, factories[name])
    service = Mock()
    create = Mock(return_value=service)
    monkeypatch.setattr(service_module, 'create_localization', create)
    manager = UDPManager(config)
    monkeypatch.setattr(manager, '_probe_ports', lambda: None)
    manager.start()
    create.assert_called_once_with(manager, manager.config)
    service.start.assert_called_once_with()
    assert len(manager.receivers)==8
    for name in ('GpsReceiver','ImuReceiver','VehicleStatusReceiver'):
        assert factories[name].call_args.kwargs['history_size']==200
    manager.close()


def source(value, received_ns):
    record = ReceivedSample(value, received_ns/1e9, 5, value.get('sim_time'),
                            received_ns=received_ns, source_stamp_ns=round(value.get('sim_time', 0)*1e9))
    return SimpleNamespace(snapshot=lambda: (value, 0., 5, received_ns/1e9), recent=lambda: (record,))


def test_pose_sender_labels_dr_and_preserves_exact_identity_without_socket():
    now = time.monotonic_ns()
    identity = dict(received_ns=now-10_000_003, source_stamp_ns=123_000_000_000,
                    sample_id=8, position_mode='dead_reckoning', last_gps_received_ns=now-5_000_000_000)
    provider = SimpleNamespace(latest_snapshot=lambda: (dict(latitude=37., longitude=126.), .01, identity))
    sender = object.__new__(PoseSender)
    sender.gps, sender.max_age_s = provider, .15
    sender.imu = source(dict(sim_time=123., quaternion=(0.,0.,0.,1.)), now-8_000_003)
    sender.status = source(dict(sim_time=123., speed=10., competition_variant=True,
                               accel=0., brake=0., steer=0., gear=4), now-9_000_003)
    sender.projector = SimpleNamespace(project=lambda *_: dict(x=1.,y=2.,yaw_raw_rad=0.,yaw_grid_rad=0.))
    sender.sequence, sender.generation, sender.checkpoint_sha256 = 0, 'generation', 'weights'
    packet = sender.sample()
    assert packet['ego_position_source'] == DR_POSITION
    assert packet['selected_samples']['gps'] == identity
    assert packet['sensor_received_ns'] == identity['received_ns']
    decoded = decode_pose(json.dumps(packet).encode())
    assert decoded.ego_position_source == DR_POSITION
    assert decoded.position_identity == identity
    provider.latest_snapshot = lambda: (None, float('inf'), dict(identity, position_mode='unavailable'))
    assert sender.sample()['ready'] is False


def test_detection_wire_preserves_dr_identity_and_requires_it():
    sender = object.__new__(DetectionSender)
    sender.sequence, sender.generation, sender.checkpoint_sha256 = 0, 'generation', 'weights'
    sender.max_age_ns = 500_000_000
    sender._send = lambda packet: dict(packet, sent_ns=time.monotonic_ns())
    stamp = time.monotonic_ns()-10_000_000
    identity = dict(position_mode='dead_reckoning', received_ns=stamp)
    packet = sender.send([], EgoPose(0.,0.,0.,0.,1.), stamp/1e9,
                         ego_position_source=DR_POSITION, position_identity=identity)
    decoded = decode(json.dumps(packet).encode())
    assert decoded.ego_position_source == DR_POSITION
    assert decoded.position_identity == identity
    with pytest.raises(ValueError, match='provenance'):
        sender.send([], EgoPose(0.,0.,0.,0.,1.), stamp/1e9, ego_position_source=DR_POSITION)


def test_perception_selects_frame_time_position_not_later_latest_dr():
    frame = ReceivedSample('lidar', 20., 1, None)
    before = ReceivedSample({'dead_reckoning': True}, 19.99, 1, 100.,
                            dict(position_mode='dead_reckoning', received_ns=19_990_000_000))
    after = ReceivedSample({'dead_reckoning': True}, 20.02, 2, 100.02)
    selected = position_at_frame((before, after), frame, 100.)
    assert selected is before
    assert position_identity(selected)['position_mode'] == 'dead_reckoning'
    raw = ReceivedSample({'latitude': 37.}, 20.01, 3, None)
    assert position_at_frame((raw,), frame, 100.) is None


def test_publisher_rejects_unavailable_provider_even_when_receipt_is_fresh():
    publisher = VePublisher(None, None, None)
    invalid = ReceivedSample(None, time.monotonic(), 1, None,
                             dict(position_mode='unavailable'))
    assert not publisher._fresh(invalid, time.monotonic())


def test_publisher_uses_frame_time_dr_and_sends_its_provenance(monkeypatch):
    now=1000.
    monkeypatch.setattr('AIM2026.Inference.main.time.monotonic', lambda: now)
    records={name:(ReceivedSample('image',now-.01,1,None),) for name in CAMERAS}
    records['lidar']=(ReceivedSample('points',now-.01,1,None,
                                   dict(source_stamp_us_within_hour=100_000_000)),)
    records['imu']=(ReceivedSample(dict(quaternion=(0.,0.,0.,1.)),now-.02,1,100.),)
    records['vehicle_status']=(ReceivedSample(dict(speed=10.),now-.02,1,100.),)
    identity=dict(position_mode='dead_reckoning',received_ns=999_980_000_000)
    records['gps']=(ReceivedSample(dict(x=12.,dead_reckoning=True),now-.02,1,100.,identity),
                    ReceivedSample(dict(x=99.,dead_reckoning=True),now,2,100.02,identity))
    udp=SimpleNamespace(recent=lambda:records, send_detection=Mock(), send_detection_not_ready=Mock())
    runtime=SimpleNamespace(infer=Mock(return_value=SimpleNamespace(
        boxes=[],signal=None,native_signal=None,inference_ms=1.)))
    projector=SimpleNamespace(project=lambda gps,imu:dict(x=gps['x'],y=0.,yaw_raw_rad=0.,yaw_grid_rad=0.))
    publisher=VePublisher(udp,runtime,projector)
    assert publisher.step()=='ready'
    assert udp.send_detection.call_args.args[1].x==12.
    assert udp.send_detection.call_args.kwargs['ego_position_source']==DR_POSITION
    assert udp.send_detection.call_args.kwargs['position_identity']['received_ns']==identity['received_ns']
    assert udp.send_detection.call_args.args[2]==now-.02  # Include IMU in freshness bound.


@pytest.mark.parametrize('source_time', [10.21, None])
def test_hold_failure_is_an_availability_barrier_in_both_time_bases(source_time):
    before = ReceivedSample(dict(dead_reckoning=True), 1.20, 1, 10.20,
                            dict(position_mode='dead_reckoning'))
    failed = ReceivedSample(None, 1.22, 2, None,
                            dict(position_mode='unavailable', reason='hold_distance_limit',
                                 received_ns=1_000_000_000, availability_changed_ns=1_220_000_000))
    records = (before, failed)
    past_frame = ReceivedSample('frame', 1.21, 3, source_time)
    current_frame = ReceivedSample('frame', 1.23, 4, source_time)
    assert position_at_frame(records, past_frame, source_time) is before
    assert position_at_frame(records, current_frame, source_time) is failed
    recovery = ReceivedSample(dict(latitude=37.), 1.24, 5, None, dict(position_mode='gps'))
    recovered_frame = ReceivedSample('frame', 1.25, 6, None)
    assert position_at_frame((*records, recovery), recovered_frame) is recovery
