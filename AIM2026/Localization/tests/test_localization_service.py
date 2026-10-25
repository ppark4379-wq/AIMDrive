"""Shared-input service checks; no simulator, model, controller or sensor socket."""
import json
import math
from pathlib import Path
from dataclasses import replace
import time
from types import SimpleNamespace

import pytest
import yaml

from AIM2026.UDP.Common import ReceivedSample
from AIM2026.Localization.service import LocalizationService, create_localization


class Receiver:
    def __init__(self):
        self.rows = []
        self.port, self.packets, self.samples = 0, 0, 0
        self.closed = False

    def push(self, value, stamp):
        self.samples += 1
        self.packets += 1
        self.rows.append(ReceivedSample(value, stamp / 1e9, self.samples, value.get('sim_time'),
                                        received_ns=stamp))

    def recent(self):
        return tuple(self.rows[-200:])

    def diagnostics(self):
        return dict(packets=self.packets, samples=self.samples)

    def close(self):
        self.closed = True


def setup_service(monkeypatch, **options):
    now = [1_000_000_000]
    monkeypatch.setattr(time, 'monotonic_ns', lambda: now[0])
    settings = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config.yaml').read_text())
    receivers = {name: Receiver() for name in ('gps', 'imu', 'vehicle_status')}
    service = LocalizationService(receivers, settings, **options)
    return now, receivers, service


def sample(now, receivers, service, *, gps=False, speed=36., poison=0.):
    ns = now[0]
    yaw = -math.radians(service.hold.geometry.convergence_degrees((0., 0., 0.)))
    receivers['imu'].push(dict(sim_time=ns / 1e9, yaw=yaw,
        quaternion=(0., 0., math.sin(yaw / 2), math.cos(yaw / 2)), angular_velocity=(0., 0., 0.)), ns)
    receivers['vehicle_status'].push(dict(signed_speed=speed, sim_time=ns / 1e9,
        yaw=poison, pos_x=poison, pos_y=poison), ns)
    if gps:
        lon, lat = service.hold.geometry.inverse.transform(*service.basis['origin_xyz'][:2])
        receivers['gps'].push(dict(latitude=lat, longitude=lon, altitude=0.), ns)
    return service.latest_snapshot()


def test_gps_blackout_recovery_stale_and_socket_ownership(monkeypatch):
    now, receivers, service = setup_service(monkeypatch)
    fix, _, meta = sample(now, receivers, service, gps=True)
    assert fix is not None and meta['position_mode'] == 'gps'
    now[0] += 100_000_000
    sample(now, receivers, service)
    now[0] += 100_000_000
    fix, age, meta = sample(now, receivers, service)
    assert meta['position_mode'] == 'dead_reckoning'
    assert service.hold._grid(fix) == pytest.approx((2., 0.), abs=1e-7)
    assert age == 0.
    assert service.recent()[-1].meta['position_mode'] == 'dead_reckoning'
    now[0] += 100_000_000
    fix, _, meta = sample(now, receivers, service, gps=True)
    assert fix is not None and meta['position_mode'] == 'gps'
    now[0] += 300_000_000
    fix, _, meta = service.latest_snapshot()
    assert fix is None and meta['position_mode'] == 'unavailable'
    assert service.recent()[-1].value is None
    service.close()
    assert not any(receiver.closed for receiver in receivers.values())
    assert service.latest_snapshot()[2]['reason'] == 'localization_closed'


def test_status_gt_and_packet_yaw_cannot_change_position(monkeypatch):
    now, receivers, service = setup_service(monkeypatch)
    sample(now, receivers, service, gps=True, poison=99999.)
    for _ in range(4):
        now[0] += 100_000_000
        fix, _, meta = sample(now, receivers, service, poison=-99999.)
    assert meta['position_mode'] == 'dead_reckoning'
    # Meridian convergence varies slightly over these four metres.
    assert service.hold._grid(fix) == pytest.approx((4., 0.), abs=1e-6)
    assert set(service.status.speed.history[-1][0]) == {'signed_speed', 'sim_time'}


def test_reverse_uses_signed_velocity(monkeypatch):
    now, receivers, service = setup_service(monkeypatch)
    sample(now, receivers, service, gps=True, speed=-36.)
    for _ in range(3):
        now[0] += 100_000_000
        fix, _, meta = sample(now, receivers, service, speed=-36.)
    assert meta['position_mode'] == 'dead_reckoning'
    assert service.hold._grid(fix) == pytest.approx((-3., 0.), abs=1e-6)


def test_no_anchor_means_no_position(monkeypatch):
    now, receivers, service = setup_service(monkeypatch)
    fix, _, meta = sample(now, receivers, service)
    assert fix is None and meta['reason'] == 'no_gps_anchor'


def test_blackout_turn_uses_imu_and_signed_speed_without_wall_inputs(monkeypatch):
    now, receivers, service = setup_service(monkeypatch)
    sample(now, receivers, service, gps=True)
    origin = service.hold.core.anchor_xy
    angle = 0.
    expected = [0., 0.]
    # A quarter circle at 10 m/s. The service receives only GPS, IMU and speed;
    # the GPS receiver is left silent after its initial anchor.
    for step in range(1, 51):
        now[0] += 20_000_000
        next_angle = step * math.pi / 100
        expected[0] += .1 * (math.cos(angle) + math.cos(next_angle))
        expected[1] += .1 * (math.sin(angle) + math.sin(next_angle))
        convergence = service.hold.geometry.convergence_degrees((*service.hold.core.xy, 0.))
        yaw = next_angle - math.radians(convergence)
        ns = now[0]
        receivers['imu'].push(dict(sim_time=ns/1e9, yaw=yaw,
            quaternion=(0., 0., math.sin(yaw/2), math.cos(yaw/2)), angular_velocity=(0., 0., 0.)), ns)
        receivers['vehicle_status'].push(dict(signed_speed=36., sim_time=ns/1e9), ns)
        fix, _, meta = service.latest_snapshot()
        angle = next_angle
    assert meta['position_mode'] == 'dead_reckoning'
    xy = service.hold._grid(fix)
    assert [xy[k] - origin[k] for k in (0, 1)] == pytest.approx(expected, abs=1e-5)
    service.close()


def test_shared_recording_and_no_gt_in_inputs(monkeypatch, tmp_path):
    output = tmp_path / 'recording'
    now, receivers, service = setup_service(monkeypatch, record_dir=output)
    sample(now, receivers, service, gps=True, poison=999999.)
    service.close()
    rows = [json.loads(line) for line in (output / 'sensor_audit.jsonl').read_text().splitlines()]
    status = next(row for row in rows if row['sensor'] == 'status')
    assert set(status['value']) == {'signed_speed', 'sim_time'}
    assert status['observed_poll_id'] == 1
    assert json.loads((output / 'plan.json').read_text())['GT_driving_inputs'] is False
    assert json.loads((output / 'recording_status.json').read_text()) == dict(
        dropped_rows=0, error=None, writer_finished=True)
    pose = json.loads((output / 'poses.jsonl').read_text().splitlines()[0])
    assert pose['selected_samples']['gps']['position_mode'] == 'gps'
    assert pose['poll_id'] == 1
    assert pose['estimator_completed_ns'] >= pose['monotonic_ns']


def test_factory_disabled_and_rejects_uncalibrated_mounts():
    manager = SimpleNamespace(receivers={})
    assert create_localization(manager, {}) is None
    with pytest.raises(ValueError, match='boolean'):
        create_localization(manager, {'Localization': {'enabled': 'true'}})
    config = dict(Localization=dict(enabled=True), GPS=dict(pos=[1., 0., 0.]), IMU=dict(rot_deg=[0., 0., 0.]))
    with pytest.raises(ValueError, match='GPS.pos'):
        create_localization(manager, config)


def test_limit_refusal_blocks_new_frames_without_erasing_prior_frame(monkeypatch):
    from AIM2026.Inference.sensors.position import position_at_frame
    now, receivers, service = setup_service(monkeypatch)
    sample(now, receivers, service, gps=True)
    for _ in range(2):
        now[0] += 100_000_000
        sample(now, receivers, service)
    valid = service.recent()[-1]
    service.hold.core.max_distance_m = 2.1
    now[0] += 20_000_000
    fix, _, meta = sample(now, receivers, service)
    assert fix is None and meta['reason'] == 'hold_distance_limit'
    records = service.recent()
    assert records[-1].meta['received_ns'] < records[-1].received_ns
    assert records[-1].received_ns == now[0]
    new_frame = SimpleNamespace(received_at=now[0] / 1e9)
    selected = position_at_frame(records, new_frame, now[0] / 1e9)
    assert selected is None or selected.value is None
    old_frame = SimpleNamespace(received_at=valid.received_at)
    assert position_at_frame(records, old_frame, valid.source_time).value is not None
    service.close()
    assert not service.recent()


def test_recording_failure_does_not_stop_position_supply(monkeypatch, tmp_path):
    now, receivers, service = setup_service(monkeypatch, record_dir=tmp_path / 'recording')
    def fail(*args):
        raise OSError('test disk failure')
    monkeypatch.setattr(service.recorder, 'pose', fail)
    fix, _, meta = sample(now, receivers, service, gps=True)
    assert fix is not None and meta['position_mode'] == 'gps'
    assert service.recorder.error == 'test disk failure'
    now[0] += 100_000_000
    assert sample(now, receivers, service)[0] is not None
    service.close()


def test_source_epoch_discards_pre_reset_pose_history(monkeypatch):
    now, receivers, service = setup_service(monkeypatch)
    sample(now, receivers, service, gps=True)
    assert service.recent()
    now[0] += 100_000_000
    sample(now, receivers, service)
    for name in ('imu', 'vehicle_status'):
        receiver = receivers[name]
        receiver.rows = [replace(receiver.rows[-1], clock_epoch=1)]
    service.poll()
    assert all(row.clock_epoch == 1 for row in service.recent())


def test_repeated_queries_do_not_refresh_sensor_receipt(monkeypatch):
    now, receivers, service = setup_service(monkeypatch)
    sample(now, receivers, service, gps=True)
    first = service.recent()[-1]
    now[0] += 50_000_000
    fix, age, meta = service.latest_snapshot()
    assert fix is not None and age == .05
    assert meta['received_ns'] == first.received_ns
    assert service.recent()[-1].sequence == first.sequence
