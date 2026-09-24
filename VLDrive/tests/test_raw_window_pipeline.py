"""Input scheduling preserves real observations and bounded framewise work."""
import threading
import copy
import queue

import pytest
import torch

from inference.input_windows import FrameGrid, RawWindowHistory, WindowInputProducer
from inference.timed_history import TimedTokenHistory


@pytest.mark.parametrize('key,value', [
    ('history_max_span_s', float('nan')), ('history_max_span_s', 2.41),
    ('history_max_span_s', 1.9), ('history_max_adjacent_gap_s', .6),
    ('history_max_adjacent_gap_s', .1), ('pin_encoder_inputs', 1),
    ('encoder_stream_priority', True), ('encoder_stream_priority', -2),
    ('encoder_batch_size', 3), ('encoder_batch_size', True),
    ('encoder_batch_wait_s', .5), ('encoder_batch_wait_s', float('inf')),
])
def test_history_and_transfer_configuration_is_bounded(key, value):
    from inference.realtime_vldrive import validate_config
    from inference.vldrive_runtime import ROOT, read_json
    cfg = copy.deepcopy(read_json(ROOT/'configs/vldrive_realtime.json'))
    cfg['runtime'][key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(cfg)


def observation(identity, stamp, *, auxiliary=None, sim_time=None):
    auxiliary = identity if auxiliary is None else auxiliary
    ids = dict(lidar=identity, front=auxiliary, left=auxiliary, right=auxiliary,
               rear=auxiliary, vehicle_status=auxiliary)
    meta = dict(sample_ids=ids, received_at={key: stamp for key in ids}, missing=[],
                front_jpeg=b'jpeg', sim_time=stamp if sim_time is None else sim_time)
    return dict(meta=meta), dict(navigation=dict(valid=True, instruction=dict(text=f'Frame {identity}.')),
                                 front_jpeg=b'jpeg', sim_time=meta['sim_time'])


def add(history, identity, stamp, **kwargs):
    return history.append(*observation(identity, stamp, **kwargs))


class SnapshotSource:
    def __init__(self):
        self.prepared_ids = []

    def prepare_snapshot(self, raw, *, device):
        assert device == 'cpu'
        identity = raw['meta']['sample_ids']['lidar']
        self.prepared_ids.append(identity)
        return dict(identity=torch.tensor(float(identity))), raw['meta']


def producer_for(history, tmp_path, *, source=None, stop=None, navigation=None):
    source = SnapshotSource() if source is None else source
    stop = threading.Event() if stop is None else stop
    cfg = dict(runtime=dict(device='cpu', max_age_s=.3, max_stream_skew_s=.15,
                            capture_hz=5., target_hz=5., encoder_max_hz=10.,
                            history_max_span_s=2.4, history_max_adjacent_gap_s=.46),
               visualization=dict(max_prediction_age_s=3.))
    state = dict(rejected_frames=0, discarded_skew=0, capture_generation=0, error=None)

    def encode_identity(_encoder, inputs):
        return torch.full((106, 256), inputs['identity'].item()), .25

    producer = WindowInputProducer(cfg, source, history, state, stop, tmp_path,
        navigation, object(), None, encode_identity,
        lambda *_args: dict(valid=True, instruction=dict(text='Straight.')))
    return producer, source


def finish_feature(producer, row, *, now=None):
    feature = torch.full((106, 256), float(row['ids']['lidar']))
    return producer.append_encoded(row, feature, None, 165., preprocessing_ms=27.,
        processing_ms=192., queue_wait_ms=0., now=row['stamp']+.2 if now is None else now)


def scheduled_rows(history, count=12, start=100.):
    grid = FrameGrid(.2, .06)
    rows = []
    for identity in range(count):
        add(history, identity, start+identity*.2)
        row = grid.select(history.raw_snapshot(), start+identity*.2+.06, .3)
        assert row is not None
        rows.append(row)
    return rows


def test_framewise_features_reuse_ten_and_preserve_original_context(tmp_path):
    history = RawWindowHistory()
    producer, source = producer_for(history, tmp_path)
    rows = scheduled_rows(history)
    for row in rows[:10]:
        assert not finish_feature(producer, row)
    assert finish_feature(producer, rows[10])
    _, first = history.snapshot(materialize=False)
    assert finish_feature(producer, rows[11])
    tensor, second = history.snapshot()
    assert tensor.shape == (1, 11, 106, 256)
    assert tensor[0, :, 0, 0].tolist() == list(range(1, 12))
    assert all(second[i]['feature'] is first[i+1]['feature'] for i in range(10))
    assert all(second[i]['stamp'] == rows[i+1]['stamp'] for i in range(11))
    assert all(second[i]['ids'] == rows[i+1]['ids'] for i in range(11))
    assert second[-1]['context']['navigation']['instruction']['text'] == 'Frame 11.'
    assert second[-1]['context']['new_vision_frames'] == 1
    assert producer.counts['prepared_windows'] == 2
    assert len(producer.features) == 11
    assert source.prepared_ids == []


def test_short_slot_gap_keeps_real_features_without_rewarming(tmp_path):
    history = RawWindowHistory()
    producer, _source = producer_for(history, tmp_path)
    rows = scheduled_rows(history, count=24)
    for row in rows[:11]:
        finish_feature(producer, row)
    assert producer.counts['prepared_windows'] == 1
    previous = history.snapshot(materialize=False)[1]
    assert finish_feature(producer, rows[12])
    selected = history.snapshot(materialize=False)[1]
    assert [row['slot'] for row in selected] == list(range(1, 11))+[12]
    assert all(selected[i]['feature'] is previous[i+1]['feature'] for i in range(10))
    assert selected[-1]['stamp']-selected[0]['stamp'] == pytest.approx(2.2)
    assert selected[-1]['context']['history_temporally_extended']
    assert producer.counts['encoded_history_gaps'] == 1
    assert producer.counts['feature_history_resets'] == 0
    assert not finish_feature(producer, rows[12])  # no repeated anchor
    for row in rows[13:23]:
        assert finish_feature(producer, row)
    assert not history.snapshot(materialize=False)[1][-1]['context']['history_temporally_extended']


def test_multiple_short_gaps_cannot_stretch_history_past_configured_bound(tmp_path):
    history = RawWindowHistory()
    producer, _source = producer_for(history, tmp_path)
    rows = scheduled_rows(history, count=20)
    for row in rows[:11]:
        finish_feature(producer, row)
    assert finish_feature(producer, rows[12])  # 2.2 s
    assert finish_feature(producer, rows[14])  # 2.4 s
    assert not finish_feature(producer, rows[16])  # would span 2.6 s
    assert producer.counts['temporal_bound_rejections'] > 0
    assert producer.state['history_selection']['selected'] < 11
    assert producer.features[-1]['stamp']-producer.features[0]['stamp'] <= 2.4+1e-8


def test_large_encoded_gap_requires_new_history(tmp_path):
    history = RawWindowHistory()
    producer, _source = producer_for(history, tmp_path)
    rows = scheduled_rows(history, count=24)
    for row in rows[:11]:
        finish_feature(producer, row)
    for row in rows[13:23]:
        assert not finish_feature(producer, row)  # 0.6 s discontinuity
    assert producer.counts['feature_history_resets'] == 1
    assert finish_feature(producer, rows[23])


@pytest.mark.parametrize('stamps,expected', [
    ([99.55,99.80],[0,1]),  # first is historical, newest is fresh
    ([99.45,99.80],[1]),    # first is older than batch bound
    ([99.55,99.69],[]),     # stale newest cannot lend freshness to first
])
def test_two_frame_batch_freshness_preserves_actual_times(tmp_path,monkeypatch,stamps,expected):
    import inference.input_windows as windows
    history = RawWindowHistory()
    producer, _ = producer_for(history, tmp_path)
    producer.batch_size = 2
    producer.prepared = queue.Queue(maxsize=2)
    monkeypatch.setattr(windows.time, 'monotonic', lambda:100.)
    for identity,stamp in enumerate(stamps):
        row=dict(stamp=stamp,ids=dict(lidar=identity),generation=0)
        producer.prepared.put(dict(row=row,inputs=dict(num_points=torch.tensor([identity+10]),
            rgb_front=torch.full((1,3,2,2),float(identity))),prepared_at=stamp+.03))
    items = producer.take_batch()
    assert [item['row']['ids']['lidar'] for item in items] == expected
    if items:
        tensors=producer.transfer_batch(items)
        assert tensors['rgb_front'][:,0,0,0].tolist() == expected
        assert tensors['num_points'].device.type == 'cpu'
        assert tensors['num_points'].tolist() == [i+10 for i in expected]
        assert [item['row']['stamp'] for item in items] == [stamps[i] for i in expected]


def test_batch_timeout_keeps_single_frame_without_duplication(tmp_path):
    import time
    history = RawWindowHistory()
    producer, _ = producer_for(history, tmp_path)
    producer.batch_size = 2
    producer.cfg['runtime']['encoder_batch_wait_s'] = .01
    stamp=time.monotonic()
    item=dict(row=dict(stamp=stamp,ids=dict(lidar=1),generation=0))
    producer.prepared.put(item)
    assert producer.take_batch() == [item]


def test_batched_runtime_keeps_per_frame_outputs_and_quantization():
    from inference.vldrive_runtime import encode_frames
    class Encoder(torch.nn.Module):
        def forward(self,inputs):
            return inputs['rgb_front'][:,None,None].expand(-1,106,256)*.123456
    features,_=encode_frames(Encoder().eval(),dict(rgb_front=torch.tensor([1.,2.])))
    assert features.shape == (2,106,256)
    assert torch.equal(features[:,0,0],(torch.tensor([1.,2.])*.123456).half().float())
    with pytest.raises(ValueError,match='one or two'):
        encode_frames(Encoder().eval(),dict(rgb_front=torch.ones(3)))


def test_batched_publish_exposes_only_newest_completed_anchor(tmp_path):
    history=RawWindowHistory()
    producer,_=producer_for(history,tmp_path)
    rows=scheduled_rows(history,count=13)
    for row in rows[:11]:finish_feature(producer,row)
    for i in (11,12):
        producer.append_encoded(rows[i],torch.full((106,256),float(i)),None,210.,
            preprocessing_ms=20.,processing_ms=250.,queue_wait_ms=0.,now=rows[12]['stamp']+.3,
            batch_size=2,publish=i==12,batch_preprocessing_ms=40.)
        assert history.snapshot(materialize=False)[1][-1]['ids']['lidar'] == (10 if i==11 else 12)
    context=history.snapshot(materialize=False)[1][-1]['context']
    assert context['new_vision_frames'] == 2 and context['window_preprocessing_ms'] == 40.
    assert producer.counts['prepared_windows'] == 2
    assert producer.counts['feature_cache_hits'] == 10+9


def test_generation_reset_rejects_old_feature_and_old_publish(tmp_path):
    history = RawWindowHistory()
    producer, _source = producer_for(history, tmp_path)
    rows = scheduled_rows(history, count=11)
    for row in rows:
        finish_feature(producer, row)
    _, first = history.snapshot(materialize=False)
    history.clear()
    assert history.snapshot() == (None, [])
    assert not history.publish(first)
    assert not finish_feature(producer, rows[-1])
    new_rows = scheduled_rows(history, count=11, start=200.)
    for row in new_rows[:10]:
        assert not finish_feature(producer, row)
    assert finish_feature(producer, new_rows[-1])
    assert all(row['generation'] == history.resets for row in producer.features)
    assert history.snapshot(materialize=False)[1][-1]['stamp'] == pytest.approx(202.)


def test_auxiliary_reuse_preserves_identity_but_lidar_must_advance():
    history = RawWindowHistory()
    assert add(history, 1, 100., auxiliary=7)
    assert add(history, 2, 100.1, auxiliary=7)
    assert history.frames[-1]['ids']['front'] == 7
    assert not add(history, 2, 100.2, auxiliary=8)
    assert not add(history, 3, 100.2, auxiliary=6)
    assert not add(history, 3, 100., auxiliary=8)
    assert add(history, 3, 100.2, auxiliary=8)


def test_grid_waits_for_nearest_real_neighbor_and_does_not_duplicate():
    history = RawWindowHistory()
    grid = FrameGrid(.2, .06)
    add(history, 0, 100.)
    assert grid.select(history.raw_snapshot(), 100.059, .3) is None
    first = grid.select(history.raw_snapshot(), 100.06, .3)
    assert first['stamp'] == first['target_stamp'] == 100.
    add(history, 1, 100.15)
    assert grid.select(history.raw_snapshot(), 100.20, .3) is None
    add(history, 2, 100.225)
    second = grid.select(history.raw_snapshot(), 100.26, .3)
    assert second['ids']['lidar'] == 2
    assert second['stamp'] == 100.225 and second['target_stamp'] == 100.2
    assert grid.select(history.raw_snapshot(), 100.27, .3) is None
    assert grid.select(history.raw_snapshot(), 100.46, .3) is None
    assert grid.missed == 1


def test_grid_reports_missing_slots_and_expires_backlog():
    history = RawWindowHistory()
    grid = FrameGrid(.2, .06)
    add(history, 0, 100.)
    assert grid.select(history.raw_snapshot(), 100.06, .3)
    add(history, 1, 100.30)
    assert grid.select(history.raw_snapshot(), 100.31, .3) is None
    assert grid.missed == 1
    for identity in range(2, 9):
        add(history, identity, 100.3+(identity-1)*.1)
    row = grid.select(history.raw_snapshot(), 101., .3)
    assert grid.expired > 0
    assert row is None or 0 <= 101.-row['stamp'] <= .3
    assert row is None or row['slot'] >= 4


def test_grid_expiry_uses_real_sample_age_not_target_age():
    history = RawWindowHistory()
    grid = FrameGrid(.2, .06)
    add(history, 0, 9.8)
    assert grid.select(history.raw_snapshot(), 9.8, .3) is None
    add(history, 1, 9.85)
    row = grid.select(history.raw_snapshot(), 10.11, .3)
    assert row is not None
    assert row['target_stamp'] == pytest.approx(9.8)
    assert row['stamp'] == pytest.approx(9.85)
    assert row['slot'] == 0


def test_observed_six_ms_grid_miss_is_accepted_without_changing_timestamp():
    # Live MORAI trace: nearest valid sample was 60.6..66.1 ms from a slot.
    old=FrameGrid(.2,.06)
    current=FrameGrid(.2,.07)
    history=RawWindowHistory(tolerance_s=.07)
    add(history,0,100.)
    assert old.select(history.raw_snapshot(),100.07,.3)
    assert current.select(history.raw_snapshot(),100.07,.3)
    add(history,1,100.134)
    assert old.select(history.raw_snapshot(),100.27,.3) is None
    selected=current.select(history.raw_snapshot(),100.27,.3)
    assert selected['stamp'] == 100.134 and selected['target_stamp'] == 100.2
    assert selected['ids']['lidar'] == 1


def test_long_gap_and_storage_limits(tmp_path):
    history = RawWindowHistory(capacity=16)
    producer, _source = producer_for(history, tmp_path)
    rows = scheduled_rows(history, count=11)
    for row in rows:
        finish_feature(producer, row)
    previous = history.snapshot(materialize=False)[1]
    assert add(history, 11, 103.1)
    assert history.resets == 1 and len(history.frames) == 1
    assert history.snapshot() == (None, [])
    assert not history.publish(previous)
    for identity in range(12, 100):
        add(history, identity, 103.1+(identity-11)*.02)
        assert len(history.frames) <= 16
    assert producer.prepared.maxsize == 1
    assert producer.features.maxlen == 11


def test_grid_contract_is_nominal_time_not_old_actual_anchor_constraint(tmp_path):
    history = RawWindowHistory()
    producer, _source = producer_for(history, tmp_path)
    selected = []
    next_raw = 0
    old_constraint_violations = 0
    for slot in range(60):
        now = 100. + slot*.2 + .06
        while 100.+next_raw*.1125 <= now+1e-8:
            add(history, next_raw, 100.+next_raw*.1125)
            next_raw += 1
        row = producer.grid.select(history.raw_snapshot(), now, .3)
        assert row is not None
        assert row['stamp'] == pytest.approx(100.+row['ids']['lidar']*.1125)
        assert abs(row['stamp']-row['target_stamp']) <= .06+1e-8
        selected.append(row)
        if finish_feature(producer, row, now=now+.192):
            rows = history.snapshot(materialize=False)[1]
            span = rows[-1]['stamp']-rows[0]['stamp']
            assert 1.88-1e-8 <= span <= 2.12+1e-8
            anchor_error = max(abs(r['stamp']-(rows[-1]['stamp']-(10-i)*.2))
                               for i,r in enumerate(rows))
            assert anchor_error <= .12+1e-8
            old_constraint_violations += anchor_error > .06+1e-8
    assert len({row['ids']['lidar'] for row in selected}) == len(selected)
    assert old_constraint_violations > 0
    assert producer.counts['prepared_windows'] == 50


def simulate_framewise(tmp_path, *, duration=20., raw_interval=.1125,
                       preprocessing_s=.027, vision_s=.165, stage2_s=.274):
    """Virtual 1-ms clock using the real grid, queues and append/publish methods.

    CPU, VE and Stage2 are independent servers with measured-cost inputs. This
    validates scheduling only; it cannot model contention on a shared GPU.
    """
    history = RawWindowHistory()
    producer, source = producer_for(history, tmp_path)
    next_raw = 0
    preprocessing = vision = language = None
    last_output_anchor = None
    completed, emitted_windows, processing_counts = [], [], []
    max_queue = max_features = max_raw = 0
    for tick in range(round(duration*1000)+1):
        now = 100.+tick/1000.
        while next_raw*raw_interval <= tick/1000.+1e-8:
            add(history, next_raw, 100.+next_raw*raw_interval)
            producer.counts['raw_frames'] += 1
            next_raw += 1
        if preprocessing is not None and preprocessing['finish'] <= now+1e-8:
            item = preprocessing['item']
            if producer.prepared.full():
                producer.prepared.get_nowait()
                producer.counts['preprocessing_queue_drops'] += 1
            producer.prepared.put_nowait(item)
            preprocessing = None
        if vision is not None and vision['finish'] <= now+1e-8:
            item = vision['item']
            feature, _elapsed = producer.encode_frame(producer.encoder, item['inputs'])
            producer.counts['encoded_frames'] += 1
            before = producer.counts['prepared_windows']
            if producer.append_encoded(item['row'], feature, None, vision_s*1000,
                    preprocessing_ms=preprocessing_s*1000,
                    processing_ms=(preprocessing_s+vision_s)*1000,
                    queue_wait_ms=vision['wait']*1000, now=now):
                rows = history.snapshot(materialize=False)[1]
                emitted_windows.append((now, rows[-1]['stamp']))
                processing_counts.append(producer.counts['prepared_windows']-before)
            vision = None
        if language is not None and language['finish'] <= now+1e-8:
            completed.append((now, language['anchor']))
            language = None
        if preprocessing is None and tick % 5 == 0:
            row = producer.grid.select(history.raw_snapshot(), now, .3)
            if row is not None:
                inputs, _meta = source.prepare_snapshot(row['raw'], device='cpu')
                producer.counts['preprocessing_frames'] += 1
                preprocessing = dict(finish=now+preprocessing_s,
                    item=dict(row=row, inputs=inputs, prepared_at=now+preprocessing_s))
        if vision is None and not producer.prepared.empty():
            item = producer.prepared.get_nowait()
            if now-item['row']['stamp'] > .3:
                producer.counts['stale_encoder_frames'] += 1
            else:
                vision = dict(item=item, finish=now+vision_s, wait=now-item['prepared_at'])
        if language is None:
            rows = history.snapshot(materialize=False)[1]
            if rows and rows[-1]['stamp'] != last_output_anchor:
                last_output_anchor = rows[-1]['stamp']
                language = dict(finish=now+stage2_s, anchor=last_output_anchor)
        max_queue = max(max_queue, producer.prepared.qsize())
        max_features = max(max_features, len(producer.features))
        max_raw = max(max_raw, len(history.frames))
    return dict(producer=producer, source=source, completed=completed,
                windows=emitted_windows, per_completion_publish_counts=processing_counts,
                bounds=(max_queue,max_features,max_raw))


def test_cpu_ve_stage2_overlap_avoids_whole_window_reencoding(tmp_path):
    result = simulate_framewise(tmp_path)
    producer = result['producer']
    windows, outputs = result['windows'], result['completed']
    assert len(windows) >= 85 and len(outputs) >= 60
    assert 3.5 <= (len(outputs)-1)/(outputs[-1][0]-outputs[0][0]) <= 3.7
    assert producer.counts['encoded_frames'] <= 100
    assert producer.counts['prepared_windows'] == producer.counts['encoded_frames']-10
    assert result['per_completion_publish_counts'] == [1]*len(windows)
    assert producer.counts['preprocessing_queue_drops'] == 0
    assert producer.counts['encoded_history_gaps'] == 0
    assert max(t-anchor for t,anchor in outputs) < .8
    assert all(r['context']['new_vision_frames'] == 1 for r in producer.features)
    assert len(result['source'].prepared_ids) == len(set(result['source'].prepared_ids))
    assert result['bounds'][0] <= 1 and result['bounds'][1] <= 11 and result['bounds'][2] <= 64


def test_overloaded_ve_drops_backlog_and_reports_history_gaps(tmp_path):
    result = simulate_framewise(tmp_path, vision_s=.3)
    producer = result['producer']
    assert producer.counts['preprocessing_queue_drops'] > 0
    assert producer.counts['encoded_history_gaps'] > 0
    assert producer.counts['encoded_frames'] <= 67
    assert producer.counts['prepared_windows'] == 0
    assert result['completed'] == []
    assert producer.state['history_selection']['selected'] < 11
    assert result['bounds'][0] <= 1 and result['bounds'][1] <= 11
    assert all(r['context']['new_vision_frames'] == 1 for r in producer.features)


def test_previous_early_skip_failure_is_not_hidden_by_fake_frames():
    old = TimedTokenHistory(capture_hz=5.)
    feature = torch.zeros(106, 256)
    for identity in range(201):
        if identity % 2 == 0:
            old.append(feature, stamp=100.+identity*.1125, ids=dict(lidar=identity))
            assert len(old.snapshot(materialize=False)[1]) != 11
    history = RawWindowHistory()
    grid = FrameGrid(.2, .06)
    selected = []
    for identity in range(101):
        stamp=100.+identity*.225
        add(history, identity, stamp)
        row=grid.select(history.raw_snapshot(), stamp+.06, .3)
        if row is not None:
            selected.append(row)
    assert grid.missed > 0
    assert len({r['ids']['lidar'] for r in selected}) == len(selected)
    assert all(abs(r['stamp']-r['target_stamp']) <= .06+1e-8 for r in selected)


def test_collect_clears_history_on_simulator_rewind(tmp_path):
    history = RawWindowHistory()
    stop = threading.Event()
    sequence = [observation(i, 100.+i*.1, sim_time=i*.1)[0] for i in range(21)]
    sequence += [observation(i, 100.+i*.1, sim_time=(i-21)*.1)[0]
                 for i in range(21, 42)]

    class StreamSource(SnapshotSource):
        def snapshot_raw(self, **_kwargs):
            if not sequence:
                stop.set()
                return None, dict(sample_ids={}, missing=['lidar'])
            raw = sequence.pop(0)
            return raw, raw['meta']

    class Navigation:
        resets = 0

        def reset(self):
            self.resets += 1

    navigation = Navigation()
    producer, source = producer_for(history, tmp_path, source=StreamSource(), stop=stop,
                                    navigation=navigation)
    producer.collect()
    assert producer.state['error'] is None
    assert navigation.resets == history.resets == 1
    assert producer.state['capture_generation'] == 1
    assert producer.counts['raw_frames'] == 42
    assert source.prepared_ids == []
    assert all(row['ids']['lidar'] >= 21 and row['generation'] == 1
               for row in history.raw_snapshot())


def test_collect_retries_same_lidar_when_delayed_camera_becomes_valid(tmp_path):
    history = RawWindowHistory()
    stop = threading.Event()
    raw, _context = observation(1, 100.)
    sequence = [(None, dict(raw['meta'], missing=['right'])), (raw, raw['meta'])]

    class StreamSource(SnapshotSource):
        def snapshot_raw(self, **_kwargs):
            if not sequence:
                stop.set()
                return None, dict(sample_ids={}, missing=['lidar'])
            return sequence.pop(0)

    producer, source = producer_for(history, tmp_path, source=StreamSource(), stop=stop)
    producer.collect()
    assert producer.state['error'] is None
    assert producer.counts['raw_retries'] == 1
    assert producer.counts['raw_frames'] == 1
    assert history.frames[-1]['ids']['lidar'] == 1
    assert source.prepared_ids == []
