"""Receive continuously, encode small scheduled batches, reuse real features.

The sampling grid is fixed across predictions. Sensor receipt stamps remain
untouched; grid timestamps describe selection targets, not fabricated samples.
"""
from bisect import bisect_left, bisect_right
from collections import deque
from contextlib import nullcontext
import json
import math
import queue
import threading
import time
import traceback

import torch

from inference.vldrive_runtime import TokenHistory


class RawWindowHistory:
    def __init__(self, interval_s=.2, tolerance_s=.06, *, reset_gap_s=1., capacity=64):
        if not 0 < tolerance_s < interval_s / 2 or reset_gap_s <= interval_s + tolerance_s:
            raise ValueError('Invalid raw history timing')
        if type(capacity) is not int or not 11 <= capacity <= 128:
            raise ValueError('Raw history capacity must be in 11..128')
        self.interval, self.tolerance = interval_s, tolerance_s
        self.reset_gap_s = reset_gap_s
        self.frames = deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.changed = threading.Event()
        self.ready = threading.Event()
        self.resets = 0
        self._encoded = []
        self.selection = dict(reason='warmup', available=0, selected=0)

    def clear(self):
        with self.lock:
            if self.frames or self._encoded:
                self.resets += 1
            self.frames.clear()
            self._encoded = []
        self.changed.set()
        self.ready.set()

    def append(self, raw, context):
        meta = raw['meta']
        stamp, ids = meta['received_at']['lidar'], dict(meta['sample_ids'])
        if stamp is None or not math.isfinite(stamp):
            return False
        with self.lock:
            if self.frames:
                previous = self.frames[-1]
                if stamp <= previous['stamp'] or ids['lidar'] <= previous['ids']['lidar']:
                    return False
                # Auxiliary streams may be reused, but never run backwards.
                if any(ids[k] < previous['ids'].get(k, ids[k]) for k in ids):
                    return False
                if stamp - previous['stamp'] > self.reset_gap_s:
                    self.frames.clear()
                    self._encoded = []
                    self.resets += 1
            self.frames.append(dict(raw=raw, stamp=stamp, ids=ids, context=context,
                                    generation=self.resets))
            oldest = stamp - (10 * self.interval + self.reset_gap_s + self.tolerance)
            while self.frames and self.frames[0]['stamp'] < oldest:
                self.frames.popleft()
        self.changed.set()
        return True

    def select_raw(self):
        with self.lock:
            frames = list(self.frames)
        if not frames:
            self.selection = dict(reason='warmup', available=0, selected=0)
            return []
        stamps = [row['stamp'] for row in frames]
        anchor, upper, selected = stamps[-1], len(frames), []
        for offset in range(11):
            target = anchor - offset * self.interval
            insertion = bisect_left(stamps, target, 0, upper)
            candidates = [i for i in (insertion-1, insertion) if 0 <= i < upper]
            if not candidates:
                break
            chosen = min(candidates, key=lambda i: abs(stamps[i]-target))
            if abs(stamps[chosen]-target) > self.tolerance + 1e-8:
                break
            selected.append(frames[chosen])
            upper = chosen
        complete = len(selected) == 11
        self.selection = dict(reason='ready' if complete else (
            'warmup' if anchor-stamps[0] < 10*self.interval-self.tolerance else 'history_time_alignment'),
            available=len(frames), selected=len(selected), anchor_received_at=anchor,
            missing_offset_s=None if complete else len(selected)*self.interval)
        return selected[::-1] if complete else []

    def raw_snapshot(self):
        with self.lock:
            return list(self.frames)

    def publish(self, rows):
        with self.lock:
            if len(rows) != 11 or any(r['generation'] != self.resets for r in rows):
                return False
            if self._encoded and rows[-1]['stamp'] <= self._encoded[-1]['stamp']:
                return False
            self._encoded = rows
        self.ready.set()
        return True

    def snapshot(self, *, materialize=True):
        with self.lock:
            rows = list(self._encoded)
        return (self.materialize(rows) if rows and materialize else None), rows

    materialize = staticmethod(TokenHistory.materialize)


class FrameGrid:
    """Choose one real sample per fixed trained-time slot, without catch-up FIFO.

    Close a slot after its tolerance window, so both neighboring observations
    can be considered. Missing slots stay missing; no duplication/interpolation.
    """
    def __init__(self, interval, tolerance):
        self.interval, self.tolerance = interval, tolerance
        self.generation = None
        self.origin = None
        self.slot = 0
        self.last_id = -1
        self.missed = 0
        self.expired = 0
        self.reason = 'waiting_for_raw'

    def select(self, frames, now, max_age):
        if not frames:
            self.reason = 'waiting_for_raw'
            return None
        generation = frames[-1]['generation']
        if self.generation != generation:
            self.generation, self.origin = generation, frames[0]['stamp']
            self.slot, self.last_id = 0, -1
        # Never replay a backlog of old slots when preprocessing falls behind.
        first_recent = max(0, math.ceil(
            (now-max_age-self.tolerance-self.origin)/self.interval-1e-8))
        if self.slot < first_recent:
            self.expired += first_recent-self.slot
            self.slot = first_recent
        stamps = [row['stamp'] for row in frames]
        while self.origin+self.slot*self.interval+self.tolerance <= now+1e-8:
            slot = self.slot
            target = self.origin+slot*self.interval
            self.slot += 1
            lower = bisect_left(stamps, target-self.tolerance-1e-8)
            upper = bisect_right(stamps, target+self.tolerance+1e-8)
            candidates = [frames[i] for i in range(lower, upper)
                          if frames[i]['generation'] == generation
                          and frames[i]['ids']['lidar'] > self.last_id
                          and 0 <= now-frames[i]['stamp'] <= max_age
                          and abs(frames[i]['stamp']-target) <= self.tolerance+1e-8]
            if not candidates:
                self.missed += 1
                self.reason = 'sampling_slot_missing'
                continue
            row = min(candidates, key=lambda r: abs(r['stamp']-target))
            self.last_id = row['ids']['lidar']
            self.reason = 'frame_selected'
            return dict(row, slot=slot, target_stamp=target)
        self.reason = 'waiting_for_sampling_slot'
        return None


class WindowInputProducer:
    def __init__(self, cfg, source, history, state, stop, output, navigation,
                 encoder, encoder_stream, encode_frame, navigation_for_frame, *, encode_frames=None):
        self.cfg, self.source, self.history = cfg, source, history
        self.state, self.stop, self.output = state, stop, output
        self.navigation, self.encoder, self.encoder_stream = navigation, encoder, encoder_stream
        self.encode_frame, self.navigation_for_frame = encode_frame, navigation_for_frame
        self.encode_frames = encode_frames
        self.grid = FrameGrid(history.interval, history.tolerance)
        self.features = deque(maxlen=11)
        runtime = cfg['runtime']
        self.max_span = runtime.get('history_max_span_s', 10*history.interval+2*history.tolerance)
        self.max_feature_gap = runtime.get('history_max_adjacent_gap_s', history.interval+2*history.tolerance)
        self.pin_inputs = runtime.get('pin_encoder_inputs', False) and torch.device(runtime['device']).type == 'cuda'
        self.batch_size = runtime.get('encoder_batch_size', 1)
        if self.batch_size not in (1, 2) or (self.batch_size == 2 and encode_frames is None):
            raise ValueError('Two-frame encoding requires the batched encoder callback')
        self.prepared = queue.Queue(maxsize=self.batch_size)
        self.counts = dict(raw_frames=0, prepared_windows=0, encoded_frames=0,
                           feature_cache_hits=0, incomplete_windows=0,
                           stale_windows=0, raw_retries=0, encoder_budget_deferrals=0,
                           sampling_slots_missing=0, sampling_slots_expired=0,
                           preprocessing_frames=0, preprocessing_queue_drops=0,
                           stale_encoder_frames=0, encoded_history_gaps=0,
                           feature_history_resets=0, temporal_bound_rejections=0,
                           extended_windows=0, encoder_batches=0)
        state['input_preparation'] = self.counts

    def collect(self):
        last_ids, last_sim_time, last_instruction = None, None, None
        rejected_signature = None
        try:
            with (self.output/'navigation.jsonl').open('x', encoding='utf-8', buffering=1) as log:
                while not self.stop.is_set():
                    # Fast snapshots only: decoding is deferred until selection.
                    raw, meta = self.source.snapshot_raw(
                        max_age=self.cfg['runtime']['max_age_s'],
                        max_skew=self.cfg['runtime']['max_stream_skew_s'])
                    ids = meta.get('sample_ids', {})
                    signature = tuple(sorted(ids.items()))
                    if raw is None:
                        self.state['missing'] = meta['missing']
                        self.state['navigation'] = None
                        if signature != rejected_signature:
                            self.state['rejected_frames'] += 1
                            if meta.get('stream_skew_s', 0) > self.cfg['runtime']['max_stream_skew_s']:
                                self.state['discarded_skew'] += 1
                            rejected_signature = signature
                        self.stop.wait(.005)
                        continue
                    self.state['missing'] = []
                    if last_ids is not None and ids['lidar'] <= last_ids['lidar']:
                        # A transient late sensor may have invalidated navigation
                        # for this anchor; restore it without fabricating a frame.
                        if self.state['navigation'] is None:
                            self.state['navigation'] = self.navigation_for_frame(self.navigation, self.cfg, meta)
                        self.stop.wait(.005)
                        continue
                    sim_time = meta.get('sim_time')
                    if sim_time is not None and last_sim_time is not None and sim_time < last_sim_time:
                        self.state['capture_generation'] += 1
                        self.history.clear()
                        if self.navigation is not None:
                            self.navigation.reset()
                    nav = self.navigation_for_frame(self.navigation, self.cfg, meta)
                    self.state['navigation'] = nav
                    if not nav['valid']:
                        if signature != rejected_signature:
                            self.state['rejected_frames'] += 1
                            log.write(json.dumps(dict(frame=ids['lidar'], navigation=nav), allow_nan=False)+'\n')
                            rejected_signature = signature
                        self.stop.wait(.005)
                        continue
                    key = nav['instruction'].get('instruction_instance_id', nav['instruction']['text'])
                    if key != last_instruction:
                        print('Navigation: '+nav['instruction']['text'], flush=True)
                        last_instruction = key
                    context = dict(navigation=nav, front_jpeg=meta['front_jpeg'], sim_time=sim_time)
                    if self.history.append(raw, context):
                        self.counts['raw_frames'] += 1
                        if rejected_signature is not None and dict(rejected_signature).get('lidar') == ids['lidar']:
                            self.counts['raw_retries'] += 1
                        log.write(json.dumps(dict(frame=ids['lidar'], navigation=nav,
                            anchor_received_at=meta['received_at']['lidar'],
                            sensor_received_at=meta['received_at'], sample_ids=ids), allow_nan=False)+'\n')
                        last_ids, last_sim_time = dict(ids), sim_time
                        rejected_signature = None
                    self.stop.wait(.001)
        except Exception:
            self.state['error'] = traceback.format_exc()
            self.stop.set()

    def preprocess(self):
        """Prepare the next scheduled frame concurrently with the current VE."""
        try:
            while not self.stop.is_set():
                self.history.changed.wait(.01)
                self.history.changed.clear()
                row = self.grid.select(self.history.raw_snapshot(), time.monotonic(),
                                       self.cfg['runtime']['max_age_s'])
                self.counts['sampling_slots_missing'] = self.grid.missed
                self.counts['sampling_slots_expired'] = self.grid.expired
                self.state['sampling_status'] = dict(reason=self.grid.reason,
                    next_slot=self.grid.slot, generation=self.grid.generation)
                if row is None:
                    continue
                started = time.perf_counter()
                inputs, meta = self.source.prepare_snapshot(row['raw'], device='cpu')
                if inputs is None:
                    self.state['rejected_frames'] += 1
                    continue
                if row['generation'] != self.history.resets:
                    continue
                if self.pin_inputs:
                    # Prepared/pending/active frames bound pinned host memory.
                    # num_points is a CPU slice bound inside PointPillars.
                    inputs = {k: v.pin_memory() if k != 'num_points' else v for k,v in inputs.items()}
                item = dict(row=row, inputs=inputs,
                            preprocessing_ms=(time.perf_counter()-started)*1000,
                            prepared_at=time.monotonic())
                self.counts['preprocessing_frames'] += 1
                try:
                    self.prepared.put_nowait(item)
                except queue.Full:
                    try:
                        self.prepared.get_nowait()
                        self.counts['preprocessing_queue_drops'] += 1
                    except queue.Empty:
                        pass
                    self.prepared.put_nowait(item)
        except Exception:
            self.state['error'] = traceback.format_exc()
            self.stop.set()

    def append_encoded(self, row, feature, ready, vision_ms, *, preprocessing_ms,
                       processing_ms, queue_wait_ms, now, batch_size=1, publish=True,
                       batch_preprocessing_ms=None):
        """Reuse real features across short gaps, within explicit time bounds."""
        if self.stop.is_set() or row['generation'] != self.history.resets:
            return False
        if self.features:
            previous = self.features[-1]
            if previous['generation'] == row['generation'] and (
                    row['slot'] <= previous['slot'] or row['stamp'] <= previous['stamp']
                    or row['ids']['lidar'] <= previous['ids']['lidar']):
                return False
            if previous['generation'] == row['generation'] and row['slot'] != previous['slot']+1:
                self.counts['encoded_history_gaps'] += 1
            if (previous['generation'] != row['generation']
                    or row['stamp']-previous['stamp'] > self.max_feature_gap+1e-8):
                self.features.clear()
                self.counts['feature_history_resets'] += 1
        context = dict(row['context'], preprocessing_ms=preprocessing_ms,
                       window_preparation_ms=processing_ms, new_vision_frames=batch_size,
                       encoder_batch_size=batch_size,
                       window_vision_ms=vision_ms,
                       window_preprocessing_ms=preprocessing_ms if batch_preprocessing_ms is None else batch_preprocessing_ms,
                       encoder_queue_wait_ms=queue_wait_ms,
                       input_ready_age_ms=(now-row['stamp'])*1000)
        self.features.append(dict(stamp=row['stamp'], ids=row['ids'],
            slot=row['slot'], target_stamp=row['target_stamp'],
            generation=row['generation'], context=context, feature=feature.detach(),
            ready=ready, vision_ms=vision_ms))
        # Never retain an arbitrarily stretched 11-frame clip under overload.
        # Timestamps/identities are the observations' actual values, not grid time.
        while self.features and row['stamp']-self.features[0]['stamp'] > self.max_span+1e-8:
            self.features.popleft()
            self.counts['temporal_bound_rejections'] += 1
        span = row['stamp']-self.features[0]['stamp']
        extended = len(self.features) == 11 and self.features[-1]['slot']-self.features[0]['slot'] > 10
        context.update(history_temporally_extended=extended,
                       history_effective_interval_s=span/10 if len(self.features)==11 else None)
        self.state['history_selection'] = dict(reason='encoding_history' if len(self.features)<11 else 'ready',
            available=len(self.history.raw_snapshot()), selected=len(self.features),
            anchor_received_at=row['stamp'], sampling_clock='fixed_monotonic_grid',
            target_stamp=row['target_stamp'], feature_span_s=span,
            max_span_s=self.max_span, max_adjacent_gap_s=self.max_feature_gap,
            temporally_extended=extended)
        if len(self.features) < 11:
            self.counts['incomplete_windows'] += 1
            return False
        if now-row['stamp'] > self.cfg['visualization']['max_prediction_age_s']:
            self.counts['stale_windows'] += 1
            return False
        if not publish:
            return False
        if self.history.publish(list(self.features)):
            self.counts['prepared_windows'] += 1
            self.counts['feature_cache_hits'] += 11-batch_size
            self.counts['extended_windows'] += int(extended)
            return True
        return False

    def take_batch(self):
        """Wait briefly for at most two chronological frames, without catch-up FIFO."""
        try:
            items = [self.prepared.get(timeout=.05)]
        except queue.Empty:
            return []
        deadline = time.monotonic()+self.cfg['runtime'].get('encoder_batch_wait_s', self.history.interval+.02)
        while len(items) < self.batch_size and not self.stop.is_set():
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                break
            try:
                items.append(self.prepared.get(timeout=min(.02, remaining)))
            except queue.Empty:
                pass
        now = time.monotonic()
        # An earlier member of a two-frame batch is intentionally one slot older.
        # The newest member must still meet the original freshness requirement.
        result = []
        for i,item in enumerate(items):
            row = item['row']
            limit = self.cfg['runtime']['max_age_s']+(len(items)-1-i)*self.history.interval
            if row['generation'] != self.history.resets:
                continue
            if now-row['stamp'] > limit:
                self.counts['stale_encoder_frames'] += 1
                continue
            result.append(item)
        # If a stale/invalid newest item was removed, it cannot lend freshness
        # to the surviving earlier frame.
        while result and now-result[-1]['row']['stamp'] > self.cfg['runtime']['max_age_s']:
            result.pop()
            self.counts['stale_encoder_frames'] += 1
        return result

    def transfer_batch(self, items):
        inputs = {}
        for key in items[0]['inputs']:
            tensors = [item['inputs'][key] for item in items]
            if len(tensors) == 1:
                value = tensors[0]
            else:
                shape = (sum(t.shape[0] for t in tensors), *tensors[0].shape[1:])
                value = torch.empty(shape, dtype=tensors[0].dtype,
                                    pin_memory=self.pin_inputs and key != 'num_points')
                torch.cat(tensors, dim=0, out=value)
            inputs[key] = value if key == 'num_points' else value.to(
                self.cfg['runtime']['device'], non_blocking=self.pin_inputs)
        return inputs

    def encode(self):
        """Encode one/two new frames, never an entire historical window again."""
        runtime = self.cfg['runtime']
        next_start = 0.
        preparer = threading.Thread(target=self.preprocess, name='vldrive-preprocess', daemon=True)
        preparer.start()
        try:
            while not self.stop.is_set():
                delay = next_start-time.monotonic()
                if delay > 0:
                    self.counts['encoder_budget_deferrals'] += 1
                    self.stop.wait(delay)
                if self.stop.is_set():
                    break
                items = self.take_batch()
                if not items or self.stop.is_set():
                    continue
                now = time.monotonic()
                next_start = now+len(items)/runtime.get('encoder_max_hz', 10.)
                started = time.perf_counter()
                with torch.cuda.stream(self.encoder_stream) if self.encoder_stream is not None else nullcontext():
                    inputs = self.transfer_batch(items)
                    if len(items) == 1:
                        feature, elapsed = self.encode_frame(self.encoder, inputs)
                        features = feature.unsqueeze(0)
                    else:
                        features, elapsed = self.encode_frames(self.encoder, inputs)
                    ready = None
                    if features.is_cuda:
                        ready = torch.cuda.Event()
                        ready.record(torch.cuda.current_stream(features.device))
                self.counts['encoded_frames'] += len(items)
                self.counts['encoder_batches'] += 1
                processing_ms = sum(item['preprocessing_ms'] for item in items)+(time.perf_counter()-started)*1000
                for index,(item,feature) in enumerate(zip(items, features)):
                    self.append_encoded(item['row'], feature, ready, elapsed,
                        preprocessing_ms=item['preprocessing_ms'], processing_ms=processing_ms,
                        queue_wait_ms=(now-item['prepared_at'])*1000, now=time.monotonic(),
                        batch_size=len(items), publish=index==len(items)-1,
                        batch_preprocessing_ms=sum(member['preprocessing_ms'] for member in items))
        except Exception:
            self.state['error'] = traceback.format_exc()
            self.stop.set()
        finally:
            self.stop.set()
            preparer.join(timeout=15)
