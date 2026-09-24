"""Sensor acquisition, anchor navigation, temporal inference and optional control."""
from datetime import datetime, timezone
from contextlib import nullcontext
import json
import queue
import threading
import time
import traceback

import torch

from inference.live_frame import CAMERA_KEYS, LiveFrameSource
from inference.vldrive_runtime import (TokenHistory, encode_frame, encode_frames, load_encoder, load_stage2,
                                      local_path, predict_window, seed_runtime)
from navigation.live import navigation_for_frame
from inference.input_windows import RawWindowHistory, WindowInputProducer


def write_report(path, report):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    pending.replace(path)


def stream_rows(source):
    return [dict(stream=name,port=port,packets=packets,samples=samples,
                 age_s=age if age != float('inf') else None)
            for name,port,packets,samples,age in source.stream_report()]


def current_window(rows, history, max_age, now=None):
    now = time.monotonic() if now is None else now
    return (len(rows)==11 and rows[-1]['generation']==history.resets
            and 0 <= now-rows[-1]['stamp'] <= max_age)


def input_available(history, state):
    nav = state['navigation']
    if isinstance(history, RawWindowHistory):
        return not state['missing'] and nav is not None and nav['valid']
    return nav is None or nav['valid']


def run_live(cfg, contract, output, report, duration=None, probe_only=False, *, navigation=None, view=None, controller=None):
    runtime, route = cfg['runtime'], cfg['route']
    source = LiveFrameSource(cfg['ports'], route_csv=local_path(route['csv']) if route['csv'] else None,
        lidar_elevations=cfg['lidar']['elevations'], lidar_axis_policy=cfg['lidar']['axis_policy'],
        bind=runtime['bind'], encoder_recipe=contract['meta']['encoder_contract'])
    stop = threading.Event()
    worker = None
    encoder_worker = None
    captured = queue.Queue(maxsize=runtime['encoder_queue_size'])
    if runtime.get('resample_history', False):
        history = RawWindowHistory(contract['history_interval_s'], runtime['history_tolerance_s'],
            reset_gap_s=runtime.get('history_reset_gap_s', 1.),
            capacity=runtime.get('input_buffer_frames', 64))
    else:
        history = TokenHistory(contract['history_interval_s'], runtime['history_tolerance_s'])
    state = dict(error=None, missing=[], discarded_skew=0, navigation=None, rejected_frames=0,
                 capture_drops=0, capture_generation=0)
    def display(**values):
        if view is not None:
            view.update(**values)
    required = {*CAMERA_KEYS, 'lidar', 'vehicle_status'}
    if cfg['navigation']['mode']=='route':
        required.add('gps')
    try:
        # Keep the UI and packet diagnostics alive before MORAI begins sending.
        began = time.monotonic()
        last_console = 0.
        while True:
            streams = stream_rows(source)
            missing = [row['stream'] for row in streams if row['stream'] in required
                       and (row['age_s'] is None or row['age_s'] > runtime['max_age_s'])]
            report.update(streams=streams,status='waiting_for_sensors',missing_streams=missing)
            display(status='waiting_for_sensors',reason=', '.join(missing),streams=streams,clear_prediction=True)
            if not missing:
                break
            elapsed = time.monotonic()-began
            if time.monotonic()-last_console >= 2:
                suffix = ' | check destination IP/ports; retrying' if elapsed>runtime['startup_timeout_s'] else ''
                print('Waiting for UDP: '+', '.join(missing)+suffix,flush=True)
                last_console = time.monotonic()
            if duration is not None and elapsed >= duration:
                report.update(status='live_unavailable',reason='No complete fresh required MORAI streams',
                              full_live_forward_tested=False)
                return
            stop.wait(.2)
        if probe_only:
            report.update(status='sensors_received',full_live_forward_tested=False)
            return
        report['stage'] = 'A_checkpoint_load'
        display(status='loading_models',reason='검증된 best checkpoint를 불러옵니다.',clear_prediction=True)
        model, report['stage2'] = load_stage2(contract,runtime['device'])
        from inference.navigation_tokens import attach_navigation_tokens
        report['navigation_tokens'] = attach_navigation_tokens(model, cfg, navigation)
        encoder = load_encoder(cfg,contract,runtime['device'])
        seed_runtime(cfg['seed'])
        encoder_stream=None
        if torch.device(runtime['device']).type=='cuda' and runtime.get('separate_encoder_stream',False):
            encoder_stream=torch.cuda.Stream(device=runtime['device'], priority=runtime.get('encoder_stream_priority', 0))
            encoder_stream.wait_stream(torch.cuda.current_stream(runtime['device']))
        report['separate_encoder_stream']=encoder_stream is not None
        report['window_execution'] = dict(optimized=runtime.get('optimized_window', False),
            resample_history=runtime.get('resample_history', False),
            capture_hz=runtime.get('capture_hz', runtime['target_hz']),
            encoder_max_hz=runtime.get('encoder_max_hz', 10.),
            encoder_batch_size=runtime.get('encoder_batch_size', 1),
            encoder_stream_priority=encoder_stream.priority if encoder_stream is not None else None,
            pin_encoder_inputs=runtime.get('pin_encoder_inputs', False),
            history_max_span_s=runtime.get('history_max_span_s', 10*history.interval+2*history.tolerance),
            history_max_adjacent_gap_s=runtime.get('history_max_adjacent_gap_s', history.interval+2*history.tolerance),
            input_buffer_frames=runtime.get('input_buffer_frames', 64),
            sampling_grid_hz=1/contract['history_interval_s'] if isinstance(history, RawWindowHistory) else None,
            capture_hz_applies_to='legacy_only' if isinstance(history, RawWindowHistory) else 'frame_capture',
            trained_interval_s=contract['history_interval_s'],
            input_assembly='streaming_grid_features' if isinstance(history, RawWindowHistory) else 'legacy',
            sampling_tolerance_basis='fixed_grid_slot' if isinstance(history, RawWindowHistory) else 'previous_frame',
            visual_kv_cache=False)
        print('Models loaded | eval | Gumbel | reconstruction off | navigation '+cfg['navigation']['mode'],flush=True)
        if isinstance(history, RawWindowHistory):
            print(f'Input: streaming {1/history.interval:g}Hz sampling grid | '
                  f'11 cached features | span <= {report["window_execution"]["history_max_span_s"]:g}s | '
                  f'VE batch <= {runtime.get("encoder_batch_size", 1)} | separate preprocessing and VE workers', flush=True)
        if controller is not None:
            controller.start(source,navigation,history,state,runtime['max_age_s'])

        def collect():
            last_seen = 0
            last_sim_time = None
            last_instruction = None
            last_capture_stamp = None
            last_capture_ids = None
            try:
                with (output/'navigation.jsonl').open('x',encoding='utf-8',buffering=1) as log:
                    while not stop.is_set():
                        _, age, sequence, preview_stamp = source.lidar.snapshot()
                        if age > runtime['max_age_s']:
                            state.update(missing=['lidar'],navigation=None)
                        if not sequence or sequence==last_seen:
                            stop.wait(.01)
                            continue
                        last_seen = sequence
                        # Avoid decoding four JPEGs and transforming LiDAR for
                        # an intermediate frame that cannot enter the history.
                        if (last_capture_stamp is not None and preview_stamp is not None
                                and 0<=preview_stamp-last_capture_stamp<history.interval-history.tolerance):
                            continue
                        preprocessing_started=time.perf_counter()
                        inputs, meta = source.build('cpu',(route['utm_east_offset'],route['utm_north_offset']),
                                                    runtime['max_age_s'])
                        preprocessing_ms=(time.perf_counter()-preprocessing_started)*1000
                        state['missing'] = meta['missing']
                        if inputs is None:
                            state['rejected_frames'] += 1
                            state['navigation'] = None
                            continue
                        stamps = list(meta['received_at'].values())
                        if max(stamps)-min(stamps)>runtime['max_stream_skew_s']:
                            state['discarded_skew'] += 1
                            state['missing'] = ['sensor_receive_time_skew']
                            state['navigation'] = None
                            continue
                        stamp, ids = meta['received_at']['lidar'], meta['sample_ids']
                        sim_time = meta.get('sim_time')
                        if sim_time is not None and last_sim_time is not None:
                            if sim_time<last_sim_time:
                                state['capture_generation'] += 1
                                history.clear()
                                if navigation is not None:
                                    navigation.reset()
                                last_sim_time = None
                                last_capture_stamp = last_capture_ids = None
                            elif sim_time==last_sim_time:
                                continue
                        nav = navigation_for_frame(navigation,cfg,meta)
                        state['navigation'] = nav
                        log.write(json.dumps(dict(frame=ids['lidar'],navigation=nav),allow_nan=False)+'\n')
                        if not nav['valid']:
                            state['rejected_frames'] += 1
                            continue
                        key = nav['instruction'].get('instruction_instance_id',nav['instruction']['text'])
                        if key!=last_instruction:
                            print('Navigation: '+nav['instruction']['text'],flush=True)
                            last_instruction = key
                        if last_capture_stamp is not None and (
                                stamp-last_capture_stamp < history.interval-history.tolerance
                                or any(ids[k]<=last_capture_ids[k] for k in ids)):
                            continue
                        item=dict(inputs=inputs,stamp=stamp,ids=ids,generation=state['capture_generation'],
                                  context=dict(navigation=nav,front_jpeg=meta['front_jpeg'],sim_time=sim_time,
                                               preprocessing_ms=preprocessing_ms))
                        try:
                            captured.put_nowait(item)
                        except queue.Full:
                            try:
                                captured.get_nowait()
                                state['capture_drops'] += 1
                            except queue.Empty:
                                pass
                            captured.put_nowait(item)
                        last_capture_stamp,last_capture_ids = stamp,ids
                        last_sim_time = sim_time
            except Exception:
                state['error'] = traceback.format_exc()
                stop.set()

        def encode_captured():
            try:
                while not stop.is_set():
                    try:
                        item=captured.get(timeout=.1)
                    except queue.Empty:
                        continue
                    # A bounded queue keeps acquisition independent of GPU work.
                    # Late frames are dropped; time is never relabeled as current.
                    if (item['generation']!=state['capture_generation'] or
                            time.monotonic()-item['stamp']>cfg['visualization']['max_prediction_age_s']):
                        state['capture_drops'] += 1
                        continue
                    with torch.cuda.stream(encoder_stream) if encoder_stream is not None else nullcontext():
                        inputs={k:v.to(runtime['device']) for k,v in item['inputs'].items()}
                        feature,elapsed=encode_frame(encoder,inputs)
                        if item['generation']!=state['capture_generation']:
                            continue
                        history.append(feature,stamp=item['stamp'],ids=item['ids'],vision_ms=elapsed,
                                       context=item['context'])
            except Exception:
                state['error']=traceback.format_exc()
                stop.set()

        if isinstance(history, RawWindowHistory):
            producer = WindowInputProducer(cfg, source, history, state, stop, output,
                navigation, encoder, encoder_stream, encode_frame, navigation_for_frame, encode_frames=encode_frames)
            collect, encode_captured = producer.collect, producer.encode
        worker = threading.Thread(target=collect,name='vldrive-capture',daemon=True)
        encoder_worker = threading.Thread(target=encode_captured,name='vldrive-vision',daemon=True)
        worker.start()
        encoder_worker.start()
        report.update(stage='E_live_waypoints',status='warming_up',outputs=0,discarded_predictions=0,
                      timing_clock='monotonic sensor receipt time; not simulator synchronization')
        started = time.monotonic()
        last_output_stamp = None
        last_prediction_at = 0.
        last_ui = 0.
        last_console = 0.
        last_report = 0.
        with (output/'waypoints.jsonl').open('x',encoding='utf-8',buffering=1) as stream:
            while not stop.is_set() and (duration is None or time.monotonic()-started<duration):
                if controller is not None:
                    controller.check_error()
                _, rows = history.snapshot(materialize=False)
                now = time.monotonic()
                output_age_limit=cfg['visualization']['max_prediction_age_s']
                ready = current_window(rows,history,output_age_limit,now)
                if not input_available(history, state):
                    ready=False
                if now-last_report >= 1.:
                    report.update(input_preparation=dict(state.get('input_preparation', {})),
                                  history_selection=dict(state.get('history_selection', {})),
                                  sampling_status=dict(state.get('sampling_status', {})))
                    write_report(output/'inference_report.json',report)
                    last_report=now
                if now-last_ui>=.2:
                    report['streams'] = stream_rows(source)
                    count = state.get('history_selection', {}).get('selected', len(rows))
                    display(streams=report['streams'],history_count=count,history_resets=history.resets,
                            navigation=state['navigation'])
                    last_ui = now
                if not ready:
                    if controller is not None:
                        controller.invalidate('history_or_navigation_unavailable')
                    nav = state['navigation']
                    reason = ', '.join(state['missing'])
                    if not reason and state.get('history_selection'):
                        selection=state['history_selection']
                        reason=f'{selection["reason"]}: {selection["selected"]}/11 encoded frames'
                    status = 'warming_up'
                    if nav is not None and not nav['valid']:
                        status, reason = 'navigation_unavailable',nav['reason']
                    elif rows and now-rows[-1]['stamp']>output_age_limit:
                        age = now-rows[-1]['stamp']
                        detail = state.get('history_selection', {})
                        status = 'stale'
                        reason = (f'완성된 입력이 {age:.2f}초 경과했습니다. '
                                  f'새 특징 이력 {detail.get("selected", 0)}/11; '
                                  f'{detail.get("reason", "waiting_for_encoded_history")}')
                    display(status=status,reason=reason,clear_prediction=True)
                    if now-last_console>=2:
                        count = state.get('history_selection', {}).get('selected', len(rows))
                        print(f'Encoded history {count}/11 | {status} | {reason}',flush=True)
                        last_console = now
                    if isinstance(history, RawWindowHistory):
                        history.ready.wait(.02)
                        history.ready.clear()
                    else:
                        stop.wait(.02)
                    continue
                if rows[-1]['stamp']==last_output_stamp:
                    if isinstance(history, RawWindowHistory):
                        history.ready.wait(.02)
                        history.ready.clear()
                    else:
                        stop.wait(.01)
                    continue
                if runtime['prediction_hz'] is not None and now-last_prediction_at < 1/runtime['prediction_hz']:
                    stop.wait(.01)
                    continue
                last_output_stamp = rows[-1]['stamp']
                last_prediction_at = now
                context = rows[-1]['context']
                instruction = context['navigation']['instruction']['text']
                features=history.materialize(rows)
                result = predict_window(model,features,instruction,
                                        optimized=runtime.get('optimized_window', False))
                # A reset/stream outage during forward cannot produce a current display.
                if (stop.is_set() or not current_window(rows,history,output_age_limit)
                        or not input_available(history, state)):
                    if controller is not None:
                        controller.invalidate('history_reset_or_stale')
                    report['discarded_predictions'] += 1
                    display(status='stale',reason='추론 중 입력 이력이 만료되었습니다.',clear_prediction=True)
                    continue
                result.update(timestamp=datetime.now(timezone.utc).isoformat(),frame=rows[-1]['ids']['lidar'],
                    instruction=instruction,navigation=context['navigation'],sim_time=context['sim_time'],
                    anchor_received_at=rows[-1]['stamp'],history_ready=True,history_generation=rows[-1]['generation'],
                    history_sample_ids=[r['ids'] for r in rows],history_timestamps_s=[r['stamp'] for r in rows],
                    history_span_s=rows[-1]['stamp']-rows[0]['stamp'],
                    history_temporally_extended=context.get('history_temporally_extended', False),
                    history_effective_interval_s=(rows[-1]['stamp']-rows[0]['stamp'])/10,
                    history_grid_timestamps_s=[r['target_stamp'] for r in rows] if 'target_stamp' in rows[-1] else None,
                    history_grid_max_offset_s=max(abs(r['stamp']-r['target_stamp']) for r in rows)
                        if 'target_stamp' in rows[-1] else None,
                    history_max_adjacent_gap_s=max(b['stamp']-a['stamp'] for a,b in zip(rows, rows[1:])),
                    history_max_offset_s=max(abs(r['stamp']-(rows[-1]['stamp']-(10-i)*contract['history_interval_s']))
                                             for i,r in enumerate(rows)),
                    vision_encoder_ms=rows[-1]['vision_ms'],
                    preprocessing_ms=context['preprocessing_ms'],
                    window_preparation_ms=context.get('window_preparation_ms'),
                    new_vision_frames=context.get('new_vision_frames', 1),
                    encoder_batch_size=context.get('encoder_batch_size', 1),
                    window_vision_ms=context.get('window_vision_ms', rows[-1]['vision_ms']),
                    window_preprocessing_ms=context.get('window_preprocessing_ms', context['preprocessing_ms']),
                    encoder_queue_wait_ms=context.get('encoder_queue_wait_ms'),
                    input_ready_age_ms=context.get('input_ready_age_ms'),
                    input_preparation=dict(state.get('input_preparation', {})),
                    total_inference_ms=context.get('window_vision_ms', rows[-1]['vision_ms'])+result['vldrive_ms'],
                    sensor_to_output_ms=(time.monotonic()-rows[-1]['stamp'])*1000)
                stream.write(json.dumps(result,allow_nan=False)+'\n')
                if controller is not None:
                    controller.submit(result)
                report.update(status='live_waypoints',outputs=report['outputs']+1,latest=result,
                              history_resets=history.resets,full_live_forward_tested=True)
                if cfg.get('navigation_token_config'):
                    report['navigation_tokens'] = model.language_model.tokenizer.describe()
                if view is not None:
                    view.show_prediction(result,context['front_jpeg'])
                write_report(output/'inference_report.json',report)
                print(f'frame {result["frame"]:05d} | tokens {result["selected_tokens"]}/106'
                      f' | VE {result["vision_encoder_ms"]:.0f} ms'
                      f' | Stage2 {result["vldrive_ms"]:.0f} ms'
                      f' | age {result["sensor_to_output_ms"]:.0f} ms | {instruction}',flush=True)
        if state['error']:
            raise RuntimeError('Sensor/encoder worker failed:\n'+state['error'])
        if not report['outputs']:
            report.update(status='live_no_complete_history',reason='No fresh navigable 11-frame window')
        report['live_elapsed_s']=time.monotonic()-started
        report['output_hz']=report['outputs']/report['live_elapsed_s']
    finally:
        stop.set()
        # Brake before waiting on GPU/capture workers or closing sensors.
        if controller is not None:
            controller.close()
            report['control']=controller.state
        if worker is not None:
            worker.join(timeout=15)
        if encoder_worker is not None:
            encoder_worker.join(timeout=15)
        report.update(streams=stream_rows(source),history_resets=history.resets,
                      discarded_skew=state['discarded_skew'],rejected_frames=state['rejected_frames'],
                      capture_queue_drops=state['capture_drops'],
                      input_preparation=dict(state.get('input_preparation', {})),
                      history_selection=dict(state.get('history_selection', {})),
                      sampling_status=dict(state.get('sampling_status', {})))
        source.close()
