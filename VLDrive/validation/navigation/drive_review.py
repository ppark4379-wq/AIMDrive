"""Whole-drive input viewer using the training geometry and window selector.

Each displayed frame ends a trailing 11-frame window (source interval 2).
All source positions are inspected, rather than only the training stride.
No model is loaded. Original recordings are read only.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageOps

from instruction.global_path.encoder import PathEncoder
from instruction.global_path.preprocess.load import load_path
from instruction.global_path.preprocess.process import process_path
from instruction.navigation.selector import select_instruction, TEMPLATES
from .recording import load_recording, digest
from .replay import DEFAULT_OPTIONS, prepare_frames, build_window, numeric_preview


def add_frame_navigation(report):
    """Frame-local preview is independent of whether a training window exists.

    Missing poses receive only an explicitly marked generic route instruction,
    never an invented turn, carried-over distance, or trainable feature vector.
    """
    encoder = PathEncoder(report['route'], **report['options'].get('encoder', {}))
    for row in report['frames']:
        projection = row['projection'] if row['path_valid'] else None
        selection = (select_instruction(encoder, [projection], **report['options'].get('instruction', {}))
                     if projection is not None else None)
        event = ((selection['event'] or encoder.next_event(projection['progress_m']))
                 if selection is not None else None)
        row.update(frame_navigation=selection['text'] if selection else TEMPLATES['follow'],
                   frame_navigation_source='current_pose' if selection else 'generic_fallback_no_valid_pose',
                   frame_kind=selection['kind'] if selection else 'follow',
                   frame_event=event,
                   frame_features=encoder.encode(projection, event) if selection else None,
                   frame_numeric_preview=numeric_preview(projection, event, encoder.options['lookahead_m']))
    return report


def build_drive_review(recording, options, *, length=11, interval=2):
    if type(length) is not int or type(interval) is not int or min(length, interval) < 1:
        raise ValueError('Window length and interval must be positive integers')
    if set(options) - {'geometry', 'encoder', 'instruction', 'max_sensor_gap_ms'}:
        raise ValueError('Unknown preprocessing options')
    if not recording.get('route'):
        raise ValueError('Whole-drive review requires a recorded planned route')
    planned = load_path(recording['route'])
    encoder = PathEncoder(process_path(planned, **options.get('geometry', {})),
                          **options.get('encoder', {}))
    gap = options.get('max_sensor_gap_ms', 150.)
    frames = prepare_frames(recording, encoder, 'matches_recorded_planned_path', max_gap_ms=gap)
    history = (length - 1) * interval
    rows, transitions = [], []
    for position, frame in enumerate(frames):
        complete = position >= history
        window = build_window(frames, encoder, position - history if complete else position,
                              length=length if complete else 1, interval=interval,
                              max_gap_ms=gap, instruction_options=options.get('instruction'))
        current = window['frames'][-1]
        valid_window = complete and window['valid_window']
        reasons = sorted({reason for source in frames[max(0, position-history):position+1]
                          for reason in source['invalid_reasons']})
        if not complete:
            reasons.append('insufficient_history')
        elif not window['continuous_source_span']:
            reasons.append('source_frame_gap')
        row = dict(current, position=position, valid_window=valid_window,
                   common_navigation=window['common_navigation'] if valid_window else None,
                   common_kind=window['common_kind'] if valid_window else None,
                   window_start_index=frames[position-history]['index'] if complete else None,
                   window_frame_indices=window['frame_indices'] if complete else [],
                   window_invalid_reasons=reasons,
                   model_features=current['features'] if valid_window else None,
                   common_semantics_ok=current['common_semantics_ok'] if valid_window else None)
        rows.append(row)
        key = row['common_kind'] if valid_window else 'invalid_window'
        if not transitions or transitions[-1]['kind'] != key:
            transitions.append(dict(position=position, index=row['index'], kind=key,
                                    text=row['common_navigation']))
    summary = dict(total_frames=len(rows), valid_frames=sum(r['path_valid'] for r in rows),
                   valid_windows=sum(r['valid_window'] for r in rows),
                   insufficient_history=min(history, len(rows)),
                   invalid_frame_reasons=dict(Counter(reason for r in rows for reason in r['invalid_reasons'])),
                   instruction_counts=dict(Counter(r['common_kind'] for r in rows if r['valid_window'])),
                   semantic_warnings=sum(r['common_semantics_ok'] is False for r in rows))
    return add_frame_navigation(dict(schema='whole-drive-navigation-review-v1', episode_id=recording['episode_id'],
                recording_source=recording['source_path'], sources=recording['sources'], options=options,
                pose_policy=recording.get('pose_policy'),
                window_length=length, frame_interval=interval, reviewed_window_stride=1,
                scope='Pre-MLP input review; numeric prose is rule-rendered, not model output. '
                      'Sensor Record scene association does not verify hardware exposure time.',
                route=encoder.route, frames=rows, transitions=transitions, summary=summary))


def write_drive_review(report, output):
    target = Path(output).resolve()
    source = Path(report['recording_source']).resolve()
    recording_root = source.parent if source.name.startswith('episode_') else source
    if target == recording_root or recording_root in target.parents:
        raise ValueError('Output must be outside the original recording')
    target.mkdir(parents=True, exist_ok=False)
    (target / 'images').mkdir()
    for i, row in enumerate(report['frames']):
        name = 'images/%06d.jpg' % i
        with Image.open(row['image_path']) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            image.thumbnail((960, 600))
            image.save(target / name, quality=85)
        row['image_sha256'] = digest(row['image_path'])
        row['preview_image'] = name
        if i % 250 == 0:
            print('Preview images: %d / %d' % (i, len(report['frames'])), flush=True)
    write_review_metadata(report, target)
    return target


def write_review_metadata(report, target):
    """Write viewer/data to a review directory whose images already exist."""
    target = Path(target)
    if not (target / 'images').is_dir():
        raise ValueError('Expected an existing review images directory')
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False)
    (target / 'review.json').write_text(payload + '\n', encoding='utf-8')
    # Escaping '<' prevents source text from closing the inline JSON script.
    template = Path(__file__).with_name('drive_viewer.html').read_text(encoding='utf-8')
    (target / 'viewer.html').write_text(template.replace('__REVIEW_DATA__', payload.replace('<', '\\u003c')),
                                       encoding='utf-8')
    fields = ['frame', 'source_ns', 'valid_frame', 'valid_window', 'instruction',
              'frame_instruction', 'frame_instruction_source', 'frame_ego_relation', 'frame_path_context',
              'progress_m', 'lateral_error_m', 'yaw_error_deg', 'current_curvature_per_m',
              'event_direction', 'start_distance_m', 'end_distance_m', 'event_curvature_per_m',
              'ego_relation', 'path_context', 'invalid_reasons']
    with (target / 'frames.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report['frames']:
            p, event, features = row['projection'], row['event'], row['model_features']
            writer.writerow(dict(frame=row['index'], source_ns=row['sensor_source_ns'],
                valid_frame=row['path_valid'], valid_window=row['valid_window'], instruction=row['common_navigation'],
                frame_instruction=row['frame_navigation'], frame_instruction_source=row['frame_navigation_source'],
                frame_ego_relation=json.dumps(row['frame_features']['ego_relation']) if row['frame_features'] else None,
                frame_path_context=json.dumps(row['frame_features']['path_context']) if row['frame_features'] else None,
                progress_m=p['progress_m'] if p else None, lateral_error_m=p['lat_err'] if p else None,
                yaw_error_deg=math.degrees(p['yaw_err']) if p else None,
                current_curvature_per_m=p['curv_now'] if p else None,
                event_direction=event['direction'] if event else None,
                start_distance_m=event['start_m']-p['progress_m'] if p and event else None,
                end_distance_m=event['end_m']-p['progress_m'] if p and event else None,
                event_curvature_per_m=event['curv_event'] if event else None,
                ego_relation=json.dumps(features['ego_relation']) if features else None,
                path_context=json.dumps(features['path_context']) if features else None,
                invalid_reasons='; '.join(row['window_invalid_reasons'])))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--map-root', type=Path)
    parser.add_argument('--route-definition', type=Path)
    parser.add_argument('--options', type=Path, default=DEFAULT_OPTIONS)
    parser.add_argument('--window-length', type=int, default=11)
    parser.add_argument('--frame-interval', type=int, default=2)
    args = parser.parse_args()
    recording = load_recording(args.recording, map_root=args.map_root, route_definition=args.route_definition)
    options = json.loads(args.options.read_text(encoding='utf-8'))
    report = build_drive_review(recording, options, length=args.window_length, interval=args.frame_interval)
    output = write_drive_review(report, args.output)
    print(json.dumps(dict(output=str(output), **report['summary']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
