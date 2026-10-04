"""Review per-frame numeric navigation on recorded camera frames, without training.

Run from VLDrive: python -m validation.navigation.replay --recording <run> --output <new-dir>
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from instruction.global_path.encoder import PathEncoder
from instruction.global_path.preprocess.load import load_path
from instruction.global_path.preprocess.process import process_path
from instruction.navigation.selector import select_instruction
from .overlay import draw_frame, contact_sheet
from .recording import load_recording, finite, synchronization, digest

DEFAULT_OPTIONS = Path(__file__).resolve().parents[2] / 'instruction/global_path/preprocess/mlp_options.json'


def nearest_distance(encoder, pose):
    xy = np.array([pose['x'], pose['y']])
    fractions = np.clip(((xy - encoder.points[:-1]) * encoder.vectors).sum(1) / encoder.lengths2, 0, 1)
    offsets = xy - (encoder.points[:-1] + fractions[:, None] * encoder.vectors)
    return float(np.linalg.norm(offsets, axis=1).min())


def prepare_frames(recording, encoder, path_identity, *, max_skew_ms=50., max_gap_ms=150.):
    if not math.isfinite(max_gap_ms) or max_gap_ms <= 0:
        raise ValueError('max_gap_ms must be positive')
    frames, previous_s, previous = [], None, None
    for frame in recording['frames']:
        stamp = frame.get('sensor_source_ns')
        if (previous is None or stamp is None or previous.get('sensor_source_ns') is None or
                frame['index'] != previous['index'] + 1 or
                not 0 < stamp - previous['sensor_source_ns'] <= max_gap_ms * 1e6):
            previous_s = None
        raw_pose = {key: frame.get(key) for key in ('x', 'y', 'yaw_rad')}
        finite_pose = all(finite(value) for value in raw_pose.values())
        pose = {key: value if finite(value) else None for key, value in raw_pose.items()}
        sync = synchronization(frame, max_skew_ms)
        candidate = encoder.project(pose, previous_s) if finite_pose and frame.get('valid') is not False else None
        reasons = []
        if not sync['valid']:
            reasons.append(sync['status'])
        if path_identity != 'matches_recorded_planned_path':
            reasons.append(path_identity)
        if not finite_pose:
            reasons.append('missing_or_invalid_recorded_pose')
        elif candidate is None:
            reasons.append('route_projection_rejected')
        if frame.get('valid') is False:
            reasons.append('source_marked_invalid')
        usable = not reasons
        projection = candidate if usable else None
        previous_s = candidate['progress_m'] if usable else None
        previous = frame
        frames.append(dict(index=frame['index'], sensor_source_ns=stamp,
                           camera_source_ns=frame.get('camera_source_ns'), pose_source_ns=frame.get('pose_source_ns'),
                           image_path=frame['image_path'], pose=pose, finite_pose=finite_pose,
                           sync=sync, route_identity=path_identity, invalid_reasons=reasons,
                           path_valid=usable, projection=projection, candidate_projection=candidate,
                           nearest_route_distance_m=nearest_distance(encoder, pose) if finite_pose else None))
    return frames


def numeric_preview(projection, event, horizon):
    if projection is None:
        return None
    progress = projection['progress_m']
    if event is None:
        message = 'No bend event within %.0f m.' % horizon
    elif event['start_m'] > progress:
        message = '%s bend starts in %.1f m; ends in %.1f m.' % (
            event['direction'].capitalize(), event['start_m'] - progress, event['end_m'] - progress)
    elif event['end_m'] >= progress:
        message = 'Inside %s bend; ends in %.1f m.' % (event['direction'], event['end_m'] - progress)
    else:
        message = 'Selected bend ended %.1f m ago (common text is stale).' % (progress - event['end_m'])
    return message + ' Lateral error %+.2f m; yaw error %+.1f deg.' % (
        projection['lat_err'], math.degrees(projection['yaw_err']))


def build_window(frames, encoder, start, *, window_id=0, length=11, interval=2,
                 max_gap_ms=150., instruction_options=None):
    """Exactly the same all-source-span continuity check as the training dataset."""
    if any(type(v) is not int or v < minimum for v, minimum in ((start, 0), (length, 1), (interval, 1))):
        raise ValueError('Invalid window parameters')
    span = (length - 1) * interval + 1
    source = frames[start:start + span]
    if len(source) != span:
        raise ValueError('Not enough recorded frames for the requested complete window')
    selected = source[::interval]
    continuous = all(b['index'] == a['index'] + 1 and
                     a['sensor_source_ns'] is not None and b['sensor_source_ns'] is not None and
                     0 < b['sensor_source_ns'] - a['sensor_source_ns'] <= max_gap_ms * 1e6
                     for a, b in zip(source, source[1:]))
    valid = all(row['path_valid'] for row in source) and continuous
    selection = select_instruction(encoder, [row['projection'] for row in selected],
                                   **(instruction_options or {})) if valid else None
    result = dict(window_id=window_id, source_start=start, source_span=span,
                  frame_indices=[row['index'] for row in selected], valid_window=valid,
                  continuous_source_span=continuous, common_navigation=selection['text'] if selection else None,
                  common_kind=selection['kind'] if selection else None, frames=[])
    for original in selected:
        row = dict(original)
        p = row['projection']
        event = (selection['event'] if selection and selection['event'] is not None else
                 encoder.next_event(p['progress_m']) if p else None)
        row['event'] = event
        row['features'] = encoder.encode(p, event)
        row['numeric_preview'] = numeric_preview(p, event, encoder.options['lookahead_m'])
        # Flag an obviously stale fixed direction/straight common sentence;
        # this is a diagnostic, not a new runtime selection policy.
        if selection and selection['kind'] == 'straight':
            row['common_semantics_ok'] = event is None
        elif selection and selection['event'] is not None:
            row['common_semantics_ok'] = event['start_m'] <= p['progress_m'] <= event['end_m']
        else:
            row['common_semantics_ok'] = True if selection and selection['kind'] == 'follow' else None
        result['frames'].append(row)
    result['semantic_warning_count'] = sum(row['common_semantics_ok'] is False for row in result['frames'])
    return result


VIEWER = '''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Recorded navigation review</title><style>
body{margin:16px;background:#101822;color:#eef4fa;font:16px sans-serif}
img{display:block;max-width:100%;max-height:80vh;margin:12px 0}
input{width:60vw}button{padding:8px 20px}small{color:#ffd27a}
</style><h2>주행 프레임과 경로 수치 검증</h2>
<p>공통 문장은 윈도우별로 고정하고, 수치는 각 프레임 pose로 계산합니다. 모델 출력이 아닙니다.</p>
<small>촬영 시각이 없으면 번호순 미리보기입니다. 윈도우 사이 이동도 실제 재생 시간이 아닙니다.</small>
<p><button id="play">재생</button> <input id="slider" type="range" min="0" value="0"><span id="label"></span></p>
<img id="frame" alt="Recorded camera and navigation diagnostics">
<script>const frames=__FRAMES__;const s=document.getElementById('slider');
const image=document.getElementById('frame');const label=document.getElementById('label');
let timer=null;s.max=frames.length-1;
function show(){const f=frames[+s.value];image.src=f.path;label.textContent=` Window ${f.window} / Frame ${f.index}`;}
function stop(){clearTimeout(timer);timer=null;document.getElementById('play').textContent='재생';}
function next(){if(+s.value>=frames.length-1){stop();return;}const a=frames[+s.value],b=frames[+s.value+1];
const dt=a.window===b.window&&a.stamp!==null&&b.stamp!==null?(b.stamp-a.stamp)/1e6:500;
timer=setTimeout(()=>{s.value=+s.value+1;show();next();},Math.max(10,Math.min(10000,dt)));}
s.oninput=()=>{stop();show();};document.getElementById('play').onclick=()=>{
if(timer!==null){stop();return;}if(+s.value===frames.length-1)s.value=0;show();
document.getElementById('play').textContent='정지';next();};show();</script></html>'''


def write_replay(recording, output, options, *, path=None, start=0, windows=1, length=11,
                 interval=2, stride=5, max_skew_ms=50., allow_unverified_preview=False, video_fps=None):
    if any(type(value) is not int or value < 1 for value in (windows, stride)):
        raise ValueError('windows and stride must be positive integers')
    if video_fps is not None and (not math.isfinite(video_fps) or video_fps <= 0):
        raise ValueError('video_fps must be positive')
    if set(options) - {'geometry', 'encoder', 'instruction', 'max_sensor_gap_ms'}:
        raise ValueError('Unknown path preprocessing option')
    planned = load_path(path) if path else load_path(recording['route']) if recording.get('route') else None
    if planned is None:
        raise ValueError('Recording has no planned path; provide an explicit --path for diagnostic review')
    identity = 'planned_path_not_recorded'
    if recording.get('route'):
        original = load_path(recording['route'])
        identity = ('matches_recorded_planned_path' if original['route_sha256'] == planned['route_sha256']
                    else 'different_recorded_planned_path')
    encoder = PathEncoder(process_path(planned, **options.get('geometry', {})), **options.get('encoder', {}))
    select_instruction(encoder, [dict(progress_m=0.)], **options.get('instruction', {}))
    gap = options.get('max_sensor_gap_ms', 150.)
    frames = prepare_frames(recording, encoder, identity, max_skew_ms=max_skew_ms, max_gap_ms=gap)
    reviews = [build_window(frames, encoder, start + i * stride, window_id=i, length=length,
                           interval=interval, max_gap_ms=gap, instruction_options=options.get('instruction'))
               for i in range(windows)]
    if not allow_unverified_preview and any(not window['valid_window'] for window in reviews):
        raise ValueError('Requested window lacks synchronized pose / matching planned route / continuity. '
                         'Use --allow-unverified-preview only to inspect invalid inputs; it never unmasks them.')
    target = Path(output).resolve()
    source = Path(recording['source_path']).resolve()
    if target == source or source in target.parents:
        raise ValueError('Review output must be outside the original recording')
    # Exclusive directory creation: no recording or previous review is overwritten.
    target.mkdir(parents=True, exist_ok=False)
    (target / 'frames').mkdir()
    playback, image_paths, writer = [], [], None
    try:
        if video_fps is not None:
            import cv2
            writer = cv2.VideoWriter(str(target / 'review.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                     video_fps, (1440, 900))
            if not writer.isOpened():
                raise RuntimeError('MP4 writer failed; PNG and HTML output do not require OpenCV')
        for window in reviews:
            for row in window['frames']:
                name = 'frames/window_%03d_frame_%06d.png' % (window['window_id'], row['index'])
                row['review_video_fps'] = video_fps
                image = draw_frame(row, window, encoder)
                image.save(target / name)
                image_paths.append(target / name)
                row['image_sha256'] = digest(row['image_path'])
                row['review_image'] = name
                playback.append(dict(path=name, window=window['window_id'], index=row['index'], stamp=row['sensor_source_ns']))
                if writer is not None:
                    import cv2
                    writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
    contact_sheet(image_paths, target / 'contact_sheet.png')
    (target / 'viewer.html').write_text(VIEWER.replace('__FRAMES__', json.dumps(playback)), encoding='utf-8')
    counts = Counter(reason for row in frames for reason in row['invalid_reasons'])
    result = dict(schema='recorded-navigation-review-v1', recording_source=recording['source_path'],
                  episode_id=recording['episode_id'], recording_format=recording['format'],
                  source_provenance=recording['sources'], planned_path=planned, processed_route=encoder.route,
                  explicit_path_source=dict(path=str(Path(path).resolve()), sha256=digest(path)) if path else None,
                  options=options, max_skew_ms=max_skew_ms,
                  window_options=dict(start=start, windows=windows, length=length, interval=interval, stride=stride),
                  source_frame_count=len(frames), valid_source_frames=sum(row['path_valid'] for row in frames),
                  invalid_source_reasons=dict(counts), valid_windows=sum(window['valid_window'] for window in reviews),
                  video_fps=video_fps,
                  video_note='Fixed-rate review playback, NOT recovered recording timing. Overlapping windows repeat frames.',
                  text_note='Numeric preview is deterministic review text; actual model input is common text + 8 numeric features.',
                  scope='Input synchronization, route projection and per-frame navigation inspection only. '
                        'Not model convergence, waypoint accuracy or closed-loop safety validation.',
                  windows=reviews)
    (target / 'review.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--path', type=Path, help='Optional metric XY path; must match the recorded planned path for valid inputs')
    parser.add_argument('--options', type=Path, default=DEFAULT_OPTIONS)
    parser.add_argument('--camera', choices=['front', 'left', 'right', 'rear'], default='front')
    parser.add_argument('--image-root', type=Path, help='Explicit portable episode camera-file root')
    parser.add_argument('--map-root', type=Path, help='Audited Sensor Record map relocation')
    parser.add_argument('--route-definition', type=Path, help='Audited Sensor Record route relocation')
    parser.add_argument('--window-start', type=int, default=0, help='Position in chronological recording frames')
    parser.add_argument('--windows', type=int, default=1)
    parser.add_argument('--window-length', type=int, default=11)
    parser.add_argument('--frame-interval', type=int, default=2)
    parser.add_argument('--window-stride', type=int, default=5)
    parser.add_argument('--max-skew-ms', type=float, default=50.)
    parser.add_argument('--video-fps', type=float, help='Optional fixed review playback rate; not source FPS')
    parser.add_argument('--allow-unverified-preview', action='store_true', help='Show invalid input warnings, never enable invalid features')
    args = parser.parse_args()
    recording = load_recording(args.recording, camera=args.camera, image_root=args.image_root,
                               map_root=args.map_root, route_definition=args.route_definition)
    options = json.loads(args.options.read_text(encoding='utf-8'))
    result = write_replay(recording, args.output, options, path=args.path, start=args.window_start,
                          windows=args.windows, length=args.window_length, interval=args.frame_interval,
                          stride=args.window_stride, max_skew_ms=args.max_skew_ms,
                          allow_unverified_preview=args.allow_unverified_preview, video_fps=args.video_fps)
    print(json.dumps(dict(output=str(args.output.resolve()), source_frames=result['source_frame_count'],
                          valid_source_frames=result['valid_source_frames'], valid_windows=result['valid_windows'],
                          invalid_reasons=result['invalid_source_reasons'])))


if __name__ == '__main__':
    main()
