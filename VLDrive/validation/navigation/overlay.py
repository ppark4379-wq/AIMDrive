"""Draw measured camera frames and navigation diagnostics, not model predictions."""
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size):
    for path in ('C:/Windows/Fonts/segoeui.ttf', 'C:/Windows/Fonts/arial.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_frame(row, window, encoder):
    canvas = Image.new('RGB', (1440, 900), '#101822')
    draw = ImageDraw.Draw(canvas)
    normal, small, heading = font(20), font(16), font(27)
    green, yellow, muted = '#78d6a0', '#ffd27a', '#aebccc'
    draw.text((24, 16), 'RECORDED CAMERA + FRAME NUMERIC NAVIGATION REVIEW', font=heading, fill='white')
    draw.text((24, 58), 'Input inspection only: no MLP / cross-attention / LLM predictions.', font=normal, fill=muted)
    if row.get('review_video_fps') is not None:
        draw.text((24, 82), 'Video review playback: %.1f fps (NOT original recording FPS).' % row['review_video_fps'],
                  font=small, fill=yellow)
    with Image.open(row['image_path']) as source:
        camera = source.convert('RGB')
        camera.thumbnail((940, 660))
        canvas.paste(camera, (20 + (940 - camera.width) // 2, 103 + (660 - camera.height) // 2))
    draw.rectangle((20, 101, 960, 765), outline='#344558', width=2)
    valid = row['path_valid']
    draw.text((985, 102), 'FRAME %04d  |  WINDOW %d' % (row['index'], window['window_id']), font=heading, fill='white')
    draw.text((985, 145), 'NUMERIC INPUT: ' + ('VALID' if valid else 'MASKED / INVALID'),
              font=normal, fill=green if valid else yellow)
    lines = ['Sync: ' + row['sync']['status'],
             'Route identity: ' + row['route_identity'],
             'Timestamp: ' + (str(row['sensor_source_ns']) + ' ns' if row['sensor_source_ns'] is not None else 'NOT RECORDED'),
             'Pose XY: ' + ('%.2f, %.2f m' % (row['pose']['x'], row['pose']['y'])
                           if row['finite_pose'] else 'missing / non-finite'),
             'Yaw: ' + ('%.1f deg (recorded)' % math.degrees(row['pose']['yaw_rad'])
                        if row['finite_pose'] else 'missing; NOT inferred from path'),
             'Nearest route: ' + ('%.2f m (diagnostic)' % row['nearest_route_distance_m']
                                  if row['nearest_route_distance_m'] is not None else 'unknown')]
    if row['sync']['skew_ms'] is not None:
        lines.append('Camera - pose time: %.1f ms' % row['sync']['skew_ms'])
    y = 185
    for line in lines:
        draw.text((985, y), line, font=small, fill=muted)
        y += 26
    p, event = row['projection'], row['event']
    if valid:
        lines = ['Along-route position: %.1f m' % p['progress_m'],
                 'Lateral error: %+.2f m (LEFT +)' % p['lat_err'],
                 'Heading error: %+.1f deg' % math.degrees(p['yaw_err']),
                 'Current curvature: %+.4f /m' % p['curv_now']]
        if event:
            lines += [event['direction'].upper() + ' bend',
                      'Start / end ahead: %+.1f / %+.1f m' %
                      (event['start_m'] - p['progress_m'], event['end_m'] - p['progress_m'])]
        else:
            lines.append('No bend event within %.0f m' % encoder.options['lookahead_m'])
    else:
        lines = ['No valid turn distances are shown.', 'Feature mask = false; no zero-error guess.',
                 'Reason: ' + ', '.join(row['invalid_reasons'])]
    for line in lines:
        # Wrap long diagnostic messages without clipping the side panel.
        words, current = line.split(), ''
        for word in words:
            candidate = (current + ' ' + word).strip()
            if draw.textlength(candidate, font=small) > 420:
                draw.text((985, y), current, font=small, fill=green if valid else yellow)
                current, y = word, y + 24
            else:
                current = candidate
        draw.text((985, y), current, font=small, fill=green if valid else yellow)
        y += 27
    # A top-down planned-route inset is a map, not a projection into the camera.
    box = (990, 540, 1420, 755)
    draw.rectangle(box, outline='#344558', fill='#162333', width=2)
    draw.text((1005, 550), 'TOP-DOWN MAP (not camera overlay)', font=small, fill=muted)
    points = np.asarray(encoder.route['points'])
    if row['finite_pose']:
        pose_xy = np.array([row['pose']['x'], row['pose']['y']])
        local = points - pose_xy
        c, s = math.cos(row['pose']['yaw_rad']), math.sin(row['pose']['yaw_rad'])
        # Ego x-forward, y-left. Display forward upward and left to the left.
        local = local @ np.array([[c, -s], [s, c]])
        origin, scale = (1205, 730), 2.
        screen = np.column_stack((origin[0] - scale * local[:, 1], origin[1] - scale * local[:, 0]))
        in_box = ((screen[:, 0] > box[0] + 8) & (screen[:, 0] < box[2] - 8) &
                  (screen[:, 1] > box[1] + 35) & (screen[:, 1] < box[3] - 8))
        for i in range(len(screen) - 1):
            if in_box[i] and in_box[i + 1]:
                draw.line([tuple(screen[i]), tuple(screen[i + 1])], fill='#65b7f2', width=3)
        draw.polygon([(origin[0], origin[1] - 9), (origin[0] - 6, origin[1] + 4),
                      (origin[0] + 6, origin[1] + 4)], fill=green if valid else yellow)
    semantic_warning = row['common_semantics_ok'] is False
    draw.text((24, 789), ('STALE COMMON TEXT: ' if semantic_warning else 'WINDOW COMMON TEXT: ') +
              (window['common_navigation'] or 'Unavailable (invalid source window)'),
              font=normal, fill=green if window['valid_window'] and not semantic_warning else yellow)
    draw.text((24, 824), 'NUMERIC PREVIEW: ' + (row['numeric_preview'] or 'Unavailable; missing sync / route / pose evidence.'),
              font=small, fill=muted)
    draw.text((24, 859), 'Rule-rendered review text is NOT a generated instruction. Frame times are never fabricated.',
              font=small, fill=muted)
    return canvas


def contact_sheet(images, destination):
    chosen = sorted(set([0, len(images) // 2, len(images) - 1]))
    sheet = Image.new('RGB', (1440, 900 * len(chosen)), '#101822')
    for offset, index in enumerate(chosen):
        with Image.open(images[index]) as frame:
            sheet.paste(frame.convert('RGB'), (0, 900 * offset))
    sheet.save(destination)
