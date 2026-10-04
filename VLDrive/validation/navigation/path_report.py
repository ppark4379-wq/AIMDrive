"""Reproducible path segmentation review, not a trained model's text output.

python -m validation.navigation.path_report --path <metric-path.txt> --output <new-directory>
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from instruction.global_path.encoder import PathEncoder
from instruction.navigation.selector import select_instruction, TEMPLATES
from instruction.global_path.preprocess.load import load_path
from instruction.global_path.preprocess.process import process_path


def projection_at(route, progress):
    """Illustrative ego exactly on the planned centerline, not recorded vehicle data."""
    index = min(len(route['curvature']) - 1, max(0, np.searchsorted(route['arc_m'], progress, side='right') - 1))
    return dict(progress_m=float(progress), lat_err=0., yaw_err=0.,
                curv_now=route['curvature'][index], segment=int(index))


def frame_preview(encoder, progress, instruction_options=None):
    projection = projection_at(encoder.route, progress)
    event = encoder.next_event(progress)
    selection = select_instruction(encoder, [projection], **(instruction_options or {}))
    features = encoder.encode(projection, event)
    if event is None:
        text = 'Follow the planned path. No qualifying bend in the next %.0f m.' % encoder.options['lookahead_m']
    else:
        start, end = event['start_m'] - progress, event['end_m'] - progress
        if start > 0:
            text = '%s bend starts in %.1f m and ends in %.1f m.' % (event['direction'].capitalize(), start, end)
        else:
            text = 'Following the %s bend; its end is %.1f m ahead.' % (event['direction'], end)
    text += ' Lateral error: 0.0 m (illustrative on-path ego).'
    return dict(progress_m=float(progress), common_navigation=selection['text'],
                common_kind=selection['kind'], event_id=event['id'] if event else None,
                numeric_preview=text, features=features)


def route_segments(route):
    """Partition the complete route; retain bends crossing the closed-route seam."""
    length, intervals = route['length_m'], []
    for event in route['events']:
        intervals.append((event['start_m'], min(event['end_m'], length), event))
        if event['end_m'] > length:
            intervals.append((0., event['end_m'] - length, event))
    segments, cursor = [], 0.
    for start, end, event in sorted(intervals, key=lambda item: item[0]):
        if start < cursor - 1e-6 or end <= start:
            raise ValueError('Bend events overlap or have invalid bounds')
        if start > cursor:
            segments.append(dict(kind='no_bend_event', start_m=cursor, end_m=start, event_id=None))
        segments.append(dict(kind=event['direction'], start_m=start, end_m=end, event_id=event['id']))
        cursor = end
    if cursor < length:
        segments.append(dict(kind='no_bend_event', start_m=cursor, end_m=length, event_id=None))
    return [dict(segment_id=i + 1, **segment) for i, segment in enumerate(segments)]


def build_report(source, options):
    loaded = load_path(source)
    if Path(source).suffix.lower() == '.txt':
        source_points = len(np.loadtxt(source, ndmin=2))
    else:
        document = json.loads(Path(source).read_text(encoding='utf-8'))
        source_points = len(document.get('route', document)['points'])
    route = process_path(loaded, **options['geometry'])
    encoder = PathEncoder(route, **options['encoder'])
    instruction_options = options.get('instruction', {})
    segments = route_segments(route)
    for segment in segments:
        segment['midpoint_example'] = frame_preview(encoder, (segment['start_m'] + segment['end_m']) / 2,
                                                   instruction_options)
    examples = []
    if route['events']:
        first = route['events'][0]
        examples.append(frame_preview(encoder, max(0., first['start_m'] - 30.), instruction_options))
        examples.append(frame_preview(encoder, (first['start_m'] + first['end_m']) / 2, instruction_options))
    for segment in sorted(segments, key=lambda row: row['end_m'] - row['start_m'], reverse=True):
        if segment['kind'] == 'no_bend_event':
            example = segment['midpoint_example']
            if example['event_id'] is None:
                examples.append(example)
                break
    return dict(schema='path-segmentation-review-v1', source_path=str(Path(source).resolve()),
        source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        source_points=source_points, distinct_points=len(loaded['points']),
        length_m=route['length_m'], closed=route['closed'],
        options=options, route=route, segments=segments, examples=examples,
        scope='Geometric bend candidates only. Not intersection, signal, speed, zone or situation labels.',
        instruction_note='numeric_preview is rule-rendered review text, NOT MLP/LLM output. Training uses common_navigation plus numeric features.',
        example_pose_note='Examples use an illustrative ego on the route: lateral/yaw error = 0. No recorded ego poses or future GT are used.')


def save_plot(report, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    route, events = report['route'], report['route']['events']
    xy, arc = np.asarray(route['points']), np.asarray(route['arc_m'])
    k = np.asarray(route['curvature'])
    colors = dict(left='#1775bd', right='#e86c18', no_bend_event='#acb8c3')
    frame_colors = np.full(len(k), colors['no_bend_event'], dtype=object)
    for segment in report['segments']:
        mask = (arc[:-1] >= segment['start_m'] - 1e-6) & (arc[:-1] < segment['end_m'] - 1e-6)
        frame_colors[mask] = colors[segment['kind']]
    fig = plt.figure(figsize=(17, 12), layout='constrained')
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 2.6], height_ratios=[1.15, 1.65, 1.2])
    path_ax = fig.add_subplot(grid[:, 0])
    path_ax.add_collection(LineCollection(np.stack((xy[:-1], xy[1:]), 1), colors=frame_colors, linewidths=3))
    path_ax.autoscale()
    path_ax.set_aspect('equal')
    path_ax.margins(.25, .04)
    path_ax.set(xlabel='Map X (m)', ylabel='Map Y (m)', title='Planned route and bend IDs')
    path_ax.grid(alpha=.18)
    path_ax.scatter(*xy[0], marker='*', s=100, c='black', zorder=5)
    path_ax.annotate('START / END', xy[0], xytext=(8, 3), textcoords='offset points', fontsize=9)
    for number, event in enumerate(events, 1):
        midpoint = ((event['start_m'] + event['end_m']) / 2) % route['length_m']
        point = np.array([np.interp(midpoint, arc, xy[:, d]) for d in range(2)])
        path_ax.scatter(*point, s=20, color=colors[event['direction']], zorder=5)
        offset = (18 if number % 2 else -25, 9 if number % 2 else -9)
        path_ax.annotate(str(number), point, xytext=offset, textcoords='offset points', fontsize=11,
            weight='bold', color=colors[event['direction']],
            arrowprops=dict(arrowstyle='-', color=colors[event['direction']], lw=.8))
    # Arrows encode route traversal, not global compass left/right.
    for progress in np.arange(180., route['length_m'], 300.):
        a, b = progress, min(progress + 15., route['length_m'])
        first = [np.interp(a, arc, xy[:, d]) for d in range(2)]
        last = [np.interp(b, arc, xy[:, d]) for d in range(2)]
        path_ax.annotate('', last, xytext=first, arrowprops=dict(arrowstyle='->', color='#303030', lw=1))
    path_ax.legend(handles=[Line2D([0], [0], color=colors[key], lw=3, label=label) for key, label in
        [('left', 'Left bend'), ('right', 'Right bend'), ('no_bend_event', 'Below event thresholds')]],
        loc='lower center', fontsize=9)

    curve_ax = fig.add_subplot(grid[0, 1])
    curve_ax.plot(arc[:-1], k, color='#3e4852', lw=.8)
    opts = report['options']['geometry']
    for threshold, style in [(opts['turn_curvature'], '--'), (opts['turn_exit_curvature'], ':')]:
        for sign in (-1, 1):
            curve_ax.axhline(sign * threshold, color='#777777', lw=.7, ls=style)
    for number, event in enumerate(events, 1):
        curve_ax.axvspan(event['start_m'], min(event['end_m'], route['length_m']),
                        color=colors[event['direction']], alpha=.16)
        curve_ax.text((event['start_m'] + event['end_m']) / 2, .085, str(number), fontsize=9, ha='center')
    curve_ax.set(xlabel='Arc distance from start s (m)', ylabel='Signed curvature (1/m)',
                 title='Left-positive curvature; dashed = seed, dotted = boundary threshold')
    curve_ax.grid(alpha=.15)

    table_ax = fig.add_subplot(grid[1, 1])
    table_ax.axis('off')
    rows = [[i, event['direction'].upper() + ('*' if event.get('reviewed_section') else ''),
             '%.1f - %.1f' % (event['start_m'], event['end_m']),
             '%+.1f' % event['angle_deg'], TEMPLATES[event['direction']]] for i, event in enumerate(events, 1)]
    table = table_ax.table(cellText=rows, colLabels=['ID', 'Bend', 'Start - end s (m)', 'Angle (deg)', 'Geometry label / instruction'],
                           colWidths=[.06, .1, .24, .14, .46], cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#d6dde3')
        if row == 0:
            cell.set_facecolor('#e7edf4')
            cell.set_text_props(weight='bold')
        elif column == 1:
            cell.set_text_props(color=colors[events[row - 1]['direction']], weight='bold')
    table_ax.set_title('Geometric bends (* = reviewed local thresholds; not intersection labels)', loc='left', fontsize=12)

    text_ax = fig.add_subplot(grid[2, 1])
    text_ax.axis('off')
    lines = ['FRAME-SPECIFIC REVIEW EXAMPLES (illustrative on-path ego)']
    for example in report['examples']:
        lines += ['', 's = %.1f m | common text: %s' % (example['progress_m'], example['common_navigation']),
                  example['numeric_preview'].replace(' Lateral error:', '\nLateral error:')]
    lines += ['', 'Seed |curvature| >= %.4f; boundary >= %.4f; merge gaps <= %.0f m.' %
              (opts['turn_curvature'], opts['turn_exit_curvature'], opts['merge_gap_m']),
              'Keep bends >= %.0f m and >= %.0f deg. Signed distances are measured ALONG the path.' %
              (opts['min_turn_length_m'], opts['min_turn_angle_deg']),
              'Gray does not mean perfectly straight. Numeric preview is rule-rendered, not generated by the MLP.']
    text_ax.text(0, .98, '\n'.join(lines), va='top', fontsize=10, linespacing=1.35)
    fig.suptitle('Global path: %s points / %.3f km / %d bend candidates' %
                 (format(report['source_points'], ','), report['length_m'] / 1000, len(events)), fontsize=17)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def write_report(source, output, options):
    report = build_report(source, options)
    target = Path(output)
    # A new directory avoids clobbering any earlier review or cache artifact.
    target.mkdir(parents=True, exist_ok=False)
    with (target / 'segments.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with (target / 'segments.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        columns = ['segment_id', 'kind', 'start_m', 'end_m', 'event_id', 'common_navigation', 'numeric_preview']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for segment in report['segments']:
            writer.writerow({**{key: segment[key] for key in columns[:5]},
                             **{key: segment['midpoint_example'][key] for key in columns[5:]}})
    save_plot(report, target / 'global_path_segments.png')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--options', type=Path, default=Path(__file__).resolve().parents[2] /
                        'instruction/global_path/preprocess/mlp_options.json')
    args = parser.parse_args()
    options = json.loads(args.options.read_text(encoding='utf-8'))
    result = write_report(args.path, args.output, options)
    print(json.dumps(dict(output=str(args.output.resolve()), points=result['source_points'],
        length_m=result['length_m'], bend_candidates=len(result['route']['events']), segments=len(result['segments']))))


if __name__ == '__main__':
    main()
