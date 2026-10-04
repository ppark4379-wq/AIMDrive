"""Plot the ACTUAL common-text selector regions, separately from geometric bends.

Each position is a hypothetical window's FIRST-frame progress on the planned
path. Numeric examples assume on-path ego, not recorded driving or model output.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from instruction.global_path.encoder import PathEncoder
from instruction.navigation.selector import select_instruction, straight_exclusion_bounds, TEMPLATES
from .path_report import build_report, frame_preview, projection_at, save_plot

DEFAULT_OPTIONS = Path(__file__).resolve().parents[2] / 'instruction/global_path/preprocess/mlp_options.json'
LABELS = dict(follow='계획 경로 추종', straight='직진 경로 추종', left='좌측 굽음', right='우측 굽음',
              gentle_left='완만한 좌측 굽음', gentle_right='완만한 우측 굽음',
              sharp_left='급한 좌측 굽음', sharp_right='급한 우측 굽음',
              left_then_right='좌측 → 우측 굽음', right_then_left='우측 → 좌측 굽음')
COLORS = dict(follow='#94a3b8', straight='#21966f', left='#277ac9', right='#e88820',
              gentle_left='#79bcf1', gentle_right='#ebbf66', sharp_left='#164d9b', sharp_right='#c44620',
              left_then_right='#9253ba', right_then_left='#d44f90')


def instruction_regions(encoder, instruction_options=None):
    """Exact open-interval selector changes; event endpoint ties remain explicit.

    next_event and the selector can change at event start/end, or when those
    boundaries cross the lookahead. The +1e-6 used for compound-event lookup is
    retained. Closed-route candidates also include adjacent lap boundaries.
    """
    length, horizon = encoder.route['length_m'], encoder.options['lookahead_m']
    cuts = {0., length}
    # Retain reviewed range boundaries even when both neighbouring sentences
    # are "follow", so the user's existing map region IDs remain traceable.
    review_bounds = straight_exclusion_bounds(encoder, (instruction_options or {}).get('straight_exclusions', ()))
    review_bounds += [(s['start_m'], s['end_m']) for s in encoder.route.get('reviewed_sections', [])]
    review_cuts = {value for bounds in review_bounds for value in bounds}
    cuts.update(review_cuts)
    shifts = (-length, 0., length) if encoder.route['closed'] else (0.,)
    for event in encoder.route['events']:
        for shift in shifts:
            start, end = event['start_m'] + shift, event['end_m'] + shift
            cuts.update(value for value in (start, end, start - horizon, end - horizon, end - horizon + 1e-6)
                        if 0 < value < length)
    cuts = sorted(cuts)
    regions = []
    for start, end in zip(cuts, cuts[1:]):
        selection = select_instruction(encoder, [projection_at(encoder.route, (start + end) / 2)],
                                       **(instruction_options or {}))
        if regions and regions[-1]['kind'] == selection['kind'] and start not in review_cuts:
            regions[-1]['end_m'] = end
        else:
            regions.append(dict(start_m=start, end_m=end, kind=selection['kind'], text=selection['text']))
    for number, region in enumerate(regions, 1):
        region['region_id'] = number
        region['midpoint_example'] = frame_preview(encoder, (region['start_m'] + region['end_m']) / 2,
                                                  instruction_options)
    return regions


def build_instruction_report(source, options):
    report = build_report(source, options)
    encoder = PathEncoder(report['route'], **options['encoder'])
    regions = instruction_regions(encoder, options.get('instruction'))
    if report['route'].get('reviewed_sections'):
        # Compare against the unrefined catalog so only region 5 acquires
        # sublabels 5a/5b/5c; the other map labels keep their original meaning.
        base_route = {key: value for key, value in report['route'].items() if key != 'reviewed_sections'}
        base_route['events'] = [e for e in base_route['events'] if not e.get('reviewed_section')]
        baseline = instruction_regions(PathEncoder(base_route, **options['encoder']), options.get('instruction'))
        groups = defaultdict(list)
        for region in regions:
            midpoint = (region['start_m'] + region['end_m']) / 2
            original = next(r for r in baseline if r['start_m'] <= midpoint < r['end_m'])
            groups[original['region_id']].append(region)
        for identity, group in groups.items():
            for index, region in enumerate(group):
                region['display_id'] = str(identity) + (chr(ord('a') + index) if len(group) > 1 else '')
    counts = defaultdict(float)
    for region in regions:
        counts[region['kind']] += region['end_m'] - region['start_m']
    report.update(schema='route-instruction-review-v1', instruction_regions=regions,
                  instruction_length_m=dict(counts),
                  unused_templates=[kind for kind in TEMPLATES if kind not in counts],
                  anchor_note='Region position is the FIRST frame of a hypothetical window. '
                              'Actual windows retain this sentence and recompute numeric features per frame.',
                  interval_note='Intervals describe the open interval interiors. At an exact boundary, '
                                'use select_instruction directly for its inclusive endpoint tie policy.',
                  common_text_note='Text is selected by the existing rule, not generated by an MLP or LLM. '
                                   'A geometric left/right bend does not necessarily select a left/right sentence.')
    return report


def save_instruction_plot(report, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib import font_manager

    korean_font = Path('C:/Windows/Fonts/malgun.ttf')
    if korean_font.is_file():
        font_manager.fontManager.addfont(str(korean_font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(korean_font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    route, regions = report['route'], report['instruction_regions']
    xy, arc = np.asarray(route['points']), np.asarray(route['arc_m'])
    mids = (arc[:-1] + arc[1:]) / 2
    line_colors = np.full(len(mids), COLORS['follow'], dtype=object)
    for region in regions:
        line_colors[(mids >= region['start_m']) & (mids < region['end_m'])] = COLORS[region['kind']]
    fig = plt.figure(figsize=(19, max(13, 6 + len(regions) * .25)), layout='constrained')
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 2.7], height_ratios=[1.15, 3.4, 1.2])
    map_ax = fig.add_subplot(grid[:, 0])
    map_ax.add_collection(LineCollection(np.stack((xy[:-1], xy[1:]), axis=1), colors=line_colors, linewidths=4))
    map_ax.autoscale()
    map_ax.set_aspect('equal')
    map_ax.margins(.28, .05)
    map_ax.set(xlabel='지도 X (m)', ylabel='지도 Y (m)', title='실제 선택되는 공통 문장별 색상')
    map_ax.grid(alpha=.15)
    map_ax.scatter(*xy[0], marker='*', s=100, color='black', zorder=5)
    map_ax.annotate('출발 / 종료', xy[0], xytext=(8, 4), textcoords='offset points', fontsize=10)
    for region in regions:
        midpoint = (region['start_m'] + region['end_m']) / 2
        point = [np.interp(midpoint, arc, xy[:, dim]) for dim in range(2)]
        offset = (14 if region['region_id'] % 2 else -24, 5 if region['region_id'] % 2 else -9)
        map_ax.annotate(region.get('display_id', str(region['region_id'])), point, xytext=offset, textcoords='offset points',
                        fontsize=10, color=COLORS[region['kind']], weight='bold',
                        arrowprops=dict(arrowstyle='-', color=COLORS[region['kind']], lw=.6))
    for s in np.arange(200., route['length_m'], 350.):
        a = [np.interp(s, arc, xy[:, dim]) for dim in range(2)]
        b = [np.interp(min(s + 15, route['length_m']), arc, xy[:, dim]) for dim in range(2)]
        map_ax.annotate('', b, xytext=a, arrowprops=dict(arrowstyle='->', color='#303030'))

    active = [kind for kind in TEMPLATES if kind in report['instruction_length_m']]
    axis = fig.add_subplot(grid[0, 1])
    for region in regions:
        index = active.index(region['kind'])
        axis.broken_barh([(region['start_m'], region['end_m'] - region['start_m'])],
                        (index - .35, .7), facecolors=COLORS[region['kind']])
    axis.set(yticks=range(len(active)), yticklabels=[LABELS[kind] for kind in active],
             xlim=(0, route['length_m']), xlabel='윈도우 첫 프레임의 경로상 위치 s (m)',
             title='현재 selector 룰의 문장 선택 구간 — 전방 60m 기준')
    axis.grid(axis='x', alpha=.18)

    table_axis = fig.add_subplot(grid[1, 1])
    table_axis.axis('off')
    rows = [[r.get('display_id', r['region_id']), '%.1f – %.1f' % (r['start_m'], r['end_m']), LABELS[r['kind']], r['text']]
            for r in regions]
    table = table_axis.table(cellText=rows, colLabels=['지도 번호', '시작 – 끝 s (m)', '뜻', '실제로 선택하는 공통 navigation 문장'],
                            colWidths=[.08, .19, .20, .53], cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.55)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#dce3ea')
        if row == 0:
            cell.set_facecolor('#e8eff6')
            cell.set_text_props(weight='bold')
        elif column in (0, 2):
            cell.set_text_props(color=COLORS[regions[row - 1]['kind']], weight='bold')

    examples_axis = fig.add_subplot(grid[2, 1])
    examples_axis.axis('off')
    options = report['options']
    encoder = PathEncoder(route, **options['encoder'])
    positions = [max(0., route['events'][0]['start_m'] - 30.)] if route['events'] else []
    refined = [e for e in route['events'] if e.get('reviewed_section')]
    if refined:
        event = refined[0]
        section = next(s for s in route['reviewed_sections'] if s['id'] == event['reviewed_section'])
        positions = [(section['start_m'] + event['start_m'] - encoder.options['lookahead_m']) / 2,
                     max(section['start_m'], event['start_m'] - 30.),
                     (event['start_m'] + event['end_m']) / 2,
                     (event['end_m'] + section['end_m']) / 2]
    for kind in (('straight',) if refined else ('straight', 'left_then_right', 'sharp_left', 'right')):
        candidates = [r for r in regions if r['kind'] == kind]
        if candidates:
            region = max(candidates, key=lambda r: r['end_m'] - r['start_m'])
            positions.append((region['start_m'] + region['end_m']) / 2)
    lines = ['수치 예시: 차량이 경로 중심에 있고 경로와 같은 방향을 바라본다고 가정 (실제 주행 아님)']
    for position in positions:
        preview = frame_preview(encoder, position, options.get('instruction'))
        p = projection_at(route, position)
        event = encoder.next_event(position)
        if event:
            detail = '%s 구간 시작까지 %+.1fm / 종료까지 %+.1fm' % (
                '좌측 굽음' if event['direction'] == 'left' else '우측 굽음',
                event['start_m'] - position, event['end_m'] - position)
        else:
            detail = '전방 60m 안에 회전 이벤트 없음'
        lines += ['s=%.1fm  |  %s' % (position, preview['common_navigation']),
                  '    ' + detail + ' | 횡오차 0m, 방향오차 0° (가정값)']
    lines += ['', ('5b: 실제 굽음은 %.1f–%.1fm. 전방 60m에서 미리 안내하므로 문장 구간은 더 일찍 시작.' %
                   (refined[0]['start_m'], refined[0]['end_m']) if refined else
                   '중요: 회전의 시작·끝과 문장 선택 구간은 다름. 짧은 회전에서는 planned path 문장이 유지될 수 있음.'),
              '문장은 윈도우별 하나. 거리·곡률·횡오차·방향오차는 각 프레임마다 계산해 MLP에 벡터로 전달.',
              '0°·0m 오차는 시각화 가정일 뿐이며, 실제 입력은 각 프레임의 기록된 pose로 계산.']
    examples_axis.text(0, 1, '\n'.join(lines), va='top', fontsize=10, linespacing=1.45)
    fig.suptitle('현재 Global Path의 구간별 Navigation | %.1fm · 기하 회전 %d개 · 실제 문장 %d종' %
                 (route['length_m'], len(route['events']), len(active)), fontsize=18)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def write_instruction_report(source, output, options):
    report = build_instruction_report(source, options)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    (target / 'instruction_regions.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (target / 'instruction_regions.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        columns = ['region_id', 'display_id', 'start_m', 'end_m', 'kind', 'text', 'midpoint_numeric_preview']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for region in report['instruction_regions']:
            writer.writerow({**{key: region.get(key, region['region_id'] if key == 'display_id' else None)
                                for key in columns[:-1]},
                             'midpoint_numeric_preview': region['midpoint_example']['numeric_preview']})
    save_instruction_plot(report, target / 'route_navigation.png')
    save_plot(report, target / 'geometry_segments.png')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--options', type=Path, default=DEFAULT_OPTIONS)
    args = parser.parse_args()
    report = write_instruction_report(args.path, args.output, json.loads(args.options.read_text(encoding='utf-8')))
    print(json.dumps(dict(output=str(args.output.resolve()), regions=len(report['instruction_regions']),
                          instruction_length_m=report['instruction_length_m'], unused_templates=report['unused_templates'])))


if __name__ == '__main__':
    main()
