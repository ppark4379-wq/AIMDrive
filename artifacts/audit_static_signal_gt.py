"""Read-only static-zone instruction/signal audit; writes only audit artifacts."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path('C:/dataset_label/static_obstacle')
OUT = Path(__file__).parent / 'static_signal_gt_20261008'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    OUT.mkdir(exist_ok=True)
    laps, gaps, total = [], [], Counter()
    for episode in sorted(ROOT.glob('*/episode_*')):
        if not episode.is_dir():
            continue
        rows = [json.loads(s) for s in (episode / 'situation_labels/labeled_frames.jsonl').read_text(encoding='utf-8-sig').splitlines()]
        signals = {r['index']: r for r in read(episode / 'route_signal_audit_v1/frames.json')}
        transitions = read(episode / 'traffic_audit_v1/transitions.json')
        zoned = [r for r in rows if r.get('zone') == 'static_obstacle']
        t0 = rows[0]['sensor_time_ns']
        reasons = Counter(r['labeling'].get('instruction_consistency', 'none') for r in zoned if not r.get('instruction'))
        total.update(reasons)
        colors = Counter(str(signals.get(r['frame_id'], {}).get('color')) for r in zoned)
        by_id = {r['frame_id']: r for r in rows}
        run = []

        def finish():
            if not run:
                return
            last = run[-1]
            after = by_id.get(last['frame_id'] + 1)
            signal = signals.get(last['frame_id'], {})
            green = next((tr for tr in transitions if tr['id'] == signal.get('light_id')
                          and tr['to_color'] in (16, 48)
                          and tr['uncertainty_source_ns'][1] >= last['sensor_time_ns']), None)
            gaps.append(dict(lap=episode.parent.name, start_frame=run[0]['frame_id'],
                             end_frame=last['frame_id'], frames=len(run),
                             start_s=round((run[0]['sensor_time_ns']-t0)/1e9, 3),
                             end_s=round((last['sensor_time_ns']-t0)/1e9, 3),
                             reasons=dict(Counter(r['labeling'].get('instruction_consistency', 'none') for r in run)),
                             signal_colors=dict(Counter(str(signals.get(r['frame_id'], {}).get('color')) for r in run)),
                             all_frames_valid=all(r.get('frame_valid') for r in run),
                             resume_frame=after['frame_id'] if after and after.get('instruction') else None,
                             resume_signal=signals.get(after['frame_id'], {}).get('color') if after else None,
                             next_green_anchor=green['sensor_anchor_indices'] if green else None,
                             resume_before_green_s=round((green['uncertainty_source_ns'][0]-after['sensor_time_ns'])/1e9, 3)
                             if green and after and after.get('instruction') else None))
            run.clear()

        for row in rows:
            if row.get('zone') == 'static_obstacle' and not row.get('instruction'):
                run.append(row)
            else:
                finish()
        finish()
        laps.append(dict(lap=episode.parent.name, zone_frames=len(zoned), missing_frames=sum(reasons.values()),
                         reasons=dict(reasons), bound_colors=dict(colors)))
    summary = dict(laps=len(laps), zone_frames=sum(r['zone_frames'] for r in laps),
                   missing_frames=sum(r['missing_frames'] for r in laps),
                   laps_with_missing=sum(bool(r['missing_frames']) for r in laps),
                   reasons=dict(total), gaps=len(gaps),
                   missing_signal_colors=dict(sum((Counter(g['signal_colors']) for g in gaps), Counter())),
                   gaps_all_frames_valid=all(g['all_frames_valid'] for g in gaps),
                   gaps_resuming_on_red=sum(g['resume_signal']==1 for g in gaps),
                   no_gap_laps=[r for r in laps if not r['missing_frames']])
    (OUT/'report.json').write_text(json.dumps(dict(summary=summary, laps=laps, gaps=gaps), indent=2), encoding='utf-8')
    for name, records in [('laps', laps), ('gaps', gaps)]:
        with (OUT/(name+'.csv')).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v, (dict,list)) else v for k,v in r.items()} for r in records)
    print(json.dumps(summary, indent=2))
    print('FIRST GAPS', json.dumps(gaps[:3], indent=2))


if __name__ == '__main__':
    main()
