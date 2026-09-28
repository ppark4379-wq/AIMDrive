"""Summarize a schema-4 observation JSONL or display interval JSONL."""
import argparse
from collections import Counter
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log')
    args = parser.parse_args()
    values, reasons = {}, Counter()
    with open(args.log, encoding='utf-8') as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A live writer can have an unfinished final line.
            if row.get('event') == 'display_interval':
                values.setdefault('display_' + row['mode'] + '_ms', []).append(row['duration_ms'])
                continue
            reasons[row.get('association_status', 'unknown')] += 1
            recovery = row.get('gap_metrics', {}).get('recovery')
            if recovery:
                for key in ('success_interval_wall_ms', 'success_interval_sensor_ms',
                            'failed_attempts_between', 'skipped_lidar_frames_between'):
                    if recovery.get(key) is not None:
                        values.setdefault(key, []).append(recovery[key])
            for stage, value in row.get('performance', {}).get('stage_wall_ms', {}).items():
                values.setdefault('stage_' + stage + '_ms', []).append(value)
    summary = {key: {'count': len(v), 'median': float(np.median(v)),
                     'p95': float(np.percentile(v, 95)), 'max': max(v)} for key, v in values.items()}
    print(json.dumps({'attempt_status_counts': dict(reasons), 'metrics': summary}, indent=2))


if __name__ == '__main__':
    main()
