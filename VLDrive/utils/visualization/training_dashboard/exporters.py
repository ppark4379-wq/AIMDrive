"""Raw metric CSVs derived from JSONL. No model or UI imports required."""
import argparse
import csv
import io
import os
from pathlib import Path
import tempfile
from utils.visualization.training_dashboard.log_reader import JsonlTail, RunMetadata
from utils.visualization.training_dashboard.metrics import MetricSeries

METRIC_COLUMNS = ('update epoch timestamp elapsed_s progress_pct eta_s expected_finish update_time_s '
                  'updates_per_sec learning_rate total_loss l_way l_prun l_rec gumbel_keep_ratio p_keep_mean '
                  'p_keep_std selected_tokens_mean selected_tokens_min selected_tokens_max selected_tokens_target '
                  'grad_norm gpu_memory_gb gpu_utilization').split()
METRIC_COLUMNS += ('train_ade_m train_fde_m trajectory_samples train_ade_recent_m train_fde_recent_m '
                   'navigation_tokens situation_tokens zone_tokens visual_tokens llm_valid_tokens '
                   'llm_max_valid_tokens llm_padded_tokens llm_padding_ratio llm_context_limit').split()
METRIC_COLUMNS += [f'zone_{i}_{key}' for i in range(6) for key in ('samples', 'ade_m', 'fde_m')]
GRADIENT_COLUMNS = 'update selector mefa qformer connector_projection tinyllama waypoint_head navigation_mlp zone_embedding'.split()


def csv_text(rows, *, gradients=False):
    stream = io.StringIO(newline='')
    columns = GRADIENT_COLUMNS if gradients else METRIC_COLUMNS
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        if gradients:
            if not row.get('module_gradients'): continue
            writer.writerow(dict(update=row['update'], **row['module_gradients']))
        else:
            flat = {key: row.get(key) for key in columns}
            for zone, stat in (row.get('zone_metrics') or {}).items():
                flat.update({f'zone_{zone}_samples': stat['samples'],
                             f'zone_{zone}_ade_m': stat['ade_sum_m']/stat['samples'],
                             f'zone_{zone}_fde_m': stat['fde_sum_m']/stat['samples']})
            writer.writerow(flat)
    return stream.getvalue()


def export_csv(run_dir, rows):
    root = Path(run_dir).resolve()
    if not root.is_dir(): raise ValueError('Run directory does not exist')
    folder = root / 'exports'
    # A symlink must not redirect writes into another run or the source log.
    if folder.resolve().parent != root: raise ValueError('Unsafe exports directory')
    folder.mkdir(exist_ok=True)
    paths = []
    for name, gradients in [('training_metrics.csv',False),('module_gradients.csv',True)]:
        target = folder / name
        if target.is_symlink(): raise ValueError('Export destination cannot be a symlink')
        descriptor, temporary_name = tempfile.mkstemp(dir=folder, suffix='.tmp')
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8-sig', newline='') as stream:
                stream.write(csv_text(rows, gradients=gradients))
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        paths.append(target)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    reader = JsonlTail(args.run_dir/'training.jsonl')
    raw = reader.read_all()
    if reader.error: parser.error(reader.error)
    rows = MetricSeries(RunMetadata(args.run_dir).context()).extend(raw)
    for path in export_csv(args.run_dir, rows): print(path)
    if reader.invalid_lines: print(f'Skipped {reader.invalid_lines} malformed complete lines')


if __name__ == '__main__': main()
