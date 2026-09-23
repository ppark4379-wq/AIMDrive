"""Shared raw-column projection and robust ETA; no fabricated missing metrics."""
from collections import deque
from datetime import datetime, timedelta, timezone
import math
import statistics
from .log_reader import number

ALIASES = {'total_loss': 'loss', 'l_way': 'waypoint_loss', 'l_prun': 'pruning_loss',
           'l_rec': 'reconstruction_loss', 'gumbel_keep_ratio': 'sampled_retention',
           'p_keep_mean': 'soft_retention', 'learning_rate': 'lr'}
DIRECT = ('update', 'epoch', 'elapsed_s', 'update_time_s', 'p_keep_std', 'selected_tokens_mean',
          'selected_tokens_min', 'selected_tokens_max', 'selected_tokens_target', 'grad_norm',
          'gpu_memory_gb', 'gpu_utilization')


def parse_time(value):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return result if result.tzinfo else None
    except (ValueError, TypeError, AttributeError):
        return None


class MetricSeries:
    def __init__(self, context=None, *, minimum_samples=20, rolling=200):
        self.context = context or {}
        self.minimum_samples = minimum_samples
        self.durations = deque(maxlen=rolling)
        self.rows = []
        self.previous = None

    def append(self, raw):
        row = {key: number(raw.get(key)) for key in DIRECT}
        row.update({key: number(raw.get(key, raw.get(alias))) for key, alias in ALIASES.items()})
        row['timestamp'] = raw.get('timestamp') if parse_time(raw.get('timestamp')) else None
        total = number(raw.get('total_updates')) or self.context.get('total_updates')
        epochs = number(raw.get('total_epochs')) or self.context.get('total_epochs')
        per_epoch = self.context.get('updates_per_epoch')
        if row['epoch'] is None and per_epoch and row['update'] is not None:
            row['epoch'] = math.ceil(row['update'] / per_epoch)
        row.update(total_updates=total, total_epochs=epochs,
                   progress_pct=100 * row['update'] / total if total and row['update'] is not None else None)
        target = number(raw.get('target_retention'))
        if target is None: target = self.context.get('target_retention')
        tokens = number(raw.get('num_visual_tokens')) or self.context.get('num_visual_tokens')
        row['target_retention'] = target
        if row['selected_tokens_target'] is None and target is not None and tokens:
            row['selected_tokens_target'] = tokens * target
        previous = self.previous
        if previous and (row['update'] <= previous['update'] or
                (row['elapsed_s'] is not None and previous['elapsed_s'] is not None and row['elapsed_s'] < previous['elapsed_s'])):
            self.durations.clear(); previous = None
        # Cadence includes checkpoint/epoch work between updates. Median limits
        # outlier influence; update_time_s is just the individual update duration.
        duration = None
        if previous and row['elapsed_s'] is not None and previous['elapsed_s'] is not None:
            duration = (row['elapsed_s'] - previous['elapsed_s']) / (row['update'] - previous['update'])
        if duration is None: duration = row['update_time_s']
        if duration is not None and duration > 0:
            self.durations.append(duration)
        if row['update_time_s'] is None and duration is not None and duration > 0:
            row['update_time_s'] = duration  # Documented cadence fallback for old logs.
        ready = len(self.durations) >= self.minimum_samples
        median = statistics.median(self.durations) if ready else None
        remaining = max(0, total-row['update']) if total else None
        eta = remaining * median if remaining is not None and median else None
        if remaining == 0: eta = 0.
        row.update(updates_per_sec=1/median if median else None, eta_s=eta, expected_finish=None)
        stamp = parse_time(row['timestamp'])
        if stamp and eta is not None:
            row['expected_finish'] = (stamp + timedelta(seconds=eta)).isoformat()
        gradients = raw.get('module_gradients')
        row['module_gradients'] = {k: number(v) for k, v in gradients.items()} if isinstance(gradients, dict) else None
        row['system'] = raw.get('system')
        self.previous = row; self.rows.append(row)
        return row

    def extend(self, rows):
        for row in rows: self.append(row)
        return self.rows


def ema(values, span=100):
    current = None; result = []; alpha = 2 / (span+1)
    for value in values:
        value = number(value)
        if value is not None:
            current = value if current is None else alpha*value + (1-alpha)*current
        result.append(current if value is not None else None)
    return result


def duration_text(value):
    if value is None: return 'estimating…'
    minutes, seconds = divmod(int(max(0,value)), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes:02d}m' if hours else f'{minutes}m {seconds:02d}s'
