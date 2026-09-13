"""Logging, iteration meters and TensorBoard writers."""
import csv
import datetime
import json
import logging
import os
import sys
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


def setup_logging(is_main_process=True):
    """Console logging; worker ranks only show warnings."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.INFO if is_main_process else logging.WARN)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)


def attach_log_file(path):
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)


class SmoothedValue:
    """Tracks a series of values: windowed median/average and the global average."""

    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """Sums count and total over ranks (the window is not synchronised)."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        values = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(values)
        self.count = int(values[0].item())
        self.total = values[1].item()

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg,
                               max=self.max, value=self.value)


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            assert isinstance(value, (float, int))
            self.meters[key].update(value)

    def __str__(self):
        return self.delimiter.join("{}: {}".format(name, str(meter)) for name, meter in self.meters.items())

    def global_avg(self):
        return self.delimiter.join("{}: {:.4f}".format(name, meter.global_avg) for name, meter in self.meters.items())

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=""):
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        total = len(iterable)
        space_fmt = ":" + str(len(str(total))) + "d"
        parts = [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
        if torch.cuda.is_available():
            parts.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(parts)
        for i, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == total - 1:
                eta = str(datetime.timedelta(seconds=int(iter_time.global_avg * (total - i))))
                memory = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else 0
                logger.info(log_msg.format(i, total, eta=eta, meters=str(self), time=str(iter_time),
                                           data=str(data_time), memory=memory))
            end = time.time()
        total_time = time.time() - start_time
        logger.info("%s Total time: %s (%.4f s / it)", header,
                    str(datetime.timedelta(seconds=int(total_time))), total_time / max(total, 1))


class NullWriter:
    """Stands in for a SummaryWriter when TensorBoard is not installed."""

    def add_scalar(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def create_tensorboard_writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            logger.warning("TensorBoard is not installed (pip install tensorboard); "
                           "metrics are still written to the CSV/JSON logs")
            return NullWriter()
    return SummaryWriter(log_dir)


def append_csv_row(path, row, write_header):
    with open(path, mode="a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_json_line(path, record):
    with open(path, "a") as stream:
        stream.write(json.dumps(record) + "\n")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
