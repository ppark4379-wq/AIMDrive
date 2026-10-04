"""Explicit spawn workers for the sealed MORAI sensor-record dataset.

Workers stay CPU-only, preserve DataLoader sampler order, and cap native thread
pools. Callers own sampler/checkpoint state and the separate worker-seed RNG.
"""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def morai_worker_init(worker_id):
    """Top-level callable so workers can start safely after CUDA initialization."""
    del worker_id
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Some builds initialize the interop pool while unpickling the dataset.
        pass
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_morai_loader(dataset, *, batch_size=1, workers=0, shuffle=False,
                      sampler=None, generator=None, pin_memory=False,
                      prefetch_factor=2, persistent_workers=True):
    """Return a loader with explicit spawn semantics, never implicit fork.

    ``generator`` must be caller-owned when workers run; keep it separate from
    the training sampler RNG so worker startup cannot perturb sample order.
    Pinning is an explicit caller choice; this helper never probes CUDA.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError('workers must be a nonnegative integer')
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if isinstance(prefetch_factor, bool) or not isinstance(prefetch_factor, int) or prefetch_factor < 1:
        raise ValueError('prefetch_factor must be a positive integer')
    if sampler is not None and shuffle:
        raise ValueError('An explicit sampler cannot be combined with shuffle')
    if workers and generator is None:
        raise ValueError('Spawn workers require a separate caller-owned generator')
    kwargs = dict(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                  sampler=sampler, num_workers=workers, generator=generator,
                  pin_memory=bool(pin_memory), drop_last=False)
    if workers:
        kwargs.update(multiprocessing_context='spawn',
                      persistent_workers=bool(persistent_workers),
                      prefetch_factor=prefetch_factor,
                      worker_init_fn=morai_worker_init)
    return DataLoader(**kwargs)
