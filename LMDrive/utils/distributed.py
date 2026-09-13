"""Distributed (torchrun) helpers.  Single-process runs need no special handling."""
import builtins
import datetime
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


@dataclass
class DistributedContext:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self):
        return self.rank == 0


def _silence_print_on_workers(is_master):
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print


def init_distributed(backend="nccl", timeout_minutes=60):
    """Use the torchrun environment when WORLD_SIZE > 1, otherwise a single device."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=timeout_minutes),
        )
        dist.barrier()
        _silence_print_on_workers(rank == 0)
        return DistributedContext(True, rank, world_size, local_rank, torch.device("cuda", local_rank))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistributedContext(False, 0, 1, 0, device)


def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_initialized() else 1


def is_main_process():
    return get_rank() == 0


def barrier():
    if is_dist_initialized():
        dist.barrier()


def reduce_mean(tensor, world_size=None):
    """Average a tensor over all ranks (returns the tensor unchanged without distributed)."""
    if not is_dist_initialized():
        return tensor
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= world_size or get_world_size()
    return reduced


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def cleanup_distributed():
    if is_dist_initialized():
        dist.destroy_process_group()


class OrderedDistributedSampler(Sampler):
    """Deterministic, non-shuffled shard per rank; pads with repeats so all ranks get the same count."""

    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset
        self.num_replicas = get_world_size() if num_replicas is None else num_replicas
        self.rank = get_rank() if rank is None else rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices += indices[: (self.total_size - len(indices))]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples
