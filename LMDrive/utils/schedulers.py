"""Learning-rate schedules of the two LMDrive stages."""
import math


class CosineLRScheduler:
    """Stage 1: per-epoch cosine decay with linear warmup (timm CosineLRScheduler, one cycle).

    step(epoch) is called after each epoch with the next epoch index.  Epochs
    beyond t_initial (the cooldown) stay at lr_min.
    """

    def __init__(self, optimizer, t_initial, lr_min=0.0, warmup_t=0, warmup_lr_init=0.0):
        assert t_initial > 0 and lr_min >= 0
        self.optimizer = optimizer
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_values = [group["initial_lr"] for group in optimizer.param_groups]
        self._set_lrs(self.base_values)
        if warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / warmup_t for v in self.base_values]
            self._set_lrs([warmup_lr_init] * len(self.base_values))
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _set_lrs(self, values):
        for group, value in zip(self.optimizer.param_groups, values):
            group["lr"] = value

    def get_lr(self, t):
        if t < self.warmup_t:
            return [self.warmup_lr_init + t * step for step in self.warmup_steps]
        if t < self.t_initial:
            return [self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t / self.t_initial))
                    for lr_max in self.base_values]
        return [self.lr_min for _ in self.base_values]

    def step(self, epoch):
        self._set_lrs(self.get_lr(epoch))


class LinearWarmupCosineLRScheduler:
    """Stage 2 (LAVIS linear_warmup_cosine_lr).

    Epoch 0 warms up linearly over warmup_steps iterations; later epochs use one
    cosine value per epoch.  step() is called before every iteration.
    """

    def __init__(self, optimizer, max_epoch, min_lr, init_lr, warmup_steps=0, warmup_start_lr=-1):
        self.optimizer = optimizer
        self.max_epoch = max_epoch
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def get_lr(self, cur_epoch, cur_step):
        if cur_epoch == 0:
            return min(self.init_lr, self.warmup_start_lr
                       + (self.init_lr - self.warmup_start_lr) * cur_step / max(self.warmup_steps, 1))
        return (self.init_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * cur_epoch / self.max_epoch)) + self.min_lr

    def step(self, cur_epoch, cur_step):
        lr = self.get_lr(cur_epoch, cur_step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
