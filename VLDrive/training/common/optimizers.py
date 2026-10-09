"""Training optimizers; existing implementations relocated without changing calculations."""
import math
import torch


def _optimizer_parameter_groups(model, lr, lr_multipliers):
    multipliers = dict(lr_multipliers or {})
    for prefix, value in multipliers.items():
        if not isinstance(prefix, str) or not prefix or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError('LR multipliers need nonempty prefixes and finite positive values')
    selected = {'default': []}
    selected.update({prefix: [] for prefix in multipliers})
    if 'default' in multipliers:
        raise ValueError('The LR prefix "default" is reserved')
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        matches = [prefix for prefix in multipliers if name.startswith(prefix)]
        if len(matches) > 1:
            raise ValueError('Overlapping LR prefixes for ' + name)
        selected[matches[0] if matches else 'default'].append(parameter)
    if any(not selected[prefix] for prefix in multipliers):
        raise ValueError('An LR multiplier prefix matched no trainable parameters')
    return [dict(params=parameters, lr=float(lr)*float(multipliers.get(name, 1.)),
                 base_lr_mult=float(multipliers.get(name, 1.)), group_name=name)
            for name, parameters in selected.items() if parameters]


def cosine_learning_rate(*, lr, step, schedule_updates, warmup_updates=0,
                         warmup_lr=0., min_lr=0.):
    """Global update-based linear warmup followed by cosine decay."""
    if (any(type(value) is not int for value in (step, schedule_updates, warmup_updates))
            or not 0 <= step <= schedule_updates or not 0 <= warmup_updates < schedule_updates):
        raise ValueError('Invalid cosine schedule position')
    if (any(not math.isfinite(value) for value in (lr, warmup_lr, min_lr))
            or lr <= 0 or not 0 <= warmup_lr <= lr or not 0 <= min_lr <= lr):
        raise ValueError('Invalid learning-rate bounds')
    if step < warmup_updates:
        return warmup_lr + (lr - warmup_lr) * step / warmup_updates
    progress = (step - warmup_updates) / (schedule_updates - warmup_updates)
    return min_lr + (lr - min_lr) * .5 * (1. + math.cos(math.pi * progress))


def set_cosine_learning_rate(optimizer, *, lr, step, schedule_updates, warmup_updates=0,
                             warmup_lr=0., min_lr=0.):
    """Keep parameter-group multipliers and legacy no-warmup defaults intact."""
    current_lr = cosine_learning_rate(lr=lr, step=step, schedule_updates=schedule_updates,
        warmup_updates=warmup_updates, warmup_lr=warmup_lr, min_lr=min_lr)
    for group in optimizer.param_groups:
        group['lr'] = current_lr * float(group.get('base_lr_mult', 1.))
    return current_lr


class CPUAdamW:
    """FP32 AdamW masters/state for a single-device FP32 or BF16 model."""
    def __init__(self, parameters, *, lr=1e-5, weight_decay=.06, fused=False):
        if type(fused) is not bool:
            raise ValueError('CPU AdamW fused must be a boolean')
        parameters = list(parameters)
        grouped = bool(parameters) and isinstance(parameters[0], dict)
        if any(isinstance(p, dict) != grouped for p in parameters):
            raise ValueError('Optimizer parameters cannot mix tensors and groups')
        groups = parameters if grouped else [dict(params=parameters)]
        groups = [dict(group, params=[p for p in group['params'] if p.requires_grad]) for group in groups]
        self.parameters = [p for group in groups for p in group['params']]
        if len({id(p) for p in self.parameters}) != len(self.parameters):
            raise ValueError('Duplicate optimizer parameter')
        if any(p.dtype not in (torch.float32, torch.bfloat16) for p in self.parameters):
            raise ValueError('CPU AdamW supports FP32 or BF16 model parameters')
        self.masters = [torch.nn.Parameter(p.detach().float().cpu().clone()) for p in self.parameters]
        master_groups, offset = [], 0
        for group in groups:
            count = len(group['params'])
            if count:
                master_groups.append(dict(group, params=self.masters[offset:offset+count]))
            offset += count
        # CPU fused kernels avoid separate passes/intermediates for each AdamW
        # operation. FP32 masters, moments and CPU/GPU transfers stay unchanged.
        self.optimizer = torch.optim.AdamW(master_groups, lr=lr, weight_decay=weight_decay,
                                           foreach=False, fused=fused)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters:
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        for parameter, master in zip(self.parameters, self.masters):
            master.grad = None if parameter.grad is None else parameter.grad.detach().to(device='cpu', dtype=torch.float32, copy=True)
        self.optimizer.step()
        for parameter, master in zip(self.parameters, self.masters):
            if master.grad is not None:
                parameter.copy_(master)
            master.grad = None

    def state_dict(self):
        return dict(implementation='cpu-fp32-adamw-v1', optimizer=self.optimizer.state_dict(),
                    masters=[p.detach() for p in self.masters])

    def load_state_dict(self, state):
        if state.get('implementation') != 'cpu-fp32-adamw-v1':
            raise ValueError('Optimizer implementation mismatch')
        if len(state['masters']) != len(self.masters):
            raise ValueError('Master parameter count mismatch')
        with torch.no_grad():
            for parameter, master, saved in zip(self.parameters, self.masters, state['masters']):
                if saved.shape != master.shape or saved.dtype != torch.float32:
                    raise ValueError('Master parameter contract mismatch')
                if not torch.equal(parameter.detach().cpu(), saved.to(parameter.dtype)):
                    raise ValueError('Restored model disagrees with FP32 optimizer masters')
                master.copy_(saved)
        self.optimizer.load_state_dict(state['optimizer'])


def make_optimizer(model, *, lr=1e-5, weight_decay=.06, state_device='cpu', lr_multipliers=None,
                   cpu_fused=False):
    if type(cpu_fused) is not bool:
        raise ValueError('optimizer.cpu_fused must be a boolean')
    groups = _optimizer_parameter_groups(model, lr, lr_multipliers)
    if state_device == 'cpu':
        return CPUAdamW(groups, lr=lr, weight_decay=weight_decay, fused=cpu_fused)
    if state_device != 'model':
        raise ValueError('optimizer.state_device must be cpu or model')
    if cpu_fused:
        raise ValueError('optimizer.cpu_fused requires state_device=cpu')
    if any(p.dtype != torch.float32 for p in model.parameters() if p.requires_grad):
        raise ValueError('BF16 training requires CPU FP32 master optimizer; use precision=fp32 for model-state AdamW')
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay, foreach=False)
