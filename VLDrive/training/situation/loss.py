"""Supervise Fs without feeding GT back into the model."""
import torch
from torch.nn import functional as F

from .labels import IGNORE


def situation_loss(prediction, targets):
    """Targets: last-frame situation labels and action_valid.

    action_valid=False masks instruction and direction losses.
    Unknown observation fields must use label -100. An all-masked batch yields
    differentiable zero; callers should skip its optimizer step.
    """
    batch = prediction['situation_embedding'].shape[0]
    action_valid = targets['action_valid']
    if action_valid.shape != (batch,) or action_valid.dtype != torch.bool:
        raise ValueError('action_valid must be boolean [B]')
    loss = prediction['situation_embedding'].sum() * 0
    metrics = {}
    for name, labels in targets['labels'].items():
        if labels.dtype != torch.long or labels.shape != (batch,):
            raise ValueError(f'{name} labels must be int64 [B]')
        valid = labels != IGNORE
        if name in ('instruction', 'direction'):
            valid = valid & action_valid
        count = int(valid.sum())
        if not count:
            continue
        if name not in prediction['logits']:
            raise ValueError(f'Missing classifier for supervised field {name}')
        logits = prediction['logits'][name][valid]
        values = labels[valid]
        if bool(((values < 0) | (values >= logits.shape[-1])).any()):
            raise ValueError(f'Out-of-vocabulary {name} label')
        summed = F.cross_entropy(logits, values, reduction='sum')
        loss = loss + summed / count
        metrics[name] = {'sum': float(summed.detach()), 'count': count,
                         'correct': int((logits.argmax(-1) == values).sum())}
    return loss, metrics
