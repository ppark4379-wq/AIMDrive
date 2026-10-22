"""Independent masked deny/allow supervision for a planned route movement."""
import torch
from torch.nn import functional as F

from models.ve.route_permission import movement_one_hot


def route_permission_loss(logits, targets, valid_mask, movement):
    """Return ``(CE, terms)`` using only rows with verified route permission.

    This function has no red/not-red label argument. A red-plus-left-arrow
    frame can therefore supervise left=allow even when its native color label
    is unknown (-1). Unknown labels/movements and NaNs in *masked* rows are
    ignored; unknown movement or malformed labels in active rows are errors.
    """
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 2
            or not logits.is_floating_point()):
        raise ValueError('Permission logits must be floating-point (B, 2)')
    batch_size = len(logits)
    for name, value in (('targets', targets), ('valid_mask', valid_mask), ('movement', movement)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(name + ' must be a tensor')
        if value.device != logits.device:
            raise ValueError(name + ' must share the logits device')
    if targets.shape != (batch_size,) or valid_mask.shape != (batch_size,):
        raise ValueError('Permission targets and validity mask must have shape (B,)')
    if valid_mask.dtype != torch.bool:
        raise TypeError('Permission validity mask must be boolean')
    if movement.shape not in ((batch_size,), (batch_size, 3)):
        raise ValueError('Movement must have shape (B,) or (B, 3)')
    count = valid_mask.sum()
    if not bool(valid_mask.any()):
        # Summing an empty view connects autograd without reading masked NaNs.
        zero = logits.reshape(-1)[:0].sum()
        return zero, {'route_permission_ce': zero, 'route_permission_valid_count': count}
    selected_targets = targets[valid_mask]
    if (not bool(torch.isfinite(selected_targets).all())
            or not bool(((selected_targets == 0) | (selected_targets == 1)).all())):
        raise ValueError('Valid permission labels must be deny=0 or allow=1')
    movement_one_hot(movement[valid_mask], int(count), device=logits.device, dtype=logits.dtype)
    selected_logits = logits[valid_mask]
    if not bool(torch.isfinite(selected_logits).all()):
        raise ValueError('Valid permission predictions must be finite')
    loss = F.cross_entropy(selected_logits, selected_targets.long())
    return loss, {'route_permission_ce': loss, 'route_permission_valid_count': count}
