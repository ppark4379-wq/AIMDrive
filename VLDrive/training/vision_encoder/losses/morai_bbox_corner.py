"""Experimental physical-corner auxiliary for raw MORAI LAV bbox outputs.

This module is opt-in only. It does not change the native target representation,
LAV losses, heading semantics, decoder, or checkpoint format. The caller retains
the native heading loss and chooses an external auxiliary weight.
"""
import torch
import torch.nn.functional as F


SIZE_FLOOR_M = 0.2
MIN_PREDICTED_SIZE_M = 0.02
SIZE_VALIDITY_WEIGHT = 2.0
HEADING_NORM_FLOOR = 1e-3
SMOOTH_L1_BETA = 0.2


def _validate_inputs(prediction, target, instance_ids, valid_mask):
    if prediction.ndim != 3 or tuple(prediction.shape[1:]) != (2500, 8):
        raise ValueError('prediction must have shape (B, 2500, 8)')
    if target.shape != prediction.shape:
        raise ValueError('target must have the same shape as prediction')
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError('prediction and target must be floating point tensors')
    if instance_ids.shape != prediction.shape[:2]:
        raise ValueError('instance_ids must have shape (B, 2500)')
    if instance_ids.dtype != torch.long:
        raise TypeError('instance_ids must have dtype torch.int64')
    if target.device != prediction.device or instance_ids.device != prediction.device:
        raise ValueError('prediction, target and instance_ids must share a device')
    if (instance_ids < -1).any():
        raise ValueError('instance_ids must be -1 or nonnegative')
    assigned = instance_ids >= 0
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool:
            raise TypeError('valid_mask must have dtype torch.bool')
        if valid_mask.device != prediction.device:
            raise ValueError('valid_mask must share the prediction device')
        if valid_mask.shape == prediction.shape[:1]:
            assigned = assigned & valid_mask[:, None]
        elif valid_mask.shape == prediction.shape[:2]:
            assigned = assigned & valid_mask
        else:
            raise ValueError('valid_mask must have shape (B,) or (B, 2500)')
    return assigned


def _corners(heading, sizes):
    transverse = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)
    along = heading * sizes[:, 0:1] * 0.5
    across = transverse * sizes[:, 1:2] * 0.5
    return torch.stack((along + across, along - across,
                        -along - across, -along + across), dim=1)


def _instance_frame_mean(cell_terms, probability, assigned, instance_ids):
    """GT-probability means per instance, equal instances then positive frames."""
    frame_ids = assigned.nonzero(as_tuple=False)[:, 0]
    pairs = torch.stack((frame_ids, instance_ids[assigned]), dim=-1)
    unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    count, width = len(unique_pairs), cell_terms.shape[1]
    instance_weight = probability.new_zeros(count).scatter_add(0, inverse, probability)
    instance_terms = cell_terms.new_zeros((count, width)).scatter_add(
        0, inverse[:, None].expand(-1, width), cell_terms * probability[:, None])
    instance_terms = instance_terms / instance_weight[:, None]
    frame_index = unique_pairs[:, 0]
    frame_count = probability.new_zeros(assigned.shape[0]).scatter_add(
        0, frame_index, torch.ones_like(instance_weight))
    frame_terms = cell_terms.new_zeros((assigned.shape[0], width)).scatter_add(
        0, frame_index[:, None].expand(-1, width), instance_terms)
    positive = frame_count > 0
    return (frame_terms[positive] / frame_count[positive, None]).mean(0)


def corner_bbox_loss(prediction, target, instance_ids, valid_mask=None):
    """Return ``(loss, terms)`` for raw ``(B, 2500, 8)`` LAV tensors.

    ``instance_ids`` is int64 ``(B,2500)`` with -1 for inactive cells; object IDs
    are local to each frame. ``valid_mask`` optionally masks frames ``(B,)`` or
    cells ``(B,2500)``. Active targets require probability in (0,1], finite
    geometry, positive full sizes and nonzero heading. Invalid active geometry
    raises; ignored cells may contain NaNs without affecting loss or gradients.

    Decode offsets by 3.5, full length by 7 and full width by 4. Convert heading
    channels [right,-forward] to [right,forward]. The shared cell-grid origin
    cancels from the corner residual. Normalize predicted heading by its norm
    clamped at 1e-3, giving finite gradients even at zero heading. This auxiliary
    does not replace the native heading loss needed to anchor nonzero direction.

    Compare the four physical corners under all four cyclic correspondences.
    Project residuals into GT longitudinal/lateral axes,
    divide by GT full length/width floored at 0.2m, average Smooth L1 (beta=0.2)
    over four corners and two coordinates, then take the minimum per cell.
    Equivalent 180-degree direction or length/width-swap plus 90-degree boxes
    therefore have zero corner error. This is a corner objective, NOT IoU loss.

    Signed predicted sizes are retained in corners, with the additional
    2 * mean(ReLU(0.02m - predicted size) / min(floored GT length, GT width))
    penalty. For each invalid size its restoring derivative has magnitude
    1/min_dimension, exceeding the corner derivative bound 0.5/min_dimension.
    Thus even the combined auxiliary pushes invalid sizes positive for rotated,
    thin boxes whose signed vertex sets can look equivalent. The native size
    loss remains in the caller's objective as well. The 2cm
    minimum is an explicit regularizer, so sub-2cm GT would not have zero total
    auxiliary even at exact prediction. No production GT dimensions are changed.

    Cell losses use GT-probability means within each instance, equal instances
    within each frame, then equal nonempty valid frames. Loss is corner mean
    PLUS size-validity mean, without an implicit external weight. Terms expose
    those two components and detached counts of selected cells, nonpositive
    predicted-size cells and sub-floor predicted-heading cells. Confidence and
    speed predictions receive no auxiliary gradient. Empty selections return
    differentiable finite zero. Computation uses at least float32.
    """
    assigned = _validate_inputs(prediction, target, instance_ids, valid_mask)
    dtype = torch.promote_types(prediction.dtype, target.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    zero = prediction.reshape(-1)[:0].to(dtype=dtype).sum()
    if not assigned.any():
        return zero, {'bbox_corner_geometry': zero, 'bbox_corner_size_validity': zero,
                      'bbox_corner_assigned_cells': zero.detach(),
                      'bbox_corner_invalid_size_cells': zero.detach(),
                      'bbox_corner_small_heading_cells': zero.detach()}
    gt = target[assigned].to(dtype=dtype)
    pred = prediction[assigned].to(dtype=dtype)
    if not torch.isfinite(gt[:, :7]).all():
        raise ValueError('assigned target probability and geometry must be finite')
    if ((gt[:, 0] <= 0) | (gt[:, 0] > 1)).any():
        raise ValueError('assigned target probability must be in (0, 1]')
    if (gt[:, 5:7] <= 0).any():
        raise ValueError('assigned target length and width must be positive')
    if not torch.isfinite(pred[:, 1:7]).all():
        raise ValueError('assigned predicted geometry must be finite')
    gt_heading = torch.stack((gt[:, 3], -gt[:, 4]), dim=-1)
    gt_norm = torch.linalg.vector_norm(gt_heading, dim=-1, keepdim=True)
    if (gt_norm <= 0).any() or not torch.isfinite(gt_norm).all():
        raise ValueError('assigned target heading must be finite and nonzero')
    gt_heading = gt_heading / gt_norm
    gt_transverse = torch.stack((-gt_heading[:, 1], gt_heading[:, 0]), dim=-1)
    pred_heading = torch.stack((pred[:, 3], -pred[:, 4]), dim=-1)
    pred_norm = torch.linalg.vector_norm(pred_heading, dim=-1, keepdim=True)
    if not torch.isfinite(pred_norm).all():
        raise ValueError('assigned predicted heading norm must be finite')
    pred_heading = pred_heading / pred_norm.clamp_min(HEADING_NORM_FLOOR)
    size_scale = gt.new_tensor([7., 4.])
    gt_size, pred_size = gt[:, 5:7] * size_scale, pred[:, 5:7] * size_scale
    center_delta = (pred[:, 1:3] - gt[:, 1:3]) * 3.5
    if not (torch.isfinite(gt_size).all() and torch.isfinite(pred_size).all()
            and torch.isfinite(center_delta).all()):
        raise ValueError('decoded physical geometry must be finite')
    denominator = gt_size.clamp_min(SIZE_FLOOR_M)
    gt_corners = _corners(gt_heading, gt_size)
    pred_corners = _corners(pred_heading, pred_size) + center_delta[:, None, :]
    alternatives = []
    for shift in range(4):
        residual = pred_corners.roll(shift, dims=1) - gt_corners
        axes_error = torch.stack(((residual * gt_heading[:, None, :]).sum(-1),
                                  (residual * gt_transverse[:, None, :]).sum(-1)), dim=-1)
        error = axes_error / denominator[:, None, :]
        alternatives.append(F.smooth_l1_loss(
            error, torch.zeros_like(error), beta=SMOOTH_L1_BETA, reduction='none').mean((1, 2)))
    corner_cell = torch.stack(alternatives, dim=-1).min(dim=-1).values
    # A dimension-specific denominator can underweight a long, negative size
    # when a 90-degree heading maps it to the GT's thin axis. Use the minimum
    # GT dimension and a factor of two to dominate the maximum corner slope.
    minimum_dimension = denominator.min(dim=-1, keepdim=True).values
    validity_cell = SIZE_VALIDITY_WEIGHT * (
        F.relu(MIN_PREDICTED_SIZE_M - pred_size) / minimum_dimension).mean(-1)
    means = _instance_frame_mean(torch.stack((corner_cell, validity_cell), dim=-1),
                                 gt[:, 0], assigned, instance_ids)
    return means.sum(), {
        'bbox_corner_geometry': means[0], 'bbox_corner_size_validity': means[1],
        'bbox_corner_assigned_cells': assigned.sum().detach(),
        'bbox_corner_invalid_size_cells': (pred_size <= 0).any(-1).sum().detach(),
        'bbox_corner_small_heading_cells': (pred_norm[:, 0] < HEADING_NORM_FLOOR).sum().detach(),
    }
