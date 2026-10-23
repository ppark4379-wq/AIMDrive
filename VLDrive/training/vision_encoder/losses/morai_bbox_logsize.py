"""Opt-in log-size loss for the two native MORAI LAV bbox size channels.

This diagnostic auxiliary keeps the eight-channel head, physical GT boxes,
native LAV objective and inference decoder unchanged. It is not enabled by any
production recipe. It measures full BEV length/width, not height or object IoU.
"""
import torch
import torch.nn.functional as F

from .morai_bbox_corner import _instance_frame_mean, _validate_inputs


MIN_PREDICTED_SIZE_M = 0.02
VALIDITY_DIMENSION_FLOOR_M = 0.2
SIZE_VALIDITY_WEIGHT = 2.0
SMOOTH_L1_BETA = 0.2


def logsize_bbox_loss(prediction, target, instance_ids, valid_mask=None):
    """Return ``(loss, terms)`` from raw ``(B,2500,8)`` native LAV tensors.

    Decode full length ``prediction[...,5]*7`` and width ``[...,6]*4``.
    The primary cell objective is the two-dimension mean Smooth L1 of
    ``log(max(predicted_size_m, .02) / GT_size_m)`` (beta=.2). The native GT
    parameter order is fixed: no swapping based on predicted heading. Center,
    heading, confidence and speed have no direct auxiliary gradient; their
    native losses remain necessary. Changes to shared model parameters can
    still affect other tasks, so output-channel independence is not a claim of
    independent training dynamics.

    For positive predictions above 2cm, reciprocal multiplicative errors have
    equal loss. The derivative includes ``1/predicted_size_m``: an undersized
    dimension receives a restoring gradient regardless of pose error, unlike
    a coupled corner objective. This differs from the earlier relative-error
    auxiliary, whose size term already was pose independent but used linear
    relative errors and was combined with center supervision.

    Add ``2*mean(relu(.02-predicted_size_m)/min(max(GT_size_m,.2)))`` to keep
    nonpositive/sub-floor predictions connected to a finite restoring
    gradient. This is a training regularizer, not an inference clamp or a GT
    edit. Sub-2cm GT can have nonzero loss even at its exact predicted size;
    the current selected MORAI objects exceed that floor. No GT dimension
    floor is applied inside the logarithmic ratio.

    ``instance_ids`` is int64 ``(B,2500)`` (-1 means inactive), and optional
    bool ``valid_mask`` masks frames ``(B,)`` or cells ``(B,2500)``. GT
    probability weights cells within each object; objects are equal within a
    frame; nonempty valid frames are equal. Only active probabilities and
    sizes must be finite/valid; pose and ignored cells are not consumed. Empty
    masks return differentiable finite zero. Half inputs compute in float32.
    """
    assigned = _validate_inputs(prediction, target, instance_ids, valid_mask)
    dtype = torch.promote_types(prediction.dtype, target.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    zero = prediction.reshape(-1)[:0].to(dtype=dtype).sum()
    if not assigned.any():
        return zero, {
            'bbox_logsize_ratio': zero,
            'bbox_logsize_validity': zero,
            'bbox_logsize_assigned_cells': zero.detach(),
            'bbox_logsize_invalid_size_cells': zero.detach(),
            'bbox_logsize_subfloor_size_cells': zero.detach(),
        }

    gt = target[assigned].to(dtype=dtype)
    pred = prediction[assigned].to(dtype=dtype)
    if not torch.isfinite(gt[:, [0, 5, 6]]).all():
        raise ValueError('assigned target probability and sizes must be finite')
    probability = gt[:, 0]
    if ((probability <= 0) | (probability > 1)).any():
        raise ValueError('assigned target probability must be in (0, 1]')
    if (gt[:, 5:7] <= 0).any():
        raise ValueError('assigned target length and width must be positive')
    if not torch.isfinite(pred[:, 5:7]).all():
        raise ValueError('assigned predicted sizes must be finite')

    scale = gt.new_tensor([7., 4.])
    gt_size = gt[:, 5:7] * scale
    pred_size = pred[:, 5:7] * scale
    if (not torch.isfinite(gt_size).all() or not torch.isfinite(pred_size).all()
            or (gt_size <= 0).any()):
        raise ValueError('decoded physical sizes must be finite and GT sizes positive')
    log_error = pred_size.clamp_min(MIN_PREDICTED_SIZE_M).log() - gt_size.log()
    ratio_cell = F.smooth_l1_loss(
        log_error, torch.zeros_like(log_error), beta=SMOOTH_L1_BETA,
        reduction='none').mean(-1)
    validity_denominator = gt_size.clamp_min(VALIDITY_DIMENSION_FLOOR_M).min(
        dim=-1, keepdim=True).values
    validity_cell = SIZE_VALIDITY_WEIGHT * (
        F.relu(MIN_PREDICTED_SIZE_M - pred_size) / validity_denominator).mean(-1)
    means = _instance_frame_mean(
        torch.stack((ratio_cell, validity_cell), dim=-1), probability,
        assigned, instance_ids)
    return means.sum(), {
        'bbox_logsize_ratio': means[0],
        'bbox_logsize_validity': means[1],
        'bbox_logsize_assigned_cells': assigned.sum().detach(),
        'bbox_logsize_invalid_size_cells': (pred_size <= 0).any(-1).sum().detach(),
        'bbox_logsize_subfloor_size_cells': (
            pred_size < MIN_PREDICTED_SIZE_M).any(-1).sum().detach(),
    }
