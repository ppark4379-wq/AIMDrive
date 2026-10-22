"""Opt-in best-owned-query aggregation of the existing MORAI corner loss.

Only query aggregation changes. Native LAV losses, physical GT, per-cell corner
and signed-size validity costs, and the caller's external weight are preserved.
GT ownership is consumed only during training, never for inference ordering.
"""
import torch
import torch.nn.functional as F

from .morai_bbox_corner import (
    HEADING_NORM_FLOOR, MIN_PREDICTED_SIZE_M, SIZE_FLOOR_M,
    SIZE_VALIDITY_WEIGHT, SMOOTH_L1_BETA, _corners, _validate_inputs,
    corner_bbox_loss,
)


def _cell_cost(prediction, target, assigned):
    """Existing corner-plus-validity formula, without aggregation or GT weights.

    The caller evaluates this selector under ``torch.no_grad``. Validation and
    geometry mirror the sealed corner objective; all assigned cells are checked
    before selection, including cells that do not ultimately win.
    """
    dtype = torch.promote_types(prediction.dtype, target.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
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
            error, torch.zeros_like(error), beta=SMOOTH_L1_BETA,
            reduction='none').mean((1, 2)))
    corner_cell = torch.stack(alternatives, dim=-1).min(dim=-1).values
    minimum_dimension = denominator.min(dim=-1, keepdim=True).values
    validity_cell = SIZE_VALIDITY_WEIGHT * (
        F.relu(MIN_PREDICTED_SIZE_M - pred_size) / minimum_dimension).mean(-1)
    return corner_cell + validity_cell


def best_owned_corner_bbox_loss(prediction, target, instance_ids, valid_mask=None):
    """Return ``(loss, terms)`` from raw ``(B,2500,8)`` LAV tensors.

    Among every valid assigned query for each (frame, local instance ID), select
    the detached minimum of the existing total per-cell corner-plus-validity
    cost. GT probability does not bias selection, and there is no probability
    threshold beyond the existing requirement that assigned GT is in (0,1].
    Exact cost ties choose the smallest global flattened cell index.

    Evaluate ``corner_bbox_loss`` on that boolean selection mask. Its unchanged
    probability mean with one selected query cancels within each object, then
    objects are equal within a frame and positive frames are equal. Only the
    winner receives auxiliary geometry gradients; confidence, speed and ignored
    queries receive none. The native losses still supervise every native query.
    The external auxiliary weight is chosen by the caller, not applied here.

    Input validation, signed-size validity, low-precision promotion and empty
    finite differentiable zero retain the original corner-loss behavior.
    Selecting among queries is discrete and need not be differentiable at ties.
    """
    assigned = _validate_inputs(prediction, target, instance_ids, valid_mask)
    selected = torch.zeros_like(assigned)
    object_count = 0
    with torch.no_grad():
        if assigned.any():
            costs = _cell_cost(prediction, target, assigned)
            positions = assigned.nonzero(as_tuple=False)
            pairs = torch.stack((positions[:, 0], instance_ids[assigned]), dim=-1)
            unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
            object_count = len(unique_pairs)
            for pair_index in range(object_count):
                members = (inverse == pair_index).nonzero(as_tuple=False)[:, 0]
                # positions/members are ascending global cell indices, so
                # torch.argmin's first minimum supplies the deterministic tie.
                winner = members[costs[members].argmin()]
                selected[positions[winner, 0], positions[winner, 1]] = True
    loss, old_terms = corner_bbox_loss(prediction, target, instance_ids, valid_mask=selected)
    objects = instance_ids.new_tensor(object_count)
    return loss, {
        'bbox_proposal_geometry': old_terms['bbox_corner_geometry'],
        'bbox_proposal_size_validity': old_terms['bbox_corner_size_validity'],
        'bbox_proposal_assigned_cells': assigned.sum().detach(),
        'bbox_proposal_selected_cells': selected.sum().detach(),
        'bbox_proposal_assigned_objects': objects,
        'bbox_proposal_selected_objects': objects,
        'bbox_proposal_positive_frames': assigned.any(dim=1).sum().detach(),
        'bbox_proposal_selected_invalid_size_cells': old_terms['bbox_corner_invalid_size_cells'],
        'bbox_proposal_selected_small_heading_cells': old_terms['bbox_corner_small_heading_cells'],
    }
