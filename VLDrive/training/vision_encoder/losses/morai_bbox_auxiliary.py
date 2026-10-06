"""Opt-in, instance-balanced relative center/size loss for raw MORAI LAV boxes.

This auxiliary leaves the eight head channels and the physical GT boxes intact.
Its external task weight is chosen by the caller; it does not replace LAV loss.
"""
import torch
import torch.nn.functional as F


def relative_bbox_loss(prediction, target, instance_ids, valid_mask=None):
    """Return ``(loss, terms)`` for raw ``(B, 2500, 8)`` LAV tensors.

    ``instance_ids`` is an int64 ``(B, 2500)`` tensor: -1 means unassigned and
    nonnegative IDs identify GT objects independently within each frame. The
    optional boolean mask has shape ``(B,)`` (frame validity) or ``(B, 2500)``.
    Only assigned, valid cells contribute, and their target probabilities must
    be in (0, 1], their geometry finite, their sizes positive, and their heading
    nonzero. Malformed active targets raise rather than silently drop objects.

    Center error is decoded to metres, rotated into the *GT* longitudinal and
    lateral axes, then divided by GT full length/width (each floored at 0.5 m).
    Decoded size error uses the same denominators. Smooth L1 (beta=0.2) averages
    the two center coordinates and two sizes separately; loss is half of each.
    Predicted sizes are not clamped, so invalid negative sizes receive gradient.
    Heading, confidence, and speed predictions have no auxiliary gradient.

    Cells are weighted by GT probability within each instance; instances have
    equal weight within each frame; frames with assigned cells have equal
    weight. Empty/invalid frames do not dilute the average. No assigned cells
    returns a finite, differentiable zero. Terms contain the *unweighted*
    ``bbox_relative_center`` and ``bbox_relative_size`` scalar tensors.
    """
    if prediction.ndim != 3 or tuple(prediction.shape[1:]) != (2500, 8):
        raise ValueError("prediction must have shape (B, 2500, 8)")
    if target.shape != prediction.shape:
        raise ValueError("target must have the same shape as prediction")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating point tensors")
    if instance_ids.shape != prediction.shape[:2]:
        raise ValueError("instance_ids must have shape (B, 2500)")
    if instance_ids.dtype != torch.long:
        raise TypeError("instance_ids must have dtype torch.int64")
    if target.device != prediction.device or instance_ids.device != prediction.device:
        raise ValueError("prediction, target and instance_ids must share a device")
    if (instance_ids < -1).any():
        raise ValueError("instance_ids must be -1 or nonnegative")

    assigned = instance_ids >= 0
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have dtype torch.bool")
        if valid_mask.device != prediction.device:
            raise ValueError("valid_mask must share the prediction device")
        if valid_mask.shape == prediction.shape[:1]:
            assigned = assigned & valid_mask[:, None]
        elif valid_mask.shape == prediction.shape[:2]:
            assigned = assigned & valid_mask
        else:
            raise ValueError("valid_mask must have shape (B,) or (B, 2500)")

    # Empty selection sum is connected to prediction without touching NaNs in
    # ignored/background cells. Multiplying prediction.sum() by zero is unsafe.
    zero = prediction.reshape(-1)[:0].sum()
    if not assigned.any():
        return zero, {"bbox_relative_center": zero, "bbox_relative_size": zero}

    selected_target = target[assigned]
    if not torch.isfinite(selected_target[:, :7]).all():
        raise ValueError("assigned target probability/geometry must be finite")
    probability = selected_target[:, 0]
    if ((probability <= 0) | (probability > 1)).any():
        raise ValueError("assigned target probability must be in (0, 1]")
    if (selected_target[:, 5:7] <= 0).any():
        raise ValueError("assigned target length and width must be positive")

    # Work in at least float32 even under mixed precision. Preserve float64 for
    # mathematical checks and callers explicitly using double precision.
    dtype = torch.promote_types(prediction.dtype, target.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    selected_target = selected_target.to(dtype=dtype)
    selected_prediction = prediction[assigned].to(dtype=dtype)
    if not torch.isfinite(selected_prediction[:, [1, 2, 5, 6]]).all():
        raise ValueError("assigned predicted center and size must be finite")
    probability = selected_target[:, 0]
    heading = torch.stack((selected_target[:, 3], -selected_target[:, 4]), dim=-1)
    heading_norm = torch.linalg.vector_norm(heading, dim=-1, keepdim=True)
    if (heading_norm <= 0).any() or not torch.isfinite(heading_norm).all():
        raise ValueError("assigned target heading must be finite and nonzero")
    heading = heading / heading_norm
    transverse = torch.stack((-heading[:, 1], heading[:, 0]), dim=-1)

    scale = selected_target.new_tensor([7., 4.])
    gt_size = selected_target[:, 5:7] * scale
    denominator = gt_size.clamp_min(0.5)
    delta = (selected_prediction[:, 1:3] - selected_target[:, 1:3]) * 3.5
    center_error = torch.stack(((delta * heading).sum(-1),
                                (delta * transverse).sum(-1)), dim=-1) / denominator
    size_error = (selected_prediction[:, 5:7] - selected_target[:, 5:7]) * scale / denominator
    center_cell = F.smooth_l1_loss(center_error, torch.zeros_like(center_error),
                                  beta=0.2, reduction="none").mean(-1)
    size_cell = F.smooth_l1_loss(size_error, torch.zeros_like(size_error),
                                beta=0.2, reduction="none").mean(-1)
    cell_terms = torch.stack((center_cell, size_cell), dim=-1)

    # Pairing frame and local object ID prevents unrelated objects with the
    # same per-frame ID from merging; unique avoids allocation by maximum ID.
    frame_ids = assigned.nonzero(as_tuple=False)[:, 0]
    pairs = torch.stack((frame_ids, instance_ids[assigned]), dim=-1)
    unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    count = len(unique_pairs)
    instance_weight = probability.new_zeros(count).scatter_add(0, inverse, probability)
    instance_terms = cell_terms.new_zeros((count, 2)).scatter_add(
        0, inverse[:, None].expand(-1, 2), cell_terms * probability[:, None])
    instance_terms = instance_terms / instance_weight[:, None]

    batch = prediction.shape[0]
    instance_frames = unique_pairs[:, 0]
    frame_count = probability.new_zeros(batch).scatter_add(
        0, instance_frames, torch.ones_like(instance_weight))
    frame_terms = instance_terms.new_zeros((batch, 2)).scatter_add(
        0, instance_frames[:, None].expand(-1, 2), instance_terms)
    positive_frames = frame_count > 0
    means = (frame_terms[positive_frames] / frame_count[positive_frames, None]).mean(0)
    return 0.5 * means.sum(), {"bbox_relative_center": means[0],
                               "bbox_relative_size": means[1]}
