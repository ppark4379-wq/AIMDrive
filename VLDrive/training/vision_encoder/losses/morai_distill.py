"""Output distillation from a frozen reference model, for geometry-only fine-tuning.

Keeps detection confidence, object speed and the native traffic-light logits of
the student on the reference model's outputs while bbox geometry (channels 1-6)
is free to change. Confidence is matched on logits so that the decision
threshold behaviour, not just the sigmoid probability, is preserved.
"""
import torch
import torch.nn.functional as F


def distillation_loss(student, reference, valid_frames=None, geometry_frames=None, *,
                      speed_frames=None, geometry_weight=1.0, negative_speed_weight=0.0):
    """``student``/``reference``: model output tuples ``(det, waypoints, light, ...)``.

    Returns ``(loss, terms)``. ``valid_frames`` is an optional ``(B,)`` bool mask
    for the detection terms (frames whose object labels are unusable still keep
    the traffic-light term). ``geometry_frames`` is an optional ``(B,)`` bool mask
    of frames whose box geometry (channels 1-6) is also held on the reference,
    weighted by the reference's detection probability, so geometry training
    aimed at one scene type cannot drift the boxes of the other scene types.

    ``speed_frames`` (optional ``(B,)`` bool) holds the speed channel on frames
    of its own, independent of ``valid_frames``: a fine-tune that frees the
    confidence/geometry of new objects on some frames can still keep every
    already-detected object's speed (weighted by the reference probability, so
    cells the reference does not detect stay free to learn). ``geometry_weight``
    scales the geometry term, which is numerically small (centimetres).
    ``negative_speed_weight`` penalises negative raw speed where the student is
    confident; downstream treats a negative speed as invalid.
    """
    det_s, det_r = student[0], reference[0].detach()
    light_s, light_r = student[2], reference[2].detach()
    if det_s.shape != det_r.shape:
        raise ValueError('student and reference detection tensors differ')
    logit = F.mse_loss(det_s[..., 0], det_r[..., 0], reduction='none').mean(-1)
    weight = torch.sigmoid(det_r[..., 0])
    speed = (weight * (det_s[..., 7] - det_r[..., 7]).abs()).sum(-1) / weight.sum(-1).clamp_min(1e-6)
    confident = torch.sigmoid(det_s[..., 0]).detach()
    negative = (confident * F.relu(-det_s[..., 7])).sum(-1) / confident.sum(-1).clamp_min(1e-6)

    def masked_mean(value, mask):
        if mask is None:
            return value.mean()
        keep = mask.to(value.dtype)
        return (value * keep).sum() / keep.sum().clamp_min(1.)

    logit = masked_mean(logit, valid_frames)
    speed_mask = speed_frames if speed_frames is not None else valid_frames
    speed = masked_mean(speed, speed_mask)
    negative = masked_mean(negative, speed_mask)
    light = F.mse_loss(light_s, light_r)
    total = logit + speed + light
    terms = {'distill_logit': logit.detach(), 'distill_speed': speed.detach(),
             'distill_light': light.detach()}
    if negative_speed_weight:
        total = total + negative_speed_weight * negative
        terms['distill_negative_speed'] = negative.detach()
    if geometry_frames is not None:
        per_frame = ((weight[..., None] * (det_s[..., 1:7] - det_r[..., 1:7]).abs()).sum((-1, -2))
                     / (6 * weight.sum(-1)).clamp_min(1e-6))
        keep = geometry_frames.to(per_frame.dtype)
        geometry = (per_frame * keep).sum() / keep.sum().clamp_min(1.)
        total = total + geometry_weight * geometry
        terms['distill_geometry'] = geometry.detach()
    return total, terms
