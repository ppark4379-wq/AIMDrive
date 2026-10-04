"""BEV head regression on GT-positive cells, independent of detection threshold."""
import torch

def box_regression_sums(prediction, target):
    mask = target[..., 0] >= .5
    count = int(mask.sum())
    if not count:
        return 0, 0., 0., 0.
    p, t = prediction[mask], target[mask]
    # Same cell centre cancels. Offsets /3.5, full length /7, full width /4.
    centre = ((p[:, 1:3] - t[:, 1:3]) * 3.5).norm(dim=-1)
    length = (p[:, 5] - t[:, 5]).abs() * 7.
    width = (p[:, 6] - t[:, 6]).abs() * 4.
    return count, float(centre.double().sum()), float(length.double().sum()), float(width.double().sum())
