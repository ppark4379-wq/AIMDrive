"""Trajectory errors in metres, weighted by samples rather than batch means."""

import torch


HORIZONS_S = (0.5, 1.0, 1.5, 2.0, 2.5)


class TokenRetentionMetrics:
    """Count deterministic selections across window/frame occurrences.

    Bounds are diagnostic acceptance criteria, not evidence of semantic token
    quality. The caller evaluates the real inference policy, without Gumbel noise.
    """
    def __init__(self, total_tokens=106):
        if total_tokens < 2:
            raise ValueError('At least two visual tokens required')
        self.total_tokens = total_tokens
        self.histogram = torch.zeros(total_tokens + 1, dtype=torch.long)

    def update(self, counts):
        value = counts.detach().double().cpu().flatten()
        if not value.numel() or not bool(torch.isfinite(value).all()) or not bool(value.eq(value.round()).all()):
            raise ValueError('Expected finite integer token counts')
        if bool(((value < 1) | (value > self.total_tokens)).any()):
            raise ValueError('Invalid retained-token count')
        self.histogram += torch.bincount(value.long(), minlength=self.total_tokens + 1)

    def compute(self, target_retention=.3, tolerance=.15):
        frames = int(self.histogram.sum())
        if not frames:
            raise ValueError('Cannot report empty token counts')
        support = self.histogram.nonzero().flatten()
        mean = float((self.histogram * torch.arange(self.total_tokens + 1)).sum()) / frames
        ratio = mean / self.total_tokens
        all_keep = int(self.histogram[-1]) / frames
        one_token = int(self.histogram[1]) / frames
        checks = dict(mean_near_target=abs(ratio - target_retention) <= tolerance,
                      not_all_keep_collapse=all_keep < .95, not_one_token_collapse=one_token < .95)
        return dict(frame_occurrences=frames, total_tokens=self.total_tokens, mean_tokens=mean,
                    min_tokens=int(support.min()), max_tokens=int(support.max()),
                    mean_retention=ratio, target_retention=target_retention, tolerance=tolerance,
                    all_keep_fraction=all_keep, one_token_fraction=one_token,
                    histogram={str(int(k)): int(self.histogram[k]) for k in support},
                    checks=checks, passed=all(checks.values()))


class TrajectoryMetrics:
    def __init__(self, *, coordinate_convention="forward_left"):
        # The expert cache preserves the LMDrive head's [left, -forward]
        # convention; existing Stage6E labels use [forward, left]. Signs do not
        # affect MAE, but swapping these axes changes its physical meaning.
        axes = {
            "forward_left": (0, 1),
            "lmdrive_left_negative_forward": (1, 0),
        }
        if coordinate_convention not in axes:
            raise ValueError("Unsupported trajectory coordinate convention: " + str(coordinate_convention))
        self.coordinate_convention = coordinate_convention
        self._longitudinal_axis, self._lateral_axis = axes[coordinate_convention]
        self._runs = {}
        self._anchors = set()

    def update(self, prediction, target, run_ids, sample_ids):
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("Prediction/target must have matching [B,5,2] shapes")
        if tuple(prediction.shape[1:]) != (5, 2):
            raise ValueError("Only the validated five-horizon XY contract is supported")
        if len(run_ids) != prediction.shape[0] or len(sample_ids) != len(run_ids):
            raise ValueError("Metric metadata batch mismatch")
        delta = prediction.detach().double().cpu() - target.detach().double().cpu()
        if not bool(torch.isfinite(delta).all()):
            raise ValueError("Nonfinite prediction or target")
        distances = delta.square().sum(-1).sqrt()
        for i, (run, sample) in enumerate(zip(run_ids, sample_ids)):
            if sample in self._anchors or not sample.startswith(run + "/"):
                raise ValueError("Duplicate or inconsistent evaluation anchor: " + sample)
            self._anchors.add(sample)
            if run not in self._runs:
                self._runs[run] = dict(count=0, distance=torch.zeros(5, dtype=torch.float64),
                                       absolute_xy=torch.zeros(5, 2, dtype=torch.float64))
            stat = self._runs[run]
            stat["count"] += 1
            stat["distance"] += distances[i]
            stat["absolute_xy"] += delta[i].abs()

    def _report(self, stat):
        count = stat["count"]
        per_horizon = stat["distance"] / count
        xy = stat["absolute_xy"] / count
        return dict(samples=count, ade_m=per_horizon.mean().item(),
                    fde_m=per_horizon[-1].item(), horizon_l2_m=per_horizon.tolist(),
                    longitudinal_mae_m=xy[:, self._longitudinal_axis].tolist(),
                    lateral_mae_m=xy[:, self._lateral_axis].tolist())

    def compute(self):
        if not self._runs:
            raise ValueError("Cannot report metrics for an empty evaluation")
        total = dict(count=sum(s["count"] for s in self._runs.values()),
                     distance=sum(s["distance"] for s in self._runs.values()),
                     absolute_xy=sum(s["absolute_xy"] for s in self._runs.values()))
        per_run = {run: self._report(stat) for run, stat in sorted(self._runs.items())}
        result = self._report(total)
        result.update(horizons_s=list(HORIZONS_S), per_run=per_run,
                      coordinate_convention=self.coordinate_convention,
                      run_macro_ade_m=sum(r["ade_m"] for r in per_run.values()) / len(per_run),
                      run_macro_fde_m=sum(r["fde_m"] for r in per_run.values()) / len(per_run),
                      aggregation="one last-frame anchor per window; sample-weighted global mean")
        return result
