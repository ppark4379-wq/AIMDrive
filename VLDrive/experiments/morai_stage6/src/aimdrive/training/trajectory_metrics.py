"""Trajectory errors in metres, weighted by samples rather than batch means."""

import torch


HORIZONS_S = (0.5, 1.0, 1.5, 2.0, 2.5)


class TrajectoryMetrics:
    def __init__(self):
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

    @staticmethod
    def _report(stat):
        count = stat["count"]
        per_horizon = stat["distance"] / count
        xy = stat["absolute_xy"] / count
        return dict(samples=count, ade_m=per_horizon.mean().item(),
                    fde_m=per_horizon[-1].item(), horizon_l2_m=per_horizon.tolist(),
                    longitudinal_mae_m=xy[:, 0].tolist(), lateral_mae_m=xy[:, 1].tolist())

    def compute(self):
        if not self._runs:
            raise ValueError("Cannot report metrics for an empty evaluation")
        total = dict(count=sum(s["count"] for s in self._runs.values()),
                     distance=sum(s["distance"] for s in self._runs.values()),
                     absolute_xy=sum(s["absolute_xy"] for s in self._runs.values()))
        per_run = {run: self._report(stat) for run, stat in sorted(self._runs.items())}
        result = self._report(total)
        result.update(horizons_s=list(HORIZONS_S), per_run=per_run,
                      run_macro_ade_m=sum(r["ade_m"] for r in per_run.values()) / len(per_run),
                      run_macro_fde_m=sum(r["fde_m"] for r in per_run.values()) / len(per_run),
                      aggregation="one last-frame anchor per window; sample-weighted global mean")
        return result
