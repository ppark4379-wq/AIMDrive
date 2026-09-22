# CCDP fixed N × C debug experiment

Historical experiment record: on 2026-09-19 the completed experiment's runner
and dedicated config were removed during cleanup. Commands below record the
original procedure and are no longer runnable. Saved results remain under
`outputs/logs/ccdp_fixed_nc_20260918/`. Current training uses D2 `mean` in
`configs/vldrive_train.json`.

At the time of this experiment the default reconstruction reduction was `sum`. This experiment adds only
`reconstruction_reduction="fixed_nc"`, guarded in the training model by
`training_variant="fixed_nc_debug"`. The latter is a validation marker and does
not change forward computation. Other repair options are rejected in this mode.

For an input with N=106 visual tokens and C=256 features:

```python
L_rec_current = numerator.mean()
L_rec_fixed_nc = numerator.mean() / 27136
```

The denominator includes every input token, independently of the selected mask.
It is not the number of dropped tokens. The mask stays differentiable. Existing
`sum` and dropped-token `mean` paths are preserved. `mean` is not used here.

`configs/ccdp_fixed_nc_diagnostic.json` controls the paired experiment. Both
branches strictly restore the same original checkpoint first. D1 then changes
only the reduction and its validation marker. Both use fresh, identical CPU
AdamW optimizers because the source checkpoint does not contain optimizer state.
The schedule starts at update 40, with the original 5595-update horizon. Each
branch runs 300 new updates on the same nine fixed training clips, with the same
per-update batch index and seed. Three temporal validation clips are held out.

Run from the repository root in the existing Ubuntu-20.04 WSL environment:

```bash
export PYTHONPATH=VLDrive/.runtime/stage2_wsl_deps:VLDrive
/opt/miniforge3/envs/lmdrive/bin/python VLDrive/tools/diagnose_ccdp_fixed_nc.py \
  --config VLDrive/configs/ccdp_fixed_nc_diagnostic.json --phase D0
/opt/miniforge3/envs/lmdrive/bin/python VLDrive/tools/diagnose_ccdp_fixed_nc.py \
  --config VLDrive/configs/ccdp_fixed_nc_diagnostic.json --phase D1
```

The runner refuses to overwrite an existing branch directory. To repeat the
pair, copy the experiment config and choose a new `output` directory. Keep all
other settings unchanged. Production training config/checkpoints are not changed.

Every update logs loss terms, weighted pruning, selector probability statistics,
probability bins, gap histogram, Gumbel masks and counterfactual argmax masks
before/after rescue. Additional fixed-batch gradient probes run at 0/120/300;
full train/validation trajectory evaluation uses both Gumbel and actual argmax
at those same steps. Probes preserve RNG state and never step the optimizer.
Saved `model-final.pt` files contain weights, not optimizer state.

`reconstruction_stats` retains raw-sum diagnostic fields for both branches.
Use `losses.reconstruction_loss` for the active objective; raw fields such as
`loss_sum_N_C_mean_frames` and `fixed_error_direct_dL_dmask_mean` are explicitly
unnormalized and should not be mistaken for D1's active loss or gradient.

The summarizer checks initial parameter hashes, forward tensors, predictions,
seed/batch/LR schedules and protected files before writing `summary.json`.
Comparing initial gradients isolates the reduction scale. Later gradients also
reflect different learned weights. Total loss values across different reductions
are not directly comparable as a driving-quality score.

These small, straight-scene diagnostic subsets do not establish general driving
performance. A 30% stochastic keep rate alone does not establish useful token
selection; inspect argmax collapse, probability separation and trajectory loss.
