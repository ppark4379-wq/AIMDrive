# VLDrive Training Dashboard

Separate, read-only Streamlit process watching one run's `training.jsonl`.
No training controls, checkpoint loading, validation charts, model imports, or GPU polling.
The optional training-side observer depends on PyTorch only, not Streamlit/Plotly.

## Files

- `app.py`, `styles.py`, `plots.py`: UI and scientific raw/EMA plots.
- `log_reader.py`: incremental binary JSONL tail and cached run metadata.
- `metrics.py`: aliases, resume-aware progress, rolling ETA, display-only EMA.
- `exporters.py`: standard-library CSV CLI and atomic `exports/` writes.
- `telemetry.py`: detached training observations and sparse module gradient snapshots.
- `demo.py`, `render_preview.py`, `dashboard_preview.png`: explicitly synthetic preview of the real UI.
- `tests/`: reader/export, live browser isolation, and tiny-model RNG/weight equivalence tests.
- `requirements.txt`: optional UI dependencies, isolated from the training environment.

## Automatic launch with training

Starting stage 2 training now starts the Dashboard server and opens the default
browser automatically, after preflight passes and before baseline evaluation:

```bash
python train.py --config vldrive_train.json
```

The config filename is resolved from `configs/`, and a new timestamped output directory
is created under `outputs/logs/`. The explicit `vldrive` stage and `--output` remain
supported. No separate visualization command is needed.
`--mode preflight` and `--mode evaluate` do not launch the Dashboard.
The `dashboard` object in `configs/vldrive_train.json` controls `enabled` (true),
`open_browser` (true), `port` (8501), and `refresh_seconds` (2, supported range 1..5).
Set `enabled` to false to disable automatic launch, or `open_browser` to false
to keep just the server. An occupied preferred port falls back to a free port.

`launcher.py` uses the isolated `.runtime/dashboard_venv` Python when available.
On this machine WSL training automatically starts its Windows Python with converted
Windows paths; no Streamlit installation in the training environment is needed.
On native Linux it prefers `.runtime/dashboard_venv/bin/python`, then the current
interpreter. Install the UI requirements in that environment on a new machine.
Startup failures are reported without stopping training. The console prints the
actual URL; `<run>/dashboard/status.json`, `bootstrap.log`, and `server.log`
contain startup details.
The browser may initially show a waiting state during baseline evaluation.
Closing the browser does not stop the server or training. The independent server
stays available after training exits so the completed run can still be reviewed.
The status file records startup status and the server PID, not a live health check.

## Manual inspection of an existing run from PowerShell

The Windows environment `.runtime/dashboard_venv` has been prepared. WSL can keep
training while this Windows process reads the same files on C:.

```powershell
cd C:\Users\autonav009\Desktop\AIMDrive\VLDrive
& .\.runtime\dashboard_venv\Scripts\python.exe -m streamlit run utils\training_dashboard\app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false -- --run-dir outputs\logs\YOUR_RUN
```

Open `http://localhost:8501`. Replace `YOUR_RUN` with the **actual** training output directory.
A missing `training.jsonl` displays a waiting state. Closing the browser or stopping
Streamlit does not stop training. To inspect the prepared synthetic UI, replace
`--run-dir outputs\logs\YOUR_RUN` with `--demo`. Refresh defaults to 2 s; `--refresh 1` to `5` is supported.

For an existing real log, a runnable example is:

```powershell
& .\.runtime\dashboard_venv\Scripts\python.exe -m streamlit run utils\training_dashboard\app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false -- --run-dir outputs\logs\vldrive_short_diagnostic_20260918
```

To install elsewhere, create a separate Python environment and run
`python -m pip install -r utils/training_dashboard/requirements.txt`.
The base training requirements do not need any UI packages.

## CSV export

Click **Export CSV**: creates both files in `<run>/exports/` and enables download buttons.
This is a snapshot at the displayed update; click Export again to include new updates.
CSV generation occurs on demand, not every refresh. The original JSONL and sidecars
are never modified. Demo mode creates download data only, not a real run.
An interrupted final line is held until its newline arrives; malformed complete
lines are counted and skipped. No reader file handle remains open between polls.
An initial very large log is scanned in bounded 8 MiB chunks per refresh.

Without Streamlit or PyTorch:

```powershell
python -m utils.training_dashboard.exporters --run-dir outputs\logs\YOUR_RUN
```

`training_metrics.csv`:

```text
update,epoch,timestamp,elapsed_s,progress_pct,eta_s,expected_finish,update_time_s,
updates_per_sec,learning_rate,total_loss,l_way,l_prun,l_rec,gumbel_keep_ratio,
p_keep_mean,p_keep_std,selected_tokens_mean,selected_tokens_min,selected_tokens_max,
selected_tokens_target,grad_norm,gpu_memory_gb,gpu_utilization
```

`module_gradients.csv`:

```text
update,selector,mefa,qformer,connector_projection,tinyllama,waypoint_head
```

Raw metrics remain raw, without EMA. Progress and ETA are explicitly derived columns.
Missing data are blank, including historical timestamps/GPU/module gradients that
were never logged. The module CSV has rows only at observed gradient snapshots.
UTF-8 BOM in on-disk CSV supports Excel; download CSV uses UTF-8.

## Existing versus new logging

| Classification | Fields / cost |
|---|---|
| Already in JSONL | `loss`, `waypoint_loss`, `pruning_loss`, `reconstruction_loss`, `sampled_retention`, `soft_retention`, `lr`, `grad_norm`, `elapsed_s` |
| Small additions | epoch/global schedule/session start, UTC timestamp, update wall time, detached `p_keep_std` and selected token mean/min/max, CUDA allocated memory, first-row system metadata |
| Sparse additional reduction | module gradient L2 before clipping, every 100 global updates by default; the full TinyLlama group scans all trainable decoder/embedding gradients |
| Deliberately absent | additional forward, per-update nvidia-smi, GPU utilization sampling, validation metrics, inferred or fabricated historical values |

Optional settings in the existing training config (defaults shown; not required):

```json
"logging": {"telemetry_enabled": true, "gradient_interval": 100}
```

`gradient_interval: 0` disables module snapshots. `telemetry_enabled: false` disables
the forward observer/module snapshots/GPU metadata, while basic existing logging and
schedule/timestamp fields remain. There is no runtime dependency on the dashboard process.
Observer exceptions are reported as `telemetry_error`; hooks are removed when training exits.

`p_keep_std` is population std across all B×T×N token probabilities in that update.
Selected-token stats are across B×T frames. Gumbel retention is the training mask
**after the existing empty-frame rescue**. Target lines/selected-token target come
from recorded model configuration (106×0.3=31.8 in the current run), not an enforced top-k.
Module gradients and the existing overall grad_norm are pre-clip L2 values.
TinyLlama includes all trainable layers and embeddings, rather than extrapolating
the first/last-layer probes in `gradient_groups()`; the other prefixes use that method directly.
GPU memory is current allocated **GiB**, despite the compatibility CSV column name `gpu_memory_gb`.

## Time and resumed runs

New rows record the absolute update, epoch and total schedule. For older logs,
total updates come from `report.json.schedule_updates` or epochs × the recorded
training-window indices; the reader never guesses from the final log line.
Session elapsed is the current invocation's **training-loop** wall time, excluding
initial model loading/preflight. It includes inter-update epoch/checkpoint work.
ETA uses the median of the most recent 200 update cadence observations, after at
least 20 samples. A new session/resetted elapsed time clears that window. Missing
`update_time_s` in old logs uses adjacent elapsed/update differences, with the first
row left unknown. Progress remains global across resume.

UI expected finish = current local time + remaining ETA. It is hidden when logs are
stale or the update schedule is complete; inactivity can also mean checkpointing
or evaluation, so it is not labeled as proof that the trainer stopped. Completion
means update schedule completion, not that final evaluation/checkpoint writing finished.
CSV expected_finish uses that row's UTC timestamp + ETA; it stays blank for old logs
without timestamps. Median ETA can understate rare long validation/checkpoint pauses.

All training tensors used by telemetry are detached. Reductions consume no RNG and
do not mutate gradients/weights or perform an extra forward. There is nevertheless
nonzero reduction/CPU-copy overhead; full TinyLlama gradient scans every 100 updates
cost more than the scalar stats. No long/full-model training was run to claim a
production overhead percentage. UI rendering/export work is entirely in the other process.
Plots retain extrema when display-decimating long runs; CSV exports all observations.

## Validation and preview

CPU/CUDA tiny-model logging on/off comparisons require identical RNG state, hard
Gumbel masks, loss, grad norm and final parameters (zero tolerance). Independent
append producers verify that the reader and live dashboard can exit while writing
continues. These are synthetic tests, not real VLDrive training.

Verified on 2026-09-19: combined existing/dashboard WSL suite **275 passed, 3 skipped,
30 subtests passed**. The skips were the optional TinyLlama checkpoint integration
test and the two Windows-only browser/server tests. WSL-to-Windows automatic launch
and server survival after bootstrap exit passed. Windows separately passed 5 launcher
tests (1 WSL-only skip), including occupied-port fallback and browser handoff, all 8 reader/CSV
tests and the real Edge live-append/export/shutdown test. The actual UI rendered
6 Plotly charts at a 1920×1080 viewport with no browser errors. Existing
`vldrive_short_diagnostic_20260918/training.jsonl` was also exported to its own
`exports/` directory; unavailable historical gradient values produced a header-only
module CSV, not invented data. No real model training or checkpoint modification occurred.

```bash
python -m pytest -q utils/training_dashboard/tests
```

Windows file/CSV tests need only the standard library:

```powershell
python -m unittest discover -s utils/training_dashboard/tests -p test_reader_export.py -v
```

For an actual browser capture, install `playwright==1.55.0` in the isolated UI environment;
the prepared Windows environment already has it. `render_preview.py` uses installed Edge,
starts its own headless local server, and shuts it down afterward:

```powershell
& .\.runtime\dashboard_venv\Scripts\python.exe utils\training_dashboard\render_preview.py
```

`dashboard_preview.png` is a **synthetic UI preview, not a learning result**.
Runtime telemetry source is included in checkpoint source provenance; UI-only source
changes are not. Existing strict checkpoint/code identity checks remain in effect,
so a checkpoint made before training-source changes is not silently exact-resumed.

Framework references: [Streamlit fragments](https://docs.streamlit.io/develop/concepts/architecture/fragments),
[Plotly scientific line styling](https://plotly.com/python/line-charts/).
