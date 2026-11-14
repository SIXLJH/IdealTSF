# IdealTSF

Implementation of **IdealTSF: Can Non-Ideal Data Contribute to Enhancing the
Performance of Time Series Forecasting Models?** (AAAI 2026).

The code provides negative-sample pretraining, positive-sample fine-tuning,
ECOS optimization, and validation-selected forecasting evaluation. An optional
lightweight channel bridge is included for high-dimensional datasets. This
bridge is an extension and should be evaluated separately from the published
IdealTSF results.

## Installation

Use Python 3.8 or newer with a compatible PyTorch build. Install the PyTorch
version appropriate for your device, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

## Datasets

Place the original benchmark CSV files under `dataset/`:

| Dataset | Preferred path | Channels |
| --- | --- | ---: |
| ETTh1 | `dataset/ETT-small/ETTh1.csv` | 7 |
| ETTh2 | `dataset/ETT-small/ETTh2.csv` | 7 |
| ETTm1 | `dataset/ETT-small/ETTm1.csv` | 7 |
| ETTm2 | `dataset/ETT-small/ETTm2.csv` | 7 |
| WTH / Weather | `dataset/weather/weather.csv` | 21 |
| ECL / Electricity | `dataset/electricity/electricity.csv` | 321 |
| Traffic | `dataset/traffic/traffic.csv` | 862 |
| Exchange | `dataset/exchange_rate/exchange_rate.csv` | 8 |

Flat paths such as `dataset/ETTm1.csv`, `dataset/weather.csv`,
`dataset/electricity.csv`, and `dataset/traffic.csv` are also supported.
Set `DATA_FILE` in a launcher or pass `--data-file` to use a different location.
Dataset files and trained weights are not included.

Validate a dataset before training:

```bash
python run_demo.py --dataset ETTm1 --pred-lens 96 --dry-run
```

## Training

Each dataset has a Bash launcher and a Windows batch launcher in `scripts/`.
Launchers run horizons 96, 192, 336, and 720 by default. For a single horizon:

Linux or Bash:

```bash
bash scripts/ETTm1.sh --pred-lens 96 --device cuda:0
bash scripts/ECL.sh --pred-lens 96 --device cuda:0
```

Windows:

```bat
scripts\ETTm1.bat --pred-lens 96 --device cuda:0
scripts\ECL.bat --pred-lens 96 --device cuda:0
```

The Python entry point can also be used directly:

```bash
python run_demo.py --dataset ETTh2 --pred-lens 96 --device cuda:0
```

When the device is `auto`, CUDA is selected when available. Use `--device cpu`
for small CPU runs.

## Editable dataset settings

Every launcher contains a complete configuration block with 55 settings.
Edit the `.sh` file on Linux or the `.bat` file on Windows, then run that
launcher. Boolean settings use `1` for enabled and `0` for disabled.

| Group | Main settings |
| --- | --- |
| Data | `DATA_ROOT`, `DATA_FILE`, `SEQ_LEN`, `PRED_LENS`, `FEATURES`, `TARGET`, `SCALE`, `STRIDE` |
| Runtime | `PYTHON_BIN`, `DEVICE`, `SEED`, `SEEDS`, `NUM_WORKERS`, `NUM_THREADS` |
| Model | `HID_DIM`, `REVIN`, `CA_MODE`, `NUM_BRIDGES` |
| Optimization | Stage epochs, batch sizes and learning rates; `WEIGHT_DECAY`, `RHO`, `PATIENCE` |
| Negative samples | `NOISE_STD`, `NOISE_SCALES`, erasing settings and stable-jump settings |
| Positive samples | Auxiliary loss weight, smoothing window, mixture coefficient and preprocessing settings |
| Execution | `OUTPUT_DIR`, `DRY_RUN`, `EVAL_ONLY`, `CHECKPOINT` |

For example, change these values in `scripts/ECL.sh`:

```bash
PRED_LENS=(96)
PRETRAIN_LR=5e-4
FINETUNE_LR=5e-5
RHO=0.9
CA_MODE="bridge"
NUM_BRIDGES=16
```

The equivalent Windows configuration is:

```bat
set "PRED_LENS=96"
set "PRETRAIN_LR=5e-4"
set "FINETUNE_LR=5e-5"
set "RHO=0.9"
set "CA_MODE=bridge"
set "NUM_BRIDGES=16"
```

Arguments supplied after a launcher override its settings. If `SEEDS` is
nonempty, it takes precedence over the single `SEED` setting. Comments beside
each variable describe its meaning and when it applies. The current stable
jump sampler is symmetric and requires `STABLE_BETA=0`.

## Batch experiments

Run all eight datasets using their platform-specific launcher settings:

```bash
python scripts/run_all.py --device cuda:0
```

Select datasets or override horizons and seeds:

```bash
python scripts/run_all.py --datasets ETTm1 ETTm2 WTH ECL Traffic --pred-lens 96 --seeds 2025 2026 2027 --device cuda:0
```

When horizons or seeds are omitted, each dataset script retains its own
settings. A failed experiment stops the batch.

## Data protocol

The following intervals are zero-based and have exclusive right endpoints.
Validation and test inputs may include earlier history, while all forecast
targets remain within their assigned split.

| Dataset family | Training rows | Validation targets | Test targets |
| --- | --- | --- | --- |
| ETTh1 / ETTh2 | `[0, 8640)` | `[8640, 11520)` | `[11520, 14400)` |
| ETTm1 / ETTm2 | `[0, 34560)` | `[34560, 46080)` | `[46080, 57600)` |
| WTH / ECL / Traffic / Exchange | First `floor(0.7N)` rows | Remaining middle rows | Last `floor(0.2N)` rows |

ETT uses fixed 12/4/4-month boundaries; ETTm has four samples per hour. The
first training forecast starts at `seq_len`. Standardization is fitted on
training rows only. Missing values are rejected by default; optional forward
fill uses past observations and does not fill leading missing values.

Windows are generated lazily. Evaluation includes the final partial batch
and averages MSE/MAE over all target elements. Metrics use standardized values
unless scaling is disabled. `M` forecasts all variables, `S` uses only the
named target, and `MS` uses all inputs to forecast the named target.

## Model variants and defaults

`CA_MODE=off` retains the original dense channel attention. `CA_MODE=bridge`
gathers channel information into a small set of learned representatives and
broadcasts it back using the existing projections. `auto` selects a mode from
the dataset and feature configuration.

| Dataset | Launcher mode | Bridges | Training batch | Evaluation batch |
| --- | --- | ---: | ---: | ---: |
| ETT datasets / Exchange | off | Unused | 32 | 256 |
| WTH | bridge | 4 | 32 | 128 |
| ECL | bridge | 16 | 32 | 32 |
| Traffic | bridge | 16 | 8 | 8 |

Common defaults are input length 512, hidden dimension 16, seed 2025, 30
pretraining epochs, 50 fine-tuning epochs, learning rates `5e-4` / `5e-5`,
weight decay `1e-4`, and ECOS radius `0.9`. Stable-jump augmentation and the
positive auxiliary loss are enabled. Positive loss weight, window and input
mixture coefficient default to `0.2`, `5` and `0.85`. Early stopping is disabled
by default (`PATIENCE=0`). These are the code defaults; reproduce and document
the configuration used for each reported experiment.

Pretraining returns its final state. Fine-tuning selects and restores the
checkpoint with the lowest validation MSE. The training entry point evaluates
test data once after restoring that checkpoint.

## Checkpoints and results

Each run creates a unique directory under `outputs/`:

| File | Content |
| --- | --- |
| `best_val.pt` | Selected weights, architecture and data metadata |
| `metrics.json` | Validation and test MSE/MAE |
| `config.json` | Resolved settings and runtime versions |
| `data_manifest.json` | Dataset hash, column order, scaler statistics and splits |
| `history.json` | Training and validation histories |
| `train.log` | Run output |

Evaluate an existing checkpoint using the original dataset and window settings:

```bash
python run_demo.py --dataset ETTm1 --seq-len 512 --pred-lens 96 --eval-only --checkpoint outputs/YOUR_RUN/best_val.pt
```

The architecture is restored from the checkpoint with strict state loading.
Data and split mismatches are rejected. To summarize completed training runs:

```bash
python scripts/summarize_results.py
```

The summarizer separates configurations and dataset snapshots, excludes
evaluation-only runs, and rejects duplicate configuration/seed entries.
Sample standard deviation is undefined for a single seed.

## References

- [IdealTSF, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/download/39826/43787).
- [TimeBridge, ICML 2025](https://proceedings.mlr.press/v267/liu25cb.html), a
  reference for cross-variable modeling. The lightweight bridge here is an
  independent simplification, not the full TimeBridge encoder.
- [ETDataset](https://github.com/zhouhaoyi/ETDataset), the ETT benchmark source.
