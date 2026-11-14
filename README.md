# IdealTSF (AAAI 2026)

Code for **IdealTSF: Can Non-Ideal Data Contribute to Enhancing the Performance of Time Series Forecasting Models?**

**Hua Wang, Jinghao Lu, and Fan Zhang**

Published in the *Proceedings of the AAAI Conference on Artificial Intelligence*, 2026.

[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/39826/43787)

## Overview

IdealTSF studies how non-ideal observations can contribute to time series forecasting through:

- **Negative-sample pretraining:** learning from perturbed historical observations.
- **Positive-sample fine-tuning:** using smoothed samples as auxiliary supervision.
- **ECOS optimization:** applying a two-step parameter update during training.

The repository supports ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic, and Exchange. It also includes an optional lightweight channel-bridge variant for high-dimensional datasets.

## Installation

Clone the repository and enter its root directory:

```bash
git clone https://github.com/SIXLJH/IdealTSF.git
cd IdealTSF
```

Use Python 3.8 or newer with a compatible PyTorch installation. An optional Conda environment can be created with:

```bash
conda create -n idealtsf python=3.10 -y
conda activate idealtsf
```

Install the PyTorch build appropriate for your device, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The main dependencies are PyTorch, NumPy, pandas, scikit-learn, and tqdm. An existing working PyTorch/CUDA environment can also be used.

## Dataset Download

### Download sources

The four ETT datasets are available from the original [ETDataset repository](https://github.com/zhouhaoyi/ETDataset). The table below links directly to their CSV files.

Weather, Electricity, Traffic, and Exchange are available from the [Time-Series-Library dataset collection](https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main). Open the corresponding folder in the table, select the CSV file, and click **Download**.

Alternative download locations provided by the [Time-Series-Library maintainers](https://github.com/thuml/Time-Series-Library#prepare-data):

- [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing)
- [Baidu Drive](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), access code: `i9iy`

### File placement

Download the original CSV files and place them at the following paths relative to the repository root:

| Dataset | Channels | Download | Local path |
| --- | ---: | --- | --- |
| ETTh1 | 7 | [CSV](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv) | `dataset/ETT-small/ETTh1.csv` |
| ETTh2 | 7 | [CSV](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv) | `dataset/ETT-small/ETTh2.csv` |
| ETTm1 | 7 | [CSV](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv) | `dataset/ETT-small/ETTm1.csv` |
| ETTm2 | 7 | [CSV](https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv) | `dataset/ETT-small/ETTm2.csv` |
| Weather (`WTH`) | 21 | [Files](https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main/weather) | `dataset/weather/weather.csv` |
| Electricity (`ECL`) | 321 | [Files](https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main/electricity) | `dataset/electricity/electricity.csv` |
| Traffic | 862 | [Files](https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main/traffic) | `dataset/traffic/traffic.csv` |
| Exchange | 8 | [Files](https://huggingface.co/datasets/thuml/Time-Series-Library/tree/main/exchange_rate) | `dataset/exchange_rate/exchange_rate.csv` |

Keep the original timestamps, column order, and values. The code performs chronological splitting and training-only standardization automatically. Dataset files are not included in this repository.

Flat paths such as `dataset/ETTm1.csv`, `dataset/weather.csv`, and `dataset/electricity.csv` are also supported. To use another location, set `DATA_FILE` in a launcher or pass `--data-file`:

```bash
python run_demo.py --dataset ETTm1 --data-file /path/to/ETTm1.csv --pred-lens 96 --dry-run
```

### Validate the dataset

Before training, check that the file can be loaded and inspect its split boundaries and window counts:

```bash
python run_demo.py --dataset ETTm1 --pred-lens 96 --dry-run
```

Add `--strict-channels` to verify the standard benchmark channel count.

## Training

Run all commands from the repository root.

### Train one dataset

**Linux / Bash:**

```bash
bash scripts/ETTm1.sh --pred-lens 96 --device cuda:0
bash scripts/ECL.sh --pred-lens 96 --device cuda:0
```

**Windows CMD:**

```bat
scripts\ETTm1.bat --pred-lens 96 --device cuda:0
scripts\ECL.bat --pred-lens 96 --device cuda:0
```

The launchers support all eight datasets:

| Dataset | Bash launcher | Windows launcher |
| --- | --- | --- |
| ETTh1 | `scripts/ETTh1.sh` | `scripts/ETTh1.bat` |
| ETTh2 | `scripts/ETTh2.sh` | `scripts/ETTh2.bat` |
| ETTm1 | `scripts/ETTm1.sh` | `scripts/ETTm1.bat` |
| ETTm2 | `scripts/ETTm2.sh` | `scripts/ETTm2.bat` |
| Weather | `scripts/WTH.sh` | `scripts/WTH.bat` |
| Electricity | `scripts/ECL.sh` | `scripts/ECL.bat` |
| Traffic | `scripts/Traffic.sh` | `scripts/Traffic.bat` |
| Exchange | `scripts/Exchange.sh` | `scripts/Exchange.bat` |

Without a horizon override, each launcher runs prediction lengths **96, 192, 336, and 720**.

The Python entry point is also available:

```bash
python run_demo.py --dataset ETTh1 --seq-len 512 --pred-lens 96 --device cuda:0
```

The default `auto` device uses CUDA when available. For small CPU runs, use `--device cpu`.

### Run multiple datasets

Run all eight datasets with their launcher configurations:

```bash
python scripts/run_all.py --device cuda:0
```

Run a subset with selected horizons and multiple seeds:

```bash
python scripts/run_all.py --datasets ETTm1 ETTm2 WTH ECL Traffic --pred-lens 96 192 336 720 --seeds 2025 2026 2027 --device cuda:0
```

The batch entry point reads the platform-specific dataset scripts. When horizons or seeds are omitted, each script retains its own settings. A failed experiment stops the batch.

## Configuration

Each launcher contains **55 configuration settings** at the top of the file. Edit the `.sh` file on Linux or the `.bat` file on Windows, then run that launcher. Boolean settings use `1` for enabled and `0` for disabled.

| Group | Main settings |
| --- | --- |
| Data | `DATA_ROOT`, `DATA_FILE`, `SEQ_LEN`, `PRED_LENS`, `FEATURES`, `TARGET`, `SCALE`, `STRIDE` |
| Runtime | `PYTHON_BIN`, `DEVICE`, `SEED`, `SEEDS`, `NUM_WORKERS`, `NUM_THREADS` |
| Model | `HID_DIM`, `REVIN`, `CA_MODE`, `NUM_BRIDGES` |
| Optimization | Stage epochs, batch sizes, learning rates, `WEIGHT_DECAY`, `RHO`, `PATIENCE` |
| Negative samples | `NOISE_STD`, `NOISE_SCALES`, erasing settings, stable-jump settings |
| Positive samples | `POSITIVE_AUX`, `POSITIVE_LOSS_WEIGHT`, `POSITIVE_WINDOW_SIZE`, `POSITIVE_MIX_ALPHA`, preprocessing settings |
| Execution | `OUTPUT_DIR`, `DRY_RUN`, `EVAL_ONLY`, `CHECKPOINT` |

For example, edit `scripts/ECL.sh`:

```bash
PRED_LENS=(96)
PRETRAIN_EPOCHS=30
FINETUNE_EPOCHS=50
PRETRAIN_BATCH_SIZE=32
FINETUNE_BATCH_SIZE=32
PRETRAIN_LR=5e-4
FINETUNE_LR=5e-5
RHO=0.9
CA_MODE="bridge"
NUM_BRIDGES=16
```

Additional command-line arguments override launcher settings:

```bash
bash scripts/ECL.sh --pred-lens 96 --finetune-lr 0.0001 --no-positive-aux
```

If `SEEDS` is nonempty, it takes precedence over the single `SEED` setting. Comments in each script explain the individual parameters and their valid values.

## Evaluation Protocol

The ETT datasets use fixed **12/4/4-month** chronological splits. ETTm contains four samples per hour. Intervals below are zero-based with exclusive right endpoints:

| Dataset family | Training rows | Validation targets | Test targets |
| --- | --- | --- | --- |
| ETTh1 / ETTh2 | `[0, 8640)` | `[8640, 11520)` | `[11520, 14400)` |
| ETTm1 / ETTm2 | `[0, 34560)` | `[34560, 46080)` | `[46080, 57600)` |
| WTH / ECL / Traffic / Exchange | First `floor(0.7N)` rows | Remaining middle rows | Last `floor(0.2N)` rows |

- The first training forecast starts at `seq_len`.
- Validation and test inputs may include earlier history, while their forecast targets stay inside the assigned split.
- Standardization is fitted on training rows only.
- Pretraining returns its final state. Fine-tuning selects and restores the checkpoint with the lowest validation MSE.
- The training entry point evaluates the test set once after checkpoint restoration.
- MSE and MAE are averaged over all target elements, including the final partial batch, in standardized space unless scaling is disabled.

`FEATURES=M` forecasts all variables. `S` uses only the named target, while `MS` uses all input variables to forecast the named target.

## Channel Interaction Modes

`CA_MODE=off` retains the original dense channel attention. `CA_MODE=bridge` uses a small set of learned representatives to gather and redistribute information across channels. `auto` selects a mode from the dataset and feature configuration.

| Dataset | Default launcher mode | Bridges | Training batch | Evaluation batch |
| --- | --- | ---: | ---: | ---: |
| ETT datasets / Exchange | off | Unused | 32 | 256 |
| WTH | bridge | 4 | 32 | 128 |
| ECL | bridge | 16 | 32 | 32 |
| Traffic | bridge | 16 | 8 | 8 |

The lightweight bridge is an optional extension inspired by cross-variable modeling. Report its results as a separate IdealTSF variant. For a controlled comparison, keep the data, seeds, and optimization settings fixed and change only `--ca`:

```bash
python run_demo.py --dataset ECL --pred-lens 96 --ca off --seed 2025
python run_demo.py --dataset ECL --pred-lens 96 --ca bridge --seed 2025
```

## Saved Results and Checkpoints

Every run creates a unique subdirectory under `outputs/`:

| File | Content |
| --- | --- |
| `best_val.pt` | Validation-selected weights, architecture, and data metadata |
| `metrics.json` | Validation and test MSE/MAE |
| `config.json` | Resolved configuration and runtime versions |
| `data_manifest.json` | Dataset hash, columns, scaler statistics, and split details |
| `history.json` | Training and validation histories |
| `train.log` | Run output |

Evaluate a saved checkpoint with its original data and window settings:

```bash
python run_demo.py --dataset ETTm1 --seq-len 512 --pred-lens 96 --eval-only --checkpoint outputs/YOUR_RUN/best_val.pt
```

The model architecture is restored from checkpoint metadata with strict weight loading. Data and split mismatches are rejected.

Summarize completed training runs:

```bash
python scripts/summarize_results.py
```

This writes `outputs/summary/all_runs.csv` and `outputs/summary/mean_std.csv`. Different configurations and dataset snapshots are grouped separately; evaluation-only reruns are excluded. Sample standard deviation is undefined for a single seed.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wang2026idealtsf,
  title={IdealTSF: Can Non-Ideal Data Contribute to Enhancing the Performance of Time Series Forecasting Models?},
  author={Wang, Hua and Lu, Jinghao and Zhang, Fan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={31},
  pages={26224--26232},
  year={2026}
}
```

## Acknowledgments

We thank [ETDataset](https://github.com/zhouhaoyi/ETDataset) and [Time-Series-Library](https://github.com/thuml/Time-Series-Library) for providing benchmark data resources, and [TimeBridge](https://github.com/Hank0626/TimeBridge) for its work on cross-variable modeling.
