#!/usr/bin/env bash
set -euo pipefail

# Edit the configuration block below. Boolean switches: 1=on, 0=off.
# CLI arguments supplied after this script override these settings.

# Interpreter (empty = current environment)
# Empty: IDEALTSF_PYTHON, then PYTHON, then python; or set an executable path.
PYTHON_BIN=""

# Dataset and windows
# This script configures this dataset only.
DATASET="ETTh2"
# Relative paths are resolved from the project root.
DATA_ROOT="dataset"
# Empty = automatic CSV lookup; set a full path to override.
DATA_FILE=""
# Historical input length.
SEQ_LEN=512
# Use only 96 for a first run.
PRED_LENS=(96 192 336 720)
# M, S, or MS.
FEATURES="M"
# Required target column for S/MS.
TARGET="OT"
# 1 = train-fitted standardization; 0 = original scale.
SCALE=1
# Sliding-window step.
STRIDE=1
# error or ffill (past values only).
MISSING_POLICY="error"
# 1 = require the standard benchmark channel count.
STRICT_CHANNELS=0

# Device, seeds and loading
# auto, cpu, cuda:0, etc.
DEVICE="auto"
# Used when SEEDS is empty.
SEED=2025
# Optional space-separated list, e.g. 2025 2026 2027; takes precedence over SEED.
SEEDS=""
# DataLoader workers.
NUM_WORKERS=0
# Empty = keep the PyTorch CPU-thread default.
NUM_THREADS=""

# Model and channel bridge
# Query/key feature dimension.
HID_DIM=16
# 1 = enable RevIN; 0 = disable.
REVIN=1
# off = original dense attention; bridge = lightweight CA; auto = dataset/feature rule.
CA_MODE="off"
# Active only in bridge mode.
NUM_BRIDGES=16

# Two-stage training and optimization
# 0 skips pretraining updates.
PRETRAIN_EPOCHS=30
# Maximum fine-tuning epochs.
FINETUNE_EPOCHS=50
# Pretraining batch size.
PRETRAIN_BATCH_SIZE=32
# Fine-tuning batch size.
FINETUNE_BATCH_SIZE=32
# Evaluation includes the final partial batch.
EVAL_BATCH_SIZE=256
# Pretraining learning rate.
PRETRAIN_LR=5e-4
# Fine-tuning learning rate.
FINETUNE_LR=5e-5
# AdamW weight decay inside ECOS.
WEIGHT_DECAY=1e-4
# ECOS perturbation radius.
RHO=0.9
# 0 = no early stopping; selection still uses validation MSE.
PATIENCE=0

# Negative-sample pretraining augmentation
# Gaussian base noise standard deviation.
NOISE_STD=0.01
# One scale per temporal smoothing level.
NOISE_SCALES=(0.1 0.05 0.02 0.01)
# Probability of structured erasing.
ERASE_PROB=0.3
# Must be >= 1 and smaller than MAX_ERASE_LENGTH.
MIN_ERASE_LENGTH=4
# Actual erasing is also bounded by SEQ_LEN/4.
MAX_ERASE_LENGTH=300
# 1 = enable stable jumps in pretraining only.
STABLE_JUMP=1
# Per-sample jump probability.
STABLE_JUMP_PROB=0.05
# Supported interval: 0.8 to 1.95.
STABLE_ALPHA=1.5
# Current sampler is symmetric: keep 0; nonzero is rejected.
STABLE_BETA=0.0
# Stable-jump magnitude scale.
STABLE_SCALE=0.005
# Stable-jump location shift.
STABLE_LOC=0.0
# 0 = no clipping; positive = amplitude bound.
STABLE_CLIP=0.05
# Jump width; 1 <= width <= SEQ_LEN.
STABLE_WIDTH=1

# Positive-sample fine-tuning augmentation
# 1 = enable positive auxiliary loss.
POSITIVE_AUX=1
# Auxiliary loss coefficient.
POSITIVE_LOSS_WEIGHT=0.2
# Smoothing window; an odd size is recommended.
POSITIVE_WINDOW_SIZE=5
# Original-input fraction in the positive sample.
POSITIVE_MIX_ALPHA=0.85
# 0 = moving average; 1 = outlier cleanup + smoothing (requires odd window).
POSITIVE_USE_PREPROCESS=0
# Used only when POSITIVE_USE_PREPROCESS=1.
POSITIVE_Z_THRESH=2.5
# IQR outlier detection; active only with preprocessing.
POSITIVE_USE_IQR=1

# Outputs and execution mode
# Every run creates a separate subdirectory.
OUTPUT_DIR="outputs"
# 1 = validate CSV/splits without training.
DRY_RUN=0
# 1 = evaluate CHECKPOINT; match its single horizon and data settings.
EVAL_ONLY=0
# Empty for training; set a path when EVAL_ONLY=1.
CHECKPOINT=""

# Build and run the command using the configuration above.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ -z "$PYTHON_BIN" ]]; then
    PYTHON_BIN="${IDEALTSF_PYTHON:-${PYTHON:-python}}"
fi

args=(
    run_demo.py
    --dataset "$DATASET"
    --data-root "$DATA_ROOT"
    --seq-len "$SEQ_LEN"
    --pred-lens "${PRED_LENS[@]}"
    --features "$FEATURES"
    --target "$TARGET"
    --stride "$STRIDE"
    --missing-policy "$MISSING_POLICY"
    --device "$DEVICE"
    --seed "$SEED"
    --num-workers "$NUM_WORKERS"
    --hid-dim "$HID_DIM"
    --ca "$CA_MODE"
    --num-bridges "$NUM_BRIDGES"
    --pretrain-epochs "$PRETRAIN_EPOCHS"
    --finetune-epochs "$FINETUNE_EPOCHS"
    --pretrain-batch-size "$PRETRAIN_BATCH_SIZE"
    --finetune-batch-size "$FINETUNE_BATCH_SIZE"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --pretrain-lr "$PRETRAIN_LR"
    --finetune-lr "$FINETUNE_LR"
    --weight-decay "$WEIGHT_DECAY"
    --rho "$RHO"
    --patience "$PATIENCE"
    --noise-std "$NOISE_STD"
    --noise-scales "${NOISE_SCALES[@]}"
    --erase-prob "$ERASE_PROB"
    --min-erase-length "$MIN_ERASE_LENGTH"
    --max-erase-length "$MAX_ERASE_LENGTH"
    --stable-jump-prob "$STABLE_JUMP_PROB"
    --stable-alpha "$STABLE_ALPHA"
    --stable-beta "$STABLE_BETA"
    --stable-scale "$STABLE_SCALE"
    --stable-loc "$STABLE_LOC"
    --stable-clip "$STABLE_CLIP"
    --stable-width "$STABLE_WIDTH"
    --positive-loss-weight "$POSITIVE_LOSS_WEIGHT"
    --positive-window-size "$POSITIVE_WINDOW_SIZE"
    --positive-mix-alpha "$POSITIVE_MIX_ALPHA"
    --positive-z-thresh "$POSITIVE_Z_THRESH"
    --output-dir "$OUTPUT_DIR"
)

append_boolean() {
    case "$2" in
        1) args+=("$1") ;;
        0) args+=("--no-${1#--}") ;;
        *) printf "Invalid boolean %s=%s; use 0 or 1.\n" "$1" "$2" >&2; exit 2 ;;
    esac
}

if [[ -n "$DATA_FILE" ]]; then
    args+=(--data-file "$DATA_FILE")
fi
append_boolean --scale "$SCALE"
append_boolean --strict-channels "$STRICT_CHANNELS"
if [[ -n "$SEEDS" ]]; then
    read -r -a configured_seeds <<< "$SEEDS"
    args+=(--seeds "${configured_seeds[@]}")
fi
if [[ -n "$NUM_THREADS" ]]; then
    args+=(--num-threads "$NUM_THREADS")
fi
append_boolean --revin "$REVIN"
append_boolean --stable-jump "$STABLE_JUMP"
append_boolean --positive-aux "$POSITIVE_AUX"
append_boolean --positive-use-preprocess "$POSITIVE_USE_PREPROCESS"
append_boolean --positive-use-iqr "$POSITIVE_USE_IQR"
append_boolean --dry-run "$DRY_RUN"
append_boolean --eval-only "$EVAL_ONLY"
if [[ -n "$CHECKPOINT" ]]; then
    args+=(--checkpoint "$CHECKPOINT")
fi

"$PYTHON_BIN" "${args[@]}" "$@"
