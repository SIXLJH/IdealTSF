@echo off
setlocal

REM Edit the configuration block below. Boolean switches: 1=on, 0=off.
REM Extra command-line arguments override the settings below.

REM Interpreter (empty = current environment)
REM Empty: IDEALTSF_PYTHON, then PYTHON, then python; or set an executable path.
set "PYTHON_BIN="

REM Dataset and windows
REM This script configures this dataset only.
set "DATASET=ETTh2"
REM Relative paths are resolved from the project root.
set "DATA_ROOT=dataset"
REM Empty = automatic CSV lookup; set a full path to override.
set "DATA_FILE="
REM Historical input length.
set "SEQ_LEN=512"
REM Use only 96 for a first run.
set "PRED_LENS=96 192 336 720"
REM M, S, or MS.
set "FEATURES=M"
REM Required target column for S/MS.
set "TARGET=OT"
REM 1 = train-fitted standardization; 0 = original scale.
set "SCALE=1"
REM Sliding-window step.
set "STRIDE=1"
REM error or ffill (past values only).
set "MISSING_POLICY=error"
REM 1 = require the standard benchmark channel count.
set "STRICT_CHANNELS=0"

REM Device, seeds and loading
REM auto, cpu, cuda:0, etc.
set "DEVICE=auto"
REM Used when SEEDS is empty.
set "SEED=2025"
REM Optional space-separated list, e.g. 2025 2026 2027; takes precedence over SEED.
set "SEEDS="
REM DataLoader workers.
set "NUM_WORKERS=0"
REM Empty = keep the PyTorch CPU-thread default.
set "NUM_THREADS="

REM Model and channel bridge
REM Query/key feature dimension.
set "HID_DIM=16"
REM 1 = enable RevIN; 0 = disable.
set "REVIN=1"
REM off = original dense attention; bridge = lightweight CA; auto = dataset/feature rule.
set "CA_MODE=off"
REM Active only in bridge mode.
set "NUM_BRIDGES=16"

REM Two-stage training and optimization
REM 0 skips pretraining updates.
set "PRETRAIN_EPOCHS=30"
REM Maximum fine-tuning epochs.
set "FINETUNE_EPOCHS=50"
REM Pretraining batch size.
set "PRETRAIN_BATCH_SIZE=32"
REM Fine-tuning batch size.
set "FINETUNE_BATCH_SIZE=32"
REM Evaluation includes the final partial batch.
set "EVAL_BATCH_SIZE=256"
REM Pretraining learning rate.
set "PRETRAIN_LR=5e-4"
REM Fine-tuning learning rate.
set "FINETUNE_LR=5e-5"
REM AdamW weight decay inside ECOS.
set "WEIGHT_DECAY=1e-4"
REM ECOS perturbation radius.
set "RHO=0.9"
REM 0 = no early stopping; selection still uses validation MSE.
set "PATIENCE=0"

REM Negative-sample pretraining augmentation
REM Gaussian base noise standard deviation.
set "NOISE_STD=0.01"
REM One scale per temporal smoothing level.
set "NOISE_SCALES=0.1 0.05 0.02 0.01"
REM Probability of structured erasing.
set "ERASE_PROB=0.3"
REM Must be >= 1 and smaller than MAX_ERASE_LENGTH.
set "MIN_ERASE_LENGTH=4"
REM Actual erasing is also bounded by SEQ_LEN/4.
set "MAX_ERASE_LENGTH=300"
REM 1 = enable stable jumps in pretraining only.
set "STABLE_JUMP=1"
REM Per-sample jump probability.
set "STABLE_JUMP_PROB=0.05"
REM Supported interval: 0.8 to 1.95.
set "STABLE_ALPHA=1.5"
REM Current sampler is symmetric: keep 0; nonzero is rejected.
set "STABLE_BETA=0.0"
REM Stable-jump magnitude scale.
set "STABLE_SCALE=0.005"
REM Stable-jump location shift.
set "STABLE_LOC=0.0"
REM 0 = no clipping; positive = amplitude bound.
set "STABLE_CLIP=0.05"
REM Jump width; 1 <= width <= SEQ_LEN.
set "STABLE_WIDTH=1"

REM Positive-sample fine-tuning augmentation
REM 1 = enable positive auxiliary loss.
set "POSITIVE_AUX=1"
REM Auxiliary loss coefficient.
set "POSITIVE_LOSS_WEIGHT=0.2"
REM Smoothing window; an odd size is recommended.
set "POSITIVE_WINDOW_SIZE=5"
REM Original-input fraction in the positive sample.
set "POSITIVE_MIX_ALPHA=0.85"
REM 0 = moving average; 1 = outlier cleanup + smoothing (requires odd window).
set "POSITIVE_USE_PREPROCESS=0"
REM Used only when POSITIVE_USE_PREPROCESS=1.
set "POSITIVE_Z_THRESH=2.5"
REM IQR outlier detection; active only with preprocessing.
set "POSITIVE_USE_IQR=1"

REM Outputs and execution mode
REM Every run creates a separate subdirectory.
set "OUTPUT_DIR=outputs"
REM 1 = validate CSV/splits without training.
set "DRY_RUN=0"
REM 1 = evaluate CHECKPOINT; match its single horizon and data settings.
set "EVAL_ONLY=0"
REM Empty for training; set a path when EVAL_ONLY=1.
set "CHECKPOINT="

REM Build and run the command using the configuration above.
if defined PYTHON_BIN goto interpreter_ready
set "PYTHON_BIN=python"
if defined PYTHON set "PYTHON_BIN=%PYTHON%"
if defined IDEALTSF_PYTHON set "PYTHON_BIN=%IDEALTSF_PYTHON%"
:interpreter_ready

set "DATA_FILE_ARG="
if defined DATA_FILE set "DATA_FILE_ARG=--data-file "%DATA_FILE%""
call :set_boolean "%SCALE%" "--scale" "--no-scale" SCALE_ARG
if errorlevel 1 exit /b 2
call :set_boolean "%STRICT_CHANNELS%" "--strict-channels" "--no-strict-channels" STRICT_CHANNELS_ARG
if errorlevel 1 exit /b 2
set "SEEDS_ARG="
if defined SEEDS set "SEEDS_ARG=--seeds %SEEDS%"
set "NUM_THREADS_ARG="
if defined NUM_THREADS set "NUM_THREADS_ARG=--num-threads %NUM_THREADS%"
call :set_boolean "%REVIN%" "--revin" "--no-revin" REVIN_ARG
if errorlevel 1 exit /b 2
call :set_boolean "%STABLE_JUMP%" "--stable-jump" "--no-stable-jump" STABLE_JUMP_ARG
if errorlevel 1 exit /b 2
call :set_boolean "%POSITIVE_AUX%" "--positive-aux" "--no-positive-aux" POSITIVE_AUX_ARG
if errorlevel 1 exit /b 2
call :set_boolean "%POSITIVE_USE_PREPROCESS%" "--positive-use-preprocess" "--no-positive-use-preprocess" POSITIVE_USE_PREPROCESS_ARG
if errorlevel 1 exit /b 2
call :set_boolean "%POSITIVE_USE_IQR%" "--positive-use-iqr" "--no-positive-use-iqr" POSITIVE_USE_IQR_ARG
if errorlevel 1 exit /b 2
call :set_boolean "%DRY_RUN%" "--dry-run" "--no-dry-run" DRY_RUN_ARG
if errorlevel 1 exit /b 2
call :set_boolean "%EVAL_ONLY%" "--eval-only" "--no-eval-only" EVAL_ONLY_ARG
if errorlevel 1 exit /b 2
set "CHECKPOINT_ARG="
if defined CHECKPOINT set "CHECKPOINT_ARG=--checkpoint "%CHECKPOINT%""

pushd "%~dp0.."
if errorlevel 1 exit /b 1
"%PYTHON_BIN%" run_demo.py ^
    --dataset "%DATASET%" ^
    --data-root "%DATA_ROOT%" ^
    %DATA_FILE_ARG% ^
    --seq-len %SEQ_LEN% ^
    --pred-lens %PRED_LENS% ^
    --features "%FEATURES%" ^
    --target "%TARGET%" ^
    %SCALE_ARG% ^
    --stride %STRIDE% ^
    --missing-policy "%MISSING_POLICY%" ^
    %STRICT_CHANNELS_ARG% ^
    --device "%DEVICE%" ^
    --seed %SEED% ^
    %SEEDS_ARG% ^
    --num-workers %NUM_WORKERS% ^
    %NUM_THREADS_ARG% ^
    --hid-dim %HID_DIM% ^
    %REVIN_ARG% ^
    --ca "%CA_MODE%" ^
    --num-bridges %NUM_BRIDGES% ^
    --pretrain-epochs %PRETRAIN_EPOCHS% ^
    --finetune-epochs %FINETUNE_EPOCHS% ^
    --pretrain-batch-size %PRETRAIN_BATCH_SIZE% ^
    --finetune-batch-size %FINETUNE_BATCH_SIZE% ^
    --eval-batch-size %EVAL_BATCH_SIZE% ^
    --pretrain-lr %PRETRAIN_LR% ^
    --finetune-lr %FINETUNE_LR% ^
    --weight-decay %WEIGHT_DECAY% ^
    --rho %RHO% ^
    --patience %PATIENCE% ^
    --noise-std %NOISE_STD% ^
    --noise-scales %NOISE_SCALES% ^
    --erase-prob %ERASE_PROB% ^
    --min-erase-length %MIN_ERASE_LENGTH% ^
    --max-erase-length %MAX_ERASE_LENGTH% ^
    %STABLE_JUMP_ARG% ^
    --stable-jump-prob %STABLE_JUMP_PROB% ^
    --stable-alpha %STABLE_ALPHA% ^
    --stable-beta %STABLE_BETA% ^
    --stable-scale %STABLE_SCALE% ^
    --stable-loc %STABLE_LOC% ^
    --stable-clip %STABLE_CLIP% ^
    --stable-width %STABLE_WIDTH% ^
    %POSITIVE_AUX_ARG% ^
    --positive-loss-weight %POSITIVE_LOSS_WEIGHT% ^
    --positive-window-size %POSITIVE_WINDOW_SIZE% ^
    --positive-mix-alpha %POSITIVE_MIX_ALPHA% ^
    %POSITIVE_USE_PREPROCESS_ARG% ^
    --positive-z-thresh %POSITIVE_Z_THRESH% ^
    %POSITIVE_USE_IQR_ARG% ^
    --output-dir "%OUTPUT_DIR%" ^
    %DRY_RUN_ARG% ^
    %EVAL_ONLY_ARG% ^
    %CHECKPOINT_ARG% ^
    %*
set "run_exit=%ERRORLEVEL%"
popd
exit /b %run_exit%

:set_boolean
if "%~1"=="1" goto boolean_on
if "%~1"=="0" goto boolean_off
echo Invalid boolean %~2=%~1; use 0 or 1. 1>&2
exit /b 2
:boolean_on
set "%~4=%~2"
exit /b 0
:boolean_off
set "%~4=%~3"
exit /b 0
