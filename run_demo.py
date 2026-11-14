"""Train/evaluate IdealTSF with chronological splits and lazy data windows."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import sys
import uuid

import numpy as np
import pandas as pd
import sklearn
import torch

from IdealTSF import IdealTSFArchitecture, Pretrainer, Trainer, TimeSeriesAugmentations
from utils.forecast_data import CHANNELS, canonical_dataset, load_benchmark
from utils.training_data import cpu_state_dict, evaluate_streaming


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_boolean_option(parser, flag, default=True):
    """Preserve --flag/--no-flag semantics on Python 3.8 and newer."""
    destination = flag[2:].replace('-', '_')
    parser.add_argument(flag, dest=destination, action='store_true')
    parser.add_argument('--no-' + flag[2:], dest=destination, action='store_false')
    parser.set_defaults(**{destination: default})


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', '--data', default='ETTh2', type=canonical_dataset)
    parser.add_argument('--data-root', default='dataset')
    parser.add_argument('--data-file', default=None)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--pred-lens', '--pred-len', type=int, nargs='+', default=[96])
    parser.add_argument('--features', choices=['M', 'S', 'MS'], default='M')
    parser.add_argument('--target', default='OT')
    add_boolean_option(parser, '--scale', default=True)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--missing-policy', choices=['error', 'ffill'], default='error')
    add_boolean_option(parser, '--strict-channels', default=False)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda:0, ...')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-threads', type=int, default=None, help='Optional CPU thread limit.')
    parser.add_argument('--pretrain-epochs', type=int, default=30)
    parser.add_argument('--finetune-epochs', type=int, default=50)
    parser.add_argument('--pretrain-batch-size', type=int, default=None)
    parser.add_argument('--finetune-batch-size', type=int, default=None)
    parser.add_argument('--eval-batch-size', type=int, default=None)
    parser.add_argument('--pretrain-lr', type=float, default=5e-4)
    parser.add_argument('--finetune-lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--rho', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default=0, help='0 keeps all fine-tuning epochs.')
    add_boolean_option(parser, '--revin', default=True)
    parser.add_argument('--hid-dim', '--hidden-dim', type=int, default=16)
    parser.add_argument('--ca', choices=['auto', 'off', 'bridge'], default='auto')
    parser.add_argument('--num-bridges', type=int, default=None)
    add_boolean_option(parser, '--stable-jump', default=True)
    parser.add_argument('--noise-std', type=float, default=0.01)
    parser.add_argument('--noise-scales', type=float, nargs='+', default=[0.1, 0.05, 0.02, 0.01])
    parser.add_argument('--erase-prob', type=float, default=0.3)
    parser.add_argument('--min-erase-length', type=int, default=4)
    parser.add_argument('--max-erase-length', type=int, default=300)
    parser.add_argument('--stable-jump-prob', type=float, default=0.05)
    parser.add_argument('--stable-alpha', type=float, default=1.5)
    parser.add_argument('--stable-beta', type=float, default=0.0,
                        help='The current sampler is symmetric: beta must be 0.')
    parser.add_argument('--stable-scale', type=float, default=0.005)
    parser.add_argument('--stable-loc', type=float, default=0.0)
    parser.add_argument('--stable-clip', type=float, default=0.05, help='0 disables clipping.')
    parser.add_argument('--stable-width', type=int, default=1)
    add_boolean_option(parser, '--positive-aux', default=True)
    parser.add_argument('--positive-loss-weight', type=float, default=0.2)
    parser.add_argument('--positive-window-size', type=int, default=5)
    parser.add_argument('--positive-mix-alpha', type=float, default=0.85)
    add_boolean_option(parser, '--positive-use-preprocess', default=False)
    parser.add_argument('--positive-z-thresh', type=float, default=2.5)
    add_boolean_option(parser, '--positive-use-iqr', default=True)
    parser.add_argument('--output-dir', default='outputs')
    add_boolean_option(parser, '--dry-run', default=False)
    add_boolean_option(parser, '--eval-only', default=False)
    parser.add_argument('--checkpoint', default=None)
    return parser


def runtime_info(device):
    return {
        'python': platform.python_version(), 'torch': str(torch.__version__),
        'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
        'cuda_runtime': torch.version.cuda, 'device': device,
        'device_name': torch.cuda.get_device_name(torch.device(device)) if device.startswith('cuda') else 'CPU',
    }


def resolve_settings(args):
    if args.device == 'auto':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu for a small check.')
    if args.num_threads is not None:
        if args.num_threads < 1:
            raise ValueError('num-threads must be positive.')
        torch.set_num_threads(args.num_threads)
    if args.ca == 'auto':
        args.ca = 'bridge' if args.features != 'S' and args.dataset in {'WTH', 'ECL', 'Traffic'} else 'off'
    if args.num_bridges is None:
        args.num_bridges = 4 if args.dataset == 'WTH' else 16
    batch = 8 if args.dataset == 'Traffic' else 32
    eval_batch = {'Traffic': 8, 'ECL': 32, 'WTH': 128}.get(args.dataset, 256)
    for name, default in [('pretrain_batch_size', batch), ('finetune_batch_size', batch), ('eval_batch_size', eval_batch)]:
        if getattr(args, name) is None:
            setattr(args, name, default)
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive.')
    if args.pretrain_epochs < 0 or args.finetune_epochs < 1:
        raise ValueError('pretrain-epochs >= 0 and finetune-epochs >= 1 are required.')
    if args.hid_dim < 1 or args.num_bridges < 1 or args.num_workers < 0 or args.patience < 0:
        raise ValueError('Invalid hid-dim, num-bridges, num-workers or patience.')
    if args.seq_len < 1 or any(h < 1 for h in args.pred_lens) or args.stride < 1:
        raise ValueError('Input length, horizons and stride must be positive.')
    if args.rho < 0 or args.pretrain_lr <= 0 or args.finetune_lr <= 0 or args.weight_decay < 0:
        raise ValueError('Invalid optimization parameters.')
    if not 0 <= args.positive_mix_alpha <= 1 or args.positive_loss_weight < 0:
        raise ValueError('Invalid positive-sample parameters.')
    if args.positive_window_size < 1:
        raise ValueError('positive-window-size must be positive.')
    if args.positive_z_thresh <= 0:
        raise ValueError('positive-z-thresh must be positive.')
    if args.positive_use_preprocess and args.positive_window_size % 2 == 0:
        raise ValueError('positive-window-size must be odd when positive-use-preprocess is enabled.')
    if not 0 <= args.erase_prob <= 1 or args.noise_std < 0 or any(s < 0 for s in args.noise_scales):
        raise ValueError('Invalid Gaussian noise or erase probability settings.')
    if args.min_erase_length < 1 or args.max_erase_length <= args.min_erase_length:
        raise ValueError('Erase lengths must satisfy 1 <= min < max.')
    if not 0 <= args.stable_jump_prob <= 1 or not 0.8 <= args.stable_alpha <= 1.95:
        raise ValueError('stable-jump-prob must be in [0,1], stable-alpha in [0.8,1.95].')
    if args.stable_beta != 0:
        raise ValueError('The current stable-jump sampler is symmetric and requires stable-beta=0.')
    if args.stable_scale < 0 or args.stable_clip < 0 or not 1 <= args.stable_width <= args.seq_len:
        raise ValueError('Invalid stable scale, clip or width (width must be <= seq-len).')
    if args.eval_only and not args.checkpoint:
        raise ValueError('--eval-only requires --checkpoint.')
    if args.checkpoint and not args.eval_only:
        raise ValueError('--checkpoint is used with --eval-only; it is not an implicit resume option.')


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_for_evaluation(args, bundle, pred_len):
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['model_config']
    expected = (bundle.train.num_channels, args.seq_len, pred_len)
    actual = (config['num_channels'], config['seq_len'], config['pred_horizon'])
    if actual != expected:
        raise ValueError(f'Checkpoint (C,L,H)={actual}; requested {expected}.')
    stored = checkpoint.get('data_metadata', {})
    for name in ['dataset', 'file_sha256', 'columns', 'scaled', 'features', 'target',
                 'missing_policy', 'scaler_mean', 'scaler_scale', 'splits']:
        if name in stored and stored[name] != bundle.metadata[name]:
            raise ValueError(f'Checkpoint/data mismatch: {name}. Use the original data and split options.')
    network = IdealTSFArchitecture(**config).to(args.device)
    network.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return network, checkpoint


def run_one_experiment(args, pred_len, seed):
    set_seed(seed)
    bundle = load_benchmark(
        args.dataset, args.seq_len, pred_len, data_root=args.data_root, data_file=args.data_file,
        features=args.features, target=args.target, scale=args.scale, stride=args.stride,
        missing_policy=args.missing_policy, strict_channels=args.strict_channels,
    )
    if args.dry_run:
        print(json.dumps(bundle.metadata, indent=2))
        return None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'{args.dataset}_L{args.seq_len}_H{pred_len}_seed{seed}_{stamp}_{uuid.uuid4().hex[:6]}'
    output = Path(args.output_dir) / run_id
    output.mkdir(parents=True, exist_ok=False)
    resolved = dict(vars(args), pred_len=pred_len, seed=seed)
    environment = runtime_info(args.device)
    write_json(output / 'config.json', {'settings': resolved, 'runtime': environment})
    write_json(output / 'data_manifest.json', bundle.metadata)
    with (output / 'train.log').open('w', encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print(f'IdealTSF | {args.dataset} | L={args.seq_len} H={pred_len} | seed={seed}')
            print(f'CA={args.ca}; stable_jump={args.stable_jump}; positive_aux={args.positive_aux}')
            print(f'Device={args.device}; data shape=({bundle.metadata["rows_used"]}, {bundle.train.num_channels})')
            print(f'Standardized metrics={args.scale}; validation MSE selects the model.')
            for name, split in bundle.metadata['splits'].items():
                print(f'{name}: target={split["target_interval"]}; windows={split["windows"]}')
            if bundle.train.num_channels != CHANNELS[args.dataset] and args.features != 'S':
                print(f'Note: {args.dataset} benchmark normally has {CHANNELS[args.dataset]} channels; this file has {bundle.train.num_channels}.')

            if args.eval_only:
                network, checkpoint = load_for_evaluation(args, bundle, pred_len)
                print(f'Restored checkpoint architecture: CA={network.ca_mode}, bridges={network.model_config["num_bridges"]}')
                pretrain_history, finetune_history = [], []
                best_epoch = checkpoint.get('best_epoch')
                checkpoint_path = Path(args.checkpoint)
            else:
                augmentations = TimeSeriesAugmentations(
                    noise_std=args.noise_std, erase_prob=args.erase_prob,
                    noise_scales=args.noise_scales,
                    min_erase_length=args.min_erase_length, max_erase_length=args.max_erase_length,
                    use_stable_jump=args.stable_jump, stable_jump_prob=args.stable_jump_prob,
                    stable_alpha=args.stable_alpha, stable_beta=args.stable_beta, stable_scale=args.stable_scale,
                    stable_loc=args.stable_loc, stable_clip=args.stable_clip, stable_width=args.stable_width,
                )
                pretrainer = Pretrainer(
                    device=args.device, num_epochs=args.pretrain_epochs,
                    batch_size=args.pretrain_batch_size, learning_rate=args.pretrain_lr,
                    weight_decay=args.weight_decay, rho=args.rho,
                    augmentations=augmentations, use_revin=args.revin,
                    ca_mode=args.ca, num_bridges=args.num_bridges, num_workers=args.num_workers,
                    hid_dim=args.hid_dim,
                )
                # Test data is deliberately not passed into either training stage.
                network = pretrainer.pretrain(bundle.train, x_val=bundle.val,
                                               eval_batch_size=args.eval_batch_size)
                trainer = Trainer(
                    pretrained_model=network, device=args.device, num_epochs=args.finetune_epochs,
                    batch_size=args.finetune_batch_size, learning_rate=args.finetune_lr,
                    weight_decay=args.weight_decay, rho=args.rho, use_revin=args.revin,
                    use_positive_aux=args.positive_aux, positive_loss_weight=args.positive_loss_weight,
                    positive_window_size=args.positive_window_size, positive_mix_alpha=args.positive_mix_alpha,
                    positive_use_preprocess=args.positive_use_preprocess,
                    positive_z_thresh=args.positive_z_thresh, positive_use_iqr=args.positive_use_iqr,
                    num_workers=args.num_workers, patience=args.patience,
                )
                checkpoint_path = output / 'best_val.pt'
                trainer.fine_tune(bundle.train, x_val=bundle.val,
                                  eval_batch_size=args.eval_batch_size, restore_best=True,
                                  save_best_path=str(checkpoint_path))
                if trainer.best_epoch is None or trainer.best_epoch < 1 or not np.isfinite(trainer.best_val_mse):
                    raise RuntimeError('No finite best-validation checkpoint was produced.')
                pretrain_history, finetune_history = pretrainer.history, trainer.history
                best_epoch = trainer.best_epoch

            val_mse, val_mae = evaluate_streaming(network, bundle.val, device=args.device,
                                                  batch_size=args.eval_batch_size)
            # Exactly one test pass, after restoring the validation-selected model.
            test_mse, test_mae = evaluate_streaming(network, bundle.test, device=args.device,
                                                    batch_size=args.eval_batch_size)
            metrics = {
                'dataset': args.dataset, 'seq_len': args.seq_len, 'pred_len': pred_len, 'seed': seed,
                'ca_mode': network.ca_mode, 'num_bridges': network.model_config['num_bridges'],
                'best_epoch': best_epoch, 'val_mse': val_mse, 'val_mae': val_mae,
                'test_mse': test_mse, 'test_mae': test_mae,
                'metrics_space': 'standardized' if args.scale else 'original',
                'features': args.features, 'evaluation_only': args.eval_only,
                'test_passes_in_this_run': 1, 'checkpoint': str(checkpoint_path),
            }
            if not args.eval_only:
                torch.save({
                    'model_state_dict': cpu_state_dict(network), 'model_config': network.model_config,
                    'best_epoch': best_epoch, 'best_val_mse': val_mse, 'best_val_mae': val_mae,
                    'data_metadata': bundle.metadata, 'run_config': resolved, 'runtime': environment,
                    'metrics': metrics,
                }, checkpoint_path)
            write_json(output / 'history.json', {'pretrain': pretrain_history, 'finetune': finetune_history})
            write_json(output / 'metrics.json', metrics)
            print('FINAL', json.dumps(metrics))
            print(f'Run files: {output}')
    return metrics


def main(argv=None):
    args = make_parser().parse_args(argv)
    resolve_settings(args)
    results = []
    for horizon in args.pred_lens:
        for seed in args.seeds or [args.seed]:
            result = run_one_experiment(args, horizon, seed)
            if result is not None:
                results.append(result)
    if results:
        print('\nResults from this invocation:')
        for result in results:
            print(f'{result["dataset"]} H={result["pred_len"]} seed={result["seed"]} '
                  f'CA={result["ca_mode"]} MSE={result["test_mse"]:.7f} MAE={result["test_mae"]:.7f}')
    return results


if __name__ == '__main__':
    main()
