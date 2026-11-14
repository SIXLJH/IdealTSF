"""Array/Dataset compatibility and element-weighted streaming evaluation."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import LabeledDataset


def as_dataset(x, y=None):
    if isinstance(x, Dataset):
        if y is not None:
            raise ValueError('Pass y=None when x is already a Dataset.')
        if len(x) == 0:
            raise ValueError('Empty dataset.')
        return x
    if y is None or len(x) == 0 or len(x) != len(y):
        raise ValueError('Nonempty x and y with matching sample counts are required.')
    return LabeledDataset(x, y)


def infer_dimensions(x, y=None):
    if isinstance(x, Dataset):
        channels, seq_len, pred_len = x.num_channels, x.seq_len, x.pred_len
    else:
        if x.ndim != 3 or y is None or y.ndim != 2 or y.shape[1] % x.shape[1]:
            raise ValueError('Expected x [N,C,L], y [N,C*H].')
        channels, seq_len, pred_len = x.shape[1], x.shape[2], y.shape[1] // x.shape[1]
    return channels, seq_len, pred_len


def target_view(tensor, channels, target_index=None):
    if target_index is None:
        return tensor
    return tensor.reshape(tensor.shape[0], channels, -1)[:, target_index, :]


def supervised_loss(network, prediction, target, criterion, target_index=None):
    channels = network.revin.num_features
    return criterion(target_view(prediction, channels, target_index),
                     target_view(target, channels, target_index))


@torch.no_grad()
def evaluate_streaming(network, x, y=None, device='cpu', batch_size=256):
    dataset = as_dataset(x, y)
    target_index = getattr(dataset, 'target_index', None)
    channels = network.revin.num_features
    # Even an unshuffled DataLoader draws a base seed. Isolate it so validation
    # never changes the RNG stream used by the next training augmentation.
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=False, generator=generator)
    was_training = network.training
    network.eval()
    squared, absolute, count = 0.0, 0.0, 0
    try:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            prediction, _ = network(xb)
            difference = (
                target_view(prediction, channels, target_index)
                - target_view(yb, channels, target_index)
            ).double()
            squared += difference.square().sum().item()
            absolute += difference.abs().sum().item()
            count += difference.numel()
    finally:
        network.train(was_training)
    if not count:
        raise ValueError('Evaluation has no target elements.')
    return squared / count, absolute / count


def cpu_state_dict(network):
    return {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
