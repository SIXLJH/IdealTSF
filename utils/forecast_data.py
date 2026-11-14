"""Chronological splits and lazy windows for the eight IdealTSF benchmarks."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


ALIASES = {
    'etth1': 'ETTh1', 'etth2': 'ETTh2', 'ettm1': 'ETTm1', 'ettm2': 'ETTm2',
    'wth': 'WTH', 'weather': 'WTH', 'ecl': 'ECL', 'electricity': 'ECL',
    'traffic': 'Traffic', 'exchange': 'Exchange', 'exchange_rate': 'Exchange',
}
CHANNELS = {'ETTh1': 7, 'ETTh2': 7, 'ETTm1': 7, 'ETTm2': 7,
            'WTH': 21, 'ECL': 321, 'Traffic': 862, 'Exchange': 8}
FILES = {
    'ETTh1': ['ETT-small/ETTh1.csv', 'ETTh1.csv'],
    'ETTh2': ['ETT-small/ETTh2.csv', 'ETTh2.csv'],
    'ETTm1': ['ETT-small/ETTm1.csv', 'ETTm1.csv'],
    'ETTm2': ['ETT-small/ETTm2.csv', 'ETTm2.csv'],
    'WTH': ['weather/weather.csv', 'weather.csv', 'WTH.csv'],
    'ECL': ['electricity/electricity.csv', 'electricity.csv', 'ECL.csv'],
    'Traffic': ['traffic/traffic.csv', 'traffic.csv', 'Traffic.csv'],
    'Exchange': ['exchange_rate/exchange_rate.csv', 'exchange_rate.csv', 'Exchange.csv'],
}


def canonical_dataset(name):
    try:
        return ALIASES[name.lower()]
    except KeyError:
        raise ValueError(f'Unknown dataset {name!r}; choose {list(CHANNELS)}.') from None


def resolve_data_file(dataset, data_root='dataset', data_file=None):
    dataset = canonical_dataset(dataset)
    if data_file is not None:
        path = Path(data_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = [Path(data_root).expanduser() / name for name in FILES[dataset]]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError('Dataset file not found. Tried: ' + ', '.join(map(str, candidates)))


def split_boundaries(dataset, length):
    """Return non-overlapping target boundaries, before adding lookback context."""
    dataset = canonical_dataset(dataset)
    if dataset.startswith('ETT'):
        factor = 4 if dataset.startswith('ETTm') else 1
        train_end, val_end, test_end = (months * 30 * 24 * factor for months in (12, 16, 20))
        if length < test_end:
            raise ValueError(f'{dataset} needs at least {test_end} rows; got {length}.')
    else:
        train_end = int(length * 0.7)
        val_end = length - int(length * 0.2)
        test_end = length
    return {'train': (0, train_end), 'val': (train_end, val_end), 'test': (val_end, test_end)}


class ForecastWindowDataset(Dataset):
    """Keep one [T,C] array; create [C,L] inputs only when a batch requests them."""

    def __init__(self, values, seq_len, pred_len, target_start, target_end,
                 stride=1, features='M', target_index=None, name=''):
        if seq_len < 1 or pred_len < 1 or stride < 1:
            raise ValueError('seq_len, pred_len and stride must be positive.')
        if values.ndim != 2 or values.dtype != np.float32:
            raise ValueError('values must be a float32 [time, channel] array.')
        self.values = values
        self.seq_len, self.pred_len, self.stride = seq_len, pred_len, stride
        self.num_channels = values.shape[1]
        self.features, self.name = features, name
        self.target_index = target_index if features == 'MS' else None
        self.target_start, self.target_end = target_start, target_end
        self.first_target = max(seq_len, target_start)
        if target_end > len(values) or target_start < 0 or target_end <= target_start:
            raise ValueError('Invalid target interval.')
        self.n_windows = (target_end - self.first_target - pred_len) // stride + 1
        if self.n_windows <= 0:
            raise ValueError(f'{name}: no complete windows for L={seq_len}, H={pred_len}.')

    def __len__(self):
        return self.n_windows

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        cut = self.first_target + index * self.stride
        # Copy only this window. Augmentation can never modify the backing data.
        x = np.array(self.values[cut - self.seq_len:cut].T, copy=True, order='C')
        y = np.array(self.values[cut:cut + self.pred_len].T, copy=True, order='C').reshape(-1)
        return torch.from_numpy(x), torch.from_numpy(y)

    def summary(self):
        return {
            'name': self.name, 'windows': len(self), 'seq_len': self.seq_len,
            'pred_len': self.pred_len, 'channels': self.num_channels,
            'target_interval': [self.target_start, self.target_end],
            'first_forecast_start': self.first_target,
            'last_forecast_end_exclusive': self.first_target + (len(self) - 1) * self.stride + self.pred_len,
            'input_interval': [self.first_target - self.seq_len,
                               self.first_target + (len(self) - 1) * self.stride],
            'stride': self.stride, 'features': self.features,
            'metric_target_index': self.target_index,
        }


@dataclass
class DataBundle:
    train: ForecastWindowDataset
    val: ForecastWindowDataset
    test: ForecastWindowDataset
    scaler: StandardScaler
    metadata: dict


def load_benchmark(dataset, seq_len, pred_len, data_root='dataset', data_file=None,
                   features='M', target='OT', scale=True, stride=1,
                   missing_policy='error', strict_channels=False):
    dataset = canonical_dataset(dataset)
    if features not in {'M', 'S', 'MS'}:
        raise ValueError('features must be M, S or MS.')
    if missing_policy not in {'error', 'ffill'}:
        raise ValueError('missing_policy must be error or ffill.')
    path = resolve_data_file(dataset, data_root, data_file)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError('Empty dataset.')
    dates = [c for c in df if c.lower() in {'date', 'datetime', 'timestamp', 'time'}]
    if dates:
        stamps = pd.to_datetime(df[dates[0]], errors='raise')
        if stamps.isna().any() or not stamps.is_monotonic_increasing or stamps.duplicated().any():
            raise ValueError('Timestamps must be unique and in chronological order.')
    columns = [c for c in df if c not in dates and not c.lower().startswith('unnamed:')]
    if features in {'S', 'MS'} and target not in columns:
        raise ValueError(f'Target {target!r} is absent; columns: {columns[:15]}.')
    # Match the uploaded ETTh reader: move the named target to the last column.
    if target in columns:
        columns = [c for c in columns if c != target] + [target]
    if features == 'S':
        columns = [target]
    if not columns:
        raise ValueError('No numeric variables found.')
    frame = df[columns].apply(pd.to_numeric, errors='raise')
    bounds = split_boundaries(dataset, len(frame))
    used_end = bounds['test'][1]
    frame = frame.iloc[:used_end].copy()
    missing_count = int(frame.isna().sum().sum())
    if missing_policy == 'ffill':
        frame = frame.ffill()  # Past observations only; no backward fill.
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Missing/nonfinite values found. ffill cannot fill leading NaNs; no future backfill is used.')
    expected = CHANNELS[dataset]
    if strict_channels and features != 'S' and len(columns) != expected:
        raise ValueError(f'{dataset}: expected {expected} channels, got {len(columns)}.')
    raw = frame.to_numpy(dtype=np.float64)
    train_end = bounds['train'][1]
    scaler = StandardScaler()
    scaler.fit(raw[:train_end])
    values = np.asarray(scaler.transform(raw) if scale else raw, dtype=np.float32, order='C')
    target_index = columns.index(target) if target in columns else None
    splits = {
        flag: ForecastWindowDataset(values, seq_len, pred_len, start, end, stride,
                                     features, target_index, flag)
        for flag, (start, end) in bounds.items()
    }
    metadata = {
        'dataset': dataset, 'file_name': path.name, 'file_sha256': digest.hexdigest(),
        'rows_in_file': len(df), 'rows_used': used_end, 'columns': columns,
        'channels': len(columns), 'expected_benchmark_channels': expected,
        'scaled': scale, 'scaler_fit_interval': [0, train_end],
        'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
        'features': features, 'target': target, 'missing_policy': missing_policy,
        'missing_values_before_fill': missing_count,
        'splits': {flag: item.summary() for flag, item in splits.items()},
        'backing_array_bytes': values.nbytes,
    }
    return DataBundle(splits['train'], splits['val'], splits['test'], scaler, metadata)
