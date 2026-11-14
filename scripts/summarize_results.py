"""Collect completed runs and summarize seeds without selecting by test error."""

import argparse
from pathlib import Path
import json

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--save-dir', default='outputs/summary')
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.output_dir).glob('*/metrics.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        if row.get('evaluation_only'):
            continue
        row['run_directory'] = str(path.parent)
        # Include optimization and augmentation settings to prevent mixing variants.
        settings = json.loads((path.parent / 'config.json').read_text(encoding='utf-8'))['settings']
        manifest = json.loads((path.parent / 'data_manifest.json').read_text(encoding='utf-8'))
        row['data_sha256'] = manifest['file_sha256']
        variant = {k: v for k, v in settings.items()
                   if k not in {'seed', 'seeds', 'pred_len', 'pred_lens', 'data_root', 'data_file',
                                'output_dir', 'device', 'num_workers', 'num_threads', 'dry_run',
                                'eval_only', 'checkpoint'}}
        row['configuration'] = json.dumps(variant, sort_keys=True)
        rows.append(row)
    if not rows:
        raise SystemExit('No completed training runs found.')
    frame = pd.DataFrame(rows)
    keys = ['dataset', 'data_sha256', 'seq_len', 'pred_len', 'ca_mode', 'metrics_space', 'features', 'configuration']
    duplicates = frame.duplicated(keys + ['seed'], keep=False)
    if duplicates.any():
        paths = frame.loc[duplicates, 'run_directory'].tolist()
        raise SystemExit('Repeated runs for the same configuration/seed. Keep the intended run before summarizing: ' + str(paths))
    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    frame.to_csv(save / 'all_runs.csv', index=False)
    grouped = frame.groupby(keys, dropna=False).agg(
        seeds=('seed', 'count'), mse_mean=('test_mse', 'mean'), mse_std=('test_mse', 'std'),
        mae_mean=('test_mae', 'mean'), mae_std=('test_mae', 'std'),
    ).reset_index()
    grouped.to_csv(save / 'mean_std.csv', index=False)
    print(grouped.drop(columns='configuration').to_string(index=False))
    print(f'Saved: {save}')


if __name__ == '__main__':
    main()
