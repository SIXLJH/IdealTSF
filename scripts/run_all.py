"""Run the eight benchmarks; stop immediately if any experiment fails."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'WTH', 'ECL', 'Traffic', 'Exchange']


def dataset_name(value):
    names = {name.lower(): name for name in DATASETS}
    names.update({'weather': 'WTH', 'electricity': 'ECL', 'exchange_rate': 'Exchange'})
    try:
        return names[value.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError('Unknown dataset: ' + value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', type=dataset_name, default=DATASETS)
    parser.add_argument('--pred-lens', type=int, nargs='+', default=None,
                        help='Optional override; omitted = use each dataset script.')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Optional override; omitted = use each dataset script.')
    args, extra = parser.parse_known_args(argv)
    root = Path(__file__).resolve().parents[1]
    forwarded = []
    if args.pred_lens is not None:
        forwarded += ['--pred-lens', *map(str, args.pred_lens)]
    if args.seeds is not None:
        forwarded += ['--seeds', *map(str, args.seeds)]
    forwarded += extra
    environment = dict(os.environ, IDEALTSF_PYTHON=sys.executable)
    for dataset in args.datasets:
        script = root / 'scripts' / (dataset + ('.bat' if os.name == 'nt' else '.sh'))
        if os.name == 'nt':
            # Batch files are executed by cmd; preserve quoted paths with spaces.
            command = subprocess.list2cmdline([str(script), *forwarded])
            print('RUN', command, flush=True)
            subprocess.run(command, shell=True, cwd=root, env=environment, check=True)
        else:
            command = ['bash', str(script), *forwarded]
            print('RUN', ' '.join(command), flush=True)
            subprocess.run(command, cwd=root, env=environment, check=True)


if __name__ == '__main__':
    main()
