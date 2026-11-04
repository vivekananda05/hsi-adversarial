#!/usr/bin/env python3
"""Save fixed HSI pixel partitions before CoOp/MaPLe-style few-shot sampling.

Examples:
  python split.py --dataset indian_pines --seed 42
  python split.py --all --root /home/vivek/HSI_DATA --seed 42
"""
import argparse
from pathlib import Path

import yaml

from datasets.hsi_splits import build_saved_split, read_saved_split, write_saved_split
from hsi_preprocessing import load_hsi_data, resolve_mat_key


CONFIG_DIR = Path(__file__).resolve().parent / 'configs/datasets'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--dataset', nargs='+', choices=sorted(p.stem for p in CONFIG_DIR.glob('*.yaml')))
    choice.add_argument('--all', action='store_true', help='Process every configured dataset; skip missing data files')
    parser.add_argument('--root', type=Path, help='Override DATASET.ROOT')
    parser.add_argument('--seed', type=int, default=42, help='Fixed master partition seed, independent of training seeds (default: 42)')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace existing master splits')
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    datasets = sorted(p.stem for p in CONFIG_DIR.glob('*.yaml')) if args.all else args.dataset
    for name in datasets:
        config = yaml.safe_load((CONFIG_DIR / (name+'.yaml')).read_text())['DATASET']
        options = config['HSI']
        root = args.root if args.root is not None else Path(config['ROOT'])
        folder = root.expanduser().resolve() / config['NAME']
        hsi_path, gt_path = folder / options['HSI_FILE'], folder / options['GT_FILE']
        missing = [str(path) for path in (hsi_path, gt_path) if not path.is_file()]
        if missing:
            message = f"{config['NAME']}: missing data: {', '.join(missing)}"
            if args.all:
                print('SKIP '+message, flush=True)
                continue
            parser.error(message)
        split_path = Path(options.get('SPLIT_FILE', 'split_{dataset}.json').format(dataset=config['NAME']))
        if not split_path.is_absolute():
            split_path = folder / split_path
        try:
            # Use exactly the loader's orientation and NaN/background handling.
            hsi, gt = load_hsi_data(str(hsi_path), str(gt_path),
                resolve_mat_key(hsi_path, options.get('HSI_KEY', ''), 3),
                resolve_mat_key(gt_path, options.get('GT_KEY', ''), 2))
            del hsi
            common = dict(stride=options.get('STRIDE', 1),
                          ignored_labels=options.get('IGNORED_LABELS', [0]),
                          train_ratio=options.get('TRAIN_RATIO', .6),
                          val_ratio=options.get('VAL_RATIO', .2))
            if split_path.exists() and not args.overwrite:
                data = read_saved_split(split_path, gt, options['CLASS_NAMES'], config['NAME'], **common)
                action = 'KEPT'
            else:
                data = build_saved_split(gt, options['CLASS_NAMES'], config['NAME'], seed=args.seed, **common)
                write_saved_split(split_path, data, overwrite=args.overwrite)
                action = 'SAVED'
        except (ValueError, OSError) as error:
            parser.error(str(error))
        counts = ', '.join(f'{split}={len(data[split])}' for split in ('train', 'val', 'test'))
        print(f"{action} {config['NAME']}: {split_path} | split seed={data['metadata']['split_seed']} | {counts}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
