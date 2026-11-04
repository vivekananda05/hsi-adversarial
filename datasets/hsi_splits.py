"""Persist fixed HSI partitions and draw seed-dependent few-shot subsets."""

import hashlib
import json
from pathlib import Path

import numpy as np



def split_class_locations(coords, train_ratio, val_ratio, num_shots, test_only=False):
    """Reserve a fixed test partition before few-shot or base/new filtering."""
    n = len(coords)
    if n < 3:
        raise ValueError('Each training class needs at least 3 labeled pixels')
    ntrain = min(max(1, int(n * train_ratio)), n - 2)
    nval = min(max(1, int(n * val_ratio)), n - ntrain - 1)
    train, val, test = coords[:ntrain], coords[ntrain:ntrain+nval], coords[ntrain+nval:]
    if test_only:
        # New-class evaluation uses only its existing held-out test partition.
        return coords[:0], coords[:0], test
    if num_shots > 0:
        train, val = train[:num_shots], val[:min(num_shots, 4)]
    return train, val, test


# Master partitions are persisted separately from seed-dependent few-shot draws.


def gt_fingerprint(gt):
    return hashlib.sha256(np.asarray(gt, dtype='<i8').tobytes(order='C')).hexdigest()


def build_saved_split(gt, class_names, dataset_name, train_ratio=.6, val_ratio=.2,
                      stride=1, ignored_labels=(0,), seed=42):
    if gt.ndim != 2 or stride < 1 or seed < 0:
        raise ValueError('Expected 2D ground truth, positive stride, and nonnegative split seed')
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError('Train/validation ratios must be positive and sum to less than one')
    invalid = set(np.unique(gt)) - set(ignored_labels) - set(range(1, len(class_names) + 1))
    if invalid:
        raise ValueError(f'Unexpected ground-truth labels: {invalid}')
    if set(ignored_labels) & set(range(1, len(class_names) + 1)):
        raise ValueError('Only background labels may be ignored with this class mapping')
    result = {
        'metadata': {
            'version': 1, 'dataset': dataset_name, 'class_names': list(class_names),
            'gt_shape': list(gt.shape), 'gt_sha256': gt_fingerprint(gt),
            'split_seed': int(seed), 'train_ratio': float(train_ratio),
            'val_ratio': float(val_ratio), 'stride': int(stride),
            'ignored_labels': sorted(int(x) for x in ignored_labels),
            'entry_format': ['row', 'column', 'label_zero_based', 'classname'],
        },
        'train': [], 'val': [], 'test': [],
    }
    rng = np.random.RandomState(seed)
    for label, classname in enumerate(class_names):
        coords = np.argwhere(gt[::stride, ::stride] == label + 1) * stride
        rng.shuffle(coords)
        groups = split_class_locations(coords, train_ratio, val_ratio, -1)
        for split, locations in zip(('train', 'val', 'test'), groups):
            result[split].extend([[int(row), int(col), label, classname] for row, col in locations])
    validate_saved_split(result, gt, class_names, dataset_name, stride, ignored_labels,
                         train_ratio, val_ratio)
    return result


def validate_saved_split(data, gt, class_names, dataset_name, stride=1,
                         ignored_labels=(0,), train_ratio=.6, val_ratio=.2):
    """Reject stale, overlapping, incomplete, or incompatible pixel partitions."""
    if not isinstance(data, dict):
        raise ValueError('Saved split must be a JSON object')
    meta = data.get('metadata', {})
    if not isinstance(meta, dict) or type(meta.get('split_seed')) is not int or meta['split_seed'] < 0:
        raise ValueError('Invalid saved split seed/metadata')
    expected = dict(version=1, dataset=dataset_name, class_names=list(class_names),
                    gt_shape=list(gt.shape), gt_sha256=gt_fingerprint(gt),
                    stride=int(stride), ignored_labels=sorted(int(x) for x in ignored_labels),
                    train_ratio=float(train_ratio), val_ratio=float(val_ratio))
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f'Saved split metadata mismatch: {key}. Regenerate with split.py --overwrite only if intentional.')
    seen = np.zeros(gt.shape, dtype=bool)
    counts = {split: np.zeros(len(class_names), dtype=int) for split in ('train', 'val', 'test')}
    for split in ('train', 'val', 'test'):
        entries = data.get(split)
        if not isinstance(entries, list):
            raise ValueError(f'Saved split is missing its {split} list')
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 4:
                raise ValueError(f'Invalid saved split entry: {entry}')
            row, col, label, classname = entry
            if any(type(value) is not int for value in (row, col, label)):
                raise ValueError('Saved locations and labels must be integers')
            if not (0 <= row < gt.shape[0] and 0 <= col < gt.shape[1] and 0 <= label < len(class_names)):
                raise ValueError('Saved location or class label is out of bounds')
            if row % stride or col % stride or gt[row, col] != label + 1 or classname != class_names[label]:
                raise ValueError('Saved location/class does not match the dataset')
            if gt[row, col] in ignored_labels or seen[row, col]:
                raise ValueError('Saved split contains an ignored or duplicate/overlapping pixel')
            seen[row, col] = True
            counts[split][label] += 1
    eligible = np.isin(gt, np.arange(1, len(class_names) + 1)) & ~np.isin(gt, ignored_labels)
    grid = np.zeros(gt.shape, dtype=bool)
    grid[::stride, ::stride] = True
    if not np.array_equal(seen, eligible & grid):
        raise ValueError('Saved partitions do not cover all eligible labeled pixels exactly once')
    if any((value == 0).any() for value in counts.values()):
        raise ValueError('Every class must have train, validation, and test pixels in the master split')
    for label in range(len(class_names)):
        total = sum(counts[key][label] for key in ('train', 'val', 'test'))
        expected_groups = split_class_locations(range(int(total)), train_ratio, val_ratio, -1)
        actual_counts = tuple(counts[key][label] for key in ('train', 'val', 'test'))
        if actual_counts != tuple(len(group) for group in expected_groups):
            raise ValueError('Saved partition counts do not match the configured split ratios')
    return data


def read_saved_split(path, gt, class_names, dataset_name, stride=1,
                     ignored_labels=(0,), train_ratio=.6, val_ratio=.2):
    with Path(path).open() as handle:
        data = json.load(handle)
    return validate_saved_split(data, gt, class_names, dataset_name, stride,
                                ignored_labels, train_ratio, val_ratio)


def write_saved_split(path, data, overwrite=False):
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f'Split already exists: {path}; use --overwrite only to deliberately replace it')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.',
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, indent=2)
            handle.write('\n')
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic no-overwrite publication, including concurrent generators.
            os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def select_saved_split(data, subsample='all', num_shots=16, seed=0):
    """Experiment seeds sample train/val only; test entries never change."""
    if subsample not in ('all', 'base', 'new'):
        raise ValueError(f'Unknown class subset: {subsample}')
    names = data['metadata']['class_names']
    midpoint = (len(names) + 1) // 2
    selected = list(range(len(names)))
    if subsample == 'base':
        selected = selected[:midpoint]
    elif subsample == 'new':
        selected = selected[midpoint:]
    relabeler = {label: index for index, label in enumerate(selected)}
    rng = np.random.RandomState(max(seed, 0))
    output = {'train': [], 'val': [], 'test': []}
    for split in ('train', 'val', 'test'):
        for label in range(len(names)):
            entries = [entry for entry in data[split] if entry[2] == label]
            if split != 'test' and num_shots > 0:
                limit = num_shots if split == 'train' else min(num_shots, 4)
                if len(entries) > limit:
                    indices = rng.choice(len(entries), size=limit, replace=False)
                    entries = [entries[index] for index in indices]
            if label not in relabeler or (subsample == 'new' and split != 'test'):
                continue
            output[split].extend([[row, col, relabeler[label], classname]
                                  for row, col, _, classname in entries])
    return output
