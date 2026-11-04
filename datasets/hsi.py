"""Full-band HSI datasets with lazy patch extraction for Dassl.

Splits are stratified by center-pixel class; neighboring patches can overlap.
RGB image transforms are intentionally bypassed for spectral tensors.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dassl.data.datasets import Datum, DatasetBase
from hsi_preprocessing import load_hsi_data, zero_pad_hsi, scale_hsi_to_unit_interval, resolve_mat_key
from dataset import flip_augmentation, radiation_augmentation
from .hsi_splits import read_saved_split, select_saved_split



class HSIPatchWrapper(Dataset):
    """Return the img/label interface used by FAP, keeping all spectral bands."""

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.items = data_source
        self.options = cfg.DATASET.HSI
        self.is_train = is_train
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0
        self.cube_pipeline = getattr(getattr(cfg, "TRAINER", None), "NAME", "") == "FAP"
        self.class_indices = {}
        for i, item in enumerate(data_source):
            self.class_indices.setdefault(item.label, []).append(i)

    def __len__(self):
        return len(self.items)

    def patch(self, item):
        row, col = item.location
        size = self.options.PATCH_SIZE
        patch = item.cube[row:row + size, col:col + size]
        return torch.from_numpy(patch.transpose(2, 0, 1).copy()).float()

    def __getitem__(self, index):
        item = self.items[index]
        original = self.patch(item)
        patch = original.unsqueeze(0)
        if self.is_train and not self.cube_pipeline:
            if self.options.FLIP:
                patch = flip_augmentation(patch)
            if self.options.RADIATION and torch.rand(()).item() < 0.5:
                patch = radiation_augmentation(patch)
            if self.options.MIXTURE and torch.rand(()).item() < 0.5:
                pool = self.class_indices[item.label]
                other = self.patch(self.items[pool[torch.randint(len(pool), ()).item()]]).unsqueeze(0)
                a, b = torch.rand(2)
                patch = (a * patch + b * other) / (a + b + 1e-6) + torch.randn_like(patch) / 25
        output = dict(img=patch.squeeze(0), label=item.label, domain=item.domain,
                      impath=item.impath, index=index, location=torch.tensor(item.location))
        if self.return_img0:
            output['img0'] = original
        return output


class HSIDataset(DatasetBase):
    dataset_wrapper = HSIPatchWrapper

    def __init__(self, cfg):
        options = cfg.DATASET.HSI
        folder = Path(cfg.DATASET.ROOT).expanduser().resolve() / self.dataset_dir
        hsi_path, gt_path = folder / options.HSI_FILE, folder / options.GT_FILE
        for path in (hsi_path, gt_path):
            if not path.is_file():
                raise FileNotFoundError(f'Missing HSI dataset file: {path}')
        split_path = Path(options.SPLIT_FILE.format(dataset=self.dataset_dir))
        if not split_path.is_absolute():
            split_path = folder / split_path
        if not split_path.is_file():
            raise FileNotFoundError(
                f'Saved dataset split missing: {split_path}. Run from FAP: '
                f'python split.py --dataset {self.dataset_dir.lower()} --root {cfg.DATASET.ROOT}')
        size, stride = options.PATCH_SIZE, options.STRIDE
        if size < 1 or size % 2 != 1 or stride < 1:
            raise ValueError('PATCH_SIZE must be positive and odd; STRIDE must be positive')
        tr, va = options.TRAIN_RATIO, options.VAL_RATIO
        if not (0 < tr < 1 and 0 < va < 1 and tr + va < 1):
            raise ValueError('Train and validation ratios must be positive and sum to less than 1')
        hsi, gt = load_hsi_data(str(hsi_path), str(gt_path),
                               resolve_mat_key(hsi_path, options.HSI_KEY, 3),
                               resolve_mat_key(gt_path, options.GT_KEY, 2))
        self.in_channels = hsi.shape[-1]
        self.raw_cube = torch.from_numpy(hsi.transpose(2, 0, 1).copy())
        self.raw_range = (float(hsi.min()), float(hsi.max()))
        hsi = scale_hsi_to_unit_interval(hsi)
        self.cube = torch.from_numpy(hsi.transpose(2, 0, 1).copy())
        cube = zero_pad_hsi(hsi, size // 2)
        names = list(options.CLASS_NAMES)
        saved = read_saved_split(split_path, gt, names, self.dataset_dir, stride,
                                 options.IGNORED_LABELS, tr, va)
        groups = select_saved_split(saved, subsample=cfg.DATASET.SUBSAMPLE_CLASSES,
                                    num_shots=cfg.DATASET.NUM_SHOTS, seed=cfg.SEED)
        print(f"Using saved split: {split_path} (partition seed={saved['metadata']['split_seed']}; "
              f"few-shot seed={max(cfg.SEED, 0)})")
        train, val, test = [], [], []
        for split, output in zip(('train', 'val', 'test'), (train, val, test)):
            for row, col, label, classname in groups[split]:
                item = Datum(impath=str(hsi_path), label=label, classname=classname)
                item.location = (row, col)
                item.cube = cube
                output.append(item)
        super().__init__(train_x=train, val=val, test=test)
