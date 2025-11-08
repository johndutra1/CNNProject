from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .datasets import DogBreedDataset


def compute_class_weights(train_csv: str, labels_csv: str):
    df = pd.read_csv(train_csv)
    counts = df['label'].value_counts().sort_index()
    # if label is string, convert to counts per breed
    if counts.empty:
        # fallback: read labels.csv
        labs = pd.read_csv(labels_csv)
        counts = labs['breed'].value_counts()
    # make weights array
    inv = 1.0 / counts.values
    weights = inv / np.sum(inv) * len(counts)
    return weights, counts


def get_dataloaders(
    data_dir: str,
    splits: Dict[str, str],
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = False,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    img_size: int = 224,
    randaugment: Optional[Tuple[int, int]] = None,
    random_erasing_p: float = 0.0,
    sampler_strategy: str = 'weighted',
):
    data_dir_p = Path(data_dir)
    # splits is a dict with keys: train,val,test pointing to csv files
    from .datasets import get_transforms
    train_tf = get_transforms('train', img_size=img_size, randaugment=randaugment, random_erasing_p=random_erasing_p)
    val_tf = get_transforms('val', img_size=img_size)

    train_ds = DogBreedDataset(data_dir, splits['train'], str(data_dir_p / 'labels.csv'), mode='train', transform=train_tf)
    val_ds = DogBreedDataset(data_dir, splits['val'], str(data_dir_p / 'labels.csv'), mode='val', transform=val_tf)
    test_ds = DogBreedDataset(data_dir, splits['test'], str(data_dir_p / 'labels.csv'), mode='test', transform=val_tf)

    # apply optional truncation for smoke-runs
    from torch.utils.data import Subset
    if limit_train is not None and limit_train > 0:
        limit_train = min(limit_train, len(train_ds))
        train_ds = Subset(train_ds, list(range(limit_train)))
    if limit_val is not None and limit_val > 0:
        limit_val = min(limit_val, len(val_ds))
        val_ds = Subset(val_ds, list(range(limit_val)))

    # compute class frequencies from train_ds
    # Note: if train_ds is a Subset, access underlying dataset
    base_train_ds = getattr(train_ds, 'dataset', train_ds)
    labels = [lab for _, lab in base_train_ds]
    counts = np.bincount(labels, minlength=base_train_ds.num_classes)
    # class weight for loss (inverse frequency)
    class_weights = 1.0 / (counts + 1e-12)

    sampler = None
    if sampler_strategy == 'weighted':
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    if sampler is not None:
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=False)

    meta = {
        'num_classes': base_train_ds.num_classes,
        'class_to_idx': base_train_ds.class_to_idx,
        'idx_to_class': base_train_ds.idx_to_class,
        'train_counts': counts,
        'class_weights': class_weights,
    }
    return train_loader, val_loader, test_loader, meta
