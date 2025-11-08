from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_transforms(mode: str, img_size: int = 224, randaugment: tuple | None = None, random_erasing_p: float = 0.0):
    """Return torchvision transforms for train/val/test.

    randaugment: optional tuple (num_ops, magnitude)
    random_erasing_p: probability for RandomErasing
    """
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if mode == 'train':
        ops: list = [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        ]
        if randaugment is not None:
            num_ops, magnitude = randaugment
            try:
                ops.append(transforms.RandAugment(num_ops=num_ops, magnitude=magnitude))
            except Exception:
                # older torchvision versions use different signature
                ops.append(transforms.RandAugment())

        ops.extend([
            transforms.ToTensor(),
            norm,
        ])

        if random_erasing_p and random_erasing_p > 0.0:
            ops.append(transforms.RandomErasing(p=random_erasing_p))

        return transforms.Compose(ops)
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            norm,
        ])


class DogBreedDataset(Dataset):
    def __init__(self, data_dir: str, split_csv: str, labels_csv: str, mode: str = 'train', transform: Optional[Callable] = None):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform or get_transforms(mode)

        # read labels mapping
        import pandas as pd

        labels_df = pd.read_csv(labels_csv, dtype={'id': str})
        breeds = sorted(labels_df['breed'].unique())
        self.class_to_idx = {b: i for i, b in enumerate(breeds)}
        self.idx_to_class = {i: b for b, i in self.class_to_idx.items()}
        self.num_classes = len(breeds)

        # read split file: expects CSV with image_path,label or id,label
        split_df = pd.read_csv(split_csv)
        rows = []
        for _, r in split_df.iterrows():
            path = r.get('image_path') or r.get('filepath') or r.get('id')
            label = r.get('label') if 'label' in r else None
            if path is None:
                continue
            # if path is id, convert to data/train/{id}.jpg
            p = Path(path)
            if not p.exists():
                # maybe it's an id
                if str(path).isdigit():
                    p = self.data_dir / 'train' / f"{int(path):04d}.jpg"
                else:
                    p = self.data_dir / path
            if not p.exists():
                # skip missing
                continue
            if label is None:
                # try infer from labels_df
                id_str = p.stem
                lab = labels_df.loc[labels_df['id'] == id_str, 'breed']
                if len(lab) == 0:
                    continue
                label = lab.iloc[0]
            rows.append((str(p.resolve()), int(self.class_to_idx[label])))

        self.samples = rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label
