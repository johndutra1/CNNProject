"""Run inference on data/test/ and write Kaggle-format submission CSV (id,breed)."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import torch

from .models import get_model
from .datasets import get_transforms
from .utils import get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default=None)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    ck = Path(args.checkpoint)
    out_path = Path(args.out)
    device = get_device() if args.device is None else torch.device(args.device)

    # load class mapping from checkpoint folder if available
    class_map_file = ck.parent / 'class_to_idx.json'
    if class_map_file.exists():
        with open(class_map_file, 'r') as fh:
            class_to_idx = json.load(fh)
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    else:
        # fallback: read labels.csv
        import pandas as pd
        labs = pd.read_csv(data_dir / 'labels.csv', dtype={'id': str})
        breeds = sorted(labs['breed'].unique())
        idx_to_class = {i: b for i, b in enumerate(breeds)}

    num_classes = len(idx_to_class)
    model = get_model('resnet50', num_classes)
    from .utils import load_checkpoint
    ckpt = load_checkpoint(str(ck), map_location=str(device))
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    # build dataset from data/test/*.jpg
    test_dir = data_dir / 'test'
    imgs = sorted([p for p in test_dir.glob('*.jpg')])
    transform = get_transforms('val')

    import torch.utils.data as data
    from PIL import Image

    class TestDS(data.Dataset):
        def __init__(self, imgs, transform):
            self.imgs = imgs
            self.transform = transform

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, i):
            p = self.imgs[i]
            img = Image.open(p).convert('RGB')
            return self.transform(img), p.stem

    ds = TestDS(imgs, transform)
    loader = data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    import csv
    rows = []
    with torch.no_grad():
        for xb, names in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits.cpu(), dim=1)
            confs, preds = probs.max(dim=1)
            for name, pred in zip(names, preds.tolist()):
                breed = idx_to_class.get(int(pred), str(pred))
                rows.append({'id': name, 'breed': breed})

    # write submission
    import pandas as pd
    pd.DataFrame(rows)[['id', 'breed']].to_csv(out_path, index=False)
    print(f'Wrote submission to {out_path} with {len(rows)} rows')


if __name__ == '__main__':
    main()
