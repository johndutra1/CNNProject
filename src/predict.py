from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .models import get_model
from .utils import load_checkpoint, get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--sample-csv', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    device = get_device()
    ss = pd.read_csv(args.sample_csv)
    # load model
    # infer num_classes from checkpoint or from sample cols
    ckpt = load_checkpoint(args.checkpoint, map_location='cpu')
    # number of classes inferred from checkpoint weights
    # instantiate model
    # here we assume resnet50
    num_classes = ckpt.get('model_state', {}).get('fc.weight', None)
    model = get_model('resnet50', num_classes=120)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    rows = []
    for _, r in ss.iterrows():
        img_id = str(r['id'])
        img_path = Path(args.data_dir) / 'test' / f"{img_id}.jpg"
        from PIL import Image
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = Image.open(img_path).convert('RGB')
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy().squeeze().tolist()
        rows.append([img_id] + probs)

    # columns: id,<breed1>,<breed2>,...
    # attempt to get breed columns from sample submission header if present
    cols = list(ss.columns)
    if len(cols) > 1:
        breed_cols = cols[1:]
    else:
        breed_cols = [f'breed_{i}' for i in range(120)]
    out_df = pd.DataFrame(rows, columns=['id'] + breed_cols)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f'Wrote submission to {args.out}')


if __name__ == '__main__':
    main()
