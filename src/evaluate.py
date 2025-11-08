from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import List

import torch
from torch import nn

from .models import get_model
from .data import get_dataloaders
from .utils import get_device, load_checkpoint


def ensure_csvs_from_txts(data_dir: Path, splits_dir: Path) -> dict:
    import pandas as pd
    out = {}
    labels_df = pd.read_csv(data_dir / 'labels.csv', dtype={'id': str})
    for name in ('train', 'val', 'test'):
        csv_path = splits_dir / f'{name}.csv'
        txt_path = splits_dir / f'{name}.txt'
        if csv_path.exists():
            out[name] = str(csv_path)
        elif txt_path.exists():
            rows = []
            for line in open(txt_path, 'r'):
                idstr = line.strip()
                if idstr.endswith('.jpg'):
                    idstr = Path(idstr).stem
                img = data_dir / 'train' / f"{idstr}.jpg"
                lab = labels_df.loc[labels_df['id'] == idstr, 'breed']
                label = lab.iloc[0] if not lab.empty else ''
                rows.append({'image_path': str(img), 'label': label})
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            out[name] = str(csv_path)
        else:
            # leave missing splits as None
            out[name] = None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--splits-path', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--output', required=True)
    p.add_argument('--img-size', type=int, default=224)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    splits_dir = Path(args.splits_path)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ensure_csvs_from_txts(data_dir, splits_dir)
    target_split = args.split
    if splits.get(target_split) is None:
        raise SystemExit(f'Missing split {target_split} in {splits_dir}')

    device = get_device()

    # choose pin_memory only for CUDA
    pin_memory = (device.type == 'cuda')
    train_loader, val_loader, test_loader, meta = get_dataloaders(str(data_dir), splits, batch_size=args.batch_size, pin_memory=pin_memory, img_size=args.img_size)
    loader = val_loader if target_split == 'val' else test_loader

    model = get_model('resnet50', meta['num_classes'])
    ck = load_checkpoint(args.checkpoint, map_location=str(device))
    model.load_state_dict(ck['model_state'])
    model.to(device)
    model.eval()

    idx_to_class = meta.get('idx_to_class', {})
    # if run dir has class_to_idx.json, prefer that
    run_dir = Path(args.checkpoint).parent
    c2i_path = run_dir / 'class_to_idx.json'
    if c2i_path.exists():
        with open(c2i_path, 'r') as fh:
            class_to_idx = json.load(fh)
            idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    all_preds: List[int] = []
    all_trues: List[int] = []
    all_confs: List[float] = []

    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits.cpu(), dim=1)
            top5 = torch.topk(probs, k=min(5, probs.size(1)), dim=1)
            # top1
            confs, preds = probs.max(dim=1)
            all_preds.extend(preds.tolist())
            # if labels present
            if y is not None:
                all_trues.extend(y.tolist())
            all_confs.extend(confs.tolist())

    # build filenames list
    base_ds = getattr(loader.dataset, 'dataset', loader.dataset)
    filenames = []
    if hasattr(base_ds, 'samples'):
        all_paths = [s[0] for s in base_ds.samples]
        if hasattr(loader.dataset, 'indices'):
            indices = list(loader.dataset.indices)
            filenames = [Path(all_paths[i]).name for i in indices]
        else:
            filenames = [Path(p).name for p in all_paths]
    else:
        filenames = [''] * len(all_preds)

    # write CSV with top5 preds as semicolon joined labels
    import csv
    with open(out_dir / f'{target_split}_predictions.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['filename', 'true_label', 'pred_label', 'pred_conf', 'top5_preds'])
        w.writeheader()
        for fn, t, p, c in zip(filenames, all_trues if all_trues else [''] * len(all_preds), all_preds, all_confs):
            # compute top5 labels for this sample
            # rerun small forward to get top5 (or reuse earlier stored top5 if we stored)
            w.writerow({'filename': fn, 'true_label': idx_to_class.get(t, '') if t != '' else '', 'pred_label': idx_to_class.get(p, str(p)), 'pred_conf': float(c), 'top5_preds': ''})

    # compute top1/top5 accuracy if labels available
    if len(all_trues) == len(all_preds) and len(all_trues) > 0:
        import numpy as np
        top1 = (np.array(all_trues) == np.array(all_preds)).mean()
        # compute top5 by re-running in batches to get top5
        top5_correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x = x.to(device)
                logits = model(x)
                probs = torch.softmax(logits.cpu(), dim=1)
                topk = torch.topk(probs, k=min(5, probs.size(1)), dim=1).indices
                for i in range(topk.size(0)):
                    total += 1
                    if y[i].item() in topk[i].tolist():
                        top5_correct += 1
        top5_acc = top5_correct / total if total > 0 else 0.0
        print(f'Top-1 accuracy: {top1:.4f}')
        print(f'Top-5 accuracy: {top5_acc:.4f}')
    else:
        print('No ground-truth labels available for top-1 accuracy')

    print(f'Wrote predictions to {out_dir}/{target_split}_predictions.csv')


if __name__ == '__main__':
    main()
