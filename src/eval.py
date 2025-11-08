"""Evaluate a checkpoint on the validation split and write metrics, predictions, and confusion matrix."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import torch
from typing import Dict, List

from .models import get_model
from .data import get_dataloaders
from .utils import get_device, load_checkpoint


def ensure_csvs_from_txts(data_dir: Path, splits_dir: Path) -> Dict[str, str]:
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
            raise SystemExit(f'Missing split: {name} in {splits_dir}')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--splits-path', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default=None)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    splits_dir = Path(args.splits_path)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ensure_csvs_from_txts(data_dir, splits_dir)

    # choose device (get_device prefers CUDA, then MPS, then CPU)
    device = get_device() if args.device is None else torch.device(args.device)
    pin_memory = (device.type == 'cuda')
    train_loader, val_loader, test_loader, meta = get_dataloaders(str(data_dir), splits, batch_size=args.batch_size, pin_memory=pin_memory)

    # build model
    model = get_model('resnet50', meta['num_classes'])
    ck = load_checkpoint(args.checkpoint, map_location=str(device))
    model.load_state_dict(ck['model_state'])
    model.to(device)
    model.eval()

    # inference
    import torch.nn as nn
    from sklearn.metrics import confusion_matrix, accuracy_score
    import pandas as pd
    all_preds = []
    all_trues = []
    all_files = []

    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits.cpu(), dim=1)
            confs, preds = probs.max(dim=1)
            all_preds.extend(preds.tolist())
            all_trues.extend(y.tolist())
            ds_samples = getattr(val_loader.dataset, 'samples', None)
            if ds_samples:
                all_files.extend([Path(s[0]).name for s in ds_samples[:len(preds)]])
            else:
                all_files.extend([''] * len(preds))

    # metrics
    acc = accuracy_score(all_trues, all_preds)
    cm = confusion_matrix(all_trues, all_preds)

    metrics = {'val_acc': float(acc)}
    with open(out_dir / 'metrics.json', 'w') as fh:
        json.dump(metrics, fh)

    # write predictions
    rows = []
    idx_to_class = meta.get('idx_to_class', {})
    for fn, t, p in zip(all_files, all_trues, all_preds):
        rows.append({'filename': fn, 'true_label': idx_to_class.get(t, str(t)), 'pred_label': idx_to_class.get(p, str(p)), 'pred_conf': ''})
    pd.DataFrame(rows).to_csv(out_dir / 'val_predictions.csv', index=False)

    # confusion matrix image (matplotlib only)
    import matplotlib.pyplot as plt
    labels = [meta.get('idx_to_class', {}).get(i, str(i)) for i in range(meta.get('num_classes', 0))]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=list(range(len(labels))), yticks=list(range(len(labels))),
           xticklabels=labels, yticklabels=labels, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    cm_path = out_dir / 'confusion_matrix.png'
    fig.savefig(cm_path)
    print(f'Wrote metrics to {out_dir}/metrics.json and predictions to {out_dir}/val_predictions.csv and {cm_path}')

    # top-5 most confused pairs (off-diagonal largest counts)
    import numpy as np
    cm2 = cm.copy()
    np.fill_diagonal(cm2, 0)
    pairs = []
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            if cm2[i, j] > 0:
                pairs.append((cm2[i, j], i, j))
    pairs.sort(reverse=True)
    print('Top-5 most confused class pairs:')
    for cnt, i, j in pairs[:5]:
        print(f'  {labels[i]} -> {labels[j]}: {cnt}')

    print(f'Wrote metrics to {out_dir}/metrics.json and predictions to {out_dir}/val_predictions.csv and confusion_matrix.png')


if __name__ == '__main__':
    main()
