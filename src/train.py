from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from contextlib import nullcontext
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from .data import get_dataloaders
from .models import get_model
from .utils import accuracy, get_device, save_checkpoint, seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--splits-path', required=True)
    p.add_argument('--limit-train', type=int, default=None, help='Limit number of training samples for smoke tests (None=no limit)')
    p.add_argument('--limit-val', type=int, default=None, help='Limit number of validation samples for smoke tests (None=no limit)')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--label-smoothing', type=float, default=0.0)
    p.add_argument('--randaugment', nargs=2, type=int, default=None, help='RandAugment: N M')
    p.add_argument('--random-erasing', type=float, default=0.0, help='RandomErasing probability')
    p.add_argument('--model', default='resnet50')
    p.add_argument('--output', default='runs/exp1')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs-warmup', type=int, default=0)
    p.add_argument('--scheduler', choices=['cosine', 'none'], default='cosine')
    p.add_argument('--optim', choices=['adamw', 'sgd'], default='adamw')
    p.add_argument('--sampler', choices=['none', 'weighted'], default='weighted')
    p.add_argument('--class-weights', choices=['none', 'auto'], default='none', help='If auto, compute class weights from training set and pass to loss')
    p.add_argument('--img-size', type=int, default=224, help='Input image size (square)')
    p.add_argument('--use-class-weights', action='store_true')
    p.add_argument('--early-stop', type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    # prefer CUDA, otherwise MPS, otherwise CPU
    import torch as _torch
    device = _torch.device("cuda" if _torch.cuda.is_available() else ("mps" if _torch.backends.mps.is_available() else "cpu"))
    print('Using device', device)

    # Allow splits to be provided as .txt lists (one id per line) or csvs. If txt, build simple csvs
    splits_dir = Path(args.splits_path)
    splits = {}
    for name in ('train', 'val', 'test'):
        csv_path = splits_dir / f'{name}.csv'
        txt_path = splits_dir / f'{name}.txt'
        if csv_path.exists():
            splits[name] = str(csv_path)
        elif txt_path.exists():
            # build csv alongside splits_dir
            df_rows = []
            import pandas as pd
            labels_df = pd.read_csv(Path(args.data_dir) / 'labels.csv', dtype={'id': str})
            for line in open(txt_path, 'r'):
                idstr = line.strip()
                if idstr.endswith('.jpg'):
                    idstr = Path(idstr).stem
                img_path = Path(args.data_dir) / 'train' / f"{idstr}.jpg"
                # find label
                lab = labels_df.loc[labels_df['id'] == idstr, 'breed']
                if lab.empty:
                    label = ''
                else:
                    label = lab.iloc[0]
                df_rows.append({'image_path': str(img_path), 'label': label})
            out_csv = splits_dir / f'{name}.csv'
            pd.DataFrame(df_rows).to_csv(out_csv, index=False)
            splits[name] = str(out_csv)
        else:
            raise SystemExit(f'Missing split file for {name} in {splits_dir} (expected {name}.csv or {name}.txt)')

    # set pin_memory only for CUDA to avoid MPS warnings
    pin_memory = (device.type == 'cuda')
    # parse randaugment tuple
    randaug = tuple(args.randaugment) if args.randaugment is not None else None

    train_loader, val_loader, test_loader, meta = get_dataloaders(
        args.data_dir,
        splits,
        batch_size=args.batch_size,
        pin_memory=pin_memory,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        img_size=args.img_size,
        randaugment=randaug,
        random_erasing_p=args.random_erasing,
        sampler_strategy=args.sampler,
    )
    model = get_model(args.model, meta['num_classes']).to(device)

    # optimizer selection
    if args.optim == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Configure AMP/GradScaler: only use CUDA AMP on CUDA; on other devices use no-op autocast
    use_cuda_amp = (device.type == 'cuda')
    if use_cuda_amp:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
    else:
        autocast = lambda *a, **k: nullcontext()
        scaler = None
    total_steps = len(train_loader) * args.epochs
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(opt, T_max=max(1, len(train_loader) * args.epochs))

    writer = SummaryWriter(log_dir=args.output)
    best_val = 0.0
    epochs_no_improve = 0

    # prepare loss function (optionally with class weights and label smoothing)
    loss_weight = None
    if args.class_weights == 'auto' or args.use_class_weights:
        cw = meta.get('class_weights')
        if cw is not None:
            loss_weight = torch.tensor(cw, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weight, label_smoothing=float(args.label_smoothing))

    for epoch in range(1, args.epochs + 1):
        # linear warmup of lr (simple)
        if args.epochs_warmup and args.epochs_warmup > 0 and epoch <= args.epochs_warmup:
            factor = float(epoch) / float(max(1, args.epochs_warmup))
            for g in opt.param_groups:
                g['lr'] = args.lr * factor
        
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            # use autocast() as a context manager; on non-CUDA this is a no-op
            with autocast():
                logits = model(x)
                loss = loss_fn(logits, y)

            if use_cuda_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            acc = accuracy(logits.detach().cpu(), y.detach().cpu())
            running_loss += loss.item()
            running_acc += acc
            if i % 10 == 0:
                writer.add_scalar('train/loss_batch', loss.item(), epoch * len(train_loader) + i)

        avg_loss = running_loss / len(train_loader)
        avg_acc = running_acc / len(train_loader)
        writer.add_scalar('train/loss', avg_loss, epoch)
        writer.add_scalar('train/acc', avg_acc, epoch)
        # log learning rate
        try:
            lr = opt.param_groups[0]['lr']
            writer.add_scalar('train/lr', lr, epoch)
        except Exception:
            pass

        # validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                # autocast is a no-op on non-CUDA
                with autocast():
                    logits = model(x)
                    loss = loss_fn(logits, y)
                val_loss += loss.item()
                val_acc += accuracy(logits.cpu(), y.cpu())
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)

        print(f"Epoch {epoch}/{args.epochs} train_loss={avg_loss:.4f} train_acc={avg_acc:.4f} val_acc={val_acc:.4f}")

        # checkpoint
        save_checkpoint(str(Path(args.output) / 'last.pt'), {'epoch': epoch, 'model_state': model.state_dict(), 'opt_state': opt.state_dict(), 'val_acc': val_acc})
        if val_acc > best_val:
            best_val = val_acc
            save_checkpoint(str(Path(args.output) / 'best.pt'), {'epoch': epoch, 'model_state': model.state_dict(), 'opt_state': opt.state_dict(), 'val_acc': val_acc})
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if args.early_stop and epochs_no_improve >= args.early_stop:
            print('Early stopping')
            break

        # step scheduler after epoch
        if scheduler is not None:
            scheduler.step()

    writer.close()

    # After training: write metrics, class mapping, and val predictions + confusion matrix using best model
    import json
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        'train_loss': float(avg_loss),
        'train_acc': float(avg_acc),
        'val_loss': float(val_loss),
        'val_acc': float(val_acc),
    }
    with open(out_dir / 'metrics.json', 'w') as fh:
        json.dump(metrics, fh)

    # save class_to_idx mapping
    with open(out_dir / 'class_to_idx.json', 'w') as fh:
        json.dump(meta.get('class_to_idx', {}), fh)

    # If best checkpoint exists, load and run inference on val set to produce predictions and confusion matrix
    best_path = out_dir / 'best.pt'
    if best_path.exists():
        from .utils import load_checkpoint
        ck = load_checkpoint(str(best_path), map_location=str(device))
        model.load_state_dict(ck['model_state'])
        model.to(device)
        model.eval()

        all_preds = []
        all_trues = []
        all_files = []
        all_confs = []
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                probs = torch.softmax(logits.cpu(), dim=1)
                confs, preds = probs.max(dim=1)
                all_preds.extend(preds.tolist())
                all_trues.extend(y.tolist())
                all_confs.extend(confs.tolist())

        # build filenames list from underlying dataset samples (handles Subset)
        base_val_ds = getattr(val_loader.dataset, 'dataset', val_loader.dataset)
        filenames = []
        if hasattr(base_val_ds, 'samples'):
            all_paths = [s[0] for s in base_val_ds.samples]
            if hasattr(val_loader.dataset, 'indices'):
                indices = list(val_loader.dataset.indices)
                filenames = [Path(all_paths[i]).name for i in indices]
            else:
                filenames = [Path(p).name for p in all_paths]
        else:
            filenames = [''] * len(all_preds)

        # write val_predictions.csv
        import csv
        with open(out_dir / 'val_predictions.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['filename', 'true_label', 'pred_label', 'pred_conf'])
            w.writeheader()
            idx_to_class = meta.get('idx_to_class', {})
            for fn, t, p, c in zip(filenames, all_trues, all_preds, all_confs):
                w.writerow({'filename': fn, 'true_label': idx_to_class.get(t, str(t)), 'pred_label': idx_to_class.get(p, str(p)), 'pred_conf': float(c)})

        # confusion matrix (matplotlib only)
        labels = [meta.get('idx_to_class', {}).get(i, str(i)) for i in range(meta.get('num_classes', 0))]
        cm = confusion_matrix(all_trues, all_preds, labels=list(range(len(labels))))
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks and label them with the respective list entries
        ax.set(xticks=list(range(len(labels))), yticks=list(range(len(labels))),
               xticklabels=labels, yticklabels=labels, ylabel='True label', xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        # annotate cells
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')
        fig.tight_layout()
        cm_path = out_dir / 'confusion_matrix.png'
        fig.savefig(cm_path)
        try:
            writer.add_figure('val/confusion_matrix', fig, global_step=0)
        except Exception:
            pass
        # save mistakes grid (6x6 of most-confident wrong predictions)
        try:
            wrongs = [i for i, (t, p) in enumerate(zip(all_trues, all_preds)) if t != p]
            wrongs_sorted = sorted(wrongs, key=lambda i: all_confs[i], reverse=True)
            top_wrongs = wrongs_sorted[:36]
            if top_wrongs:
                import matplotlib.pyplot as plt
                from PIL import Image
                fig2, axes = plt.subplots(6, 6, figsize=(12, 12))
                axes = axes.flatten()
                for ax in axes:
                    ax.axis('off')
                for ax, idx in zip(axes, top_wrongs):
                    img_path = None
                    # map filename back to full path if available
                    if hasattr(base_val_ds, 'samples'):
                        if hasattr(val_loader.dataset, 'indices'):
                            full_paths = [p for p in all_paths]
                            # indices already used earlier
                            try:
                                sel = list(val_loader.dataset.indices)[idx]
                                img_path = full_paths[sel]
                            except Exception:
                                img_path = None
                        else:
                            img_path = all_paths[idx]
                    if img_path and Path(img_path).exists():
                        im = Image.open(img_path).convert('RGB')
                        im = im.resize((args.img_size, args.img_size))
                        ax.imshow(im)
                        t = all_trues[idx]
                        p = all_preds[idx]
                        conf = all_confs[idx]
                        ax.set_title(f"{meta.get('idx_to_class', {}).get(p, p)}\ntrue:{meta.get('idx_to_class', {}).get(t, t)} {conf:.2f}", fontsize=8)
                fig2.tight_layout()
                mistakes_path = out_dir / 'mistakes_grid.png'
                fig2.savefig(mistakes_path)
        except Exception:
            mistakes_path = None

        print(f'Wrote artifacts to {out_dir}')
        print(f'Best model saved to: {best_path}')
        print(f'Confusion matrix saved to: {cm_path}')
        if mistakes_path:
            print(f'Mistakes grid saved to: {mistakes_path}')
    else:
        print('No best.pt found; skipping val predictions save')


if __name__ == '__main__':
    main()
