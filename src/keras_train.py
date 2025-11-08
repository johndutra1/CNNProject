"""Keras training script using tf.keras.applications backbones and tf.data.

Features:
- Supports MobileNetV2 and EfficientNetV2B0 backbones (imagenet weights)
- Keras preprocessing layers for augmentation (RandomFlip/Rotation/Zoom)
- Deterministic fine-tune-last-N by unfreezing last N layers of the base model
- Saves: best.keras, history.csv, history.png, metrics.json, val_predictions.csv,
  confusion_matrix.png, sample_grid.png, and appends experiments.csv
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard


def plot_history(history, output_path: Path):
    """Plot training history and save to disk."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    ax2.plot(history.history['accuracy'], label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Val Acc')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'history.png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, output_path: Path, title='Confusion Matrix', fname: str | None = None):
    """Plot and save confusion matrix.

    Args:
        y_true: iterable of true labels
        y_pred: iterable of predicted labels
        class_names: list of class names (order used for labels)
        output_path: directory to save the plot
        title: plot title
        fname: optional filename (e.g., 'confusion_matrix.png'). If None, defaults to 'confusion_matrix.png'.
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_name = fname or 'confusion_matrix.png'
    plt.savefig(output_path / out_name)
    plt.close()


def build_backbone(name: str, img_size: int):
    name = name.lower()
    if name.startswith('efficientnetv2') or 'efficientnet' in name:
        # Use EfficientNetV2B0 for predictability
        base = tf.keras.applications.EfficientNetV2B0(
            include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3), pooling='avg'
        )
    elif 'mobilenet' in name:
        base = tf.keras.applications.MobileNetV2(
            include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3), pooling='avg'
        )
    else:
        raise ValueError(f'Unsupported backbone: {name}')
    return base


def make_dataset_from_split(data_dir: Path, splits_path: Path, split: str, img_size: int, batch_size: int,
                            limit: int | None = None, shuffle: bool = True, augment: bool = False,
                            backbone: str | None = None):
    # Read split ids
    split_file = splits_path / f'{split}.txt'
    with open(split_file, 'r') as fh:
        ids = [l.strip() for l in fh]
    if limit is not None:
        ids = ids[:limit]

    labels_df = pd.read_csv(data_dir / 'labels.csv', dtype={'id': str})
    labels_df['id'] = labels_df['id'].astype(str)
    le = LabelEncoder()
    labels_df['label_idx'] = le.fit_transform(labels_df['breed'])
    class_names = list(le.classes_)

    paths = []
    y = []
    for img_id in ids:
        if img_id.endswith('.jpg'):
            img_id = Path(img_id).stem
        img_path = data_dir / 'train' / f'{img_id}.jpg'
        if not img_path.exists():
            continue
        paths.append(str(img_path))
        y.append(int(labels_df[labels_df['id'] == img_id]['label_idx'].iloc[0]))

    ds = tf.data.Dataset.from_tensor_slices((paths, y))

    # choose backbone-specific preprocess_input
    preprocess_fn = None
    if backbone is not None:
        bn = backbone.lower()
        try:
            if 'efficientnet' in bn:
                from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as eff_pre
                preprocess_fn = eff_pre
            elif 'mobilenet' in bn:
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_pre
                preprocess_fn = mb_pre
        except Exception:
            preprocess_fn = None

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)
        if preprocess_fn is not None:
            img = preprocess_fn(img)
        else:
            img = img / 255.0
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, class_names, paths, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--splits-path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', default='efficientnetv2b0', help='efficientnetv2b0 or mobilenet_v2')
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--warmup-epochs', type=int, default=1, help='Epochs to train with base frozen when fine-tuning')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--label-smoothing', type=float, default=0.0)
    p.add_argument('--fine-tune-last', type=int, default=0, help='Unfreeze last N layers of the base model')
    p.add_argument('--limit-train', type=int, default=None)
    p.add_argument('--limit-val', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--export-samples', type=int, default=25)
    args = p.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    splits_path = Path(args.splits_path)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds, class_names, train_paths, train_labels = make_dataset_from_split(
        data_dir, splits_path, 'train', args.img_size, args.batch_size, limit=args.limit_train, shuffle=True, augment=False, backbone=args.model
    )
    val_ds, _, val_paths, val_labels = make_dataset_from_split(
        data_dir, splits_path, 'val', args.img_size, args.batch_size, limit=args.limit_val, shuffle=False, augment=False, backbone=args.model
    )

    # Save class mapping
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    with open(out / 'class_to_idx.json', 'w') as fh:
        json.dump(class_to_idx, fh, indent=2)

    # Build model
    base = build_backbone(args.model, args.img_size)
    base.trainable = False

    inputs = tf.keras.Input(shape=(args.img_size, args.img_size, 3))
    # Augmentation layers
    x = tf.keras.layers.RandomFlip('horizontal')(inputs)
    x = tf.keras.layers.RandomRotation(0.06)(x)
    x = tf.keras.layers.RandomZoom(0.08)(x)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # Handle label smoothing: some Keras versions don't accept label_smoothing on
    # SparseCategoricalCrossentropy. To be compatible, implement label smoothing
    # by converting sparse labels to smoothed one-hot vectors inside a loss fn.
    if args.label_smoothing and args.label_smoothing > 0.0:
        num_classes = len(class_names)

        def loss_fn(y_true, y_pred):
            # y_true: shape (batch,), integer class indices
            y_true = tf.cast(y_true, tf.int32)
            y_true_oh = tf.one_hot(y_true, depth=num_classes)
            smooth = float(args.label_smoothing)
            y_true_smoothed = y_true_oh * (1.0 - smooth) + (smooth / float(num_classes))
            # use categorical_crossentropy on smoothed one-hot labels
            return tf.keras.losses.categorical_crossentropy(y_true_smoothed, y_pred)

        loss = loss_fn
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Callbacks
    ckpt = ModelCheckpoint(out / 'best.keras', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True, verbose=1)
    csvlogger = CSVLogger(out / 'history.csv')
    tb = TensorBoard(log_dir=out / 'logs')
    callbacks = [ckpt, early, csvlogger, tb]

    # Training: if fine-tune-last > 0, do warmup then unfreeze last N layers
    if args.fine_tune_last > 0 and args.warmup_epochs > 0:
        # Warmup with base frozen
        history1 = model.fit(train_ds, validation_data=val_ds, epochs=args.warmup_epochs, callbacks=callbacks, verbose=1)

        # Unfreeze last N layers of base
        total_layers = len(base.layers)
        n = min(args.fine_tune_last, total_layers)
        for layer in base.layers[:-n]:
            layer.trainable = False
        for layer in base.layers[-n:]:
            layer.trainable = True

        # Recompile with a lower LR for fine-tuning (drop LR by 10x)
        fine_lr = float(args.lr) / 10.0
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_lr), loss=loss, metrics=['accuracy'])
        remaining = max(1, args.epochs - args.warmup_epochs)
        history2 = model.fit(train_ds, validation_data=val_ds, epochs=remaining, callbacks=callbacks, verbose=1)
        # merge histories
        history = history1
        for k, v in history2.history.items():
            history.history.setdefault(k, []).extend(v)
    else:
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)

    # Save history plot
    plot_history(history, out)

    # Predictions on validation
    val_preds = model.predict(val_ds)
    pred_labels = [class_names[p.argmax()] for p in val_preds]
    pred_conf = [float(p.max()) for p in val_preds]

    val_filenames = [Path(p).stem for p in val_paths]
    val_true = [class_names[i] for i in val_labels]

    pd.DataFrame({
        'filename': val_filenames,
        'true_label': val_true,
        'pred_label': pred_labels,
        'pred_conf': pred_conf
    }).to_csv(out / 'val_predictions.csv', index=False)

    # Confusion matrix and sample grid
    plot_confusion_matrix(val_true, pred_labels, class_names, out, fname='confusion_matrix.png')

    # Sample grid: use first N samples (or random selection)
    try:
        from .make_sample_grid import make_prediction_grid
        sample_n = min(args.export_samples, len(val_paths))
        sample_paths = val_paths[:sample_n]
        sample_true = val_true[:sample_n]
        sample_pred = pred_labels[:sample_n]
        make_prediction_grid(sample_paths, sample_true, sample_pred, out, n_samples=sample_n)
    except Exception:
        # fallback: create a small grid in-place
        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
        axes = axes.ravel()
        for i in range(min(25, len(val_paths))):
            img = tf.io.read_file(val_paths[i])
            img = tf.image.decode_jpeg(img, channels=3)
            axes[i].imshow(img)
            color = 'green' if val_true[i] == pred_labels[i] else 'red'
            axes[i].set_title(f'T:{val_true[i]}\\nP:{pred_labels[i]}', color=color, fontsize=8)
            axes[i].axis('off')
        for j in range(i+1, 25):
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(out / 'sample_grid.png')
        plt.close()

    # Metrics from best epoch (approximate: max val_accuracy)
    best_idx = int(np.argmax(history.history.get('val_accuracy', [0])))
    metrics = {
        'train_acc': float(history.history.get('accuracy', [0])[best_idx]) if 'accuracy' in history.history else 0.0,
        'val_acc': float(history.history.get('val_accuracy', [0])[best_idx]) if 'val_accuracy' in history.history else 0.0,
        'train_loss': float(history.history.get('loss', [0])[best_idx]) if 'loss' in history.history else 0.0,
        'val_loss': float(history.history.get('val_loss', [0])[best_idx]) if 'val_loss' in history.history else 0.0,
    }
    with open(out / 'metrics.json', 'w') as fh:
        json.dump(metrics, fh, indent=2)

    # Also save a legacy HDF5 copy for compatibility
    try:
        model.save(out / 'best.h5')
    except Exception:
        # best.keras is primary; best.h5 is optional
        pass

    # Append experiments.csv
    exp = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': args.model,
        'img_size': args.img_size,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'label_smoothing': args.label_smoothing,
        'fine_tune_last': args.fine_tune_last,
        'train_acc': metrics['train_acc'],
        'val_acc': metrics['val_acc'],
        'notes': ''
    }
    exp_csv = Path('experiments.csv')
    dfexp = pd.DataFrame([exp])
    # append atomically: write header only if file doesn't exist
    if exp_csv.exists():
        dfexp.to_csv(exp_csv, mode='a', header=False, index=False)
    else:
        dfexp.to_csv(exp_csv, mode='w', header=True, index=False)


if __name__ == '__main__':
    main()