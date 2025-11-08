"""Evaluate a trained Keras model on validation or test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def make_prediction_grid(image_paths: list[str], true_labels: list[str], 
                        pred_labels: list[str], output_path: Path, n_samples: int = 25):
    """Create a grid of sample predictions."""
    # Select random samples if we have more than n_samples
    if len(image_paths) > n_samples:
        indices = np.random.choice(len(image_paths), n_samples, replace=False)
    else:
        indices = range(len(image_paths))
    
    # Create 5x5 grid
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    axes = axes.ravel()
    
    for idx, i in enumerate(indices):
        # Load and display image
        img = tf.io.read_file(image_paths[i])
        img = tf.image.decode_jpeg(img, channels=3)
        axes[idx].imshow(img)
        
        # Add title with true and predicted labels
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        title = f'True: {true_labels[i]}\nPred: {pred_labels[i]}'
        axes[idx].set_title(title, color=color, fontsize=8)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / f'sample_preds_{len(indices)}.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--splits-path', required=True)
    parser.add_argument('--checkpoint', '--model', dest='model', required=True, help='Path to best.keras or best.h5 (or directory containing them)')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--export-samples', type=int, default=0)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    splits_path = Path(args.splits_path)
    model_path = Path(args.model)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    # If user passed a directory, prefer best.keras then best.h5 inside it
    if model_path.is_dir():
        cand1 = model_path / 'best.keras'
        cand2 = model_path / 'best.h5'
        if cand1.exists():
            model_path = cand1
        elif cand2.exists():
            model_path = cand2
        else:
            raise FileNotFoundError(f'No best.keras or best.h5 found in {model_path}')

    # Load without compiling to avoid needing custom loss/optimizer objects
    model = tf.keras.models.load_model(model_path, compile=False)
    img_size = args.img_size
    
    # Load class mapping
    model_dir = model_path.parent
    with open(model_dir / 'class_to_idx.json', 'r') as f:
        class_to_idx = json.load(f)
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    
    # Read split file and create dataset
    split_file = splits_path / f'{args.split}.txt'
    image_paths = []
    true_labels = []
    skipped = 0
    labels_df = None
    if args.split == 'val':
        labels_df = pd.read_csv(data_dir / 'labels.csv', dtype={'id': str})

    with open(split_file, 'r') as f:
        for line in f:
            img_id = line.strip()
            if img_id.endswith('.jpg'):
                img_id = Path(img_id).stem

            # For test set, we might not have labels
            img_path = data_dir / ('train' if args.split == 'val' else 'test') / f'{img_id}.jpg'
            if not img_path.exists():
                skipped += 1
                print(f"Skipping missing file: {img_path}")
                continue
            image_paths.append(str(img_path))

            if args.split == 'val':
                label = labels_df[labels_df['id'] == img_id]['breed'].iloc[0]
                true_labels.append(label)

    if skipped:
        print(f"Skipped {skipped} missing files from {split_file}")
    
    # Create tf.data.Dataset
    # Prefer backbone-specific preprocessing if we can detect backbone from model layers
    preprocess_fn = None
    for layer in model.layers:
        lname = getattr(layer, 'name', '').lower()
        if 'efficientnet' in lname:
            try:
                from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as eff_pre
                preprocess_fn = eff_pre
                break
            except Exception:
                preprocess_fn = None
        if 'mobilenet' in lname:
            try:
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_pre
                preprocess_fn = mb_pre
                break
            except Exception:
                preprocess_fn = None

    def preprocess(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)
        if preprocess_fn is not None:
            img = preprocess_fn(img)
        else:
            img = img / 255.0
        return img

    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Get predictions
    predictions = model.predict(ds)
    pred_labels = [idx_to_class[p.argmax()] for p in predictions]
    pred_conf = [float(p.max()) for p in predictions]
    
    # Create output dataframe
    df = pd.DataFrame()
    df['filename'] = [Path(p).stem for p in image_paths]
    if args.split == 'val':
        df['true_label'] = true_labels
    df['pred_label'] = pred_labels
    df['pred_conf'] = pred_conf
    
    # Save predictions
    df.to_csv(output_path / f'{args.split}_predictions.csv', index=False)
    
    # If validation split, create confusion matrix
    if args.split == 'val':
        classes = sorted(set(true_labels))
        # prefer using the shared helper
        try:
            from .keras_train import plot_confusion_matrix
            plot_confusion_matrix(true_labels, pred_labels, classes, output_path, title=f'Confusion Matrix - {args.split.title()} Set', fname=f'{args.split}_confusion_matrix.png')
        except Exception:
            # fallback: plot inline
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            cm = confusion_matrix(true_labels, pred_labels, labels=classes)
            plt.figure(figsize=(12, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.title(f'Confusion Matrix - {args.split.title()} Set')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(output_path / f'{args.split}_confusion_matrix.png')
            plt.close()
    
    # Export sample predictions if requested
    if args.export_samples > 0 and len(image_paths) >= args.export_samples:
        # We saved predictions CSV earlier; prefer using it with the utility
        preds_csv = output_path / f'{args.split}_predictions.csv'
        try:
            from .make_sample_grid import make_prediction_grid
            # write sample_preds_{split}.png
            make_prediction_grid(preds_csv, Path(args.data_dir), output_path, n_samples=args.export_samples, output_fname=f'sample_preds_{args.split}.png')
        except Exception:
            # fallback: build grid from arrays
            make_prediction_grid(preds_csv, Path(args.data_dir), output_path, n_samples=args.export_samples, output_fname=f'sample_preds_{args.split}.png')


if __name__ == '__main__':
    main()