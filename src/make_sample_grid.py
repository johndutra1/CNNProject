"""Utility script to create a grid of sample predictions."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def make_prediction_grid(csv_path: Path, data_dir: Path, output_path: Path, n_samples: int = 25, output_fname: str | None = None):
    """Create a grid of sample predictions from a predictions CSV.

    If output_fname is provided it will be used as the file name (under output_path).
    Otherwise 'prediction_grid.png' is used.
    """
    # Read predictions
    df = pd.read_csv(csv_path)
    
    # Sample rows if we have more than n_samples
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
    
    # Create 5x5 grid
    n_rows = int(np.ceil(len(df) / 5))
    fig, axes = plt.subplots(n_rows, 5, figsize=(15, 3*n_rows))
    axes = axes.ravel()
    
    for idx, (_, row) in enumerate(df.iterrows()):
        # Determine image path based on filename
        img_path = data_dir / ('train' if 'true_label' in df.columns else 'test') / f"{row['filename']}.jpg"
        
        # Load and display image
        img = tf.io.read_file(str(img_path))
        img = tf.image.decode_jpeg(img, channels=3)
        axes[idx].imshow(img.numpy().astype('uint8'))
        
        # Format title
        if 'true_label' in df.columns:
            color = 'green' if row['true_label'] == row['pred_label'] else 'red'
            title = f"True: {row['true_label']}\nPred: {row['pred_label']}\nConf: {row['pred_conf']:.2f}"
        else:
            color = 'black'
            title = f"Pred: {row['pred_label']}\nConf: {row['pred_conf']:.2f}"
        
        axes[idx].set_title(title, color=color, fontsize=8)
        axes[idx].axis('off')
    
    # Hide empty subplots
    for idx in range(len(df), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    fname = output_fname or 'prediction_grid.png'
    plt.savefig(output_path / fname, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions-csv', required=True, help='Path to predictions CSV')
    parser.add_argument('--data-dir', required=True, help='Path to data directory containing train/test folders')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--samples', type=int, default=25, help='Number of samples to show')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    make_prediction_grid(Path(args.predictions_csv), data_dir, output_path, args.samples)


if __name__ == '__main__':
    main()