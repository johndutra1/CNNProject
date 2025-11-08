"""Download Kaggle competition data using the kaggle CLI and unzip train/test.

Usage:
python src/download_data.py --competition dog-breed-identification --out-dir data
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

from .utils import ensure_dir, run_cmd


def kaggle_available() -> bool:
    try:
        run_cmd(['kaggle', '--version'], capture_output=False)
        return True
    except Exception:
        return False


def unzip_if_needed(zip_path: Path, out_dir: Path):
    if not zip_path.exists():
        return False
    # Check if expected output exists (naive: if train/ folder exists for train.zip)
    if zip_path.stem == 'train' and (out_dir / 'train').exists():
        return False
    if zip_path.stem == 'test' and (out_dir / 'test').exists():
        return False

    print(f"Unzipping {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--competition', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))

    if not kaggle_available():
        print("kaggle CLI not found. Install it with 'pip install kaggle' and set up credentials (\n"
              "either create ~/.kaggle/kaggle.json or export KAGGLE_USERNAME and KAGGLE_KEY).")
        sys.exit(1)

    # Download competition files if not present
    expected_files = ['train.zip', 'test.zip', 'labels.csv']
    missing = [f for f in expected_files if not (out_dir / f).exists()]
    if missing:
        print(f"Downloading competition data for {args.competition} into {out_dir}")
        run_cmd(['kaggle', 'competitions', 'download', '-c', args.competition, '-p', str(out_dir)])
    else:
        print("All expected files already exist, skipping download.")

    # Unzip train and test
    unzip_if_needed(out_dir / 'train.zip', out_dir)
    unzip_if_needed(out_dir / 'test.zip', out_dir)

    # Print summary counts
    train_dir = out_dir / 'train'
    test_dir = out_dir / 'test'
    n_train = len(list(train_dir.glob('*.jpg'))) if train_dir.exists() else 0
    n_test = len(list(test_dir.glob('*.jpg'))) if test_dir.exists() else 0

    print(f"Summary:\n  train images: {n_train}\n  test images: {n_test}")


if __name__ == '__main__':
    main()
