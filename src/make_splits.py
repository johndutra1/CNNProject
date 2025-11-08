"""Create stratified train/val/test splits and write simple text lists.

Produces three plain text files (one filename per line, relative to data/train/):
  {out_dir}/train.txt
  {out_dir}/val.txt
  {out_dir}/test.txt

Example:
  python -m src.make_splits --data-dir data --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42 --out-dir data/splits
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_three_split(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    """Return (train_df, val_df, test_df) stratified by 'breed'."""
    if not (0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0):
        raise ValueError('ratios must be between 0 and 1')
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError('train/val/test ratios must sum to 1.0')

    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(df, stratify=df['breed'], test_size=temp_ratio, random_state=seed)
    # split temp into val and test
    val_frac = val_ratio / temp_ratio if temp_ratio > 0 else 0.0
    if temp_ratio > 0:
        val_df, test_df = train_test_split(temp_df, stratify=temp_df['breed'], test_size=(1 - val_frac), random_state=seed)
    else:
        val_df = df.iloc[0:0].copy()
        test_df = df.iloc[0:0].copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def ids_from_df(df: pd.DataFrame) -> List[str]:
    return [str(x).strip() for x in df['id'].tolist()]


def write_list_file(out_path: Path, ids: Iterable[str]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as fh:
        for _id in ids:
            fh.write(f"{_id}.jpg\n")


def main():
    parser = argparse.ArgumentParser(description='Create stratified train/val/test plain-text lists')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--train-ratio', type=float, default=0.70)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', default='data/splits')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    labels_csv = data_dir / 'labels.csv'
    if not labels_csv.exists():
        raise SystemExit(f"labels.csv not found in {data_dir} — run download_data.py first")

    out_dir = Path(args.out_dir)
    train_txt = out_dir / 'train.txt'
    val_txt = out_dir / 'val.txt'
    test_txt = out_dir / 'test.txt'

    # idempotency: if files exist, do nothing
    if train_txt.exists() and val_txt.exists() and test_txt.exists():
        print(f"Split files already exist at {out_dir}; skipping generation.")
        return

    df = pd.read_csv(labels_csv, dtype={"id": str})
    required = {'id', 'breed'}
    if not required.issubset(set(df.columns)):
        raise SystemExit('labels.csv must contain id and breed columns')

    train_df, val_df, test_df = stratified_three_split(df, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    train_ids = ids_from_df(train_df)
    val_ids = ids_from_df(val_df)
    test_ids = ids_from_df(test_df)

    # write simple lists (filenames relative to data/train/)
    write_list_file(train_txt, train_ids)
    write_list_file(val_txt, val_ids)
    write_list_file(test_txt, test_ids)

    # summary
    n_train = len(train_ids)
    n_val = len(val_ids)
    n_test = len(test_ids)
    total_images_on_disk = len(list((data_dir / 'train').glob('*.jpg')))

    print('Wrote split files:')
    print(f'  {train_txt} ({n_train} lines)')
    print(f'  {val_txt} ({n_val} lines)')
    print(f'  {test_txt} ({n_test} lines)')

    # overlap checks
    s_train: Set[str] = set(train_ids)
    s_val: Set[str] = set(val_ids)
    s_test: Set[str] = set(test_ids)
    inter_tv = s_train & s_val
    inter_tt = s_train & s_test
    inter_vt = s_val & s_test

    print('\nSplit counts:')
    print(f'  train: {n_train}\n  val: {n_val}\n  test: {n_test}')
    if inter_tv or inter_tt or inter_vt:
        print('\nOverlap detected between splits:')
        if inter_tv:
            print(f'  train ∩ val: {len(inter_tv)}')
        if inter_tt:
            print(f'  train ∩ test: {len(inter_tt)}')
        if inter_vt:
            print(f'  val ∩ test: {len(inter_vt)}')
    else:
        print('\nNo overlaps between train/val/test (OK)')

    total_from_splits = n_train + n_val + n_test
    print(f'\nTotal from splits: {total_from_splits}')
    print(f'Total images on disk (data/train): {total_images_on_disk}')
    if total_from_splits == total_images_on_disk:
        print('Total matches number of images on disk (OK)')
    else:
        print('WARNING: total from splits does NOT equal number of images on disk')


if __name__ == '__main__':
    main()
