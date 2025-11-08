"""Verify split lists: no overlap, files exist, labels present in labels.csv."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Set

import csv


def read_txt_ids(p: Path) -> Set[str]:
    ids = set()
    with open(p, 'r') as fh:
        for line in fh:
            s = line.strip()
            if s.endswith('.jpg'):
                s = Path(s).stem
            ids.add(s)
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--splits-path', required=True)
    p.add_argument('--labels-csv', required=True)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    splits_dir = Path(args.splits_path)
    labels_csv = Path(args.labels_csv)

    if not labels_csv.exists():
        print(f'labels csv not found: {labels_csv}')
        sys.exit(2)

    labels_ids = set()
    with open(labels_csv, newline='') as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            labels_ids.add(str(r['id']).strip())

    any_fail = False
    split_ids = {}
    for name in ('train', 'val', 'test'):
        ptxt = splits_dir / f'{name}.txt'
        if not ptxt.exists():
            print(f'Missing split file: {ptxt}')
            any_fail = True
            continue
        ids = read_txt_ids(ptxt)
        split_ids[name] = ids
        print(f'{name}: {len(ids)} entries')

    # overlap checks
    inter = False
    for a in ('train', 'val', 'test'):
        for b in ('train', 'val', 'test'):
            if a >= b:
                continue
            s = split_ids.get(a, set()) & split_ids.get(b, set())
            if s:
                print(f'Overlap between {a} and {b}: {len(s)}')
                inter = True
    if inter:
        any_fail = True
    else:
        print('No overlaps between splits (OK)')

    # existence checks and labels
    missing_files = 0
    missing_labels = 0
    total = 0
    for name, ids in split_ids.items():
        for idstr in ids:
            total += 1
            img = data_dir / 'train' / f"{idstr}.jpg"
            if not img.exists():
                missing_files += 1
            if idstr not in labels_ids:
                missing_labels += 1

    print(f'Total entries across splits: {total}')
    if missing_files:
        print(f'MISSING FILES under data/train/: {missing_files}')
        any_fail = True
    else:
        print('All split files exist under data/train/ (OK)')
    if missing_labels:
        print(f'MISSING LABELS in labels.csv: {missing_labels}')
        any_fail = True
    else:
        print('All ids present in labels.csv (OK)')

    if any_fail:
        print('Verification FAILED')
        sys.exit(3)
    print('Verification PASSED')


if __name__ == '__main__':
    main()
