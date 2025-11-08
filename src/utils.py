from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import torch
import numpy as np


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def save_checkpoint(path: str, state: Dict[str, Any]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location=None) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def get_device():
    # prefer CUDA if available, otherwise MPS on macOS, otherwise CPU
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None:
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    return torch.device('cpu')
"""Small utility helpers for dataset scripts."""
import os
import subprocess
import sys
import csv
from pathlib import Path
from typing import List


def ensure_dir(path: str) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def run_cmd(cmd: List[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    try:
        if capture_output:
            return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        else:
            return subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
        if capture_output:
            print(e.stdout, file=sys.stderr)
            print(e.stderr, file=sys.stderr)
        raise


def read_labels_csv(path: str) -> List[dict]:
    rows = []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows


def write_csv(path: str, rows: List[dict], fieldnames: List[str]):
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def safe_symlink_or_copy(src: str, dst: str):
    """Attempt to create a symlink; if not possible, copy the file."""
    src_p = Path(src)
    dst_p = Path(dst)
    ensure_dir(str(dst_p.parent))
    try:
        if dst_p.exists():
            return
        # try creating a symlink
        dst_p.symlink_to(src_p.resolve())
    except Exception:
        # fallback to copying
        from shutil import copy2

        copy2(src_p, dst_p)
"""Small utility helpers for dataset scripts."""
import os
import subprocess
import sys
import csv
from pathlib import Path
from typing import List


def ensure_dir(path: str) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def run_cmd(cmd: List[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    try:
        if capture_output:
            return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        else:
            return subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}", file=sys.stderr)
        if capture_output:
            print(e.stdout, file=sys.stderr)
            print(e.stderr, file=sys.stderr)
        raise


def read_labels_csv(path: str) -> List[dict]:
    rows = []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows


def write_csv(path: str, rows: List[dict], fieldnames: List[str]):
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def safe_symlink_or_copy(src: str, dst: str):
    """Attempt to create a relative symlink; if not possible, copy the file."""
    src_p = Path(src)
    dst_p = Path(dst)
    ensure_dir(str(dst_p.parent))
    try:
        if dst_p.exists():
            return
        # try creating a symlink
        dst_p.symlink_to(src_p.resolve())
    except Exception:
        # fallback to copying
        from shutil import copy2

        copy2(src_p, dst_p)
