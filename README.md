# Dog Breed Identification - CNN Project

This repository implements a CNN classifier for the Kaggle Dog Breed Identification challenge using either PyTorch or TensorFlow/Keras with TF Hub.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Configure Kaggle API:

```bash
pip install kaggle
# Either create ~/.kaggle/kaggle.json with your credentials
# or export credentials as environment variables:
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

3. Download dataset:

```bash
python -m src.download_data --competition dog-breed-identification --out-dir data
```

Quick summary (start here)

- Findings: a tuned EfficientNetV2B0 model trained here achieved val_acc ≈ 82% on the validation split. The ready-to-grade packet and artifacts live in `READY_FOR_GRADE.md` and `runs/keras_tuneB/`.
- To reproduce quickly: follow the steps in `READY_FOR_GRADE.md` — it contains exact commands to run training and evaluation and a checklist of files to submit.

4. Create splits:

```bash
python -m src.make_splits.py --data-dir data --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

## Training (Keras)

Train using a Keras `tf.keras.applications` backbone (EfficientNetV2B0 by default) with transfer learning and optional fine-tuning:

```bash
python -m src.keras_train \
  --data-dir data \
  --splits-path data/splits \
  --model efficientnetv2b0 \
  --img-size 224 \
  --epochs 15 \
  --warmup-epochs 1 \
  --fine-tune-last 20 \
  --batch-size 32 \
  --lr 0.001 \
  --output runs/keras_exp1 \
  --export-samples 25
```

This will save the following artifacts under the output directory (`runs/keras_exp1`):

- `best.keras` — Best model weights (ModelCheckpoint)
- `history.csv` and `history.png` — training/validation loss and accuracy
- `confusion_matrix.png` — confusion matrix on validation set
- `val_predictions.csv` — validation predictions (filename,true_label,pred_label,pred_conf)
- `metrics.json` — train/val metrics (best epoch)
- `sample_grid.png` — 5×5 labeled sample grid from validation
- `logs/` — TensorBoard logs

Notes:
- Augmentation is applied using Keras preprocessing layers (`RandomFlip`, `RandomRotation`, `RandomZoom`).
- Fine-tuning: use `--warmup-epochs` to train the head/base frozen, then `--fine-tune-last N` to unfreeze and fine-tune the last N layers of the backbone.

### macOS / Apple GPU (optional)

If you want to use Apple Silicon GPU acceleration on macOS, follow Apple's instructions and install the macOS-specific TensorFlow packages (these are provided by Apple and may require `tensorflow-macos` and `tensorflow-metal`). Example (may vary with macOS/TF versions):

```bash
# Install macOS-optimized TF (example; check official Apple/TF docs for latest guidance)
pip install -U pip
pip install "tensorflow-macos==2.16.*"
pip install "tensorflow-metal"
```

If you prefer the standard Linux/CPU packages or are running on a non-Apple GPU, the regular `tensorflow` wheel from PyPI (listed in `requirements.txt`) will be installed by `pip install -r requirements.txt`.

## Evaluation

Evaluate on validation set with sample grid:

```bash
python -m src.keras_evaluate \
  --data-dir data \
  --splits-path data/splits \
  --model runs/keras_mobilenet/best.keras \
  --split val \
  --export-samples 25 \
  --output runs/keras_mobilenet/eval
```

For test set:

```bash
python -m src.keras_evaluate \
  --data-dir data \
  --splits-path data/splits \
  --model runs/keras_mobilenet/best.keras \
  --split test \
  --output runs/keras_mobilenet/test
```

## Sample Prediction Grid

Generate a grid of sample predictions:

```bash
python -m src.make_sample_grid \
  --predictions-csv runs/keras_mobilenet/eval/val_predictions.csv \
  --data-dir data \
  --output runs/keras_mobilenet/samples \
  --samples 25
```

## Legacy PyTorch Implementation

The original PyTorch implementation remains available:

Export MPS fallback (macOS):
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Train:
```bash
python -m src.train --data-dir data --splits-path data/splits --epochs 20 --batch-size 64 --lr 3e-4 --output runs/pytorch_exp1
```

Evaluate:
```bash
python -m src.eval --data-dir data --splits-path data/splits --checkpoint runs/pytorch_exp1/best.pt
```

## Notes

- All paths are relative to the project root
- Experiments are logged in `experiments.csv`
- Raw images are not tracked in git (see `.gitignore`)
- Both Keras and PyTorch implementations use the same data splits
