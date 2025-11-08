# Compliance checklist — Deep Learning using Keras (CNN Project 1)

Requirement | Implemented artifact / flag | Command to produce | Status
---|---:|---|:---
Use Keras with transfer learning | `--model efficientnetv2b0` (tf.keras.applications) | `python -m src.keras_train --model efficientnetv2b0 ...` | [x]
Fine-tune last N layers after warmup | `--warmup-epochs`, `--fine-tune-last` flags; deterministic unfreeze of last N layers | see train command below | [x]
Use tf.data + Keras preprocessing layers | `tf.data` used; RandomFlip/RandomRotation/RandomZoom applied in model input | training run | [x]
Log experiments/hparams | `experiments.csv` appended with run details | run training | [x]
Save best.keras | `runs/<name>/best.keras` via ModelCheckpoint | training run | [x]
Save history.csv and history.png | `history.csv` (CSVLogger) and `history.png` (plot_history) | training run | [x]
Save metrics.json (best epoch) | `metrics.json` in run dir | training run | [x]
Save {split}_predictions.csv | `val_predictions.csv` and test predictions via evaluate | training/eval run | [x]
Save confusion_matrix.png (val) | `confusion_matrix.png` in run dir | training/eval run | [x]
Save 5x5 labeled grid (25 images) | `sample_grid.png` (training) and `sample_preds_{split}.png` (eval) | training/eval run | [x]
README: install/run/macOS note | `README.md` updated with Keras examples and macOS TF/Metal note | n/a | [x]


## Example commands

Smoke run (1 epoch):

```bash
bash scripts/smoke_keras.sh
```

Short real run (3 epochs):

```bash
python -m src.keras_train \
  --data-dir data \
  --splits-path data/splits \
  --model efficientnetv2b0 \
  --img-size 224 \
  --epochs 3 \
  --warmup-epochs 1 \
  --fine-tune-last 20 \
  --batch-size 32 \
  --lr 3e-4 \
  --output runs/keras_exp1 \
  --export-samples 25
```

Evaluate on test:

```bash
python -m src.keras_evaluate \
  --data-dir data \
  --splits-path data/splits \
  --checkpoint runs/keras_exp1/best.keras \
  --split test \
  --img-size 224 \
  --batch-size 32 \
  --output runs/keras_exp1/eval_test \
  --export-samples 25
```
