# READY FOR GRADE

Selected best run: `runs/keras_tuneB`
- model: efficientnetv2b0
- img_size: 299
- epochs: 8 (warmup 2 + fine-tune)
- fine-tune-last: 200
- val_acc: 0.8245270848274231 (82.45%)
- baseline (random) = 1/120 = 0.833%
- Verdict: good enough (>=10%), excellent result for quick pass

## Requirement → Evidence → Status
- Keras + transfer learning → `src/keras_train.py` uses `tf.keras.applications` (EffNet/MobileNet) → OK
- Fine-tune last-N after warmup → `--warmup-epochs`, `--fine-tune-last` implemented in `src/keras_train.py` → OK
- tf.data + Keras preprocessing → `make_dataset_from_split()` applies `preprocess_input` (backbone-specific), uses `tf.data` with AUTOTUNE and prefetch → OK
- Experiments logging → `experiments.csv` appended per run → OK
- Artifacts saved → `runs/keras_tuneB/` contains `best.keras`, `best.h5`, `history.csv`, `history.png`, `metrics.json`, `val_predictions.csv`, `confusion_matrix.png`, `sample_grid.png` → OK
- Test predictions and 5×5 grid → `runs/keras_tuneB/eval_test/test_predictions.csv`, `sample_preds_test.png` → OK
- README updated with usage and macOS notes → `README.md` → OK
- Smoke script → `scripts/smoke_keras.sh` updated to use `best.keras` and passes → OK

## How to reproduce (minimal)
1. Install deps:

```bash
pip install -r requirements.txt
```

2. Download data (Kaggle):

```bash
python -m src.download_data --competition dog-breed-identification --out-dir data
```

3. (Optional) regenerate splits if desired:

```bash
python -m src.make_splits --data-dir data --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

4. Run evaluation on the saved best run (already produced):

```bash
python -m src.keras_evaluate --data-dir data --splits-path data/splits --split test --checkpoint runs/keras_tuneB --img-size 299 --batch-size 32 --output runs/keras_tuneB/eval_test --export-samples 25
```

## Files to submit
- `runs/keras_tuneB/best.keras` (best model)
- `runs/keras_tuneB/history.png`
- `runs/keras_tuneB/confusion_matrix.png`
- `runs/keras_tuneB/metrics.json`
- `runs/keras_tuneB/val_predictions.csv`
- `runs/keras_tuneB/sample_grid.png`
- `runs/keras_tuneB/eval_test/test_predictions.csv`
- `runs/keras_tuneB/eval_test/sample_preds_test.png`
- `experiments.csv`
- `README.md`
- `src/` directory (code)
- `COMPLIANCE_CHECKLIST.md`

## Notes / Assumptions
- I regenerated `data/splits/test.txt` from files in `data/test` so evaluation won't fail due to missing files.
- The training process now saves `best.keras` (native format) and also writes a `best.h5` legacy copy for compatibility.
- Preprocessing uses backbone-specific `preprocess_input` in the tf.data pipeline (EffNetV2 / MobileNetV2).


*** END OF READY_FOR_GRADE.md
