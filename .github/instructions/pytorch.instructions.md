---
description: "Use when working with PyTorch model development, training logic, data augmentation, inference, or MLflow tracking in the ocuscan project."
applyTo:
  - "src/**/*.py"
---

## PyTorch & Training Conventions

- **Model Architecture**: We use ImageNet-pretrained `EfficientNetB0` via the `timm` library. The classification head is custom for 6 classes (`normal`, `ocp`, `ocp_chronic`, `post_viral_ded`, `sjs`, `symblepharon`).
- **Anti-Overfitting Arsenal**: Do not remove anti-overfitting mechanisms due to small dataset size: MixUp (α=0.2), EMA (Exponential Moving Average via `ModelEmaV2`), Label Smoothing (ε=0.1), and `WeightedRandomSampler` for class imbalance.
- **Two-Phase Training Strategy**:
  - **Phase 1 (`--phase 1`)**: Promotes head-only training with frozen backbone (`AdamW`, lr=1e-3).
  - **Phase 2 (`--phase 2`)**: Fine-tunes the top 3 EfficientNet blocks + head (`AdamW`, lr=2e-5).
- **MLflow Tracking Dashboard**: `mlruns/` directories store parameters, metrics, and models. Always use `mlflow` context managers in training/eval loops.
- **Metric Priority**: The key validation metric and the one used for Early Stopping is **Validation Macro-F1**, NOT Accuracy, since classes are imbalanced.

## Dataset and Labelling
- See [docs/labelling_convention.md](../../docs/labelling_convention.md) for strict dataset labelling rules.
- **Never modify** `dataset/labels.csv` manually. Use `src/dataset.py` or fallback `src/rebuild_labels.py` if missing.
