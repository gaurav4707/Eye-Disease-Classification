# OcuScan AI
**Anterior Segment Ocular Disease Detection**  
v2.0 · 6-Class EfficientNetB0 Classifier · Streamlit App · Docker

---

## Classes

| idx | class_key | Display Name | ICD-10 | Severity |
|-----|-----------|--------------|--------|----------|
| 0 | `normal` | Normal | Z01.01 | None |
| 1 | `ocp` | OCP (Ocular Cicatricial Pemphigoid) | L12.1 | Medium |
| 2 | `ocp_chronic` | OCP Chronic | L12.1 | High |
| 3 | `post_viral_ded` | Post-Viral DED | H04.12 | Low |
| 4 | `sjs` | SJS (Stevens-Johnson Syndrome) | L51.1 | High |
| 5 | `symblepharon` | Symblepharon | H11.23 | High |

---

## Project Structure

```
ocuscan_ai/
├── dataset/
│   ├── normal/              ← anterior segment images
│   ├── ocp/
│   ├── ocp_chronic/
│   ├── post_viral_ded/
│   ├── sjs/
│   ├── symblepharon/
│   └── labels.csv           ← master CSV (filepath, class_key, class_idx, split)
├── notebooks/
│   ├── EDA.ipynb            ← Phase 1: class distribution, mean images, histograms
│   ├── training.ipynb       ← Phase 2: loss curves, per-class F1, OCP analysis
│   └── evaluation.ipynb     ← Phase 3: full metrics, AUC-ROC, calibration
├── src/
│   ├── dataset.py           ← EyeDiseaseDataset, DataLoader factory, stratified split
│   ├── augmentation.py      ← Albumentations pipeline (9 transforms, training only)
│   ├── model.py             ← EfficientNetB0 + Grad-CAM, freeze/unfreeze helpers
│   ├── train.py             ← Two-phase training loop, MixUp, EMA, WeightedSampler
│   ├── evaluate.py          ← All metrics, confusion matrix, AUC-ROC, SVM baseline
│   ├── predict.py           ← Single-image inference + TTA → PredictionResult
│   ├── db.py                ← SQLite schema, seed data, CRUD helpers
│   └── utils.py             ← Image loading, PredictionResult, temperature scaling
├── models/                  ← .pt checkpoints (phase1_best.pt, phase2_best.pt)
├── results/                 ← metrics.json, figures, Grad-CAM outputs
├── app/
│   └── streamlit_app.py     ← Web application (4 screens)
├── docs/
│   └── labelling_convention.md
├── requirements.txt
└── Dockerfile
```

---

## Setup

```bash
conda create -n ocuscan python=3.10 -y
conda activate ocuscan
pip install -r requirements.txt

# Verify environment
python src/dataset.py
```

---

## Dataset Setup

1. Add anterior segment images to `dataset/<class_key>/` folders.
2. Follow the labelling rules in `docs/labelling_convention.md` exactly.
3. Run `python src/dataset.py` to build `labels.csv` and apply the stratified 70/15/15 split.
4. Open `notebooks/EDA.ipynb` and run all cells to validate dataset quality.

> If `dataset.py` fails to generate `labels.csv`, use `src/rebuild_labels.py` as a fallback.

---

## Training

### Phase 1 — Frozen backbone, head only

```bash
python src/train.py --phase 1
```

- AdamW: `lr=1e-3`, `weight_decay=1e-3`
- Max 40 epochs, early stopping `patience=7` on `val_macro_f1`
- MixUp `α=0.2`, label smoothing `ε=0.1`, WeightedRandomSampler, EMA `decay=0.99`
- Saves best checkpoint to `models/phase1_best.pt`

### Phase 2 — Fine-tune top 3 EfficientNet blocks

```bash
python src/train.py --phase 2
```

- Loads `models/phase1_best.pt` automatically
- AdamW: `lr=5e-6`, `weight_decay=1e-4`
- Max 25 epochs, early stopping `patience=7`
- Saves best checkpoint to `models/phase2_best.pt`

### Options

```bash
python src/train.py --phase 1 --image-size 160   # progressive resizing
python src/train.py --phase 1 --no-mixup          # ablation: disable MixUp
python src/train.py --phase 2 --resume models/phase1_best.pt
```

### View training curves

```bash
mlflow ui
# open http://localhost:5000
```

Or open `notebooks/training.ipynb` for loss curves, per-class F1 bars, and the OCP/OCP Chronic discrimination plot.

---

## Evaluation

```bash
python src/evaluate.py
```

Outputs saved to `results/`:
- `metrics.json` — accuracy, macro F1, per-class P/R/F1, SJS recall, OCP confusion rate
- `confusion_matrix.png` — 6×6 heatmap with OCP/OCP Chronic highlighted
- `roc_curves.png` — per-class AUC-ROC
- `calibration_curve.png` — reliability diagram (apply temperature scaling if overconfident)
- `cv_results.json` — 5-fold CV mean ± std Macro-F1
- `svm_baseline.json` — frozen EfficientNetB0 features + SVM(RBF) comparison

---

## Inference

```python
from src.predict import Predictor

predictor = Predictor()   # auto-loads phase2_best.pt
result = predictor.predict('path/to/image.jpg', save_gradcam=True)

print(result.predicted_display)   # e.g. "OCP (Ocular Cicatricial Pemphigoid)"
print(f"{result.confidence:.1%}") # e.g. "84.3%"
print(result.flagged)             # True if confidence < 60%
```

CLI:

```bash
python src/predict.py --image path/to/image.jpg
python src/predict.py --image path/to/image.jpg --gradcam
python src/predict.py --image path/to/image.jpg --no-tta   # single-pass, faster
```

---

## Running the App

```bash
streamlit run app/streamlit_app.py
```

### Docker

```bash
docker build -t ocuscan-ai .
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/ocuscan.db:/app/ocuscan.db \
  ocuscan-ai
```

App accessible at [http://localhost:8501](http://localhost:8501)

---

## Implementation Plan

| Phase | Week | Title | Status |
|-------|------|-------|--------|
| Ph 1 | Week 1 | Foundation & Data | ✅ Complete |
| Ph 2 | Week 2 | Model Training | ✅ Complete |
| Ph 3 | Week 3 | Evaluation & Baseline | ⏳ Pending |
| Ph 4 | Week 4 | Application & Deployment | ⏳ Pending |

### Phase 2 deliverables

| Day | File | Status |
|-----|------|--------|
| Day 8 | `src/model.py` — EfficientNetB0 + Grad-CAM | ✅ |
| Day 9 | `src/train.py` — training loop, MixUp, EMA, sampler | ✅ |
| Day 10–11 | Phase 1 + Phase 2 training runs | ⏳ Run locally |
| Day 12 | MLflow review — OCP vs OCP Chronic per-class F1 | ⏳ After training |
| Day 13 | `notebooks/training.ipynb` — curves + comparison | ✅ |
| Day 14 | `src/predict.py` — inference + TTA + Grad-CAM | ✅ |

---

## Training Recipe (anti-overfitting, 41 images)

| Technique | Setting | Where |
|-----------|---------|-------|
| Label smoothing | ε = 0.1 | `train.py` CrossEntropyLoss |
| Weight decay | 1e-3 (Ph1) / 1e-4 (Ph2) | `train.py` AdamW |
| Dropout | 0.5 / 0.3 | `model.py` classifier head |
| WeightedRandomSampler | Inverse-frequency | `train.py` DataLoader |
| MixUp | α = 0.2 | `train.py` training loop |
| Model EMA | decay = 0.99 | `train.py` (used for val + checkpoint) |
| Early stopping | patience = 7 | `train.py` on `val_macro_f1` |
| TTA at inference | original + hflip | `predict.py` |
| Temperature scaling | fit on val logits | `utils.py` TemperatureScaler |
| 5-fold CV | StratifiedKFold | `evaluate.py` Day 18 |

---

## Key Metrics to Monitor

| Metric | Why it matters |
|--------|----------------|
| `val_macro_f1` | Primary metric — equal weight across all 6 classes |
| `val_sjs_recall` | Clinical priority — missing SJS is highest-stakes error |
| `val_ocp_f1` vs `val_ocp_chronic_f1` | Hardest discrimination; similar appearance |
| `val_symblepharon_f1` | Sign-detection reliability |
| Train acc vs val F1 gap | Overfitting diagnostic — flag if gap > 0.4 |

---

## Overfitting Warning Signs

With only 41 images, monitor for these patterns during training:

```
Train Accuracy → 100%  +  Val Macro-F1 stagnates below 0.5  →  memorisation
Train Loss → 0          +  Val Loss increasing                →  stop immediately
```

If this occurs: reduce MixUp `α`, increase `weight_decay`, or check Grad-CAM — if it highlights eyelashes or skin instead of the fornix or adhesion bands, the model is learning shortcuts.

---

## Medical Disclaimer

> OcuScan AI is a research tool intended to assist clinicians.  
> It is **not** a diagnostic device and must not be used as the sole basis for clinical decisions.  
> All outputs require review by a qualified ophthalmologist.  
> Confidence scores below 60% are automatically flagged for human review.

---

## Dataset Provenance

All images are proprietary clinical photographs. No public dataset was used.
Augmentation is the primary data expansion strategy (9 medically realistic transforms).
No augmented images appear in validation or test sets.

---

*OcuScan AI v2.0 · June 2025 · Confidential*
