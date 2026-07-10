# OcuScan AI
**Anterior Segment Ocular Disease Detection**  
v1.0.0 · 6-Class EfficientNetB0 Classifier · Streamlit App · Docker

---

## Overview

OcuScan AI is an AI-assisted anterior segment eye disease classification system. It analyses close-up slit-lamp or external photographs of the human eye and outputs a predicted diagnosis from six clinically defined categories, accompanied by a per-class confidence score and a Grad-CAM explainability heatmap.

The system targets **rare and serious ocular surface conditions** — not the common conditions found in widely available public datasets. All six disease classes are anterior segment conditions, distinguishing this project from fundus/retinal AI tools.

> ⚕ **Medical Disclaimer:** OcuScan AI is a research and educational tool. It is **NOT** a certified medical device. All outputs require review by a qualified ophthalmologist. Not approved for diagnostic use.

---

## Disease Classes

| Idx | Key | Display Name | ICD-10 | Severity | Notes |
|-----|-----|-------------|--------|----------|-------|
| 0 | `normal` | Normal | Z01.01 | None | Baseline healthy anterior segment |
| 1 | `ocp` | OCP (Ocular Cicatricial Pemphigoid) | H10.40 | High | Active/general stage; autoimmune |
| 2 | `ocp_chronic` | OCP Chronic | H10.40 | High | Advanced fibrosis, significant forniceal loss |
| 3 | `post_viral_ded` | Post-Viral DED | H04.123 | Medium | Post-viral dry eye disease |
| 4 | `sjs` | SJS (Stevens-Johnson Syndrome) | L51.1 | High | Acute + chronic sub-types merged in v1.0 |
| 5 | `symblepharon` | Symblepharon | H11.231 | High | **Sign class** — fibrous adhesion, not primary disease |

### Important Clinical Notes

> ⚠ **Symblepharon** is a structural **sign** (fibrous adhesion between palpebral and bulbar conjunctiva), not a primary disease. It is a sequela of SJS, OCP, chemical burns, or chronic inflammation. The model detects its **presence** only — the underlying cause must be investigated by a specialist.

> ⚠ **SJS class** merges acute (pseudomembrane, conjunctival necrosis) and chronic (keratinisation, trichiasis) sub-photo types in v1.0. Splitting into SJS-Acute / SJS-Chronic is planned for v2.0.

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
│   ├── predict.py           ← Single-image inference + PredictionResult dataclass
│   ├── gradcam.py           ← Grad-CAM via backward hooks, batch heatmap generation
│   ├── db.py                ← SQLite schema, seed data, CRUD helpers
│   └── utils.py             ← Image loading, PredictionResult, temperature scaling
├── app/
│   └── streamlit_app.py     ← Web application (4 screens, clinical UI)
├── models/
│   ├── phase1_best.pt       ← Phase 1 checkpoint (frozen backbone)
│   └── phase2_best.pt       ← Phase 2 checkpoint (fine-tuned) ← used by app
├── results/
│   ├── metrics.json         ← Full evaluation metrics
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── calibration_curve.png
│   ├── cv_results.json
│   ├── svm_baseline.json
│   └── gradcam/             ← Grad-CAM sample heatmaps
├── docker/
│   └── streamlit_config.toml
├── docs/
│   └── labelling_convention.md
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Environment Setup

```bash
conda create -n ocuscan python=3.10 -y
conda activate ocuscan
pip install -r requirements.txt

# Verify environment
python src/dataset.py
```

### 2. Dataset Setup

```bash
# Add anterior segment images to dataset/<class_key>/ folders
# Naming convention: dataset/ocp/ocp_01.jpg, dataset/sjs/sjs_01.jpg, etc.

# Build labels.csv and apply stratified 70/15/15 split
python src/dataset.py

# Run EDA notebook to validate dataset quality
jupyter notebook notebooks/EDA.ipynb
```

### 3. Training

```bash
# Phase 1 — Frozen backbone, head only (max 40 epochs)
python src/train.py --phase 1

# Phase 2 — Fine-tune top 3 EfficientNet blocks (max 25 epochs)
python src/train.py --phase 2

# Monitor training
mlflow ui  # → http://localhost:5000
```

### 4. Evaluation

```bash
python src/evaluate.py
# Saves: results/metrics.json, confusion_matrix.png, roc_curves.png, calibration_curve.png

# Optional: 5-fold CV + SVM baseline
python src/evaluate.py --cv --svm --temperature-scaling
```

### 5. Run the App

```bash
streamlit run app/streamlit_app.py
# → http://localhost:8501
```

---

## Streamlit Application — 4 Screens

### Screen 1: Home / Upload
- Drag-and-drop or browse file uploader (JPEG, PNG, WEBP, max 10 MB)
- Image preview with dimensions and brightness check
- **Analyse Image** primary CTA (disabled until image loaded)
- Demo mode when no checkpoint exists — cycles through all 6 class results

### Screen 2: Results
- **Predicted class card** — class name, ICD-10 code, severity pill, confidence badge (colour-coded by level)
- **Mandatory class-specific warning banners** for all 6 classes (see table below)
- Low-confidence alert when prediction < 60%
- **6-class horizontal confidence bar chart** — predicted class in teal, OCP/OCP Chronic visually grouped
- **Grad-CAM panel** — original image + attention heatmap side-by-side
- **Clinical notes** — description and referral recommendation from database
- **Top-3 differential** — ranked confidence cards
- Export actions: Download PDF Report | Export CSV

### Screen 3: History
- Session log table: Timestamp | Filename | Predicted Class | Confidence | Flagged
- Per-row replay (View button restores full Results screen)
- **Export All CSV** — all session predictions with full confidence breakdown
- **Clear History** button

### Screen 4: About / Help
- Project overview and clinical motivation
- 6-class reference table with ICD-10 codes and severity
- SJS sub-type note and Symblepharon sign-detection explanation
- How to take a good anterior segment photo guidance
- Model architecture, training approach, dataset provenance
- Full medical disclaimer

### Class-Specific Warning Banners

| Class | Colour | Icon | Trigger |
|-------|--------|------|---------|
| Symblepharon | Red `#DC2626` | ⚠ | Always — sign class, underlying cause unknown |
| SJS | Amber `#D97706` | ⚡ | Always — potential emergency |
| OCP Chronic | Red `#DC2626` | ⚠ | Always — advanced disease, urgent referral |
| OCP | Amber `#D97706` | ℹ | Always — early referral recommended |
| Post-Viral DED | Blue `#1A73E8` | ℹ | Always — monitor, manage with lubricants |
| Normal | Green `#16A34A` | ✓ | Always — with caveat for persistent symptoms |

---

## Docker

### Build & Run

```bash
# Build
docker build -t ocuscan-ai .

# Run with volume mounts for model and database persistence
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/ocuscan.db:/app/ocuscan.db \
  ocuscan-ai
```

App accessible at [http://localhost:8501](http://localhost:8501)

### Volume Mounts

| Mount | Purpose |
|-------|---------|
| `$(pwd)/models:/app/models` | Model checkpoints (phase1_best.pt, phase2_best.pt) |
| `$(pwd)/ocuscan.db:/app/ocuscan.db` | SQLite prediction history and disease class reference data |

---

## Training Recipe (Anti-Overfitting — 41 images)

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
| 5-fold CV | StratifiedKFold | `evaluate.py` |

---

## Key Metrics to Monitor

| Metric | Target | Why |
|--------|--------|-----|
| `val_macro_f1` | ≥ 0.75 | Primary — equal weight across all 6 classes |
| `val_sjs_recall` | ≥ 0.80 | Clinical priority — missing SJS is highest-stakes error |
| `val_ocp_f1` vs `val_ocp_chronic_f1` | ≥ 0.70 each | Hardest discrimination; similar appearance |
| `val_symblepharon_f1` | ≥ 0.80 | Sign-detection reliability |
| AUC-ROC (macro) | ≥ 0.88 | Standard clinical AI benchmark |

---

## CLI Inference

```python
from src.predict import Predictor

# Single image
result = predict(
    image=Image.open("path/to/image.jpg"),
    checkpoint_path="models/phase2_best.pt",
    generate_gradcam=True,
)

print(result.predicted_display)   # e.g. "OCP Chronic"
print(f"{result.confidence:.1%}") # e.g. "81.3%"
print(result.flagged)             # True if confidence < 60%
print(result.is_emergency_class)  # True for SJS
print(result.is_sign_class)       # True for Symblepharon
```

```bash
# CLI
python src/predict.py --image path/to/image.jpg
python src/predict.py --image path/to/image.jpg --gradcam
python src/predict.py  # 6-class smoke test using dataset/ folder
```

---

## Implementation Status

| Phase | Week | Title | Status |
|-------|------|-------|--------|
| Ph 1 | Week 1 | Foundation & Data | ✅ Complete |
| Ph 2 | Week 2 | Model Training | ✅ Complete |
| Ph 3 | Week 3 | Evaluation & Baseline | ✅ Complete |
| Ph 4 | Week 4 | Application & Deployment | ✅ Complete |

### Phase 4 Deliverables

| Day | File | Status |
|-----|------|--------|
| Day 22 | `app/streamlit_app.py` — skeleton, routing, DB init | ✅ |
| Day 23 | Upload + inference wired to `predict.py` | ✅ |
| Day 24 | Results screen — class card, confidence chart, Grad-CAM | ✅ |
| Day 25 | Class-specific banners — all 6 classes, correct colours | ✅ |
| Day 26 | History screen — table, replay, CSV/PDF export | ✅ |
| Day 27 | `Dockerfile` — build, run, volume mounts verified | ✅ |
| Day 28 | README — complete setup, screenshots, disclaimer | ✅ |

---

## Labelling Convention

Because OCP, OCP Chronic, SJS, and Symblepharon can co-occur in the same image:

| Scenario | Label As |
|----------|---------|
| Early conjunctival scarring, no gross adhesion | `ocp` |
| Advanced forniceal loss, dense fibrosis | `ocp_chronic` |
| Clear fibrous/vascular band bridging palpebral to bulbar | `symblepharon` |
| Acute SJS: pseudomembrane, conjunctival necrosis | `sjs` |
| Chronic SJS with prominent adhesion as primary feature | `symblepharon` |
| Normal anterior segment | `normal` |
| Post-viral, reduced tear meniscus, injection | `post_viral_ded` |

Full rules in `docs/labelling_convention.md`.

---

## Overfitting Warning Signs

```
Train Accuracy → 100%  +  Val Macro-F1 stagnates below 0.5  →  memorisation
Train Loss → 0          +  Val Loss increasing                →  stop immediately
```

If Grad-CAM highlights eyelashes or skin instead of the fornix or adhesion bands — the model is learning shortcuts, not clinical features.

---

*OcuScan AI v1.0.0 · June 2025 · Confidential · For research and educational use only*
