# OcuScan AI
**Anterior Segment Ocular Disease Detection**  
v2.0 · 6-Class EfficientNetB0 Classifier · Streamlit App · Docker

---

## Classes

| idx | class_key | Display Name | ICD-10 |
|-----|-----------|--------------|--------|
| 0 | `normal` | Normal | Z01.01 |
| 1 | `ocp` | OCP (Ocular Cicatricial Pemphigoid) | L12.1 |
| 2 | `ocp_chronic` | OCP Chronic | L12.1 |
| 3 | `post_viral_ded` | Post-Viral DED | H04.12 |
| 4 | `sjs` | SJS (Stevens-Johnson Syndrome) | L51.1 |
| 5 | `symblepharon` | Symblepharon | H11.23 |

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
│   ├── EDA.ipynb            ← Phase 1 EDA
│   ├── training.ipynb       ← Phase 2 training curves
│   └── evaluation.ipynb     ← Phase 3 metrics
├── src/
│   ├── dataset.py           ← EyeDiseaseDataset, DataLoader factory
│   ├── augmentation.py      ← Albumentations pipeline (9 transforms)
│   ├── model.py             ← EfficientNetB0 + Grad-CAM
│   ├── train.py             ← Training loop + MLflow
│   ├── evaluate.py          ← All metrics, confusion matrix, AUC-ROC
│   ├── predict.py           ← Single-image inference → PredictionResult
│   ├── db.py                ← SQLite schema and helpers
│   └── utils.py             ← Image loading, PredictionResult, temp scaling
├── models/                  ← .pt checkpoints
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
# Create conda environment
conda create -n ocuscan python=3.10 -y
conda activate ocuscan

# Install dependencies
pip install -r requirements.txt

# Test environment
python src/dataset.py
```

---

## Dataset Setup

1. Add anterior segment images to `dataset/<class_key>/` folders.
2. Follow the labelling rules in `docs/labelling_convention.md` exactly.
3. Run `python src/dataset.py` to build `labels.csv` and apply stratified split.
4. Open `notebooks/EDA.ipynb` and run all cells to validate the dataset.

---

## Training

```bash
# Phase 1 — train head only (frozen backbone)
python src/train.py --phase 1

# Phase 2 — fine-tune top 3 EfficientNet blocks
python src/train.py --phase 2

# View training curves in MLflow
mlflow ui
```

---

## Evaluation

```bash
python src/evaluate.py
# Results saved to results/metrics.json, results/confusion_matrix.png, etc.
```

---

## Running the App

```bash
streamlit run app/streamlit_app.py
```

### Docker

```bash
docker build -t ocuscan-ai .
docker run -p 8501:8501 -v $(pwd)/models:/app/models -v $(pwd)/ocuscan.db:/app/ocuscan.db ocuscan-ai
```

App accessible at [http://localhost:8501](http://localhost:8501)

---

## Implementation Plan

| Phase | Week | Title | Status |
|-------|------|-------|--------|
| Ph 1 | Week 1 | Foundation & Data | ✅ In Progress |
| Ph 2 | Week 2 | Model Training | ⏳ Pending |
| Ph 3 | Week 3 | Evaluation & Baseline | ⏳ Pending |
| Ph 4 | Week 4 | Application & Deployment | ⏳ Pending |

---

## Medical Disclaimer

> OcuScan AI is a research tool intended to assist clinicians.  
> It is **not** a diagnostic device and must not be used as the sole basis for clinical decisions.  
> All outputs require review by a qualified ophthalmologist.  
> Confidence scores below 60% are automatically flagged for human review.

---

## Dataset Provenance

All images are proprietary clinical photographs from [institution]. No public dataset was used.
Augmentation is the primary data expansion strategy. No augmented images appear in validation or
test sets.

---

*OcuScan AI v2.0 — June 2025 — Confidential*
