# OcuScan AI — Labelling Convention
## docs/labelling_convention.md
**Version:** 2.0 | **Date:** June 2025 | **Status:** Active

---

## Purpose
This document defines the single-label assignment rules for all anterior segment images in the
OcuScan AI dataset. Because OCP, OCP Chronic, SJS, and Symblepharon can co-occur in the same
image, every image must receive exactly one class label. Use the rules below in priority order.

---

## Class Index Mapping (FIXED — do not change)

| idx | class_key       | Display Name                          |
|-----|-----------------|---------------------------------------|
| 0   | normal          | Normal                                |
| 1   | ocp             | OCP (Ocular Cicatricial Pemphigoid)   |
| 2   | ocp_chronic     | OCP Chronic                           |
| 3   | post_viral_ded  | Post-Viral DED (Dry Eye Disease)      |
| 4   | sjs             | SJS (Stevens-Johnson Syndrome)        |
| 5   | symblepharon    | Symblepharon                          |

---

## Labelling Rules (apply in this order)

### Rule 1 — Symblepharon override
If a **clear fibrous or vascular adhesion band** bridging palpebral to bulbar conjunctiva is the
**primary visual feature**, label as `symblepharon` — regardless of the underlying cause (OCP,
SJS, or other).

> _Rationale:_ The model needs to learn the sign of adhesion as a distinct class. Mixing adhesion
> into OCP/SJS adds intra-class variance without gain.

### Rule 2 — SJS (acute vs chronic)
- **Acute SJS**: pseudomembrane, conjunctival necrosis, eyelid erythema → label `sjs`
- **Chronic SJS**: keratinisation, trichiasis, foreshortening → label `sjs`
- **Exception**: if chronic SJS image shows a dominant adhesion band as the primary feature →
  label `symblepharon` per Rule 1

> _Note:_ SJS contains two sub-photo types (acute + chronic). This intentional intra-class
> variance is documented. Do not split into separate classes.

### Rule 3 — OCP Chronic vs OCP
- **OCP Chronic**: advanced forniceal loss, dense fibrous scarring, possible corneal involvement
  → label `ocp_chronic`
- **OCP (early)**: subconjunctival fibrosis only, early/mild forniceal foreshortening, no
  corneal involvement → label `ocp`
- **Ambiguous cases**: if you cannot distinguish OCP from OCP Chronic, flag the image in
  `labels.csv` with a note (see Flagging Protocol below). Do not guess.

> _Critical:_ OCP vs OCP Chronic is the hardest discrimination in the dataset. When in doubt,
> flag rather than assign.

### Rule 4 — Post-Viral DED
Conjunctival injection, reduced tear meniscus, possible punctate staining, no fibrosis,
no adhesion → label `post_viral_ded`

### Rule 5 — Normal
Clear anterior segment, no conjunctival injection, no fibrosis, no adhesion, no opacity
→ label `normal`

---

## Flagging Protocol
Ambiguous images must still be included in `labels.csv` with their best-guess label plus a
flag column:

```csv
filepath,class_key,class_idx,split,ambiguous_flag,ambiguous_note
dataset/ocp/img_047.jpg,ocp,1,train,1,"Could be ocp_chronic — forniceal loss borderline"
```

Flagged images:
- Are included in training (not discarded)
- Are excluded from per-class quality metrics
- Are listed in `EDA.ipynb` for manual review

---

## Co-occurrence Reference Table

| Scenario | Label As |
|---|---|
| Early conjunctival scarring, no gross adhesion | `ocp` |
| Advanced forniceal loss, dense fibrosis ± corneal involvement | `ocp_chronic` |
| Clear fibrous/vascular band bridging palpebral to bulbar conjunctiva | `symblepharon` |
| Acute SJS: pseudomembrane, conjunctival necrosis | `sjs` |
| Chronic SJS: keratinisation, trichiasis — no dominant adhesion band | `sjs` |
| Normal anterior segment | `normal` |
| Post-viral presentation, reduced tear meniscus, injection | `post_viral_ded` |

---

## labels.csv Schema

All images must be registered in `dataset/labels.csv`. One row per image.

| Column | Type | Description |
|---|---|---|
| `filepath` | string | Relative path from project root, e.g. `dataset/ocp/img_001.jpg` |
| `class_key` | string | One of the 6 class keys above |
| `class_idx` | int | Corresponding index 0–5 |
| `split` | string | `train`, `val`, or `test` — assigned by `src/dataset.py` split step |
| `ambiguous_flag` | int | 1 if reviewer was uncertain; 0 otherwise |
| `ambiguous_note` | string | Free-text note for flagged images; empty otherwise |

---

## Version History

| Version | Date | Change |
|---|---|---|
| 1.0 | May 2025 | Initial 4-class convention |
| 2.0 | June 2025 | Added OCP Chronic and Symblepharon; co-occurrence rules formalised |
