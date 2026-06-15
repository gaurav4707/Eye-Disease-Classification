# OcuScan AI — Labelling Convention
**docs/labelling_convention.md** | v2.0 | June 2025

This document defines the labelling rules for all images assigned to the six anterior segment
disease classes. It must be read before labelling any image and consulted whenever an ambiguous
case arises. All labelling decisions are final once `labels.csv` is committed.

---

## 1. Class Index Mapping

The following mapping is **fixed** and must not change between labelling, training, and inference.

| Idx | Key              | Display Name                          |
|-----|------------------|---------------------------------------|
| 0   | `normal`         | Normal                                |
| 1   | `ocp`            | OCP (Ocular Cicatricial Pemphigoid)   |
| 2   | `ocp_chronic`    | OCP Chronic                           |
| 3   | `post_viral_ded` | Post-Viral DED                        |
| 4   | `sjs`            | SJS (Stevens-Johnson Syndrome)        |
| 5   | `symblepharon`   | Symblepharon                          |

---

## 2. Per-Class Labelling Rules

### 2.1 `normal`
Label as **normal** when:
- The anterior segment shows a clear conjunctiva with no adhesion, scarring, or opacity.
- No active inflammation, injection, or fibrous band is visible.
- Tear meniscus appears normal; no epithelial irregularity evident.

**Do NOT label as normal if** any conjunctival scarring, forniceal changes, or adhesion is
present, even if subtle.

---

### 2.2 `ocp` — OCP (Ocular Cicatricial Pemphigoid)
Label as **ocp** when:
- The image shows **early to moderate** subconjunctival fibrosis.
- Forniceal foreshortening is **present but mild** — the fornix is reduced but not obliterated.
- There may be conjunctival symblepharon formation beginning, but no established fibrous bridge.
- No gross corneal involvement (pannus, opacity) is visible.

**Key discriminator vs `ocp_chronic`:** Fornix is reduced but still visible. Fibrosis is pale/
early. The palpebral conjunctiva still shows distinct surface texture.

---

### 2.3 `ocp_chronic` — OCP Chronic
Label as **ocp_chronic** when:
- The image shows **advanced** subconjunctival fibrosis with **dense, white/pale fibrosis**.
- Forniceal foreshortening is **severe or complete** — the inferior and/or superior fornix is
  largely obliterated.
- Corneal involvement may be present (pannus, peripheral opacity, vascularisation).
- Multiple symblepharon bands or a continuous cicatricial membrane may be visible.

**Key discriminator vs `ocp`:** Fibrosis is denser and whiter. Fornix largely gone.
If in doubt between `ocp` and `ocp_chronic`, choose the class that best matches the
**dominant visual feature** (extent of forniceal loss).

---

### 2.4 `post_viral_ded` — Post-Viral DED
Label as **post_viral_ded** when:
- The image shows conjunctival **injection** (redness) without adhesion or scarring.
- Reduced or absent tear meniscus is visible.
- Mild epithelial irregularity or punctate staining pattern may be present.
- The image was taken in the context of a known or suspected post-viral dry eye presentation.
- No fibrotic bands or symblepharon are visible.

---

### 2.5 `sjs` — SJS (Stevens-Johnson Syndrome)
The SJS class merges two sub-photo types that were treated separately in earlier versions.
Both are labelled `sjs`:

| Sub-type  | Visual Features                                                                 |
|-----------|---------------------------------------------------------------------------------|
| **Acute** | Pseudomembrane (grey/white conjunctival membrane), conjunctival necrosis,        |
|           | haemorrhage, severe injection.                                                  |
| **Chronic**| Keratinisation of the conjunctiva, trichiasis, metaplastic lashes, corneal      |
|           | vascularisation, severe dry eye, entropion.                                     |

**Exception — symblepharon priority rule:** If the dominant visual feature of a chronic SJS
image is the **adhesion band itself** (a clearly defined fibrous bridge between palpebral
and bulbar conjunctiva), label it as `symblepharon` instead of `sjs`.

---

### 2.6 `symblepharon` — Symblepharon
Label as **symblepharon** when:
- A **clearly visible fibrous or vascular band** bridges the palpebral conjunctiva to the
  bulbar conjunctiva.
- The adhesion band is the **primary and dominant visual feature** of the image.
- The band may be partial (one segment) or total (spanning the entire lid-globe junction).
- The underlying cause may be OCP, SJS, chemical/thermal injury, or other cicatrising disease —
  **the cause does not affect this labelling rule**.

**Do NOT label as symblepharon if:**
- The adhesion is not clearly identifiable as a bridge between lid and globe.
- The dominant feature is fibrosis, keratinisation, or conjunctival scarring without a discrete
  adhesion band → use `ocp`, `ocp_chronic`, or `sjs` as appropriate.

---

## 3. Ambiguous Case Decision Tree

```
Is a clear fibrous adhesion band visible between lid and globe?
├── YES → label symblepharon (regardless of presumed cause)
└── NO
    ├── Is there pseudomembrane, necrosis, or keratinisation with known SJS context?
    │   └── YES → label sjs
    ├── Is there subconjunctival fibrosis with forniceal foreshortening?
    │   ├── Advanced / dense fibrosis, fornix largely gone → ocp_chronic
    │   └── Early / mild fibrosis, fornix still partially visible → ocp
    ├── Is there conjunctival injection + reduced tear meniscus, post-viral context?
    │   └── YES → post_viral_ded
    └── No pathological features → normal
```

---

## 4. Co-occurrence Rules

Because OCP, SJS, and Symblepharon can co-occur in the same eye, the following
priority order governs labelling when **multiple features are present**:

1. **Symblepharon** — if a discrete adhesion band is the primary feature, always wins.
2. **SJS** — if acute pseudomembrane or necrosis is present.
3. **OCP Chronic** — if advanced fibrosis dominates over other features.
4. **OCP** — early fibrosis without discrete adhesion band.
5. **Post-Viral DED** — injection, DED signs, no fibrosis.
6. **Normal** — no pathological features.

---

## 5. Images to Exclude

The following images should be **excluded from the dataset entirely** (not assigned any class):

- Images where the anterior segment is not the primary subject (fundus images, OCT cross-sections).
- Images that are overexposed, completely dark, or show severe motion blur obscuring all features.
- Images of the external face/eyelid only without visible conjunctiva or cornea.
- Duplicate images (identical pixel content) — keep only one copy.
- Heavily watermarked images where the watermark obscures pathological features.

---

## 6. Audit Trail

All labelling decisions for ambiguous cases must be noted in `dataset/labelling_log.csv` with
the following columns:

| Column        | Description                                         |
|---------------|-----------------------------------------------------|
| `filepath`    | Relative path to the image                          |
| `assigned_class` | Final class_key assigned                         |
| `ambiguity`   | Brief description of why the image was ambiguous    |
| `decision`    | The decision rationale                              |
| `reviewer`    | Initials of the person who made the final decision  |
| `date`        | ISO 8601 date of decision                           |

This log is committed alongside `labels.csv` and reviewed during Week 3 evaluation
to identify systematic labelling errors that may explain model confusion patterns.

---

## 7. Version History

| Version | Date       | Change                                           |
|---------|------------|--------------------------------------------------|
| v1.0    | 2025-05-01 | Initial convention for 4-class dataset           |
| v2.0    | 2025-06-01 | Added Symblepharon as 6th class; SJS sub-types   |
|         |            | merged; co-occurrence priority rules formalised  |
