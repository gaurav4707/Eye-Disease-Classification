# OcuScan AI — Dataset Labelling Convention
v1.0 · June 2025 · Anterior Segment Disease Images

---

## Purpose

Because OCP, OCP Chronic, SJS, and Symblepharon can visually co-occur in the same anterior
segment image, this document defines the tiebreaking rules used to assign each image to exactly
one class. All annotators must follow these rules. Ambiguous cases should be flagged in
`labels.csv` using the `ambiguous_flag` and `ambiguous_note` columns.

---

## Primary Rule

**Assign an image to the class that represents the PRIMARY visual feature in that image.**

If two classes are both represented, the class whose features are more prominent, more extensive,
or more clinically relevant should win. Document the runner-up class in `ambiguous_note`.

---

## Class-Specific Rules

### `normal` (idx 0)
- Clear conjunctiva, no detectable abnormality
- No adhesion bands visible
- No significant inflammatory change
- No conjunctival injection suggesting active disease
- **Disqualified if:** any visible subconjunctival fibrosis, forniceal foreshortening, adhesion band, or keratinisation is present

### `ocp` — Ocular Cicatricial Pemphigoid, active stage (idx 1)
- Subconjunctival fibrosis visible
- Early forniceal foreshortening (reduced inferior fornix depth)
- No gross adhesion band spanning the entire fornix
- No advanced dense fibrosis filling the fornix
- **Assign as `ocp_chronic` if:** fibrosis is dense, fornix is substantially obliterated, or corneal involvement (pannus) is present
- **Assign as `symblepharon` if:** a distinct fibrous adhesion band connecting palpebral to bulbar conjunctiva is the primary visible feature

### `ocp_chronic` — OCP, chronic/end-stage (idx 2)
- Dense subconjunctival fibrosis
- Significant or complete forniceal foreshortening
- Possible corneal involvement (pannus, vascularisation)
- Severe dry eye features (keratinisation, reduced tear meniscus)
- **Rule:** This class requires evidence of ADVANCED structural change. Early OCP = `ocp`. Advanced = `ocp_chronic`.
- **Assign as `symblepharon` if:** a distinct adhesion band is the primary visual feature even in the context of chronic OCP

### `post_viral_ded` — Post-Viral Dry Eye Disease (idx 3)
- Conjunctival injection (redness)
- Reduced tear meniscus height
- Possible punctate epithelial erosions (not visible on gross photography)
- History of recent viral conjunctivitis (if available)
- **No significant scarring, fibrosis, or adhesion**
- **Disqualified if:** any conjunctival scarring or adhesion visible — reassign to `ocp`, `ocp_chronic`, or `symblepharon`

### `sjs` — Stevens-Johnson Syndrome (idx 4)

**Acute SJS sub-type:**
- Pseudomembrane formation
- Conjunctival haemorrhage / necrosis
- Acute inflammatory changes with lid margin involvement

**Chronic SJS sub-type:**
- Keratinisation of the conjunctival surface
- Trichiasis (misdirected lashes)
- Symblepharon (if present but NOT the primary image feature — see below)
- Corneal opacification / vascularisation

**Assign as `symblepharon` if:** the dominant and most clinically prominent feature in the image is a discrete fibrous adhesion band, even if the underlying aetiology is known to be SJS.

### `symblepharon` (idx 5 — sign detection class)
- A visible fibrous or fibrovascular adhesion band connecting the palpebral (eyelid inner surface) conjunctiva to the bulbar (eyeball surface) conjunctiva
- The adhesion band is the **primary and dominant** visual feature of the image
- Underlying aetiology may be OCP, SJS, chemical burn, thermal burn, or any other cicatricial cause
- **IMPORTANT:** Assign to `symblepharon` regardless of known or suspected underlying cause, as long as the adhesion band is the primary feature

---

## Tiebreaker Scenarios

| Image Description | Correct Label | Reasoning |
|------------------|---------------|-----------|
| Early scarring, slight fornix shallowing | `ocp` | No dense fibrosis, no adhesion |
| Dense fibrosis, fornix nearly obliterated, no discrete band | `ocp_chronic` | Advanced stage, no primary adhesion |
| Dense fibrosis + visible adhesion band spanning fornix | `symblepharon` | Adhesion band is primary feature |
| Acute SJS: pseudomembrane, necrosis | `sjs` | Acute features dominate |
| Chronic SJS: keratinisation, trichiasis, no prominent band | `sjs` | Keratinisation is primary |
| Chronic SJS: prominent adhesion band | `symblepharon` | Adhesion is primary feature |
| Chemical burn sequela with adhesion band | `symblepharon` | Sign detection, cause-agnostic |
| Mild injection, reduced tear film, no scarring | `post_viral_ded` | No structural change |
| Clear conjunctiva, no abnormality | `normal` | No detectable pathology |

---

## Flagging Ambiguous Cases

In `labels.csv`, use:
- `ambiguous_flag = 1` — image is ambiguous between two classes
- `ambiguous_note = "ocp vs ocp_chronic — forniceal loss present but not fully obliterated"` — free text description

Ambiguous images are still assigned to one class (the primary labeller's best judgement). They are:
- Tracked separately during analysis
- Excluded from disagreement calculations in inter-rater reliability studies
- Candidates for expert clinical review in v2.0

---

## Quality Requirements

All images in the dataset must meet these minimum quality standards:

| Requirement | Minimum |
|------------|---------|
| Resolution | ≥ 224 × 224 pixels |
| Focus | Eye region in focus; acceptable slight peripheral blur |
| Illumination | Adequate; neither completely dark nor overexposed |
| Eye coverage | Anterior segment (conjunctiva, cornea, visible fornix) fills ≥ 50% of frame |
| Format | JPEG, PNG, or WEBP |
| File size | < 10 MB |

**Reject if:** image is a fundus photograph, OCT scan, retinal image, or systemic skin lesion photograph.

---

## Inter-Rater Reliability

For each ambiguous image (where `ambiguous_flag = 1`), a second independent annotator should
review and record their label in a separate column. Cohen's kappa should be computed for the
complete dataset before model training. Target kappa ≥ 0.75 (substantial agreement).

---

*OcuScan AI Labelling Convention v1.0 · June 2025 · Confidential*
