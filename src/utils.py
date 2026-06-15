"""
OcuScan AI — src/utils.py
Phase 3 (augments Phase 1/2 version)
Shared utilities: image loading, class label mapping, CSV export, temperature scaling.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ── Class constants ────────────────────────────────────────────────────────────

CLASS_NAMES: list[str] = [
    "normal",         # idx 0
    "ocp",            # idx 1
    "ocp_chronic",    # idx 2
    "post_viral_ded", # idx 3
    "sjs",            # idx 4  (2 sub-photo types merged)
    "symblepharon",   # idx 5  (sign detection)
]

DISPLAY_NAMES: dict[str, str] = {
    "normal": "Normal",
    "ocp": "OCP (Ocular Cicatricial Pemphigoid)",
    "ocp_chronic": "OCP Chronic",
    "post_viral_ded": "Post-Viral DED",
    "sjs": "SJS (Stevens-Johnson Syndrome)",
    "symblepharon": "Symblepharon",
}

ICD10_CODES: dict[str, str] = {
    "normal": "Z01.01",
    "ocp": "H10.40",
    "ocp_chronic": "H10.40",
    "post_viral_ded": "H04.123",
    "sjs": "L51.1",
    "symblepharon": "H11.231",
}

SEVERITY_TIERS: dict[str, str] = {
    "normal": "None",
    "ocp": "Medium",
    "ocp_chronic": "High",
    "post_viral_ded": "Medium",
    "sjs": "High",
    "symblepharon": "High",
}

EMERGENCY_CLASSES: set[str] = {"sjs"}
SIGN_CLASSES: set[str] = {"symblepharon"}

# ── Image I/O ──────────────────────────────────────────────────────────────────

def load_image(path: str | Path) -> Image.Image:
    """Load an image from disk as RGB PIL.Image."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def validate_image(image: Image.Image, min_dim: int = 64) -> tuple[bool, str]:
    """
    Basic sanity check for an uploaded anterior segment image.
    Returns (is_valid, reason_if_invalid).
    Note: does NOT perform clinical validation — only structural checks.
    """
    w, h = image.size
    if w < min_dim or h < min_dim:
        return False, f"Image too small ({w}×{h}); minimum {min_dim}px per side."
    if image.mode not in {"RGB", "L", "RGBA"}:
        return False, f"Unexpected colour mode: {image.mode}."
    arr = np.array(image.convert("RGB"))
    mean_brightness = float(arr.mean())
    if mean_brightness < 5:
        return False, "Image appears to be completely black."
    if mean_brightness > 250:
        return False, "Image appears to be completely white / overexposed."
    return True, ""


# ── Class label utilities ──────────────────────────────────────────────────────

def class_key_to_idx(class_key: str) -> int:
    """Return the integer index for a class key. Raises KeyError if unknown."""
    try:
        return CLASS_NAMES.index(class_key)
    except ValueError:
        raise KeyError(f"Unknown class key: '{class_key}'. Valid: {CLASS_NAMES}")


def class_idx_to_key(idx: int) -> str:
    """Return the class key for an integer index."""
    if not 0 <= idx < len(CLASS_NAMES):
        raise IndexError(f"Class index {idx} out of range [0, {len(CLASS_NAMES)-1}].")
    return CLASS_NAMES[idx]


def class_idx_to_display(idx: int) -> str:
    return DISPLAY_NAMES[class_idx_to_key(idx)]


def probs_to_confidence_dict(probs: np.ndarray | list[float]) -> dict[str, float]:
    """Map a 6-element probability array to {class_key: score} dict."""
    probs = list(probs)
    if len(probs) != len(CLASS_NAMES):
        raise ValueError(f"Expected {len(CLASS_NAMES)} probabilities, got {len(probs)}.")
    return {k: round(float(p), 6) for k, p in zip(CLASS_NAMES, probs)}


# ── Temperature scaling (standalone, no torch dependency) ─────────────────────

class TemperatureScaler:
    """
    Post-hoc temperature scaling for probability calibration.
    Fit on val logits; transform test logits before prediction.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        """
        Optimise temperature on validation set.
        logits: (N, C) raw model logits.
        labels: (N,) integer class indices.
        """
        from scipy.optimize import minimize_scalar

        def neg_log_likelihood(T: float) -> float:
            scaled = logits / T
            log_sum_exp = np.log(np.sum(np.exp(scaled - scaled.max(1, keepdims=True)), axis=1))
            nll = -(scaled[np.arange(len(labels)), labels] - scaled.max(1) - log_sum_exp)
            return float(nll.mean())

        result = minimize_scalar(neg_log_likelihood, bounds=(0.05, 20.0), method="bounded")
        self.temperature = float(result.x)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated softmax probabilities."""
        scaled = logits / self.temperature
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump({"temperature": self.temperature}, f)

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaler":
        with open(path) as f:
            d = json.load(f)
        return cls(temperature=d["temperature"])


# ── CSV export ─────────────────────────────────────────────────────────────────

def export_predictions_csv(rows: list[dict], out_path: str | Path) -> None:
    """
    Export a list of prediction dicts to CSV.

    Expected keys per row (all optional extras kept):
        session_id, filename, predicted_class, predicted_display,
        confidence, flagged, created_at, model_version
    """
    if not rows:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten confidence_json if present
    flat_rows = []
    for r in rows:
        row = dict(r)
        if "confidence_json" in row and isinstance(row["confidence_json"], str):
            try:
                conf_dict = json.loads(row["confidence_json"])
                for k, v in conf_dict.items():
                    row[f"conf_{k}"] = round(v, 4)
            except (json.JSONDecodeError, TypeError):
                pass
            del row["confidence_json"]
        flat_rows.append(row)

    fieldnames = list(flat_rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"[utils] CSV exported → {out_path} ({len(rows)} rows)")


# ── Session ID ────────────────────────────────────────────────────────────────

def new_session_id() -> str:
    return str(uuid.uuid4())


# ── Database helpers ───────────────────────────────────────────────────────────

def get_db_connection(db_path: str = "ocuscan.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = "ocuscan.db") -> None:
    """
    Create database tables if they don't exist.
    Schema: predictions, disease_classes, model_versions.
    """
    conn = get_db_connection(db_path)
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id              TEXT    NOT NULL,
            filename                TEXT    NOT NULL,
            image_path              TEXT,
            predicted_class         TEXT    NOT NULL,
            predicted_display       TEXT    NOT NULL,
            confidence              REAL    NOT NULL,
            confidence_json         TEXT    NOT NULL,
            gradcam_path            TEXT,
            flagged                 INTEGER NOT NULL DEFAULT 0,
            symblepharon_warning_shown INTEGER NOT NULL DEFAULT 0,
            sjs_emergency_shown     INTEGER NOT NULL DEFAULT 0,
            created_at              DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            model_version           TEXT    NOT NULL DEFAULT 'v1.0.0',
            inference_ms            INTEGER
        );

        CREATE TABLE IF NOT EXISTS disease_classes (
            id                  INTEGER PRIMARY KEY,
            class_key           TEXT    NOT NULL UNIQUE,
            display_name        TEXT    NOT NULL,
            icd10_code          TEXT    NOT NULL,
            severity_tier       TEXT    NOT NULL,
            is_sign             INTEGER NOT NULL DEFAULT 0,
            sjs_subtype_note    TEXT,
            description         TEXT    NOT NULL,
            referral_text       TEXT    NOT NULL,
            warning_banner_text TEXT    NOT NULL,
            banner_colour       TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_versions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            version_tag         TEXT    NOT NULL,
            architecture        TEXT    NOT NULL DEFAULT 'EfficientNetB0',
            checkpoint_path     TEXT    NOT NULL,
            num_classes         INTEGER NOT NULL DEFAULT 6,
            class_names_json    TEXT    NOT NULL,
            val_macro_f1        REAL,
            val_accuracy        REAL,
            ocp_vs_ocpchronic_f1 TEXT,
            sjs_recall          REAL,
            trained_at          DATETIME,
            is_active           INTEGER NOT NULL DEFAULT 0,
            notes               TEXT
        );
    """)
    conn.commit()
    _seed_disease_classes(conn)
    conn.close()
    print(f"[utils] Database initialised → {db_path}")


def _seed_disease_classes(conn: sqlite3.Connection) -> None:
    """Seed the disease_classes reference table if empty."""
    c = conn.cursor()
    if c.execute("SELECT COUNT(*) FROM disease_classes").fetchone()[0] > 0:
        return  # already seeded

    seed_data = [
        (
            0, "normal", "Normal", "Z01.01", "None", 0, None,
            "No anterior segment pathology detected. Conjunctiva appears clear with no adhesion or opacity.",
            "No referral indicated. Routine follow-up as clinically appropriate.",
            "No ocular pathology detected. This is a screening aid only — clinical judgement must prevail.",
            "#22c55e",
        ),
        (
            1, "ocp", "OCP (Ocular Cicatricial Pemphigoid)", "H10.40", "Medium", 0, None,
            "Subconjunctival fibrosis with early forniceal foreshortening. "
            "An autoimmune blistering disease with conjunctival cicatrisation.",
            "Urgent ophthalmology referral. Systemic immunosuppression often required.",
            "CAUTION: OCP detected. Urgent specialist review recommended. "
            "Do not delay — disease progression can lead to blindness.",
            "#f59e0b",
        ),
        (
            2, "ocp_chronic", "OCP Chronic", "H10.40", "High", 0, None,
            "Advanced forniceal loss, dense subconjunctival fibrosis, possible corneal involvement. "
            "Late-stage ocular cicatricial pemphigoid.",
            "Emergency ophthalmology referral. Corneal protection may be required.",
            "WARNING: Chronic OCP detected. Immediate specialist review essential. "
            "Risk of corneal exposure and permanent vision loss.",
            "#ef4444",
        ),
        (
            3, "post_viral_ded", "Post-Viral DED", "H04.123", "Medium", 0, None,
            "Post-viral dry eye disease. Conjunctival injection, reduced tear meniscus, "
            "possible punctate epithelial staining.",
            "Ophthalmology review. Lubricants and anti-inflammatory drops as first line.",
            "NOTE: Post-Viral Dry Eye Disease detected. "
            "Ocular surface management advised — seek specialist review.",
            "#3b82f6",
        ),
        (
            4, "sjs", "SJS (Stevens-Johnson Syndrome)", "L51.1", "High", 0,
            "SJS class merges acute (pseudomembrane, necrosis) and chronic (keratinisation, "
            "trichiasis) sub-photo types in v1.0.",
            "Stevens-Johnson Syndrome — severe mucocutaneous reaction. "
            "Acute: pseudomembrane and conjunctival necrosis. "
            "Chronic: symblepharon, trichiasis, keratinisation.",
            "EMERGENCY: Immediate ophthalmology and dermatology referral. "
            "Systemic management required urgently.",
            "⚠ EMERGENCY: SJS pattern detected. Immediate specialist review required. "
            "Ocular surface preservation is time-critical.",
            "#f97316",
        ),
        (
            5, "symblepharon", "Symblepharon", "H11.231", "High", 1, None,
            "Fibrous adhesion band between palpebral and bulbar conjunctiva. "
            "May indicate sequela of SJS, OCP, or chemical/thermal injury.",
            "Urgent ophthalmology referral. Cause identification essential. "
            "Surgical intervention may be required.",
            "WARNING: Symblepharon detected. Urgent specialist review required. "
            "This may indicate progressive cicatricial disease.",
            "#ef4444",
        ),
    ]

    c.executemany(
        """INSERT INTO disease_classes
           (id, class_key, display_name, icd10_code, severity_tier, is_sign,
            sjs_subtype_note, description, referral_text, warning_banner_text, banner_colour)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        seed_data,
    )
    conn.commit()
    print("[utils] disease_classes table seeded.")


def save_prediction_to_db(result, db_path: str = "ocuscan.db") -> int:
    """
    Insert a PredictionResult into the predictions table.
    Returns the new row id.
    """
    conn = get_db_connection(db_path)
    c = conn.cursor()
    c.execute(
        """INSERT INTO predictions
           (session_id, filename, image_path, predicted_class, predicted_display,
            confidence, confidence_json, gradcam_path, flagged,
            symblepharon_warning_shown, sjs_emergency_shown,
            created_at, model_version, inference_ms)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            result.session_id,
            result.filename,
            getattr(result, "image_path", None),
            result.predicted_class,
            result.predicted_display,
            result.confidence,
            json.dumps(result.confidence_all),
            result.gradcam_path,
            int(result.flagged),
            int(result.is_sign_class),
            int(result.is_emergency_class),
            datetime.now(timezone.utc).isoformat(),
            result.model_version,
            result.inference_ms,
        ),
    )
    row_id = c.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_class_info(class_key: str, db_path: str = "ocuscan.db") -> dict:
    """Return the disease_classes row for a given class key as a dict."""
    conn = get_db_connection(db_path)
    row = conn.execute(
        "SELECT * FROM disease_classes WHERE class_key = ?", (class_key,)
    ).fetchone()
    conn.close()
    if row is None:
        raise KeyError(f"Class key '{class_key}' not found in disease_classes table.")
    return dict(row)


def get_session_history(session_id: str, db_path: str = "ocuscan.db") -> list[dict]:
    """Return all predictions for a session, newest first."""
    conn = get_db_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM predictions WHERE session_id = ? ORDER BY created_at DESC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Metrics formatting helpers ─────────────────────────────────────────────────

def format_metrics_summary(metrics_json_path: str | Path) -> str:
    """
    Return a human-readable summary string from a metrics.json file.
    Used in the evaluation notebook and Streamlit About page.
    """
    with open(metrics_json_path) as f:
        m = json.load(f)

    lines = [
        "OcuScan AI — Model Performance Summary",
        "=" * 42,
        f"Accuracy          : {m.get('accuracy', 'N/A')}",
        f"Macro F1          : {m.get('macro_f1', 'N/A')}",
        f"Macro AUC-ROC     : {m.get('macro_auc_roc', 'N/A')}",
        "",
        "Per-class F1:",
    ]
    for key, vals in m.get("per_class", {}).items():
        lines.append(f"  {vals['display']:<28}: {vals['f1']}")

    ocp = m.get("ocp_vs_ocp_chronic", {})
    lines += [
        "",
        f"OCP ↔ OCP Chronic confusion rate : {ocp.get('ocp_confusion_rate', 'N/A')}",
    ]
    cs = m.get("clinical_stakes", {})
    lines += [
        f"SJS Recall               : {cs.get('sjs_recall', 'N/A')}",
        f"Symblepharon Precision   : {cs.get('symblepharon_precision', 'N/A')}",
    ]
    return "\n".join(lines)


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def register_model_version(
    version_tag: str,
    checkpoint_path: str,
    metrics: Optional[dict] = None,
    notes: Optional[str] = None,
    db_path: str = "ocuscan.db",
    set_active: bool = True,
) -> None:
    """
    Register a trained checkpoint in model_versions table.
    If set_active=True, deactivates all other versions first.
    """
    conn = get_db_connection(db_path)
    c = conn.cursor()

    if set_active:
        c.execute("UPDATE model_versions SET is_active = 0")

    ocp_f1 = None
    if metrics:
        pc = metrics.get("per_class", {})
        if "ocp" in pc and "ocp_chronic" in pc:
            ocp_f1 = json.dumps(
                {"ocp": pc["ocp"]["f1"], "ocp_chronic": pc["ocp_chronic"]["f1"]}
            )

    c.execute(
        """INSERT INTO model_versions
           (version_tag, architecture, checkpoint_path, num_classes, class_names_json,
            val_macro_f1, val_accuracy, ocp_vs_ocpchronic_f1, sjs_recall,
            trained_at, is_active, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            version_tag,
            "EfficientNetB0",
            checkpoint_path,
            len(CLASS_NAMES),
            json.dumps(CLASS_NAMES),
            metrics.get("macro_f1") if metrics else None,
            metrics.get("accuracy") if metrics else None,
            ocp_f1,
            (metrics or {}).get("clinical_stakes", {}).get("sjs_recall"),
            datetime.now(timezone.utc).isoformat(),
            int(set_active),
            notes,
        ),
    )
    conn.commit()
    conn.close()
    print(f"[utils] Model version '{version_tag}' registered.")
