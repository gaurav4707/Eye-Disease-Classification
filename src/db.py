"""
OcuScan AI — src/db.py
SQLite database initialisation, seeding, and access helpers.

Tables: predictions, disease_classes, model_versions
(schema per Backend Schema v2.0)
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "ocuscan.db"

# ─────────────────────────────────────────────────────────────────────────────
# Schema DDL
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id                  TEXT    NOT NULL,
    filename                    TEXT    NOT NULL,
    image_path                  TEXT,
    predicted_class             TEXT    NOT NULL,
    predicted_display           TEXT    NOT NULL,
    confidence                  REAL    NOT NULL,
    confidence_json             TEXT    NOT NULL,
    gradcam_path                TEXT,
    flagged                     INTEGER NOT NULL DEFAULT 0,
    symblepharon_warning_shown  INTEGER NOT NULL DEFAULT 0,
    sjs_emergency_shown         INTEGER NOT NULL DEFAULT 0,
    created_at                  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_version               TEXT    NOT NULL,
    inference_ms                INTEGER
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
    version_tag         TEXT    NOT NULL UNIQUE,
    architecture        TEXT    NOT NULL,
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
"""

# ─────────────────────────────────────────────────────────────────────────────
# Seed data for disease_classes
# ─────────────────────────────────────────────────────────────────────────────

DISEASE_CLASSES_SEED = [
    {
        'id': 0,
        'class_key': 'normal',
        'display_name': 'Normal',
        'icd10_code': 'Z01.01',
        'severity_tier': 'None',
        'is_sign': 0,
        'sjs_subtype_note': None,
        'description': (
            'Anterior segment appears normal. Clear conjunctiva, '
            'no adhesion, no opacity, no significant pathology detected.'
        ),
        'referral_text': 'No referral indicated. Routine follow-up as per clinical protocol.',
        'warning_banner_text': (
            'Normal — No pathology detected. '
            'Clinical correlation is always recommended.'
        ),
        'banner_colour': '#2E7D32',
    },
    {
        'id': 1,
        'class_key': 'ocp',
        'display_name': 'OCP (Ocular Cicatricial Pemphigoid)',
        'icd10_code': 'L12.1',
        'severity_tier': 'Medium',
        'is_sign': 0,
        'sjs_subtype_note': None,
        'description': (
            'Features consistent with Ocular Cicatricial Pemphigoid (OCP) — '
            'subconjunctival fibrosis and early forniceal foreshortening. '
            'An autoimmune condition requiring systemic evaluation.'
        ),
        'referral_text': (
            'Referral to ophthalmologist and dermatology/rheumatology recommended. '
            'Biopsy may be required for confirmation.'
        ),
        'warning_banner_text': (
            'OCP Suspected — Ocular Cicatricial Pemphigoid is an autoimmune condition. '
            'Urgent specialist referral is advised. Do not delay systemic workup.'
        ),
        'banner_colour': '#F57C00',
    },
    {
        'id': 2,
        'class_key': 'ocp_chronic',
        'display_name': 'OCP Chronic',
        'icd10_code': 'L12.1',
        'severity_tier': 'High',
        'is_sign': 0,
        'sjs_subtype_note': None,
        'description': (
            'Advanced Ocular Cicatricial Pemphigoid — dense forniceal fibrosis, '
            'significant forniceal loss, and possible corneal involvement. '
            'High risk of vision-threatening complications.'
        ),
        'referral_text': (
            'Urgent ophthalmology referral. Systemic immunosuppression likely required. '
            'Corneal status must be assessed immediately.'
        ),
        'warning_banner_text': (
            'OCP Chronic — Advanced scarring detected. '
            'URGENT: Specialist ophthalmology referral required. '
            'Vision-threatening complications possible.'
        ),
        'banner_colour': '#C62828',
    },
    {
        'id': 3,
        'class_key': 'post_viral_ded',
        'display_name': 'Post-Viral DED',
        'icd10_code': 'H04.12',
        'severity_tier': 'Low',
        'is_sign': 0,
        'sjs_subtype_note': None,
        'description': (
            'Features consistent with Post-Viral Dry Eye Disease — '
            'conjunctival injection, reduced tear meniscus, possible punctate staining. '
            'Often follows viral conjunctivitis.'
        ),
        'referral_text': (
            'Lubricating eye drops recommended. '
            'Referral if symptoms persist beyond 4–6 weeks or worsen.'
        ),
        'warning_banner_text': (
            'Post-Viral DED — Dry Eye Disease features detected. '
            'Lubricant therapy and follow-up advised.'
        ),
        'banner_colour': '#1565C0',
    },
    {
        'id': 4,
        'class_key': 'sjs',
        'display_name': 'SJS (Stevens-Johnson Syndrome)',
        'icd10_code': 'L51.1',
        'severity_tier': 'High',
        'is_sign': 0,
        'sjs_subtype_note': (
            'SJS class includes both acute (pseudomembrane, conjunctival necrosis) '
            'and chronic (keratinisation, trichiasis) sub-photo types. '
            'Merged in v1.0 due to limited chronic-phase samples.'
        ),
        'description': (
            'Features consistent with Stevens-Johnson Syndrome — a severe mucocutaneous '
            'reaction. Ocular involvement may include pseudomembrane formation, '
            'conjunctival necrosis (acute), or keratinisation and trichiasis (chronic).'
        ),
        'referral_text': (
            'EMERGENCY: If acute presentation, immediate hospital admission required. '
            'Chronic SJS: urgent ophthalmology for ongoing management and complication prevention.'
        ),
        'warning_banner_text': (
            '⚠ SJS EMERGENCY — Stevens-Johnson Syndrome features detected. '
            'IMMEDIATE medical attention required. '
            'Contact emergency services if acute presentation.'
        ),
        'banner_colour': '#FF6F00',
    },
    {
        'id': 5,
        'class_key': 'symblepharon',
        'display_name': 'Symblepharon',
        'icd10_code': 'H11.23',
        'severity_tier': 'High',
        'is_sign': 1,
        'sjs_subtype_note': None,
        'description': (
            'Symblepharon detected — fibrous adhesion between palpebral and bulbar conjunctiva. '
            'This is a sign of severe ocular surface disease, often secondary to OCP, '
            'SJS, chemical injury, or chronic inflammation.'
        ),
        'referral_text': (
            'Urgent ophthalmology referral. '
            'Underlying cause must be identified and treated to prevent progression.'
        ),
        'warning_banner_text': (
            'Symblepharon Detected — Fibrous adhesion identified. '
            'URGENT: Specialist referral required to identify and treat the underlying cause.'
        ),
        'banner_colour': '#B71C1C',
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a sqlite3 connection with row_factory set."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH, seed: bool = True) -> sqlite3.Connection:
    """
    Create tables and seed disease_classes if not already present.
    Returns open connection.
    """
    conn = get_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()

    if seed:
        _seed_disease_classes(conn)

    print(f"  [DB] Initialised → {db_path}")
    return conn


def _seed_disease_classes(conn: sqlite3.Connection):
    """Insert disease_classes seed rows (skip if already seeded)."""
    existing = conn.execute("SELECT COUNT(*) FROM disease_classes").fetchone()[0]
    if existing == len(DISEASE_CLASSES_SEED):
        return  # already seeded

    conn.execute("DELETE FROM disease_classes")
    for row in DISEASE_CLASSES_SEED:
        conn.execute(
            """INSERT INTO disease_classes
               (id, class_key, display_name, icd10_code, severity_tier,
                is_sign, sjs_subtype_note, description, referral_text,
                warning_banner_text, banner_colour)
               VALUES (:id, :class_key, :display_name, :icd10_code, :severity_tier,
                :is_sign, :sjs_subtype_note, :description, :referral_text,
                :warning_banner_text, :banner_colour)""",
            row,
        )
    conn.commit()
    print(f"  [DB] disease_classes seeded ({len(DISEASE_CLASSES_SEED)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# CRUD helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_prediction(result, conn: sqlite3.Connection) -> int:
    """
    Insert a PredictionResult into predictions table.
    Returns the new row id.
    """
    cursor = conn.execute(
        """INSERT INTO predictions
           (session_id, filename, image_path, predicted_class, predicted_display,
            confidence, confidence_json, gradcam_path, flagged,
            symblepharon_warning_shown, sjs_emergency_shown,
            model_version, inference_ms)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            result.session_id,
            result.filename,
            None,
            result.predicted_class,
            result.predicted_display,
            result.confidence,
            json.dumps(result.confidence_all),
            result.gradcam_path,
            int(result.flagged),
            int(result.is_sign_class),
            int(result.is_emergency_class),
            result.model_version,
            result.inference_ms,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_class_info(class_key: str, conn: sqlite3.Connection) -> Optional[dict]:
    """Return disease_classes row for class_key as dict, or None."""
    row = conn.execute(
        "SELECT * FROM disease_classes WHERE class_key = ?", (class_key,)
    ).fetchone()
    return dict(row) if row else None


def get_session_history(session_id: str, conn: sqlite3.Connection) -> list[dict]:
    """Return all predictions for a session, newest first."""
    rows = conn.execute(
        "SELECT * FROM predictions WHERE session_id = ? ORDER BY created_at DESC",
        (session_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_active_model_version(conn: sqlite3.Connection) -> Optional[dict]:
    """Return the currently active model_versions row, or None."""
    row = conn.execute(
        "SELECT * FROM model_versions WHERE is_active = 1 LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def register_model_version(version_tag: str,
                            checkpoint_path: str,
                            conn: sqlite3.Connection,
                            val_macro_f1: Optional[float] = None,
                            val_accuracy: Optional[float] = None,
                            sjs_recall: Optional[float] = None,
                            notes: Optional[str] = None) -> int:
    """
    Register a new model checkpoint and set it as active.
    Deactivates all previous versions.
    """
    from dataset import CLASS_NAMES  # local import to avoid circular
    conn.execute("UPDATE model_versions SET is_active = 0")
    cursor = conn.execute(
        """INSERT INTO model_versions
           (version_tag, architecture, checkpoint_path, num_classes,
            class_names_json, val_macro_f1, val_accuracy, sjs_recall,
            trained_at, is_active, notes)
           VALUES (?,?,?,?,?,?,?,?,?,1,?)""",
        (
            version_tag,
            'EfficientNetB0',
            checkpoint_path,
            6,
            json.dumps(CLASS_NAMES),
            val_macro_f1,
            val_accuracy,
            sjs_recall,
            datetime.now(timezone.utc).isoformat(),
            notes,
        ),
    )
    conn.commit()
    return cursor.lastrowid


if __name__ == '__main__':
    print("OcuScan AI — src/db.py")
    conn = init_db()
    print("disease_classes rows:")
    for row in conn.execute("SELECT id, class_key, display_name, severity_tier FROM disease_classes"):
        print(f"  [{row['id']}] {row['class_key']:<18} {row['display_name']:<40} {row['severity_tier']}")
    conn.close()
    print("[OK] db.py passed.")
