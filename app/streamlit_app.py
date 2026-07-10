"""
OcuScan AI — app/streamlit_app.py
Phase 4 | Week 4 | v1.0.0
Streamlit web application — 4 screens: Home/Upload, Results, History, About.

Architecture:
  - Session state manages current_screen, current_result, history list
  - SQLite (db.py) persists predictions and reads disease_classes reference data
  - predict.py handles all inference; gradcam.py handles heatmap generation
  - reportlab generates single-result PDF; csv stdlib handles batch export
  - Custom CSS injected via st.markdown for clinical dashboard styling
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import sys
import tempfile
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

# ── Path resolution ────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(APP_DIR))

# ── Local imports (guarded for demo without model checkpoint) ──────────────────
try:
    from predict import (
        CLASS_NAMES,
        DISPLAY_NAMES,
        ICD10_CODES,
        LOW_CONFIDENCE_THRESHOLD,
        PredictionResult,
        predict,
    )
    _PREDICT_AVAILABLE = True
except ImportError:
    _PREDICT_AVAILABLE = False
    CLASS_NAMES = ["normal", "ocp", "ocp_chronic", "post_viral_ded", "sjs", "symblepharon"]
    DISPLAY_NAMES = {
        "normal": "Normal",
        "ocp": "OCP (Ocular Cicatricial Pemphigoid)",
        "ocp_chronic": "OCP Chronic",
        "post_viral_ded": "Post-Viral DED",
        "sjs": "SJS (Stevens-Johnson Syndrome)",
        "symblepharon": "Symblepharon",
    }
    ICD10_CODES = {
        "normal": "Z01.01",
        "ocp": "H10.40",
        "ocp_chronic": "H10.40",
        "post_viral_ded": "H04.123",
        "sjs": "L51.1",
        "symblepharon": "H11.231",
    }
    LOW_CONFIDENCE_THRESHOLD = 0.60

try:
    from db import init_db, save_prediction, get_class_info, get_session_history
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

try:
    from utils import validate_image, export_predictions_csv
    _UTILS_AVAILABLE = True
except ImportError:
    _UTILS_AVAILABLE = False


# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_VERSION = "v1.0.0"
CHECKPOINT_PATH = str(PROJECT_ROOT / "models" / "phase2_best.pt")
DB_PATH = str(PROJECT_ROOT / "ocuscan.db")
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ACCEPTED_TYPES = ["jpg", "jpeg", "png", "webp"]

# ── Design System (from UI/UX Spec) ───────────────────────────────────────────
COLORS = {
    "primary": "#0F766E",
    "primary_dark": "#134E4A",
    "bg": "#F8FAFC",
    "surface": "#FFFFFF",
    "text_primary": "#1C1C1E",
    "text_secondary": "#6B7280",
    "border": "#E5E7EB",
    "high_conf": "#16A34A",
    "med_conf": "#D97706",
    "low_conf": "#DC2626",
    "info_blue": "#1A73E8",
}

# Per-class banner configuration (colour, icon, text)
CLASS_BANNERS = {
    "symblepharon": {
        "bg": "#DC2626",
        "text_color": "#FFFFFF",
        "icon": "⚠",
        "title": "SIGN DETECTED — Not a Primary Diagnosis",
        "body": (
            "Symblepharon is a structural sign — a fibrous adhesion between palpebral and bulbar "
            "conjunctiva. This model detects the <strong>presence</strong> of the adhesion but does "
            "<strong>NOT</strong> identify the underlying cause (SJS, OCP, chemical burn, or other). "
            "Specialist evaluation is mandatory to determine aetiology and prevent progression."
        ),
    },
    "sjs": {
        "bg": "#D97706",
        "text_color": "#FFFFFF",
        "icon": "⚡",
        "title": "EMERGENCY RISK — Stevens-Johnson Syndrome",
        "body": (
            "Acute SJS is an ophthalmic emergency. If the patient presents with active ocular "
            "inflammation, pseudomembrane formation, or conjunctival necrosis, seek urgent "
            "ophthalmology review immediately. <br/><em>Note: this class includes both acute "
            "(inflammatory) and chronic (scarring) SJS sub-types merged in v1.0.</em>"
        ),
    },
    "ocp_chronic": {
        "bg": "#DC2626",
        "text_color": "#FFFFFF",
        "icon": "⚠",
        "title": "ADVANCED OCP — Severe Scarring Stage",
        "body": (
            "Chronic OCP indicates advanced conjunctival fibrosis and significant forniceal loss. "
            "Systemic immunosuppression should already be in place. If the patient is not under "
            "specialist care, urgent referral to ophthalmology and a mucous membrane pemphigoid "
            "physician is required."
        ),
    },
    "ocp": {
        "bg": "#D97706",
        "text_color": "#FFFFFF",
        "icon": "ℹ",
        "title": "EARLY OCP — Active Ocular Cicatricial Pemphigoid",
        "body": (
            "The model has classified this as general OCP rather than chronic/end-stage. "
            "Early referral to an ocular surface specialist and mucous membrane pemphigoid "
            "physician is recommended. Do not delay systemic workup — disease can progress "
            "rapidly to an advanced stage."
        ),
    },
    "post_viral_ded": {
        "bg": "#1A73E8",
        "text_color": "#FFFFFF",
        "icon": "ℹ",
        "title": "POST-VIRAL DED — Dry Eye Disease",
        "body": (
            "Post-Viral Dry Eye Disease is usually manageable with lubricants and anti-inflammatory "
            "drops. Monitor for progression. If symptoms persist beyond 3 months or worsen, "
            "specialist review is recommended."
        ),
    },
    "normal": {
        "bg": "#16A34A",
        "text_color": "#FFFFFF",
        "icon": "✓",
        "title": "NO PATHOLOGY DETECTED",
        "body": (
            "No anterior segment disease features detected in this image. If the patient has "
            "persistent clinical symptoms despite this result, clinical review by a qualified "
            "ophthalmologist is still advised."
        ),
    },
}

SEVERITY_DISPLAY = {
    "High": ("High Risk", "#DC2626"),
    "Medium": ("Medium Risk", "#D97706"),
    "None": ("No Pathology", "#16A34A"),
}

# ── Disease class reference data (fallback if db unavailable) ─────────────────
DISEASE_CLASS_DATA = {
    "normal": {
        "description": "Anterior segment appears normal. Clear conjunctiva, no adhesion, no opacity, no significant pathology detected.",
        "referral_text": "No referral indicated. Routine follow-up as per clinical protocol.",
        "severity_tier": "None",
        "icd10_code": "Z01.01",
        "is_sign": False,
        "sjs_subtype_note": None,
    },
    "ocp": {
        "description": "Features consistent with Ocular Cicatricial Pemphigoid — subconjunctival fibrosis and early forniceal foreshortening. An autoimmune condition requiring systemic evaluation.",
        "referral_text": "Referral to ophthalmologist and dermatology/rheumatology recommended. Biopsy may be required for confirmation.",
        "severity_tier": "Medium",
        "icd10_code": "H10.40",
        "is_sign": False,
        "sjs_subtype_note": None,
    },
    "ocp_chronic": {
        "description": "Advanced Ocular Cicatricial Pemphigoid — dense forniceal fibrosis, significant forniceal loss, and possible corneal involvement. High risk of vision-threatening complications.",
        "referral_text": "Urgent ophthalmology referral. Systemic immunosuppression likely required. Corneal status must be assessed immediately.",
        "severity_tier": "High",
        "icd10_code": "H10.40",
        "is_sign": False,
        "sjs_subtype_note": None,
    },
    "post_viral_ded": {
        "description": "Features consistent with Post-Viral Dry Eye Disease — conjunctival injection, reduced tear meniscus, possible punctate staining. Often follows viral conjunctivitis.",
        "referral_text": "Lubricating eye drops recommended. Referral if symptoms persist beyond 4–6 weeks or worsen.",
        "severity_tier": "Medium",
        "icd10_code": "H04.123",
        "is_sign": False,
        "sjs_subtype_note": None,
    },
    "sjs": {
        "description": "Features consistent with Stevens-Johnson Syndrome — a severe mucocutaneous reaction. Ocular involvement may include pseudomembrane formation, conjunctival necrosis (acute), or keratinisation and trichiasis (chronic).",
        "referral_text": "EMERGENCY: If acute presentation, immediate hospital admission required. Chronic SJS: urgent ophthalmology for ongoing management.",
        "severity_tier": "High",
        "icd10_code": "L51.1",
        "is_sign": False,
        "sjs_subtype_note": "SJS class merges acute (pseudomembrane, necrosis) and chronic (keratinisation, trichiasis) sub-photo types in v1.0.",
    },
    "symblepharon": {
        "description": "Symblepharon detected — fibrous adhesion between palpebral and bulbar conjunctiva. This is a sign of severe ocular surface disease, often secondary to OCP, SJS, chemical injury, or chronic inflammation.",
        "referral_text": "Urgent ophthalmology referral. Underlying cause must be identified and treated to prevent progression.",
        "severity_tier": "High",
        "icd10_code": "H11.231",
        "is_sign": True,
        "sjs_subtype_note": None,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# CSS Injection
# ═══════════════════════════════════════════════════════════════════════════════

def inject_css() -> None:
    """Inject full custom CSS matching the OcuScan AI clinical dashboard spec."""
    st.markdown(
        """
        <style>
        /* ── Google Fonts ─────────────────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* ── Global reset ─────────────────────────────────────────────────── */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, Arial, sans-serif;
            color: #1C1C1E;
        }

        /* ── App background ───────────────────────────────────────────────── */
        .stApp {
            background-color: #F8FAFC;
        }

        /* ── Hide default Streamlit chrome ────────────────────────────────── */
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 920px; }

        /* ── Sidebar ──────────────────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background-color: #134E4A !important;
            border-right: 1px solid #0F766E;
        }
        [data-testid="stSidebar"] * { color: #FFFFFF !important; }
        [data-testid="stSidebar"] .stRadio label {
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.15s;
            display: block;
            font-size: 14px;
            font-weight: 500;
        }
        [data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(255,255,255,0.1);
        }

        /* ── Card component ───────────────────────────────────────────────── */
        .ocuscan-card {
            background: #FFFFFF;
            border-radius: 8px;
            border: 1px solid #E5E7EB;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }

        /* ── Primary button ───────────────────────────────────────────────── */
        .stButton > button[kind="primary"],
        .stButton > button {
            background-color: #0F766E !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            transition: background 0.15s !important;
            cursor: pointer !important;
        }
        .stButton > button:hover {
            background-color: #134E4A !important;
        }
        .stButton > button:disabled {
            opacity: 0.4 !important;
            cursor: not-allowed !important;
        }

        /* ── Download buttons ─────────────────────────────────────────────── */
        .stDownloadButton > button {
            background-color: #FFFFFF !important;
            color: #0F766E !important;
            border: 1.5px solid #0F766E !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            transition: all 0.15s !important;
        }
        .stDownloadButton > button:hover {
            background-color: #0F766E !important;
            color: #FFFFFF !important;
        }

        /* ── Screen headings ──────────────────────────────────────────────── */
        .screen-title {
            font-size: 24px;
            font-weight: 700;
            color: #134E4A;
            border-bottom: 2px solid #0F766E;
            padding-bottom: 8px;
            margin-bottom: 20px;
        }

        /* ── Predicted class card ─────────────────────────────────────────── */
        .class-name-display {
            font-size: 28px;
            font-weight: 700;
            color: #134E4A;
            margin-bottom: 4px;
            line-height: 1.2;
        }
        .icd10-display {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: #6B7280;
            margin-bottom: 12px;
        }

        /* ── Confidence badge ─────────────────────────────────────────────── */
        .conf-badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 15px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: #FFFFFF;
            margin-right: 8px;
        }
        .severity-pill {
            display: inline-block;
            padding: 3px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            color: #FFFFFF;
            vertical-align: middle;
        }

        /* ── Warning banners ──────────────────────────────────────────────── */
        .warning-banner {
            border-radius: 6px;
            padding: 14px 18px;
            margin: 16px 0;
            border-left: 4px solid rgba(0,0,0,0.3);
        }
        .warning-banner .banner-title {
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .warning-banner .banner-body {
            font-size: 13px;
            line-height: 1.55;
            opacity: 0.95;
        }

        /* ── Confidence chart container ───────────────────────────────────── */
        .chart-caption {
            font-size: 12px;
            color: #6B7280;
            text-align: center;
            margin-top: 4px;
        }

        /* ── Clinical notes card ──────────────────────────────────────────── */
        .notes-label {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #6B7280;
            margin-bottom: 4px;
        }
        .notes-body {
            font-size: 14px;
            color: #1C1C1E;
            line-height: 1.6;
            margin-bottom: 12px;
        }
        .referral-box {
            background: #F0FDF4;
            border-left: 3px solid #16A34A;
            padding: 10px 14px;
            border-radius: 0 6px 6px 0;
            font-size: 13px;
            color: #166534;
            line-height: 1.5;
        }
        .referral-box.urgent {
            background: #FEF2F2;
            border-color: #DC2626;
            color: #991B1B;
        }

        /* ── History table ────────────────────────────────────────────────── */
        .hist-row {
            display: flex;
            align-items: center;
            padding: 10px 14px;
            border-bottom: 1px solid #E5E7EB;
            gap: 12px;
            font-size: 13px;
        }
        .hist-row:last-child { border-bottom: none; }
        .hist-row:hover { background: #F8FAFC; }

        /* ── Medical disclaimer ───────────────────────────────────────────── */
        .medical-disclaimer {
            background: #FFF7ED;
            border: 1px solid #FED7AA;
            border-radius: 6px;
            padding: 12px 16px;
            font-size: 12px;
            color: #92400E;
            margin-top: 24px;
            line-height: 1.5;
        }

        /* ── Sidebar nav ──────────────────────────────────────────────────── */
        .sidebar-logo {
            font-size: 22px;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.5px;
            margin-bottom: 2px;
        }
        .sidebar-subtitle {
            font-size: 11px;
            color: rgba(255,255,255,0.7);
            margin-bottom: 20px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .model-badge {
            background: rgba(255,255,255,0.15);
            border-radius: 4px;
            padding: 4px 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: rgba(255,255,255,0.85);
            display: inline-block;
            margin-bottom: 6px;
        }
        .class-count-badge {
            background: #0F766E;
            border-radius: 20px;
            padding: 3px 10px;
            font-size: 11px;
            font-weight: 600;
            color: #FFFFFF;
            display: inline-block;
        }

        /* ── Low confidence alert ─────────────────────────────────────────── */
        .low-conf-alert {
            background: #FFF7ED;
            border: 1.5px solid #D97706;
            border-radius: 6px;
            padding: 12px 16px;
            font-size: 13px;
            color: #92400E;
            font-weight: 500;
            margin: 12px 0;
        }

        /* ── Upload drop zone ─────────────────────────────────────────────── */
        [data-testid="stFileUploader"] {
            border: 2px dashed #0F766E !important;
            border-radius: 8px !important;
            background: #F0FDFA !important;
        }

        /* ── Metric tiles ─────────────────────────────────────────────────── */
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 12px 16px !important;
        }

        /* ── Footer ───────────────────────────────────────────────────────── */
        .sidebar-footer {
            position: absolute;
            bottom: 16px;
            font-size: 11px;
            color: rgba(255,255,255,0.5);
            left: 16px;
            right: 16px;
        }

        /* ── About class table ────────────────────────────────────────────── */
        .class-table-row {
            display: grid;
            grid-template-columns: 140px 70px 90px 1fr;
            gap: 8px;
            padding: 10px 12px;
            border-bottom: 1px solid #E5E7EB;
            font-size: 13px;
            align-items: center;
        }
        .class-table-row:nth-child(even) { background: #F8FAFC; }
        .class-table-header {
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #6B7280;
            background: #F8FAFC;
            border-bottom: 2px solid #E5E7EB;
        }

        /* ── Focus ring for accessibility ─────────────────────────────────── */
        :focus-visible {
            outline: 2px solid #0F766E;
            outline-offset: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Session State Bootstrap
# ═══════════════════════════════════════════════════════════════════════════════

def init_session_state() -> None:
    """Initialise all session state keys needed across screens."""
    defaults = {
        "screen": "Home",
        "session_id": str(uuid.uuid4()),
        "current_result": None,
        "history": [],
        "uploaded_image": None,
        "uploaded_filename": None,
        "replay_result": None,
        "db_conn": None,
        "model_loaded": False,
        "load_error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_db():
    """Return a lazy-initialised SQLite connection."""
    if not _DB_AVAILABLE:
        return None
    if st.session_state.db_conn is None:
        try:
            from db import init_db
            conn = init_db(db_path=Path(DB_PATH), seed=True)
            st.session_state.db_conn = conn
        except Exception as e:
            pass  # graceful: db unavailable
    return st.session_state.db_conn


# ═══════════════════════════════════════════════════════════════════════════════
# Shared UI Components
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    """Render sidebar and return the selected screen name."""
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">OcuScan AI</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-subtitle">Anterior Segment Classifier</div>',
            unsafe_allow_html=True,
        )

        screen = st.radio(
            "Navigate",
            options=["Home", "History", "About"],
            label_visibility="collapsed",
            key="nav_radio",
        )

        st.markdown("<hr style='border-color:rgba(255,255,255,0.2);margin:16px 0'/>", unsafe_allow_html=True)
        st.markdown('<div class="model-badge">Model: EfficientNetB0</div>', unsafe_allow_html=True)
        st.markdown('<div class="model-badge">v1.0.0</div>', unsafe_allow_html=True)
        st.markdown('<br/><div class="class-count-badge">6 classes</div>', unsafe_allow_html=True)

        n_hist = len(st.session_state.history)
        if n_hist > 0:
            st.markdown(
                f"<br/><div style='font-size:12px;color:rgba(255,255,255,0.6);margin-top:8px'>"
                f"Session: {n_hist} prediction{'s' if n_hist != 1 else ''}</div>",
                unsafe_allow_html=True,
            )

        # Checkpoint status
        checkpoint = Path(CHECKPOINT_PATH)
        if checkpoint.exists():
            st.markdown(
                "<div style='font-size:11px;color:#6EE7B7;margin-top:12px'>● Model loaded</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size:11px;color:#FCA5A5;margin-top:12px'>○ No checkpoint — demo mode</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div style='position:fixed;bottom:12px;left:16px;font-size:10px;"
            "color:rgba(255,255,255,0.35)'>OcuScan AI v1.0.0 · For research use only</div>",
            unsafe_allow_html=True,
        )

    return screen


def render_medical_disclaimer() -> None:
    st.markdown(
        """
        <div class="medical-disclaimer">
        ⚕ <strong>Medical Disclaimer:</strong> OcuScan AI is a research and educational tool intended
        to <em>assist</em> clinicians. It is <strong>NOT</strong> a certified medical device and must
        not be used as the sole basis for any clinical decision. All outputs require review by a
        qualified ophthalmologist. Confidence scores below 60% are automatically flagged.
        Not approved for diagnostic use in any jurisdiction.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_warning_banner(class_key: str) -> None:
    """Render the mandatory class-specific warning banner."""
    cfg = CLASS_BANNERS.get(class_key)
    if not cfg:
        return
    bg = cfg["bg"]
    txt = cfg["text_color"]
    icon = cfg["icon"]
    title = cfg["title"]
    body = cfg["body"]
    st.markdown(
        f"""
        <div class="warning-banner" style="background:{bg};color:{txt};">
          <div class="banner-title">{icon}&nbsp;&nbsp;{title}</div>
          <div class="banner-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_chart(confidence_all: dict[str, float], predicted_class: str) -> None:
    """
    Render horizontal confidence bar chart — all 6 classes sorted descending.
    OCP and OCP Chronic have a subtle shared background to indicate they are related.
    """
    sorted_items = sorted(confidence_all.items(), key=lambda x: x[1], reverse=True)
    labels = [DISPLAY_NAMES.get(k, k) for k, _ in sorted_items]
    values = [v * 100 for _, v in sorted_items]
    keys = [k for k, _ in sorted_items]

    colors_bars = []
    for k in keys:
        if k == predicted_class:
            colors_bars.append("#0F766E")
        elif k in ("ocp", "ocp_chronic"):
            colors_bars.append("#7FBFBC")  # soft teal for related classes
        else:
            colors_bars.append("#D1D5DB")

    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Highlight OCP/OCP Chronic group background
    ocp_positions = [i for i, k in enumerate(keys) if k in ("ocp", "ocp_chronic")]
    if len(ocp_positions) >= 2:
        y_low = min(ocp_positions) - 0.45
        y_high = max(ocp_positions) + 0.45
        ax.axhspan(y_low, y_high, facecolor="#F0FDFA", alpha=0.8, zorder=0)

    bars = ax.barh(labels, values, color=colors_bars, height=0.55, zorder=2)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            min(val + 1.5, 99),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            color="#1C1C1E",
            fontweight="600",
        )

    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", fontsize=9, color="#6B7280")
    ax.xaxis.set_major_locator(plt.MultipleLocator(25))
    ax.tick_params(axis="y", labelsize=10, colors="#1C1C1E")
    ax.tick_params(axis="x", labelsize=8, colors="#6B7280")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E5E7EB")
    ax.spines["bottom"].set_color("#E5E7EB")
    ax.grid(axis="x", color="#E5E7EB", alpha=0.7, zorder=1)
    ax.invert_yaxis()

    plt.tight_layout(pad=0.8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown(
        "<p class='chart-caption'>▲ OCP & OCP Chronic are visually grouped — related conditions with distinct staging</p>",
        unsafe_allow_html=True,
    )


def get_gradcam_overlay(result, original_image: Image.Image) -> Optional[Image.Image]:
    """Attempt to generate Grad-CAM overlay; returns None on failure."""
    if result.gradcam_path and Path(result.gradcam_path).exists():
        return Image.open(result.gradcam_path).convert("RGB")

    if not _PREDICT_AVAILABLE:
        return None

    try:
        from gradcam import GradCAM
        from predict import load_model, preprocess

        model = load_model(CHECKPOINT_PATH, device="cpu")
        tensor = preprocess(original_image).to("cpu")
        gcam = GradCAM(model)
        pred_idx = CLASS_NAMES.index(result.predicted_class)
        heatmap = gcam.compute(tensor, class_idx=pred_idx)
        overlay = GradCAM.overlay(original_image, heatmap)
        gcam.remove_hooks()
        return overlay
    except Exception:
        return None


def render_gradcam_panel(result, original_image: Image.Image) -> None:
    """Render side-by-side original image and Grad-CAM overlay."""
    cls_key = result.predicted_class
    attention_caption = {
        "symblepharon": "Model Attention Map — adhesion region highlighted",
        "ocp": "Model Attention Map — forniceal region highlighted",
        "ocp_chronic": "Model Attention Map — forniceal/fibrosis region highlighted",
        "sjs": "Model Attention Map — pseudomembrane/keratinisation region highlighted",
        "post_viral_ded": "Model Attention Map — ocular surface region",
        "normal": "Model Attention Map — baseline reference",
    }.get(cls_key, "Model Attention Map")

    overlay = get_gradcam_overlay(result, original_image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            original_image.resize((224, 224)),
            caption="Anterior Segment Image",
            use_column_width=True,
        )
    with col2:
        if overlay is not None:
            st.image(
                overlay,
                caption=attention_caption,
                use_column_width=True,
            )
        else:
            # Show placeholder with message
            st.markdown(
                """
                <div style="background:#F3F4F6;border-radius:8px;padding:40px;text-align:center;
                color:#6B7280;font-size:13px;min-height:224px;display:flex;align-items:center;
                justify-content:center;flex-direction:column;">
                <div style='font-size:24px;margin-bottom:8px'>🔬</div>
                Grad-CAM unavailable<br/>
                <span style='font-size:11px'>Run with model checkpoint to enable</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(attention_caption)


def get_class_data(class_key: str) -> dict:
    """Get disease class data from DB or fallback dict."""
    conn = get_db()
    if conn and _DB_AVAILABLE:
        try:
            row = get_class_info(class_key, conn)
            if row:
                return row
        except Exception:
            pass
    return DISEASE_CLASS_DATA.get(class_key, {})


# ═══════════════════════════════════════════════════════════════════════════════
# Export Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def generate_csv_bytes(history: list[dict]) -> bytes:
    """Serialise session history list to CSV bytes."""
    if not history:
        return b""
    buf = io.StringIO()
    fieldnames = [
        "timestamp", "filename", "predicted_class", "predicted_display",
        "confidence", "flagged", "icd10_code", "inference_ms", "model_version",
    ] + [f"conf_{k}" for k in CLASS_NAMES]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in history:
        conf_all = row.get("confidence_all", {})
        flat = {f"conf_{k}": round(conf_all.get(k, 0.0), 4) for k in CLASS_NAMES}
        writer.writerow({**row, **flat})
    return buf.getvalue().encode("utf-8")


def generate_pdf_report(result, original_image: Image.Image) -> bytes:
    """
    Generate a single-result PDF report using reportlab.
    Returns PDF bytes.
    """
    try:
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm, mm
        from reportlab.platypus import (
            HRFlowable,
            Image as RLImage,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
    except ImportError:
        # Fallback: plain text PDF-like bytes (reportlab not installed)
        lines = [
            "OcuScan AI — Prediction Report",
            "=" * 42,
            f"Predicted Class  : {result.predicted_display}",
            f"Confidence       : {result.confidence:.1%}",
            f"ICD-10           : {result.icd10_code}",
            f"Flagged          : {'Yes — low confidence' if result.flagged else 'No'}",
            f"Session ID       : {result.session_id}",
            f"Model Version    : {result.model_version}",
            "",
            "All Class Scores:",
        ]
        for k, v in sorted(result.confidence_all.items(), key=lambda x: -x[1]):
            lines.append(f"  {DISPLAY_NAMES.get(k, k):<32}: {v:.4f}")
        lines += [
            "",
            "MEDICAL DISCLAIMER: This is a research tool. Not a certified medical device.",
            "All outputs require review by a qualified ophthalmologist.",
        ]
        return "\n".join(lines).encode("utf-8")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    teal = rl_colors.HexColor("#0F766E")
    dark = rl_colors.HexColor("#134E4A")
    grey = rl_colors.HexColor("#6B7280")
    red = rl_colors.HexColor("#DC2626")
    amber = rl_colors.HexColor("#D97706")
    green = rl_colors.HexColor("#16A34A")

    h1 = ParagraphStyle("H1", parent=styles["Heading1"], textColor=dark, fontSize=18, spaceAfter=4)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=teal, fontSize=13, spaceAfter=4, spaceBefore=12)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=15, spaceAfter=6)
    caption = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8, textColor=grey)
    mono = ParagraphStyle("Mono", parent=styles["Code"], fontSize=9, leading=13)

    # severity colour
    sev = DISEASE_CLASS_DATA.get(result.predicted_class, {}).get("severity_tier", "High")
    sev_colour = {"High": red, "Medium": amber, "None": green}.get(sev, grey)

    story = []

    # ── Header ──────────────────────────────────────────────────────────────
    story.append(Paragraph("OcuScan AI — Prediction Report", h1))
    story.append(HRFlowable(width="100%", thickness=2, color=teal, spaceAfter=10))

    # ── Metadata row ─────────────────────────────────────────────────────────
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    meta_data = [
        ["Generated", ts],
        ["Session ID", result.session_id[:16] + "…"],
        ["Model Version", result.model_version],
        ["Filename", result.filename or "—"],
    ]
    meta_table = Table(meta_data, colWidths=[4 * cm, 12 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (0, -1), grey),
        ("TEXTCOLOR", (1, 0), (1, -1), rl_colors.HexColor("#1C1C1E")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # ── Prediction card ───────────────────────────────────────────────────────
    story.append(Paragraph("Prediction Result", h2))
    pred_data = [
        ["Predicted Class", result.predicted_display],
        ["ICD-10 Code", result.icd10_code],
        ["Confidence", f"{result.confidence:.1%}"],
        ["Severity", sev],
        ["Flagged (low confidence)", "YES — unreliable result" if result.flagged else "No"],
        ["Emergency Class", "YES — SJS" if result.is_emergency_class else "No"],
        ["Sign Detection Class", "YES — Symblepharon" if result.is_sign_class else "No"],
    ]
    pred_table = Table(pred_data, colWidths=[6 * cm, 10 * cm])
    pred_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), grey),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [rl_colors.white, rl_colors.HexColor("#F8FAFC")]),
        ("TEXTCOLOR", (1, 2), (1, 2), sev_colour),  # confidence in severity colour
        ("FONTNAME", (1, 2), (1, 2), "Helvetica-Bold"),
        ("FONTSIZE", (1, 2), (1, 2), 12),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 10))

    # ── Warning banner ─────────────────────────────────────────────────────────
    banner = CLASS_BANNERS.get(result.predicted_class)
    if banner:
        from reportlab.platypus import KeepTogether
        from reportlab.lib.styles import ParagraphStyle as PS
        import html
        clean_body = banner["body"].replace("<strong>", "").replace("</strong>", "").replace("<br/>", " ").replace("<em>", "").replace("</em>", "")
        warn_style = ParagraphStyle(
            "Warn", parent=styles["Normal"], fontSize=9,
            backColor=rl_colors.HexColor(banner["bg"]),
            textColor=rl_colors.white,
            borderPadding=(8, 10, 8, 10),
            leading=14,
        )
        warn_title = f"{banner['icon']} {banner['title']}: {clean_body}"
        story.append(Paragraph(warn_title, warn_style))
        story.append(Spacer(1, 10))

    # ── All class scores ───────────────────────────────────────────────────────
    story.append(Paragraph("All Class Confidence Scores", h2))
    sorted_conf = sorted(result.confidence_all.items(), key=lambda x: -x[1])
    conf_data = [["Class", "Display Name", "Score"]]
    for k, v in sorted_conf:
        mark = "◀ predicted" if k == result.predicted_class else ""
        conf_data.append([k, DISPLAY_NAMES.get(k, k), f"{v:.4f}  {mark}"])
    conf_table = Table(conf_data, colWidths=[4 * cm, 7 * cm, 5 * cm])
    conf_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), teal),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#F8FAFC")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.3, rl_colors.HexColor("#E5E7EB")),
    ]))
    story.append(conf_table)
    story.append(Spacer(1, 12))

    # ── Clinical notes ────────────────────────────────────────────────────────
    cdata = get_class_data(result.predicted_class)
    if cdata:
        story.append(Paragraph("Clinical Notes", h2))
        if cdata.get("description"):
            story.append(Paragraph(f"<b>Description:</b> {cdata['description']}", body))
        if cdata.get("referral_text"):
            story.append(Paragraph(f"<b>Referral Recommendation:</b> {cdata['referral_text']}", body))
        if cdata.get("sjs_subtype_note"):
            story.append(Paragraph(f"<i>SJS Note: {cdata['sjs_subtype_note']}</i>", caption))
        story.append(Spacer(1, 8))

    # ── Symblepharon note ──────────────────────────────────────────────────────
    if result.is_sign_class:
        sign_note = ParagraphStyle(
            "SignNote", parent=styles["Normal"], fontSize=9,
            borderColor=red, borderWidth=1, borderPadding=8,
            textColor=rl_colors.HexColor("#991B1B"),
            leading=13,
        )
        story.append(Paragraph(
            "⚠ SIGN CLASS: Symblepharon is a structural sign, not a primary disease. "
            "This model detects the presence of conjunctival adhesion. "
            "The underlying cause must be investigated by a specialist.",
            sign_note,
        ))
        story.append(Spacer(1, 8))

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=grey, spaceBefore=8, spaceAfter=8))
    story.append(Paragraph(
        "MEDICAL DISCLAIMER: OcuScan AI is a research and educational tool. "
        "It is NOT a certified medical device and must not be used as the sole basis for any "
        "clinical decision. All outputs require review by a qualified ophthalmologist. "
        "For research and educational use only.",
        caption,
    ))

    doc.build(story)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# Demo mode — create synthetic PredictionResult for UI testing without model
# ═══════════════════════════════════════════════════════════════════════════════

def _make_demo_result(class_key: str = "ocp", confidence: float = 0.74) -> "PredictionResult":
    """Create a synthetic result for demo/test mode when checkpoint unavailable."""
    import random
    remaining = 1.0 - confidence
    others = [k for k in CLASS_NAMES if k != class_key]
    probs = [random.uniform(0, remaining) for _ in others]
    total = sum(probs)
    normalised = [p / total * remaining for p in probs]
    conf_all = {class_key: confidence}
    for k, v in zip(others, normalised):
        conf_all[k] = round(v, 6)

    # Use a plain namespace object instead of a class with a shadowed
    # `confidence` property — defining a class attribute and a same-named
    # property in one class body raises NameError at class-definition time
    # because the property isn't bound yet when the attribute assignment runs.
    from types import SimpleNamespace

    return SimpleNamespace(
        session_id=str(uuid.uuid4()),
        filename="demo_image.jpg",
        predicted_class=class_key,
        predicted_display=DISPLAY_NAMES.get(class_key, class_key),
        icd10_code=ICD10_CODES.get(class_key, ""),
        confidence=confidence,
        confidence_all=conf_all,
        is_sign_class=(class_key == "symblepharon"),
        is_emergency_class=(class_key == "sjs"),
        flagged=(confidence < LOW_CONFIDENCE_THRESHOLD),
        gradcam_path=None,
        model_version="v1.0.0-demo",
        inference_ms=412,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Screen: Home / Upload
# ═══════════════════════════════════════════════════════════════════════════════

def render_home_screen() -> None:
    st.markdown('<div class="screen-title">Upload & Analyse</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style='font-size:14px;color:#6B7280;margin-bottom:20px;line-height:1.6;'>
        Upload a close-up photograph of the anterior eye for AI-assisted classification of:
        <strong>Normal, OCP, OCP Chronic, Post-Viral DED, SJS</strong>, or <strong>Symblepharon</strong>.
        Accepted formats: JPEG, PNG, WEBP — maximum 10 MB.
        </div>
        """,
        unsafe_allow_html=True,
    )

    checkpoint_exists = Path(CHECKPOINT_PATH).exists()

    if not checkpoint_exists:
        st.info(
            "**Demo Mode** — No model checkpoint found at `models/phase2_best.pt`. "
            "Upload an image to preview the UI with a simulated result. "
            "Complete Phase 2 training to enable real inference.",
            icon="ℹ️",
        )

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Drop your anterior segment eye photograph here",
        type=ACCEPTED_TYPES,
        help="JPEG, PNG, or WEBP · Max 10 MB · Single image",
        label_visibility="visible",
    )

    image: Optional[Image.Image] = None
    filename: Optional[str] = None

    if uploaded_file is not None:
        # Size check
        if uploaded_file.size > MAX_UPLOAD_BYTES:
            st.error(
                f"⛔ File exceeds {MAX_UPLOAD_MB} MB ({uploaded_file.size / 1e6:.1f} MB). "
                "Please upload a smaller image.",
                icon=None,
            )
            return

        try:
            image = Image.open(uploaded_file).convert("RGB")
            filename = uploaded_file.name
        except Exception as e:
            st.error(f"Unable to read image file: {e}")
            return

        # Basic validation
        w, h = image.size
        if w < 64 or h < 64:
            st.error("Image is too small (minimum 64 × 64 pixels). Please upload a higher-resolution image.")
            return

        arr = np.array(image)
        mean_brightness = float(arr.mean())
        if mean_brightness < 5:
            st.error("Image appears completely black. Please check the file.")
            return
        if mean_brightness > 250:
            st.error("Image appears overexposed (completely white). Please use a properly lit photograph.")
            return

        # Preview
        st.markdown('<div class="ocuscan-card">', unsafe_allow_html=True)
        col_prev, col_info = st.columns([1, 1.5])
        with col_prev:
            st.image(image, caption="Image Preview", use_column_width=True)
        with col_info:
            st.markdown(f"**Filename:** `{filename}`")
            st.markdown(f"**Dimensions:** {w} × {h} px")
            st.markdown(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.markdown(f"**Brightness:** {mean_brightness:.0f} / 255")
        st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.uploaded_image = image
        st.session_state.uploaded_filename = filename

    # ── Analyse button ────────────────────────────────────────────────────────
    analyse_disabled = (image is None)
    if st.button(
        "🔬 Analyse Image",
        disabled=analyse_disabled,
        help="Click after uploading an image",
        use_container_width=False,
    ):
        _run_inference(image, filename, checkpoint_exists)

    render_medical_disclaimer()


def _run_inference(image: Image.Image, filename: str, checkpoint_exists: bool) -> None:
    """Run inference (or demo) and navigate to Results screen."""
    spinner_text = "Processing anterior segment image…"
    with st.spinner(spinner_text):
        try:
            if checkpoint_exists and _PREDICT_AVAILABLE:
                result = predict(
                    image=image,
                    checkpoint_path=CHECKPOINT_PATH,
                    device="cpu",
                    filename=filename,
                    session_id=st.session_state.session_id,
                    model_version=MODEL_VERSION,
                    generate_gradcam=True,
                )
            else:
                # Demo mode — cycle through classes for variety
                _demo_classes = CLASS_NAMES
                _idx = len(st.session_state.history) % len(_demo_classes)
                result = _make_demo_result(
                    class_key=_demo_classes[_idx],
                    confidence=round(0.55 + 0.30 * (len(st.session_state.history) % 3) / 2, 2),
                )

            # Persist to session history
            hist_row = {
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                "filename": filename,
                "predicted_class": result.predicted_class,
                "predicted_display": DISPLAY_NAMES.get(result.predicted_class, result.predicted_class),
                "confidence": result.confidence,
                "confidence_all": result.confidence_all,
                "icd10_code": ICD10_CODES.get(result.predicted_class, ""),
                "flagged": result.flagged,
                "inference_ms": result.inference_ms,
                "model_version": result.model_version,
                "_image": image,  # store for replay (memory, not db)
                "_result": result,
            }
            st.session_state.history.insert(0, hist_row)

            # Save to SQLite
            conn = get_db()
            if conn and _DB_AVAILABLE:
                try:
                    save_prediction(result, conn)
                except Exception:
                    pass

            st.session_state.current_result = result
            st.session_state._current_image = image
            st.session_state.screen = "Results"
            st.rerun()

        except Exception as exc:
            import traceback
            with st.expander("Technical details (click to expand)"):
                st.code(traceback.format_exc())
            st.error(
                f"An error occurred during analysis: {exc}. Please try again.",
                icon="⛔",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Screen: Results
# ═══════════════════════════════════════════════════════════════════════════════

def render_results_screen() -> None:
    result = st.session_state.get("current_result")
    image: Optional[Image.Image] = st.session_state.get("_current_image")

    if result is None:
        st.info("No result to display. Upload an image on the Home screen.")
        if st.button("← Go to Home"):
            st.session_state.screen = "Home"
            st.rerun()
        return

    cls_key: str = result.predicted_class
    conf: float = result.confidence
    cdata = get_class_data(cls_key)
    severity = cdata.get("severity_tier", "High")
    sev_label, sev_colour = SEVERITY_DISPLAY.get(severity, ("High Risk", "#DC2626"))
    icd10 = ICD10_CODES.get(cls_key, cdata.get("icd10_code", ""))
    conf_pct = conf * 100

    # Confidence badge colour
    if conf >= 0.75:
        badge_colour = "#16A34A"
    elif conf >= 0.50:
        badge_colour = "#D97706"
    else:
        badge_colour = "#DC2626"

    st.markdown('<div class="screen-title">Classification Result</div>', unsafe_allow_html=True)

    # ── Predicted class card ──────────────────────────────────────────────────
    st.markdown('<div class="ocuscan-card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="class-name-display">{DISPLAY_NAMES.get(cls_key, cls_key)}</div>
        <div class="icd10-display">ICD-10: {icd10}</div>
        <span class="conf-badge" style="background:{badge_colour};">{conf_pct:.1f}%</span>
        <span class="severity-pill" style="background:{sev_colour};">{sev_label}</span>
        """,
        unsafe_allow_html=True,
    )

    # Low confidence alert (above class banner if also flagged)
    if result.flagged:
        st.markdown(
            '<div class="low-conf-alert">⚠ <strong>Low Confidence</strong> — '
            f"Result confidence is {conf_pct:.1f}%, below the 60% reliability threshold. "
            "Do <strong>not</strong> use this output for clinical decisions without specialist review.</div>",
            unsafe_allow_html=True,
        )

    # Class-specific warning banner (mandatory)
    render_warning_banner(cls_key)

    # Symblepharon sign note
    if result.is_sign_class:
        st.markdown(
            """
            <div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:6px;
            padding:10px 14px;font-size:13px;color:#991B1B;margin:8px 0;">
            <strong>⚠ Sign Class:</strong> Symblepharon is a structural sign (fibrous adhesion between
            palpebral and bulbar conjunctiva), not a primary disease. This model detects its presence
            but does NOT identify the underlying cause. Underlying cause must be investigated by a specialist.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # SJS sub-type note
    if cls_key == "sjs":
        sjs_note = cdata.get("sjs_subtype_note") or "This class merges acute and chronic SJS sub-types."
        st.markdown(
            f'<div style="font-size:12px;color:#6B7280;font-style:italic;margin-top:4px;">ℹ {sjs_note}</div>',
            unsafe_allow_html=True,
        )

    if result.inference_ms:
        st.markdown(
            f'<div style="font-size:11px;color:#6B7280;margin-top:8px;text-align:right;">Inference: {result.inference_ms} ms</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Confidence bar chart ──────────────────────────────────────────────────
    st.markdown('<div class="ocuscan-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:16px;font-weight:600;color:#134E4A;margin-bottom:12px;">Class Confidence Scores</div>', unsafe_allow_html=True)
    render_confidence_chart(result.confidence_all, cls_key)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Grad-CAM panel ────────────────────────────────────────────────────────
    if image is not None:
        st.markdown('<div class="ocuscan-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:16px;font-weight:600;color:#134E4A;margin-bottom:12px;">Model Attention Map (Grad-CAM)</div>', unsafe_allow_html=True)
        render_gradcam_panel(result, image)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Clinical notes card ───────────────────────────────────────────────────
    st.markdown('<div class="ocuscan-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:16px;font-weight:600;color:#134E4A;margin-bottom:12px;">Clinical Notes</div>', unsafe_allow_html=True)

    if cdata.get("description"):
        st.markdown('<div class="notes-label">Description</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="notes-body">{cdata["description"]}</div>', unsafe_allow_html=True)

    if cdata.get("referral_text"):
        urgency_cls = "urgent" if severity == "High" else ""
        st.markdown(
            f'<div class="referral-box {urgency_cls}"><strong>Referral Recommendation:</strong> {cdata["referral_text"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Top 3 differential ────────────────────────────────────────────────────
    sorted_conf = sorted(result.confidence_all.items(), key=lambda x: -x[1])
    top3 = sorted_conf[:3]
    cols3 = st.columns(3)
    for i, (k, v) in enumerate(top3):
        with cols3[i]:
            label = "1st" if i == 0 else ("2nd" if i == 1 else "3rd")
            rank_colour = badge_colour if i == 0 else "#9CA3AF"
            st.markdown(
                f"""
                <div class="ocuscan-card" style="text-align:center;padding:14px 10px;">
                  <div style="font-size:11px;color:#6B7280;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">{label}</div>
                  <div style="font-size:13px;font-weight:700;color:#134E4A;margin:4px 0;">{DISPLAY_NAMES.get(k, k)}</div>
                  <div style="font-size:16px;font-weight:800;color:{rank_colour};">{v*100:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Action buttons ────────────────────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)
    col_a, col_b, col_c, col_d = st.columns([1.5, 1.5, 1.5, 1])

    with col_a:
        if st.button("🔄 Analyse Another Image", use_container_width=True):
            st.session_state.screen = "Home"
            st.session_state.current_result = None
            st.session_state._current_image = None
            st.rerun()

    with col_b:
        # PDF export
        try:
            pdf_bytes = generate_pdf_report(result, image)
            st.download_button(
                label="📄 Download Report (PDF)",
                data=pdf_bytes,
                file_name=f"ocuscan_{cls_key}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            if st.button("📄 Report (unavailable)", disabled=True, use_container_width=True):
                pass

    with col_c:
        # Single-result CSV
        row = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "filename": result.filename,
            "predicted_class": result.predicted_class,
            "predicted_display": DISPLAY_NAMES.get(result.predicted_class, result.predicted_class),
            "confidence": result.confidence,
            "confidence_all": result.confidence_all,
            "icd10_code": ICD10_CODES.get(result.predicted_class, ""),
            "flagged": result.flagged,
            "inference_ms": result.inference_ms,
            "model_version": result.model_version,
        }
        csv_bytes = generate_csv_bytes([row])
        st.download_button(
            label="📊 Export CSV",
            data=csv_bytes,
            file_name=f"ocuscan_result_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    render_medical_disclaimer()


# ═══════════════════════════════════════════════════════════════════════════════
# Screen: History
# ═══════════════════════════════════════════════════════════════════════════════

def render_history_screen() -> None:
    st.markdown('<div class="screen-title">Prediction History</div>', unsafe_allow_html=True)

    history = st.session_state.history

    # Top action row
    col_info, col_export, col_clear = st.columns([2, 1.2, 1])
    with col_info:
        if history:
            st.markdown(
                f"<div style='font-size:13px;color:#6B7280;padding-top:8px;'>"
                f"{len(history)} prediction{'s' if len(history) != 1 else ''} this session</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size:13px;color:#6B7280;padding-top:8px;'>No predictions yet</div>",
                unsafe_allow_html=True,
            )

    with col_export:
        if history:
            export_rows = [
                {k: v for k, v in row.items() if not k.startswith("_")}
                for row in history
            ]
            csv_all = generate_csv_bytes(export_rows)
            st.download_button(
                "📊 Export All (CSV)",
                data=csv_all,
                file_name=f"ocuscan_session_{st.session_state.session_id[:8]}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with col_clear:
        if history:
            if st.button(
                "🗑 Clear History",
                help="Remove all predictions from this session",
                use_container_width=True,
            ):
                st.session_state.history = []
                st.rerun()

    if not history:
        st.markdown(
            """
            <div class="ocuscan-card" style="text-align:center;padding:48px 20px;color:#6B7280;">
              <div style='font-size:32px;margin-bottom:12px;'>🔬</div>
              <div style='font-size:15px;font-weight:600;'>No predictions yet</div>
              <div style='font-size:13px;margin-top:6px;'>Upload an anterior segment image on the Home screen to get started.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Column headers
    st.markdown(
        """
        <div style="display:grid;grid-template-columns:140px 1fr 140px 100px 80px 60px;
        gap:8px;padding:8px 12px;background:#F8FAFC;border:1px solid #E5E7EB;
        border-radius:8px 8px 0 0;font-size:11px;font-weight:600;text-transform:uppercase;
        letter-spacing:0.05em;color:#6B7280;">
          <div>Timestamp</div>
          <div>Filename</div>
          <div>Predicted Class</div>
          <div>Confidence</div>
          <div>Flagged</div>
          <div>Action</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for i, row in enumerate(history):
        conf = row.get("confidence", 0.0)
        conf_pct = conf * 100
        if conf >= 0.75:
            badge_col = "#16A34A"
        elif conf >= 0.50:
            badge_col = "#D97706"
        else:
            badge_col = "#DC2626"

        flagged_html = (
            '<span style="color:#DC2626;font-weight:600;">⚠ Yes</span>'
            if row.get("flagged")
            else '<span style="color:#16A34A;">✓ No</span>'
        )

        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns:140px 1fr 140px 100px 80px 60px;
            gap:8px;padding:10px 12px;background:{'#FFFFFF' if i % 2 == 0 else '#F8FAFC'};
            border-left:1px solid #E5E7EB;border-right:1px solid #E5E7EB;
            border-bottom:1px solid #E5E7EB;font-size:13px;align-items:center;">
              <div style="color:#6B7280;font-size:12px;">{row.get('timestamp', '—')}</div>
              <div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
              font-family:monospace;font-size:12px;">{row.get('filename', '—')}</div>
              <div style="font-weight:600;color:#134E4A;">{row.get('predicted_display', row.get('predicted_class',''))}</div>
              <div><span class="conf-badge" style="background:{badge_col};font-size:12px;">{conf_pct:.1f}%</span></div>
              <div>{flagged_html}</div>
              <div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # View button in a real column (can't use markdown for buttons)
        if i == 0 or True:
            pass

    # Render view buttons below (streamlit limitation — buttons must be in st context)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Show replay buttons in a separate pass
    st.markdown("**Replay a result:**", unsafe_allow_html=False)
    for i, row in enumerate(history[:10]):  # show up to 10 replay buttons
        if st.button(
            f"▶ View: {row.get('predicted_display', row.get('predicted_class', ''))} "
            f"({row.get('confidence', 0)*100:.1f}%) — {row.get('filename', '')}",
            key=f"replay_{i}",
        ):
            result = row.get("_result")
            img = row.get("_image")
            if result is not None:
                st.session_state.current_result = result
                st.session_state._current_image = img
                st.session_state.screen = "Results"
                st.rerun()

    render_medical_disclaimer()


# ═══════════════════════════════════════════════════════════════════════════════
# Screen: About / Help
# ═══════════════════════════════════════════════════════════════════════════════

def render_about_screen() -> None:
    st.markdown('<div class="screen-title">About OcuScan AI</div>', unsafe_allow_html=True)

    # ── Project overview ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="ocuscan-card">
          <div style='font-size:16px;font-weight:600;color:#134E4A;margin-bottom:10px;'>Project Overview</div>
          <div style='font-size:14px;color:#1C1C1E;line-height:1.7;'>
            OcuScan AI is an AI-assisted anterior segment eye disease classification system. It analyses
            close-up slit-lamp or external photographs of the human eye and outputs a predicted diagnosis
            from six clinically defined categories, accompanied by a per-class confidence score and a
            Grad-CAM explainability heatmap.
            <br/><br/>
            The system targets rare and serious ocular surface conditions — not the common conditions
            found in widely available public datasets. <strong>All six classes are anterior segment
            conditions</strong>, distinguishing this project from fundus or retinal AI tools.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Disease class table ───────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:16px;font-weight:600;color:#134E4A;margin:20px 0 10px;">Disease Classes</div>',
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        """
        <div style="display:grid;grid-template-columns:50px 180px 80px 80px 1fr;
        gap:8px;padding:8px 12px;background:#F8FAFC;border:1px solid #E5E7EB;
        border-radius:8px 8px 0 0;font-size:11px;font-weight:600;text-transform:uppercase;
        letter-spacing:0.05em;color:#6B7280;">
          <div>Idx</div><div>Class</div><div>ICD-10</div><div>Severity</div><div>Description</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for i, cls_key in enumerate(CLASS_NAMES):
        cdata = get_class_data(cls_key)
        sev = cdata.get("severity_tier", "High")
        _, sev_col = SEVERITY_DISPLAY.get(sev, ("High Risk", "#DC2626"))
        sev_label, _ = SEVERITY_DISPLAY.get(sev, ("High Risk", "#DC2626"))
        icd = ICD10_CODES.get(cls_key, cdata.get("icd10_code", ""))
        desc = cdata.get("description", "")[:120] + ("…" if len(cdata.get("description", "")) > 120 else "")
        sign_tag = ' <span style="font-size:10px;background:#EFF6FF;color:#1D4ED8;padding:1px 6px;border-radius:10px;">sign</span>' if cdata.get("is_sign") else ""
        bg = "#FFFFFF" if i % 2 == 0 else "#F8FAFC"
        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns:50px 180px 80px 80px 1fr;
            gap:8px;padding:10px 12px;background:{bg};border-left:1px solid #E5E7EB;
            border-right:1px solid #E5E7EB;border-bottom:1px solid #E5E7EB;font-size:13px;align-items:start;">
              <div style="font-family:monospace;color:#6B7280;">{i}</div>
              <div style="font-weight:600;color:#134E4A;">{DISPLAY_NAMES.get(cls_key, cls_key)}{sign_tag}</div>
              <div style="font-family:monospace;font-size:12px;">{icd}</div>
              <div><span class="severity-pill" style="background:{sev_col};font-size:11px;">{sev_label}</span></div>
              <div style="color:#1C1C1E;line-height:1.5;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Important notes ───────────────────────────────────────────────────────
    st.markdown("<br/>", unsafe_allow_html=True)

    with st.expander("⚠ SJS Class — Sub-type Note", expanded=False):
        st.markdown(
            """
            The **SJS class** in v1.0 merges two distinct clinical sub-types:
            - **Acute SJS** — pseudomembrane formation, conjunctival necrosis, active inflammation
            - **Chronic SJS** — keratinisation, trichiasis, symblepharon formation

            These are intentionally merged because insufficient images per sub-type were available
            to train a reliable sub-classifier. Model performance on this class may be lower due to
            intra-class visual variation. Splitting into **SJS-Acute / SJS-Chronic** is planned for v2.0.
            """,
        )

    with st.expander("⚠ Symblepharon — Sign Detection Notice", expanded=False):
        st.markdown(
            """
            **Symblepharon is a structural sign, not a primary disease.**

            A symblepharon is a fibrous adhesion band between the palpebral (eyelid) and bulbar
            (eyeball) conjunctiva. It is a sequela of several conditions:
            - Stevens-Johnson Syndrome
            - Ocular Cicatricial Pemphigoid
            - Chemical or thermal burns
            - Severe chronic inflammation

            The OcuScan AI model detects the **presence** of the adhesion band in the image.
            It does **NOT** diagnose the underlying cause. Every Symblepharon result must be
            followed by specialist evaluation to identify and treat the aetiology.
            """,
        )

    with st.expander("📷 How to Take a Good Anterior Segment Photo", expanded=False):
        st.markdown(
            """
            For best classification accuracy, your image should:
            - **Be a close-up** — the eye should fill most of the frame
            - **Show the anterior segment clearly** — conjunctiva, cornea, and visible fornix
            - **Have good illumination** — slit-lamp images are ideal; external photos must be well-lit
            - **Be in focus** — blurry images reduce model reliability
            - **Avoid occlusion** — minimal eyelid/eyelash coverage of the region of interest
            - **Use JPEG or PNG** at ≥ 224 × 224 pixels resolution

            **Not suitable:** fundus photographs, OCT scans, retinal images, or systemic skin images.
            """,
        )

    # ── Model information ─────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:16px;font-weight:600;color:#134E4A;margin:20px 0 10px;">Model Information</div>',
        unsafe_allow_html=True,
    )

    checkpoint_exists = Path(CHECKPOINT_PATH).exists()
    model_status = "✅ Loaded" if checkpoint_exists else "⚠ Not found (demo mode)"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="ocuscan-card">
            <div class="notes-label">Architecture</div>
            <div class="notes-body"><strong>EfficientNetB0</strong> (ImageNet pretrained via timm)<br/>
            Custom 6-class classifier head: GlobalAvgPool → Dropout(0.5) → Linear(1280, 256) →
            ReLU → Dropout(0.3) → Linear(256, 6)</div>
            <div class="notes-label">Input</div>
            <div class="notes-body">224 × 224 × 3 RGB — anterior segment photographs</div>
            <div class="notes-label">Output</div>
            <div class="notes-body">6-class softmax probability vector</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="ocuscan-card">
            <div class="notes-label">Training</div>
            <div class="notes-body">
            <strong>Phase 1:</strong> Frozen backbone, head only, AdamW lr=1e-3<br/>
            <strong>Phase 2:</strong> Top-3 blocks unfrozen, lr=5e-6<br/>
            MixUp α=0.2 · Label smoothing ε=0.1 · EMA decay=0.99
            </div>
            <div class="notes-label">Checkpoint</div>
            <div class="notes-body"><code>{CHECKPOINT_PATH}</code><br/>{model_status}</div>
            <div class="notes-label">Version</div>
            <div class="notes-body">{MODEL_VERSION}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Dataset provenance ────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="ocuscan-card">
        <div style='font-size:14px;font-weight:600;color:#134E4A;margin-bottom:6px;'>Dataset Provenance</div>
        <div style='font-size:13px;color:#1C1C1E;line-height:1.6;'>
        All training images are proprietary clinical photographs from anterior segment examinations.
        <strong>No public dataset was used.</strong> Augmentation (9 medically realistic transforms
        including CLAHE, HueSaturationValue, CoarseDropout, and GaussNoise) is the primary data
        expansion strategy. Augmented images are never included in the validation or test sets.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Full disclaimer ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="medical-disclaimer" style="margin-top:24px;">
        <strong>Full Medical Disclaimer:</strong> OcuScan AI v1.0.0 is a research prototype
        developed for educational and research purposes only. It is NOT a licensed or certified
        medical device in any jurisdiction (including CE Mark or FDA 510(k)). It must not be used
        as the sole or primary basis for any clinical diagnosis, treatment decision, or patient
        management plan. All AI-generated outputs require verification by a qualified
        ophthalmologist or relevant clinical specialist. Confidence scores below 60% are
        automatically flagged as unreliable. OcuScan AI v2.0 · June 2025 · Confidential.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main Application Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="OcuScan AI — Anterior Segment Classifier",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": "OcuScan AI v1.0.0 — For research use only",
        },
    )

    inject_css()
    init_session_state()

    # Sidebar navigation
    selected_screen = render_sidebar()

    # The sidebar radio widget retains its previous value across reruns even
    # when a programmatic transition (Analyse -> Results, Replay -> Results,
    # Analyse Another -> Home) set st.session_state.screen earlier in the same
    # rerun. Only let the radio drive navigation when the user actually
    # changed it this run; otherwise the stale radio value would clobber the
    # programmatic transition every time.
    prev_radio_value = st.session_state.get("_prev_nav_radio")
    if selected_screen != prev_radio_value:
        st.session_state.screen = selected_screen
    st.session_state["_prev_nav_radio"] = selected_screen

    # Screen router
    screen = st.session_state.screen
    if screen == "Home":
        render_home_screen()
    elif screen == "Results":
        render_results_screen()
    elif screen == "History":
        render_history_screen()
    elif screen == "About":
        render_about_screen()
    else:
        render_home_screen()


if __name__ == "__main__":
    main()
