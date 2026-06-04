"""
OcuScan AI — src/utils.py
Shared utilities: image loading, class label mapping, CSV export,
temperature scaling, and helper functions used across the codebase.
"""

import io
import json
import csv
import time
import uuid
import sqlite3
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── Re-export CLASS_NAMES for convenience ─────────────────────────────────────
from dataset import CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES

PROJECT_ROOT = Path(__file__).parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: Union[str, Path]) -> Image.Image:
    """Load any supported image format as PIL RGB."""
    return Image.open(path).convert('RGB')


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load image from raw bytes (e.g. Streamlit UploadedFile.read())."""
    return Image.open(io.BytesIO(data)).convert('RGB')


def validate_image(image: Image.Image,
                   min_size: int = 64,
                   max_size: int = 4096) -> tuple[bool, str]:
    """
    Basic anterior segment image validation.
    Returns (is_valid, reason_string).
    """
    w, h = image.size
    if w < min_size or h < min_size:
        return False, f"Image too small ({w}×{h}). Minimum {min_size}px."
    if w > max_size or h > max_size:
        return False, f"Image too large ({w}×{h}). Maximum {max_size}px."
    if image.mode not in ('RGB', 'L', 'RGBA'):
        return False, f"Unsupported colour mode: {image.mode}"
    return True, "OK"


# ─────────────────────────────────────────────────────────────────────────────
# Class label mapping
# ─────────────────────────────────────────────────────────────────────────────

# ICD-10 codes per class (for reporting)
ICD10_CODES = {
    'normal':         'Z01.01',
    'ocp':            'L12.1',
    'ocp_chronic':    'L12.1',
    'post_viral_ded': 'H04.12',
    'sjs':            'L51.1',
    'symblepharon':   'H11.23',
}

SEVERITY_TIERS = {
    'normal':         'None',
    'ocp':            'Medium',
    'ocp_chronic':    'High',
    'post_viral_ded': 'Low',
    'sjs':            'High',
    'symblepharon':   'High',
}

BANNER_COLOURS = {
    'normal':         '#2E7D32',   # green
    'ocp':            '#F57C00',   # amber
    'ocp_chronic':    '#C62828',   # red
    'post_viral_ded': '#1565C0',   # blue
    'sjs':            '#FF6F00',   # amber-emergency
    'symblepharon':   '#B71C1C',   # deep red
}

IS_SIGN_CLASS = {cls: (cls == 'symblepharon') for cls in CLASS_NAMES}
IS_EMERGENCY_CLASS = {cls: (cls == 'sjs') for cls in CLASS_NAMES}

CONFIDENCE_THRESHOLD = 0.60   # below this → flagged for review


def class_key_to_display(class_key: str) -> str:
    """Map class_key → human-readable display name."""
    display_map = {
        'normal':         'Normal',
        'ocp':            'OCP (Ocular Cicatricial Pemphigoid)',
        'ocp_chronic':    'OCP Chronic',
        'post_viral_ded': 'Post-Viral DED',
        'sjs':            'SJS (Stevens-Johnson Syndrome)',
        'symblepharon':   'Symblepharon',
    }
    return display_map.get(class_key, class_key)


# ─────────────────────────────────────────────────────────────────────────────
# PredictionResult dataclass (canonical data model — matches backend schema)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    session_id: str
    filename: str
    predicted_class: str           # class_key
    predicted_display: str         # human display name
    confidence: float              # top-1 confidence 0.0–1.0
    confidence_all: dict           # {class_key: score} all 6 classes
    is_sign_class: bool            # True for symblepharon
    is_emergency_class: bool       # True for sjs
    gradcam_path: Optional[str]    = None
    model_version: str             = 'v1.0.0'
    inference_ms: Optional[int]    = None
    flagged: bool                  = False   # True if confidence < 0.60

    def to_dict(self) -> dict:
        d = asdict(self)
        d['confidence_json'] = json.dumps(d.pop('confidence_all'))
        return d


def build_prediction_result(probs: torch.Tensor,
                             filename: str,
                             session_id: Optional[str] = None,
                             model_version: str = 'v1.0.0',
                             inference_ms: Optional[int] = None) -> PredictionResult:
    """
    Build PredictionResult from softmax probability tensor.

    Args:
        probs:        1-D tensor of shape [NUM_CLASSES] — softmax probabilities.
        filename:     Original uploaded filename.
        session_id:   UUID string; generated if None.
        model_version: Checkpoint tag.
        inference_ms: Latency from preprocess → forward.
    """
    if probs.ndim == 2:
        probs = probs.squeeze(0)

    probs_np = probs.detach().cpu().numpy()
    top_idx = int(np.argmax(probs_np))
    top_class = CLASS_NAMES[top_idx]
    top_conf = float(probs_np[top_idx])

    confidence_all = {cls: float(probs_np[i]) for i, cls in enumerate(CLASS_NAMES)}

    return PredictionResult(
        session_id=session_id or str(uuid.uuid4()),
        filename=filename,
        predicted_class=top_class,
        predicted_display=class_key_to_display(top_class),
        confidence=top_conf,
        confidence_all=confidence_all,
        is_sign_class=IS_SIGN_CLASS[top_class],
        is_emergency_class=IS_EMERGENCY_CLASS[top_class],
        model_version=model_version,
        inference_ms=inference_ms,
        flagged=top_conf < CONFIDENCE_THRESHOLD,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Temperature scaling (post-hoc calibration)
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaler(torch.nn.Module):
    """
    Single-parameter temperature scaling for model calibration.
    Applied after training; does not change accuracy, only confidence calibration.
    Usage:
        scaler = TemperatureScaler()
        scaler.fit(logits_val, labels_val)
        calibrated_probs = scaler(logits)
    """

    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / self.temperature, dim=-1)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor,
            lr: float = 0.01, max_iter: int = 50) -> float:
        """
        Optimise temperature on validation logits/labels using NLL loss.
        Returns final temperature value.
        """
        self.train()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        nll_criterion = torch.nn.CrossEntropyLoss()

        def eval_fn():
            optimizer.zero_grad()
            scaled = logits / self.temperature
            loss = nll_criterion(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        self.eval()
        t = self.temperature.item()
        print(f"  Temperature calibrated: T = {t:.4f}")
        return t

    def save(self, path: Union[str, Path]):
        torch.save({'temperature': self.temperature.item()}, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TemperatureScaler':
        scaler = cls()
        data = torch.load(path, map_location='cpu')
        scaler.temperature = torch.nn.Parameter(
            torch.tensor([data['temperature']])
        )
        return scaler


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def export_predictions_csv(results: list[PredictionResult],
                            output_path: Union[str, Path]) -> Path:
    """
    Export a list of PredictionResult objects to CSV.
    Returns the output path.
    """
    output_path = Path(output_path)
    fieldnames = [
        'session_id', 'filename', 'predicted_class', 'predicted_display',
        'confidence', 'is_sign_class', 'is_emergency_class',
        'flagged', 'model_version', 'inference_ms', 'gradcam_path',
    ] + [f'conf_{cls}' for cls in CLASS_NAMES]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                'session_id': r.session_id,
                'filename': r.filename,
                'predicted_class': r.predicted_class,
                'predicted_display': r.predicted_display,
                'confidence': f'{r.confidence:.4f}',
                'is_sign_class': int(r.is_sign_class),
                'is_emergency_class': int(r.is_emergency_class),
                'flagged': int(r.flagged),
                'model_version': r.model_version,
                'inference_ms': r.inference_ms or '',
                'gradcam_path': r.gradcam_path or '',
            }
            for cls in CLASS_NAMES:
                row[f'conf_{cls}'] = f"{r.confidence_all.get(cls, 0.0):.4f}"
            writer.writerow(row)

    print(f"  CSV exported → {output_path}  ({len(results)} rows)")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def seed_everything(seed: int = 42):
    """Seed Python random, NumPy, and PyTorch for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_confidence(conf: float) -> str:
    """Format confidence as percentage string."""
    return f"{conf * 100:.1f}%"


if __name__ == '__main__':
    print("OcuScan AI — src/utils.py")
    print("Device:", get_device())
    print("CLASS_NAMES:", CLASS_NAMES)
    print("ICD-10 codes:", ICD10_CODES)
    print("[OK] utils.py loaded successfully.")
