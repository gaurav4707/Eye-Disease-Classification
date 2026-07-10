"""
OcuScan AI — src/predict.py
Phase 2 Day 14 / Phase 3 integration
Single-image inference pipeline: load checkpoint → preprocess → forward → PredictionResult.
Used by both the evaluation suite (Phase 3) and the Streamlit app (Phase 4).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── Class constants (must match training order exactly) ────────────────────────
CLASS_NAMES: list[str] = [
    "normal",          # idx 0
    "ocp",             # idx 1
    "ocp_chronic",     # idx 2
    "post_viral_ded",  # idx 3
    "sjs",             # idx 4
    "symblepharon",    # idx 5
]

DISPLAY_NAMES: dict[str, str] = {
    "normal":         "Normal",
    "ocp":            "OCP (Ocular Cicatricial Pemphigoid)",
    "ocp_chronic":    "OCP Chronic",
    "post_viral_ded": "Post-Viral DED",
    "sjs":            "SJS (Stevens-Johnson Syndrome)",
    "symblepharon":   "Symblepharon",
}

ICD10_CODES: dict[str, str] = {
    "normal":         "Z01.01",
    "ocp":            "H10.40",
    "ocp_chronic":    "H10.40",
    "post_viral_ded": "H04.123",
    "sjs":            "L51.1",
    "symblepharon":   "H11.231",
}

# Clinical thresholds
LOW_CONFIDENCE_THRESHOLD = 0.60
EMERGENCY_CLASSES: set[str] = {"sjs"}
SIGN_CLASSES: set[str] = {"symblepharon"}


# ── PredictionResult dataclass ─────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """
    Full inference result — one instance per uploaded image.
    Passed between predict.py → Streamlit screens → database.
    """
    session_id: str
    filename: str

    # Top-1 prediction
    predicted_class: str          # class_key  e.g. 'ocp_chronic'
    predicted_display: str        # human name e.g. 'OCP Chronic'
    confidence: float             # top-1 softmax score  0.0–1.0

    # All-class scores
    confidence_all: dict[str, float]  # {class_key: score} for all 6 classes

    # Clinical flags
    is_sign_class: bool           # True for symblepharon
    is_emergency_class: bool      # True for sjs
    flagged: bool                 # True if confidence < LOW_CONFIDENCE_THRESHOLD

    # Optional artefacts
    image_path: Optional[str] = None
    gradcam_path: Optional[str] = None
    model_version: str = "v1.0.0"
    inference_ms: Optional[int] = None

    # Derived helpers
    @property
    def icd10_code(self) -> str:
        return ICD10_CODES.get(self.predicted_class, "")

    @property
    def confidence_json(self) -> str:
        return json.dumps(self.confidence_all)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["icd10_code"] = self.icd10_code
        d["confidence_json"] = self.confidence_json
        return d

    def top_n(self, n: int = 3) -> list[tuple[str, str, float]]:
        """Return top-n (class_key, display_name, score) tuples sorted by score."""
        sorted_classes = sorted(
            self.confidence_all.items(), key=lambda x: x[1], reverse=True
        )
        return [
            (k, DISPLAY_NAMES.get(k, k), round(v, 4))
            for k, v in sorted_classes[:n]
        ]


# ── Model loader (lazy singleton) ─────────────────────────────────────────────

_loaded_model = None
_loaded_checkpoint_path: Optional[str] = None


def load_model(checkpoint_path: str, device: str = "cpu"):
    """
    Load OcuScanModel from a .pt checkpoint.
    Caches the model so repeated calls don't reload from disk.
    """
    global _loaded_model, _loaded_checkpoint_path

    if _loaded_model is not None and _loaded_checkpoint_path == checkpoint_path:
        return _loaded_model

    try:
        from model import OcuScanModel
    except ImportError:
        raise ImportError(
            "src/model.py not found. Complete Phase 2 (Day 8) before running predict.py."
        )

    model = OcuScanModel(num_classes=len(CLASS_NAMES))
    model.load_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    _loaded_model = model
    _loaded_checkpoint_path = checkpoint_path
    return model


def clear_model_cache() -> None:
    """Force next call to load_model() to reload from disk (e.g. after checkpoint update)."""
    global _loaded_model, _loaded_checkpoint_path
    _loaded_model = None
    _loaded_checkpoint_path = None


# ── Preprocessing ──────────────────────────────────────────────────────────────

def _get_val_transform():
    """Return val_transform from augmentation.py, or a minimal fallback."""
    try:
        from augmentation import val_transform
        return val_transform
    except ImportError:
        import torchvision.transforms as T
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def preprocess(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL Image for inference.
    Returns a [1, 3, 224, 224] float32 tensor.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = _get_val_transform()
    transformed = transform(image=np.array(image))
    tensor = transformed["image"] if isinstance(transformed, dict) else transformed
    return tensor.unsqueeze(0)         # [1, 3, 224, 224]


# ── Core predict function ──────────────────────────────────────────────────────

def predict(
    image: Image.Image,
    checkpoint_path: str = "models/phase2_best.pt",
    device: str = "cpu",
    filename: str = "upload.jpg",
    session_id: Optional[str] = None,
    model_version: str = "v1.0.0",
    generate_gradcam: bool = False,
    image_save_path: Optional[str] = None,
) -> PredictionResult:
    """
    Full single-image inference pipeline.

    Parameters
    ----------
    image            : PIL.Image (any mode; converted to RGB internally)
    checkpoint_path  : path to .pt checkpoint file
    device           : 'cuda' or 'cpu'
    filename         : original upload filename (for display/logging)
    session_id       : browser session UUID; generated if None
    model_version    : version tag from model_versions table
    generate_gradcam : if True, compute Grad-CAM overlay and attach path to result
    image_save_path  : if provided, save the processed image here

    Returns
    -------
    PredictionResult dataclass with all fields populated
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    # ── Validate image ─────────────────────────────────────────────────────────
    from utils import validate_image
    is_valid, reason = validate_image(image)
    if not is_valid:
        raise ValueError(f"Image validation failed: {reason}")

    # ── Save original image (optional) ────────────────────────────────────────
    if image_save_path:
        Path(image_save_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(image_save_path)

    # ── Load model ─────────────────────────────────────────────────────────────
    model = load_model(checkpoint_path, device)

    # ── Preprocess ────────────────────────────────────────────────────────────
    tensor = preprocess(image).to(device)

    # ── Forward pass ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)                           # [1, 6]
        probs = F.softmax(logits, dim=1).squeeze(0)     # [6]
    inference_ms = int((time.perf_counter() - t0) * 1000)

    # ── Build result ───────────────────────────────────────────────────────────
    probs_np = probs.cpu().numpy()
    top_idx = int(probs_np.argmax())
    top_key = CLASS_NAMES[top_idx]
    top_conf = float(probs_np[top_idx])

    confidence_all = {
        k: round(float(probs_np[i]), 6) for i, k in enumerate(CLASS_NAMES)
    }

    result = PredictionResult(
        session_id=session_id,
        filename=filename,
        predicted_class=top_key,
        predicted_display=DISPLAY_NAMES[top_key],
        confidence=round(top_conf, 6),
        confidence_all=confidence_all,
        is_sign_class=(top_key in SIGN_CLASSES),
        is_emergency_class=(top_key in EMERGENCY_CLASSES),
        flagged=(top_conf < LOW_CONFIDENCE_THRESHOLD),
        image_path=image_save_path,
        gradcam_path=None,
        model_version=model_version,
        inference_ms=inference_ms,
    )

    # ── Grad-CAM (optional) ────────────────────────────────────────────────────
    if generate_gradcam:
        try:
            from gradcam import GradCAM
            gcam = GradCAM(model)
            # Re-run with grad enabled (tensor must leave no_grad context)
            tensor_grad = preprocess(image).to(device)
            heatmap = gcam.compute(tensor_grad, class_idx=top_idx)
            overlay = GradCAM.overlay(image, heatmap)
            gcam.remove_hooks()

            # Save overlay
            gcam_dir = Path("results/gradcam")
            gcam_dir.mkdir(parents=True, exist_ok=True)
            gcam_filename = f"gradcam_{session_id[:8]}_{top_key}.png"
            gcam_path = str(gcam_dir / gcam_filename)
            overlay.save(gcam_path)
            result.gradcam_path = gcam_path
        except Exception as e:
            print(f"[predict] Grad-CAM generation failed (non-fatal): {e}")

    return result


def predict_from_path(
    image_path: str,
    **kwargs,
) -> PredictionResult:
    """
    Convenience wrapper: load image from disk path, then call predict().
    Passes the filename from the path unless overridden in kwargs.
    """
    path = Path(image_path)
    image = Image.open(path).convert("RGB")
    kwargs.setdefault("filename", path.name)
    return predict(image, **kwargs)


# ── Batch predict ──────────────────────────────────────────────────────────────

def predict_batch(
    image_paths: list[str],
    checkpoint_path: str = "models/phase2_best.pt",
    device: str = "cpu",
    session_id: Optional[str] = None,
    generate_gradcam: bool = False,
) -> list[PredictionResult]:
    """
    Run prediction on a list of image file paths.
    All share the same session_id if provided.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    results = []
    for path in image_paths:
        try:
            r = predict_from_path(
                path,
                checkpoint_path=checkpoint_path,
                device=device,
                session_id=session_id,
                generate_gradcam=generate_gradcam,
            )
            results.append(r)
        except Exception as e:
            print(f"[predict_batch] Skipping {path}: {e}")
    return results


# ── CLI — 6-class smoke test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

    parser = argparse.ArgumentParser(description="OcuScan AI — Single-image inference test")
    parser.add_argument(
        "--checkpoint", default="models/phase2_best.pt", help="Path to .pt checkpoint"
    )
    parser.add_argument(
        "--image", default=None,
        help="Path to a single image. If omitted, runs a 6-class smoke test using dataset/.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM overlay")
    args = parser.parse_args()

    if args.image:
        # Single image
        result = predict_from_path(
            args.image,
            checkpoint_path=args.checkpoint,
            device=args.device,
            generate_gradcam=args.gradcam,
        )
        print(f"\nFile    : {result.filename}")
        print(f"Predicted : {result.predicted_display} ({result.predicted_class})")
        print(f"Confidence: {result.confidence:.4f}  {'⚠ LOW' if result.flagged else ''}")
        print(f"ICD-10  : {result.icd10_code}")
        print(f"Emergency : {result.is_emergency_class}  |  Sign class: {result.is_sign_class}")
        print(f"Inference : {result.inference_ms} ms")
        print("\nAll class scores:")
        for k, disp, score in result.top_n(6):
            bar = "█" * int(score * 30)
            print(f"  {disp:<32}: {score:.4f}  {bar}")
        if result.gradcam_path:
            print(f"\nGrad-CAM: {result.gradcam_path}")
    else:
        # 6-class smoke test
        print("Running 6-class smoke test (1 image per class from dataset/)...\n")
        dataset_dir = Path("dataset")
        passed, failed = 0, 0
        for class_key in CLASS_NAMES:
            cls_dir = dataset_dir / class_key
            if not cls_dir.exists():
                print(f"  [{class_key}] SKIP — folder not found")
                continue
            imgs = sorted(
                [f for f in cls_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            )
            if not imgs:
                print(f"  [{class_key}] SKIP — no images found")
                continue
            try:
                result = predict_from_path(
                    str(imgs[0]),
                    checkpoint_path=args.checkpoint,
                    device=args.device,
                    generate_gradcam=args.gradcam,
                )
                correct = "✓" if result.predicted_class == class_key else "✗"
                print(
                    f"  [{class_key:<16}] pred={result.predicted_class:<16}"
                    f" conf={result.confidence:.3f}  {correct}  {result.inference_ms}ms"
                )
                passed += 1
            except Exception as e:
                print(f"  [{class_key}] FAIL — {e}")
                failed += 1

        print(f"\nSmoke test complete: {passed} passed, {failed} failed")
        if failed:
            sys.exit(1)
