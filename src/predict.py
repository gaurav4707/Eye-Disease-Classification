"""
OcuScan AI — src/predict.py
Single-image inference pipeline.

Returns a PredictionResult dataclass used by both the Streamlit app and
the evaluation script.

TTA (Test-Time Augmentation):
  Two variants averaged by default: original + horizontal flip.
  Keeps latency reasonable for the Streamlit app (≈2× single-pass time).

Usage:
  from predict import Predictor
  predictor = Predictor(checkpoint_path='models/phase2_best.pt')
  result = predictor.predict('path/to/image.jpg')

CLI:
  python src/predict.py --image path/to/image.jpg
  python src/predict.py --image path/to/image.jpg --no-tta --gradcam
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from augmentation import get_val_transform
from dataset import CLASS_NAMES, NUM_CLASSES
from model import OcuScanModel, load_checkpoint
from utils import (PredictionResult, build_prediction_result, get_device,
                   load_image, load_image_from_bytes, validate_image)

MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
GRADCAM_DIR = PROJECT_ROOT / 'results' / 'gradcam'


# ─────────────────────────────────────────────────────────────────────────────
# Predictor class
# ─────────────────────────────────────────────────────────────────────────────

class Predictor:
    """
    Wraps the OcuScanModel for single-image inference.

    Loads the model once on init; predict() is fast thereafter.
    Grad-CAM is computed on demand (set save_gradcam=True or call separately).

    Args:
        checkpoint_path: Path to .pt checkpoint (default: models/phase2_best.pt,
                         fallback: models/phase1_best.pt).
        device:          'cpu', 'cuda', 'mps', or None (auto-detect).
        image_size:      Resize target (must match training; default 224).
        use_tta:         Average original + horizontal-flip predictions.
    """

    def __init__(self,
                 checkpoint_path: Optional[Union[str, Path]] = None,
                 device: Optional[str] = None,
                 image_size: int = 224,
                 use_tta: bool = True):

        self.device = (torch.device(device) if device
                       else get_device())
        self.image_size = image_size
        self.use_tta = use_tta
        self.transform = get_val_transform(image_size=image_size)
        self.model_version = 'v1.0.0'

        # Resolve checkpoint path
        if checkpoint_path is None:
            for candidate in [
                MODELS_DIR / 'phase2_best.pt',
                MODELS_DIR / 'phase1_best.pt',
            ]:
                if candidate.exists():
                    checkpoint_path = candidate
                    break
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                "No checkpoint found. Train first with src/train.py, "
                "or pass checkpoint_path explicitly."
            )
        checkpoint_path = Path(checkpoint_path)

        # Load model
        self.model = OcuScanModel(pretrained=False, freeze_backbone=False).to(self.device)
        ckpt = load_checkpoint(checkpoint_path, self.model, device=str(self.device))
        self.model.eval()

        self.model_version = ckpt.get('version_tag', checkpoint_path.stem)
        print(f"  [Predictor] Ready — checkpoint: {checkpoint_path.name} | "
              f"device: {self.device} | TTA: {use_tta}")

    # ── Core inference ────────────────────────────────────────────────────────

    def _preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        """PIL RGB → [1, 3, H, W] float32 tensor on self.device."""
        img_np = np.array(pil_image.convert('RGB'))
        transformed = self.transform(image=img_np)
        tensor = transformed['image']       # [3, H, W]
        return tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]

    @torch.no_grad()
    def _forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass → softmax probabilities [NUM_CLASSES]."""
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0)  # [6]
        return probs

    @torch.no_grad()
    def _forward_tta(self, pil_image: Image.Image) -> torch.Tensor:
        """
        TTA: average probabilities over two variants.
          1. Original image
          2. Horizontal flip
        Keeping to 2 variants balances accuracy gain vs latency.
        """
        img_np = np.array(pil_image.convert('RGB'))
        variants = [
            img_np,
            np.fliplr(img_np),
        ]
        probs_list = []
        for v in variants:
            t = self.transform(image=v)['image'].unsqueeze(0).to(self.device)
            logits = self.model(t)
            probs_list.append(F.softmax(logits, dim=-1))

        avg_probs = torch.stack(probs_list, dim=0).mean(dim=0).squeeze(0)
        return avg_probs  # [6]

    def predict(self,
                source: Union[str, Path, bytes, Image.Image],
                filename: Optional[str] = None,
                session_id: Optional[str] = None,
                save_gradcam: bool = False) -> PredictionResult:
        """
        Run inference on a single image.

        Args:
            source:       File path, raw bytes, or PIL Image.
            filename:     Original filename (for display / DB storage).
            session_id:   Session UUID string; auto-generated if None.
            save_gradcam: If True, compute and save Grad-CAM heatmap to
                          results/gradcam/ and set result.gradcam_path.

        Returns:
            PredictionResult dataclass (see utils.py).
        """
        # ── Load image ────────────────────────────────────────────────────────
        if isinstance(source, (str, Path)):
            pil_image = load_image(source)
            if filename is None:
                filename = Path(source).name
        elif isinstance(source, bytes):
            pil_image = load_image_from_bytes(source)
        elif isinstance(source, Image.Image):
            pil_image = source.convert('RGB')
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        if filename is None:
            filename = 'image.jpg'

        # ── Validate ──────────────────────────────────────────────────────────
        is_valid, reason = validate_image(pil_image)
        if not is_valid:
            raise ValueError(f"Invalid image: {reason}")

        # ── Inference ─────────────────────────────────────────────────────────
        t_start = time.perf_counter()

        self.model.eval()
        if self.use_tta:
            probs = self._forward_tta(pil_image)
        else:
            tensor = self._preprocess(pil_image)
            probs = self._forward(tensor)

        inference_ms = int((time.perf_counter() - t_start) * 1000)

        # ── Build result ──────────────────────────────────────────────────────
        result = build_prediction_result(
            probs=probs,
            filename=filename,
            session_id=session_id,
            model_version=self.model_version,
            inference_ms=inference_ms,
        )

        # ── Optional Grad-CAM ─────────────────────────────────────────────────
        if save_gradcam:
            gradcam_path = self._compute_and_save_gradcam(pil_image, result)
            result.gradcam_path = str(gradcam_path)

        return result

    def _compute_and_save_gradcam(self,
                                   pil_image: Image.Image,
                                   result: PredictionResult) -> Path:
        """
        Compute Grad-CAM for predicted class, save overlay to disk.
        Returns path to saved file.
        """
        import cv2

        GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

        # Grad-CAM needs gradients — use single pass without TTA
        tensor = self._preprocess(pil_image)
        class_idx = CLASS_NAMES.index(result.predicted_class)

        heatmap = self.model.get_gradcam(tensor, class_idx=class_idx,
                                         output_size=self.image_size)

        # Build original image array (resize to match heatmap)
        import numpy as np
        orig_np = np.array(pil_image.convert('RGB'))
        orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
        orig_resized = cv2.resize(orig_bgr, (self.image_size, self.image_size))

        overlay = self.model.overlay_gradcam(orig_resized, heatmap, alpha=0.4)

        out_path = GRADCAM_DIR / f"{result.session_id}_{result.predicted_class}.jpg"
        cv2.imwrite(str(out_path), overlay)
        print(f"  [Grad-CAM] Saved → {out_path}")
        return out_path

    def predict_batch(self,
                      sources: list,
                      session_id: Optional[str] = None,
                      save_gradcam: bool = False) -> list[PredictionResult]:
        """
        Run predict() for a list of image sources.
        Returns list of PredictionResult in input order.
        """
        results = []
        for source in sources:
            try:
                r = self.predict(source, session_id=session_id,
                                 save_gradcam=save_gradcam)
                results.append(r)
            except Exception as e:
                print(f"  [WARN] predict failed for {source}: {e}")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OcuScan AI — single-image inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to anterior segment image')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to .pt checkpoint (auto-detects if omitted)')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable Test-Time Augmentation')
    parser.add_argument('--gradcam', action='store_true',
                        help='Compute and save Grad-CAM overlay')
    parser.add_argument('--image-size', type=int, default=224)
    args = parser.parse_args()

    print("OcuScan AI — src/predict.py")
    print("=" * 50)

    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        use_tta=not args.no_tta,
    )

    result = predictor.predict(
        source=args.image,
        save_gradcam=args.gradcam,
    )

    print(f"\n  File:       {result.filename}")
    print(f"  Prediction: {result.predicted_display}")
    print(f"  Confidence: {result.confidence * 100:.1f}%")
    print(f"  Flagged:    {result.flagged}  (< 60% threshold)")
    print(f"  Is sign:    {result.is_sign_class}  (symblepharon)")
    print(f"  Emergency:  {result.is_emergency_class}  (SJS)")
    print(f"  Latency:    {result.inference_ms} ms")
    if result.gradcam_path:
        print(f"  Grad-CAM:   {result.gradcam_path}")

    print("\n  Per-class probabilities:")
    for cls, score in sorted(result.confidence_all.items(),
                              key=lambda x: x[1], reverse=True):
        bar = '█' * int(score * 30)
        print(f"    {cls:<20} {score:5.1%}  {bar}")
