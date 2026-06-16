"""
OcuScan AI — src/model.py
EfficientNetB0 classifier with custom 6-class head and Grad-CAM support.

Architecture (TRD §3.1):
  EfficientNetB0 backbone (ImageNet pretrained via timm)
  → GlobalAvgPool
  → Dropout(0.5)          ← upgraded from 0.4 for small-dataset regularisation
  → Linear(1280, 256)
  → ReLU
  → Dropout(0.3)          ← upgraded from 0.2
  → Linear(256, 6)
  → Softmax (inference) / raw logits (training)

Grad-CAM is attached to the last EfficientNet conv layer via backward hooks.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
from typing import Optional

from dataset import CLASS_NAMES, NUM_CLASSES

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class OcuScanModel(nn.Module):
    """
    EfficientNetB0 with a custom 6-class classifier head.

    Args:
        num_classes:  Number of output classes (default 6 — fixed per TRD).
        pretrained:   Load ImageNet weights via timm (True for training).
        freeze_backbone: If True, all EfficientNet parameters are frozen
                          (Phase 1 training mode).
    """

    def __init__(self,
                 num_classes: int = NUM_CLASSES,
                 pretrained: bool = True,
                 freeze_backbone: bool = True):
        super().__init__()

        # Load EfficientNetB0 without the default classification head
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,          # removes the default head
            global_pool='avg',      # GlobalAvgPool2d included
        )
        # backbone output feature dim = 1280 for EfficientNetB0
        in_features = self.backbone.num_features  # 1280

        # Custom classifier head (TRD §3.1, upgraded dropout per anti-overfitting plan)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),              # upgraded: 0.4 → 0.5
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),              # upgraded: 0.2 → 0.3
            nn.Linear(256, num_classes),
        )

        # Grad-CAM hooks (populated lazily on first call to get_gradcam)
        self._gradcam_gradients: Optional[torch.Tensor] = None
        self._gradcam_activations: Optional[torch.Tensor] = None
        self._hook_handles = []

        if freeze_backbone:
            self.freeze_backbone()

    # ── Backbone freeze / unfreeze ────────────────────────────────────────────

    def freeze_backbone(self):
        """Freeze all backbone parameters (Phase 1)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  [Model] Backbone frozen — training head only.")

    def unfreeze_top_blocks(self, n_blocks: int = 3):
        """
        Unfreeze the top N EfficientNet blocks for fine-tuning (Phase 2).
        EfficientNetB0 has 7 MBConv blocks (blocks[0]–blocks[6]).
        Top 3 = blocks[4], blocks[5], blocks[6] + conv_head + bn2.
        """
        # First re-freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze top N blocks
        blocks = list(self.backbone.blocks)
        for block in blocks[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # Always unfreeze the final conv + BN
        for layer in [self.backbone.conv_head, self.backbone.bn2]:
            for param in layer.parameters():
                param.requires_grad = True

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [Model] Unfrozen top {n_blocks} blocks + conv_head/bn2. "
              f"Trainable params: {n_trainable:,}")

    def unfreeze_all(self):
        """Unfreeze every parameter (used for full fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True
        n = sum(p.numel() for p in self.parameters())
        print(f"  [Model] All params unfrozen. Total: {n:,}")

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224] float32 tensor
        Returns:
            logits: [B, 6] — raw (unscaled) logits, NOT softmax
        """
        features = self.extract_features(x)     # [B, 1280]
        logits = self.classifier(features)  # [B, 6]
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent features from the EfficientNetB0 backbone before the custom classifier head.
        Args:
            x: [B, 3, 224, 224] float32 tensor
        Returns:
            features: [B, 1280] float32 tensor
        """
        return self.backbone(x)

    # ── Grad-CAM ──────────────────────────────────────────────────────────────

    def _register_gradcam_hooks(self):
        """Register forward/backward hooks on the last conv layer."""
        self._remove_gradcam_hooks()

        # Target layer: last block's last depthwise conv
        # In timm EfficientNetB0: backbone.blocks[-1][-1].conv_dw
        try:
            target_layer = self.backbone.blocks[-1][-1].conv_dw
        except (AttributeError, IndexError):
            # Fallback: conv_head
            target_layer = self.backbone.conv_head

        def save_activation(module, input, output):
            self._gradcam_activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self._gradcam_gradients = grad_output[0].detach()

        h1 = target_layer.register_forward_hook(save_activation)
        h2 = target_layer.register_full_backward_hook(save_gradient)
        self._hook_handles = [h1, h2]

    def _remove_gradcam_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def get_gradcam(self,
                    x: torch.Tensor,
                    class_idx: Optional[int] = None,
                    output_size: int = 224) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single input image.

        Args:
            x:           [1, 3, H, W] input tensor (NOT batched > 1).
            class_idx:   Class index to generate CAM for.
                         If None, uses predicted class (argmax of logits).
            output_size: Resize heatmap to this square size in pixels.

        Returns:
            heatmap: [H, W, 3] uint8 numpy array (BGR, jet colormap)
                     suitable for cv2 imwrite or overlay.
        """
        self._register_gradcam_hooks()
        self.eval()

        x = x.requires_grad_(True)
        logits = self.forward(x)           # [1, 6]

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward pass for the target class
        self.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        self._remove_gradcam_hooks()

        # Grad-CAM: global average pool of gradients → weight activations
        gradients = self._gradcam_gradients    # [1, C, H', W']
        activations = self._gradcam_activations  # [1, C, H', W']

        if gradients is None or activations is None:
            raise RuntimeError("Grad-CAM hooks did not fire. Check target layer.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * activations).sum(dim=1).squeeze()  # [H', W']
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        # Resize to output_size × output_size
        cam_resized = cv2.resize(cam, (output_size, output_size))

        # Apply jet colormap → BGR uint8
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        return heatmap   # [H, W, 3] BGR uint8

    def overlay_gradcam(self,
                        original_img: np.ndarray,
                        heatmap: np.ndarray,
                        alpha: float = 0.4) -> np.ndarray:
        """
        Blend Grad-CAM heatmap onto the original image.

        Args:
            original_img: [H, W, 3] uint8 (RGB or BGR, any size).
            heatmap:      [H, W, 3] uint8 BGR from get_gradcam().
            alpha:        Heatmap opacity (0 = original only, 1 = heatmap only).

        Returns:
            overlay: [H, W, 3] uint8 blended image.
        """
        h, w = original_img.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_resized, alpha, 0)
        return overlay


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model: OcuScanModel,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    val_macro_f1: float,
                    path: Path,
                    extra: Optional[dict] = None):
    """Save full training checkpoint (model + optimizer state)."""
    payload = {
        'epoch': epoch,
        'val_macro_f1': val_macro_f1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': CLASS_NAMES,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"  [Checkpoint] Saved → {path}  (epoch {epoch}, F1={val_macro_f1:.4f})")


def load_checkpoint(path: Path,
                    model: OcuScanModel,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    device: str = 'cpu') -> dict:
    """Load checkpoint into model (and optionally optimizer)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f"  [Checkpoint] Loaded ← {path}  "
          f"(epoch {ckpt.get('epoch','?')}, F1={ckpt.get('val_macro_f1','?')})")
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke-test  (python src/model.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    # Define get_device locally (utils module may not export it)
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("OcuScan AI — src/model.py smoke test")
    print("=" * 50)

    device = get_device()
    print(f"  Device: {device}")

    # ── Phase 1: frozen backbone ──────────────────────────────────────────────
    print("\n[1] Phase 1 — frozen backbone")
    model = OcuScanModel(pretrained=True, freeze_backbone=True).to(device)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {n_total:,}")
    print(f"  Trainable params: {n_trainable:,}  (head only)")

    # Forward pass
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, NUM_CLASSES), f"Expected (1,{NUM_CLASSES}), got {logits.shape}"
    print(f"  Forward pass OK → logits shape: {logits.shape}")

    # Batch forward
    x_batch = torch.randn(8, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(x_batch)
    assert out.shape == (8, NUM_CLASSES)
    print(f"  Batch forward OK → shape: {out.shape}")

    # ── Phase 2: unfreeze top 3 blocks ───────────────────────────────────────
    print("\n[2] Phase 2 — unfreeze top 3 blocks")
    model.unfreeze_top_blocks(n_blocks=3)
    n_trainable2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable2 > n_trainable, "Phase 2 should have more trainable params"

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    print("\n[3] Grad-CAM test")
    model.train(False)
    x_single = torch.randn(1, 3, 224, 224).to(device)
    heatmap = model.get_gradcam(x_single, class_idx=0)
    assert heatmap.shape == (224, 224, 3), f"Expected (224,224,3), got {heatmap.shape}"
    assert heatmap.dtype == np.uint8
    print(f"  Grad-CAM OK → heatmap shape: {heatmap.shape}, dtype: {heatmap.dtype}")

    # ── Classifier head inspection ────────────────────────────────────────────
    print("\n[4] Classifier head:")
    for name, module in model.classifier.named_modules():
        if name:
            print(f"  {name}: {module}")

    print("\n[OK] src/model.py passed all checks.")
