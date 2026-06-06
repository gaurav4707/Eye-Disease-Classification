"""
OcuScan AI — src/augmentation.py
Albumentations augmentation pipeline for anterior segment images.

All augmentations are medically realistic — no transform creates an appearance
impossible in a real anterior segment photograph.

Phase 1, Day 6 deliverable.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─────────────────────────────────────────────────────────────────────────────
# Training transform  (9 transforms per TRD §4.2 + resize + normalize)
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transform(image_size: int = 224) -> A.Compose:
    """
    Full augmentation pipeline applied only to the training set at load time.
    Never applied to val or test sets.

    Transform order follows Albumentations best-practice:
      1. Spatial transforms (flip, shift/scale/rotate)
      2. Colour/intensity transforms
      3. Noise & blur
      4. Dropout (last before normalise, to operate on spatial content)
      5. Resize → Normalize → ToTensorV2

    Returns:
        A.Compose pipeline. Call as: augmented = transform(image=np_hwc_uint8)
    """
    return A.Compose([
        # ── Spatial ──────────────────────────────────────────────────────────
        # Eyes are symmetric; flipping is medically valid
        A.HorizontalFlip(p=0.5),

        # Patient head position variation; slight rotation only
        A.Affine(
            translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            border_mode=0,   # constant (black) padding
            p=0.5,
        ),

        # ── Colour / intensity ────────────────────────────────────────────────
        # Slit-lamp illumination varies significantly between exams
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.4,
        ),

        # Colour temperature differences between clinical cameras
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.3,
        ),

        # Exposure variation between clinical settings
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.2,
        ),

        # CLAHE: enhances local contrast — critical for subtle forniceal changes
        # (key for OCP vs OCP Chronic discrimination)
        A.CLAHE(
            clip_limit=3.0,
            tile_grid_size=(8, 8),
            p=0.3,
        ),

        # ── Noise & blur ──────────────────────────────────────────────────────
        # Sensor noise in clinical cameras
        A.GaussNoise(
            std_range=(0.02, 0.11),   # approx var 5–30 in [0,255] space
            p=0.2,
        ),

        # Slight defocus in clinical photography
        A.GaussianBlur(
            blur_limit=3,
            p=0.15,
        ),

        # ── Dropout (robustness to occlusion) ─────────────────────────────────
        # Robustness to partial occlusion (lashes, specular reflections)
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(8, 24),
            hole_width_range=(8, 24),
            fill=0,
            p=0.2,
        ),

        # ── Resize + Normalise + Tensor ───────────────────────────────────────
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),   # ImageNet stats (matching timm/EfficientNet)
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int = 224) -> A.Compose:
    """
    Validation / test transform.  Resize + normalize only — NO augmentation.
    This is the transform used for inference in src/predict.py as well.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Named accessors (used by src/train.py and app/streamlit_app.py)
# ─────────────────────────────────────────────────────────────────────────────

train_transform = get_train_transform()
val_transform   = get_val_transform()


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke-test + visual grid  (python src/augmentation.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import random
    from pathlib import Path
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    PROJECT_ROOT = Path(__file__).parent.parent
    DATASET_DIR  = PROJECT_ROOT / 'dataset'
    OUTPUT_PATH  = PROJECT_ROOT / 'results' / 'augmentation_grid.png'

    from dataset import CLASS_NAMES

    print("OcuScan AI — src/augmentation.py visual grid test")
    print("=" * 55)

    # Collect up to 2 sample images per class
    samples = []
    for cls in CLASS_NAMES:
        cls_dir = DATASET_DIR / cls
        if not cls_dir.exists():
            continue
        images = [f for f in cls_dir.iterdir()
                  if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        if images:
            samples.append((cls, random.choice(images)))

    if not samples:
        print("  [INFO] No images found in dataset/ — skipping visual grid.")
        print("  Add images to dataset/<class_key>/ and re-run.")
        print("\n  Pipeline configured with 9 transforms:")
        transforms = [
            "HorizontalFlip (p=0.5)",
            "ShiftScaleRotate (p=0.5, shift=0.05, scale=0.1, rotate=15°)",
            "RandomBrightnessContrast (p=0.4, limit=0.25)",
            "HueSaturationValue (p=0.3, hue=10, sat=30, val=20)",
            "RandomGamma (p=0.2, gamma=80–120)",
            "CLAHE (p=0.3, clip_limit=3.0)",
            "GaussNoise (p=0.2, var=5–30)",
            "GaussianBlur (p=0.15, limit=3)",
            "CoarseDropout (p=0.2, max_holes=6, 8–24px)",
        ]
        for i, t in enumerate(transforms, 1):
            print(f"  {i:2}. {t}")
        sys.exit(0)

    # Build grid: each row = one class, 8 augmented versions
    n_aug = 8
    transform = get_train_transform()
    n_classes = len(samples)
    fig, axes = plt.subplots(n_classes, n_aug + 1, figsize=(2.5 * (n_aug + 1), 2.5 * n_classes))
    if n_classes == 1:
        axes = [axes]

    for row_idx, (cls_name, img_path) in enumerate(samples):
        img_np = np.array(Image.open(img_path).convert('RGB'))

        # Column 0: original
        axes[row_idx][0].imshow(img_np)
        axes[row_idx][0].set_title(f'{cls_name}\n(original)', fontsize=7)
        axes[row_idx][0].axis('off')

        # Columns 1–8: augmented
        for aug_idx in range(n_aug):
            aug = transform(image=img_np)['image']  # Tensor [3, H, W]
            # Denormalize for display
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            display = aug.numpy().transpose(1, 2, 0)
            display = (display * std + mean).clip(0, 1)
            axes[row_idx][aug_idx + 1].imshow(display)
            axes[row_idx][aug_idx + 1].set_title(f'aug {aug_idx + 1}', fontsize=7)
            axes[row_idx][aug_idx + 1].axis('off')

    plt.suptitle('OcuScan AI — Augmentation Grid (8 samples per class)', fontsize=10)
    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=100, bbox_inches='tight')
    print(f"  Visual grid saved → {OUTPUT_PATH}")
    print("  [OK] augmentation.py pipeline confirmed.")
