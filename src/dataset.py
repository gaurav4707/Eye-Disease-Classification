"""
OcuScan AI — src/dataset.py
EyeDiseaseDataset class, DataLoader factory, and train/val/test split logic.
"""

import os
import csv
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

# ── Fixed class index mapping — NEVER change between train and inference ──────
CLASS_NAMES = [
    'normal',          # idx 0
    'ocp',             # idx 1
    'ocp_chronic',     # idx 2
    'post_viral_ded',  # idx 3
    'sjs',             # idx 4  (2 sub-photo types merged)
    'symblepharon',    # idx 5  (sign detection class)
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# ── Dataset root (resolved relative to this file) ─────────────────────────────
_SRC_DIR = Path(__file__).parent
PROJECT_ROOT = _SRC_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
LABELS_CSV = DATASET_DIR / "labels.csv"


# ─────────────────────────────────────────────────────────────────────────────
# labels.csv helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_labels_csv(dataset_dir: Path = DATASET_DIR,
                     output_path: Path = LABELS_CSV,
                     seed: int = 42) -> pd.DataFrame:
    """
    Scan dataset/<class_key>/ folders, create labels.csv with filepath,
    class_key, class_idx columns.  Split column is populated by
    apply_stratified_split().  Call this once after organising your dataset.
    """
    rows = []
    for class_key in CLASS_NAMES:
        class_dir = dataset_dir / class_key
        if not class_dir.exists():
            print(f"  [WARN] Folder missing: {class_dir}")
            continue
        image_files = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        ])
        for img_path in image_files:
            rows.append({
                'filepath': str(img_path.relative_to(PROJECT_ROOT)),
                'class_key': class_key,
                'class_idx': CLASS_TO_IDX[class_key],
                'split': '',           # populated by apply_stratified_split
                'ambiguous_flag': 0,
                'ambiguous_note': '',
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  labels.csv written → {output_path}  ({len(df)} images)")
    return df


def apply_stratified_split(df: pd.DataFrame,
                            val_frac: float = 0.15,
                            test_frac: float = 0.15,
                            seed: int = 42) -> pd.DataFrame:
    """
    StratifiedShuffleSplit 70/15/15 per class.
    OCP and OCP Chronic are split independently (per TRD §4.3).
    Augmented images (if any) must NOT be in val/test — handled by caller.
    """
    df = df.copy()
    df['split'] = ''

    # First split: train vs (val + test)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(val_frac + test_frac), random_state=seed
    )
    train_idx, temp_idx = next(sss1.split(df, df['class_idx']))

    temp_df = df.iloc[temp_idx].copy()
    relative_test_frac = test_frac / (val_frac + test_frac)

    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_test_frac, random_state=seed
    )
    val_local_idx, test_local_idx = next(
        sss2.split(temp_df, temp_df['class_idx'])
    )

    val_idx = temp_df.index[val_local_idx]
    test_idx = temp_df.index[test_local_idx]

    df.loc[df.index[train_idx], 'split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'
    df.loc[test_idx, 'split'] = 'test'

    return df


def verify_no_leakage(df: pd.DataFrame) -> bool:
    """
    Verify that no filepath appears in more than one split.
    Returns True if clean, raises AssertionError otherwise.
    """
    counts = df.groupby('filepath')['split'].nunique()
    leaky = counts[counts > 1]
    assert len(leaky) == 0, f"Data leakage detected for: {leaky.index.tolist()}"
    print("  [OK] No data leakage detected.")
    return True


def verify_split_counts(df: pd.DataFrame) -> dict:
    """Print per-class, per-split counts and return as dict."""
    summary = {}
    print("\n  Per-class split counts:")
    print(f"  {'class':<18} {'train':>7} {'val':>7} {'test':>7} {'total':>7}")
    print("  " + "-" * 46)
    for cls in CLASS_NAMES:
        cls_df = df[df['class_key'] == cls]
        t = (cls_df['split'] == 'train').sum()
        v = (cls_df['split'] == 'val').sum()
        te = (cls_df['split'] == 'test').sum()
        total = len(cls_df)
        print(f"  {cls:<18} {t:>7} {v:>7} {te:>7} {total:>7}")
        summary[cls] = {'train': int(t), 'val': int(v), 'test': int(te)}
    print()
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# EyeDiseaseDataset
# ─────────────────────────────────────────────────────────────────────────────

class EyeDiseaseDataset(Dataset):
    """
    PyTorch Dataset for OcuScan AI anterior segment images.

    Args:
        df:         DataFrame with columns [filepath, class_idx, split].
        split:      'train', 'val', or 'test'.
        transform:  Albumentations transform (applied to numpy HWC uint8 image).
        project_root: Path to resolve relative filepaths in df.

    Returns (per __getitem__):
        image_tensor:  torch.Tensor [3, H, W] float32 in [0, 1]
        label:         int class index
        filepath:      str (for debugging / error tracking)
    """

    def __init__(self,
                 df: pd.DataFrame,
                 split: str,
                 transform=None,
                 project_root: Path = PROJECT_ROOT):
        assert split in ('train', 'val', 'test'), \
            f"split must be 'train', 'val', or 'test'; got '{split}'"

        if split is not None and 'split' in df.columns:
            self.df = df[df['split'] == split].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)
        self.split = split
        self.transform = transform
        self.project_root = project_root

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split='{split}'. "
                             "Check labels.csv or run apply_stratified_split().")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.project_root / row['filepath']

        # Load image as RGB numpy array (H, W, 3) uint8
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {img_path}\n{e}")

        # Apply Albumentations transform
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']  # returns torch.Tensor if ToTensorV2 used

        # If transform did not include ToTensorV2, convert manually
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        label = int(row['class_idx'])
        filepath = str(row['filepath'])
        return image, label, filepath

    def class_counts(self) -> dict:
        """Return {class_key: count} for this split."""
        counts = self.df['class_key'].value_counts().to_dict()
        return {cls: counts.get(cls, 0) for cls in CLASS_NAMES}

    def class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency class weights for CrossEntropyLoss.
        Returns tensor of shape [NUM_CLASSES].
        """
        counts = [self.df[self.df['class_idx'] == i].shape[0]
                  for i in range(NUM_CLASSES)]
        counts = np.array(counts, dtype=np.float32)
        counts = np.where(counts == 0, 1.0, counts)  # avoid div-by-zero
        weights = 1.0 / counts
        weights = weights / weights.sum() * NUM_CLASSES   # normalise
        return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(df: pd.DataFrame,
                     train_transform=None,
                     val_transform=None,
                     batch_size: int = 8,
                     num_workers: int = 0,
                     pin_memory: bool = False,
                     seed: int = 42) -> dict:
    """
    Build train / val / test DataLoaders.

    Args:
        df:              Full labels DataFrame (all splits).
        train_transform: Albumentations transform for training set.
        val_transform:   Albumentations transform for val/test (no augmentation).
        batch_size:      Per TRD: 8 (constrained by dataset size).
        num_workers:     0 for Windows/debug; 2–4 for Linux production.
        pin_memory:      True when training on GPU.

    Returns:
        dict with keys 'train', 'val', 'test' → DataLoader instances.
    """

    def _worker_init(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    train_ds = EyeDiseaseDataset(df, 'train', transform=train_transform)
    val_ds   = EyeDiseaseDataset(df, 'val',   transform=val_transform)
    test_ds  = EyeDiseaseDataset(df, 'test',  transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke-test  (python src/dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    print("OcuScan AI — src/dataset.py smoke test")
    print("=" * 50)

    # Check dataset folder structure
    print("\n[1] Dataset folder check:")
    all_present = True
    total_images = 0
    for cls in CLASS_NAMES:
        cls_dir = DATASET_DIR / cls
        if cls_dir.exists():
            n = len([f for f in cls_dir.iterdir()
                     if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}])
            print(f"  {cls:<22} → {n:>4} images")
            total_images += n
        else:
            print(f"  {cls:<22} → [MISSING]")
            all_present = False
    print(f"\n  Total images found: {total_images}")

    if total_images == 0:
        print("\n  [INFO] Dataset is empty — this is expected before images are added.")
        print("  Run build_labels_csv() + apply_stratified_split() after adding images.")
        sys.exit(0)

    # Build or load labels.csv
    print("\n[2] labels.csv:")
    if LABELS_CSV.exists():
        df = pd.read_csv(LABELS_CSV)
        print(f"  Loaded existing labels.csv ({len(df)} rows)")
    else:
        print("  Building labels.csv from folder scan...")
        df = build_labels_csv()

    # Apply split if not already done
    if 'split' not in df.columns or df['split'].eq('').all():
        print("\n[3] Applying stratified 70/15/15 split...")
        df = apply_stratified_split(df)
        df.to_csv(LABELS_CSV, index=False)
        print("  Split applied and saved.")
    else:
        print("\n[3] Split already present in labels.csv")

    # Verify
    print("\n[4] Verifying no data leakage...")
    verify_no_leakage(df)

    # Count summary
    print("\n[5] Per-class split counts:")
    verify_split_counts(df)

    # Class index verification
    print("[6] CLASS_NAMES verification:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  [{idx}] {name}")

    print("\n[OK] src/dataset.py passed all checks.")
    print("     Add images to dataset/<class_key>/ and re-run to build labels.csv.")
