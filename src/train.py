"""
OcuScan AI — src/train.py
Two-phase training loop with all anti-overfitting upgrades for 41-image dataset.

Phase 1 (--phase 1):
  - Frozen backbone, head-only training
  - AdamW, lr=1e-3, weight_decay=1e-3
  - CosineAnnealingLR T_max=20
  - Max 40 epochs, early stopping patience=7 on val_macro_f1
  - MixUp (α=0.2), Label smoothing (ε=0.1), WeightedRandomSampler, EMA

Phase 2 (--phase 2):
  - Top 3 EfficientNet blocks + head unfrozen
  - AdamW, lr=5e-6, weight_decay=1e-4  (TRD §3.1)
  - CosineAnnealingLR T_max=20
  - Max 25 epochs, early stopping patience=7
  - Same MixUp / EMA / label smoothing

Usage:
  python src/train.py --phase 1
  python src/train.py --phase 2 --resume models/phase1_best.pt
  python src/train.py --phase 1 --no-mixup          # ablation
  python src/train.py --phase 1 --image-size 160    # progressive resize
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from augmentation import get_train_transform, get_val_transform
from dataset import (CLASS_NAMES, LABELS_CSV, NUM_CLASSES, EyeDiseaseDataset,
                     apply_stratified_split)
from model import OcuScanModel, load_checkpoint, save_checkpoint
from utils import get_device, seed_everything

MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# MixUp
# ─────────────────────────────────────────────────────────────────────────────

def mixup_batch(x: torch.Tensor,
                y: torch.Tensor,
                alpha: float = 0.2):
    """
    MixUp augmentation for a single batch.
    Returns mixed_x, y_a, y_b, lambda — caller computes mixed loss.
    Applied only during training; val/test use clean images.
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mix, y_a, y_b, lam


def mixup_criterion(criterion: nn.Module,
                    pred: torch.Tensor,
                    y_a: torch.Tensor,
                    y_b: torch.Tensor,
                    lam: float) -> torch.Tensor:
    """Compute interpolated loss for MixUp batch."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────────────────────────────────────
# WeightedRandomSampler factory
# ─────────────────────────────────────────────────────────────────────────────

def make_weighted_sampler(dataset: EyeDiseaseDataset) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so every batch sees rare classes more
    frequently. Inverse-frequency weighting per sample.
    """
    class_counts = np.array([
        (dataset.df['class_idx'] == i).sum() for i in range(NUM_CLASSES)
    ], dtype=np.float32)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)  # avoid /0
    class_weight = 1.0 / class_counts

    sample_weights = np.array([
        class_weight[int(label)] for label in dataset.df['class_idx']
    ], dtype=np.float32)

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    print(f"  [Sampler] WeightedRandomSampler built — {len(sample_weights)} samples")
    print(f"            Class weights: { {CLASS_NAMES[i]: f'{class_weight[i]:.3f}' for i in range(NUM_CLASSES)} }")
    return sampler


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Monitor val_macro_f1 (maximise). Stop if no improvement after `patience`
    epochs. Saves best checkpoint automatically.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4,
                 checkpoint_path: Optional[Path] = None):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_score = -np.inf
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, score: float, model, optimizer, epoch: int,
                 ema_model=None) -> bool:
        """
        Returns True if training should stop.
        Saves checkpoint if score improved.
        """
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            if self.checkpoint_path is not None:
                # Save EMA weights if available (better generalisation)
                save_model = ema_model.module if ema_model is not None else model
                save_checkpoint(save_model, optimizer, epoch, score,
                                self.checkpoint_path)
        else:
            self.counter += 1
            print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs "
                  f"(best F1={self.best_score:.4f} @ epoch {self.best_epoch})")
            if self.counter >= self.patience:
                print(f"  [EarlyStopping] Stopping. Best epoch: {self.best_epoch}, "
                      f"Best val_macro_f1: {self.best_score:.4f}")
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> dict:
    """
    Run model on loader, return loss + per-class + macro metrics.
    Uses clean images (no MixUp, no augmentation).
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None,
                             labels=list(range(NUM_CLASSES)), zero_division=0)
    accuracy = float((all_preds == all_labels).mean())

    return {
        'loss': total_loss / max(n_batches, 1),
        'macro_f1': float(macro_f1),
        'accuracy': accuracy,
        'per_class_f1': {CLASS_NAMES[i]: float(per_class_f1[i])
                         for i in range(NUM_CLASSES)},
        'sjs_recall': float(per_class_f1[CLASS_NAMES.index('sjs')]),
        'symblepharon_f1': float(per_class_f1[CLASS_NAMES.index('symblepharon')]),
        'ocp_f1': float(per_class_f1[CLASS_NAMES.index('ocp')]),
        'ocp_chronic_f1': float(per_class_f1[CLASS_NAMES.index('ocp_chronic')]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(phase: int,
          resume: Optional[Path] = None,
          image_size: int = 224,
          use_mixup: bool = True,
          seed: int = 42) -> dict:
    """
    Run one training phase. Returns dict of best metrics.

    Args:
        phase:       1 (frozen backbone) or 2 (top-3 blocks unfrozen).
        resume:      Path to Phase 1 checkpoint to initialise Phase 2 from.
        image_size:  Input resolution (224 standard; 160 for progressive resize).
        use_mixup:   Enable MixUp augmentation.
        seed:        Random seed for reproducibility.
    """
    seed_everything(seed)
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  OcuScan AI — Phase {phase} Training")
    print(f"  Device: {device} | Image size: {image_size} | MixUp: {use_mixup}")
    print(f"{'='*60}\n")

    # ── Load labels.csv ───────────────────────────────────────────────────────
    if not LABELS_CSV.exists():
        raise FileNotFoundError(
            f"labels.csv not found at {LABELS_CSV}. "
            "Run src/dataset.py to build it first."
        )
    df = pd.read_csv(LABELS_CSV)
    if 'split' not in df.columns or df['split'].eq('').all():
        print("  [Data] Applying stratified split...")
        df = apply_stratified_split(df)
        df.to_csv(LABELS_CSV, index=False)

    # ── Transforms ───────────────────────────────────────────────────────────
    train_tfm = get_train_transform(image_size=image_size)
    val_tfm   = get_val_transform(image_size=image_size)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = EyeDiseaseDataset(df, 'train', transform=train_tfm)
    val_ds   = EyeDiseaseDataset(df, 'val',   transform=val_tfm)

    print(f"  [Data] Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"  [Data] Train class counts: {train_ds.class_counts()}")

    # ── Sampler + DataLoaders ─────────────────────────────────────────────────
    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        sampler=sampler,        # replaces shuffle=True
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = OcuScanModel(pretrained=True, freeze_backbone=(phase == 1)).to(device)

    if phase == 2:
        if resume is None:
            resume = MODELS_DIR / 'phase1_best.pt'
        if resume.exists():
            load_checkpoint(resume, model, device=str(device))
        else:
            print(f"  [WARN] Phase 2 resume checkpoint not found at {resume}. "
                  "Training from ImageNet weights.")
        model.unfreeze_top_blocks(n_blocks=3)
    else:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [Model] Trainable params (head only): {n_trainable:,}")

    # ── Loss (label smoothing ε=0.1) ──────────────────────────────────────────
    class_weights = train_ds.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.1,    # reduces overconfidence on 41-image dataset
    )
    # Val loss without label smoothing — for clean monitoring
    val_criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if phase == 1:
        # Head-only training: higher LR + strong weight decay
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3,
            weight_decay=1e-3,   # upgraded from 1e-4 for small dataset
        )
        max_epochs = 40
    else:
        # Fine-tuning: low LR to preserve ImageNet features in backbone
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=5e-6,
            weight_decay=1e-4,   # lighter regularisation for backbone
        )
        max_epochs = 25

    # ── Scheduler (CosineAnnealingLR, T_max=20) ───────────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-7
    )

    # ── Model EMA (decay=0.99 for small datasets) ─────────────────────────────
    ema_model = ModelEmaV2(model, decay=0.99, device=device)

    # ── Early stopping ────────────────────────────────────────────────────────
    ckpt_path = MODELS_DIR / f'phase{phase}_best.pt'
    early_stopper = EarlyStopping(patience=7, checkpoint_path=ckpt_path)

    # ── MLflow tracking ───────────────────────────────────────────────────────
    mlflow.set_experiment(f"ocuscan_phase{phase}")
    run_name = f"phase{phase}_img{image_size}_{'mixup' if use_mixup else 'nomixup'}"

    history = []

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            'phase': phase,
            'image_size': image_size,
            'max_epochs': max_epochs,
            'lr': optimizer.param_groups[0]['lr'],
            'weight_decay': optimizer.param_groups[0]['weight_decay'],
            'label_smoothing': 0.1,
            'mixup_alpha': 0.2 if use_mixup else 0.0,
            'ema_decay': 0.99,
            'early_stopping_patience': 7,
            'batch_size': 8,
            'n_train': len(train_ds),
            'n_val': len(val_ds),
        })

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()

            # ── Training epoch ────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels, _ in train_loader:
                images, labels = images.to(device), labels.to(device)

                if use_mixup:
                    images, y_a, y_b, lam = mixup_batch(images, labels, alpha=0.2)
                    logits = model(images)
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                    # For accuracy tracking, use un-mixed labels
                    preds = logits.argmax(dim=1)
                    train_correct += (lam * (preds == y_a).float() +
                                      (1 - lam) * (preds == y_b).float()).sum().item()
                else:
                    logits = model(images)
                    loss = criterion(logits, labels)
                    train_correct += (logits.argmax(1) == labels).sum().item()

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping (extra stability for tiny batches)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                ema_model.update(model)

                train_loss += loss.item()
                train_total += labels.size(0)

            scheduler.step()
            train_acc = train_correct / max(train_total, 1)
            train_loss_avg = train_loss / max(len(train_loader), 1)

            # ── Validation (use EMA model) ────────────────────────────────────
            val_metrics = evaluate(ema_model.module, val_loader, val_criterion, device)
            epoch_time = time.time() - epoch_start

            # ── Logging ───────────────────────────────────────────────────────
            print(
                f"  Epoch {epoch:3d}/{max_epochs} | "
                f"Train loss: {train_loss_avg:.4f} | acc: {train_acc:.3f} | "
                f"Val loss: {val_metrics['loss']:.4f} | "
                f"Val macro-F1: {val_metrics['macro_f1']:.4f} | "
                f"Val acc: {val_metrics['accuracy']:.3f} | "
                f"SJS recall: {val_metrics['sjs_recall']:.3f} | "
                f"OCP F1: {val_metrics['ocp_f1']:.3f} | "
                f"OCP-Chr F1: {val_metrics['ocp_chronic_f1']:.3f} | "
                f"{epoch_time:.1f}s"
            )

            row = {
                'epoch': epoch,
                'train_loss': train_loss_avg,
                'train_acc': train_acc,
                'lr': optimizer.param_groups[0]['lr'],
                **{f'val_{k}': v for k, v in val_metrics.items()
                   if not isinstance(v, dict)},
                **{f'val_f1_{k}': v
                   for k, v in val_metrics['per_class_f1'].items()},
            }
            history.append(row)

            mlflow.log_metrics({
                'train_loss': train_loss_avg,
                'train_acc': train_acc,
                'val_loss': val_metrics['loss'],
                'val_macro_f1': val_metrics['macro_f1'],
                'val_accuracy': val_metrics['accuracy'],
                'val_sjs_recall': val_metrics['sjs_recall'],
                'val_ocp_f1': val_metrics['ocp_f1'],
                'val_ocp_chronic_f1': val_metrics['ocp_chronic_f1'],
                'lr': optimizer.param_groups[0]['lr'],
            }, step=epoch)

            # ── Early stopping check ──────────────────────────────────────────
            stop = early_stopper(
                val_metrics['macro_f1'], model, optimizer, epoch,
                ema_model=ema_model
            )
            if stop:
                break

        # ── Save training history ──────────────────────────────────────────────
        history_path = RESULTS_DIR / f'phase{phase}_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\n  [History] Saved → {history_path}")

        # Log best metrics
        best = {
            'best_val_macro_f1': early_stopper.best_score,
            'best_epoch': early_stopper.best_epoch,
        }
        mlflow.log_metrics(best)
        mlflow.log_artifact(str(ckpt_path))

        print(f"\n  ✓ Phase {phase} complete.")
        print(f"    Best val_macro_f1 = {early_stopper.best_score:.4f} "
              f"(epoch {early_stopper.best_epoch})")
        print(f"    Checkpoint saved → {ckpt_path}")

    return {**best, 'history': history}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from typing import Optional   # needed for EarlyStopping type hint above

    parser = argparse.ArgumentParser(description='OcuScan AI Training')
    parser.add_argument('--phase', type=int, choices=[1, 2], default=1,
                        help='Training phase (1=frozen backbone, 2=fine-tune)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to Phase 1 checkpoint for Phase 2 init')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input resolution (default 224; use 160 for progressive)')
    parser.add_argument('--no-mixup', action='store_true',
                        help='Disable MixUp augmentation (ablation)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    resume_path = Path(args.resume) if args.resume else None

    results = train(
        phase=args.phase,
        resume=resume_path,
        image_size=args.image_size,
        use_mixup=not args.no_mixup,
        seed=args.seed,
    )
    print(f"\n  Final: best_val_macro_f1={results['best_val_macro_f1']:.4f}")
