"""
OcuScan AI — src/evaluate.py
Phase 3 | Day 15-17
Full evaluation: accuracy, per-class P/R/F1, confusion matrix, AUC-ROC, calibration.
Saves results/metrics.json.
"""

import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd
from model import load_checkpoint
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibrationDisplay
from sklearn.preprocessing import label_binarize

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Local imports ──────────────────────────────────────────────────────────────
# These are available after Phase 1 & 2 are complete.
# Guarded imports allow evaluate.py to be imported without crashing in isolation.
try:
    from dataset import EyeDiseaseDataset, CLASS_NAMES, make_dataloaders
    from model import OcuScanModel
    from augmentation import val_transform
except ImportError:
    CLASS_NAMES = [
        "normal",
        "ocp",
        "ocp_chronic",
        "post_viral_ded",
        "sjs",
        "symblepharon",
    ]

DISPLAY_NAMES = {
    "normal": "Normal",
    "ocp": "OCP",
    "ocp_chronic": "OCP Chronic",
    "post_viral_ded": "Post-Viral DED",
    "sjs": "SJS",
    "symblepharon": "Symblepharon",
}

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Temperature scaling ────────────────────────────────────────────────────────

class TemperatureScaler:
    """
    Simple post-hoc temperature scaling for calibration.
    Fit on validation set; apply before final test evaluation.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Find optimal temperature via grid search minimising NLL.
        logits: (N, C) raw model logits (before softmax).
        labels: (N,) integer class indices.
        Returns the fitted temperature value.
        """
        from scipy.optimize import minimize_scalar

        def nll(T):
            scaled = logits / T
            # Log-softmax then NLL
            log_probs = scaled - np.log(np.sum(np.exp(scaled), axis=1, keepdims=True))
            return -np.mean(log_probs[np.arange(len(labels)), labels])

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(getattr(result, "x"))
        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated softmax probabilities."""
        scaled = logits / self.temperature
        exp_s = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        return exp_s / exp_s.sum(axis=1, keepdims=True)


# ── Core metric computation ────────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
) -> dict:
    """
    Compute the full evaluation suite and return a metrics dict.

    Parameters
    ----------
    y_true  : (N,) integer ground-truth labels
    y_pred  : (N,) integer predicted labels
    probs   : (N, C) softmax probabilities
    class_names : list of class key strings

    Returns
    -------
    metrics : dict ready for JSON serialisation
    """
    n_classes = len(class_names)

    # ── Basic ──────────────────────────────────────────────────────────────────
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )

    # precision_recall_fscore_support may return None for some outputs in edge
    # cases; ensure we have arrays to index into to avoid 'None is not
    # subscriptable' errors.
    if precision is None:
        precision = np.zeros(n_classes, dtype=float)
    if recall is None:
        recall = np.zeros(n_classes, dtype=float)
    if f1 is None:
        f1 = np.zeros(n_classes, dtype=float)
    if support is None:
        support = np.zeros(n_classes, dtype=int)

    per_class = {}
    for i, key in enumerate(class_names):
        per_class[key] = {
            "display": DISPLAY_NAMES.get(key, key),
            "precision": round(float(np.asarray(precision)[i]), 4),
            "recall": round(float(np.asarray(recall)[i]), 4),
            "f1": round(float(np.asarray(f1)[i]), 4),
            "support": int(support[i]),
        }

    # ── OCP / OCP Chronic explicit confusion rate ──────────────────────────────
    ocp_idx = class_names.index("ocp")
    ocp_chr_idx = class_names.index("ocp_chronic")

    ocp_mask = y_true == ocp_idx
    ocp_chr_mask = y_true == ocp_chr_idx

    ocp_confused_as_chronic = (
        int(np.sum((y_true == ocp_idx) & (y_pred == ocp_chr_idx)))
    )
    ocp_chr_confused_as_ocp = (
        int(np.sum((y_true == ocp_chr_idx) & (y_pred == ocp_idx)))
    )
    ocp_total = int(np.sum(ocp_mask))
    ocp_chr_total = int(np.sum(ocp_chr_mask))

    ocp_confusion_rate = (
        (ocp_confused_as_chronic + ocp_chr_confused_as_ocp)
        / max(ocp_total + ocp_chr_total, 1)
    )

    # ── SJS Recall & Symblepharon Precision (clinical stakes) ─────────────────
    sjs_idx = class_names.index("sjs")
    symb_idx = class_names.index("symblepharon")
    sjs_recall = float(np.asarray(recall)[sjs_idx])
    symb_precision = float(np.asarray(precision)[symb_idx])

    # ── AUC-ROC ───────────────────────────────────────────────────────────────
    y_bin = np.asarray(label_binarize(y_true, classes=list(range(n_classes))))
    # Ensure probs is a dense numpy array. Some callers may pass a
    # scipy.sparse spmatrix which does not support numpy-style
    # __getitem__ indexing; convert to a dense array first.
    # If probs is a sparse matrix or implements toarray(), convert it
    # to a dense numpy array. Otherwise, ensure it's a numpy array.
    toarray = getattr(probs, "toarray", None)
    if callable(toarray):
        probs = np.asarray(toarray())
    else:
        probs = np.asarray(probs)
    auc_per_class = {}
    macro_auc = None
    try:
        for i, key in enumerate(class_names):
            auc_per_class[key] = round(
                float(roc_auc_score(y_bin[:, i], probs[:, i])), 4
            )
        macro_auc = round(
            float(roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")), 4
        )
    except ValueError as e:
        auc_per_class = {k: None for k in class_names}
        macro_auc = None

    metrics = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_auc_roc": macro_auc,
        "auc_roc_per_class": auc_per_class,
        "per_class": per_class,
        "ocp_vs_ocp_chronic": {
            "ocp_confused_as_chronic": ocp_confused_as_chronic,
            "ocp_chronic_confused_as_ocp": ocp_chr_confused_as_ocp,
            "ocp_confusion_rate": round(ocp_confusion_rate, 4),
        },
        "clinical_stakes": {
            "sjs_recall": round(sjs_recall, 4),
            "symblepharon_precision": round(symb_precision, 4),
        },
    }
    return metrics


# ── Confusion matrix plot ──────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    save_path: Path = RESULTS_DIR / "confusion_matrix.png",
) -> None:
    """
    Plot a 6×6 normalised confusion matrix.
    OCP vs OCP Chronic off-diagonal cells are highlighted with a bold border.
    """
    display_labels = [DISPLAY_NAMES.get(k, k) for k in class_names]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        linewidths=0.5,
        linecolor="lightgrey",
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Normalised proportion"},
    )

    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title("OcuScan AI — Confusion Matrix (Normalised)", fontsize=14, pad=14)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    # ── Bold red border on OCP ↔ OCP Chronic confusion cells ──────────────────
    ocp_idx = class_names.index("ocp")
    ocp_chr_idx = class_names.index("ocp_chronic")
    for (row, col) in [(ocp_idx, ocp_chr_idx), (ocp_chr_idx, ocp_idx)]:
        ax.add_patch(
            Rectangle(
                (col, row), 1, 1,
                fill=False, edgecolor="#e63946", linewidth=3, zorder=5,
            )
        )

    # Annotation for highlighted cells
    fig.text(
        0.13, 0.01,
        "■ Red border = OCP ↔ OCP Chronic confusion cells (key discriminator)",
        fontsize=8, color="#e63946",
    )

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved → {save_path}")


# ── AUC-ROC curves ────────────────────────────────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    save_path: Path = RESULTS_DIR / "roc_curves.png",
) -> None:
    """Plot per-class ROC curves (one-vs-rest) plus macro average."""
    n_classes = len(class_names)
    # label_binarize may return a sparse matrix (spmatrix) which doesn't
    # implement __getitem__ in some scipy versions; convert to dense array
    # to ensure indexing like y_bin[:, i] works.
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    # Ensure it's a dense numpy array
    try:
        y_bin = np.asarray(y_bin.toarray())  # type: ignore[attr-defined]
    except AttributeError:
        y_bin = np.asarray(y_bin)

    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(9, 7))

    mean_fpr = np.linspace(0, 1, 200)
    tprs = []

    for i, (key, col) in enumerate(zip(class_names, colors)):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            auc_val = roc_auc_score(y_bin[:, i], probs[:, i])
            ax.plot(
                fpr, tpr,
                label=f"{DISPLAY_NAMES.get(key, key)} (AUC={auc_val:.3f})",
                color=col, lw=1.8,
            )
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        except ValueError:
            pass

    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        macro_auc = np.mean(np.asarray([roc_auc_score(y_bin[:, i], probs[:, i]) for i in range(n_classes)]))
        ax.plot(
            mean_fpr, mean_tpr,
            color="black", lw=2.5, linestyle="--",
            label=f"Macro avg (AUC≈{macro_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Random classifier")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("OcuScan AI — AUC-ROC Curves (One-vs-Rest)", fontsize=14)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] ROC curves saved → {save_path}")


# ── Calibration curve ─────────────────────────────────────────────────────────

def plot_calibration_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    save_path: Path = RESULTS_DIR / "calibration_curve.png",
    n_bins: int = 10,
) -> dict:
    """
    Reliability diagram for each class (one-vs-rest) plus mean calibration error.
    Returns dict with ECE per class.
    """
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    # Ensure a dense ndarray for numpy-style indexing.
    y_bin = np.asarray(y_bin)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    ece_scores = {}

    for i, key in enumerate(class_names):
        ax = axes[i]
        try:
            prob_true, prob_pred = calibration_curve(
                y_bin[:, i], probs[:, i], n_bins=n_bins, strategy="uniform"
            )
            # ECE (expected calibration error)
            bin_sizes = np.histogram(probs[:, i], bins=n_bins, range=(0, 1))[0]
            ece = float(
                np.sum(
                    np.abs(prob_true - prob_pred)
                    * bin_sizes[: len(prob_true)]
                    / len(y_true)
                )
            )
            ece_scores[key] = round(ece, 4)

            ax.plot(prob_pred, prob_true, "s-", color="#2563eb", markersize=5, label="Model")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
            ax.fill_between(prob_pred, prob_pred, prob_true, alpha=0.15, color="#2563eb")
            ax.set_title(f"{DISPLAY_NAMES.get(key, key)}\nECE={ece:.3f}", fontsize=10)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel("Mean predicted probability", fontsize=8)
            ax.set_ylabel("Fraction of positives", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        except Exception as exc:
            ax.text(0.5, 0.5, f"No data\n{exc}", ha="center", va="center", fontsize=8)
            ece_scores[key] = None

    fig.suptitle("OcuScan AI — Calibration Curves (Reliability Diagrams)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Calibration curves saved → {save_path}")
    return ece_scores


# ── Formatted metrics table ───────────────────────────────────────────────────

def print_metrics_table(metrics: dict) -> None:
    """Pretty-print the metrics dict to stdout."""
    sep = "─" * 72
    print(f"\n{'OcuScan AI — Evaluation Report':^72}")
    print(sep)
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Macro F1   : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  Macro AUC  : {metrics['macro_auc_roc']}")
    print(sep)
    print(f"  {'Class':<22} {'Precision':>9} {'Recall':>9} {'F1':>9} {'Support':>9}")
    print(f"  {'─'*22} {'─'*9} {'─'*9} {'─'*9} {'─'*9}")
    for key, v in metrics["per_class"].items():
        print(
            f"  {v['display']:<22} {v['precision']:>9.4f} {v['recall']:>9.4f}"
            f" {v['f1']:>9.4f} {v['support']:>9}"
        )
    print(sep)
    ocp = metrics["ocp_vs_ocp_chronic"]
    print(f"  OCP ↔ OCP Chronic confusion rate : {ocp['ocp_confusion_rate']:.4f}")
    print(f"    OCP predicted as OCP Chronic   : {ocp['ocp_confused_as_chronic']}")
    print(f"    OCP Chronic predicted as OCP   : {ocp['ocp_chronic_confused_as_ocp']}")
    print(sep)
    cs = metrics["clinical_stakes"]
    print(f"  SJS Recall            : {cs['sjs_recall']:.4f}  (high clinical stakes)")
    print(f"  Symblepharon Precision: {cs['symblepharon_precision']:.4f}")
    print(sep)


# ── 5-Fold cross-validation ───────────────────────────────────────────────────

def run_cross_validation(
    labels_csv: str,
    checkpoint_path: str,
    device: str = "cpu",
    n_splits: int = 5,
    batch_size: int = 8,
) -> dict:
    """
    StratifiedKFold cross-validation on the training set.
    Returns mean ± std for macro F1 and per-class F1.
    Saves results/cv_results.json.
    """
    from sklearn.model_selection import StratifiedKFold
    import torch

    try:
        from dataset import EyeDiseaseDataset, CLASS_NAMES
        from model import OcuScanModel
        from augmentation import val_transform
    except ImportError:
        raise RuntimeError("Phase 1 & 2 src modules not available.")

    df = pd.read_csv(labels_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    X = train_df["filepath"].values
    y = train_df["class_idx"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_macro_f1 = []
    fold_per_class_f1: dict[str, list] = {k: [] for k in CLASS_NAMES}

    model = OcuScanModel(num_classes=len(CLASS_NAMES))
    load_checkpoint(
    Path(checkpoint_path),
    model,
    device=device
)
    model.to(device)
    model.eval()

    for fold_idx, (_, val_idx) in enumerate(skf.split(X, y)):
        print(f"[cv] Fold {fold_idx + 1}/{n_splits} …")
        fold_paths = X[val_idx].tolist()
        fold_labels = y[val_idx].tolist()

        fold_df = pd.DataFrame({"filepath": fold_paths, "class_idx": fold_labels})
        dataset = EyeDiseaseDataset(
            df=fold_df,
            split="val",
            transform=val_transform,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_preds, all_true = [], []
        with torch.no_grad():
            for imgs, lbls, _ in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(lbls.numpy())

        all_preds = np.array(all_preds)
        all_true = np.array(all_true)

        mf1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
        fold_macro_f1.append(mf1)

        _, _, per_f1, _ = precision_recall_fscore_support(
            all_true, all_preds, labels=list(range(len(CLASS_NAMES))), zero_division=0
        )
        per_f1 = np.asarray(per_f1)
        for i, k in enumerate(CLASS_NAMES):
            # per_f1 is an array of floats; index directly and cast to float
            fold_per_class_f1[k].append(float(per_f1[i]))

    cv_results = {
        "n_splits": n_splits,
        "macro_f1_mean": round(float(np.mean(fold_macro_f1)), 4),
        "macro_f1_std": round(float(np.std(fold_macro_f1)), 4),
        "macro_f1_folds": [round(v, 4) for v in fold_macro_f1],
        "per_class_f1": {
            k: {
                "mean": round(float(np.mean(vs)), 4),
                "std": round(float(np.std(vs)), 4),
                "folds": [round(v, 4) for v in vs],
            }
            for k, vs in fold_per_class_f1.items()
        },
    }

    out_path = RESULTS_DIR / "cv_results.json"
    with open(out_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"[cv] Cross-validation results saved → {out_path}")
    print(
        f"[cv] Macro F1: {cv_results['macro_f1_mean']:.4f}"
        f" ± {cv_results['macro_f1_std']:.4f}"
    )
    return cv_results


# ── SVM baseline ──────────────────────────────────────────────────────────────

def run_svm_baseline(
    labels_csv: str,
    checkpoint_path: str,
    device: str = "cpu",
    batch_size: int = 8,
) -> dict:
    """
    Day 19: Freeze EfficientNetB0, extract 1280-dim features, train SVM(RBF),
    evaluate on test set.  Saves results/svm_baseline.json.
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    import torch

    try:
        from dataset import EyeDiseaseDataset, CLASS_NAMES
        from model import OcuScanModel
        from augmentation import val_transform
    except ImportError:
        raise RuntimeError("Phase 1 & 2 src modules not available.")

    df = pd.read_csv(labels_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    model = OcuScanModel(num_classes=len(CLASS_NAMES))
    load_checkpoint(
    Path(checkpoint_path),
    model,
    device=device
)
    model.to(device)
    model.eval()

    def extract_features(split_df):
    # split_df is already filtered to the desired rows.
    # Temporarily tag them with a sentinel so EyeDiseaseDataset
    # can match without conflicting with real split names.
        _df = split_df.copy()
        _df["split"] = "_extract"
        dataset = EyeDiseaseDataset(df=_df, split="_extract", transform=val_transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        feats, ys = [], []
        with torch.no_grad():
            for imgs, lbls, _ in loader:
                imgs = imgs.to(device)
                f = model.extract_features(imgs)  # 1280-dim GlobalAvgPool output
                feats.append(f.cpu().numpy())
                ys.extend(lbls.numpy())
        return np.vstack(feats), np.array(ys)

    print("[svm] Extracting training features …")
    X_train, y_train = extract_features(train_df)
    print("[svm] Extracting test features …")
    X_test, y_test = extract_features(test_df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("[svm] GridSearchCV for SVM(RBF) …")
    param_grid = {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]}
    svm = GridSearchCV(
        SVC(kernel="rbf", probability=True, random_state=42),
        param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    svm.fit(X_train, y_train)
    print(f"[svm] Best params: {svm.best_params_}")

    y_pred_svm = svm.predict(X_test)
    svm_probs = svm.predict_proba(X_test)

    svm_acc = float(accuracy_score(y_test, y_pred_svm))
    svm_macro_f1 = float(f1_score(y_test, y_pred_svm, average="macro", zero_division=0))
    prec, rec, f1_, supp = precision_recall_fscore_support(
        y_test, y_pred_svm, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    prec = np.atleast_1d(prec)
    rec = np.atleast_1d(rec)
    f1_ = np.atleast_1d(f1_)

    svm_per_class = {
        k: {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1_[i]), 4),
        }
        for i, k in enumerate(CLASS_NAMES)
    }

    try:
        y_bin = label_binarize(y_test, classes=list(range(len(CLASS_NAMES))))
        svm_macro_auc = round(
            float(roc_auc_score(y_bin, svm_probs, average="macro", multi_class="ovr")), 4
        )
    except Exception:
        svm_macro_auc = None

    baseline = {
        "model": "SVM(RBF) on frozen EfficientNetB0 features",
        "best_params": svm.best_params_,
        "accuracy": round(svm_acc, 4),
        "macro_f1": round(svm_macro_f1, 4),
        "macro_auc_roc": svm_macro_auc,
        "per_class": svm_per_class,
    }

    out_path = RESULTS_DIR / "svm_baseline.json"
    with open(out_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"[svm] SVM baseline saved → {out_path}")
    print(f"[svm] Macro F1: {svm_macro_f1:.4f} | Accuracy: {svm_acc:.4f}")
    return baseline


# ── Main evaluation entry point ────────────────────────────────────────────────

def evaluate(
    labels_csv: str,
    checkpoint_path: str,
    device: str = "cpu",
    batch_size: int = 8,
    apply_temperature: bool = False,
) -> dict:
    """
    Full evaluation run on the held-out test set.
    Saves metrics.json, confusion_matrix.png, roc_curves.png, calibration_curve.png.
    """
    import torch

    try:
        from dataset import EyeDiseaseDataset, CLASS_NAMES
        from model import OcuScanModel
        from augmentation import val_transform
    except ImportError:
        raise RuntimeError(
            "Phase 1 & 2 source modules (dataset.py, model.py, augmentation.py) not found. "
            "Ensure you are running from the project root with src/ on the path."
        )

    print(f"[evaluate] Loading model from {checkpoint_path} …")
    model = OcuScanModel(num_classes=len(CLASS_NAMES))
    from pathlib import Path

    load_checkpoint(Path(checkpoint_path), model, device=device)
    model.to(device)
    model.eval()

    df = pd.read_csv(labels_csv)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    def run_inference(subset_df):
        # EyeDiseaseDataset expects a dataframe and a split name (see other usages)
        split_name = subset_df["split"].iloc[0] if "split" in subset_df.columns else "test"
        dataset = EyeDiseaseDataset(df=subset_df, split=split_name, transform=val_transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_logits, all_true = [], []
        with torch.no_grad():
            for imgs, lbls, _ in loader:
                imgs = imgs.to(device)
                logits = model(imgs)
                all_logits.append(logits.cpu().numpy())
                all_true.extend(lbls.numpy())
        return np.vstack(all_logits), np.array(all_true)

    # ── Optionally fit temperature scaling on val set ──────────────────────────
    if apply_temperature:
        print("[evaluate] Fitting temperature scaler on validation set …")
        val_logits, val_true = run_inference(val_df)
        scaler = TemperatureScaler()
        T = scaler.fit(val_logits, val_true)
        print(f"[evaluate] Temperature = {T:.4f}")
    else:
        scaler = None

    # ── Test set inference ─────────────────────────────────────────────────────
    print("[evaluate] Running inference on test set …")
    t0 = time.perf_counter()
    test_logits, y_true = run_inference(test_df)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    if scaler is not None:
        probs = scaler.transform(test_logits)
    else:
        probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()

    y_pred = probs.argmax(axis=1)

    # ── Compute metrics ────────────────────────────────────────────────────────
    metrics = compute_all_metrics(y_true, y_pred, probs, CLASS_NAMES)
    metrics["inference_ms_total"] = elapsed_ms
    metrics["n_test_samples"] = len(y_true)

    if apply_temperature and scaler:
        metrics["temperature_scaling"] = {"applied": True, "temperature": round(scaler.temperature, 4)}
    else:
        metrics["temperature_scaling"] = {"applied": False}

    # ── Calibration ───────────────────────────────────────────────────────────
    ece_scores = plot_calibration_curve(y_true, probs)
    metrics["ece_per_class"] = ece_scores

    # ── Save metrics.json ──────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] metrics.json saved → {out_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print_metrics_table(metrics)
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curves(y_true, probs)

    return metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="OcuScan AI — Phase 3 Evaluation")
    parser.add_argument(
        "--labels-csv",
        default="dataset/labels.csv",
        help="Path to labels.csv (must contain split column)",
    )
    parser.add_argument(
        "--checkpoint",
        default="models/phase2_best.pt",
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device: cuda or cpu",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--temperature-scaling",
        action="store_true",
        help="Apply temperature scaling (fit on val set before test evaluation)",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run 5-fold cross-validation after main evaluation",
    )
    parser.add_argument(
        "--svm",
        action="store_true",
        help="Run SVM baseline after main evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    args = _parse_args()

    metrics = evaluate(
        labels_csv=args.labels_csv,
        checkpoint_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        apply_temperature=args.temperature_scaling,
    )

    if args.cv:
        run_cross_validation(
            labels_csv=args.labels_csv,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )

    if args.svm:
        run_svm_baseline(
            labels_csv=args.labels_csv,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
