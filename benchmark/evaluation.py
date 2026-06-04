"""Evaluation harness: AUROC, AUPRC, ECE, adaptive ECE, Brier, SCE.

Implements all Phase A metrics with per-class binary decomposition
for multi-label tasks. ECE uses 15-bin equal-width by default and
adaptive equal-mass bins as a robustness check.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from config import ECE_N_BINS

logger = logging.getLogger(__name__)


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = ECE_N_BINS,
    strategy: str = "equal_width",
) -> float:
    """Expected Calibration Error for binary classification.

    Args:
        probs: predicted probabilities, shape (N,)
        labels: binary labels, shape (N,)
        n_bins: number of bins
        strategy: 'equal_width' or 'equal_mass'
    """
    if strategy == "equal_width":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "equal_mass":
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(probs, quantiles)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        bin_edges = np.unique(bin_edges)
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    ece = 0.0
    n = len(probs)

    for i in range(len(bin_edges) - 1):
        mask = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
        if i == 0:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_confidence = probs[mask].mean()
        avg_accuracy = labels[mask].mean()
        ece += (n_bin / n) * abs(avg_confidence - avg_accuracy)

    return float(ece)


def compute_signed_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = ECE_N_BINS,
) -> float:
    """Signed calibration error: negative = underconfident, positive = overconfident."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    sce = 0.0
    n = len(probs)

    for i in range(n_bins):
        mask = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
        if i == 0:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_confidence = probs[mask].mean()
        avg_accuracy = labels[mask].mean()
        sce += (n_bin / n) * (avg_confidence - avg_accuracy)

    return float(sce)


def compute_reliability_diagram_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = ECE_N_BINS,
) -> dict:
    """Compute bin-level data for reliability diagrams."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins_data = []

    for i in range(n_bins):
        mask = (probs > bin_edges[i]) & (probs <= bin_edges[i + 1])
        if i == 0:
            mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])
        n_bin = mask.sum()

        if n_bin == 0:
            bins_data.append({
                "bin_lower": float(bin_edges[i]),
                "bin_upper": float(bin_edges[i + 1]),
                "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                "n_samples": 0,
                "avg_confidence": None,
                "avg_accuracy": None,
            })
        else:
            bins_data.append({
                "bin_lower": float(bin_edges[i]),
                "bin_upper": float(bin_edges[i + 1]),
                "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                "n_samples": int(n_bin),
                "avg_confidence": float(probs[mask].mean()),
                "avg_accuracy": float(labels[mask].mean()),
            })

    return {"bins": bins_data, "n_bins": n_bins}


def evaluate_binary(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Full evaluation for a single binary task."""
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")

    try:
        auprc = average_precision_score(labels, probs)
    except ValueError:
        auprc = float("nan")

    brier = brier_score_loss(labels, probs)
    ece_ew = compute_ece(probs, labels, strategy="equal_width")
    ece_em = compute_ece(probs, labels, strategy="equal_mass")
    sce = compute_signed_calibration_error(probs, labels)
    rel_diag = compute_reliability_diagram_data(probs, labels)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "brier": float(brier),
        "ece_equal_width": float(ece_ew),
        "ece_equal_mass": float(ece_em),
        "sce": float(sce),
        "reliability_diagram": rel_diag,
    }


def evaluate_multilabel(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: list,
) -> dict:
    """Evaluate multi-label predictions with per-class binary decomposition."""
    n_classes = probs.shape[1]
    per_class = {}

    for c in range(n_classes):
        class_name = class_names[c] if c < len(class_names) else f"class_{c}"
        if labels[:, c].sum() < 2 or (1 - labels[:, c]).sum() < 2:
            logger.warning(f"Skipping {class_name}: insufficient positive/negative samples")
            continue
        per_class[class_name] = evaluate_binary(probs[:, c], labels[:, c])

    valid_metrics = [m for m in per_class.values() if not np.isnan(m["auroc"])]

    macro = {}
    if valid_metrics:
        for key in ["auroc", "auprc", "brier", "ece_equal_width", "ece_equal_mass", "sce"]:
            values = [m[key] for m in valid_metrics if not np.isnan(m[key])]
            macro[key] = float(np.mean(values)) if values else float("nan")
    else:
        for key in ["auroc", "auprc", "brier", "ece_equal_width", "ece_equal_mass", "sce"]:
            macro[key] = float("nan")

    return {"per_class": per_class, "macro": macro}


def evaluate_model_on_dataset(
    model_id: str,
    dataset_id: str,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    class_names: list,
) -> dict:
    """Top-level evaluation for one model-dataset cell."""
    result = evaluate_multilabel(test_probs, test_labels, class_names)
    result["model_id"] = model_id
    result["dataset_id"] = dataset_id
    logger.info(
        f"Eval {model_id}/{dataset_id}: "
        f"macro AUROC={result['macro']['auroc']:.4f}, "
        f"ECE={result['macro']['ece_equal_width']:.4f}, "
        f"Brier={result['macro']['brier']:.4f}, "
        f"SCE={result['macro']['sce']:.4f}"
    )
    return result
