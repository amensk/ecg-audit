"""Calibration metrics: ECE (equal-width & adaptive), SCE, Brier, AUROC.

All metrics operate on per-class binary problems and are macro-averaged.
Definitions match the benchmark used in the accompanying paper.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np


def ece_equal_width(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins."""
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += m.sum() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece / n)


def adaptive_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Adaptive ECE: equal-count bins (quantile binning)."""
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    n = len(y_true)
    order = np.argsort(y_prob)
    yp, yt = y_prob[order], y_true[order]
    ece = 0.0
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        ece += (hi - lo) * abs(yt[lo:hi].mean() - yp[lo:hi].mean())
    return float(ece / n)


def signed_calibration_error(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """SCE = mean(prob) - mean(label). Positive = overconfident."""
    return float(np.asarray(y_prob, float).mean() - np.asarray(y_true, float).mean())


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_prob, float) - np.asarray(y_true, float)) ** 2))


def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


@dataclass
class AuditReport:
    """Result of a calibration audit."""
    classes: list
    per_class: dict = field(default_factory=dict)
    macro_ece: float = float("nan")
    macro_adaptive_ece: float = float("nan")
    macro_sce: float = float("nan")
    macro_brier: float = float("nan")
    macro_auroc: float = float("nan")
    n_classes_evaluated: int = 0

    def to_dict(self) -> dict:
        return {
            "classes": list(self.classes),
            "macro_ece": self.macro_ece,
            "macro_adaptive_ece": self.macro_adaptive_ece,
            "macro_sce": self.macro_sce,
            "macro_brier": self.macro_brier,
            "macro_auroc": self.macro_auroc,
            "n_classes_evaluated": self.n_classes_evaluated,
            "per_class": self.per_class,
        }

    def __repr__(self) -> str:
        return (f"AuditReport(macro_auroc={self.macro_auroc:.4f}, "
                f"macro_ece={self.macro_ece:.4f}, macro_sce={self.macro_sce:+.4f}, "
                f"k={self.n_classes_evaluated}/{len(self.classes)})")


class CalibrationAuditor:
    """Audit probability calibration of multi-label classifiers.

    Parameters
    ----------
    classes : sequence of str
        Class names (columns of the probability/label matrices).
    n_bins : int
        Number of bins for ECE.
    min_pos, min_neg : int
        Minimum positive / negative test support for a class to be included
        in macro metrics (guards against degenerate single-class AUROC/ECE).
    """

    def __init__(self, classes: Sequence[str], n_bins: int = 15,
                 min_pos: int = 10, min_neg: int = 10):
        self.classes = list(classes)
        self.n_bins = n_bins
        self.min_pos = min_pos
        self.min_neg = min_neg

    def audit(self, probs: np.ndarray, labels: np.ndarray) -> AuditReport:
        probs = np.asarray(probs, float)
        labels = np.asarray(labels, float)
        if probs.ndim == 1:
            probs = probs[:, None]; labels = labels[:, None]
        if probs.shape != labels.shape:
            raise ValueError(f"probs {probs.shape} and labels {labels.shape} must match")
        if probs.shape[1] != len(self.classes):
            raise ValueError(f"{probs.shape[1]} columns != {len(self.classes)} classes")

        per_class, eces, aeces, sces, briers, aurocs = {}, [], [], [], [], []
        for c, name in enumerate(self.classes):
            yt, yp = labels[:, c], probs[:, c]
            npos, nneg = int(yt.sum()), int(len(yt) - yt.sum())
            included = npos >= self.min_pos and nneg >= self.min_neg
            rec = {"n_pos": npos, "n_neg": nneg, "included": included}
            if included:
                e = ece_equal_width(yt, yp, self.n_bins)
                ae = adaptive_ece(yt, yp, self.n_bins)
                s = signed_calibration_error(yt, yp)
                b = brier(yt, yp)
                a = auroc(yt, yp)
                rec.update({"ece": e, "adaptive_ece": ae, "sce": s, "brier": b, "auroc": a})
                eces.append(e); aeces.append(ae); sces.append(s); briers.append(b)
                if not np.isnan(a):
                    aurocs.append(a)
            per_class[name] = rec

        rep = AuditReport(classes=self.classes, per_class=per_class)
        if eces:
            rep.macro_ece = float(np.mean(eces))
            rep.macro_adaptive_ece = float(np.mean(aeces))
            rep.macro_sce = float(np.mean(sces))
            rep.macro_brier = float(np.mean(briers))
            rep.macro_auroc = float(np.mean(aurocs)) if aurocs else float("nan")
            rep.n_classes_evaluated = len(eces)
        return rep
