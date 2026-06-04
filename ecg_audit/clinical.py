"""Clinical utility: decision-curve analysis (net benefit) and subgroup calibration."""
from __future__ import annotations
from typing import Sequence
import numpy as np

from .metrics import ece_equal_width, signed_calibration_error, auroc


def net_benefit(y_true, y_prob, threshold: float) -> float:
    """Net benefit at a decision threshold (Vickers & Elkin, 2006).

    NB = TP/n - FP/n * (pt / (1 - pt))
    """
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    n = len(y_true)
    pred = y_prob >= threshold
    tp = float(np.sum(pred & (y_true == 1)))
    fp = float(np.sum(pred & (y_true == 0)))
    if threshold >= 1.0:
        return 0.0
    return tp / n - fp / n * (threshold / (1.0 - threshold))


def decision_curve(y_true, y_prob, thresholds=None) -> dict:
    """Net-benefit curve plus treat-all / treat-none references."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.5, 50)
    y_true = np.asarray(y_true, float)
    prev = float(y_true.mean())
    model_nb, treat_all = [], []
    for t in thresholds:
        model_nb.append(net_benefit(y_true, y_prob, t))
        treat_all.append(prev - (1 - prev) * (t / (1 - t)) if t < 1 else 0.0)
    return {
        "thresholds": list(map(float, thresholds)),
        "model_net_benefit": model_nb,
        "treat_all": treat_all,
        "treat_none": [0.0] * len(thresholds),
        "prevalence": prev,
    }


def subgroup_calibration(probs, labels, groups, classes: Sequence[str],
                         n_bins: int = 15) -> dict:
    """Compute macro ECE / SCE / AUROC within each subgroup.

    Parameters
    ----------
    groups : array of group labels (one per sample).
    """
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, float)
    groups = np.asarray(groups)
    if probs.ndim == 1:
        probs, labels = probs[:, None], labels[:, None]
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        eces, sces, aurocs = [], [], []
        for c in range(probs.shape[1]):
            yt, yp = labels[m, c], probs[m, c]
            if len(np.unique(yt)) < 2:
                continue
            eces.append(ece_equal_width(yt, yp, n_bins))
            sces.append(signed_calibration_error(yt, yp))
            a = auroc(yt, yp)
            if not np.isnan(a):
                aurocs.append(a)
        out[str(g)] = {
            "n": int(m.sum()),
            "macro_ece": float(np.mean(eces)) if eces else float("nan"),
            "macro_sce": float(np.mean(sces)) if sces else float("nan"),
            "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        }
    return out
