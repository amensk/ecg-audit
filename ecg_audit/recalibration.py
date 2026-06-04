"""Post-hoc recalibration methods for multi-label probabilities.

All methods are rank-preserving per class (monotone), so AUROC is preserved
exactly (Platt, temperature) or near-exactly (isotonic, up to ties).
"""
from __future__ import annotations
from typing import Sequence
import numpy as np


def platt_scale(p_cal, y_cal, p_test):
    """Platt scaling (1-D logistic regression on probabilities)."""
    from sklearn.linear_model import LogisticRegression
    p_cal, y_cal, p_test = map(lambda a: np.asarray(a, float), (p_cal, y_cal, p_test))
    if len(np.unique(y_cal)) < 2:
        return p_test.copy()
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    lr.fit(p_cal.reshape(-1, 1), y_cal.astype(int))
    return lr.predict_proba(p_test.reshape(-1, 1))[:, 1]


def isotonic_calibrate(p_cal, y_cal, p_test):
    """Isotonic regression calibration."""
    from sklearn.isotonic import IsotonicRegression
    p_cal, y_cal, p_test = map(lambda a: np.asarray(a, float), (p_cal, y_cal, p_test))
    if len(np.unique(y_cal)) < 2:
        return p_test.copy()
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    return iso.predict(p_test)


def temperature_scale(p_cal, y_cal, p_test, lr: float = 0.01, max_iter: int = 50):
    """Temperature scaling via LBFGS on calibration NLL (no torch dependency)."""
    p_cal, y_cal, p_test = map(lambda a: np.asarray(a, float), (p_cal, y_cal, p_test))
    if len(np.unique(y_cal)) < 2:
        return p_test.copy()
    eps = 1e-7
    logit_cal = np.log(np.clip(p_cal, eps, 1 - eps) / np.clip(1 - p_cal, eps, 1 - eps))

    def nll(T):
        ps = 1.0 / (1.0 + np.exp(-logit_cal / T))
        ps = np.clip(ps, eps, 1 - eps)
        return -np.mean(y_cal * np.log(ps) + (1 - y_cal) * np.log(1 - ps))

    # simple 1-D golden-section search over T in [0.2, 5]
    a, b = 0.2, 5.0
    gr = (np.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(60):
        if nll(c) < nll(d):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    T = (a + b) / 2
    logit_te = np.log(np.clip(p_test, eps, 1 - eps) / np.clip(1 - p_test, eps, 1 - eps))
    return 1.0 / (1.0 + np.exp(-logit_te / T))


_METHODS = {"platt": platt_scale, "isotonic": isotonic_calibrate,
            "temperature": temperature_scale}


def recalibrate(method: str, p_cal, y_cal, p_test):
    """Recalibrate a single class with the named method."""
    if method not in _METHODS:
        raise ValueError(f"unknown method '{method}'; choose from {list(_METHODS)}")
    return _METHODS[method](p_cal, y_cal, p_test)


def recalibrate_multilabel(method: str, p_cal, y_cal, p_test):
    """Apply a recalibration method independently to each class column."""
    p_cal, y_cal, p_test = map(lambda a: np.asarray(a, float), (p_cal, y_cal, p_test))
    out = np.array(p_test, dtype=float)
    for c in range(p_test.shape[1]):
        out[:, c] = recalibrate(method, p_cal[:, c], y_cal[:, c], p_test[:, c])
    return out


class PhysioCalib:
    """Severity-weighted isotonic calibration.

    Standard isotonic regression is fit per class, but the calibration of
    high-severity conditions is regularised toward the empirical curve more
    strongly (via sample reweighting), allocating more effective calibration
    capacity to clinically critical classes. Rank-preserving per class.

    Parameters
    ----------
    severity_weights : dict[str, float]
        Map from class name to severity weight (>0). Classes absent from the
        map default to weight 1.0.
    """

    def __init__(self, severity_weights: dict | None = None):
        self.severity_weights = severity_weights or {}
        self._iso = {}
        self.classes_ = None

    def fit(self, probs_cal, labels_cal, classes: Sequence[str]):
        from sklearn.isotonic import IsotonicRegression
        probs_cal = np.asarray(probs_cal, float)
        labels_cal = np.asarray(labels_cal, float)
        self.classes_ = list(classes)
        for c, name in enumerate(self.classes_):
            yc, pc = labels_cal[:, c], probs_cal[:, c]
            if len(np.unique(yc)) < 2:
                self._iso[name] = None
                continue
            w = float(self.severity_weights.get(name, 1.0))
            # severity weight emphasises positive examples of critical classes
            sample_weight = np.where(yc > 0.5, w, 1.0)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(pc, yc, sample_weight=sample_weight)
            self._iso[name] = iso
        return self

    def predict(self, probs_test):
        if self.classes_ is None:
            raise RuntimeError("PhysioCalib.fit must be called before predict")
        probs_test = np.asarray(probs_test, float)
        out = np.array(probs_test, dtype=float)
        for c, name in enumerate(self.classes_):
            iso = self._iso.get(name)
            if iso is not None:
                out[:, c] = iso.predict(probs_test[:, c])
        return out
