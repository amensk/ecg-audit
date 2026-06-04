import numpy as np
from sklearn.metrics import roc_auc_score
from ecg_audit import (platt_scale, isotonic_calibrate, temperature_scale,
                       recalibrate_multilabel, PhysioCalib, ece_equal_width)


def _miscalibrated(n=4000, seed=0):
    """Generate overconfident probabilities with a known better calibration."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)
    p_true = 1 / (1 + np.exp(-z))
    y = (rng.uniform(0, 1, n) < p_true).astype(float)
    # overconfident: push probs toward 0/1
    p_over = np.clip(1 / (1 + np.exp(-1.8 * z)), 1e-4, 1 - 1e-4)
    return y, p_over


def test_platt_preserves_auroc():
    y, p = _miscalibrated()
    yc, pc = y[:2000], p[:2000]; yt, pt = y[2000:], p[2000:]
    pr = platt_scale(pc, yc, pt)
    assert abs(roc_auc_score(yt, pr) - roc_auc_score(yt, pt)) < 1e-9


def test_isotonic_reduces_ece():
    y, p = _miscalibrated()
    yc, pc = y[:2000], p[:2000]; yt, pt = y[2000:], p[2000:]
    pr = isotonic_calibrate(pc, yc, pt)
    assert ece_equal_width(yt, pr) < ece_equal_width(yt, pt)


def test_temperature_reduces_ece_and_preserves_ranking():
    y, p = _miscalibrated()
    yc, pc = y[:2000], p[:2000]; yt, pt = y[2000:], p[2000:]
    pr = temperature_scale(pc, yc, pt)
    assert ece_equal_width(yt, pr) <= ece_equal_width(yt, pt) + 1e-6
    # monotone transform preserves AUROC
    assert abs(roc_auc_score(yt, pr) - roc_auc_score(yt, pt)) < 1e-6


def test_multilabel_shape():
    y, p = _miscalibrated(n=1000)
    Y = np.stack([y, 1 - y], axis=1); P = np.stack([p, 1 - p], axis=1)
    out = recalibrate_multilabel("isotonic", P[:500], Y[:500], P[500:])
    assert out.shape == (500, 2)


def test_physiocalib_runs_and_preserves_shape():
    y, p = _miscalibrated(n=2000)
    Y = np.stack([y, y], axis=1); P = np.stack([p, p], axis=1)
    pc = PhysioCalib(severity_weights={"MI": 3.0}).fit(P[:1000], Y[:1000], ["MI", "NORM"])
    out = pc.predict(P[1000:])
    assert out.shape == (1000, 2)
    assert np.all((out >= 0) & (out <= 1))
