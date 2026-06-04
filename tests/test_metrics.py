import numpy as np
import pytest
from ecg_audit import (CalibrationAuditor, ece_equal_width, adaptive_ece,
                       signed_calibration_error, brier, auroc)


def test_perfect_calibration_low_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(float)  # perfectly calibrated by construction
    assert ece_equal_width(y, p, 15) < 0.02
    assert adaptive_ece(y, p, 15) < 0.02


def test_overconfident_positive_sce():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1.0])
    p = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])  # mean prob > mean label
    assert signed_calibration_error(y, p) > 0


def test_brier_bounds():
    y = np.array([0, 1, 0, 1.0])
    assert brier(y, y) == 0.0
    assert 0 <= brier(y, 1 - y) <= 1.0


def test_auroc_perfect_and_degenerate():
    y = np.array([0, 0, 1, 1.0]); p = np.array([0.1, 0.2, 0.8, 0.9])
    assert auroc(y, p) == 1.0
    assert np.isnan(auroc(np.ones(5), np.random.rand(5)))  # single class -> nan


def test_auditor_min_support_excludes_degenerate():
    # class 0 has only 1 positive -> excluded; class 1 balanced -> included
    n = 200
    rng = np.random.default_rng(1)
    labels = np.zeros((n, 2))
    labels[0, 0] = 1                      # 1 positive only
    labels[:100, 1] = 1                   # 100 positives
    probs = rng.uniform(0, 1, (n, 2))
    rep = CalibrationAuditor(classes=["rare", "bal"], min_pos=10, min_neg=10).audit(probs, labels)
    assert rep.per_class["rare"]["included"] is False
    assert rep.per_class["bal"]["included"] is True
    assert rep.n_classes_evaluated == 1


def test_auditor_shape_validation():
    aud = CalibrationAuditor(classes=["a", "b"])
    with pytest.raises(ValueError):
        aud.audit(np.zeros((10, 3)), np.zeros((10, 3)))  # 3 cols != 2 classes
