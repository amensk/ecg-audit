"""ecg-audit: calibration auditing for ECG foundation models.

Quick start
-----------
>>> import numpy as np
>>> from ecg_audit import CalibrationAuditor, PhysioCalib
>>> auditor = CalibrationAuditor(classes=["NORM", "MI", "STTC", "CD", "HYP"])
>>> report = auditor.audit(probs, labels)
>>> print(report.macro_ece, report.macro_sce)
"""
from .metrics import (
    CalibrationAuditor, AuditReport,
    ece_equal_width, adaptive_ece, signed_calibration_error, brier, auroc,
)
from .recalibration import (
    PhysioCalib, recalibrate, recalibrate_multilabel,
    platt_scale, isotonic_calibrate, temperature_scale,
)
from .clinical import net_benefit, decision_curve, subgroup_calibration
from .plotting import reliability_diagram, reliability_curve
from .leaderboard import composite_score, make_entry, submit, rank

__version__ = "0.1.0"

__all__ = [
    "CalibrationAuditor", "AuditReport",
    "ece_equal_width", "adaptive_ece", "signed_calibration_error", "brier", "auroc",
    "PhysioCalib", "recalibrate", "recalibrate_multilabel",
    "platt_scale", "isotonic_calibrate", "temperature_scale",
    "net_benefit", "decision_curve", "subgroup_calibration",
    "reliability_diagram", "reliability_curve",
    "composite_score", "make_entry", "submit", "rank",
    "__version__",
]
