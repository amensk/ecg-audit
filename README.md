# ecg-audit

**Calibration auditing and recalibration for ECG foundation models.**

`ecg-audit` is a small, dependency-light Python package for measuring and
improving the *probability calibration* of ECG classifiers — not just their
ranking (AUROC). It accompanies the paper **"When Better Rankings Mean Worse
Probabilities: A Calibration Audit of ECG Foundation Models"**, and provides the
exact metrics, recalibrators, decision-curve analysis, subgroup tools, and a
leaderboard scaffold used there.

> Clinical motivation: a model can rank patients perfectly (high AUROC) yet emit
> systematically over- or under-confident probabilities. When those probabilities
> drive treatment thresholds, miscalibration is a patient-safety issue that AUROC
> cannot see.

## Why calibration?

Across a 6-model × 4-dataset benchmark (PTB-XL, CPSC2018, CODE-15%,
MIMIC-IV-ECG), we find:

- **Discrimination ≠ calibration.** On multi-label PTB-XL, the best-ranking
  foundation model (ST-MEM, AUROC 0.838) is the *worst* calibrated among the
  encoder FMs (ECE 0.124), while a supervised S4D baseline is best calibrated
  (ECE 0.064). HuBERT-ECG, a masked-prediction model, has the highest ECE (0.136).
- **Post-hoc recalibration helps a lot and costs little.** Isotonic regression
  cuts PTB-XL ECE by **43–71%** at ≤1.4% AUROC cost.
- **Within-domain pretraining is not sufficient.** On MIMIC-IV-ECG, ECG-FM
  (which was *pretrained on MIMIC-IV-ECG*) trails MERL and even the S4D baseline
  on linear-probe AUROC.
- **Calibration is context-dependent.** The same model's ECE varies several-fold
  across datasets, so calibration must be audited on locally representative data.

All headline numbers are reproducible from the artifacts in `results/` and the
scripts in `benchmark/`.

## Install

```bash
pip install git+https://github.com/amensk/ecg-audit.git
# or, from a clone:
pip install -e .
```

Requires Python ≥ 3.9 and `numpy`, `scipy`, `scikit-learn`, `pandas`,
`matplotlib`.

## Quick start (Python API)

```python
import numpy as np
from ecg_audit import CalibrationAuditor, PhysioCalib

probs  = np.load("predictions.npy")  # (n, n_classes), per-class probabilities
labels = np.load("labels.npy")       # (n, n_classes), binary
classes = ["NORM", "MI", "STTC", "CD", "HYP"]

# Audit raw calibration (ECE, adaptive ECE, SCE, Brier, AUROC; macro + per class)
report = CalibrationAuditor(classes=classes).audit(probs, labels)
print(report)                        # AuditReport(macro_auroc=..., macro_ece=..., ...)
print(report.macro_ece, report.macro_sce)

# Severity-weighted isotonic recalibration
pc = PhysioCalib(severity_weights={"MI": 3.0, "STTC": 2.0, "CD": 1.5,
                                   "HYP": 1.2, "NORM": 0.5})
pc.fit(probs_cal, labels_cal, classes)
probs_calibrated = pc.predict(probs_test)
```

Standard recalibrators are one call each:

```python
from ecg_audit import recalibrate_multilabel
p_isotonic    = recalibrate_multilabel("isotonic",    p_cal, y_cal, p_test)
p_platt       = recalibrate_multilabel("platt",       p_cal, y_cal, p_test)
p_temperature = recalibrate_multilabel("temperature", p_cal, y_cal, p_test)
```

Clinical utility and subgroup tools:

```python
from ecg_audit import decision_curve, net_benefit, subgroup_calibration
dc = decision_curve(y_true, y_prob)                 # net-benefit vs treat-all/none
sg = subgroup_calibration(probs, labels, age_group, classes)  # ECE/SCE/AUROC by group
```

## Quick start (CLI)

```bash
# Audit a predictions CSV (columns: <COND>_prob, <COND>_true)
ecg-audit audit predictions.csv \
  --conditions MI,STTC,NORM \
  --output report.json \
  --plot reliability/

# Recalibrate with isotonic regression
ecg-audit recalibrate --method isotonic \
  --cal-data cal.csv --test-data test.csv --output recalibrated.csv

# Append a calibration-aware entry to a community leaderboard
ecg-audit submit --model ST-MEM --dataset PTB-XL --auroc 0.838 --ece 0.124 \
  --leaderboard leaderboard.json
```

## Repository layout

```
ecg_audit/        the pip-installable package (metrics, recalibration, clinical, plotting, cli)
tests/            unit tests (pytest)
benchmark/        research scripts to reproduce the paper's benchmark
paper/            LaTeX source, figures, and compiled PDF
results/          machine-readable result artifacts (JSON/CSV) backing every claim
```

Large inputs (the clinical ECG datasets and third-party model checkpoints) are
**not** included — they are governed by their own licenses / data use
agreements. `benchmark/` documents how to obtain and stage them.

## Reproducing the benchmark

```bash
pip install -e .
python benchmark/run_benchmark_v2.py        # 6 models x 4 datasets, bootstrap CIs
python benchmark/run_code15_expanded.py     # multi-part CODE-15% (rare-class support)
python benchmark/make_paper_assets_v2.py    # figures + LaTeX tables from results JSON
```

Feature extraction requires the model checkpoints and datasets to be staged
under `workspace/` as documented in `benchmark/`. Cached per-class metrics and
summaries are committed under `results/` so the figures and tables can be
regenerated without re-running feature extraction.

## Tests

```bash
pip install -e ".[test]"
pytest -q          # 14 tests covering metrics, recalibration, and clinical utility
```

## Citation

```bibtex
@article{ecgaudit2026,
  title  = {When Better Rankings Mean Worse Probabilities:
            A Calibration Audit of ECG Foundation Models},
  author = {AutoR Research Workflow},
  year   = {2026}
}
```

## License

MIT (code). The datasets and model checkpoints used in the benchmark are **not**
redistributed and remain under their respective licenses and data use
agreements (PTB-XL, CPSC2018, CODE-15%, MIMIC-IV-ECG, and the foundation-model
checkpoints).
