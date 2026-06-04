#!/usr/bin/env python3
"""
TASK 1 - Extended calibration metrics for ALL cells in results/probs_manifest.json.

For each cell, over its valid classes (ac.valid_classes on y_test), compute macro AND
per-class: AUROC, ECE(equal-width), adaptive ECE, classwise-ECE (= mean per-class ECE),
Brier, NLL, SCE, calibration slope, calibration intercept. Metrics computed on TEST.

Writes:
  results/extended_metrics.json  (cell -> {macro {..}, per_class {cls -> {..}}})
  results/extended_metrics.csv   (one row per cell with macro values)

Uses ONLY the cached probs via ac.load_cell. No fabrication.
"""
import json, sys, csv
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
sys.path.insert(0, str(WS / "code"))
import analysis_common as ac

RESULTS = WS / "results"
MANIFEST = json.loads((RESULTS / "probs_manifest.json").read_text())

# metric registry: name -> fn(y, p)
METRIC_FNS = {
    "auroc": ac.auroc,
    "ece": ac.ece_equal_width,
    "adaptive_ece": ac.adaptive_ece,
    "brier": ac.brier,
    "nll": ac.nll,
    "sce": ac.sce,
}

def per_class_metrics(y, p):
    """Full per-class metric dict for one class column (binary y, prob p)."""
    d = {}
    for name, fn in METRIC_FNS.items():
        v = fn(y, p)
        d[name] = (None if (isinstance(v, float) and np.isnan(v)) else float(v))
    slope, intercept = ac.calibration_slope_intercept(y, p)
    d["cal_slope"] = (None if np.isnan(slope) else float(slope))
    d["cal_intercept"] = (None if np.isnan(intercept) else float(intercept))
    return d

def _mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else None

def main():
    out = {}
    csv_rows = []
    for cell in MANIFEST["cells"]:
        mid, did = cell["model_id"], cell["dataset_id"]
        key = f"{mid}_{did}"
        c = ac.load_cell(mid, did)
        Y, P = c["y_test"], c["p_test"]
        classes = list(c["classes"])
        # valid classes recomputed on y_test (authoritative)
        cols = ac.valid_classes(Y)

        per_class = {}
        for ci in cols:
            per_class[classes[ci]] = per_class_metrics(Y[:, ci], P[:, ci])

        # macro = mean over per-class metrics (only valid classes)
        macro = {}
        for name in list(METRIC_FNS.keys()) + ["cal_slope", "cal_intercept"]:
            macro[name] = _mean([per_class[classes[ci]][name] for ci in cols])
        # classwise-ECE reported explicitly = mean per-class equal-width ECE
        macro["classwise_ece"] = _mean([per_class[classes[ci]]["ece"] for ci in cols])

        out[key] = {
            "model_id": mid, "model": cell["model"],
            "dataset_id": did, "dataset": cell["dataset"],
            "n_test": int(len(Y)), "n_cal": int(len(c["y_cal"])),
            "valid_class_idx": cols,
            "valid_classes": [classes[ci] for ci in cols],
            "macro": macro,
            "per_class": per_class,
        }

        row = {"cell": key, "model_id": mid, "model": cell["model"],
               "dataset_id": did, "dataset": cell["dataset"],
               "n_test": int(len(Y)), "n_valid_classes": len(cols)}
        for name in ["auroc", "ece", "adaptive_ece", "classwise_ece",
                     "brier", "nll", "sce", "cal_slope", "cal_intercept"]:
            row[f"macro_{name}"] = macro[name]
        csv_rows.append(row)

    (RESULTS / "extended_metrics.json").write_text(json.dumps(out, indent=2))
    fields = list(csv_rows[0].keys())
    with open(RESULTS / "extended_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)

    # ---- print requested table: macro NLL, slope, intercept, adaptive ECE for PTB-XL (D1) & MIMIC (D5)
    def fmt(v):
        return "  nan " if v is None else f"{v:6.4f}"
    print("\n=== TASK 1: Macro metrics for PTB-XL (D1) and MIMIC-IV-ECG (D5) cells ===")
    print(f"{'cell':9s} {'dataset':14s} {'NLL':>7s} {'slope':>7s} {'intercept':>9s} {'adaECE':>7s}")
    for r in csv_rows:
        if r["dataset_id"] in ("D1", "D5"):
            print(f"{r['cell']:9s} {r['dataset']:14s} "
                  f"{fmt(r['macro_nll'])} {fmt(r['macro_cal_slope'])} "
                  f"{fmt(r['macro_cal_intercept']):>9s} {fmt(r['macro_adaptive_ece'])}")
    print(f"\nWrote results/extended_metrics.json and results/extended_metrics.csv "
          f"({len(csv_rows)} cells)")

if __name__ == "__main__":
    main()
