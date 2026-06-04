"""H4 calibration-size micro-ablation pilot for Stage 05, attempt 33.

Tests H4: "The learned MLP head outperforms temperature/Platt/isotonic at
large calibration sets but underperforms at small sets due to overfitting."

Uses ONLY cached logits from s4d_real_pilot_predictions.npz and
monotone_recalibration_pilot_predictions.npz — no GPU or FM loading needed.

Ablation sizes: 10, 20, 50, 75, 100, 150, full (~200)
Repetitions per size: 5 (random seeds 42-46)
Methods: temperature_scaling, monotone_platt, isotonic_regression, learned_mlp
Datasets: D1 (PTB-XL, 5 classes), D3 (CODE-15%, 6 classes)
Models: M6 (configured S4D), M2 (ST-MEM probe)

All evaluation is on the full fixed test set to remove test-set variance.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

WORKSPACE = Path(__file__).resolve().parent.parent
RESULTS = WORKSPACE / "results"
NOTES = WORKSPACE / "notes"
CODE = WORKSPACE / "code"
sys.path.insert(0, str(CODE))

from recalibration import MLPCalibrator
from evaluation import compute_ece

SEED = 42
# Calibration subsample sizes to test
ABLATION_SIZES = [10, 20, 50, 75, 100, 150, "full"]
N_REPS = 5
MIN_CLASS_SUPPORT = 2  # skip class if fewer positives in subsampled cal set


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def macro_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Macro-averaged ECE (equal-width) over all classes."""
    n_classes = probs.shape[1]
    ece_vals = []
    for c in range(n_classes):
        if labels[:, c].sum() < 1 or (1 - labels[:, c]).sum() < 1:
            continue
        ece_vals.append(compute_ece(probs[:, c], labels[:, c], n_bins=n_bins))
    return float(np.mean(ece_vals)) if ece_vals else float("nan")


def macro_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    n_classes = probs.shape[1]
    brier_vals = []
    for c in range(n_classes):
        if labels[:, c].sum() < 1 or (1 - labels[:, c]).sum() < 1:
            continue
        brier_vals.append(brier_score_loss(labels[:, c], probs[:, c]))
    return float(np.mean(brier_vals)) if brier_vals else float("nan")


def macro_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    n_classes = probs.shape[1]
    auroc_vals = []
    for c in range(n_classes):
        if labels[:, c].sum() < 1 or (1 - labels[:, c]).sum() < 1:
            continue
        try:
            auroc_vals.append(roc_auc_score(labels[:, c], probs[:, c]))
        except Exception:
            pass
    return float(np.mean(auroc_vals)) if auroc_vals else float("nan")


# ---------------------------------------------------------------------------
# Calibration fitters
# ---------------------------------------------------------------------------

def fit_temperature(cal_logits: np.ndarray, cal_labels: np.ndarray,
                    test_logits: np.ndarray) -> np.ndarray:
    """Per-class temperature scaling via NLL minimization."""
    n_classes = cal_logits.shape[1]
    temps = np.ones(n_classes)
    for c in range(n_classes):
        y = cal_labels[:, c]
        if y.sum() < MIN_CLASS_SUPPORT or (len(y) - y.sum()) < MIN_CLASS_SUPPORT:
            continue
        def nll(T, logits=cal_logits[:, c], labels=y):
            scaled = logits / max(T[0], 0.01)
            p = 1 / (1 + np.exp(-scaled))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
        try:
            res = minimize(nll, x0=[1.0], method="L-BFGS-B", bounds=[(0.01, 100.0)])
            temps[c] = float(res.x[0])
        except Exception:
            pass
    scaled_test = test_logits / temps[np.newaxis, :]
    return expit(scaled_test)


def fit_monotone_platt(cal_logits: np.ndarray, cal_labels: np.ndarray,
                       test_logits: np.ndarray) -> np.ndarray:
    """Per-class Platt scaling; falls back to temperature for negative slopes."""
    n_classes = cal_logits.shape[1]
    result_probs = expit(test_logits).copy()
    for c in range(n_classes):
        y = cal_labels[:, c]
        if y.sum() < MIN_CLASS_SUPPORT or (len(y) - y.sum()) < MIN_CLASS_SUPPORT:
            continue
        try:
            lr = LogisticRegression(C=1e6, fit_intercept=True, max_iter=1000)
            lr.fit(cal_logits[:, c:c+1], y)
            slope = float(lr.coef_[0, 0])
            if slope <= 0:
                # fallback to temperature scaling for this class
                def nll(T, logits=cal_logits[:, c], labels=y):
                    p = 1 / (1 + np.exp(-logits / max(T[0], 0.01)))
                    p = np.clip(p, 1e-10, 1 - 1e-10)
                    return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
                res = minimize(nll, x0=[1.0], method="L-BFGS-B", bounds=[(0.01, 100.0)])
                result_probs[:, c] = expit(test_logits[:, c] / float(res.x[0]))
            else:
                result_probs[:, c] = lr.predict_proba(test_logits[:, c:c+1])[:, 1]
        except Exception:
            pass
    return result_probs


def fit_isotonic(cal_probs: np.ndarray, cal_labels: np.ndarray,
                 test_probs: np.ndarray) -> np.ndarray:
    """Per-class isotonic regression on probabilities."""
    n_classes = cal_probs.shape[1]
    result_probs = test_probs.copy()
    for c in range(n_classes):
        y = cal_labels[:, c]
        if y.sum() < MIN_CLASS_SUPPORT or (len(y) - y.sum()) < MIN_CLASS_SUPPORT:
            continue
        try:
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            ir.fit(cal_probs[:, c], y)
            result_probs[:, c] = ir.predict(test_probs[:, c])
        except Exception:
            pass
    return result_probs


def fit_mlp(cal_logits: np.ndarray, cal_labels: np.ndarray,
            test_logits: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Learned MLP head via MLPCalibrator from recalibration.py."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    calibrator = MLPCalibrator()
    try:
        calibrator.fit(cal_logits.astype(np.float32), cal_labels.astype(np.float32))
        return calibrator.transform(test_logits.astype(np.float32)).astype(np.float64)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "MLP fit failed (n=%d): %s; returning raw probs.", len(cal_logits), exc
        )
        return expit(test_logits)


# ---------------------------------------------------------------------------
# Main ablation loop
# ---------------------------------------------------------------------------

def run_ablation(
    dataset_id: str,
    dataset_name: str,
    model_id: str,
    model_name: str,
    cal_logits: np.ndarray,
    cal_labels: np.ndarray,
    test_logits: np.ndarray,
    test_probs_raw: np.ndarray,
    test_labels: np.ndarray,
    logger: logging.Logger,
) -> list[dict]:
    cal_probs = expit(cal_logits)
    n_cal_full = len(cal_logits)
    rows = []

    for size in ABLATION_SIZES:
        n = n_cal_full if size == "full" else int(size)
        if n > n_cal_full:
            n = n_cal_full  # cap at available

        for rep in range(N_REPS):
            seed = SEED + rep
            rng = np.random.default_rng(seed)
            if n >= n_cal_full:
                idx = np.arange(n_cal_full)
                effective_n = n_cal_full
            else:
                idx = rng.choice(n_cal_full, size=n, replace=False)
                effective_n = n

            sub_logits = cal_logits[idx]
            sub_probs = cal_probs[idx]
            sub_labels = cal_labels[idx]

            # Count min class support
            min_pos = int(sub_labels.sum(axis=0).min())
            min_neg = int((1 - sub_labels).sum(axis=0).min())

            base_row = {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "model_id": model_id,
                "model_name": model_name,
                "cal_size": size if size == "full" else n,
                "effective_n": effective_n,
                "rep": rep,
                "seed": seed,
                "min_class_pos_in_cal": min_pos,
                "min_class_neg_in_cal": min_neg,
            }

            for method_name, probs_fn in [
                ("temperature_scaling", lambda: fit_temperature(sub_logits, sub_labels, test_logits)),
                ("monotone_platt", lambda: fit_monotone_platt(sub_logits, sub_labels, test_logits)),
                ("isotonic_regression", lambda: fit_isotonic(sub_probs, sub_labels, test_probs_raw)),
                ("learned_mlp_head", lambda s=seed: fit_mlp(sub_logits, sub_labels, test_logits, seed=s)),
            ]:
                try:
                    cal_test_probs = probs_fn()
                    ece = macro_ece(cal_test_probs, test_labels)
                    brier = macro_brier(cal_test_probs, test_labels)
                    auroc = macro_auroc(cal_test_probs, test_labels)
                    status = "ok"
                except Exception as exc:
                    logger.warning("Method %s failed for %s/%s n=%d rep=%d: %s",
                                   method_name, dataset_id, model_id, effective_n, rep, exc)
                    ece = brier = auroc = float("nan")
                    status = f"error:{type(exc).__name__}"

                rows.append({
                    **base_row,
                    "method": method_name,
                    "ece_ew": ece,
                    "brier": brier,
                    "auroc": auroc,
                    "status": status,
                })
                logger.info("  %s/%s n=%s rep=%d %s: ECE=%.4f Brier=%.4f AUROC=%.4f",
                            dataset_id, model_id, size, rep, method_name, ece, brier, auroc)
    return rows


# ---------------------------------------------------------------------------
# Summary statistics over reps
# ---------------------------------------------------------------------------

def summarize_ablation(rows: list[dict]) -> list[dict]:
    """Compute mean ± std over reps for each (dataset, model, size, method) cell."""
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["dataset_id"], r["model_id"], r["cal_size"], r["method"])
        groups[key].append(r)

    summary = []
    def _sort_key(item):
        (ds, mod, sz, mth) = item[0]
        return (ds, mod, 9999 if sz == "full" else int(sz), mth)

    for (dataset_id, model_id, cal_size, method), group in sorted(groups.items(), key=_sort_key):
        ece_vals = [r["ece_ew"] for r in group if not np.isnan(r["ece_ew"])]
        brier_vals = [r["brier"] for r in group if not np.isnan(r["brier"])]
        auroc_vals = [r["auroc"] for r in group if not np.isnan(r["auroc"])]
        summary.append({
            "dataset_id": dataset_id,
            "dataset_name": group[0]["dataset_name"],
            "model_id": model_id,
            "model_name": group[0]["model_name"],
            "cal_size": cal_size,
            "effective_n": group[0]["effective_n"],
            "method": method,
            "n_reps": len(group),
            "n_valid_reps": len(ece_vals),
            "ece_ew_mean": float(np.mean(ece_vals)) if ece_vals else float("nan"),
            "ece_ew_std": float(np.std(ece_vals)) if len(ece_vals) > 1 else 0.0,
            "brier_mean": float(np.mean(brier_vals)) if brier_vals else float("nan"),
            "brier_std": float(np.std(brier_vals)) if len(brier_vals) > 1 else 0.0,
            "auroc_mean": float(np.mean(auroc_vals)) if auroc_vals else float("nan"),
            "auroc_std": float(np.std(auroc_vals)) if len(auroc_vals) > 1 else 0.0,
        })
    return summary


def rank_methods_per_cell(summary: list[dict]) -> list[dict]:
    """For each (dataset, model, cal_size), rank methods by ECE and Brier."""
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in summary:
        key = (r["dataset_id"], r["model_id"], r["cal_size"])
        groups[key].append(r)

    ranking_rows = []
    def _sort_key2(item):
        (ds, mod, sz) = item[0]
        return (ds, mod, 9999 if sz == "full" else int(sz))

    for (dataset_id, model_id, cal_size), group in sorted(groups.items(), key=_sort_key2):
        valid = [r for r in group if not np.isnan(r["ece_ew_mean"])]
        if not valid:
            continue
        ece_sorted = sorted(valid, key=lambda r: r["ece_ew_mean"])
        brier_sorted = sorted(valid, key=lambda r: r["brier_mean"])
        ece_rank = {r["method"]: i + 1 for i, r in enumerate(ece_sorted)}
        brier_rank = {r["method"]: i + 1 for i, r in enumerate(brier_sorted)}
        for r in group:
            ranking_rows.append({
                "dataset_id": dataset_id,
                "dataset_name": r["dataset_name"],
                "model_id": model_id,
                "model_name": r["model_name"],
                "cal_size": cal_size,
                "effective_n": r["effective_n"],
                "method": r["method"],
                "ece_ew_mean": r["ece_ew_mean"],
                "brier_mean": r["brier_mean"],
                "ece_rank": ece_rank.get(r["method"], 99),
                "brier_rank": brier_rank.get(r["method"], 99),
                "best_by_ece": ece_rank.get(r["method"]) == 1,
                "best_by_brier": brier_rank.get(r["method"]) == 1,
                "mlp_worst_by_ece": ece_rank.get("learned_mlp_head") == len(valid),
                "mlp_worst_by_brier": brier_rank.get("learned_mlp_head") == len(valid),
            })
    return ranking_rows


def h4_analysis(ranking_rows: list[dict]) -> dict:
    """Assess H4 support from micro-ablation: is MLP worst at small n?"""
    from collections import defaultdict
    # Group by (dataset_id, model_id, cal_size) — one entry per method
    # Find cells where learned_mlp_head has the worst ECE rank
    mlp_worst_ece_by_size: dict[tuple, list[bool]] = defaultdict(list)
    mlp_worst_brier_by_size: dict[tuple, list[bool]] = defaultdict(list)
    for r in ranking_rows:
        if r["method"] == "learned_mlp_head":
            key = r["cal_size"]
            mlp_worst_ece_by_size[key].append(r["ece_rank"] == 4)
            mlp_worst_brier_by_size[key].append(r["brier_rank"] == 4)

    analysis = {}
    for size in sorted(mlp_worst_ece_by_size.keys(),
                       key=lambda x: (0, int(x)) if str(x).isdigit() else (1, 0)):
        ece_worst = mlp_worst_ece_by_size[size]
        brier_worst = mlp_worst_brier_by_size[size]
        analysis[str(size)] = {
            "mlp_worst_ece_fraction": sum(ece_worst) / len(ece_worst) if ece_worst else float("nan"),
            "mlp_worst_brier_fraction": sum(brier_worst) / len(brier_worst) if brier_worst else float("nan"),
            "n_cells": len(ece_worst),
        }
    return analysis


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    log_path = NOTES / "h4_size_ablation_pilot.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_path), mode="w"),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("H4 calibration-size micro-ablation pilot starting.")

    # Load cached predictions
    grad_pred = dict(np.load(RESULTS / "recalibration_gradient_pilot_predictions.npz"))
    s4d_pred = dict(np.load(RESULTS / "s4d_real_pilot_predictions.npz"))

    # Dataset metadata
    DATASETS = {
        "D1": {"name": "PTB-XL", "n_classes": 5,
               "class_names": ["NORM", "MI", "STTC", "CD", "HYP"]},
        "D3": {"name": "CODE-15%", "n_classes": 6,
               "class_names": ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]},
    }
    MODELS = {
        "M6_s4d_configured": "Configured S4D",
        "M2_stmem_probe": "ST-MEM probe",
    }

    all_rows = []

    for dataset_id, ds_info in DATASETS.items():
        dataset_name = ds_info["name"]
        # Calibration labels come from S4D pilot (same split for both models)
        cal_labels = s4d_pred[f"{dataset_id}_cal_labels"].astype(np.float64)
        test_labels = s4d_pred[f"{dataset_id}_test_labels"].astype(np.float64)

        for model_id, model_name in MODELS.items():
            key = f"{dataset_id}_{model_id}"
            cal_logits = grad_pred[f"{key}_cal_logits"].astype(np.float64)
            test_logits = grad_pred[f"{key}_test_logits"].astype(np.float64)
            test_probs_raw = grad_pred[f"{key}_test_raw_probs"].astype(np.float64)

            logger.info("Running ablation: %s / %s (cal_n=%d, test_n=%d)",
                        dataset_id, model_id, len(cal_logits), len(test_logits))
            rows = run_ablation(
                dataset_id, dataset_name, model_id, model_name,
                cal_logits, cal_labels,
                test_logits, test_probs_raw, test_labels,
                logger,
            )
            all_rows.extend(rows)

    # Summary and ranking
    summary = summarize_ablation(all_rows)
    ranking = rank_methods_per_cell(summary)
    h4 = h4_analysis(ranking)

    logger.info("H4 analysis by cal_size (MLP worst fraction):")
    for size, stats in h4.items():
        logger.info("  n=%s: mlp_worst_ece=%.2f, mlp_worst_brier=%.2f (n_cells=%d)",
                    size, stats["mlp_worst_ece_fraction"],
                    stats["mlp_worst_brier_fraction"], stats["n_cells"])

    # Write outputs
    out_json = RESULTS / "h4_size_ablation_pilot_results.json"
    out_summary_csv = RESULTS / "h4_size_ablation_pilot_summary.csv"
    out_ranking_csv = RESULTS / "h4_size_ablation_pilot_ranking.csv"
    out_raw_csv = RESULTS / "h4_size_ablation_pilot_raw.csv"

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "05_experimentation",
        "experiment": "h4_calibration_size_micro_ablation",
        "scope": (
            "H4 micro-ablation: all 4 calibration methods on cached 1000-record "
            "PTB-XL/CODE-15% pilot logits, varying cal set size {10,20,50,75,100,150,full}, "
            "5 reps per size, evaluated on fixed full test set."
        ),
        "publication_grade": False,
        "ablation_sizes": ABLATION_SIZES,
        "n_reps": N_REPS,
        "total_rows": len(all_rows),
        "h4_analysis_by_cal_size": h4,
        "summary": summary,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("JSON written to %s", out_json)

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(out_summary_csv, summary)
    write_csv(out_ranking_csv, ranking)
    write_csv(out_raw_csv, all_rows)
    logger.info("CSVs written to %s, %s, %s", out_summary_csv, out_ranking_csv, out_raw_csv)
    logger.info("H4 micro-ablation pilot complete. %d raw rows, %d summary rows.",
                len(all_rows), len(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
