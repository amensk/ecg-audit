"""Expanded real-data pilot for Stage 05 refinement.

Runs a larger PTB-XL + CODE-15% pilot than ``run_minimal_real_pilot.py`` and
adds two controls requested in the refinement pass:

1. guarded per-class Platt scaling, which skips calibration fitting when the
   calibration or test split has too few positive/negative examples;
2. per-class DeLong tests comparing raw vs guarded-Platt AUROC to diagnose
   whether the earlier PTB-XL AUROC anomaly persists on a larger subset.

This is still a pilot: S4-lite is intentionally reduced and ST-MEM uses a
logistic linear probe on frozen features.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from evaluation import evaluate_model_on_dataset
from run_minimal_real_pilot import (
    RESULTS,
    NOTES,
    SEED,
    _evaluate_stmem_probe,
    _load_code15,
    _load_ptbxl,
    _load_stmem_model,
    _multilabel_split,
    _standardize,
    _train_s4_lite,
)
from statistical_tests import delong_test


def _setup_logging() -> logging.Logger:
    NOTES.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("expanded_real_pilot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(NOTES / "expanded_real_pilot.log", mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _fit_guarded_platt(
    cal_logits: np.ndarray,
    cal_labels: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    min_cal_pos: int,
    min_cal_neg: int,
    min_test_pos: int,
    min_test_neg: int,
) -> dict:
    raw_probs = 1.0 / (1.0 + np.exp(-test_logits))
    calibrated = raw_probs.copy().astype(np.float64)
    params = []
    for c in range(cal_logits.shape[1]):
        y_cal = cal_labels[:, c]
        y_test = test_labels[:, c]
        cal_pos = int(y_cal.sum())
        cal_neg = int(len(y_cal) - cal_pos)
        test_pos = int(y_test.sum())
        test_neg = int(len(y_test) - test_pos)
        guard = {
            "class_index": c,
            "cal_pos": cal_pos,
            "cal_neg": cal_neg,
            "test_pos": test_pos,
            "test_neg": test_neg,
            "min_cal_pos": min_cal_pos,
            "min_cal_neg": min_cal_neg,
            "min_test_pos": min_test_pos,
            "min_test_neg": min_test_neg,
        }
        if cal_pos < min_cal_pos or cal_neg < min_cal_neg:
            guard["status"] = "skipped_insufficient_calibration_support"
            calibrated[:, c] = raw_probs[:, c]
            params.append(guard)
            continue
        if test_pos < min_test_pos or test_neg < min_test_neg:
            guard["status"] = "fit_but_excluded_from_delong_due_to_test_support"
        else:
            guard["status"] = "fit"
        lr = LogisticRegression(solver="lbfgs", max_iter=1000, C=1e3)
        lr.fit(cal_logits[:, c : c + 1], y_cal)
        calibrated[:, c] = lr.predict_proba(test_logits[:, c : c + 1])[:, 1]
        guard["coef"] = lr.coef_.tolist()
        guard["intercept"] = lr.intercept_.tolist()
        params.append(guard)
    return {"probs": calibrated, "params": params}


def _delong_raw_vs_platt(
    labels: np.ndarray,
    raw_probs: np.ndarray,
    platt_probs: np.ndarray,
    class_names: list[str],
    platt_params: list[dict],
    min_test_pos: int,
    min_test_neg: int,
) -> dict:
    out = {}
    param_by_class = {int(p["class_index"]): p for p in platt_params}
    for c, name in enumerate(class_names):
        y = labels[:, c]
        pos = int(y.sum())
        neg = int(len(y) - pos)
        row = {
            "class_name": name,
            "test_pos": pos,
            "test_neg": neg,
            "platt_status": param_by_class.get(c, {}).get("status", "unknown"),
        }
        if pos < min_test_pos or neg < min_test_neg:
            row["status"] = "skipped_insufficient_test_support"
            out[name] = row
            continue
        try:
            dl = delong_test(y, raw_probs[:, c], platt_probs[:, c])
            row.update(dl)
            row["status"] = "computed"
            row["raw_minus_platt_auc"] = dl["diff"]
            row["platt_minus_raw_auc"] = -dl["diff"]
            row["nominal_significant_degradation"] = dl["p_value"] < 0.05 and dl["diff"] > 0
        except Exception as exc:
            row["status"] = "failed"
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
        out[name] = row
    return out


def _flatten_rows(out: dict) -> list[dict]:
    rows = []
    for row in out["phase_a_metrics"]:
        m = row["metrics"]
        rows.append({
            "phase": "A_expanded_real_pilot",
            "dataset_id": row["dataset_id"],
            "dataset_name": row["dataset_name"],
            "model_id": row["model_id"],
            "model_name": row["model_name"],
            "recal_method": "none",
            "auroc": m.get("auroc"),
            "auprc": m.get("auprc"),
            "ece_ew": m.get("ece_equal_width"),
            "ece_em": m.get("ece_equal_mass"),
            "brier": m.get("brier"),
            "sce": m.get("sce"),
        })
    for row in out["recalibration_metrics"]:
        m = row["metrics"]
        rows.append({
            "phase": "B_expanded_real_pilot",
            "dataset_id": row["dataset_id"],
            "dataset_name": row["dataset_name"],
            "model_id": row["model_id"],
            "model_name": row["model_name"],
            "recal_method": row["recal_method"],
            "auroc": m.get("auroc"),
            "auprc": m.get("auprc"),
            "ece_ew": m.get("ece_equal_width"),
            "ece_em": m.get("ece_equal_mass"),
            "brier": m.get("brier"),
            "sce": m.get("sce"),
        })
    return rows


def _flatten_delong_rows(out: dict) -> list[dict]:
    rows = []
    for comparison in out["delong_raw_vs_guarded_platt"]:
        for class_name, result in comparison["per_class"].items():
            rows.append({
                "dataset_id": comparison["dataset_id"],
                "dataset_name": comparison["dataset_name"],
                "model_id": comparison["model_id"],
                "model_name": comparison["model_name"],
                "recal_method": comparison["recal_method"],
                "class_name": class_name,
                "status": result.get("status"),
                "platt_status": result.get("platt_status"),
                "test_pos": result.get("test_pos"),
                "test_neg": result.get("test_neg"),
                "auc_raw": result.get("auc_a"),
                "auc_platt": result.get("auc_b"),
                "raw_minus_platt_auc": result.get("raw_minus_platt_auc"),
                "platt_minus_raw_auc": result.get("platt_minus_raw_auc"),
                "z_statistic": result.get("z_statistic"),
                "p_value": result.get("p_value"),
                "nominal_significant_degradation": result.get(
                    "nominal_significant_degradation"
                ),
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--min-cal-pos", type=int, default=10)
    parser.add_argument("--min-cal-neg", type=int, default=10)
    parser.add_argument("--min-test-pos", type=int, default=10)
    parser.add_argument("--min-test-neg", type=int, default=10)
    args = parser.parse_args()

    logger = _setup_logging()
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "05_experimentation",
        "experiment": "expanded_real_data_pilot_guarded_platt_delong",
        "scope": (
            "PTB-XL + CODE-15%, expanded subset, S4-lite baseline, ST-MEM linear "
            "probe, guarded Platt recalibration, per-class DeLong diagnostics"
        ),
        "publication_grade": False,
        "parameters": vars(args),
        "limitations": [
            "Expanded real-data pilot, not full benchmark.",
            "S4-lite remains reduced relative to the approved S4D baseline.",
            "ST-MEM remains a logistic linear probe on frozen features.",
            "Only Platt scaling is tested in this refinement pass.",
            "Per-class DeLong tests are diagnostic and not Bonferroni-powered for final H6 claims.",
        ],
        "datasets": {},
        "phase_a_metrics": [],
        "recalibration_metrics": [],
        "delong_raw_vs_guarded_platt": [],
    }

    stmem_status, stmem_model = _load_stmem_model(logger)
    out["fm_checkpoint"] = stmem_status

    for loader in (_load_ptbxl, _load_code15):
        try:
            ds = loader(args.max_records, logger)
            split = _multilabel_split(ds.patient_ids, ds.labels)
            ds.signals = _standardize(ds.signals, split["train"])
            test_labels = ds.labels[split["test"]]

            out["datasets"][ds.dataset_id] = {
                "dataset_name": ds.dataset_name,
                "n_loaded": int(len(ds.labels)),
                "n_train": int(len(split["train"])),
                "n_cal": int(len(split["cal"])),
                "n_test": int(len(split["test"])),
                "n_classes": int(ds.labels.shape[1]),
                "class_names": ds.class_names,
                "label_prevalence": {
                    name: float(ds.labels[:, i].mean()) for i, name in enumerate(ds.class_names)
                },
                "test_class_counts": {
                    name: {
                        "pos": int(test_labels[:, i].sum()),
                        "neg": int(len(test_labels) - test_labels[:, i].sum()),
                    }
                    for i, name in enumerate(ds.class_names)
                },
                "leakage_overlap_counts": split["leakage"],
                "source_note": ds.source_note,
            }

            model_results = []
            s4_result = _train_s4_lite(ds, split, logger)
            model_results.append({
                "model_id": "M6_s4_lite",
                "model_name": "S4-lite expanded real pilot",
                "result": s4_result,
                "extra": {},
            })
            if stmem_model is not None:
                stmem_result = _evaluate_stmem_probe(stmem_model, ds, split, logger)
                model_results.append({
                    "model_id": "M2_stmem_probe",
                    "model_name": "ST-MEM frozen linear probe expanded real pilot",
                    "result": stmem_result,
                    "extra": {
                        "feature_shape": stmem_result["feature_shape"],
                        "probe_params": stmem_result["probe_params"],
                    },
                })

            for model_result in model_results:
                result = model_result["result"]
                out["phase_a_metrics"].append({
                    "dataset_id": ds.dataset_id,
                    "dataset_name": ds.dataset_name,
                    "model_id": model_result["model_id"],
                    "model_name": model_result["model_name"],
                    "metrics": result["raw_metrics"]["macro"],
                    "per_class": result["raw_metrics"]["per_class"],
                    **model_result["extra"],
                })
                platt = _fit_guarded_platt(
                    result["cal_logits"],
                    ds.labels[split["cal"]],
                    result["test_logits"],
                    test_labels,
                    args.min_cal_pos,
                    args.min_cal_neg,
                    args.min_test_pos,
                    args.min_test_neg,
                )
                platt_metrics = evaluate_model_on_dataset(
                    f"{model_result['model_id']}_guarded_platt",
                    ds.dataset_id,
                    platt["probs"],
                    test_labels,
                    ds.class_names,
                )
                out["recalibration_metrics"].append({
                    "dataset_id": ds.dataset_id,
                    "dataset_name": ds.dataset_name,
                    "model_id": model_result["model_id"],
                    "model_name": model_result["model_name"],
                    "recal_method": "guarded_platt_scaling",
                    "metrics": platt_metrics["macro"],
                    "per_class": platt_metrics["per_class"],
                    "platt_params": platt["params"],
                })
                delong = _delong_raw_vs_platt(
                    test_labels,
                    result["test_probs"],
                    platt["probs"],
                    ds.class_names,
                    platt["params"],
                    args.min_test_pos,
                    args.min_test_neg,
                )
                out["delong_raw_vs_guarded_platt"].append({
                    "dataset_id": ds.dataset_id,
                    "dataset_name": ds.dataset_name,
                    "model_id": model_result["model_id"],
                    "model_name": model_result["model_name"],
                    "recal_method": "guarded_platt_scaling",
                    "per_class": delong,
                    "n_nominal_significant_degradation": sum(
                        1 for d in delong.values()
                        if d.get("nominal_significant_degradation")
                    ),
                })
        except Exception as exc:
            logger.exception("Expanded pilot failed for loader %s", loader.__name__)
            out["datasets"][loader.__name__] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    failed_datasets = [
        key for key, value in out["datasets"].items()
        if isinstance(value, dict) and value.get("status") == "failed"
    ]
    out["failed_datasets"] = failed_datasets
    out["status"] = "completed" if not failed_datasets else "completed_partial_with_dataset_blocker"

    json_path = RESULTS / "expanded_real_pilot_results.json"
    csv_path = RESULTS / "expanded_real_pilot_summary.csv"
    delong_csv_path = RESULTS / "expanded_real_pilot_delong.csv"
    json_path.write_text(json.dumps(out, indent=2))
    rows = _flatten_rows(out)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "dataset_id",
                "dataset_name",
                "model_id",
                "model_name",
                "recal_method",
                "auroc",
                "auprc",
                "ece_ew",
                "ece_em",
                "brier",
                "sce",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    delong_rows = _flatten_delong_rows(out)
    with delong_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset_id",
                "dataset_name",
                "model_id",
                "model_name",
                "recal_method",
                "class_name",
                "status",
                "platt_status",
                "test_pos",
                "test_neg",
                "auc_raw",
                "auc_platt",
                "raw_minus_platt_auc",
                "platt_minus_raw_auc",
                "z_statistic",
                "p_value",
                "nominal_significant_degradation",
            ],
        )
        writer.writeheader()
        writer.writerows(delong_rows)

    print(json.dumps({
        "status": out["status"],
        "results": str(json_path),
        "summary": str(csv_path),
        "delong_summary": str(delong_csv_path),
        "datasets": out["datasets"],
        "n_delong_comparisons": len(out["delong_raw_vs_guarded_platt"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
