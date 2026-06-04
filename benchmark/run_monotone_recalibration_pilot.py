"""Monotone recalibration guard for the Stage 05 real-data pilot.

This script reruns the same-split S4D and ST-MEM pilot recalibration with a
failure-isolation rule: per-class Platt scaling is accepted only when the
fitted slope is nonnegative. Negative Platt slopes fall back to temperature
scaling, which is monotone in the original logits and therefore cannot invert
rankings.
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
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from evaluation import evaluate_model_on_dataset
from run_minimal_real_pilot import (
    NOTES,
    RESULTS,
    SEED,
    _evaluate_stmem_probe,
    _load_code15,
    _load_ptbxl,
    _load_stmem_model,
    _multilabel_split,
    _standardize,
)
from statistical_tests import delong_test


def _setup_logging() -> logging.Logger:
    NOTES.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("monotone_recalibration_pilot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(NOTES / "monotone_recalibration_pilot.log", mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _fit_temperature(cal_logits: np.ndarray, cal_labels: np.ndarray) -> dict:
    labels = cal_labels.astype(np.float64)
    logits = cal_logits.astype(np.float64)

    def nll(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        z = logits / temperature
        return float(np.mean(np.logaddexp(0.0, z) - labels * z))

    result = minimize_scalar(
        nll,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-4},
    )
    temperature = float(np.exp(result.x))
    return {
        "temperature": temperature,
        "objective": float(result.fun),
        "success": bool(result.success),
        "n_iter": int(result.nfev),
    }


def _fit_monotone_platt_with_temperature_fallback(
    cal_logits: np.ndarray,
    cal_labels: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    min_cal_pos: int,
    min_cal_neg: int,
    min_test_pos: int,
    min_test_neg: int,
    min_platt_slope: float,
) -> dict:
    raw_probs = expit(test_logits)
    calibrated = raw_probs.copy().astype(np.float64)
    params = []

    for c in range(cal_logits.shape[1]):
        y_cal = cal_labels[:, c]
        y_test = test_labels[:, c]
        cal_pos = int(y_cal.sum())
        cal_neg = int(len(y_cal) - cal_pos)
        test_pos = int(y_test.sum())
        test_neg = int(len(y_test) - test_pos)
        row = {
            "class_index": c,
            "cal_pos": cal_pos,
            "cal_neg": cal_neg,
            "test_pos": test_pos,
            "test_neg": test_neg,
            "min_cal_pos": min_cal_pos,
            "min_cal_neg": min_cal_neg,
            "min_test_pos": min_test_pos,
            "min_test_neg": min_test_neg,
            "min_platt_slope": min_platt_slope,
        }

        if cal_pos < min_cal_pos or cal_neg < min_cal_neg:
            row.update({
                "status": "skipped_insufficient_calibration_support",
                "used_method": "raw_identity",
            })
            calibrated[:, c] = raw_probs[:, c]
            params.append(row)
            continue

        if test_pos < min_test_pos or test_neg < min_test_neg:
            row["delong_status_hint"] = "exclude_due_to_test_support"
        else:
            row["delong_status_hint"] = "eligible"

        try:
            lr = LogisticRegression(solver="lbfgs", max_iter=1000, C=1e3)
            lr.fit(cal_logits[:, c : c + 1], y_cal)
            slope = float(lr.coef_[0, 0])
            row["platt_slope"] = slope
            row["platt_intercept"] = float(lr.intercept_[0])
            if slope >= min_platt_slope:
                calibrated[:, c] = lr.predict_proba(test_logits[:, c : c + 1])[:, 1]
                row.update({
                    "status": "fit_platt_nonnegative_slope",
                    "used_method": "platt_scaling",
                })
            else:
                temp = _fit_temperature(cal_logits[:, c], y_cal)
                calibrated[:, c] = expit(test_logits[:, c] / temp["temperature"])
                row.update({
                    "status": "fallback_temperature_negative_platt_slope",
                    "used_method": "temperature_scaling",
                    **temp,
                })
        except Exception as exc:
            temp = _fit_temperature(cal_logits[:, c], y_cal)
            calibrated[:, c] = expit(test_logits[:, c] / temp["temperature"])
            row.update({
                "status": f"fallback_temperature_after_platt_error:{type(exc).__name__}",
                "used_method": "temperature_scaling",
                "platt_error": str(exc),
                **temp,
            })
        params.append(row)

    return {"probs": calibrated, "params": params}


def _delong_raw_vs_recalibrated(
    labels: np.ndarray,
    raw_probs: np.ndarray,
    recal_probs: np.ndarray,
    class_names: list[str],
    params: list[dict],
    min_test_pos: int,
    min_test_neg: int,
) -> dict:
    out = {}
    param_by_class = {int(p["class_index"]): p for p in params}
    for c, class_name in enumerate(class_names):
        y = labels[:, c]
        pos = int(y.sum())
        neg = int(len(y) - pos)
        param = param_by_class.get(c, {})
        row = {
            "class_name": class_name,
            "test_pos": pos,
            "test_neg": neg,
            "recal_status": param.get("status", "unknown"),
            "used_method": param.get("used_method", "unknown"),
        }
        if pos < min_test_pos or neg < min_test_neg:
            row["status"] = "skipped_insufficient_test_support"
            out[class_name] = row
            continue
        try:
            dl = delong_test(y, raw_probs[:, c], recal_probs[:, c])
            row.update(dl)
            row["status"] = "computed"
            row["raw_minus_recal_auc"] = dl["diff"]
            row["recal_minus_raw_auc"] = -dl["diff"]
            row["nominal_significant_degradation"] = dl["p_value"] < 0.05 and dl["diff"] > 0
        except Exception as exc:
            row["status"] = "failed"
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
        out[class_name] = row
    return out


def _metric_row(phase: str, dataset_id: str, dataset_name: str, model_id: str,
                model_name: str, recal_method: str, metrics: dict) -> dict:
    return {
        "phase": phase,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "model_id": model_id,
        "model_name": model_name,
        "recal_method": recal_method,
        "auroc": metrics.get("auroc"),
        "auprc": metrics.get("auprc"),
        "ece_ew": metrics.get("ece_equal_width"),
        "ece_em": metrics.get("ece_equal_mass"),
        "brier": metrics.get("brier"),
        "sce": metrics.get("sce"),
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_s4d_inputs() -> tuple[dict, dict]:
    result_path = RESULTS / "s4d_real_pilot_results.json"
    pred_path = RESULTS / "s4d_real_pilot_predictions.npz"
    if not result_path.exists() or not pred_path.exists():
        raise FileNotFoundError("S4D pilot outputs are required before monotone recalibration.")
    return json.loads(result_path.read_text()), dict(np.load(pred_path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--min-cal-pos", type=int, default=10)
    parser.add_argument("--min-cal-neg", type=int, default=10)
    parser.add_argument("--min-test-pos", type=int, default=10)
    parser.add_argument("--min-test-neg", type=int, default=10)
    parser.add_argument("--min-platt-slope", type=float, default=0.0)
    args = parser.parse_args()

    logger = _setup_logging()
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    s4d_results, s4d_arrays = _load_s4d_inputs()
    stmem_status, stmem_model = _load_stmem_model(logger)

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "05_experimentation",
        "experiment": "monotone_recalibration_guard_pilot",
        "scope": (
            "Same 1000-record PTB-XL + CODE-15% pilot; S4D predictions reused "
            "from configured S4D run; ST-MEM probe recomputed; Platt accepted only "
            "with nonnegative slope, otherwise temperature scaling fallback"
        ),
        "publication_grade": False,
        "parameters": vars(args),
        "s4d_source_artifact": "results/s4d_real_pilot_predictions.npz",
        "stmem_checkpoint": stmem_status,
        "datasets": {},
        "phase_a_metrics": [],
        "recalibration_metrics": [],
        "delong_raw_vs_monotone_recalibration": [],
    }

    summary_rows = []
    param_rows = []
    delong_rows = []
    prediction_arrays = {}

    for loader in (_load_ptbxl, _load_code15):
        try:
            ds = loader(args.max_records, logger)
            split = _multilabel_split(ds.patient_ids, ds.labels)
            ds.signals = _standardize(ds.signals, split["train"])
            cal_labels = ds.labels[split["cal"]]
            test_labels = ds.labels[split["test"]]

            out["datasets"][ds.dataset_id] = {
                "dataset_name": ds.dataset_name,
                "n_loaded": int(len(ds.labels)),
                "n_train": int(len(split["train"])),
                "n_cal": int(len(split["cal"])),
                "n_test": int(len(split["test"])),
                "n_classes": int(ds.labels.shape[1]),
                "class_names": ds.class_names,
                "leakage_overlap_counts": split["leakage"],
            }

            model_inputs = []
            prefix = ds.dataset_id
            if f"{prefix}_test_logits" in s4d_arrays:
                model_inputs.append({
                    "model_id": "M6_s4d_configured",
                    "model_name": "Configured S4D constrained pilot",
                    "cal_logits": s4d_arrays[f"{prefix}_cal_logits"],
                    "test_logits": s4d_arrays[f"{prefix}_test_logits"],
                    "test_probs": s4d_arrays[f"{prefix}_test_probs"],
                    "source": "s4d_real_pilot_predictions.npz",
                })
            if stmem_model is not None:
                stmem = _evaluate_stmem_probe(stmem_model, ds, split, logger)
                model_inputs.append({
                    "model_id": "M2_stmem_probe",
                    "model_name": "ST-MEM frozen linear probe expanded real pilot",
                    "cal_logits": stmem["cal_logits"],
                    "test_logits": stmem["test_logits"],
                    "test_probs": stmem["test_probs"],
                    "source": "recomputed_in_monotone_recalibration_pilot",
                    "feature_shape": stmem["feature_shape"],
                })

            for model in model_inputs:
                raw_metrics = evaluate_model_on_dataset(
                    model["model_id"],
                    ds.dataset_id,
                    model["test_probs"],
                    test_labels,
                    ds.class_names,
                )
                out["phase_a_metrics"].append({
                    "dataset_id": ds.dataset_id,
                    "dataset_name": ds.dataset_name,
                    "model_id": model["model_id"],
                    "model_name": model["model_name"],
                    "metrics": raw_metrics["macro"],
                    "per_class": raw_metrics["per_class"],
                    "source": model["source"],
                    "feature_shape": model.get("feature_shape"),
                })
                summary_rows.append(_metric_row(
                    "A_monotone_guard_pilot",
                    ds.dataset_id,
                    ds.dataset_name,
                    model["model_id"],
                    model["model_name"],
                    "none",
                    raw_metrics["macro"],
                ))

                recal = _fit_monotone_platt_with_temperature_fallback(
                    model["cal_logits"],
                    cal_labels,
                    model["test_logits"],
                    test_labels,
                    args.min_cal_pos,
                    args.min_cal_neg,
                    args.min_test_pos,
                    args.min_test_neg,
                    args.min_platt_slope,
                )
                recal_metrics = evaluate_model_on_dataset(
                    f"{model['model_id']}_monotone_guard",
                    ds.dataset_id,
                    recal["probs"],
                    test_labels,
                    ds.class_names,
                )
                out["recalibration_metrics"].append({
                    "dataset_id": ds.dataset_id,
                    "dataset_name": ds.dataset_name,
                    "model_id": model["model_id"],
                    "model_name": model["model_name"],
                    "recal_method": "monotone_platt_or_temperature",
                    "metrics": recal_metrics["macro"],
                    "per_class": recal_metrics["per_class"],
                    "recalibration_params": recal["params"],
                })
                summary_rows.append(_metric_row(
                    "B_monotone_guard_pilot",
                    ds.dataset_id,
                    ds.dataset_name,
                    model["model_id"],
                    model["model_name"],
                    "monotone_platt_or_temperature",
                    recal_metrics["macro"],
                ))

                delong = _delong_raw_vs_recalibrated(
                    test_labels,
                    model["test_probs"],
                    recal["probs"],
                    ds.class_names,
                    recal["params"],
                    args.min_test_pos,
                    args.min_test_neg,
                )
                out["delong_raw_vs_monotone_recalibration"].append({
                    "dataset_id": ds.dataset_id,
                    "dataset_name": ds.dataset_name,
                    "model_id": model["model_id"],
                    "model_name": model["model_name"],
                    "recal_method": "monotone_platt_or_temperature",
                    "per_class": delong,
                    "n_nominal_significant_degradation": sum(
                        1 for d in delong.values()
                        if d.get("nominal_significant_degradation")
                    ),
                })

                for p in recal["params"]:
                    class_name = ds.class_names[int(p["class_index"])]
                    param_rows.append({
                        "dataset_id": ds.dataset_id,
                        "dataset_name": ds.dataset_name,
                        "model_id": model["model_id"],
                        "model_name": model["model_name"],
                        "class_index": p.get("class_index"),
                        "class_name": class_name,
                        "status": p.get("status"),
                        "used_method": p.get("used_method"),
                        "platt_slope": p.get("platt_slope"),
                        "platt_intercept": p.get("platt_intercept"),
                        "temperature": p.get("temperature"),
                        "cal_pos": p.get("cal_pos"),
                        "cal_neg": p.get("cal_neg"),
                        "test_pos": p.get("test_pos"),
                        "test_neg": p.get("test_neg"),
                    })

                for class_name, result in delong.items():
                    delong_rows.append({
                        "dataset_id": ds.dataset_id,
                        "dataset_name": ds.dataset_name,
                        "model_id": model["model_id"],
                        "model_name": model["model_name"],
                        "recal_method": "monotone_platt_or_temperature",
                        "class_name": class_name,
                        "status": result.get("status"),
                        "recal_status": result.get("recal_status"),
                        "used_method": result.get("used_method"),
                        "test_pos": result.get("test_pos"),
                        "test_neg": result.get("test_neg"),
                        "auc_raw": result.get("auc_a"),
                        "auc_recal": result.get("auc_b"),
                        "raw_minus_recal_auc": result.get("raw_minus_recal_auc"),
                        "recal_minus_raw_auc": result.get("recal_minus_raw_auc"),
                        "z_statistic": result.get("z_statistic"),
                        "p_value": result.get("p_value"),
                        "nominal_significant_degradation": result.get(
                            "nominal_significant_degradation"
                        ),
                    })

                key_prefix = f"{ds.dataset_id}_{model['model_id']}"
                prediction_arrays[f"{key_prefix}_test_raw_probs"] = model["test_probs"]
                prediction_arrays[f"{key_prefix}_test_recal_probs"] = recal["probs"]
                prediction_arrays[f"{key_prefix}_test_labels"] = test_labels
                prediction_arrays[f"{key_prefix}_cal_logits"] = model["cal_logits"]
                prediction_arrays[f"{key_prefix}_test_logits"] = model["test_logits"]
        except Exception as exc:
            logger.exception("Monotone recalibration pilot failed for loader %s", loader.__name__)
            out["datasets"][loader.__name__] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    failed = [
        key for key, value in out["datasets"].items()
        if isinstance(value, dict) and value.get("status") == "failed"
    ]
    out["failed_datasets"] = failed
    out["status"] = "completed" if not failed else "completed_partial_with_dataset_blocker"

    json_path = RESULTS / "monotone_recalibration_pilot_results.json"
    summary_path = RESULTS / "monotone_recalibration_pilot_summary.csv"
    params_path = RESULTS / "monotone_recalibration_pilot_params.csv"
    delong_path = RESULTS / "monotone_recalibration_pilot_delong.csv"
    predictions_path = RESULTS / "monotone_recalibration_pilot_predictions.npz"

    json_path.write_text(json.dumps(out, indent=2))
    _write_csv(
        summary_path,
        summary_rows,
        [
            "phase", "dataset_id", "dataset_name", "model_id", "model_name",
            "recal_method", "auroc", "auprc", "ece_ew", "ece_em", "brier", "sce",
        ],
    )
    _write_csv(
        params_path,
        param_rows,
        [
            "dataset_id", "dataset_name", "model_id", "model_name", "class_index",
            "class_name", "status", "used_method", "platt_slope", "platt_intercept",
            "temperature", "cal_pos", "cal_neg", "test_pos", "test_neg",
        ],
    )
    _write_csv(
        delong_path,
        delong_rows,
        [
            "dataset_id", "dataset_name", "model_id", "model_name", "recal_method",
            "class_name", "status", "recal_status", "used_method", "test_pos",
            "test_neg", "auc_raw", "auc_recal", "raw_minus_recal_auc",
            "recal_minus_raw_auc", "z_statistic", "p_value",
            "nominal_significant_degradation",
        ],
    )
    if prediction_arrays:
        np.savez_compressed(predictions_path, **prediction_arrays)

    print(json.dumps({
        "status": out["status"],
        "results": str(json_path),
        "summary": str(summary_path),
        "params": str(params_path),
        "delong": str(delong_path),
        "predictions": str(predictions_path) if prediction_arrays else None,
        "datasets": out["datasets"],
        "n_param_rows": len(param_rows),
        "n_delong_rows": len(delong_rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
