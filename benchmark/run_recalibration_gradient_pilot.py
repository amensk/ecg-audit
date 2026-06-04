"""Recalibration-method gradient pilot for Stage 05.

This script extends the same 1,000-record PTB-XL/CODE-15% configured S4D and
ST-MEM pilot with the two remaining real recalibrators from the approved study
design: isotonic regression and a learned MLP head. The monotone
Platt-or-temperature outputs from ``run_monotone_recalibration_pilot.py`` are
used as the safety baseline for H3/H4 comparisons.
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
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from evaluation import evaluate_model_on_dataset
from recalibration import MLPCalibrator
from run_minimal_real_pilot import NOTES, RESULTS, SEED
from run_monotone_recalibration_pilot import (
    _delong_raw_vs_recalibrated,
    _metric_row,
    _write_csv,
)
from statistical_tests import compute_brier_gap_closure, compute_gap_closure


DATASET_ORDER = ["D1", "D3"]
MODEL_ORDER = ["M6_s4d_configured", "M2_stmem_probe"]
METHOD_ORDER = [
    "monotone_platt_or_temperature",
    "isotonic_regression",
    "learned_mlp_head",
]


def _setup_logging() -> logging.Logger:
    NOTES.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("recalibration_gradient_pilot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(
        NOTES / "recalibration_gradient_pilot.log", mode="w"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _load_inputs() -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    result_path = RESULTS / "monotone_recalibration_pilot_results.json"
    pred_path = RESULTS / "monotone_recalibration_pilot_predictions.npz"
    s4d_pred_path = RESULTS / "s4d_real_pilot_predictions.npz"
    missing = [
        str(p)
        for p in [result_path, pred_path, s4d_pred_path]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Recalibration gradient pilot requires prior monotone and S4D outputs: "
            + ", ".join(missing)
        )
    return (
        json.loads(result_path.read_text()),
        dict(np.load(pred_path)),
        dict(np.load(s4d_pred_path)),
    )


def _model_metadata(monotone_results: dict) -> dict[tuple[str, str], dict]:
    out = {}
    for row in monotone_results.get("phase_a_metrics", []):
        out[(row["dataset_id"], row["model_id"])] = {
            "dataset_name": row["dataset_name"],
            "model_name": row["model_name"],
            "source": row.get("source"),
            "feature_shape": row.get("feature_shape"),
        }
    return out


def _fit_isotonic_per_class(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    class_names: list[str],
    min_cal_pos: int,
    min_cal_neg: int,
) -> dict:
    calibrated = test_probs.copy().astype(np.float64)
    params = []
    for c, class_name in enumerate(class_names):
        y_cal = cal_labels[:, c]
        y_test = test_labels[:, c]
        cal_pos = int(y_cal.sum())
        cal_neg = int(len(y_cal) - cal_pos)
        row = {
            "class_index": c,
            "class_name": class_name,
            "cal_pos": cal_pos,
            "cal_neg": cal_neg,
            "test_pos": int(y_test.sum()),
            "test_neg": int(len(y_test) - y_test.sum()),
        }
        if cal_pos < min_cal_pos or cal_neg < min_cal_neg:
            row.update({
                "status": "skipped_insufficient_calibration_support",
                "used_method": "raw_identity",
                "n_thresholds": 0,
            })
            calibrated[:, c] = test_probs[:, c]
            params.append(row)
            continue

        try:
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            ir.fit(cal_probs[:, c], y_cal)
            calibrated[:, c] = ir.predict(test_probs[:, c])
            row.update({
                "status": "fit_isotonic",
                "used_method": "isotonic_regression",
                "n_thresholds": int(len(getattr(ir, "X_thresholds_", []))),
                "x_thresholds": getattr(ir, "X_thresholds_", np.array([])).tolist(),
                "y_thresholds": getattr(ir, "y_thresholds_", np.array([])).tolist(),
            })
        except Exception as exc:
            row.update({
                "status": f"failed_isotonic:{type(exc).__name__}",
                "used_method": "raw_identity",
                "error": str(exc),
                "n_thresholds": 0,
            })
            calibrated[:, c] = test_probs[:, c]
        params.append(row)
    return {"probs": calibrated, "params": params}


def _fit_mlp_head(
    cal_logits: np.ndarray,
    cal_labels: np.ndarray,
    test_logits: np.ndarray,
) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    calibrator = MLPCalibrator()
    calibrator.fit(cal_logits.astype(np.float32), cal_labels.astype(np.float32))
    probs = calibrator.transform(test_logits.astype(np.float32)).astype(np.float64)
    return {"probs": probs, "params": calibrator.get_params()}


def _metric_delta(raw_metrics: dict, recal_metrics: dict) -> dict:
    return {
        "delta_auroc": recal_metrics.get("auroc") - raw_metrics.get("auroc"),
        "delta_ece_ew": recal_metrics.get("ece_equal_width")
        - raw_metrics.get("ece_equal_width"),
        "delta_ece_em": recal_metrics.get("ece_equal_mass")
        - raw_metrics.get("ece_equal_mass"),
        "delta_brier": recal_metrics.get("brier") - raw_metrics.get("brier"),
        "delta_sce": recal_metrics.get("sce") - raw_metrics.get("sce"),
        "relative_ece_reduction": (
            (raw_metrics.get("ece_equal_width") - recal_metrics.get("ece_equal_width"))
            / raw_metrics.get("ece_equal_width")
            if raw_metrics.get("ece_equal_width")
            else None
        ),
        "relative_brier_reduction": (
            (raw_metrics.get("brier") - recal_metrics.get("brier"))
            / raw_metrics.get("brier")
            if raw_metrics.get("brier")
            else None
        ),
    }


def _rank_rows(rows: list[dict]) -> list[dict]:
    ranked = []
    groups = {}
    for row in rows:
        if row["recal_method"] == "none":
            continue
        key = (row["dataset_id"], row["model_id"])
        groups.setdefault(key, []).append(row)

    for (dataset_id, model_id), group in groups.items():
        ece_sorted = sorted(group, key=lambda r: (np.inf if r["ece_ew"] is None else r["ece_ew"]))
        brier_sorted = sorted(group, key=lambda r: (np.inf if r["brier"] is None else r["brier"]))
        ece_rank = {id(row): rank + 1 for rank, row in enumerate(ece_sorted)}
        brier_rank = {id(row): rank + 1 for rank, row in enumerate(brier_sorted)}
        for row in group:
            ranked.append({
                "dataset_id": dataset_id,
                "dataset_name": row["dataset_name"],
                "model_id": model_id,
                "model_name": row["model_name"],
                "recal_method": row["recal_method"],
                "ece_ew": row["ece_ew"],
                "brier": row["brier"],
                "auroc": row["auroc"],
                "ece_rank": ece_rank[id(row)],
                "brier_rank": brier_rank[id(row)],
                "best_by_ece": ece_rank[id(row)] == 1,
                "best_by_brier": brier_rank[id(row)] == 1,
            })
    return ranked


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-cal-pos", type=int, default=10)
    parser.add_argument("--min-cal-neg", type=int, default=10)
    parser.add_argument("--min-test-pos", type=int, default=10)
    parser.add_argument("--min-test-neg", type=int, default=10)
    args = parser.parse_args()

    logger = _setup_logging()
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    monotone_results, pred, s4d_pred = _load_inputs()
    metadata = _model_metadata(monotone_results)

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "05_experimentation",
        "experiment": "recalibration_gradient_same_split_pilot",
        "scope": (
            "Same 1000-record PTB-XL + CODE-15% pilot; configured S4D and "
            "ST-MEM; compares monotone Platt/temperature safety baseline, "
            "isotonic regression, and learned MLP head."
        ),
        "publication_grade": False,
        "parameters": vars(args),
        "source_artifacts": {
            "monotone_results": "results/monotone_recalibration_pilot_results.json",
            "monotone_predictions": "results/monotone_recalibration_pilot_predictions.npz",
            "s4d_predictions": "results/s4d_real_pilot_predictions.npz",
        },
        "datasets": monotone_results.get("datasets", {}),
        "raw_metrics": [],
        "recalibration_metrics": [],
        "delong_raw_vs_recalibration": [],
        "h3_gap_closure": [],
    }

    summary_rows = []
    ranking_input_rows = []
    ranking_rows = []
    delong_rows = []
    param_rows = []
    prediction_arrays = {}
    model_raw_metrics = {}
    model_recal_metrics = {}

    for dataset_id in DATASET_ORDER:
        ds_info = monotone_results["datasets"][dataset_id]
        dataset_name = ds_info["dataset_name"]
        class_names = ds_info["class_names"]
        cal_labels = s4d_pred[f"{dataset_id}_cal_labels"].astype(np.float64)

        for model_id in MODEL_ORDER:
            key_prefix = f"{dataset_id}_{model_id}"
            meta = metadata[(dataset_id, model_id)]
            model_name = meta["model_name"]

            cal_logits = pred[f"{key_prefix}_cal_logits"].astype(np.float64)
            cal_probs = expit(cal_logits)
            test_logits = pred[f"{key_prefix}_test_logits"].astype(np.float64)
            test_probs = pred[f"{key_prefix}_test_raw_probs"].astype(np.float64)
            test_labels = pred[f"{key_prefix}_test_labels"].astype(np.float64)
            monotone_probs = pred[f"{key_prefix}_test_recal_probs"].astype(np.float64)

            raw_eval = evaluate_model_on_dataset(
                model_id, dataset_id, test_probs, test_labels, class_names
            )
            raw_metrics = raw_eval["macro"]
            model_raw_metrics[(dataset_id, model_id)] = raw_metrics
            out["raw_metrics"].append({
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "model_id": model_id,
                "model_name": model_name,
                "metrics": raw_metrics,
                "per_class": raw_eval["per_class"],
                "source": meta.get("source"),
                "feature_shape": meta.get("feature_shape"),
            })
            raw_row = _metric_row(
                "A_recalibration_gradient_pilot",
                dataset_id,
                dataset_name,
                model_id,
                model_name,
                "none",
                raw_metrics,
            )
            summary_rows.append(raw_row)

            method_outputs = {
                "monotone_platt_or_temperature": {
                    "probs": monotone_probs,
                    "params": {"source": "monotone_recalibration_pilot_predictions.npz"},
                }
            }
            method_outputs["isotonic_regression"] = _fit_isotonic_per_class(
                cal_probs,
                cal_labels,
                test_probs,
                test_labels,
                class_names,
                args.min_cal_pos,
                args.min_cal_neg,
            )
            try:
                method_outputs["learned_mlp_head"] = _fit_mlp_head(
                    cal_logits,
                    cal_labels,
                    test_logits,
                )
            except Exception as exc:
                logger.exception(
                    "MLP recalibration failed for %s/%s; using raw identity.",
                    dataset_id,
                    model_id,
                )
                method_outputs["learned_mlp_head"] = {
                    "probs": test_probs.copy(),
                    "params": {
                        "status": f"failed_mlp:{type(exc).__name__}",
                        "error": str(exc),
                    },
                }

            for method_name in METHOD_ORDER:
                method = method_outputs[method_name]
                probs = np.clip(method["probs"], 0.0, 1.0)
                recal_eval = evaluate_model_on_dataset(
                    f"{model_id}_{method_name}",
                    dataset_id,
                    probs,
                    test_labels,
                    class_names,
                )
                recal_metrics = recal_eval["macro"]
                deltas = _metric_delta(raw_metrics, recal_metrics)
                model_recal_metrics[(dataset_id, model_id, method_name)] = recal_metrics

                out["recalibration_metrics"].append({
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "model_id": model_id,
                    "model_name": model_name,
                    "recal_method": method_name,
                    "metrics": recal_metrics,
                    "metric_delta_vs_raw": deltas,
                    "per_class": recal_eval["per_class"],
                    "params": method.get("params"),
                })
                row = _metric_row(
                    "B_recalibration_gradient_pilot",
                    dataset_id,
                    dataset_name,
                    model_id,
                    model_name,
                    method_name,
                    recal_metrics,
                )
                row.update(deltas)
                summary_rows.append(row)
                ranking_input_rows.append(row)

                delong = _delong_raw_vs_recalibrated(
                    test_labels,
                    test_probs,
                    probs,
                    class_names,
                    [
                        {
                            "class_index": i,
                            "status": method_name,
                            "used_method": method_name,
                        }
                        for i in range(len(class_names))
                    ],
                    args.min_test_pos,
                    args.min_test_neg,
                )
                out["delong_raw_vs_recalibration"].append({
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "model_id": model_id,
                    "model_name": model_name,
                    "recal_method": method_name,
                    "per_class": delong,
                    "n_nominal_significant_degradation": sum(
                        1 for d in delong.values()
                        if d.get("nominal_significant_degradation")
                    ),
                })
                for class_name, result in delong.items():
                    delong_rows.append({
                        "dataset_id": dataset_id,
                        "dataset_name": dataset_name,
                        "model_id": model_id,
                        "model_name": model_name,
                        "recal_method": method_name,
                        "class_name": class_name,
                        "status": result.get("status"),
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

                params = method.get("params")
                if method_name == "isotonic_regression":
                    for p in params:
                        param_rows.append({
                            "dataset_id": dataset_id,
                            "dataset_name": dataset_name,
                            "model_id": model_id,
                            "model_name": model_name,
                            "recal_method": method_name,
                            "class_index": p.get("class_index"),
                            "class_name": p.get("class_name"),
                            "status": p.get("status"),
                            "used_method": p.get("used_method"),
                            "cal_pos": p.get("cal_pos"),
                            "cal_neg": p.get("cal_neg"),
                            "test_pos": p.get("test_pos"),
                            "test_neg": p.get("test_neg"),
                            "n_thresholds": p.get("n_thresholds"),
                            "n_parameters": None,
                        })
                elif method_name == "learned_mlp_head":
                    param_rows.append({
                        "dataset_id": dataset_id,
                        "dataset_name": dataset_name,
                        "model_id": model_id,
                        "model_name": model_name,
                        "recal_method": method_name,
                        "class_index": None,
                        "class_name": "__all__",
                        "status": params.get("status", "fit_mlp"),
                        "used_method": method_name,
                        "cal_pos": None,
                        "cal_neg": None,
                        "test_pos": None,
                        "test_neg": None,
                        "n_thresholds": None,
                        "n_parameters": params.get("n_parameters"),
                    })

                prediction_arrays[f"{key_prefix}_{method_name}_test_probs"] = probs

            prediction_arrays[f"{key_prefix}_test_raw_probs"] = test_probs
            prediction_arrays[f"{key_prefix}_test_logits"] = test_logits
            prediction_arrays[f"{key_prefix}_test_labels"] = test_labels
            prediction_arrays[f"{key_prefix}_cal_logits"] = cal_logits

    ranking_rows = _rank_rows(ranking_input_rows)

    for dataset_id in DATASET_ORDER:
        s4_raw = model_raw_metrics[(dataset_id, "M6_s4d_configured")]
        fm_raw = model_raw_metrics[(dataset_id, "M2_stmem_probe")]
        for method_name in METHOD_ORDER:
            fm_post = model_recal_metrics[(dataset_id, "M2_stmem_probe", method_name)]
            ece_closure = compute_gap_closure(
                fm_raw["ece_equal_width"],
                fm_post["ece_equal_width"],
                s4_raw["ece_equal_width"],
            )
            brier_closure = compute_brier_gap_closure(
                fm_raw["brier"],
                fm_post["brier"],
                s4_raw["brier"],
            )
            row = {
                "dataset_id": dataset_id,
                "dataset_name": monotone_results["datasets"][dataset_id]["dataset_name"],
                "fm_model_id": "M2_stmem_probe",
                "s4_model_id": "M6_s4d_configured",
                "recal_method": method_name,
                "fm_ece_pre": fm_raw["ece_equal_width"],
                "fm_ece_post": fm_post["ece_equal_width"],
                "s4_ece": s4_raw["ece_equal_width"],
                "ece_gap_closure": ece_closure,
                "fm_brier_pre": fm_raw["brier"],
                "fm_brier_post": fm_post["brier"],
                "s4_brier": s4_raw["brier"],
                "brier_gap_closure": brier_closure,
                "excluded_ece_gap_closure": ece_closure is None,
                "excluded_brier_gap_closure": brier_closure is None,
            }
            out["h3_gap_closure"].append(row)

    out["method_rankings"] = ranking_rows
    out["status"] = "completed"

    json_path = RESULTS / "recalibration_gradient_pilot_results.json"
    summary_path = RESULTS / "recalibration_gradient_pilot_summary.csv"
    ranking_path = RESULTS / "recalibration_gradient_pilot_method_ranking.csv"
    delong_path = RESULTS / "recalibration_gradient_pilot_delong.csv"
    params_path = RESULTS / "recalibration_gradient_pilot_params.csv"
    h3_path = RESULTS / "recalibration_gradient_pilot_h3_gap_closure.csv"
    predictions_path = RESULTS / "recalibration_gradient_pilot_predictions.npz"

    _write_json(json_path, out)
    _write_csv(
        summary_path,
        summary_rows,
        [
            "phase", "dataset_id", "dataset_name", "model_id", "model_name",
            "recal_method", "auroc", "auprc", "ece_ew", "ece_em", "brier", "sce",
            "delta_auroc", "delta_ece_ew", "delta_ece_em", "delta_brier",
            "delta_sce", "relative_ece_reduction", "relative_brier_reduction",
        ],
    )
    _write_csv(
        ranking_path,
        ranking_rows,
        [
            "dataset_id", "dataset_name", "model_id", "model_name", "recal_method",
            "ece_ew", "brier", "auroc", "ece_rank", "brier_rank",
            "best_by_ece", "best_by_brier",
        ],
    )
    _write_csv(
        delong_path,
        delong_rows,
        [
            "dataset_id", "dataset_name", "model_id", "model_name", "recal_method",
            "class_name", "status", "test_pos", "test_neg", "auc_raw", "auc_recal",
            "raw_minus_recal_auc", "recal_minus_raw_auc", "z_statistic",
            "p_value", "nominal_significant_degradation",
        ],
    )
    _write_csv(
        params_path,
        param_rows,
        [
            "dataset_id", "dataset_name", "model_id", "model_name", "recal_method",
            "class_index", "class_name", "status", "used_method", "cal_pos",
            "cal_neg", "test_pos", "test_neg", "n_thresholds", "n_parameters",
        ],
    )
    _write_csv(
        h3_path,
        out["h3_gap_closure"],
        [
            "dataset_id", "dataset_name", "fm_model_id", "s4_model_id",
            "recal_method", "fm_ece_pre", "fm_ece_post", "s4_ece",
            "ece_gap_closure", "fm_brier_pre", "fm_brier_post", "s4_brier",
            "brier_gap_closure", "excluded_ece_gap_closure",
            "excluded_brier_gap_closure",
        ],
    )
    np.savez_compressed(predictions_path, **prediction_arrays)

    print(json.dumps({
        "status": out["status"],
        "results": str(json_path),
        "summary": str(summary_path),
        "rankings": str(ranking_path),
        "delong": str(delong_path),
        "params": str(params_path),
        "h3_gap_closure": str(h3_path),
        "predictions": str(predictions_path),
        "n_summary_rows": len(summary_rows),
        "n_ranking_rows": len(ranking_rows),
        "n_delong_rows": len(delong_rows),
        "n_param_rows": len(param_rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
