"""Configured S4D real-data pilot for Stage 05 refinement.

Runs the same 1,000-record PTB-XL + CODE-15% pilot as
``run_expanded_real_pilot.py`` but replaces the reduced S4-lite comparator
with the configured S4D architecture from ``s4_baseline.py``. The search is
intentionally constrained for local feasibility while preserving the approved
S4D model shape.
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

from config import S4Hyperparams
from evaluation import evaluate_model_on_dataset
from run_expanded_real_pilot import (
    _delong_raw_vs_platt,
    _fit_guarded_platt,
)
from run_minimal_real_pilot import (
    NOTES,
    RESULTS,
    SEED,
    _load_code15,
    _load_ptbxl,
    _multilabel_split,
    _standardize,
)
from s4_baseline import get_s4_predictions, train_s4_single


def _setup_logging() -> logging.Logger:
    NOTES.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("s4d_real_pilot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(NOTES / "s4d_real_pilot.log", mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)

    s4_logger = logging.getLogger("s4_baseline")
    s4_logger.setLevel(logging.INFO)
    s4_logger.handlers.clear()
    s4_logger.addHandler(stream)
    s4_logger.addHandler(file_handler)
    return logger


def _train_s4d_constrained(
    ds,
    split: dict[str, np.ndarray],
    learning_rates: list[float],
    weight_decays: list[float],
    max_epochs: int,
    patience: int,
    batch_size: int,
    logger: logging.Logger,
) -> dict:
    hp = S4Hyperparams(
        learning_rates=learning_rates,
        weight_decays=weight_decays,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
    )
    train_sigs = ds.signals[split["train"]]
    train_lbls = ds.labels[split["train"]]
    cal_sigs = ds.signals[split["cal"]]
    cal_lbls = ds.labels[split["cal"]]

    config_results = []
    best = None
    for lr in learning_rates:
        for wd in weight_decays:
            logger.info(
                "%s configured S4D search config lr=%s wd=%s epochs=%s patience=%s batch=%s",
                ds.dataset_id,
                lr,
                wd,
                max_epochs,
                patience,
                batch_size,
            )
            result = train_s4_single(
                train_sigs,
                train_lbls,
                cal_sigs,
                cal_lbls,
                ds.labels.shape[1],
                lr,
                wd,
                hp,
            )
            config_row = {
                "lr": lr,
                "weight_decay": wd,
                "best_cal_auroc": float(result["best_auroc"]),
                "max_epochs": max_epochs,
                "patience": patience,
                "batch_size": batch_size,
            }
            config_results.append(config_row)
            logger.info(
                "%s configured S4D result lr=%s wd=%s best_cal_auroc=%.4f",
                ds.dataset_id,
                lr,
                wd,
                result["best_auroc"],
            )
            if best is None or result["best_auroc"] > best["best_auroc"]:
                best = result

    assert best is not None
    cal_preds = get_s4_predictions(best["model"], cal_sigs, batch_size=batch_size)
    test_preds = get_s4_predictions(
        best["model"], ds.signals[split["test"]], batch_size=batch_size
    )
    raw_metrics = evaluate_model_on_dataset(
        "M6_s4d_configured",
        ds.dataset_id,
        test_preds["probs"],
        ds.labels[split["test"]],
        ds.class_names,
    )
    return {
        "model": best["model"],
        "cal_logits": cal_preds["logits"],
        "cal_probs": cal_preds["probs"],
        "test_logits": test_preds["logits"],
        "test_probs": test_preds["probs"],
        "raw_metrics": raw_metrics,
        "search_results": config_results,
        "selected_config": {
            "lr": float(best["lr"]),
            "weight_decay": float(best["wd"]),
            "best_cal_auroc": float(best["best_auroc"]),
            "architecture": {
                "class": "S4DModel",
                "n_leads": 12,
                "d_model": 128,
                "d_state": 64,
                "n_layers": 4,
                "dropout": 0.1,
                "n_classes": int(ds.labels.shape[1]),
            },
        },
    }


def _flatten_summary_rows(out: dict) -> list[dict]:
    rows = []
    for row in out["phase_a_metrics"]:
        m = row["metrics"]
        rows.append({
            "phase": "A_s4d_real_pilot",
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
            "phase": "B_s4d_real_pilot",
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


def _load_expanded_stmem_rows() -> list[dict]:
    path = RESULTS / "expanded_real_pilot_summary.csv"
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("model_id") == "M2_stmem_probe"]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[3e-4, 1e-3])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[1e-4])
    parser.add_argument("--max-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
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
        "experiment": "configured_s4d_real_data_pilot",
        "scope": (
            "PTB-XL + CODE-15%, same 1000-record pilot, configured S4D "
            "architecture with constrained hyperparameter search, guarded Platt "
            "recalibration, per-class DeLong diagnostics"
        ),
        "publication_grade": False,
        "parameters": vars(args),
        "limitations": [
            "Pilot-scale subset, not full benchmark.",
            "Configured S4D architecture is used, but search is constrained to keep the local run feasible.",
            "Signals use the same 2250-sample crop/pad geometry as the ST-MEM pilot for a same-split comparison.",
            "Only guarded Platt scaling is tested in this refinement pass.",
            "Per-class DeLong tests are diagnostic and not Bonferroni-powered for final H6 claims.",
        ],
        "datasets": {},
        "phase_a_metrics": [],
        "recalibration_metrics": [],
        "delong_raw_vs_guarded_platt": [],
        "prediction_archive_keys": [],
    }

    prediction_arrays = {}
    for loader in (_load_ptbxl, _load_code15):
        try:
            ds = loader(args.max_records, logger)
            split = _multilabel_split(ds.patient_ids, ds.labels)
            ds.signals = _standardize(ds.signals, split["train"])
            test_labels = ds.labels[split["test"]]
            cal_labels = ds.labels[split["cal"]]

            out["datasets"][ds.dataset_id] = {
                "dataset_name": ds.dataset_name,
                "n_loaded": int(len(ds.labels)),
                "n_train": int(len(split["train"])),
                "n_cal": int(len(split["cal"])),
                "n_test": int(len(split["test"])),
                "n_classes": int(ds.labels.shape[1]),
                "class_names": ds.class_names,
                "label_prevalence": {
                    name: float(ds.labels[:, i].mean())
                    for i, name in enumerate(ds.class_names)
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

            s4d = _train_s4d_constrained(
                ds,
                split,
                args.learning_rates,
                args.weight_decays,
                args.max_epochs,
                args.patience,
                args.batch_size,
                logger,
            )
            out["phase_a_metrics"].append({
                "dataset_id": ds.dataset_id,
                "dataset_name": ds.dataset_name,
                "model_id": "M6_s4d_configured",
                "model_name": "Configured S4D constrained pilot",
                "metrics": s4d["raw_metrics"]["macro"],
                "per_class": s4d["raw_metrics"]["per_class"],
                "search_results": s4d["search_results"],
                "selected_config": s4d["selected_config"],
            })

            platt = _fit_guarded_platt(
                s4d["cal_logits"],
                cal_labels,
                s4d["test_logits"],
                test_labels,
                args.min_cal_pos,
                args.min_cal_neg,
                args.min_test_pos,
                args.min_test_neg,
            )
            platt_metrics = evaluate_model_on_dataset(
                "M6_s4d_configured_guarded_platt",
                ds.dataset_id,
                platt["probs"],
                test_labels,
                ds.class_names,
            )
            out["recalibration_metrics"].append({
                "dataset_id": ds.dataset_id,
                "dataset_name": ds.dataset_name,
                "model_id": "M6_s4d_configured",
                "model_name": "Configured S4D constrained pilot",
                "recal_method": "guarded_platt_scaling",
                "metrics": platt_metrics["macro"],
                "per_class": platt_metrics["per_class"],
                "platt_params": platt["params"],
            })
            delong = _delong_raw_vs_platt(
                test_labels,
                s4d["test_probs"],
                platt["probs"],
                ds.class_names,
                platt["params"],
                args.min_test_pos,
                args.min_test_neg,
            )
            out["delong_raw_vs_guarded_platt"].append({
                "dataset_id": ds.dataset_id,
                "dataset_name": ds.dataset_name,
                "model_id": "M6_s4d_configured",
                "model_name": "Configured S4D constrained pilot",
                "recal_method": "guarded_platt_scaling",
                "per_class": delong,
                "n_nominal_significant_degradation": sum(
                    1 for d in delong.values()
                    if d.get("nominal_significant_degradation")
                ),
            })

            prefix = ds.dataset_id
            arrays = {
                f"{prefix}_cal_logits": s4d["cal_logits"],
                f"{prefix}_cal_probs": s4d["cal_probs"],
                f"{prefix}_cal_labels": cal_labels,
                f"{prefix}_test_logits": s4d["test_logits"],
                f"{prefix}_test_probs": s4d["test_probs"],
                f"{prefix}_test_platt_probs": platt["probs"],
                f"{prefix}_test_labels": test_labels,
            }
            prediction_arrays.update(arrays)
            out["prediction_archive_keys"].extend(sorted(arrays))
        except Exception as exc:
            logger.exception("Configured S4D pilot failed for loader %s", loader.__name__)
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

    json_path = RESULTS / "s4d_real_pilot_results.json"
    summary_path = RESULTS / "s4d_real_pilot_summary.csv"
    delong_path = RESULTS / "s4d_real_pilot_delong.csv"
    comparison_path = RESULTS / "s4d_vs_stmem_real_pilot_comparison.csv"
    predictions_path = RESULTS / "s4d_real_pilot_predictions.npz"

    json_path.write_text(json.dumps(out, indent=2))
    summary_rows = _flatten_summary_rows(out)
    _write_csv(
        summary_path,
        summary_rows,
        [
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
    delong_rows = _flatten_delong_rows(out)
    _write_csv(
        delong_path,
        delong_rows,
        [
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

    stmem_rows = _load_expanded_stmem_rows()
    comparison_rows = []
    for row in summary_rows:
        comparison = dict(row)
        comparison["source_artifact"] = "s4d_real_pilot_summary.csv"
        comparison_rows.append(comparison)
    for row in stmem_rows:
        comparison = {
            "phase": row["phase"],
            "dataset_id": row["dataset_id"],
            "dataset_name": row["dataset_name"],
            "model_id": row["model_id"],
            "model_name": row["model_name"],
            "recal_method": row["recal_method"],
            "auroc": row["auroc"],
            "auprc": row["auprc"],
            "ece_ew": row["ece_ew"],
            "ece_em": row["ece_em"],
            "brier": row["brier"],
            "sce": row["sce"],
            "source_artifact": "expanded_real_pilot_summary.csv",
        }
        comparison_rows.append(comparison)
    _write_csv(
        comparison_path,
        comparison_rows,
        [
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
            "source_artifact",
        ],
    )
    if prediction_arrays:
        np.savez_compressed(predictions_path, **prediction_arrays)

    print(json.dumps({
        "status": out["status"],
        "results": str(json_path),
        "summary": str(summary_path),
        "delong_summary": str(delong_path),
        "comparison": str(comparison_path),
        "predictions": str(predictions_path) if prediction_arrays else None,
        "datasets": out["datasets"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
