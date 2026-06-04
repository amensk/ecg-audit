"""Phase A runner: Calibration audit (H1, H2, H5).

Evaluates all 6 models on all available datasets, producing
AUROC, AUPRC, ECE, Brier, and SCE metrics.
"""

import json
import logging
from pathlib import Path

import numpy as np

from config import (
    ALL_MODEL_IDS,
    FM_IDS,
    DATASET_IDS,
    DATASET_NAMES,
    ALL_MODEL_NAMES,
    RESULTS_DIR,
    FIGURES_DIR,
)
from evaluation import evaluate_model_on_dataset

logger = logging.getLogger(__name__)


def run_phase_a(
    predictions: dict,
    labels_by_dataset: dict,
    class_names_by_dataset: dict,
    output_dir: Path = RESULTS_DIR,
) -> dict:
    """Execute Phase A evaluation across all model-dataset pairs.

    Args:
        predictions: {dataset_id: {model_id: {"test": {"logits": ..., "probs": ...}}}}
        labels_by_dataset: {dataset_id: {"test": labels_array}}
        class_names_by_dataset: {dataset_id: [class_names]}

    Returns:
        phase_a_results: {dataset_id: {model_id: evaluation_dict}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_a_results = {}

    for dataset_id in predictions:
        phase_a_results[dataset_id] = {}

        for model_id in predictions[dataset_id]:
            test_probs = predictions[dataset_id][model_id]["test"]["probs"]
            test_labels = labels_by_dataset[dataset_id]["test"]
            class_names = class_names_by_dataset[dataset_id]

            result = evaluate_model_on_dataset(
                model_id, dataset_id, test_probs, test_labels, class_names
            )

            result["test_probs"] = test_probs
            result["test_labels"] = test_labels

            phase_a_results[dataset_id][model_id] = result

    save_phase_a_results(phase_a_results, output_dir)
    return phase_a_results


def save_phase_a_results(results: dict, output_dir: Path):
    """Save Phase A results to structured JSON and CSV files."""
    summary_rows = []
    full_results = {}

    for dataset_id, dataset_results in results.items():
        full_results[dataset_id] = {}
        for model_id, model_results in dataset_results.items():
            serializable = {
                "model_id": model_id,
                "model_name": ALL_MODEL_NAMES.get(model_id, model_id),
                "dataset_id": dataset_id,
                "dataset_name": DATASET_NAMES.get(dataset_id, dataset_id),
                "macro": model_results["macro"],
                "per_class": {
                    k: {kk: vv for kk, vv in v.items() if kk != "reliability_diagram"}
                    for k, v in model_results.get("per_class", {}).items()
                },
            }
            full_results[dataset_id][model_id] = serializable

            summary_rows.append({
                "model_id": model_id,
                "model_name": ALL_MODEL_NAMES.get(model_id, model_id),
                "dataset_id": dataset_id,
                "dataset_name": DATASET_NAMES.get(dataset_id, dataset_id),
                **{f"macro_{k}": v for k, v in model_results["macro"].items()},
            })

    with open(output_dir / "phase_a_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    import csv

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(output_dir / "phase_a_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    logger.info(f"Phase A results saved: {len(summary_rows)} model-dataset evaluations")
