"""Phase B runner: Recalibration intervention (H3, H4, H6).

Applies all 4 recalibration methods to all 5 FMs on all datasets,
then evaluates calibrated predictions.
"""

import json
import logging
from pathlib import Path

import numpy as np

from config import (
    FM_IDS,
    RECAL_IDS,
    RECAL_NAMES,
    FM_NAMES,
    DATASET_NAMES,
    RESULTS_DIR,
)
from recalibration import run_all_recalibrations
from evaluation import evaluate_multilabel

logger = logging.getLogger(__name__)


def run_phase_b(
    predictions: dict,
    labels_by_dataset: dict,
    class_names_by_dataset: dict,
    output_dir: Path = RESULTS_DIR,
) -> dict:
    """Execute Phase B recalibration for all FM-dataset-method combinations.

    Args:
        predictions: {dataset_id: {model_id: {"cal": {...}, "test": {...}}}}
        labels_by_dataset: {dataset_id: {"cal": ..., "test": ...}}
        class_names_by_dataset: {dataset_id: [class_names]}

    Returns:
        phase_b_results: {dataset_id: {fm_id: {method_id: evaluation_dict}}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    recal_dir = output_dir / "recalibration"
    recal_dir.mkdir(exist_ok=True)

    phase_b_results = {}

    for dataset_id in predictions:
        phase_b_results[dataset_id] = {}

        for fm_id in FM_IDS:
            if fm_id not in predictions[dataset_id]:
                continue

            preds = predictions[dataset_id][fm_id]
            cal_logits = preds["cal"]["logits"]
            cal_probs = preds["cal"]["probs"]
            cal_labels = labels_by_dataset[dataset_id]["cal"]
            test_logits = preds["test"]["logits"]
            test_probs = preds["test"]["probs"]
            test_labels = labels_by_dataset[dataset_id]["test"]
            class_names = class_names_by_dataset[dataset_id]

            recal_results = run_all_recalibrations(
                model_id=fm_id,
                dataset_id=dataset_id,
                cal_logits=cal_logits,
                cal_probs=cal_probs,
                cal_labels=cal_labels,
                test_logits=test_logits,
                test_probs=test_probs,
                output_dir=recal_dir,
            )

            phase_b_results[dataset_id][fm_id] = {}
            for method_id, recal_result in recal_results.items():
                eval_result = evaluate_multilabel(
                    recal_result["calibrated_probs"], test_labels, class_names
                )
                eval_result["method_id"] = method_id
                eval_result["method_name"] = recal_result["method_name"]
                eval_result["calibrated_probs"] = recal_result["calibrated_probs"]
                phase_b_results[dataset_id][fm_id][method_id] = eval_result

    save_phase_b_results(phase_b_results, output_dir)
    return phase_b_results


def save_phase_b_results(results: dict, output_dir: Path):
    """Save Phase B results to structured JSON and CSV."""
    summary_rows = []
    full_results = {}

    for dataset_id, ds_results in results.items():
        full_results[dataset_id] = {}
        for fm_id, fm_results in ds_results.items():
            full_results[dataset_id][fm_id] = {}
            for method_id, method_results in fm_results.items():
                serializable = {
                    "method_id": method_id,
                    "method_name": RECAL_NAMES.get(method_id, method_id),
                    "macro": method_results["macro"],
                    "per_class": method_results.get("per_class", {}),
                }
                full_results[dataset_id][fm_id][method_id] = serializable

                summary_rows.append({
                    "fm_id": fm_id,
                    "fm_name": FM_NAMES.get(fm_id, fm_id),
                    "dataset_id": dataset_id,
                    "dataset_name": DATASET_NAMES.get(dataset_id, dataset_id),
                    "method_id": method_id,
                    "method_name": RECAL_NAMES.get(method_id, method_id),
                    **{f"macro_{k}": v for k, v in method_results["macro"].items()},
                })

    with open(output_dir / "phase_b_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    import csv

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(output_dir / "phase_b_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    logger.info(
        f"Phase B results saved: {len(summary_rows)} FM-dataset-method evaluations"
    )
