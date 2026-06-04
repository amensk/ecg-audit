"""H4 calibration size ablation infrastructure.

Stratified subsampling of calibration sets at {500, 1K, 2K, 5K, 10K}
with 5 repetitions per size. Runs all 4 recalibration methods per subsample.
"""

import logging
from pathlib import Path

import numpy as np

from config import (
    H4_ABLATION_SIZES,
    H4_ABLATION_REPS,
    H4_PRIMARY_DATASETS,
    FM_IDS,
    RECAL_IDS,
    RANDOM_SEED,
)
from recalibration import apply_recalibration
from evaluation import evaluate_multilabel

logger = logging.getLogger(__name__)


def stratified_subsample(
    indices: np.ndarray,
    labels: np.ndarray,
    target_size: int,
    seed: int,
) -> np.ndarray:
    """Stratified subsample preserving label distribution."""
    rng = np.random.RandomState(seed)

    if target_size >= len(indices):
        return indices.copy()

    if labels.ndim > 1:
        label_keys = ["".join(map(str, row.astype(int))) for row in labels]
    else:
        label_keys = labels.tolist()

    from collections import Counter

    label_counts = Counter(label_keys)
    label_to_indices = {}
    for i, lk in enumerate(label_keys):
        label_to_indices.setdefault(lk, []).append(i)

    selected = []
    remaining_budget = target_size

    for lk, count in sorted(label_counts.items(), key=lambda x: x[1]):
        n_select = max(1, int(round(count / len(indices) * target_size)))
        n_select = min(n_select, count, remaining_budget)
        pool = label_to_indices[lk]
        chosen = rng.choice(pool, size=n_select, replace=False)
        selected.extend(chosen)
        remaining_budget -= n_select

    if remaining_budget > 0:
        all_chosen = set(selected)
        remaining = [i for i in range(len(indices)) if i not in all_chosen]
        if remaining:
            extra = rng.choice(remaining, size=min(remaining_budget, len(remaining)), replace=False)
            selected.extend(extra)

    return indices[np.array(selected[:target_size])]


def run_h4_ablation(
    fm_id: str,
    dataset_id: str,
    cal_logits: np.ndarray,
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_logits: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    class_names: list,
) -> dict:
    """Run the full H4 ablation for one FM-dataset pair.

    Returns results keyed by calibration size, then method, then repetition.
    """
    results = {}
    cal_indices = np.arange(len(cal_logits))

    for size in H4_ABLATION_SIZES:
        if size > len(cal_indices):
            logger.info(
                f"Skipping size {size} for {fm_id}/{dataset_id}: "
                f"cal set only has {len(cal_indices)} samples"
            )
            continue

        size_results = {}
        for method_id in RECAL_IDS:
            rep_eces = []
            for rep in range(H4_ABLATION_REPS):
                seed = RANDOM_SEED + rep * 1000 + size
                sub_idx = stratified_subsample(cal_indices, cal_labels, size, seed)

                sub_logits = cal_logits[sub_idx]
                sub_probs = cal_probs[sub_idx]
                sub_labels = cal_labels[sub_idx]

                recal_result = apply_recalibration(
                    method_id, sub_logits, sub_probs, sub_labels,
                    test_logits, test_probs,
                )

                eval_result = evaluate_multilabel(
                    recal_result["calibrated_probs"], test_labels, class_names
                )
                rep_eces.append(eval_result["macro"]["ece_equal_width"])

            size_results[method_id] = {
                "ece_values": rep_eces,
                "mean_ece": float(np.mean(rep_eces)),
                "std_ece": float(np.std(rep_eces)),
            }

        results[str(size)] = size_results
        logger.info(f"H4 ablation {fm_id}/{dataset_id} size={size}: done")

    full_results = {}
    for method_id in RECAL_IDS:
        recal_result = apply_recalibration(
            method_id, cal_logits, cal_probs, cal_labels, test_logits, test_probs,
        )
        eval_result = evaluate_multilabel(
            recal_result["calibrated_probs"], test_labels, class_names
        )
        full_results[method_id] = {
            "ece_values": [eval_result["macro"]["ece_equal_width"]],
            "mean_ece": eval_result["macro"]["ece_equal_width"],
            "std_ece": 0.0,
        }
    results["full"] = full_results

    return results


def run_all_h4_ablations(
    predictions: dict,
    labels_by_dataset: dict,
    class_names_by_dataset: dict,
    output_dir: Path,
) -> dict:
    """Run H4 ablation for all FM-dataset pairs on primary datasets."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for dataset_id in H4_PRIMARY_DATASETS:
        if dataset_id not in predictions:
            continue
        for fm_id in FM_IDS:
            if fm_id not in predictions[dataset_id]:
                continue

            preds = predictions[dataset_id][fm_id]
            result = run_h4_ablation(
                fm_id=fm_id,
                dataset_id=dataset_id,
                cal_logits=preds["cal"]["logits"],
                cal_probs=preds["cal"]["probs"],
                cal_labels=labels_by_dataset[dataset_id]["cal"],
                test_logits=preds["test"]["logits"],
                test_probs=preds["test"]["probs"],
                test_labels=labels_by_dataset[dataset_id]["test"],
                class_names=class_names_by_dataset[dataset_id],
            )
            all_results[f"{fm_id}_{dataset_id}"] = result

    with open(output_dir / "h4_ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results
