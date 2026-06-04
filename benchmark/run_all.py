"""Main orchestrator for the ECG FM calibration study.

Coordinates the full experimental pipeline:
1. Data preparation (load, preprocess, split)
2. Feature extraction (frozen FMs)
3. Linear probe training (FMs) and S4 training (baseline)
4. Phase A evaluation (calibration audit)
5. Phase B recalibration (intervention)
6. H4 ablation
7. Statistical hypothesis testing
8. Visualization
9. Artifact logging

Usage:
    python run_all.py --data-root /path/to/datasets --checkpoints-root /path/to/checkpoints
    python run_all.py --phase A  # run only Phase A
    python run_all.py --phase B  # run only Phase B
    python run_all.py --phase stats  # run only statistical tests (requires saved results)
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

from config import (
    WORKSPACE,
    RESULTS_DIR,
    FIGURES_DIR,
    DATA_DIR,
    FM_IDS,
    ALL_MODEL_IDS,
    DATASET_IDS,
    RANDOM_SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / "experiment.log"),
    ],
)
logger = logging.getLogger(__name__)

CACHE_DIR = WORKSPACE / "cache"


def _save_cache(name: str, obj):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Cached {name}")


def _load_cache(name: str):
    path = CACHE_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}. Run prior phases first.")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded cached {name}")
    return obj


def parse_args():
    parser = argparse.ArgumentParser(description="ECG FM Calibration Study")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root directory containing raw datasets")
    parser.add_argument("--checkpoints-root", type=Path, required=True,
                        help="Root directory containing FM checkpoints")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "data", "features", "probes", "A", "B", "h4", "stats", "viz"],
                        help="Which phase to run")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Subset of datasets to use (e.g., D1 D3 D4)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Subset of models to use (e.g., M1 M2 M6)")
    parser.add_argument("--skip-mimic", action="store_true",
                        help="Skip MIMIC-IV-ECG (D2) if access unavailable")
    return parser.parse_args()


def step_data(args) -> dict:
    """Step 1: Load, preprocess, and split all datasets."""
    from data_pipeline import prepare_dataset

    datasets = args.datasets or DATASET_IDS
    if args.skip_mimic and "D2" in datasets:
        datasets = [d for d in datasets if d != "D2"]

    prepared = {}
    for dataset_id in datasets:
        try:
            data = prepare_dataset(
                dataset_id,
                args.data_root,
                output_dir=WORKSPACE / "processed" / dataset_id,
            )
            prepared[dataset_id] = data
            logger.info(f"Prepared {dataset_id}: {data['signals'].shape[0]} ECGs")
        except (FileNotFoundError, NotImplementedError) as e:
            logger.warning(f"Skipping {dataset_id}: {e}")

    _save_cache("prepared_data", prepared)
    return prepared


def step_features(args, prepared_data: dict) -> dict:
    """Step 2: Extract frozen FM features for all datasets."""
    from models import load_fm, extract_and_save_features

    model_ids = [m for m in (args.models or FM_IDS) if m != "M6"]
    features = {}

    checkpoint_map = {
        "M1": args.checkpoints_root / "merl" / "merl_checkpoint.pth",
        "M2": args.checkpoints_root / "st_mem" / "st_mem_checkpoint.pth",
        "M3": args.checkpoints_root / "hubert_ecg" / "hubert_ecg_checkpoint.pth",
        "M4": args.checkpoints_root / "ecgfm_ked" / "ecgfm_ked_checkpoint.pth",
        "M5": args.checkpoints_root / "ecg_fm" / "ecg_fm_checkpoint.pth",
    }

    for model_id in model_ids:
        checkpoint = checkpoint_map.get(model_id)
        if checkpoint is None or not checkpoint.exists():
            logger.warning(f"Checkpoint not found for {model_id} at {checkpoint}")
            continue

        fm = load_fm(model_id, checkpoint)
        features[model_id] = {}

        for dataset_id, data in prepared_data.items():
            feats = extract_and_save_features(
                fm, data["signals"], data["split_indices"],
                output_dir=WORKSPACE / "features" / model_id,
                dataset_id=dataset_id,
            )
            features[model_id][dataset_id] = feats

    _save_cache("features", features)
    return features


def step_probes(args, prepared_data: dict, features: dict) -> dict:
    """Step 3: Train linear probes (FMs) and S4 baseline."""
    from linear_probe import run_linear_probe_pipeline
    from s4_baseline import run_s4_pipeline

    predictions = {}

    for dataset_id, data in prepared_data.items():
        predictions[dataset_id] = {}

        for model_id, model_features in features.items():
            if dataset_id not in model_features:
                continue

            result = run_linear_probe_pipeline(
                model_id=model_id,
                dataset_id=dataset_id,
                features=model_features[dataset_id],
                labels=data["labels"],
                split_indices=data["split_indices"],
                n_classes=data["n_classes"],
                output_dir=WORKSPACE / "probes",
            )
            predictions[dataset_id][model_id] = result["predictions"]

        if "M6" in (args.models or ALL_MODEL_IDS):
            s4_result = run_s4_pipeline(
                dataset_id=dataset_id,
                signals=data["signals"],
                labels=data["labels"],
                split_indices=data["split_indices"],
                n_classes=data["n_classes"],
                output_dir=WORKSPACE / "probes",
            )
            predictions[dataset_id]["M6"] = s4_result["predictions"]

    _save_cache("predictions", predictions)
    return predictions


def _build_labels_and_classnames(prepared_data: dict):
    labels_by_dataset = {}
    class_names_by_dataset = {}
    for dataset_id, data in prepared_data.items():
        labels_by_dataset[dataset_id] = {
            split: data["labels"][data["split_indices"][split]]
            for split in ["train", "cal", "test"]
        }
        class_names_by_dataset[dataset_id] = data["class_names"]
    return labels_by_dataset, class_names_by_dataset


def step_phase_a(predictions: dict, prepared_data: dict) -> dict:
    """Step 4: Phase A calibration audit."""
    from run_phase_a import run_phase_a

    labels_by_dataset, class_names_by_dataset = _build_labels_and_classnames(prepared_data)
    result = run_phase_a(predictions, labels_by_dataset, class_names_by_dataset)
    _save_cache("phase_a_results", result)
    return result


def step_phase_b(predictions: dict, prepared_data: dict) -> dict:
    """Step 5: Phase B recalibration intervention."""
    from run_phase_b import run_phase_b

    labels_by_dataset, class_names_by_dataset = _build_labels_and_classnames(prepared_data)
    result = run_phase_b(predictions, labels_by_dataset, class_names_by_dataset)
    _save_cache("phase_b_results", result)
    return result


def step_h4(predictions: dict, prepared_data: dict) -> dict:
    """Step 6: H4 calibration size ablation."""
    from h4_ablation import run_all_h4_ablations

    labels_by_dataset, class_names_by_dataset = _build_labels_and_classnames(prepared_data)
    result = run_all_h4_ablations(
        predictions, labels_by_dataset, class_names_by_dataset,
        output_dir=RESULTS_DIR / "h4_ablation",
    )
    _save_cache("h4_results", result)
    return result


def step_stats(phase_a_results: dict, phase_b_results: dict, h4_results: dict) -> dict:
    """Step 7: Statistical hypothesis testing."""
    from statistical_tests import (
        h1_test_all, h2_sign_analysis, h3_gap_closure,
        h4_method_size_interaction, h5_cross_dataset_consistency,
        h6_auroc_preservation, h3_brier_sensitivity,
    )

    stats = {}
    stats["H1"] = h1_test_all(phase_a_results)
    stats["H2"] = h2_sign_analysis(phase_a_results)
    stats["H3"] = h3_gap_closure(phase_a_results, phase_b_results)
    stats["H3_brier_sensitivity"] = h3_brier_sensitivity(phase_a_results, phase_b_results)
    stats["H4"] = h4_method_size_interaction(h4_results)
    stats["H5"] = h5_cross_dataset_consistency(phase_a_results)
    stats["H6"] = h6_auroc_preservation(phase_a_results, phase_b_results)

    verdicts = {}
    for i in range(1, 7):
        key = f"H{i}"
        data = stats[key]
        verdicts[key] = data.get(f"h{i}_supported", data.get(f"h{i}_overall"))
    verdicts["H3_brier_consistent"] = stats["H3_brier_sensitivity"]["h3_brier_supported"]
    stats["verdict_summary"] = verdicts

    with open(RESULTS_DIR / "hypothesis_test_results.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    logger.info(f"Hypothesis verdicts: {verdicts}")
    _save_cache("stats_results", stats)
    return stats


def step_viz(phase_a_results, phase_b_results, h3_results, h4_results):
    """Step 8: Generate all figures."""
    from visualization import generate_all_figures

    generate_all_figures(
        phase_a_results, phase_b_results, h3_results, h4_results, FIGURES_DIR
    )


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(RANDOM_SEED)

    logger.info("=" * 60)
    logger.info("ECG FM Calibration Study — Starting")
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Checkpoints root: {args.checkpoints_root}")
    logger.info("=" * 60)

    if args.phase == "all":
        prepared = step_data(args)
        features = step_features(args, prepared)
        predictions = step_probes(args, prepared, features)
        phase_a = step_phase_a(predictions, prepared)
        phase_b = step_phase_b(predictions, prepared)
        h4 = step_h4(predictions, prepared)
        stats = step_stats(phase_a, phase_b, h4)
        step_viz(phase_a, phase_b, stats["H3"], h4)
    elif args.phase == "data":
        step_data(args)
    elif args.phase == "features":
        prepared = _load_cache("prepared_data")
        step_features(args, prepared)
    elif args.phase == "probes":
        prepared = _load_cache("prepared_data")
        features = _load_cache("features")
        step_probes(args, prepared, features)
    elif args.phase == "A":
        prepared = _load_cache("prepared_data")
        predictions = _load_cache("predictions")
        step_phase_a(predictions, prepared)
    elif args.phase == "B":
        prepared = _load_cache("prepared_data")
        predictions = _load_cache("predictions")
        step_phase_b(predictions, prepared)
    elif args.phase == "h4":
        prepared = _load_cache("prepared_data")
        predictions = _load_cache("predictions")
        step_h4(predictions, prepared)
    elif args.phase == "stats":
        phase_a = _load_cache("phase_a_results")
        phase_b = _load_cache("phase_b_results")
        h4 = _load_cache("h4_results")
        step_stats(phase_a, phase_b, h4)
    elif args.phase == "viz":
        phase_a = _load_cache("phase_a_results")
        phase_b = _load_cache("phase_b_results")
        stats = _load_cache("stats_results")
        h4 = _load_cache("h4_results")
        step_viz(phase_a, phase_b, stats["H3"], h4)

    logger.info("=" * 60)
    logger.info("Experiment pipeline complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
