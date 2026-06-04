"""Smoke test for verifying FM wrappers, data pipeline, and recalibration.

Runs with synthetic data to catch import path and shape mismatches before
committing to full dataset runs. Can also verify individual FM checkpoints.

Usage:
    python smoke_test.py                          # test all non-FM components
    python smoke_test.py --checkpoint-dir /path    # also test FM loading
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

from config import (
    RANDOM_SEED, N_LEADS, SIGNAL_LENGTH, ECE_N_BINS,
    FM_IDS, FM_NAMES, RECAL_IDS, RECAL_NAMES,
    S4Hyperparams, MLPRecalConfig, LinearProbeConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

N_SAMPLES = 200
N_CLASSES = 5


def make_synthetic_data():
    rng = np.random.RandomState(RANDOM_SEED)
    signals = rng.randn(N_SAMPLES, N_LEADS, SIGNAL_LENGTH).astype(np.float32)
    labels = (rng.rand(N_SAMPLES, N_CLASSES) > 0.7).astype(np.float32)
    for i in range(N_CLASSES):
        if labels[:, i].sum() < 5:
            labels[rng.choice(N_SAMPLES, 5, replace=False), i] = 1.0
    patient_ids = np.arange(N_SAMPLES)
    return signals, labels, patient_ids


def test_splitting():
    from data_pipeline import patient_stratified_split, verify_no_leakage, build_split_indices
    _, labels, patient_ids = make_synthetic_data()
    split_patients = patient_stratified_split(patient_ids, labels)
    verify_no_leakage(split_patients)
    indices = build_split_indices(patient_ids, split_patients)
    assert sum(len(v) for v in indices.values()) == N_SAMPLES
    logger.info("PASS: splitting")


def test_normalization():
    from data_pipeline import compute_normalization_stats, normalize_signals
    signals, _, _ = make_synthetic_data()
    stats = compute_normalization_stats(signals)
    normed = normalize_signals(signals, stats)
    assert normed.shape == signals.shape
    per_lead_mean = np.mean(normed, axis=(0, 2))
    assert np.allclose(per_lead_mean, 0, atol=0.1)
    logger.info("PASS: normalization")


def test_evaluation():
    from evaluation import evaluate_multilabel
    rng = np.random.RandomState(RANDOM_SEED)
    probs = rng.rand(100, N_CLASSES).astype(np.float32)
    labels = (rng.rand(100, N_CLASSES) > 0.5).astype(np.float32)
    for i in range(N_CLASSES):
        if labels[:, i].sum() < 3:
            labels[:3, i] = 1.0
    class_names = [f"class_{i}" for i in range(N_CLASSES)]
    result = evaluate_multilabel(probs, labels, class_names)
    assert "macro" in result
    assert "per_class" in result
    for metric in ["auroc", "auprc", "brier", "ece_equal_width", "ece_equal_mass", "sce"]:
        assert metric in result["macro"], f"Missing metric: {metric}"
    logger.info("PASS: evaluation")


def test_recalibration():
    from recalibration import apply_recalibration
    rng = np.random.RandomState(RANDOM_SEED)
    cal_logits = rng.randn(80, N_CLASSES).astype(np.float32)
    cal_probs = 1 / (1 + np.exp(-cal_logits))
    cal_labels = (rng.rand(80, N_CLASSES) > 0.5).astype(np.float32)
    test_logits = rng.randn(40, N_CLASSES).astype(np.float32)
    test_probs = 1 / (1 + np.exp(-test_logits))

    for method_id in RECAL_IDS:
        result = apply_recalibration(
            method_id, cal_logits, cal_probs, cal_labels, test_logits, test_probs
        )
        assert result["calibrated_probs"].shape == (40, N_CLASSES)
        assert np.all(result["calibrated_probs"] >= 0)
        assert np.all(result["calibrated_probs"] <= 1)
        logger.info(f"PASS: recalibration {RECAL_NAMES[method_id]}")


def test_linear_probe():
    from linear_probe import train_linear_probe, get_predictions, LinearProbe
    rng = np.random.RandomState(RANDOM_SEED)
    train_feats = rng.randn(100, 64).astype(np.float32)
    train_labels = (rng.rand(100, N_CLASSES) > 0.5).astype(np.float32)
    val_feats = rng.randn(30, 64).astype(np.float32)
    val_labels = (rng.rand(30, N_CLASSES) > 0.5).astype(np.float32)
    config = LinearProbeConfig(max_epochs=3, patience=2)
    result = train_linear_probe(train_feats, train_labels, val_feats, val_labels, N_CLASSES, config)
    preds = get_predictions(result["model"], val_feats)
    assert preds["logits"].shape == (30, N_CLASSES)
    assert preds["probs"].shape == (30, N_CLASSES)
    logger.info("PASS: linear probe")


def test_s4d():
    from s4_baseline import S4DModel
    model = S4DModel(n_leads=N_LEADS, n_classes=N_CLASSES)
    x = torch.randn(4, N_LEADS, SIGNAL_LENGTH)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, N_CLASSES)
    logger.info("PASS: S4D forward pass")


def test_statistical():
    from statistical_tests import (
        compute_gap_closure, compute_brier_gap_closure, delong_test,
    )
    gc = compute_gap_closure(0.10, 0.06, 0.04)
    assert gc is not None
    assert abs(gc - (0.10 - 0.06) / (0.10 - 0.04)) < 1e-6
    assert compute_gap_closure(0.03, 0.02, 0.04) is None

    bgc = compute_brier_gap_closure(0.20, 0.15, 0.10)
    assert bgc is not None

    rng = np.random.RandomState(RANDOM_SEED)
    y = (rng.rand(200) > 0.5).astype(float)
    p1 = rng.rand(200)
    p2 = rng.rand(200)
    dl = delong_test(y, p1, p2)
    assert "p_value" in dl
    assert "z_statistic" in dl
    logger.info("PASS: statistical tests")


def test_visualization():
    from visualization import plot_reliability_diagram
    import tempfile
    bins_data = [
        {"bin_center": i / 10, "bin_lower": (i - 0.5) / 10, "bin_upper": (i + 0.5) / 10,
         "n_samples": 20, "avg_confidence": i / 10, "avg_accuracy": i / 10 + 0.02}
        for i in range(1, 11)
    ]
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        plot_reliability_diagram(bins_data, "TestModel", "TestDataset", "TestClass",
                                 Path(f.name), ece=0.05, sce=-0.02)
    logger.info("PASS: visualization")


def test_fm_checkpoint(model_id: str, checkpoint_path: Path):
    from models import FM_REGISTRY
    wrapper_cls = FM_REGISTRY[model_id]
    wrapper = wrapper_cls(checkpoint_path)
    try:
        wrapper.load()
        x = np.random.randn(2, N_LEADS, SIGNAL_LENGTH).astype(np.float32)
        feats = wrapper.extract_features(x)
        logger.info(
            f"PASS: {FM_NAMES[model_id]} — embedding shape {feats.shape}, "
            f"dim={wrapper.embedding_dim}"
        )
    except Exception as e:
        logger.error(f"FAIL: {FM_NAMES[model_id]} — {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    args = parser.parse_args()

    tests = [
        ("Splitting", test_splitting),
        ("Normalization", test_normalization),
        ("Evaluation", test_evaluation),
        ("Recalibration", test_recalibration),
        ("Linear probe", test_linear_probe),
        ("S4D", test_s4d),
        ("Statistical tests", test_statistical),
        ("Visualization", test_visualization),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            logger.error(f"FAIL: {name} — {e}")
            traceback.print_exc()
            failed += 1

    if args.checkpoint_dir and args.checkpoint_dir.exists():
        checkpoint_map = {
            "M1": args.checkpoint_dir / "merl" / "merl_checkpoint.pth",
            "M2": args.checkpoint_dir / "st_mem" / "st_mem_checkpoint.pth",
            "M3": args.checkpoint_dir / "hubert_ecg" / "hubert_ecg_checkpoint.pth",
            "M4": args.checkpoint_dir / "ecgfm_ked" / "ecgfm_ked_checkpoint.pth",
            "M5": args.checkpoint_dir / "ecg_fm" / "ecg_fm_checkpoint.pth",
        }
        for model_id, cp in checkpoint_map.items():
            if cp.exists():
                test_fm_checkpoint(model_id, cp)
            else:
                logger.warning(f"SKIP: {FM_NAMES[model_id]} — checkpoint not found at {cp}")

    logger.info(f"\nResults: {passed} passed, {failed} failed out of {passed + failed} tests")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
