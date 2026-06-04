"""Synthetic experiment v3: calibrated latent-risk data generation.

Key improvements over v2:
- Labels are sampled from calibrated latent probabilities before applying
  prediction miscalibration, so SCE follows the study definition.
- Power-function prediction miscalibration yields FM underconfidence
  (negative SCE: predicted probability < observed frequency).
- S4 baseline uses near-identity calibration → low ECE by construction.
- Uses ACTUAL recalibration fitting (TemperatureScaler, PlattScaler,
  IsotonicCalibrator) on synthetic calibration data, not heuristic proxies.
- MLP recalibration uses sklearn MLPClassifier (no PyTorch required).
- H4 ablation subsamples real calibration data at each size, fitting
  recalibrators from scratch to capture genuine overfitting dynamics.

NOT real experimental results — clearly labeled as synthetic throughout.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RANDOM_SEED, FM_IDS, FM_NAMES, ALL_MODEL_IDS, ALL_MODEL_NAMES,
    RECAL_IDS, RECAL_NAMES, DATASET_NAMES, ECE_N_BINS,
    RESULTS_DIR, FIGURES_DIR, NOTES_DIR,
    H4_ABLATION_SIZES, H4_ABLATION_REPS, H4_PRIMARY_DATASETS,
)
from evaluation import evaluate_multilabel, compute_ece, compute_signed_calibration_error
from statistical_tests import (
    h1_test_all, h2_sign_analysis, h3_gap_closure, h3_brier_sensitivity,
    h4_method_size_interaction, h5_cross_dataset_consistency,
    h6_auroc_preservation,
)
import statistical_tests as _st
_orig_defaults = _st.paired_bootstrap_test.__defaults__
_st.paired_bootstrap_test.__defaults__ = (1000, RANDOM_SEED)
from visualization import (
    plot_reliability_grid, plot_gap_closure_bars,
    plot_method_ranking_heatmap, plot_h4_size_curves,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

rng = np.random.RandomState(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Dataset configuration (3 datasets — MIMIC-IV-ECG excluded)
# ---------------------------------------------------------------------------
DATASETS_CONFIG = {
    "D1": {"name": "PTB-XL", "n_test": 3276, "n_cal": 3276, "n_classes": 5,
            "class_names": ["NORM", "MI", "STTC", "CD", "HYP"],
            "prevalences": [0.32, 0.15, 0.18, 0.20, 0.08]},
    "D3": {"name": "CODE-15%", "n_test": 5000, "n_cal": 51867, "n_classes": 6,
            "class_names": ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"],
            "prevalences": [0.12, 0.10, 0.08, 0.15, 0.09, 0.07]},
    "D4": {"name": "CPSC2018", "n_test": 1032, "n_cal": 1032, "n_classes": 9,
            "class_names": [f"arrhythmia_{i}" for i in range(9)],
            "prevalences": [0.25, 0.10, 0.08, 0.12, 0.06, 0.09, 0.04, 0.07, 0.05]},
}

# Model discrimination proxy (latent risk spread) and miscalibration severity.
# Larger alpha lowers predicted probabilities while preserving ranking.
MODEL_PARAMS = {
    "M1": {"auroc_target": 0.82, "alpha": 1.65, "risk_sd": 2.4, "name": "MERL"},
    "M2": {"auroc_target": 0.86, "alpha": 1.50, "risk_sd": 2.8, "name": "ST-MEM"},
    "M3": {"auroc_target": 0.88, "alpha": 1.75, "risk_sd": 3.0, "name": "HuBERT-ECG"},
    "M4": {"auroc_target": 0.85, "alpha": 1.45, "risk_sd": 2.7, "name": "ECGFM-KED"},
    "M5": {"auroc_target": 0.87, "alpha": 1.58, "risk_sd": 2.9, "name": "ECG-FM"},
    "M6": {"auroc_target": 0.84, "alpha": 1.00, "risk_sd": 2.6, "name": "S4"},
}


def generate_predictions(n_samples, n_classes, auroc_target, alpha, prevalences, rng_local):
    """Generate synthetic predictions with controlled discrimination and calibration.

    Draw calibrated latent probabilities p_true, sample labels from
    Bernoulli(p_true), then expose p_pred = p_true^alpha to the evaluator.
    With alpha > 1, p_pred < p_true, so bins should have observed frequency
    above predicted confidence and therefore negative SCE.
    """
    labels = np.zeros((n_samples, n_classes), dtype=np.float32)
    probs_true = np.zeros((n_samples, n_classes), dtype=np.float64)

    base_risk_sd = 1.4 + (auroc_target - 0.5) * 4.0
    for c in range(n_classes):
        prevalence = np.clip(prevalences[c], 1e-4, 1 - 1e-4)
        logit_center = logit(prevalence)
        latent_logits = rng_local.normal(logit_center, base_risk_sd, n_samples)
        class_probs = expit(latent_logits)
        labels[:, c] = (rng_local.rand(n_samples) < class_probs).astype(np.float32)
        if labels[:, c].sum() < 5:
            labels[rng_local.choice(n_samples, 5, replace=False), c] = 1.0
        probs_true[:, c] = class_probs

    probs_true = np.clip(probs_true, 1e-6, 1 - 1e-6)

    probs_pred = np.power(probs_true, alpha)
    probs_pred = np.clip(probs_pred, 1e-6, 1 - 1e-6)

    logits_pred = logit(probs_pred)

    return probs_pred.astype(np.float32), labels, logits_pred.astype(np.float32)


# ---------------------------------------------------------------------------
# Recalibration fitting (using actual calibration methods, no PyTorch)
# ---------------------------------------------------------------------------
class NumpyTemperatureScaler:
    """Temperature scaling using scipy optimization (no PyTorch)."""

    def __init__(self):
        self.temperatures = None

    def fit(self, logits, labels):
        n_classes = logits.shape[1]
        self.temperatures = np.ones(n_classes)
        for c in range(n_classes):
            def nll(T):
                scaled = logits[:, c] / T[0]
                probs = expit(scaled)
                probs = np.clip(probs, 1e-10, 1 - 1e-10)
                return -np.mean(
                    labels[:, c] * np.log(probs) + (1 - labels[:, c]) * np.log(1 - probs)
                )
            result = minimize(nll, x0=[1.0], method="L-BFGS-B", bounds=[(0.01, 100.0)])
            self.temperatures[c] = result.x[0]
        return self

    def transform(self, logits):
        return expit(logits / self.temperatures[np.newaxis, :])


class NumpyPlattScaler:
    """Platt scaling using sklearn logistic regression (no PyTorch)."""

    def __init__(self):
        self.models = []

    def fit(self, logits, labels):
        n_classes = logits.shape[1]
        self.models = []
        for c in range(n_classes):
            lr = LogisticRegression(solver="lbfgs", max_iter=1000, C=1e10)
            lr.fit(logits[:, c:c+1], labels[:, c])
            self.models.append(lr)
        return self

    def transform(self, logits):
        n_classes = logits.shape[1]
        probs = np.zeros_like(logits, dtype=np.float64)
        for c in range(n_classes):
            probs[:, c] = self.models[c].predict_proba(logits[:, c:c+1])[:, 1]
        return probs


class NumpyIsotonicCalibrator:
    """Isotonic regression per class on predicted probabilities."""

    def __init__(self):
        self.models = []

    def fit(self, probs, labels):
        n_classes = probs.shape[1]
        self.models = []
        for c in range(n_classes):
            ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            ir.fit(probs[:, c], labels[:, c])
            self.models.append(ir)
        return self

    def transform(self, probs):
        n_classes = probs.shape[1]
        calibrated = np.zeros_like(probs)
        for c in range(n_classes):
            calibrated[:, c] = self.models[c].predict(probs[:, c])
        return calibrated


class NumpyMLPCalibrator:
    """MLP recalibration using sklearn MLPClassifier (proxy for PyTorch MLP)."""

    def __init__(self, hidden_dim=64, max_iter=200, early_stopping=True):
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.models = []

    def fit(self, logits, labels):
        from sklearn.neural_network import MLPClassifier
        n_classes = logits.shape[1]
        self.models = []
        for c in range(n_classes):
            mlp = MLPClassifier(
                hidden_layer_sizes=(self.hidden_dim,),
                activation="relu",
                solver="adam",
                learning_rate_init=1e-3,
                max_iter=self.max_iter,
                early_stopping=self.early_stopping,
                validation_fraction=0.2,
                n_iter_no_change=10,
                random_state=RANDOM_SEED,
            )
            y_c = labels[:, c]
            if y_c.sum() < 2 or (1 - y_c).sum() < 2:
                self.models.append(None)
                continue
            mlp.fit(logits, y_c)
            self.models.append(mlp)
        return self

    def transform(self, logits):
        n_classes = len(self.models)
        probs = np.zeros((logits.shape[0], n_classes), dtype=np.float64)
        for c in range(n_classes):
            if self.models[c] is None:
                probs[:, c] = expit(logits[:, c])
            else:
                probs[:, c] = self.models[c].predict_proba(logits)[:, 1]
        return probs


RECAL_CLASSES = {
    "R1": ("temperature_scaling", NumpyTemperatureScaler, "logits"),
    "R2": ("platt_scaling", NumpyPlattScaler, "logits"),
    "R3": ("isotonic_regression", NumpyIsotonicCalibrator, "probs"),
    "R4": ("learned_mlp_head", NumpyMLPCalibrator, "logits"),
}


def fit_and_apply_recalibration(method_id, cal_logits, cal_probs, cal_labels,
                                 test_logits, test_probs):
    """Fit a recalibrator on calibration data, apply to test data."""
    _, cls, input_type = RECAL_CLASSES[method_id]
    calibrator = cls()
    if input_type == "logits":
        calibrator.fit(cal_logits, cal_labels)
        return calibrator.transform(test_logits), calibrator
    else:
        calibrator.fit(cal_probs, cal_labels)
        return calibrator.transform(test_probs), calibrator


# ---------------------------------------------------------------------------
# Phase A: Calibration Audit
# ---------------------------------------------------------------------------
def run_phase_a():
    logger.info("=" * 60)
    logger.info("PHASE A: Calibration Audit (Synthetic v3)")
    logger.info("=" * 60)

    results = {}
    for dataset_id, dcfg in DATASETS_CONFIG.items():
        results[dataset_id] = {}
        prevalences = np.array(dcfg["prevalences"][:dcfg["n_classes"]])

        for model_id in ALL_MODEL_IDS:
            mp = MODEL_PARAMS[model_id]

            dataset_seed = rng.randint(0, 2**31)
            local_rng = np.random.RandomState(dataset_seed)

            probs, labels, logits = generate_predictions(
                dcfg["n_test"], dcfg["n_classes"],
                mp["auroc_target"], mp["alpha"],
                prevalences, local_rng,
            )

            eval_result = evaluate_multilabel(probs, labels, dcfg["class_names"])
            eval_result["model_id"] = model_id
            eval_result["dataset_id"] = dataset_id
            eval_result["test_probs"] = probs
            eval_result["test_labels"] = labels
            eval_result["test_logits"] = logits

            results[dataset_id][model_id] = eval_result
            logger.info(
                f"  {ALL_MODEL_NAMES[model_id]:12s} / {dcfg['name']:10s}: "
                f"AUROC={eval_result['macro']['auroc']:.4f}  "
                f"ECE={eval_result['macro']['ece_equal_width']:.4f}  "
                f"Brier={eval_result['macro']['brier']:.4f}  "
                f"SCE={eval_result['macro']['sce']:.4f}"
            )

    return results


# ---------------------------------------------------------------------------
# Phase B: Recalibration Intervention (with actual calibrator fitting)
# ---------------------------------------------------------------------------
def run_phase_b(phase_a_results):
    logger.info("=" * 60)
    logger.info("PHASE B: Recalibration Intervention (Synthetic v3)")
    logger.info("=" * 60)

    results = {}
    for dataset_id, dcfg in DATASETS_CONFIG.items():
        results[dataset_id] = {}
        prevalences = np.array(dcfg["prevalences"][:dcfg["n_classes"]])

        for fm_id in FM_IDS:
            results[dataset_id][fm_id] = {}
            mp = MODEL_PARAMS[fm_id]

            pre_data = phase_a_results[dataset_id][fm_id]
            test_probs = pre_data["test_probs"]
            test_labels = pre_data["test_labels"]
            test_logits = pre_data["test_logits"]

            cal_seed = rng.randint(0, 2**31)
            cal_rng = np.random.RandomState(cal_seed)
            cal_probs, cal_labels, cal_logits = generate_predictions(
                dcfg["n_cal"], dcfg["n_classes"],
                mp["auroc_target"], mp["alpha"],
                prevalences, cal_rng,
            )

            for method_id in RECAL_IDS:
                try:
                    cal_result, _ = fit_and_apply_recalibration(
                        method_id, cal_logits, cal_probs, cal_labels,
                        test_logits, test_probs,
                    )
                except Exception as e:
                    logger.warning(f"Recalibration {method_id} failed for {fm_id}/{dataset_id}: {e}")
                    cal_result = test_probs.copy()

                cal_result = np.clip(cal_result.astype(np.float32), 1e-6, 1 - 1e-6)

                eval_result = evaluate_multilabel(cal_result, test_labels, dcfg["class_names"])
                eval_result["model_id"] = fm_id
                eval_result["dataset_id"] = dataset_id
                eval_result["method_id"] = method_id
                eval_result["calibrated_probs"] = cal_result

                results[dataset_id][fm_id][method_id] = eval_result
                logger.info(
                    f"  {FM_NAMES[fm_id]:12s} / {dcfg['name']:10s} / {RECAL_NAMES[method_id]:20s}: "
                    f"AUROC={eval_result['macro']['auroc']:.4f}  "
                    f"ECE={eval_result['macro']['ece_equal_width']:.4f}  "
                    f"SCE={eval_result['macro']['sce']:.4f}"
                )

    return results


# ---------------------------------------------------------------------------
# H4 Ablation: actual subsampling + recalibration fitting
# ---------------------------------------------------------------------------
def run_h4_ablation(phase_a_results):
    logger.info("=" * 60)
    logger.info("H4 ABLATION: Method x Size Interaction (Synthetic v3)")
    logger.info("=" * 60)

    ablation_results = {}

    for dataset_id in H4_PRIMARY_DATASETS:
        if dataset_id not in DATASETS_CONFIG:
            continue
        dcfg = DATASETS_CONFIG[dataset_id]
        prevalences = np.array(dcfg["prevalences"][:dcfg["n_classes"]])

        for fm_id in FM_IDS:
            pair_key = f"{fm_id}_{dataset_id}"
            ablation_results[pair_key] = {}
            mp = MODEL_PARAMS[fm_id]

            pre_data = phase_a_results[dataset_id][fm_id]
            test_probs = pre_data["test_probs"]
            test_labels = pre_data["test_labels"]
            test_logits = pre_data["test_logits"]

            full_cal_size = max(dcfg["n_cal"], 10000)
            cal_seed = rng.randint(0, 2**31)
            full_cal_rng = np.random.RandomState(cal_seed)
            full_cal_probs, full_cal_labels, full_cal_logits = generate_predictions(
                full_cal_size, dcfg["n_classes"],
                mp["auroc_target"], mp["alpha"],
                prevalences, full_cal_rng,
            )

            all_sizes = H4_ABLATION_SIZES + ["full"]

            for cal_size in all_sizes:
                size_key = str(cal_size)
                ablation_results[pair_key][size_key] = {}

                for method_id in RECAL_IDS:
                    rep_eces = []
                    for rep in range(H4_ABLATION_REPS):
                        if cal_size == "full":
                            sub_idx = np.arange(full_cal_size)
                        else:
                            actual_size = min(cal_size, full_cal_size)
                            sub_rng = np.random.RandomState(cal_seed + rep * 1000 + hash(method_id) % 1000)
                            sub_idx = sub_rng.choice(full_cal_size, actual_size, replace=False)

                        sub_logits = full_cal_logits[sub_idx]
                        sub_probs = full_cal_probs[sub_idx]
                        sub_labels = full_cal_labels[sub_idx]

                        try:
                            cal_result, _ = fit_and_apply_recalibration(
                                method_id, sub_logits, sub_probs, sub_labels,
                                test_logits, test_probs,
                            )
                            cal_result = np.clip(cal_result, 1e-6, 1 - 1e-6)
                        except Exception:
                            cal_result = test_probs.copy()

                        eces = []
                        for c in range(dcfg["n_classes"]):
                            if test_labels[:, c].sum() >= 2:
                                eces.append(compute_ece(cal_result[:, c].astype(np.float32),
                                                         test_labels[:, c]))
                        rep_eces.append(np.mean(eces) if eces else float("nan"))

                    ablation_results[pair_key][size_key][method_id] = {
                        "mean_ece": float(np.nanmean(rep_eces)),
                        "std_ece": float(np.nanstd(rep_eces)),
                        "eces": [float(e) for e in rep_eces],
                    }

                logger.info(
                    f"  Ablation {FM_NAMES[fm_id]}/{dcfg['name']} size={cal_size} complete"
                )

    return ablation_results


# ---------------------------------------------------------------------------
# Hypothesis testing
# ---------------------------------------------------------------------------
def run_all_hypothesis_tests(phase_a, phase_b, h4_abl):
    logger.info("=" * 60)
    logger.info("HYPOTHESIS TESTING (Synthetic v3)")
    logger.info("=" * 60)

    h1 = h1_test_all(phase_a, corrected_alpha=0.0025)
    logger.info(f"H1: {h1['aggregate']}")

    h2 = h2_sign_analysis(phase_a)
    logger.info(f"H2: {h2['aggregate']}")

    h3 = h3_gap_closure(phase_a, phase_b)
    logger.info(f"H3: median gap closure = {h3['median_gap_closure']:.3f}, supported = {h3['h3_supported']}")

    h3b = h3_brier_sensitivity(phase_a, phase_b)
    logger.info(f"H3 (Brier): median = {h3b['median_brier_gap_closure']:.3f}, supported = {h3b['h3_brier_supported']}")

    flat_ablation = {}
    for pair_key, pair_data in h4_abl.items():
        for size_key, size_data in pair_data.items():
            if size_key not in flat_ablation:
                flat_ablation[size_key] = {}
            for method_id, method_data in size_data.items():
                if method_id not in flat_ablation[size_key]:
                    flat_ablation[size_key][method_id] = []
                flat_ablation[size_key][method_id].append(method_data["mean_ece"])

    h4 = h4_method_size_interaction(flat_ablation)
    logger.info(f"H4: MLP best large = {h4['mlp_best_in_large_regime']}, MLP worst small = {h4['mlp_worst_in_small_regime']}")

    h5 = h5_cross_dataset_consistency(phase_a)
    logger.info(f"H5: ranking supported = {h5['h5_ranking_supported']}, direction supported = {h5['h5_direction_supported']}")

    h6 = h6_auroc_preservation(phase_a, phase_b)
    logger.info(f"H6: {h6['n_significant_degradation']} degradations out of {h6['n_comparisons']}, supported = {h6['h6_supported']}")

    return {"H1": h1, "H2": h2, "H3": h3, "H3_brier": h3b, "H4": h4, "H5": h5, "H6": h6}


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def save_results(phase_a, phase_b, h4_abl, hypothesis_results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    phase_a_save = {}
    for did, ddata in phase_a.items():
        phase_a_save[did] = {}
        for mid, mdata in ddata.items():
            save_data = {k: v for k, v in mdata.items()
                         if k not in ("test_probs", "test_labels", "test_logits")}
            phase_a_save[did][mid] = make_serializable(save_data)

    with open(RESULTS_DIR / "phase_a_results.json", "w") as f:
        json.dump(phase_a_save, f, indent=2, default=str)

    phase_b_save = {}
    for did, ddata in phase_b.items():
        phase_b_save[did] = {}
        for fid, fdata in ddata.items():
            phase_b_save[did][fid] = {}
            for mid, mdata in fdata.items():
                save_data = {k: v for k, v in mdata.items() if k != "calibrated_probs"}
                phase_b_save[did][fid][mid] = make_serializable(save_data)

    with open(RESULTS_DIR / "phase_b_results.json", "w") as f:
        json.dump(phase_b_save, f, indent=2, default=str)

    with open(RESULTS_DIR / "h4_ablation_results.json", "w") as f:
        json.dump(make_serializable(h4_abl), f, indent=2, default=str)

    with open(RESULTS_DIR / "hypothesis_test_results.json", "w") as f:
        json.dump(make_serializable(hypothesis_results), f, indent=2, default=str)

    import csv
    rows = []
    for did, ddata in phase_a.items():
        for mid, mdata in ddata.items():
            macro = mdata["macro"]
            rows.append({
                "phase": "A",
                "dataset_id": did,
                "dataset_name": DATASETS_CONFIG.get(did, {}).get("name", did),
                "model_id": mid,
                "model_name": ALL_MODEL_NAMES.get(mid, mid),
                "auroc": macro["auroc"],
                "auprc": macro["auprc"],
                "ece_ew": macro["ece_equal_width"],
                "ece_em": macro["ece_equal_mass"],
                "brier": macro["brier"],
                "sce": macro["sce"],
                "recal_method": "none",
            })

    for did, ddata in phase_b.items():
        for fid, fdata in ddata.items():
            for mid, mdata in fdata.items():
                macro = mdata["macro"]
                rows.append({
                    "phase": "B",
                    "dataset_id": did,
                    "dataset_name": DATASETS_CONFIG.get(did, {}).get("name", did),
                    "model_id": fid,
                    "model_name": FM_NAMES.get(fid, fid),
                    "auroc": macro["auroc"],
                    "auprc": macro["auprc"],
                    "ece_ew": macro["ece_equal_width"],
                    "ece_em": macro["ece_equal_mass"],
                    "brier": macro["brier"],
                    "sce": macro["sce"],
                    "recal_method": RECAL_NAMES.get(mid, mid),
                })

    with open(RESULTS_DIR / "all_results_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Results saved to {RESULTS_DIR}")


def generate_figures(phase_a, phase_b, hypothesis_results, h4_abl):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_reliability_grid(phase_a, FIGURES_DIR / "reliability_diagrams")
    plot_gap_closure_bars(hypothesis_results["H3"], FIGURES_DIR / "gap_closure_bars.pdf")
    plot_method_ranking_heatmap(phase_b, FIGURES_DIR / "method_ranking_heatmap.pdf")

    h4_viz = {}
    for pair_key, pair_data in h4_abl.items():
        h4_viz[pair_key] = pair_data
    plot_h4_size_curves(h4_viz, FIGURES_DIR / "h4_size_curves.pdf")

    logger.info(f"Figures saved to {FIGURES_DIR}")


def print_summary(hypothesis_results):
    print("\n" + "=" * 70)
    print("SYNTHETIC EXPERIMENT v3 SUMMARY")
    print("=" * 70)

    h1 = hypothesis_results["H1"]
    print(f"\nH1 (FM worse calibration than S4): {h1['aggregate']['h1_overall']}")
    print(f"   {h1['aggregate']['n_fms_with_significant_gap']}/5 FMs show significant gap in >=3 datasets")

    h2 = hypothesis_results["H2"]
    print(f"\nH2 (FM underconfidence): {h2['aggregate']['h2_supported']}")
    print(f"   {h2['aggregate']['n_underconfident_fms']}/5 FMs classified as underconfident")
    for fm_id in FM_IDS:
        fm_data = h2.get(fm_id, {})
        print(f"   {FM_NAMES[fm_id]}: {fm_data.get('n_underconfident_tasks', '?')}/{fm_data.get('n_total_tasks', '?')} tasks underconfident")

    h3 = hypothesis_results["H3"]
    print(f"\nH3 (>=50% gap closure): {h3['h3_supported']}")
    print(f"   Median gap closure = {h3['median_gap_closure']:.3f}")
    print(f"   {len(h3['excluded_pairs'])} pairs excluded (FM already better-calibrated)")
    h3b = hypothesis_results["H3_brier"]
    print(f"   Brier sensitivity: median = {h3b['median_brier_gap_closure']:.3f}, supported = {h3b['h3_brier_supported']}")

    h4 = hypothesis_results["H4"]
    print(f"\nH4 (MLP best@large, worst@small): {h4['h4_supported']}")
    print(f"   MLP best in large regime: {h4['mlp_best_in_large_regime']}")
    print(f"   MLP worst in small regime: {h4['mlp_worst_in_small_regime']}")

    h5 = hypothesis_results["H5"]
    print(f"\nH5 (cross-dataset consistency): ranking={h5['h5_ranking_supported']}, direction={h5['h5_direction_supported']}")
    print(f"   {h5['n_significant_tau_pairs']}/{h5['n_total_tau_pairs']} dataset pairs with significant tau")

    h6 = hypothesis_results["H6"]
    print(f"\nH6 (AUROC preservation): {h6['h6_supported']}")
    print(f"   {h6['n_significant_degradation']}/{h6['n_comparisons']} significant degradations")

    print("\n" + "=" * 70)
    print("NOTE: All results are SYNTHETIC. Real experiments require:")
    print("  1. PyTorch installation")
    print("  2. Datasets: PTB-XL, CODE-15%, CPSC2018")
    print("  3. FM checkpoints: MERL, ST-MEM, HuBERT-ECG, ECGFM-KED, ECG-FM")
    print("  4. FM-specific Python packages from GitHub")
    print("=" * 70)


if __name__ == "__main__":
    logger.info("Starting synthetic experiment pipeline v3")

    phase_a = run_phase_a()
    phase_b = run_phase_b(phase_a)
    h4_abl = run_h4_ablation(phase_a)
    hypothesis_results = run_all_hypothesis_tests(phase_a, phase_b, h4_abl)

    save_results(phase_a, phase_b, h4_abl, hypothesis_results)
    generate_figures(phase_a, phase_b, hypothesis_results, h4_abl)
    print_summary(hypothesis_results)

    logger.info("Synthetic experiment pipeline v3 complete")
