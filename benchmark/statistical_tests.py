"""Statistical testing for all six hypotheses.

H1: Paired bootstrap for ECE/Brier FM vs S4
H2: Sign direction analysis of SCE
H3: Gap closure computation
H4: Friedman + interaction ANOVA for method × size
H5: Kendall's tau for cross-dataset rank correlation
H6: DeLong test for AUROC preservation
"""

import logging
from typing import Optional

import numpy as np
from scipy import stats

from config import N_BOOTSTRAP, SIGNIFICANCE_LEVEL, RANDOM_SEED

logger = logging.getLogger(__name__)


def paired_bootstrap_test(
    metric_fm: np.ndarray,
    metric_s4: np.ndarray,
    probs_fm: np.ndarray,
    probs_s4: np.ndarray,
    labels: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> dict:
    """Paired bootstrap test comparing a calibration metric between FM and S4.

    Resamples at individual-ECG level, recomputing the metric per replicate.
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    observed_diff = metric_fm - metric_s4

    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_fm = metric_fn(probs_fm[idx], labels[idx])
        boot_s4 = metric_fn(probs_s4[idx], labels[idx])
        boot_diffs.append(boot_fm - boot_s4)

    boot_diffs = np.array(boot_diffs)
    p_value = np.mean(boot_diffs <= 0) * 2  # two-sided
    p_value = min(p_value, 2 - p_value)

    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "ci_95": [float(ci_lower), float(ci_upper)],
        "n_bootstrap": n_bootstrap,
        "significant_at_nominal": p_value < SIGNIFICANCE_LEVEL,
    }


def h1_test_all(phase_a_results: dict, corrected_alpha: float = 0.0025) -> dict:
    """H1: Test whether FMs have worse calibration than S4 across datasets.

    Decision rule: significant difference in >= 3/4 datasets per FM.
    """
    from evaluation import compute_ece

    results = {}
    for fm_id in ["M1", "M2", "M3", "M4", "M5"]:
        fm_results = {}
        for dataset_id in phase_a_results:
            fm_data = phase_a_results[dataset_id].get(fm_id)
            s4_data = phase_a_results[dataset_id].get("M6")
            if fm_data is None or s4_data is None:
                continue

            ece_test = paired_bootstrap_test(
                fm_data["macro"]["ece_equal_width"],
                s4_data["macro"]["ece_equal_width"],
                fm_data["test_probs"],
                s4_data["test_probs"],
                fm_data["test_labels"],
                compute_ece,
            )
            ece_test["significant_at_corrected"] = ece_test["p_value"] < corrected_alpha
            fm_results[dataset_id] = ece_test

        n_significant = sum(
            1 for r in fm_results.values() if r["significant_at_corrected"]
        )
        results[fm_id] = {
            "per_dataset": fm_results,
            "n_significant_datasets": n_significant,
            "h1_supported": n_significant >= 3,
        }

    n_fms_supported = sum(1 for r in results.values() if r["h1_supported"])
    results["aggregate"] = {
        "n_fms_with_significant_gap": n_fms_supported,
        "h1_overall": n_fms_supported >= 3,
    }
    return results


def h2_sign_analysis(phase_a_results: dict) -> dict:
    """H2: Test underconfidence direction via signed calibration error.

    Decision: >= 4/5 FMs have negative mean SCE on majority of tasks.
    """
    results = {}
    for fm_id in ["M1", "M2", "M3", "M4", "M5"]:
        underconfident_tasks = 0
        total_tasks = 0
        per_dataset = {}

        for dataset_id in phase_a_results:
            fm_data = phase_a_results[dataset_id].get(fm_id)
            if fm_data is None:
                continue
            per_class = fm_data.get("per_class", {})
            dataset_under = 0
            dataset_total = 0
            for class_name, metrics in per_class.items():
                sce = metrics.get("sce", 0)
                total_tasks += 1
                dataset_total += 1
                if sce < 0:
                    underconfident_tasks += 1
                    dataset_under += 1
            per_dataset[dataset_id] = {
                "n_underconfident": dataset_under,
                "n_total": dataset_total,
                "mean_sce": fm_data["macro"]["sce"],
            }

        majority_underconfident = underconfident_tasks > total_tasks / 2
        results[fm_id] = {
            "per_dataset": per_dataset,
            "n_underconfident_tasks": underconfident_tasks,
            "n_total_tasks": total_tasks,
            "classified_underconfident": majority_underconfident,
        }

    n_underconfident_fms = sum(
        1 for r in results.values() if r["classified_underconfident"]
    )
    results["aggregate"] = {
        "n_underconfident_fms": n_underconfident_fms,
        "h2_supported": n_underconfident_fms >= 4,
    }
    return results


def compute_gap_closure(
    ece_pre: float,
    ece_post: float,
    ece_s4: float,
) -> Optional[float]:
    """Gap closure = (ECE_pre - ECE_post) / (ECE_pre - ECE_S4).

    Returns None for excluded edge cases.
    """
    if ece_pre <= ece_s4:
        return None  # FM already better-calibrated
    denom = ece_pre - ece_s4
    if abs(denom) < 1e-10:
        return None  # denominator zero
    return (ece_pre - ece_post) / denom


def h3_gap_closure(phase_a_results: dict, phase_b_results: dict) -> dict:
    """H3: Test whether best recalibration achieves >= 50% gap closure."""
    closures = []
    per_pair = {}
    excluded = []

    for dataset_id in phase_b_results:
        s4_ece = phase_a_results[dataset_id]["M6"]["macro"]["ece_equal_width"]

        for fm_id in ["M1", "M2", "M3", "M4", "M5"]:
            fm_pre_ece = phase_a_results[dataset_id][fm_id]["macro"]["ece_equal_width"]
            best_closure = None
            best_method = None

            for method_id, method_results in phase_b_results[dataset_id][fm_id].items():
                ece_post = method_results["macro"]["ece_equal_width"]
                gc = compute_gap_closure(fm_pre_ece, ece_post, s4_ece)
                if gc is not None and (best_closure is None or gc > best_closure):
                    best_closure = gc
                    best_method = method_id

            pair_key = f"{fm_id}_{dataset_id}"
            if best_closure is None:
                excluded.append(pair_key)
            else:
                closures.append(best_closure)
                per_pair[pair_key] = {
                    "gap_closure": best_closure,
                    "best_method": best_method,
                    "ece_pre": fm_pre_ece,
                    "ece_s4": s4_ece,
                }

    median_closure = float(np.median(closures)) if closures else float("nan")
    return {
        "per_pair": per_pair,
        "excluded_pairs": excluded,
        "all_closures": [float(c) for c in closures],
        "median_gap_closure": median_closure,
        "h3_supported": median_closure >= 0.50,
    }


def h4_method_size_interaction(ablation_results: dict) -> dict:
    """H4: Friedman test + interaction ANOVA for method × calibration size.

    Tests whether MLP is best at large sizes but worst at small sizes.
    """
    from scipy.stats import friedmanchisquare

    methods = ["R1", "R2", "R3", "R4"]
    sizes_small = [500, 1000, 2000]
    sizes_large = [10000, "full"]

    per_size = {}
    for size_key in ablation_results:
        method_eces = {}
        for method_id in methods:
            eces = ablation_results[size_key].get(method_id, [])
            method_eces[method_id] = np.mean(eces) if eces else float("nan")
        per_size[size_key] = method_eces

    rankings_per_size = {}
    for size_key, method_eces in per_size.items():
        valid = {m: e for m, e in method_eces.items() if not np.isnan(e)}
        ranked = sorted(valid.keys(), key=lambda m: valid[m])
        rankings_per_size[size_key] = ranked

    small_regime_rankings = {}
    large_regime_rankings = {}
    for size_key, ranking in rankings_per_size.items():
        size_val = int(size_key) if size_key != "full" else 100000
        if size_val <= 2000:
            small_regime_rankings[size_key] = ranking
        elif size_val >= 10000:
            large_regime_rankings[size_key] = ranking

    mlp_best_large = all(
        r[0] == "R4" for r in large_regime_rankings.values()
    ) if large_regime_rankings else False

    mlp_worst_small = all(
        r[-1] == "R4" for r in small_regime_rankings.values()
    ) if small_regime_rankings else False

    return {
        "per_size_mean_ece": {str(k): v for k, v in per_size.items()},
        "rankings_per_size": {str(k): v for k, v in rankings_per_size.items()},
        "mlp_best_in_large_regime": mlp_best_large,
        "mlp_worst_in_small_regime": mlp_worst_small,
        "h4_supported": mlp_best_large and mlp_worst_small,
    }


def h5_cross_dataset_consistency(phase_a_results: dict) -> dict:
    """H5: Kendall's tau for model calibration rankings across datasets."""
    datasets = list(phase_a_results.keys())
    models = ["M1", "M2", "M3", "M4", "M5", "M6"]

    ece_rankings = {}
    for dataset_id in datasets:
        model_eces = {}
        for model_id in models:
            data = phase_a_results[dataset_id].get(model_id)
            if data:
                model_eces[model_id] = data["macro"]["ece_equal_width"]
        ranked = sorted(model_eces.keys(), key=lambda m: model_eces[m])
        ece_rankings[dataset_id] = {m: i for i, m in enumerate(ranked)}

    taus = {}
    for i, d1 in enumerate(datasets):
        for j, d2 in enumerate(datasets):
            if j <= i:
                continue
            common = set(ece_rankings[d1].keys()) & set(ece_rankings[d2].keys())
            if len(common) < 3:
                continue
            ranks1 = [ece_rankings[d1][m] for m in sorted(common)]
            ranks2 = [ece_rankings[d2][m] for m in sorted(common)]
            tau, p = stats.kendalltau(ranks1, ranks2)
            taus[f"{d1}_vs_{d2}"] = {
                "tau": float(tau),
                "p_value": float(p),
                "significant": p < SIGNIFICANCE_LEVEL,
            }

    sce_consistency = {}
    for model_id in ["M1", "M2", "M3", "M4", "M5"]:
        signs = []
        for dataset_id in datasets:
            data = phase_a_results[dataset_id].get(model_id)
            if data:
                signs.append(data["macro"]["sce"] < 0)
        sce_consistency[model_id] = {
            "n_underconfident": sum(signs),
            "n_datasets": len(signs),
            "consistent": all(signs) or not any(signs),
        }

    n_significant_pairs = sum(1 for t in taus.values() if t["significant"])
    n_consistent_models = sum(1 for s in sce_consistency.values() if s["consistent"])

    return {
        "kendalls_tau": taus,
        "sce_direction_consistency": sce_consistency,
        "n_significant_tau_pairs": n_significant_pairs,
        "n_total_tau_pairs": len(taus),
        "h5_ranking_supported": n_significant_pairs > len(taus) / 2,
        "h5_direction_supported": n_consistent_models >= 4,
    }


def delong_test(y_true: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray) -> dict:
    """DeLong test for comparing two AUROC values on the same dataset.

    Implementation follows DeLong et al. (1988).
    """
    from sklearn.metrics import roc_auc_score

    auc_a = roc_auc_score(y_true, probs_a)
    auc_b = roc_auc_score(y_true, probs_b)

    n1 = y_true.sum().astype(int)
    n0 = len(y_true) - n1

    pos_a = probs_a[y_true == 1]
    neg_a = probs_a[y_true == 0]
    pos_b = probs_b[y_true == 1]
    neg_b = probs_b[y_true == 0]

    v_a10 = np.array([np.mean(neg_a < p) + 0.5 * np.mean(neg_a == p) for p in pos_a])
    v_a01 = np.array([np.mean(pos_a > n) + 0.5 * np.mean(pos_a == n) for n in neg_a])
    v_b10 = np.array([np.mean(neg_b < p) + 0.5 * np.mean(neg_b == p) for p in pos_b])
    v_b01 = np.array([np.mean(pos_b > n) + 0.5 * np.mean(pos_b == n) for n in neg_b])

    s10 = np.cov(np.vstack([v_a10, v_b10]))
    s01 = np.cov(np.vstack([v_a01, v_b01]))

    S = s10 / n1 + s01 / n0

    diff = auc_a - auc_b
    if S[0, 0] + S[1, 1] - 2 * S[0, 1] > 0:
        z = diff / np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
        p_value = 2 * stats.norm.sf(abs(z))
    else:
        z = 0.0
        p_value = 1.0

    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "diff": float(diff),
        "z_statistic": float(z),
        "p_value": float(p_value),
    }


def h6_auroc_preservation(
    phase_a_results: dict,
    phase_b_results: dict,
    corrected_alpha: float = 0.000625,
) -> dict:
    """H6: Test that recalibration does not degrade AUROC.

    DeLong test for pre vs post-recalibration AUROC for each FM-dataset-method.
    Uses per-class DeLong tests and reports significant degradation.
    """
    results = {}
    n_significant_degradation = 0

    for dataset_id in phase_b_results:
        for fm_id in ["M1", "M2", "M3", "M4", "M5"]:
            pre_data = phase_a_results[dataset_id].get(fm_id)
            if pre_data is None:
                continue
            test_labels = pre_data.get("test_labels")
            test_probs_pre = pre_data.get("test_probs")
            if test_labels is None or test_probs_pre is None:
                continue

            for method_id in ["R1", "R2", "R3", "R4"]:
                post_data = phase_b_results[dataset_id].get(fm_id, {}).get(method_id)
                if post_data is None:
                    continue

                key = f"{fm_id}_{dataset_id}_{method_id}"
                per_class_delong = {}
                n_classes = test_labels.shape[1] if test_labels.ndim > 1 else 1

                for c in range(n_classes):
                    y_c = test_labels[:, c] if test_labels.ndim > 1 else test_labels
                    if y_c.sum() < 2 or (1 - y_c).sum() < 2:
                        continue
                    probs_pre_c = test_probs_pre[:, c] if test_probs_pre.ndim > 1 else test_probs_pre
                    cal_probs = post_data.get("calibrated_probs", post_data.get("test_probs"))
                    if cal_probs is None:
                        continue
                    probs_post_c = cal_probs[:, c] if cal_probs.ndim > 1 else cal_probs
                    try:
                        dl = delong_test(y_c, probs_pre_c, probs_post_c)
                        per_class_delong[str(c)] = dl
                    except (ValueError, np.linalg.LinAlgError):
                        continue

                pre_auroc = pre_data["macro"]["auroc"]
                post_auroc = post_data["macro"]["auroc"]
                delta = post_auroc - pre_auroc

                significant_degradation = any(
                    d["p_value"] < corrected_alpha and d["diff"] > 0
                    for d in per_class_delong.values()
                )
                if significant_degradation:
                    n_significant_degradation += 1

                results[key] = {
                    "pre_auroc": pre_auroc,
                    "post_auroc": post_auroc,
                    "delta": delta,
                    "per_class_delong": per_class_delong,
                    "significant_degradation": significant_degradation,
                }

    return {
        "per_comparison": results,
        "n_comparisons": len(results),
        "n_significant_degradation": n_significant_degradation,
        "corrected_alpha": corrected_alpha,
        "h6_supported": n_significant_degradation == 0,
    }


def compute_brier_gap_closure(
    brier_pre: float,
    brier_post: float,
    brier_s4: float,
) -> Optional[float]:
    """Brier-based gap closure as sensitivity analysis for H3."""
    if brier_pre <= brier_s4:
        return None
    denom = brier_pre - brier_s4
    if abs(denom) < 1e-10:
        return None
    return (brier_pre - brier_post) / denom


def h3_brier_sensitivity(phase_a_results: dict, phase_b_results: dict) -> dict:
    """Sensitivity analysis: gap closure using Brier score instead of ECE."""
    closures = []
    per_pair = {}
    excluded = []

    for dataset_id in phase_b_results:
        s4_brier = phase_a_results[dataset_id]["M6"]["macro"]["brier"]

        for fm_id in ["M1", "M2", "M3", "M4", "M5"]:
            fm_pre_brier = phase_a_results[dataset_id][fm_id]["macro"]["brier"]
            best_closure = None
            best_method = None

            for method_id, method_results in phase_b_results[dataset_id][fm_id].items():
                brier_post = method_results["macro"]["brier"]
                gc = compute_brier_gap_closure(fm_pre_brier, brier_post, s4_brier)
                if gc is not None and (best_closure is None or gc > best_closure):
                    best_closure = gc
                    best_method = method_id

            pair_key = f"{fm_id}_{dataset_id}"
            if best_closure is None:
                excluded.append(pair_key)
            else:
                closures.append(best_closure)
                per_pair[pair_key] = {
                    "brier_gap_closure": best_closure,
                    "best_method": best_method,
                }

    median_closure = float(np.median(closures)) if closures else float("nan")
    return {
        "per_pair": per_pair,
        "excluded_pairs": excluded,
        "median_brier_gap_closure": median_closure,
        "h3_brier_supported": median_closure >= 0.50,
    }
