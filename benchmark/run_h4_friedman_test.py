"""
H4 Friedman rank test on the h4_size_ablation_pilot_raw.csv.

Tests whether method rankings are non-uniform at each calibration size,
and specifically whether the MLP head's rank-last position at small n
is statistically significant across the four model-dataset cells.

Statistical design:
  - For each cal_size, build a (blocks x methods) ECE matrix where
    blocks = model-dataset cell x repetition (4 cells x 5 reps = 20 blocks).
  - Apply scipy.stats.friedmanchisquare to each cal_size.
  - Follow with pairwise Wilcoxon signed-rank (MLP vs each other method)
    at small n (n <= 75) using the same block structure.
  - Compute Spearman rank correlation between numeric cal_size and MLP's
    mean ECE rank across cal_sizes.
  - Save machine-readable JSON and a flat CSV of per-size Friedman results.
"""

import json
import csv
import datetime
import sys
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, spearmanr, rankdata
from itertools import combinations

WORKSPACE = "/Users/ameenk/AutoR/runs/20260523_210450/workspace"
RESULTS = os.path.join(WORKSPACE, "results")
NOTES = os.path.join(WORKSPACE, "notes")

RAW_CSV = os.path.join(RESULTS, "h4_size_ablation_pilot_raw.csv")

# Methods to include (no raw baseline)
METHODS = ["temperature_scaling", "monotone_platt", "isotonic_regression", "learned_mlp_head"]
METHOD_LABELS = {
    "temperature_scaling": "Temperature",
    "monotone_platt": "Monotone Platt",
    "isotonic_regression": "Isotonic",
    "learned_mlp_head": "Learned MLP",
}
# Map cal_size to numeric for correlation; 'full' -> 200
SIZE_NUMERIC = {"10": 10, "20": 20, "50": 50, "75": 75, "100": 100, "150": 150, "full": 200}
SMALL_N_THRESHOLD = 75  # sizes considered "small" for post-hoc focus


def load_and_pivot(raw_csv: str) -> dict:
    """
    Returns dict keyed by cal_size (str), each value is a DataFrame with
    columns = methods and rows = blocks (one row per cell x rep combination).
    """
    df = pd.read_csv(raw_csv)
    df = df[df["method"].isin(METHODS)].copy()
    df["block"] = df["dataset_id"] + "_" + df["model_id"] + "_rep" + df["rep"].astype(str)

    pivots = {}
    for size, grp in df.groupby("cal_size"):
        pivot = grp.pivot_table(index="block", columns="method", values="ece_ew", aggfunc="first")
        # Keep only rows with all four methods present
        pivot = pivot.dropna()
        pivots[str(size)] = pivot[METHODS]
    return pivots


def friedman_per_size(pivots: dict) -> list:
    """Run Friedman test per calibration size."""
    rows = []
    for size_str, pivot in sorted(pivots.items(), key=lambda x: SIZE_NUMERIC.get(x[0], 9999)):
        n_blocks = len(pivot)
        groups = [pivot[m].values for m in METHODS]
        stat, pval = friedmanchisquare(*groups)

        # Mean ECE rank per method (lower = better)
        ece_matrix = pivot.values  # shape (n_blocks, 4)
        ranks = np.apply_along_axis(rankdata, axis=1, arr=ece_matrix)  # rank within each block
        mean_ranks = {METHODS[i]: float(np.mean(ranks[:, i])) for i in range(len(METHODS))}
        mlp_mean_rank = mean_ranks["learned_mlp_head"]
        mlp_worst_fraction = float(np.mean(ranks[:, METHODS.index("learned_mlp_head")] == len(METHODS)))

        rows.append({
            "cal_size": size_str,
            "cal_size_numeric": SIZE_NUMERIC.get(size_str, 9999),
            "n_blocks": n_blocks,
            "friedman_statistic": round(float(stat), 4),
            "friedman_pvalue": round(float(pval), 6),
            "significant_at_0.05": bool(pval < 0.05),
            "significant_at_0.01": bool(pval < 0.01),
            "mean_ece_rank_temperature": round(mean_ranks["temperature_scaling"], 3),
            "mean_ece_rank_platt": round(mean_ranks["monotone_platt"], 3),
            "mean_ece_rank_isotonic": round(mean_ranks["isotonic_regression"], 3),
            "mean_ece_rank_mlp": round(mean_ranks["learned_mlp_head"], 3),
            "mlp_worst_fraction": round(mlp_worst_fraction, 3),
            "best_method": METHODS[int(np.argmin(list(mean_ranks.values())))],
        })
    return rows


def pairwise_wilcoxon_mlp(pivots: dict, small_n_sizes: list) -> list:
    """
    For each method pair involving MLP at small-n sizes, run Wilcoxon signed-rank
    test on the ECE difference (pooled across small-n sizes).
    """
    # Collect ECE differences: MLP - other_method, across all (size, block) pairs
    results = []
    comparators = [m for m in METHODS if m != "learned_mlp_head"]

    for size_str in small_n_sizes:
        if size_str not in pivots:
            continue
        pivot = pivots[size_str]
        mlp_vals = pivot["learned_mlp_head"].values
        for other in comparators:
            other_vals = pivot[other].values
            diff = mlp_vals - other_vals  # positive = MLP worse
            if len(diff) < 6:
                stat, pval = float("nan"), float("nan")
                n_pairs = len(diff)
            else:
                try:
                    stat, pval = wilcoxon(diff, alternative="greater")  # H1: MLP ECE > other
                    n_pairs = len(diff)
                except Exception as e:
                    stat, pval = float("nan"), float("nan")
                    n_pairs = len(diff)
            results.append({
                "cal_size": size_str,
                "cal_size_numeric": SIZE_NUMERIC.get(size_str, 9999),
                "comparison": f"mlp_vs_{other}",
                "method_a": "learned_mlp_head",
                "method_b": other,
                "n_pairs": n_pairs,
                "wilcoxon_statistic": round(float(stat), 4) if not np.isnan(stat) else None,
                "wilcoxon_pvalue": round(float(pval), 6) if not np.isnan(pval) else None,
                "significant_at_0.05": bool(pval < 0.05) if not np.isnan(pval) else False,
                "mean_diff": round(float(np.mean(diff)), 6),
                "median_diff": round(float(np.median(diff)), 6),
                "fraction_mlp_worse": round(float(np.mean(diff > 0)), 3),
            })
    return results


def spearman_rank_vs_calsize(friedman_rows: list) -> dict:
    """Spearman correlation between cal_size_numeric and MLP's mean ECE rank."""
    sizes = [r["cal_size_numeric"] for r in friedman_rows]
    mlp_ranks = [r["mean_ece_rank_mlp"] for r in friedman_rows]
    corr, pval = spearmanr(sizes, mlp_ranks)
    return {
        "spearman_rho": round(float(corr), 4),
        "spearman_pvalue": round(float(pval), 6),
        "interpretation": (
            "Negative rho means MLP rank improves (lower rank number = better) as cal_size grows. "
            "Positive rho means MLP rank worsens as cal_size grows."
        ),
        "n_points": len(sizes),
    }


def mlp_rank_distribution(pivots: dict) -> dict:
    """Count how often MLP ranks 1st, 2nd, 3rd, 4th at each cal_size."""
    dist = {}
    for size_str, pivot in sorted(pivots.items(), key=lambda x: SIZE_NUMERIC.get(x[0], 9999)):
        ece_matrix = pivot.values
        ranks = np.apply_along_axis(rankdata, axis=1, arr=ece_matrix)
        mlp_ranks = ranks[:, METHODS.index("learned_mlp_head")].astype(int).tolist()
        counts = {str(r): int(sum(1 for v in mlp_ranks if v == r)) for r in range(1, 5)}
        dist[size_str] = {
            "cal_size_numeric": SIZE_NUMERIC.get(size_str, 9999),
            "n_blocks": len(pivot),
            "rank_counts": counts,
            "fraction_rank_4": round(counts.get("4", 0) / len(pivot), 3),
            "fraction_rank_4_or_3": round((counts.get("4", 0) + counts.get("3", 0)) / len(pivot), 3),
        }
    return dist


def run_all():
    pivots = load_and_pivot(RAW_CSV)
    cal_sizes_sorted = sorted(pivots.keys(), key=lambda x: SIZE_NUMERIC.get(x, 9999))
    small_n_sizes = [s for s in cal_sizes_sorted if SIZE_NUMERIC.get(s, 9999) <= SMALL_N_THRESHOLD]

    # --- Friedman per size ---
    friedman_rows = friedman_per_size(pivots)

    # --- Pairwise Wilcoxon at small n ---
    wilcoxon_rows = pairwise_wilcoxon_mlp(pivots, small_n_sizes)

    # --- Spearman: cal_size vs MLP rank ---
    spearman_result = spearman_rank_vs_calsize(friedman_rows)

    # --- MLP rank distribution ---
    rank_dist = mlp_rank_distribution(pivots)

    # Summary: is MLP significantly last?
    all_small_n_friedman = [r for r in friedman_rows if r["cal_size_numeric"] <= SMALL_N_THRESHOLD]
    n_sig_small = sum(1 for r in all_small_n_friedman if r["significant_at_0.05"])
    n_sig_small_01 = sum(1 for r in all_small_n_friedman if r["significant_at_0.01"])

    all_large_n_friedman = [r for r in friedman_rows if r["cal_size_numeric"] > SMALL_N_THRESHOLD]
    n_sig_large = sum(1 for r in all_large_n_friedman if r["significant_at_0.05"])

    # MLP fraction rank-last at small n pooled
    small_n_wilcoxon_sig = [w for w in wilcoxon_rows if w.get("significant_at_0.05")]

    h4_verdict = {
        "small_n_sizes_tested": small_n_sizes,
        "large_n_sizes_tested": [s for s in cal_sizes_sorted if SIZE_NUMERIC.get(s, 9999) > SMALL_N_THRESHOLD],
        "friedman_significant_at_small_n_p05": n_sig_small,
        "friedman_significant_at_small_n_p01": n_sig_small_01,
        "total_small_n_sizes": len(all_small_n_friedman),
        "friedman_significant_at_large_n_p05": n_sig_large,
        "total_large_n_sizes": len(all_large_n_friedman),
        "spearman_calsize_vs_mlp_rank": spearman_result,
        "wilcoxon_mlp_vs_others_significant_pairs": len(small_n_wilcoxon_sig),
        "total_wilcoxon_pairs_tested": len(wilcoxon_rows),
        "mlp_rank_last_fraction_n10": rank_dist.get("10", {}).get("fraction_rank_4", None),
        "mlp_rank_last_fraction_n20": rank_dist.get("20", {}).get("fraction_rank_4", None),
        "mlp_rank_last_fraction_n50": rank_dist.get("50", {}).get("fraction_rank_4", None),
        "interpretation": (
            "Friedman tests whether method ECE rankings are non-uniform across the 20 blocks "
            "(4 model-dataset cells x 5 reps) at each calibration size. "
            "Wilcoxon signed-rank tests whether MLP ECE is specifically higher than each other method "
            "at small n (alternative=greater). "
            "Spearman rho < 0 indicates MLP rank improves as cal_size increases."
        ),
    }

    result = {
        "generated_at": datetime.datetime.now().isoformat(),
        "stage": "05_experimentation",
        "experiment": "h4_friedman_rank_test",
        "source_csv": "results/h4_size_ablation_pilot_raw.csv",
        "methods_compared": METHODS,
        "small_n_threshold": SMALL_N_THRESHOLD,
        "n_blocks_per_size": 20,
        "friedman_per_calsize": friedman_rows,
        "pairwise_wilcoxon_mlp_vs_others_small_n": wilcoxon_rows,
        "spearman_calsize_vs_mlp_rank": spearman_result,
        "mlp_rank_distribution_per_size": rank_dist,
        "h4_verdict": h4_verdict,
    }

    # Save JSON
    json_path = os.path.join(RESULTS, "h4_friedman_test_results.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save flat CSV of Friedman results
    csv_path = os.path.join(RESULTS, "h4_friedman_test_per_size.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(friedman_rows[0].keys()))
        w.writeheader()
        w.writerows(friedman_rows)

    # Save flat CSV of Wilcoxon results
    wil_csv_path = os.path.join(RESULTS, "h4_wilcoxon_mlp_vs_others.csv")
    with open(wil_csv_path, "w", newline="") as f:
        if wilcoxon_rows:
            w = csv.DictWriter(f, fieldnames=list(wilcoxon_rows[0].keys()))
            w.writeheader()
            w.writerows(wilcoxon_rows)

    return result, friedman_rows, wilcoxon_rows


if __name__ == "__main__":
    log_path = os.path.join(NOTES, "h4_friedman_test.log")

    class Tee:
        def __init__(self, path):
            self.f = open(path, "w")
            self.orig = sys.stdout
        def write(self, s):
            self.orig.write(s)
            self.f.write(s)
        def flush(self):
            self.orig.flush()
            self.f.flush()

    sys.stdout = Tee(log_path)

    print(f"[{datetime.datetime.now().isoformat()}] H4 Friedman rank test")

    result, friedman_rows, wilcoxon_rows = run_all()

    print("\n=== Friedman test results per calibration size ===")
    print(f"{'Cal size':>10} {'n_blocks':>9} {'chi2':>8} {'p-value':>10} {'sig':>5} {'MLP mean rank':>14} {'MLP rank-last':>14}")
    for r in friedman_rows:
        print(f"{r['cal_size']:>10} {r['n_blocks']:>9} {r['friedman_statistic']:>8.3f} "
              f"{r['friedman_pvalue']:>10.4f} {'*' if r['significant_at_0.05'] else '':>5} "
              f"{r['mean_ece_rank_mlp']:>14.3f} {r['mlp_worst_fraction']:>14.3f}")

    print("\n=== Wilcoxon: MLP vs others at small n (n <= 75) ===")
    print(f"{'Cal size':>10} {'Comparison':>35} {'n_pairs':>8} {'stat':>8} {'p-value':>10} {'sig':>5} {'frac MLP worse':>15}")
    for w in wilcoxon_rows:
        print(f"{w['cal_size']:>10} {w['comparison']:>35} {w['n_pairs']:>8} "
              f"{str(w['wilcoxon_statistic'] or 'nan'):>8} {str(w['wilcoxon_pvalue'] or 'nan'):>10} "
              f"{'*' if w.get('significant_at_0.05') else '':>5} {w['fraction_mlp_worse']:>15.3f}")

    print("\n=== Spearman: cal_size vs MLP ECE rank ===")
    sp = result["spearman_calsize_vs_mlp_rank"]
    print(f"  rho = {sp['spearman_rho']:.4f}, p = {sp['spearman_pvalue']:.4f}, n = {sp['n_points']}")

    v = result["h4_verdict"]
    print("\n=== H4 verdict summary ===")
    print(f"  Friedman significant at small n (p<0.05): {v['friedman_significant_at_small_n_p05']}/{v['total_small_n_sizes']} sizes")
    print(f"  Friedman significant at small n (p<0.01): {v['friedman_significant_at_small_n_p01']}/{v['total_small_n_sizes']} sizes")
    print(f"  MLP fraction rank-last: n=10: {v['mlp_rank_last_fraction_n10']}, n=20: {v['mlp_rank_last_fraction_n20']}, n=50: {v['mlp_rank_last_fraction_n50']}")
    print(f"  Wilcoxon significant pairs: {v['wilcoxon_mlp_vs_others_significant_pairs']}/{v['total_wilcoxon_pairs_tested']}")

    sys.stdout.f.close()
    sys.stdout = sys.stdout.orig
    print("Done. Log:", log_path)
