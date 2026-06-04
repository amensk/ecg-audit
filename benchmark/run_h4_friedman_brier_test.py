"""
H4 Friedman + Wilcoxon rank tests on Brier score.

Mirrors run_h4_friedman_test.py but uses the `brier` column instead of `ece_ew`.
Brier score is a proper scoring rule, so confirming the ECE-based H4 finding
holds under Brier strengthens the evidence that MLP overfitting at small n is
metric-independent.
"""

import json
import csv
import datetime
import sys
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, spearmanr, rankdata

WORKSPACE = "/Users/ameenk/AutoR/runs/20260523_210450/workspace"
RESULTS = os.path.join(WORKSPACE, "results")
NOTES = os.path.join(WORKSPACE, "notes")

RAW_CSV = os.path.join(RESULTS, "h4_size_ablation_pilot_raw.csv")

METHODS = ["temperature_scaling", "monotone_platt", "isotonic_regression", "learned_mlp_head"]
SIZE_NUMERIC = {"10": 10, "20": 20, "50": 50, "75": 75, "100": 100, "150": 150, "full": 200}
SMALL_N_THRESHOLD = 75


def load_and_pivot(raw_csv, metric):
    df = pd.read_csv(raw_csv)
    df = df[df["method"].isin(METHODS)].copy()
    df["block"] = df["dataset_id"] + "_" + df["model_id"] + "_rep" + df["rep"].astype(str)
    pivots = {}
    for size, grp in df.groupby("cal_size"):
        pivot = grp.pivot_table(index="block", columns="method", values=metric, aggfunc="first")
        pivot = pivot.dropna()
        pivots[str(size)] = pivot[METHODS]
    return pivots


def friedman_per_size(pivots):
    rows = []
    for size_str, pivot in sorted(pivots.items(), key=lambda x: SIZE_NUMERIC.get(x[0], 9999)):
        n_blocks = len(pivot)
        groups = [pivot[m].values for m in METHODS]
        stat, pval = friedmanchisquare(*groups)
        ece_matrix = pivot.values
        ranks = np.apply_along_axis(rankdata, axis=1, arr=ece_matrix)
        mean_ranks = {METHODS[i]: float(np.mean(ranks[:, i])) for i in range(len(METHODS))}
        mlp_worst_fraction = float(np.mean(ranks[:, METHODS.index("learned_mlp_head")] == len(METHODS)))
        rows.append({
            "cal_size": size_str,
            "cal_size_numeric": SIZE_NUMERIC.get(size_str, 9999),
            "n_blocks": n_blocks,
            "friedman_statistic": round(float(stat), 4),
            "friedman_pvalue": round(float(pval), 6),
            "significant_at_0.05": bool(pval < 0.05),
            "significant_at_0.01": bool(pval < 0.01),
            "mean_brier_rank_temperature": round(mean_ranks["temperature_scaling"], 3),
            "mean_brier_rank_platt": round(mean_ranks["monotone_platt"], 3),
            "mean_brier_rank_isotonic": round(mean_ranks["isotonic_regression"], 3),
            "mean_brier_rank_mlp": round(mean_ranks["learned_mlp_head"], 3),
            "mlp_worst_fraction": round(mlp_worst_fraction, 3),
            "best_method": METHODS[int(np.argmin(list(mean_ranks.values())))],
        })
    return rows


def pairwise_wilcoxon_mlp(pivots, small_n_sizes):
    results = []
    comparators = [m for m in METHODS if m != "learned_mlp_head"]
    for size_str in small_n_sizes:
        if size_str not in pivots:
            continue
        pivot = pivots[size_str]
        mlp_vals = pivot["learned_mlp_head"].values
        for other in comparators:
            other_vals = pivot[other].values
            diff = mlp_vals - other_vals
            if len(diff) < 6:
                stat, pval = float("nan"), float("nan")
                n_pairs = len(diff)
            else:
                try:
                    stat, pval = wilcoxon(diff, alternative="greater")
                    n_pairs = len(diff)
                except Exception:
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
                "fraction_mlp_worse": round(float(np.mean(diff > 0)), 3),
            })
    return results


def run_all():
    pivots = load_and_pivot(RAW_CSV, "brier")
    cal_sizes_sorted = sorted(pivots.keys(), key=lambda x: SIZE_NUMERIC.get(x, 9999))
    small_n_sizes = [s for s in cal_sizes_sorted if SIZE_NUMERIC.get(s, 9999) <= SMALL_N_THRESHOLD]

    friedman_rows = friedman_per_size(pivots)
    wilcoxon_rows = pairwise_wilcoxon_mlp(pivots, small_n_sizes)

    sizes = [r["cal_size_numeric"] for r in friedman_rows]
    mlp_ranks = [r["mean_brier_rank_mlp"] for r in friedman_rows]
    corr, pval = spearmanr(sizes, mlp_ranks)
    spearman_result = {
        "spearman_rho": round(float(corr), 4),
        "spearman_pvalue": round(float(pval), 6),
        "n_points": len(sizes),
    }

    # Rank distribution
    rank_dist = {}
    for size_str, pivot in sorted(pivots.items(), key=lambda x: SIZE_NUMERIC.get(x[0], 9999)):
        ece_matrix = pivot.values
        ranks = np.apply_along_axis(rankdata, axis=1, arr=ece_matrix)
        mlp_ranks_list = ranks[:, METHODS.index("learned_mlp_head")].astype(int).tolist()
        counts = {str(r): int(sum(1 for v in mlp_ranks_list if v == r)) for r in range(1, 5)}
        rank_dist[size_str] = {
            "cal_size_numeric": SIZE_NUMERIC.get(size_str, 9999),
            "n_blocks": len(pivot),
            "rank_counts": counts,
            "fraction_rank_4": round(counts.get("4", 0) / len(pivot), 3),
        }

    all_small_n = [r for r in friedman_rows if r["cal_size_numeric"] <= SMALL_N_THRESHOLD]
    n_sig_small_05 = sum(1 for r in all_small_n if r["significant_at_0.05"])
    n_sig_small_01 = sum(1 for r in all_small_n if r["significant_at_0.01"])
    wil_sig = [w for w in wilcoxon_rows if w.get("significant_at_0.05")]

    result = {
        "generated_at": datetime.datetime.now().isoformat(),
        "stage": "05_experimentation",
        "experiment": "h4_friedman_brier_test",
        "metric": "brier",
        "source_csv": "results/h4_size_ablation_pilot_raw.csv",
        "small_n_threshold": SMALL_N_THRESHOLD,
        "n_blocks_per_size": 20,
        "friedman_per_calsize": friedman_rows,
        "pairwise_wilcoxon_mlp_vs_others_small_n": wilcoxon_rows,
        "spearman_calsize_vs_mlp_rank": spearman_result,
        "mlp_rank_distribution_per_size": rank_dist,
        "h4_brier_verdict": {
            "friedman_significant_at_small_n_p05": n_sig_small_05,
            "friedman_significant_at_small_n_p01": n_sig_small_01,
            "total_small_n_sizes": len(all_small_n),
            "wilcoxon_significant_pairs": len(wil_sig),
            "total_wilcoxon_pairs": len(wilcoxon_rows),
            "mlp_rank_last_fraction_n10": rank_dist.get("10", {}).get("fraction_rank_4"),
            "mlp_rank_last_fraction_n20": rank_dist.get("20", {}).get("fraction_rank_4"),
            "mlp_rank_last_fraction_n50": rank_dist.get("50", {}).get("fraction_rank_4"),
            "spearman_rho": spearman_result["spearman_rho"],
            "spearman_pvalue": spearman_result["spearman_pvalue"],
        },
    }

    json_path = os.path.join(RESULTS, "h4_friedman_brier_test_results.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    csv_path = os.path.join(RESULTS, "h4_friedman_brier_test_per_size.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(friedman_rows[0].keys()))
        w.writeheader()
        w.writerows(friedman_rows)

    wil_csv = os.path.join(RESULTS, "h4_wilcoxon_brier_mlp_vs_others.csv")
    with open(wil_csv, "w", newline="") as f:
        if wilcoxon_rows:
            w = csv.DictWriter(f, fieldnames=list(wilcoxon_rows[0].keys()))
            w.writeheader()
            w.writerows(wilcoxon_rows)

    return result, friedman_rows, wilcoxon_rows, spearman_result


if __name__ == "__main__":
    log_path = os.path.join(NOTES, "h4_friedman_brier_test.log")

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
    print(f"[{datetime.datetime.now().isoformat()}] H4 Friedman Brier test")

    result, friedman_rows, wilcoxon_rows, spearman_result = run_all()

    print("\n=== Friedman test (Brier) per calibration size ===")
    print(f"{'Cal size':>10} {'chi2':>8} {'p-value':>10} {'sig':>5} {'MLP mean rank':>14} {'MLP rank-last':>14}")
    for r in friedman_rows:
        print(f"{r['cal_size']:>10} {r['friedman_statistic']:>8.3f} {r['friedman_pvalue']:>10.4f} "
              f"{'*' if r['significant_at_0.05'] else '':>5} "
              f"{r['mean_brier_rank_mlp']:>14.3f} {r['mlp_worst_fraction']:>14.3f}")

    print("\n=== Wilcoxon Brier: MLP vs others at small n (n <= 75) ===")
    for w in wilcoxon_rows:
        print(f"  n={w['cal_size']:>4} {w['comparison']:>35} p={str(w['wilcoxon_pvalue'] or 'nan'):>10} "
              f"{'*' if w.get('significant_at_0.05') else ''}")

    print(f"\nSpearman (cal_size vs MLP Brier rank): rho={spearman_result['spearman_rho']:.4f}, "
          f"p={spearman_result['spearman_pvalue']:.4f}")

    v = result["h4_brier_verdict"]
    print(f"\nFriedman sig at small n (p<0.05): {v['friedman_significant_at_small_n_p05']}/{v['total_small_n_sizes']}")
    print(f"MLP rank-last: n=10: {v['mlp_rank_last_fraction_n10']}, n=20: {v['mlp_rank_last_fraction_n20']}, n=50: {v['mlp_rank_last_fraction_n50']}")
    print(f"Wilcoxon sig pairs: {v['wilcoxon_significant_pairs']}/{v['total_wilcoxon_pairs']}")

    sys.stdout.f.close()
    sys.stdout = sys.stdout.orig
    print("Done. Log:", log_path)
