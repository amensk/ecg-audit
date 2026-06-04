"""
H4 Nemenyi post-hoc test: pairwise significance for all method pairs at each calibration size.

Uses the Nemenyi critical difference (CD) approach from Demsar (2006):
  CD = q_alpha * sqrt(k*(k+1) / (6*N))

where:
  k = 4 (number of methods)
  N = number of blocks per size (model x dataset x rep combinations)
  q_alpha values from Demsar Table 5 (k=4, studentized range / sqrt(2))

Also runs pairwise two-sided Wilcoxon signed-rank tests on raw ECE and Brier values
per size for robustness.

Input:  results/h4_size_ablation_pilot_raw.csv
Output:
  results/h4_nemenyi_test_results.json
  results/h4_nemenyi_pairwise_ece.csv   (k*(k-1)/2 pairs x 7 sizes)
  results/h4_nemenyi_pairwise_brier.csv
  notes/h4_nemenyi_test.log
"""

import json
import logging
import os
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
RAW_CSV = BASE / "results/h4_size_ablation_pilot_raw.csv"
OUT_JSON = BASE / "results/h4_nemenyi_test_results.json"
OUT_ECE_CSV = BASE / "results/h4_nemenyi_pairwise_ece.csv"
OUT_BRIER_CSV = BASE / "results/h4_nemenyi_pairwise_brier.csv"
LOG_PATH = BASE / "notes/h4_nemenyi_test.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH, mode="w")],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nemenyi critical-difference constants (Demsar 2006, Table 5, k=4)
# q_alpha are the critical values q such that CD = q * sqrt(k*(k+1)/(6*N))
# ---------------------------------------------------------------------------
Q_ALPHA = {0.10: 2.291, 0.05: 2.569, 0.01: 3.102}
K = 4  # number of methods


def critical_difference(n_blocks: int, alpha: float) -> float:
    """Nemenyi CD for k=4 methods and given number of blocks."""
    return Q_ALPHA[alpha] * np.sqrt(K * (K + 1) / (6 * n_blocks))


def rank_methods_in_block(vals: dict[str, float]) -> dict[str, float]:
    """Assign ranks (1=best=lowest ECE/Brier) to methods within one block."""
    sorted_methods = sorted(vals, key=lambda m: vals[m])
    return {m: i + 1 for i, m in enumerate(sorted_methods)}


def nemenyi_pairwise(df_size: pd.DataFrame, metric: str) -> dict:
    """
    Compute Nemenyi CD-based pairwise significance for one calibration size.

    Returns dict with keys:
      - mean_ranks: dict method -> mean rank
      - n_blocks: int
      - CD_05: float
      - CD_10: float
      - pairs: list of {method_a, method_b, rank_diff, sig_05, sig_10,
                         wilcoxon_p, wilcoxon_sig_05}
    """
    methods = sorted(df_size["method"].unique())

    # Build blocks: (dataset_id, model_id, rep) -> {method -> metric_value}
    blocks: dict = {}
    for _, row in df_size.iterrows():
        key = (row["dataset_id"], row["model_id"], row["rep"])
        if key not in blocks:
            blocks[key] = {}
        blocks[key][row["method"]] = row[metric]

    # Keep only complete blocks (all 4 methods present)
    complete_blocks = {k: v for k, v in blocks.items() if len(v) == K}
    n_blocks = len(complete_blocks)
    log.info(f"  metric={metric} n_blocks={n_blocks} (complete blocks with all 4 methods)")

    if n_blocks < 2:
        return {"error": "insufficient blocks", "n_blocks": n_blocks}

    # Rank within each block (1=best)
    block_ranks: list[dict] = []
    for block_vals in complete_blocks.values():
        block_ranks.append(rank_methods_in_block(block_vals))

    # Mean rank per method
    mean_ranks = {
        m: np.mean([br[m] for br in block_ranks if m in br]) for m in methods
    }

    cd_05 = critical_difference(n_blocks, 0.05)
    cd_10 = critical_difference(n_blocks, 0.10)

    log.info(f"  CD(0.05)={cd_05:.3f}  CD(0.10)={cd_10:.3f}")
    for m, r in sorted(mean_ranks.items(), key=lambda x: x[1]):
        log.info(f"  mean rank {m}: {r:.3f}")

    # Pairwise comparisons
    pairs = []
    for m_a, m_b in combinations(methods, 2):
        rank_diff = abs(mean_ranks[m_a] - mean_ranks[m_b])
        sig_05 = rank_diff > cd_05
        sig_10 = rank_diff > cd_10

        # Wilcoxon signed-rank on raw metric values (two-sided, Bonferroni k*(k-1)/2=6)
        vals_a = [complete_blocks[k][m_a] for k in complete_blocks]
        vals_b = [complete_blocks[k][m_b] for k in complete_blocks]
        diffs = [a - b for a, b in zip(vals_a, vals_b)]
        # skip if all differences are zero
        if all(d == 0 for d in diffs):
            w_p = 1.0
        else:
            try:
                _, w_p = wilcoxon(vals_a, vals_b, alternative="two-sided")
            except Exception:
                w_p = float("nan")

        # Bonferroni-corrected threshold: 0.05 / 6 = 0.00833
        w_sig_bonf = (not np.isnan(w_p)) and (w_p < 0.00833)

        higher_method = m_a if mean_ranks[m_a] > mean_ranks[m_b] else m_b
        lower_method = m_b if mean_ranks[m_a] > mean_ranks[m_b] else m_a

        log.info(
            f"  {m_a} vs {m_b}: rank_diff={rank_diff:.3f} "
            f"sig05_CD={sig_05} sig10_CD={sig_10} "
            f"wilcoxon_p={w_p:.4f} sig_bonf={w_sig_bonf}"
        )

        pairs.append({
            "method_a": m_a,
            "method_b": m_b,
            "mean_rank_a": mean_ranks[m_a],
            "mean_rank_b": mean_ranks[m_b],
            "rank_diff": rank_diff,
            "higher_rank_method": higher_method,
            "lower_rank_method": lower_method,
            "CD_05": cd_05,
            "CD_10": cd_10,
            "sig_CD_05": sig_05,
            "sig_CD_10": sig_10,
            "wilcoxon_p": round(w_p, 6) if not np.isnan(w_p) else None,
            "wilcoxon_sig_bonf": w_sig_bonf,
        })

    return {
        "n_blocks": n_blocks,
        "mean_ranks": mean_ranks,
        "CD_05": cd_05,
        "CD_10": cd_10,
        "pairs": pairs,
    }


def main():
    log.info("=== H4 Nemenyi Post-Hoc Test ===")
    log.info(f"Input: {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)
    log.info(f"Loaded {len(df)} rows; columns: {list(df.columns)}")

    cal_sizes = sorted(df["cal_size"].unique())
    log.info(f"Calibration sizes: {cal_sizes}")

    results = {
        "generated_at": datetime.now().isoformat(),
        "stage": "05_experimentation",
        "experiment": "h4_nemenyi_post_hoc",
        "method": "Nemenyi_critical_difference",
        "k_methods": K,
        "q_alpha_table": Q_ALPHA,
        "metrics": ["ece_ew", "brier"],
        "description": (
            "Nemenyi post-hoc pairwise significance for H4 ablation. "
            "CD = q_alpha * sqrt(k*(k+1)/(6*N)) with k=4, N=blocks per size. "
            "Also reports two-sided Wilcoxon signed-rank with Bonferroni correction "
            "(threshold 0.05/6 = 0.00833 for 6 pairs)."
        ),
        "sizes": {},
    }

    pairwise_rows_ece = []
    pairwise_rows_brier = []

    for size in cal_sizes:
        df_size = df[df["cal_size"] == size]
        log.info(f"\n--- cal_size={size}, n_rows={len(df_size)} ---")

        for metric, row_store in [("ece_ew", pairwise_rows_ece), ("brier", pairwise_rows_brier)]:
            log.info(f"  Running Nemenyi for metric={metric}")
            res = nemenyi_pairwise(df_size, metric)
            if "error" in res:
                log.warning(f"  Skipped: {res['error']}")
                continue

            results["sizes"].setdefault(str(size), {})[metric] = res

            for pair in res["pairs"]:
                row_store.append({
                    "cal_size": size,
                    "method_a": pair["method_a"],
                    "method_b": pair["method_b"],
                    "mean_rank_a": round(pair["mean_rank_a"], 3),
                    "mean_rank_b": round(pair["mean_rank_b"], 3),
                    "rank_diff": round(pair["rank_diff"], 3),
                    "higher_rank_method": pair["higher_rank_method"],
                    "CD_05": round(pair["CD_05"], 3),
                    "CD_10": round(pair["CD_10"], 3),
                    "sig_CD_05": pair["sig_CD_05"],
                    "sig_CD_10": pair["sig_CD_10"],
                    "wilcoxon_p": pair["wilcoxon_p"],
                    "wilcoxon_sig_bonf": pair["wilcoxon_sig_bonf"],
                })

    # Count significant pairs across all sizes
    for metric, row_store, label in [
        ("ece_ew", pairwise_rows_ece, "ECE"),
        ("brier", pairwise_rows_brier, "Brier"),
    ]:
        sig_cd05 = sum(r["sig_CD_05"] for r in row_store)
        sig_wilcox = sum(r["wilcoxon_sig_bonf"] for r in row_store)
        total = len(row_store)
        log.info(
            f"\n{label} summary: {sig_cd05}/{total} pairs significant by CD(0.05); "
            f"{sig_wilcox}/{total} by Wilcoxon-Bonferroni"
        )

    # MLP-specific summary
    log.info("\n=== MLP vs Others (ECE, CD 0.05) ===")
    mlp_rows = [r for r in pairwise_rows_ece if "learned_mlp_head" in (r["method_a"], r["method_b"])]
    for size in cal_sizes:
        size_rows = [r for r in mlp_rows if r["cal_size"] == size]
        sig_count = sum(r["sig_CD_05"] for r in size_rows)
        log.info(f"  n={size}: MLP sig-different from {sig_count}/{len(size_rows)} other methods")

    # Add summary to results
    results["summary"] = {
        "ece_sig_cd05_count": sum(r["sig_CD_05"] for r in pairwise_rows_ece),
        "ece_sig_wilcoxon_bonf_count": sum(r["wilcoxon_sig_bonf"] for r in pairwise_rows_ece),
        "brier_sig_cd05_count": sum(r["sig_CD_05"] for r in pairwise_rows_brier),
        "brier_sig_wilcoxon_bonf_count": sum(r["wilcoxon_sig_bonf"] for r in pairwise_rows_brier),
        "total_pairs_tested": len(pairwise_rows_ece),
        "n_sizes": len(cal_sizes),
        "n_pairs_per_size": K * (K - 1) // 2,
    }

    # Save outputs
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nSaved: {OUT_JSON}")

    pd.DataFrame(pairwise_rows_ece).to_csv(OUT_ECE_CSV, index=False)
    log.info(f"Saved: {OUT_ECE_CSV} ({len(pairwise_rows_ece)} rows)")

    pd.DataFrame(pairwise_rows_brier).to_csv(OUT_BRIER_CSV, index=False)
    log.info(f"Saved: {OUT_BRIER_CSV} ({len(pairwise_rows_brier)} rows)")

    log.info(f"\nLog: {LOG_PATH}")
    log.info("=== Done ===")

    return results


if __name__ == "__main__":
    main()
