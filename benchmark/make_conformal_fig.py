#!/usr/bin/env python3
"""Figure: split-conformal coverage vs average set size for MI and AF endpoints.

Left column: empirical coverage vs target (validity check) by method/alpha.
Right column: average prediction-set size by method (efficiency), grouped by
alpha, showing how recalibration changes conformal efficiency at fixed coverage.

Reads results/conformal.csv (produced by run_conformal.py). Writes
writing/figures/fig_conformal.pdf and .png. Does not touch manuscript files.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
FIGDIR = WS / "writing" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(WS / "results" / "conformal.csv")

ENDPOINTS = ["MI", "AF"]
METHODS = ["raw", "platt", "isotonic", "temperature"]
ALPHAS = [0.1, 0.2]
MCOLOR = {"raw": "#444444", "platt": "#1f77b4",
          "isotonic": "#d62728", "temperature": "#2ca02c"}
MMARK = {"raw": "o", "platt": "s", "isotonic": "^", "temperature": "D"}

fig, axes = plt.subplots(len(ENDPOINTS), 2, figsize=(11, 8))

for r, ep in enumerate(ENDPOINTS):
    sub = df[df.endpoint == ep]

    # ---- LEFT: coverage vs avg set size scatter (per model x dataset x alpha) ----
    axL = axes[r, 0]
    for alpha in ALPHAS:
        tgt = 1.0 - alpha
        axL.axvline(tgt, color="grey", ls="--", lw=1, alpha=0.7)
        for method in METHODS:
            s = sub[(sub.alpha == alpha) & (sub.method == method)]
            axL.scatter(s.coverage, s.avg_set_size,
                        c=MCOLOR[method], marker=MMARK[method], s=42,
                        alpha=0.8, edgecolors="white", linewidths=0.5,
                        label=method if alpha == ALPHAS[0] else None)
    axL.set_xlabel("Empirical marginal coverage")
    axL.set_ylabel("Average prediction-set size")
    axL.set_title(f"{ep}: coverage vs set size\n(dashed = targets 0.90 / 0.80; "
                  f"points = each model$\\times$dataset$\\times\\alpha$)")
    axL.grid(alpha=0.25)
    if r == 0:
        axL.legend(title="recal. method", fontsize=8, loc="best")

    # ---- RIGHT: mean avg set size by method, grouped by alpha (efficiency) ----
    axR = axes[r, 1]
    width = 0.18
    x = np.arange(len(ALPHAS))
    for j, method in enumerate(METHODS):
        means, errs = [], []
        for alpha in ALPHAS:
            s = sub[(sub.alpha == alpha) & (sub.method == method)]
            means.append(s.avg_set_size.mean())
            errs.append(s.avg_set_size.std(ddof=1) / np.sqrt(max(len(s), 1)))
        axR.bar(x + (j - 1.5) * width, means, width,
                yerr=errs, capsize=3, color=MCOLOR[method],
                label=method, edgecolor="black", linewidth=0.4)
    axR.set_xticks(x)
    axR.set_xticklabels([f"$\\alpha$={a}\n(target {1-a:.2f})" for a in ALPHAS])
    axR.set_ylabel("Mean average set size (±SE)")
    axR.set_title(f"{ep}: conformal efficiency by method\n"
                  f"(at fixed coverage; lower = more efficient)")
    axR.grid(alpha=0.25, axis="y")
    if r == 0:
        axR.legend(fontsize=8, ncol=2, loc="best")

fig.suptitle("Split-conformal prediction (one-vs-rest): coverage validity and "
             "efficiency under recalibration", fontsize=12, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(FIGDIR / "fig_conformal.pdf", bbox_inches="tight")
fig.savefig(FIGDIR / "fig_conformal.png", dpi=150, bbox_inches="tight")
print(f"Wrote {FIGDIR/'fig_conformal.pdf'} and .png")
