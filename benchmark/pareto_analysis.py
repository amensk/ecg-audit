#!/usr/bin/env python3
"""
Analysis 2: AUROC-ECE Pareto Frontier + Composite Score
- Load benchmark results (skip ECGFM-KED CODE-15% partial)
- Compute Pareto frontier per dataset
- Composite score C(alpha) = alpha*AUROC + (1-alpha)*(1-ECE_normalized)
- Save: results/pareto_analysis.json
- Figure: writing/figures/fig7_pareto_frontier.pdf + .png
"""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
RESULTS = WS / "results"
FIGS = WS / "writing" / "figures"
FIGS.mkdir(exist_ok=True)

COLORS = {
    "MERL":      "#e74c3c",
    "ST-MEM":    "#2980b9",
    "ECGFM-KED": "#e67e22",
    "ECG-FM":    "#8e44ad",
    "S4D":       "#27ae60",
}
MARKERS = {
    "MERL":      "o",
    "ST-MEM":    "s",
    "ECGFM-KED": "D",
    "ECG-FM":    "^",
    "S4D":       "P",
}
PRETRAIN = {
    "MERL":      "contrastive",
    "ST-MEM":    "masked-recon.",
    "ECGFM-KED": "knowledge-enh.",
    "ECG-FM":    "contrastive+gen.",
    "S4D":       "supervised",
}
SKIP_PARTIAL = {("ECGFM-KED", "CODE-15%")}  # partial/invalid


def pareto_front(points):
    """Return indices of Pareto-optimal points (maximize AUROC, minimize ECE)."""
    pts = np.array(points)  # shape (N, 2): [AUROC, ECE]
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j.AUROC >= i.AUROC and j.ECE <= i.ECE (strictly better on one)
            if pts[j, 0] >= pts[i, 0] and pts[j, 1] <= pts[i, 1]:
                if pts[j, 0] > pts[i, 0] or pts[j, 1] < pts[i, 1]:
                    is_pareto[i] = False
                    break
    return np.where(is_pareto)[0]


def composite_score(auroc, ece, ece_max, alpha):
    """C(alpha) = alpha*AUROC + (1-alpha)*(1 - ECE/ECE_max)"""
    ece_norm = ece / ece_max if ece_max > 0 else 0.0
    return alpha * auroc + (1 - alpha) * (1 - ece_norm)


def main():
    with open(RESULTS / "full_benchmark_results.json") as f:
        data = json.load(f)

    # Organize by dataset
    datasets = {}
    for r in data["results"]:
        model = r["model_name"]
        ds = r["dataset_name"]
        key = (model, ds)
        if key in SKIP_PARTIAL:
            print(f"Skipping partial: {key}")
            continue
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append({
            "model": model,
            "auroc": r["phase_a"]["macro_auroc"],
            "ece": r["phase_a"]["macro_ece_ew"],
            "iso_ece": r["phase_b"]["isotonic"]["macro_ece_ew"],
            "platt_ece": r["phase_b"]["platt"]["macro_ece_ew"],
        })

    alphas = [0.5, 0.7, 0.9]
    pareto_results = {}

    for ds, cells in datasets.items():
        aurocs = [c["auroc"] for c in cells]
        eces = [c["ece"] for c in cells]
        iso_eces = [c["iso_ece"] for c in cells]
        models = [c["model"] for c in cells]

        ece_max = max(eces)
        iso_ece_max = max(iso_eces)

        # Pareto frontier
        points = list(zip(aurocs, eces))
        pareto_idx = pareto_front(points)

        # Composite scores
        composite_raw = {}
        composite_iso = {}
        for alpha in alphas:
            composite_raw[alpha] = [composite_score(a, e, ece_max, alpha)
                                    for a, e in zip(aurocs, eces)]
            composite_iso[alpha] = [composite_score(a, e, iso_ece_max, alpha)
                                    for a, e in zip(aurocs, iso_eces)]

        pareto_results[ds] = {
            "models": models,
            "auroc": aurocs,
            "ece_raw": eces,
            "ece_iso": iso_eces,
            "ece_max_raw": ece_max,
            "ece_max_iso": iso_ece_max,
            "pareto_indices": pareto_idx.tolist(),
            "pareto_models": [models[i] for i in pareto_idx],
            "composite_raw": {str(a): composite_raw[a] for a in alphas},
            "composite_iso": {str(a): composite_iso[a] for a in alphas},
            "composite_raw_winner_alpha07": models[np.argmax(composite_raw[0.7])],
            "composite_iso_winner_alpha07": models[np.argmax(composite_iso[0.7])],
        }

        print(f"\n=== {ds} ===")
        print(f"  Pareto models: {[models[i] for i in pareto_idx]}")
        for alpha in alphas:
            winner = models[np.argmax(composite_raw[alpha])]
            iso_winner = models[np.argmax(composite_iso[alpha])]
            print(f"  α={alpha}: raw winner={winner}, iso winner={iso_winner}")

    # Save JSON
    with open(RESULTS / "pareto_analysis.json", "w") as f:
        json.dump(pareto_results, f, indent=2)
    print(f"\nSaved: {RESULTS}/pareto_analysis.json")

    # Figure: 3-panel scatter + Pareto front
    ds_order = ["PTB-XL", "CPSC2018", "CODE-15%"]
    ds_available = [d for d in ds_order if d in pareto_results]
    n_panels = len(ds_available)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    for ax, ds in zip(axes, ds_available):
        pr = pareto_results[ds]
        models = pr["models"]
        aurocs = pr["auroc"]
        eces = pr["ece_raw"]
        pareto_idx = pr["pareto_indices"]

        # Plot all models
        for i, (m, a, e) in enumerate(zip(models, aurocs, eces)):
            label_str = f"{m} ({PRETRAIN[m]})"
            ax.scatter(a, e * 100, color=COLORS[m], marker=MARKERS[m],
                       s=140, zorder=5, label=label_str,
                       edgecolors="black", linewidth=0.5)
            offset = (4, 4)
            if m == "ST-MEM":
                offset = (4, -10)
            ax.annotate(m, (a, e * 100), textcoords="offset points",
                        xytext=offset, fontsize=7.5)

        # Draw Pareto front
        pfront = sorted([(aurocs[i], eces[i] * 100) for i in pareto_idx])
        if len(pfront) >= 2:
            px, py = zip(*pfront)
            ax.plot(px, py, "k--", linewidth=1.5, alpha=0.6, label="Pareto front", zorder=3)

        # Composite score contour (alpha=0.7)
        # Draw iso-composite lines
        a_range = np.linspace(min(aurocs) - 0.02, max(aurocs) + 0.02, 200)
        ece_max = pr["ece_max_raw"]
        for c_val, style in [(0.85, ":"), (0.80, "--")]:
            # C = 0.7*a + 0.3*(1 - e/ece_max) => e/ece_max = 1 - (C - 0.7*a)/0.3
            e_vals = ece_max * (1 - (c_val - 0.7 * a_range) / 0.3) * 100
            valid = (e_vals >= 0) & (e_vals <= ece_max * 100 * 1.5)
            if valid.sum() > 1:
                ax.plot(a_range[valid], e_vals[valid], color="gray",
                        linestyle=style, alpha=0.4, linewidth=0.8)

        ax.set_xlabel("Macro AUROC", fontsize=11)
        ax.set_ylabel("Macro ECE × 100", fontsize=11)
        ax.set_title(ds, fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
        ax.grid(True, alpha=0.3)

        # Annotate winner
        winner = pr["composite_raw_winner_alpha07"]
        ax.text(0.97, 0.97, f"α=0.7 winner:\n{winner}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(
        "AUROC–ECE Pareto Frontier per Dataset\n"
        "Dashed line = Pareto front; grey contours = composite score C(α=0.7) iso-lines",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    for ext in ["pdf", "png"]:
        path = FIGS / f"fig7_pareto_frontier.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
