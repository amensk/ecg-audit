#!/usr/bin/env python3
"""
Analysis 5: Calibration-Set Size Curves
- Load h4_size_ablation_pilot_summary.csv
- Plot ECE vs log(n) with std bands for each method
- Fit power-law extrapolation to n=500 and n=1000
- PTB-XL panel + CODE-15% panel
- Save: writing/figures/fig10_calsize_curves.pdf + .png
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
RESULTS = WS / "results"
FIGS = WS / "writing" / "figures"
FIGS.mkdir(exist_ok=True)

METHOD_COLORS = {
    "isotonic_regression": "#2980b9",
    "monotone_platt":      "#e74c3c",
    "learned_mlp_head":    "#8e44ad",
    "temperature_scaling": "#27ae60",
}
METHOD_LABELS = {
    "isotonic_regression": "Isotonic regression",
    "monotone_platt":      "Platt scaling",
    "learned_mlp_head":    "MLP head",
    "temperature_scaling": "Temperature scaling",
}
METHOD_LS = {
    "isotonic_regression": "-",
    "monotone_platt":      "--",
    "learned_mlp_head":    ":",
    "temperature_scaling": "-.",
}


def power_law(x, a, b):
    """ECE ~ a * n^(-b)"""
    return a * np.power(x, -b)


def fit_and_extrapolate(ns, eces, extrap_ns):
    """Fit power law and extrapolate."""
    try:
        valid = ~np.isnan(eces) & (np.array(ns) > 0) & (np.array(eces) > 0)
        if valid.sum() < 3:
            return None, None
        popt, _ = curve_fit(power_law, np.array(ns)[valid], np.array(eces)[valid],
                            p0=[0.2, 0.3], maxfev=5000)
        extrap_vals = power_law(np.array(extrap_ns), *popt)
        return extrap_vals, popt
    except Exception as e:
        print(f"  Power law fit failed: {e}")
        return None, None


def main():
    df = pd.read_csv(RESULTS / "h4_size_ablation_pilot_summary.csv")
    print("Columns:", df.columns.tolist())
    print(df.head(3))

    datasets = df["dataset_name"].unique()
    ds_order = ["PTB-XL", "CODE-15%"]
    ds_available = [d for d in ds_order if d in datasets]

    methods = df["method"].unique().tolist()
    extrap_ns = [500, 1000]

    # Deduplicate models per dataset (use ST-MEM probe as primary)
    model_priority = "ST-MEM probe"

    fig, axes = plt.subplots(1, len(ds_available), figsize=(6.5 * len(ds_available), 5.5))
    if len(ds_available) == 1:
        axes = [axes]

    crossover_info = {}

    for ax, ds in zip(axes, ds_available):
        df_ds = df[df["dataset_name"] == ds].copy()
        # Use ST-MEM probe if available, else first model
        models_in_ds = df_ds["model_name"].unique()
        if "ST-MEM probe" in models_in_ds:
            df_ds = df_ds[df_ds["model_name"] == "ST-MEM probe"]
        else:
            df_ds = df_ds[df_ds["model_name"] == models_in_ds[0]]

        # Map cal_size to numeric
        df_ds = df_ds.copy()
        df_ds["n_numeric"] = pd.to_numeric(df_ds["effective_n"], errors="coerce")
        df_ds = df_ds[df_ds["n_numeric"].notna()]
        df_ds = df_ds.sort_values("n_numeric")

        # Track crossover between isotonic and platt
        iso_interp = {}
        platt_interp = {}

        for method in methods:
            dm = df_ds[df_ds["method"] == method].groupby("n_numeric").agg(
                ece_mean=("ece_ew_mean", "mean"),
                ece_std=("ece_ew_std", "mean"),
            ).reset_index()

            if dm.empty:
                continue

            ns = dm["n_numeric"].values
            ece_mean = dm["ece_mean"].values
            ece_std = dm["ece_std"].values

            color = METHOD_COLORS.get(method, "gray")
            ls = METHOD_LS.get(method, "-")
            label = METHOD_LABELS.get(method, method)

            # Plot with std bands
            ax.plot(ns, ece_mean, color=color, linestyle=ls, linewidth=2,
                    marker="o", markersize=5, label=label, zorder=5)
            ax.fill_between(ns, ece_mean - ece_std, ece_mean + ece_std,
                            color=color, alpha=0.12)

            # Power-law extrapolation
            extrap_vals, popt = fit_and_extrapolate(ns.tolist(), ece_mean.tolist(), extrap_ns)
            if extrap_vals is not None:
                ax.plot(extrap_ns, extrap_vals, color=color, linestyle=ls,
                        marker="*", markersize=10, alpha=0.6, zorder=4)
                # Add "extrapolated" text
                for n_e, v_e in zip(extrap_ns, extrap_vals):
                    ax.annotate(f"extrap.\n{v_e:.3f}", (n_e, v_e),
                                textcoords="offset points", xytext=(4, 4),
                                fontsize=6.5, color=color, alpha=0.8)

            if method == "isotonic_regression":
                iso_interp = dict(zip(ns, ece_mean))
            if method == "monotone_platt":
                platt_interp = dict(zip(ns, ece_mean))

        # Draw extrapolation region
        max_observed_n = df_ds["n_numeric"].max()
        ax.axvline(max_observed_n, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.text(max_observed_n + 15, ax.get_ylim()[1] * 0.95, "← observed | extrapolated →",
                fontsize=7, color="gray", ha="left", va="top", style="italic")

        # Mark crossover between isotonic and platt
        common_ns = sorted(set(iso_interp.keys()) & set(platt_interp.keys()))
        crossover_n = None
        for i in range(len(common_ns) - 1):
            n1, n2 = common_ns[i], common_ns[i + 1]
            iso1, iso2 = iso_interp[n1], iso_interp[n2]
            platt1, platt2 = platt_interp[n1], platt_interp[n2]
            # Crossover where isotonic goes below platt
            if iso1 >= platt1 and iso2 <= platt2:
                crossover_n = (n1 + n2) / 2
                crossover_ece = (iso1 + iso2) / 2
                ax.annotate("Crossover\n(iso < platt)",
                            (crossover_n, crossover_ece),
                            xytext=(crossover_n + 10, crossover_ece + 0.01),
                            arrowprops=dict(arrowstyle="->", color="black", lw=1),
                            fontsize=8, color="black")
                crossover_info[ds] = {"n": crossover_n, "ece": crossover_ece}
                break

        ax.set_xscale("log")
        ax.set_xlabel("Calibration Set Size (n)", fontsize=11)
        ax.set_ylabel("Macro ECE", fontsize=11)
        ax.set_title(f"{ds} (ST-MEM probe)", fontsize=12, fontweight="bold")

        # Add extrap label to x ticks
        xticks = list(ax.get_xticks())
        ax.set_xticks(sorted(set(xticks + extrap_ns)))
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "Calibration Error vs Calibration Set Size\n"
        "Filled bands = ±1 std over 5 repetitions. Stars = power-law extrapolation (labelled).",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    for ext in ["pdf", "png"]:
        path = FIGS / f"fig10_calsize_curves.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)

    print("\nCrossovers:", crossover_info)
    print("Done.")


if __name__ == "__main__":
    main()
