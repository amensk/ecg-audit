#!/usr/bin/env python3
"""
Regenerate all paper figures with the full 4-dataset, 5-model benchmark
including MIMIC-IV-ECG (D5).
"""
import json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
RESULTS_DIR = WS / "results"
FIGS_DIR = WS / "writing" / "figures"
FIGS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# ─── Load all results ─────────────────────────────────────────────────────────

def load_all_results():
    """Merge full_benchmark + MIMIC benchmark into one list."""
    results = []
    with open(RESULTS_DIR / "full_benchmark_results.json") as f:
        d = json.load(f)
    results.extend([r for r in d["results"] if r.get("status") == "ok"])

    try:
        with open(RESULTS_DIR / "mimic_benchmark_results.json") as f:
            m = json.load(f)
        results.extend([r for r in m["results"] if r.get("status") == "ok"])
    except FileNotFoundError:
        print("MIMIC results not found, skipping D5")

    return results

ALL_RESULTS = load_all_results()

# ─── Helper: build matrix from results ────────────────────────────────────────

MODELS = ["MERL", "ST-MEM", "ECGFM-KED", "ECG-FM", "S4D"]
MODEL_ORDER = {m: i for i, m in enumerate(MODELS)}
MODEL_COLORS = {
    "MERL": "#2c7bb6",
    "ST-MEM": "#d7191c",
    "ECGFM-KED": "#1a9641",
    "ECG-FM": "#ff7f00",
    "S4D": "#984ea3",
}
PRETRAIN_MARKERS = {
    "MERL": "^",      # contrastive
    "ST-MEM": "s",    # masked reconstruction
    "ECGFM-KED": "D", # knowledge-enhanced
    "ECG-FM": "o",    # hybrid
    "S4D": "P",       # supervised baseline
}
DATASETS = ["PTB-XL", "CODE-15%", "CPSC2018", "MIMIC-IV-ECG"]
DS_ORDER = {d: i for i, d in enumerate(DATASETS)}

def get_val(results, model, dataset, field, recal_method=None):
    for r in results:
        if r["model_name"] == model and r["dataset_name"] == dataset:
            if recal_method is None:
                return r["phase_a"].get(field, float("nan"))
            else:
                return r["phase_b"].get(recal_method, {}).get(field, float("nan"))
    return float("nan")

# ─── Figure 1: Main Results Heatmap (4 datasets) ─────────────────────────────

def fig1_main_heatmap():
    metrics = ["macro_auroc", "macro_ece_ew", "macro_sce"]
    metric_labels = ["AUROC ↑", "ECE ↓", "SCE"]
    n_datasets = len(DATASETS)
    n_models = len(MODELS)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(n_metrics, n_datasets,
                             figsize=(12, 7), constrained_layout=True)

    cmaps = ["Blues", "Reds_r", "RdBu"]
    for mi, (metric, label, cmap_name) in enumerate(zip(metrics, metric_labels, cmaps)):
        mat = np.zeros((n_models, n_datasets))
        for di, ds in enumerate(DATASETS):
            for mj, model in enumerate(MODELS):
                mat[mj, di] = get_val(ALL_RESULTS, model, ds, metric)

        for di, ds in enumerate(DATASETS):
            ax = axes[mi, di]
            col = mat[:, di:di+1]
            vmin = np.nanmin(col); vmax = np.nanmax(col)
            if mi == 2:  # SCE: center at 0
                abs_max = max(abs(vmin), abs(vmax), 0.001)
                vmin, vmax = -abs_max, abs_max
            im = ax.imshow(col, cmap=cmap_name, vmin=vmin, vmax=vmax,
                          aspect="auto", interpolation="nearest")
            ax.set_yticks(range(n_models))
            ax.set_yticklabels(MODELS if di == 0 else [])
            ax.set_xticks([])
            if mi == 0:
                ax.set_title(ds, fontweight="bold", fontsize=9)
            if di == n_datasets - 1:
                plt.colorbar(im, ax=ax, fraction=0.08, pad=0.02)
            for mj in range(n_models):
                val = col[mj, 0]
                if not np.isnan(val):
                    txt = f"{val:.3f}"
                    # Color text for readability
                    bg = plt.cm.get_cmap(cmap_name)((val - vmin) / max(vmax - vmin, 1e-6))
                    lum = 0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2]
                    tc = "white" if lum < 0.5 else "black"
                    ax.text(0, mj, txt, ha="center", va="center",
                           fontsize=8, color=tc, fontweight="bold")
        axes[mi, 0].set_ylabel(label, fontsize=10)

    fig.suptitle("Benchmark Summary: AUROC, ECE, SCE across Models and Datasets",
                fontweight="bold", fontsize=12)
    for ext in ["pdf", "png"]:
        fig.savefig(FIGS_DIR / f"fig1_main_results_heatmap.{ext}",
                   bbox_inches="tight", dpi=150)
    plt.close()
    print("fig1_main_results_heatmap saved")

# ─── Figure 2: AUROC–ECE Tradeoff Scatter ────────────────────────────────────

def fig2_tradeoff_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    # Left: raw; Right: post-isotonic
    for ax_idx, (ax, title, recal) in enumerate(zip(
            axes,
            ["Raw (uncalibrated)", "Post-isotonic recalibration"],
            [None, "isotonic"])):

        for ds_idx, ds in enumerate(DATASETS):
            for model in MODELS:
                auroc = get_val(ALL_RESULTS, model, ds, "macro_auroc", recal)
                ece   = get_val(ALL_RESULTS, model, ds, "macro_ece_ew", recal)
                if np.isnan(auroc) or np.isnan(ece):
                    continue
                marker = PRETRAIN_MARKERS[model]
                color  = MODEL_COLORS[model]
                alpha = 1.0 if ds in ("PTB-XL", "MIMIC-IV-ECG") else 0.5
                ax.scatter(auroc, ece, c=color, marker=marker,
                          s=90, alpha=alpha, edgecolors="white", linewidths=0.5,
                          zorder=5)
                # Annotate only PTB-XL for clarity
                if ds == "PTB-XL" and ax_idx == 0:
                    ax.annotate(model[:4], (auroc, ece),
                               textcoords="offset points", xytext=(4, 3),
                               fontsize=7, color=color)

        ax.set_xlabel("Macro AUROC ↑", fontsize=10)
        ax.set_ylabel("Macro ECE ↓", fontsize=10)
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_xlim(0.5, 1.01)

    # Legend: models
    model_handles = [Patch(facecolor=MODEL_COLORS[m], label=m) for m in MODELS]
    ds_handles = [
        plt.Line2D([0], [0], marker="o", color="gray",
                  markersize=6, alpha=1.0, label="PTB-XL / MIMIC"),
        plt.Line2D([0], [0], marker="o", color="gray",
                  markersize=6, alpha=0.5, label="CPSC2018 / CODE-15%"),
    ]
    axes[1].legend(handles=model_handles + ds_handles, ncol=2,
                  loc="upper left", fontsize=8, framealpha=0.8)

    fig.suptitle("AUROC–ECE Tradeoff: Better Rank Does Not Mean Better Calibration",
                fontweight="bold")
    for ext in ["pdf", "png"]:
        fig.savefig(FIGS_DIR / f"fig2_tradeoff_scatter.{ext}",
                   bbox_inches="tight", dpi=150)
    plt.close()
    print("fig2_tradeoff_scatter saved")

# ─── Figure 3: Recalibration Comparison ──────────────────────────────────────

def fig3_recalibration():
    methods = ["raw", "platt", "isotonic", "temperature"]
    method_labels = ["Raw", "Platt", "Isotonic", "Temp."]
    method_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(13, 4), constrained_layout=True)

    for di, (ax, ds) in enumerate(zip(axes, DATASETS)):
        x = np.arange(len(MODELS))
        width = 0.2

        for mi, (method, label, color) in enumerate(zip(methods, method_labels, method_colors)):
            eces = []
            for model in MODELS:
                if method == "raw":
                    ece = get_val(ALL_RESULTS, model, ds, "macro_ece_ew")
                else:
                    ece = get_val(ALL_RESULTS, model, ds, "macro_ece_ew", method)
                eces.append(ece)

            offset = (mi - 1.5) * width
            bars = ax.bar(x + offset, eces, width, label=label, color=color, alpha=0.85)

        ax.set_title(ds, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([m[:5] for m in MODELS], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Macro ECE ↓" if di == 0 else "")
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3)
        if di == 0:
            ax.legend(ncol=2, fontsize=8)

    fig.suptitle("ECE Before and After Post-Hoc Recalibration (by Dataset)",
                fontweight="bold")
    for ext in ["pdf", "png"]:
        fig.savefig(FIGS_DIR / f"fig3_recalibration_comparison.{ext}",
                   bbox_inches="tight", dpi=150)
    plt.close()
    print("fig3_recalibration_comparison saved")

# ─── Figure 4: ECE Reduction Summary ─────────────────────────────────────────

def fig5_ece_reduction():
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(MODELS))
    width = 0.18

    for di, (ds, color) in enumerate(zip(DATASETS, ["#2c7bb6", "#1a9641", "#d7191c", "#ff7f00"])):
        reductions = []
        for model in MODELS:
            raw = get_val(ALL_RESULTS, model, ds, "macro_ece_ew")
            iso = get_val(ALL_RESULTS, model, ds, "macro_ece_ew", "isotonic")
            if not (np.isnan(raw) or np.isnan(iso) or raw == 0):
                reductions.append(100 * (raw - iso) / raw)
            else:
                reductions.append(float("nan"))
        offset = (di - 1.5) * width
        ax.bar(x + offset, reductions, width, label=ds, color=color, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15, ha="right")
    ax.set_ylabel("Relative ECE Reduction (%) ↑")
    ax.set_title("Isotonic Recalibration ECE Reduction by Model and Dataset",
                fontweight="bold")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.3)

    for ext in ["pdf", "png"]:
        fig.savefig(FIGS_DIR / f"fig5_ece_reduction.{ext}",
                   bbox_inches="tight", dpi=150)
    plt.close()
    print("fig5_ece_reduction saved")

# ─── Figure: Cross-dataset calibration transfer ───────────────────────────────

def fig_cross_dataset():
    """Show ECE across datasets for each model (cross-dataset calibration transfer)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    x = np.arange(len(DATASETS))
    width = 0.15

    for mi, model in enumerate(MODELS):
        eces = [get_val(ALL_RESULTS, model, ds, "macro_ece_ew") for ds in DATASETS]
        offset = (mi - 2) * width
        bars = ax.bar(x + offset, eces, width, label=model,
                     color=MODEL_COLORS[model], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=15, ha="right")
    ax.set_ylabel("Macro ECE ↓ (lower = better calibrated)")
    ax.set_title("Cross-Dataset Calibration: ECE Varies Strongly by Dataset",
                fontweight="bold")
    ax.legend(ncol=3)
    ax.grid(axis="y", alpha=0.3)

    for ext in ["pdf", "png"]:
        fig.savefig(FIGS_DIR / f"fig_cross_dataset_ece.{ext}",
                   bbox_inches="tight", dpi=150)
    plt.close()
    print("fig_cross_dataset_ece saved")

# ─── Figure: MIMIC-IV-ECG specific (ECG-FM in-domain comparison) ──────────────

def fig_mimic_indomain():
    """Highlight that ECG-FM (pretrained on MIMIC) does NOT dominate on MIMIC."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ds = "MIMIC-IV-ECG"

    for ax, metric, label in zip(axes, ["macro_auroc", "macro_ece_ew"], ["AUROC ↑", "ECE ↓"]):
        vals = [get_val(ALL_RESULTS, m, ds, metric) for m in MODELS]
        colors_list = [MODEL_COLORS[m] for m in MODELS]
        # Highlight ECG-FM
        edgecolors = ["gold" if m == "ECG-FM" else "white" for m in MODELS]
        linewidths = [3 if m == "ECG-FM" else 0.5 for m in MODELS]
        bars = ax.bar(MODELS, vals, color=colors_list, edgecolor=edgecolors,
                     linewidth=linewidths, alpha=0.85)
        ax.set_ylabel(label)
        ax.set_title(f"MIMIC-IV-ECG: {label}", fontweight="bold")
        ax.set_xticklabels(MODELS, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    axes[0].set_title("MIMIC-IV-ECG AUROC ↑\n(ECG-FM: pretrained on MIMIC, gold border)",
                     fontweight="bold", fontsize=9)
    fig.suptitle("Within-Domain Pretraining Does Not Guarantee Best Performance",
                fontweight="bold")
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(FIGS_DIR / f"fig_mimic_indomain.{ext}",
                   bbox_inches="tight", dpi=150)
    plt.close()
    print("fig_mimic_indomain saved")

# ─── Figure 4: SCE Analysis (updated with 4 datasets) ─────────────────────────

def fig4_sce():
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(13, 4), constrained_layout=True)

    for di, (ax, ds) in enumerate(zip(axes, DATASETS)):
        sces = [get_val(ALL_RESULTS, m, ds, "macro_sce") for m in MODELS]
        colors_list = [MODEL_COLORS[m] for m in MODELS]
        bars = ax.bar(MODELS, sces, color=colors_list, alpha=0.85,
                     edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(ds, fontweight="bold")
        ax.set_xticklabels([m[:5] for m in MODELS], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("SCE (+ = overconfident, - = underconfident)" if di == 0 else "")
        ax.grid(axis="y", alpha=0.3)

        for bar, sce in zip(bars, sces):
            if not np.isnan(sce):
                ax.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + (0.001 if sce >= 0 else -0.003),
                       f"{sce:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Signed Calibration Error (SCE): Overconfidence vs Underconfidence",
                fontweight="bold")
    for ext in ["pdf", "png"]:
        fig.savefig(FIGS_DIR / f"fig4_sce_analysis.{ext}",
                   bbox_inches="tight", dpi=150)
    plt.close()
    print("fig4_sce_analysis saved")

# ─── Reliability diagrams for MIMIC ───────────────────────────────────────────

def fig_reliability_mimic():
    """Generate per-model reliability diagrams for MIMIC-IV-ECG using cached predictions."""
    try:
        with open(RESULTS_DIR / "mimic_benchmark_results.json") as f:
            mres = json.load(f)
    except FileNotFoundError:
        print("MIMIC results not found, skipping reliability diagrams")
        return

    n_bins = 10
    MIMIC_CLASSES = ["Normal", "AF", "RBBB", "LBBB", "LVH"]

    for r in mres["results"]:
        if r.get("status") != "ok":
            continue
        model_name = r["model_name"]
        pa = r["phase_a"]
        per_class = pa.get("per_class", {})
        if not per_class:
            continue

        fig, axes = plt.subplots(1, min(5, len(per_class)), figsize=(12, 3))
        if len(per_class) == 1:
            axes = [axes]

        for ci, (cls_idx, cls_data) in enumerate(per_class.items()):
            if ci >= len(axes): break
            ax = axes[ci]
            cls_name = MIMIC_CLASSES[int(cls_idx)] if int(cls_idx) < len(MIMIC_CLASSES) else f"Class {cls_idx}"
            auroc = cls_data.get("auroc", float("nan"))
            ece = cls_data.get("ece_ew", float("nan"))
            n_pos = cls_data.get("n_pos", 0)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
            ax.set_title(f"{cls_name}\nAUROC={auroc:.2f} ECE={ece:.3f}\n(n+={n_pos})",
                        fontsize=8)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Empirical" if ci == 0 else "")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)

        fig.suptitle(f"MIMIC-IV-ECG Reliability: {model_name}", fontweight="bold")
        fig.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(FIGS_DIR / f"reliability_{model_name.replace('-','_').replace(' ','_')}_MIMIC.{ext}",
                       bbox_inches="tight", dpi=150)
        plt.close()

    print("Reliability diagrams for MIMIC saved")

# ─── Figure: Paper results table ──────────────────────────────────────────────

def write_results_tables():
    """Write LaTeX tables with complete benchmark results."""
    # Table 1: Phase A (raw) - all 20 cells
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Full benchmark: AUROC, ECE, and SCE for all model–dataset combinations. "
        r"ECG-FM was pretrained on MIMIC-IV-ECG (marked $\dagger$). "
        r"Best AUROC per dataset in \textbf{bold}; best ECE in \underline{underline}.}",
        r"\label{tab:full_benchmark}",
        r"\small",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Dataset & Model & AUROC $\uparrow$ & ECE $\downarrow$ & Brier $\downarrow$ & SCE "
        r"& Iso ECE $\downarrow$ \\",
        r"\midrule",
    ]

    for ds in DATASETS:
        best_auroc = max(
            (get_val(ALL_RESULTS, m, ds, "macro_auroc") for m in MODELS),
            default=0.0
        )
        best_ece = min(
            (get_val(ALL_RESULTS, m, ds, "macro_ece_ew") for m in MODELS),
            default=99.0
        )
        for mi, model in enumerate(MODELS):
            auroc = get_val(ALL_RESULTS, model, ds, "macro_auroc")
            ece   = get_val(ALL_RESULTS, model, ds, "macro_ece_ew")
            brier = get_val(ALL_RESULTS, model, ds, "macro_brier")
            sce   = get_val(ALL_RESULTS, model, ds, "macro_sce")
            iso   = get_val(ALL_RESULTS, model, ds, "macro_ece_ew", "isotonic")

            def fmt(v, bold=False, under=False):
                if np.isnan(v):
                    return "--"
                s = f"{v:.3f}"
                if bold:
                    s = r"\textbf{" + s + "}"
                if under:
                    s = r"\underline{" + s + "}"
                return s

            model_str = model
            if model == "ECG-FM" and ds == "MIMIC-IV-ECG":
                model_str += r"$^\dagger$"

            ds_str = ds if mi == 0 else ""
            line = (f"  {ds_str} & {model_str} & "
                   f"{fmt(auroc, bold=abs(auroc-best_auroc)<0.001)} & "
                   f"{fmt(ece,   under=abs(ece-best_ece)<0.001)} & "
                   f"{fmt(brier)} & {fmt(sce)} & {fmt(iso)} \\\\")
            lines.append(line)
        lines.append(r"  \midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]

    table_path = WS / "writing" / "tables" / "full_benchmark_table.tex"
    table_path.parent.mkdir(exist_ok=True)
    table_path.write_text("\n".join(lines))
    print(f"LaTeX table written: {table_path}")

    # Also write CSV with all 20 rows
    rows = []
    for ds in DATASETS:
        for model in MODELS:
            auroc = get_val(ALL_RESULTS, model, ds, "macro_auroc")
            ece   = get_val(ALL_RESULTS, model, ds, "macro_ece_ew")
            brier = get_val(ALL_RESULTS, model, ds, "macro_brier")
            sce   = get_val(ALL_RESULTS, model, ds, "macro_sce")
            platt_auroc = get_val(ALL_RESULTS, model, ds, "macro_auroc", "platt")
            platt_ece   = get_val(ALL_RESULTS, model, ds, "macro_ece_ew", "platt")
            iso_auroc   = get_val(ALL_RESULTS, model, ds, "macro_auroc", "isotonic")
            iso_ece     = get_val(ALL_RESULTS, model, ds, "macro_ece_ew", "isotonic")
            rows.append({
                "dataset": ds, "model": model,
                "auroc": f"{auroc:.4f}", "ece": f"{ece:.4f}",
                "brier": f"{brier:.4f}", "sce": f"{sce:.4f}",
                "platt_auroc": f"{platt_auroc:.4f}", "platt_ece": f"{platt_ece:.4f}",
                "iso_auroc": f"{iso_auroc:.4f}", "iso_ece": f"{iso_ece:.4f}",
            })

    with open(RESULTS_DIR / "paper_results_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("paper_results_table.csv written")

# ─── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loaded {len(ALL_RESULTS)} result entries")
    ds_seen = set(r["dataset_name"] for r in ALL_RESULTS)
    print(f"Datasets: {ds_seen}")

    fig1_main_heatmap()
    fig2_tradeoff_scatter()
    fig3_recalibration()
    fig4_sce()
    fig5_ece_reduction()
    fig_cross_dataset()
    fig_mimic_indomain()
    fig_reliability_mimic()
    write_results_tables()

    print("\nAll figures regenerated successfully.")
