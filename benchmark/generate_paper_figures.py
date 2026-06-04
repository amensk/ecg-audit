#!/usr/bin/env python3
"""Generate all figures for the paper using real benchmark results."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
RESULTS = WS / "results"
FIGURES = WS / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "figure.dpi": 150,
})

MODEL_COLORS = {
    "MERL":      "#e41a1c",   # red (contrastive)
    "ST-MEM":    "#377eb8",   # blue (masked reconstruction)
    "ECGFM-KED": "#ff7f00",   # orange (knowledge-enhanced)
    "ECG-FM":    "#984ea3",   # purple (contrastive+generative)
    "S4D":       "#4daf4a",   # green (supervised)
}
OBJECTIVE_LABELS = {
    "MERL":      "Contrastive\n(InfoNCE)",
    "ST-MEM":    "Masked\nReconstruction",
    "ECGFM-KED": "Knowledge-\nEnhanced",
    "ECG-FM":    "Contrastive+\nGenerative",
    "S4D":       "Supervised\n(no pretraining)",
}
DATASET_COLORS = {"PTB-XL": "#1f78b4", "CPSC2018": "#33a02c", "CODE-15%": "#6a3d9a"}

def load_results():
    with open(RESULTS / "full_benchmark_results.json") as f:
        data = json.load(f)
    rows = []
    for r in data["results"]:
        if r.get("status") != "ok":
            continue
        pa = r["phase_a"]
        pb = r.get("phase_b", {})
        row = {
            "model": r["model_name"],
            "dataset": r["dataset_name"],
            "n_test": r.get("n_test", 0),
            "auroc": pa["macro_auroc"], "ece": pa["macro_ece_ew"],
            "brier": pa["macro_brier"], "sce": pa["macro_sce"],
            "auroc_platt": pb.get("platt", {}).get("macro_auroc", np.nan),
            "ece_platt":   pb.get("platt", {}).get("macro_ece_ew", np.nan),
            "auroc_iso":   pb.get("isotonic", {}).get("macro_auroc", np.nan),
            "ece_iso":     pb.get("isotonic", {}).get("macro_ece_ew", np.nan),
            "auroc_temp":  pb.get("temperature", {}).get("macro_auroc", np.nan),
            "ece_temp":    pb.get("temperature", {}).get("macro_ece_ew", np.nan),
        }
        row["delta_ece_iso"] = row["ece_iso"] - row["ece"]
        row["delta_auroc_iso"] = row["auroc_iso"] - row["auroc"]
        row["rel_ece_reduction_iso"] = (row["ece"] - row["ece_iso"]) / (row["ece"] + 1e-9)
        rows.append(row)
    return pd.DataFrame(rows)


def fig1_main_results_table(df):
    """Figure 1: Main results heatmap — AUROC and ECE across all model × dataset cells."""
    models = ["MERL", "ST-MEM", "ECGFM-KED", "ECG-FM", "S4D"]
    datasets = ["PTB-XL", "CPSC2018", "CODE-15%"]

    # Pivot tables
    def pivot(df, col, model_order, ds_order):
        p = df.pivot(index="model", columns="dataset", values=col)
        return p.reindex(index=model_order, columns=ds_order)

    auroc_p = pivot(df, "auroc", models, datasets)
    ece_p   = pivot(df, "ece",   models, datasets)
    sce_p   = pivot(df, "sce",   models, datasets)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Real Benchmark Results: 4 Foundation Models + S4D Baseline × 3 Datasets",
                 fontsize=12, fontweight="bold", y=1.03)

    for ax, data_p, title, cmap, fmt in [
        (axes[0], auroc_p, "AUROC (higher=better)", "Blues", ".3f"),
        (axes[1], ece_p,   "ECE (lower=better)",    "Reds_r",".3f"),
        (axes[2], sce_p,   "SCE (negative=underconfident)", "RdBu_r", ".3f"),
    ]:
        vals = data_p.values.astype(float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        im = ax.imshow(vals, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=20, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.set_title(title, fontweight="bold", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(len(models)):
            for j in range(len(datasets)):
                v = vals[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                            fontsize=8, color="white" if (v-vmin)/(vmax-vmin+1e-9) > 0.6 else "black")

    plt.tight_layout()
    path = FIGURES / "fig1_main_results_heatmap.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf",".png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig2_discrimination_calibration_tradeoff(df):
    """Figure 2: AUROC vs ECE scatter, two panels (PTB-XL and CPSC2018)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Discrimination–Calibration Tradeoff by Pretraining Objective",
                 fontsize=12, fontweight="bold")

    for ax, ds in zip(axes, ["PTB-XL", "CPSC2018"]):
        sub = df[df["dataset"] == ds]
        for _, row in sub.iterrows():
            model = row["model"]
            c = MODEL_COLORS.get(model, "gray")
            ax.scatter(row["ece"], row["auroc"], color=c, s=180, zorder=5,
                       edgecolors="k", linewidths=0.7)
            ax.annotate(model, (row["ece"], row["auroc"]),
                        xytext=(5, 4), textcoords="offset points", fontsize=8.5)
        ax.set_xlabel("ECE (equal-width, 15 bins)", fontsize=10)
        ax.set_ylabel("Macro AUROC", fontsize=10)
        ax.set_title(f"{ds}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Legend by objective type
    patches = [mpatches.Patch(color=v, label=f"{k}\n({OBJECTIVE_LABELS[k].replace(chr(10),' ')})")
               for k, v in MODEL_COLORS.items()]
    axes[1].legend(handles=patches, fontsize=7, loc="lower right", ncol=1)
    plt.tight_layout()
    path = FIGURES / "fig2_tradeoff_scatter.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf",".png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig3_recalibration_comparison(df):
    """Figure 3: ECE before vs after recalibration (isotonic and Platt) per model × dataset."""
    models = ["MERL", "ST-MEM", "ECGFM-KED", "ECG-FM", "S4D"]
    datasets = ["PTB-XL", "CPSC2018", "CODE-15%"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("ECE Before and After Recalibration (Isotonic and Platt Scaling)",
                 fontsize=12, fontweight="bold")

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds]
        x = np.arange(len(models))
        w = 0.25
        raw_vals  = [sub[sub["model"] == m]["ece"].values[0] if len(sub[sub["model"]==m]) > 0 else np.nan for m in models]
        platt_vals = [sub[sub["model"] == m]["ece_platt"].values[0] if len(sub[sub["model"]==m]) > 0 else np.nan for m in models]
        iso_vals  = [sub[sub["model"] == m]["ece_iso"].values[0] if len(sub[sub["model"]==m]) > 0 else np.nan for m in models]
        colors = [MODEL_COLORS.get(m, "gray") for m in models]
        ax.bar(x - w, raw_vals, w, color=colors, alpha=0.35, edgecolor="k", lw=0.5, label="Raw")
        ax.bar(x,      platt_vals, w, color=colors, alpha=0.65, edgecolor="k", lw=0.5, hatch="//", label="Platt")
        ax.bar(x + w, iso_vals,  w, color=colors, alpha=1.0,  edgecolor="k", lw=0.5, hatch="", label="Isotonic")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8.5)
        ax.set_ylabel("ECE (equal-width)", fontsize=9)
        ax.set_title(ds, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, max([v for v in raw_vals + platt_vals + iso_vals if not np.isnan(v)]) * 1.25)
        if ds == "PTB-XL":
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(facecolor="gray", alpha=0.35, label="Raw"),
                                Patch(facecolor="gray", alpha=0.65, hatch="//", label="Platt"),
                                Patch(facecolor="gray", label="Isotonic")], fontsize=8)
    plt.tight_layout()
    path = FIGURES / "fig3_recalibration_comparison.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf",".png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig4_sce_analysis(df):
    """Figure 4: SCE (signed calibration error) — underconfidence vs overconfidence."""
    models = ["MERL", "ST-MEM", "ECG-FM", "ECGFM-KED", "S4D"]
    datasets = ["PTB-XL", "CPSC2018"]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Signed Calibration Error by Model and Dataset\n"
                 "(positive = overconfident; negative = underconfident)",
                 fontsize=11, fontweight="bold")

    x = np.arange(len(models))
    w = 0.35
    for i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds]
        sce_vals = [sub[sub["model"]==m]["sce"].values[0] if len(sub[sub["model"]==m])>0 else np.nan for m in models]
        colors = [MODEL_COLORS.get(m, "gray") for m in models]
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, sce_vals, w, color=colors,
                      alpha=0.7 + i * 0.2, edgecolor="k", lw=0.5,
                      label=ds)
    ax.axhline(0, color="k", linewidth=1.2, linestyle="--")
    ax.fill_between([-0.5, len(models)-0.5], [-0.015, -0.015], [0, 0],
                    alpha=0.06, color="blue", label="Underconfident zone")
    ax.fill_between([-0.5, len(models)-0.5], [0, 0], [0.015, 0.015],
                    alpha=0.06, color="red", label="Overconfident zone")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({OBJECTIVE_LABELS[m].replace(chr(10),' ')})" for m in models],
                       fontsize=8)
    ax.set_ylabel("SCE = mean(confidence) − mean(accuracy)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(-0.5, len(models) - 0.5)
    plt.tight_layout()
    path = FIGURES / "fig4_sce_analysis.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf",".png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig5_ece_reduction_summary(df):
    """Figure 5: Relative ECE reduction after isotonic recalibration."""
    models = ["MERL", "ST-MEM", "ECG-FM", "ECGFM-KED", "S4D"]
    datasets = ["PTB-XL", "CPSC2018", "CODE-15%"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("Relative ECE Reduction After Isotonic Recalibration\n"
                 "(positive = ECE improved; labeled with raw ECE → recal ECE)",
                 fontsize=10, fontweight="bold")

    x = np.arange(len(models))
    w = 0.25
    offsets = [-w, 0, w]
    for i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds]
        reductions = []
        for m in models:
            row = sub[sub["model"]==m]
            if len(row) > 0:
                reductions.append(float(row["rel_ece_reduction_iso"].values[0]))
            else:
                reductions.append(np.nan)
        bars = ax.bar(x + offsets[i], reductions, w, color=DATASET_COLORS.get(ds, "gray"),
                      alpha=0.8, edgecolor="k", lw=0.5, label=ds)
        for xi, (m, r) in enumerate(zip(models, reductions)):
            if not np.isnan(r):
                row = sub[sub["model"]==m]
                if len(row) > 0:
                    ece_raw = float(row["ece"].values[0])
                    ece_iso = float(row["ece_iso"].values[0])
                    ax.text(xi + offsets[i], r + 0.01,
                            f"{ece_raw:.3f}→{ece_iso:.3f}",
                            ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.axhline(0, color="k", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Relative ECE reduction = (ECE_raw−ECE_iso) / ECE_raw", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(-0.3, 1.05)
    plt.tight_layout()
    path = FIGURES / "fig5_ece_reduction.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(str(path).replace(".pdf",".png"), bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def save_latex_table(df):
    """Save a comprehensive LaTeX table for the paper."""
    models = ["MERL", "ST-MEM", "ECG-FM", "ECGFM-KED", "S4D"]
    datasets = ["PTB-XL", "CPSC2018", "CODE-15%"]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Full benchmark results: 4 ECG foundation models + S4D supervised baseline across 3 datasets.",
        r"Phase~A = raw (uncalibrated). Phase~B = isotonic recalibration.",
        r"ECGFM-KED results on CODE-15\% are marked (partial) due to partial architecture reconstruction.",
        r"HuBERT-ECG omitted: corrupted checkpoint. MIMIC-IV-ECG omitted: PhysioNet credential required.}",
        r"\label{tab:full-benchmark}",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Dataset & \multicolumn{4}{c}{Phase A (raw)} & \multicolumn{2}{c}{Phase B (isotonic)} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
        r" & & AUROC & ECE & Brier & SCE & AUROC & ECE \\",
        r"\midrule",
    ]

    prev_model = None
    for model in models:
        for ds in datasets:
            sub = df[(df["model"]==model) & (df["dataset"]==ds)]
            if len(sub) == 0:
                continue
            row = sub.iloc[0]
            is_partial = (model == "ECGFM-KED" and ds == "CODE-15%")
            ece_iso = row["ece_iso"]
            auroc_iso = row["auroc_iso"]

            model_label = model if model != prev_model else ""
            prev_model = model

            partial_note = r"$^*$" if is_partial else ""
            # Bold best AUROC per dataset
            line = (f"{model_label} & {ds} & "
                    f"{row['auroc']:.3f} & {row['ece']:.3f} & "
                    f"{row['brier']:.3f} & {row['sce']:+.3f} & "
                    f"{auroc_iso:.3f} & {ece_iso:.3f}{partial_note} \\\\")
            lines.append(line)
        if model != models[-1]:
            lines.append(r"\addlinespace[2pt]")

    lines += [
        r"\bottomrule",
        r"\multicolumn{8}{l}{$^*$ ECGFM-KED CODE-15\%: partial architecture load (35 missing keys); AUROC near chance.} \\",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    table_path = WS / "writing" / "tables" / "full_benchmark_table.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines))
    print(f"Saved LaTeX table: {table_path}")


def main():
    print(f"Generating paper figures at {datetime.now().isoformat()[:19]}")
    df = load_results()
    print(f"Loaded {len(df)} result rows:")
    print(df[["model","dataset","auroc","ece","sce"]].to_string(index=False))

    fig1_main_results_table(df)
    fig2_discrimination_calibration_tradeoff(df)
    fig3_recalibration_comparison(df)
    fig4_sce_analysis(df)
    fig5_ece_reduction_summary(df)
    save_latex_table(df)

    # Save structured results CSV for paper
    df.to_csv(RESULTS / "paper_results_table.csv", index=False)
    print("\nAll figures and tables saved.")
    print("Results summary:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
