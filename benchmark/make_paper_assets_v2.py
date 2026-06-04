#!/usr/bin/env python3
"""
Generate all paper figures + LaTeX tables from benchmark_v2_results.json
(6 models x 4 datasets, bootstrap CIs, MIMIC v2 labels with MI).
"""
import json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
RESULTS = WS / "results"
FIGS = WS / "writing" / "figures"
TABLES = WS / "writing" / "tables"
FIGS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150,
})

MODELS = ["MERL", "ST-MEM", "HuBERT-ECG", "ECGFM-KED", "ECG-FM", "S4D"]
DATASETS = ["PTB-XL", "CODE-15%", "CPSC2018", "MIMIC-IV-ECG"]
MODEL_COLORS = {
    "MERL": "#2c7bb6", "ST-MEM": "#d7191c", "HuBERT-ECG": "#8c510a",
    "ECGFM-KED": "#1a9641", "ECG-FM": "#ff7f00", "S4D": "#984ea3",
}
PRETRAIN_MARKERS = {  # marker encodes pretraining objective family
    "MERL": "^",        # contrastive
    "ST-MEM": "s",      # masked reconstruction
    "HuBERT-ECG": "X",  # masked hidden-unit prediction
    "ECGFM-KED": "D",   # knowledge-enhanced
    "ECG-FM": "o",      # hybrid contrastive+generative
    "S4D": "P",         # supervised
}

with open(RESULTS / "benchmark_v2_results.json") as f:
    DATA = json.load(f)
R = {(r["model_name"], r["dataset_name"]): r
     for r in DATA["results"] if r.get("status") == "ok"}


def get(model, ds, field, recal=None):
    r = R.get((model, ds))
    if r is None:
        return np.nan
    src = r["phase_a"] if recal is None else r["phase_b"].get(recal, {})
    return src.get(field, np.nan)


def get_ci(model, ds, key):
    r = R.get((model, ds))
    if r is None:
        return None
    return r["phase_a"].get(key)


# ── fig1: heatmap AUROC / ECE / SCE ───────────────────────────────────────────
def fig1():
    metrics = [("macro_auroc", "AUROC ↑", "Blues"),
               ("macro_ece_ew", "ECE ↓", "Reds_r"),
               ("macro_sce", "SCE", "RdBu")]
    fig, axes = plt.subplots(len(metrics), len(DATASETS), figsize=(13, 7.5),
                             constrained_layout=True)
    for mi, (metric, label, cmap) in enumerate(metrics):
        for di, ds in enumerate(DATASETS):
            ax = axes[mi, di]
            col = np.array([[get(m, ds, metric)] for m in MODELS])
            vmin, vmax = np.nanmin(col), np.nanmax(col)
            if mi == 2:
                a = max(abs(vmin), abs(vmax), 1e-3); vmin, vmax = -a, a
            im = ax.imshow(col, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_yticks(range(len(MODELS)))
            ax.set_yticklabels(MODELS if di == 0 else [])
            ax.set_xticks([])
            if mi == 0:
                ax.set_title(ds, fontweight="bold", fontsize=9)
            if di == len(DATASETS) - 1:
                plt.colorbar(im, ax=ax, fraction=0.09, pad=0.02)
            for r_ in range(len(MODELS)):
                v = col[r_, 0]
                if not np.isnan(v):
                    cmap_obj = matplotlib.colormaps[cmap]
                    bg = cmap_obj((v - vmin) / max(vmax - vmin, 1e-6))
                    lum = 0.299*bg[0]+0.587*bg[1]+0.114*bg[2]
                    ax.text(0, r_, f"{v:.3f}", ha="center", va="center", fontsize=7.5,
                            color="white" if lum < 0.5 else "black", fontweight="bold")
        axes[mi, 0].set_ylabel(label, fontsize=10)
    fig.suptitle("Benchmark: AUROC, ECE, SCE across 6 Models × 4 Datasets",
                 fontweight="bold", fontsize=12)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig1_main_results_heatmap.{ext}", bbox_inches="tight", dpi=150)
    plt.close()
    print("fig1 done")


# ── fig2: tradeoff scatter with CI error bars ─────────────────────────────────
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, title, recal in zip(axes, ["Raw (uncalibrated)", "Post-isotonic"],
                                [None, "isotonic"]):
        for ds in DATASETS:
            alpha = 1.0 if ds in ("PTB-XL", "MIMIC-IV-ECG") else 0.45
            for m in MODELS:
                a = get(m, ds, "macro_auroc", recal)
                e = get(m, ds, "macro_ece_ew", recal)
                if np.isnan(a) or np.isnan(e):
                    continue
                if recal is None:
                    aci, eci = get_ci(m, ds, "macro_auroc_ci"), get_ci(m, ds, "macro_ece_ci")
                    if aci and eci and alpha == 1.0:
                        xerr = [[max(0.0, a-aci[0])], [max(0.0, aci[1]-a)]]
                        yerr = [[max(0.0, e-eci[0])], [max(0.0, eci[1]-e)]]
                        ax.errorbar(a, e, xerr=xerr, yerr=yerr, fmt="none",
                                    ecolor=MODEL_COLORS[m], alpha=0.35, lw=0.8, zorder=3)
                ax.scatter(a, e, c=MODEL_COLORS[m], marker=PRETRAIN_MARKERS[m],
                           s=85, alpha=alpha, edgecolors="white", linewidths=0.5, zorder=5)
        ax.set_xlabel("Macro AUROC ↑"); ax.set_ylabel("Macro ECE ↓")
        ax.set_title(title, fontweight="bold"); ax.grid(alpha=0.3); ax.set_xlim(0.5, 1.01)
    handles = [Patch(facecolor=MODEL_COLORS[m], label=m) for m in MODELS]
    handles += [plt.Line2D([0],[0], marker="o", color="gray", markersize=6, alpha=1.0, label="PTB-XL/MIMIC (CI)"),
                plt.Line2D([0],[0], marker="o", color="gray", markersize=6, alpha=0.45, label="CPSC/CODE-15%")]
    axes[1].legend(handles=handles, ncol=2, loc="upper right", fontsize=7.5, framealpha=0.85)
    fig.suptitle("Discrimination–Calibration Tradeoff (error bars: bootstrap 95% CI)",
                 fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig2_tradeoff_scatter.{ext}", bbox_inches="tight", dpi=150)
    plt.close()
    print("fig2 done")


# ── fig3: recalibration bars ──────────────────────────────────────────────────
def fig3():
    methods = [("raw", "Raw", "#e41a1c"), ("platt", "Platt", "#377eb8"),
               ("isotonic", "Isotonic", "#4daf4a"), ("temperature", "Temp.", "#ff7f00")]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(14, 4), constrained_layout=True)
    x = np.arange(len(MODELS)); w = 0.2
    for di, (ax, ds) in enumerate(zip(axes, DATASETS)):
        for mi, (mth, lab, col) in enumerate(methods):
            eces = [get(m, ds, "macro_ece_ew", None if mth == "raw" else mth) for m in MODELS]
            ax.bar(x + (mi-1.5)*w, eces, w, label=lab, color=col, alpha=0.85)
        ax.set_title(ds, fontweight="bold"); ax.set_xticks(x)
        ax.set_xticklabels([m[:5] for m in MODELS], rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel("Macro ECE ↓" if di == 0 else ""); ax.grid(axis="y", alpha=0.3)
        if di == 0:
            ax.legend(ncol=2, fontsize=7.5)
    fig.suptitle("ECE Before/After Post-Hoc Recalibration", fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig3_recalibration_comparison.{ext}", bbox_inches="tight", dpi=150)
    plt.close()
    print("fig3 done")


# ── fig4: SCE bars ────────────────────────────────────────────────────────────
def fig4():
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(14, 4), constrained_layout=True)
    for di, (ax, ds) in enumerate(zip(axes, DATASETS)):
        sces = [get(m, ds, "macro_sce") for m in MODELS]
        ax.bar(MODELS, sces, color=[MODEL_COLORS[m] for m in MODELS], alpha=0.85)
        ax.axhline(0, color="black", lw=1, ls="--")
        ax.set_title(ds, fontweight="bold")
        ax.set_xticklabels([m[:5] for m in MODELS], rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel("SCE (+over / −under)" if di == 0 else ""); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Signed Calibration Error by Model and Dataset", fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig4_sce_analysis.{ext}", bbox_inches="tight", dpi=150)
    plt.close()
    print("fig4 done")


# ── fig5: ECE reduction (isotonic) with CI whiskers ───────────────────────────
def fig5():
    fig, ax = plt.subplots(figsize=(9, 4.3))
    x = np.arange(len(MODELS)); w = 0.2
    colors = ["#2c7bb6", "#1a9641", "#d7191c", "#ff7f00"]
    for di, (ds, c) in enumerate(zip(DATASETS, colors)):
        reds, los, his = [], [], []
        for m in MODELS:
            raw = get(m, ds, "macro_ece_ew"); iso = get(m, ds, "macro_ece_ew", "isotonic")
            red = 100*(raw-iso)/raw if (raw and not np.isnan(raw) and not np.isnan(iso)) else np.nan
            reds.append(red)
            r = R.get((m, ds)); ci = r.get("isotonic_ece_reduction_ci") if r else None
            if ci and not np.isnan(red):
                los.append(max(0, red-ci[0])); his.append(max(0, ci[1]-red))
            else:
                los.append(0); his.append(0)
        ax.bar(x + (di-1.5)*w, reds, w, label=ds, color=c, alpha=0.85,
               yerr=[los, his], capsize=2, error_kw={"lw": 0.7, "alpha": 0.6})
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(MODELS, rotation=15, ha="right")
    ax.set_ylabel("Relative ECE Reduction (%) ↑")
    ax.set_title("Isotonic Recalibration ECE Reduction (95% CI whiskers)", fontweight="bold")
    ax.legend(ncol=2); ax.grid(axis="y", alpha=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig5_ece_reduction.{ext}", bbox_inches="tight", dpi=150)
    plt.close()
    print("fig5 done")


# ── fig_cross_dataset ─────────────────────────────────────────────────────────
def fig_cross():
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    x = np.arange(len(DATASETS)); w = 0.14
    for mi, m in enumerate(MODELS):
        eces = [get(m, ds, "macro_ece_ew") for ds in DATASETS]
        ax.bar(x + (mi-2.5)*w, eces, w, label=m, color=MODEL_COLORS[m], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(DATASETS, rotation=12, ha="right")
    ax.set_ylabel("Macro ECE ↓"); ax.legend(ncol=3, fontsize=7.5)
    ax.set_title("Cross-Dataset Calibration: ECE Varies Strongly by Dataset", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_cross_dataset_ece.{ext}", bbox_inches="tight", dpi=150)
    plt.close()
    print("fig_cross done")


# ── fig_mimic_indomain ────────────────────────────────────────────────────────
def fig_mimic():
    ds = "MIMIC-IV-ECG"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, lab in zip(axes, ["macro_auroc", "macro_ece_ew"], ["AUROC ↑", "ECE ↓"]):
        vals = [get(m, ds, metric) for m in MODELS]
        edge = ["gold" if m == "ECG-FM" else "white" for m in MODELS]
        lw = [3 if m == "ECG-FM" else 0.5 for m in MODELS]
        bars = ax.bar(MODELS, vals, color=[MODEL_COLORS[m] for m in MODELS],
                      edgecolor=edge, linewidth=lw, alpha=0.88)
        ax.set_ylabel(lab); ax.set_xticklabels(MODELS, rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.004, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7.5)
    axes[0].set_title("MIMIC-IV-ECG AUROC ↑\n(ECG-FM pretrained on MIMIC, gold border)",
                      fontweight="bold", fontsize=9)
    axes[1].set_title("MIMIC-IV-ECG ECE ↓", fontweight="bold", fontsize=9)
    fig.suptitle("Within-Domain Pretraining ≠ Best Linear-Probe Performance", fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_mimic_indomain.{ext}", bbox_inches="tight", dpi=150)
    plt.close()
    print("fig_mimic done")


# ── Pareto / composite (recomputed from v2) ───────────────────────────────────
def pareto_and_table():
    def pareto_front(pts):  # pts: list of (name, auroc, ece); maximize auroc, minimize ece
        front = []
        for name, a, e in pts:
            dominated = any((a2 >= a and e2 <= e and (a2 > a or e2 < e))
                            for n2, a2, e2 in pts if n2 != name)
            if not dominated:
                front.append(name)
        return front

    def composite(a, e, emax, alpha):
        return alpha*a + (1-alpha)*(1 - e/emax)

    out = {}
    for ds in DATASETS:
        pts = [(m, get(m, ds, "macro_auroc"), get(m, ds, "macro_ece_ew"))
               for m in MODELS if not np.isnan(get(m, ds, "macro_auroc"))]
        # exclude ECGFM-KED on CODE-15% (partial reconstruction, near chance)
        if ds == "CODE-15%":
            pts = [p for p in pts if p[0] != "ECGFM-KED"]
        emax = max(e for _, _, e in pts)
        front = pareto_front(pts)
        comp = {}
        for alpha in (0.5, 0.7, 0.9):
            scores = {n: composite(a, e, emax, alpha) for n, a, e in pts}
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            comp[f"alpha_{alpha}"] = {"winner": ranked[0][0],
                                      "winner_score": round(ranked[0][1], 4),
                                      "last": ranked[-1][0],
                                      "last_score": round(ranked[-1][1], 4),
                                      "ranking": [n for n, _ in ranked]}
        out[ds] = {"pareto_front": front, "composite": comp,
                   "points": {n: {"auroc": round(a, 4), "ece": round(e, 4)} for n, a, e in pts}}
    (RESULTS / "pareto_analysis_v2.json").write_text(json.dumps(out, indent=2))
    print("pareto_analysis_v2.json written")
    return out


# ── LaTeX main results table (with CIs) + per-class support table + CSV ───────
def tables():
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Full benchmark across 6 models and 4 datasets (24 cells). "
        r"AUROC and ECE shown with bootstrap 95\% CIs. ECG-FM was pretrained on "
        r"MIMIC-IV-ECG ($\dagger$). Iso-ECE is macro ECE after isotonic recalibration. "
        r"$k$ = number of classes meeting min-support ($\geq$10 positives and "
        r"$\geq$10 negatives in the test split). Best AUROC per dataset in "
        r"\textbf{bold}; best raw ECE \underline{underlined}.}",
        r"\label{tab:full-benchmark}", r"\small",
        r"\begin{tabular}{llcccccc}", r"\toprule",
        r"Dataset & Model & AUROC (95\% CI) $\uparrow$ & ECE (95\% CI) $\downarrow$ "
        r"& Brier & SCE & Iso-ECE & $k$ \\", r"\midrule",
    ]
    for ds in DATASETS:
        aurocs = {m: get(m, ds, "macro_auroc") for m in MODELS}
        eces = {m: get(m, ds, "macro_ece_ew") for m in MODELS}
        best_a = np.nanmax(list(aurocs.values()))
        best_e = np.nanmin(list(eces.values()))
        def fmt_ci(val, ci):
            # clamp display so the point estimate lies within [lo, hi]
            # (ECE is positively biased under bootstrap resampling)
            if not ci:
                return f"{val:.3f}"
            lo, hi = min(ci[0], val), max(ci[1], val)
            return f"{val:.3f} [{lo:.3f},{hi:.3f}]"
        for mi, m in enumerate(MODELS):
            r = R.get((m, ds))
            if r is None:
                continue
            pa = r["phase_a"]
            a, e = pa["macro_auroc"], pa["macro_ece_ew"]
            astr = fmt_ci(a, pa.get("macro_auroc_ci"))
            estr = fmt_ci(e, pa.get("macro_ece_ci"))
            if abs(a-best_a) < 1e-9:
                astr = r"\textbf{" + astr + "}"
            if abs(e-best_e) < 1e-9:
                estr = r"\underline{" + estr + "}"
            mname = m + (r"$^\dagger$" if (m == "ECG-FM" and ds == "MIMIC-IV-ECG") else "")
            dsn = (ds if mi == 0 else "").replace("%", r"\%")
            iso = r["phase_b"]["isotonic"]["macro_ece_ew"]
            lines.append(f"  {dsn} & {mname} & {astr} & {estr} & "
                         f"{pa['macro_brier']:.3f} & {pa['macro_sce']:+.3f} & "
                         f"{iso:.3f} & {pa['n_classes_evaluated']} \\\\")
        lines.append(r"  \midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    (TABLES / "full_benchmark_table.tex").write_text("\n".join(lines))
    print("full_benchmark_table.tex written")

    # MIMIC per-class support table
    r = R.get(("ST-MEM", "MIMIC-IV-ECG"))
    if r:
        pc = r["phase_a"]["per_class"]
        sl = [r"\begin{table}[t]", r"\centering",
              r"\caption{MIMIC-IV-ECG per-class support (2{,}000 records, natural prevalence "
              r"from machine-measurement reports). Classes with $\geq$10 positives and "
              r"$\geq$10 negatives in the test split are included in macro metrics.}",
              r"\label{tab:mimic-support}", r"\small",
              r"\begin{tabular}{lccc}", r"\toprule",
              r"Class & Test positives & Test negatives & Included \\", r"\midrule"]
        names = {"SR": "Sinus rhythm", "AF": "Atrial fibrillation", "RBBB": "RBBB",
                 "LBBB": "LBBB", "LVH": "LVH", "MI": "Myocardial infarction"}
        for cls, d in pc.items():
            inc = r"\checkmark" if d.get("included") else r"--"
            sl.append(f"  {names.get(cls, cls)} & {d['n_pos']} & {d['n_neg']} & {inc} \\\\")
        sl += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (TABLES / "mimic_support_table.tex").write_text("\n".join(sl))
        print("mimic_support_table.tex written")

    # CSV
    rows = []
    for ds in DATASETS:
        for m in MODELS:
            r = R.get((m, ds))
            if r is None:
                continue
            pa = r["phase_a"]
            rows.append({
                "dataset": ds, "model": m, "n_test": r["n_test"],
                "n_classes": pa["n_classes_evaluated"],
                "auroc": f"{pa['macro_auroc']:.4f}",
                "auroc_ci": str(pa.get("macro_auroc_ci")),
                "ece": f"{pa['macro_ece_ew']:.4f}",
                "ece_ci": str(pa.get("macro_ece_ci")),
                "brier": f"{pa['macro_brier']:.4f}", "sce": f"{pa['macro_sce']:.4f}",
                "platt_ece": f"{r['phase_b']['platt']['macro_ece_ew']:.4f}",
                "isotonic_ece": f"{r['phase_b']['isotonic']['macro_ece_ew']:.4f}",
                "isotonic_auroc": f"{r['phase_b']['isotonic']['macro_auroc']:.4f}",
            })
    with open(RESULTS / "paper_results_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"paper_results_table.csv written ({len(rows)} rows)")


if __name__ == "__main__":
    n = len(R)
    print(f"Loaded {n} ok cells: {sorted(set(k[1] for k in R))}")
    fig1(); fig2(); fig3(); fig4(); fig5(); fig_cross(); fig_mimic()
    pareto_and_table()
    tables()
    print("ALL ASSETS DONE")
