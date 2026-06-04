#!/usr/bin/env python3
"""
Expanded Decision-Curve Analysis (net benefit) on REAL cached probabilities.

Goes beyond the single existing ST-MEM/PTB-XL/MI curve: every model present is
analysed one-vs-rest for the MI and AF endpoints across all supported datasets,
for RAW, isotonic- and Platt-recalibrated probabilities. Recalibrators are fit on
the CAL split and applied to the TEST split (no leakage). Treat-all / treat-none
references are included. Integrated (trapezoidal) net benefit over the clinically
plausible threshold range is reported, plus the area above treat-all.

Inputs  : results/probs/{MID}_{DID}.npz via analysis_common.load_cell
Outputs : results/dca_expanded.json
          results/dca_expanded.csv
          writing/figures/fig_dca_mi.pdf (+png)
          writing/figures/fig_dca_af.pdf (+png)

Real data only. Manuscript / probs caches / benchmark_v2_results.json untouched.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
RESULTS = WS / "results"
FIGS = WS / "writing" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WS / "code"))
import analysis_common as ac  # noqa: E402

# ── endpoint / dataset plan ────────────────────────────────────────────────────
# endpoint -> list of (dataset_id, class_name)
PLAN = {
    "MI": [("D1", "MI"), ("D5", "MI")],
    "AF": [("D3", "AF"), ("D4", "AF"), ("D5", "AF")],
}
# clinically plausible threshold ranges (inclusive of upper bound), step 0.01
RANGES = {
    "MI": (0.05, 0.30),
    "AF": (0.05, 0.50),
}
ALL_MODELS = ["M1", "M2", "M3", "M4", "M5", "M6"]
MODEL_NAME = {"M1": "MERL", "M2": "ST-MEM", "M3": "HuBERT-ECG",
              "M4": "ECGFM-KED", "M5": "ECG-FM", "M6": "S4D"}
DATASET_NAME = {"D1": "PTB-XL", "D3": "CODE-15%", "D4": "CPSC2018", "D5": "MIMIC-IV-ECG"}
SKIP = {("M4", "D3")}  # documented exclusion

KEY_THRESH = [0.10, 0.15, 0.20]


def thresholds_for(endpoint):
    lo, hi = RANGES[endpoint]
    # arange can drop the endpoint due to fp error; add a small epsilon
    return np.round(np.arange(lo, hi + 1e-9, 0.01), 4)


def net_benefit(y, p, t):
    """NB(t) = TP/n - FP/n * (t/(1-t)). Matches ecg_audit.net_benefit."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    n = len(y)
    if t >= 1.0:
        return 0.0
    pred = p >= t
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    return tp / n - fp / n * (t / (1.0 - t))


def treat_all_curve(y, thr):
    y = np.asarray(y, float)
    prev = float(y.mean())
    return np.array([prev - (1 - prev) * (t / (1 - t)) if t < 1 else 0.0 for t in thr])


def integrated(curve, thr):
    """Trapezoidal area of curve over the threshold range."""
    return float(np.trapezoid(np.asarray(curve, float), np.asarray(thr, float)))


def nearest_idx(thr, target):
    return int(np.argmin(np.abs(thr - target)))


def build():
    recals = ac.get_recalibrators()
    platt, iso = recals["platt"], recals["isotonic"]

    results = {}          # nested endpoint/dataset/model/method
    flat_rows = []        # for csv
    # cache loaded cells + selection metrics for the figures
    fig_cache = {}        # (endpoint, did) -> dict with thr, per-model curves, selection

    for endpoint, dlist in PLAN.items():
        thr = thresholds_for(endpoint)
        thr_keys = [f"{t:.2f}" for t in thr]
        results.setdefault(endpoint, {})

        for did, cls in dlist:
            dnode = results[endpoint].setdefault(did, {})
            present_models = []
            per_model_curves = {}   # mid -> {raw,isotonic,platt,treat_all, auroc, ece, treat_all_int}
            treat_all_int_ref = None

            for mid in ALL_MODELS:
                if (mid, did) in SKIP:
                    continue
                try:
                    cell = ac.load_cell(mid, did)
                except FileNotFoundError:
                    continue
                classes = list(cell["classes"])
                if cls not in classes:
                    continue
                ci = classes.index(cls)

                y_test = cell["y_test"][:, ci].astype(float)
                p_test = cell["p_test"][:, ci].astype(float)
                y_cal = cell["y_cal"][:, ci].astype(float)
                p_cal = cell["p_cal"][:, ci].astype(float)

                # support guard (matches probs_manifest min_pos/min_neg policy)
                npos, nneg = int(y_test.sum()), int(len(y_test) - y_test.sum())
                if npos < ac.MIN_POS or nneg < ac.MIN_NEG:
                    continue

                # recalibrate: fit on cal, apply to test (signature: p_cal,y_cal,p_test)
                p_platt = np.asarray(platt(p_cal, y_cal, p_test), float)
                p_iso = np.asarray(iso(p_cal, y_cal, p_test), float)

                # net-benefit curves
                curve_raw = np.array([net_benefit(y_test, p_test, t) for t in thr])
                curve_platt = np.array([net_benefit(y_test, p_platt, t) for t in thr])
                curve_iso = np.array([net_benefit(y_test, p_iso, t) for t in thr])
                curve_all = treat_all_curve(y_test, thr)
                curve_none = np.zeros_like(thr)

                int_raw = integrated(curve_raw, thr)
                int_platt = integrated(curve_platt, thr)
                int_iso = integrated(curve_iso, thr)
                int_all = integrated(curve_all, thr)
                if treat_all_int_ref is None:
                    treat_all_int_ref = int_all

                # selection metrics (computed from the SAME real test probs)
                au = ac.auroc(y_test, p_test)
                ec = ac.ece_equal_width(y_test, p_test)

                method_curves = {
                    "raw": (curve_raw, int_raw),
                    "isotonic": (curve_iso, int_iso),
                    "platt": (curve_platt, int_platt),
                    "treat_all": (curve_all, int_all),
                }
                # treat_none stored once per (endpoint,dataset,model) as reference
                mnode = dnode.setdefault(mid, {})
                for method, (curve, integ) in method_curves.items():
                    mnode[method] = {
                        "nb_curve": {k: float(v) for k, v in zip(thr_keys, curve)},
                        "integrated_nb": float(integ),
                        "integrated_vs_treat_all": float(integ - int_all),
                    }
                    flat_rows.append({
                        "endpoint": endpoint,
                        "dataset": DATASET_NAME[did],
                        "model": MODEL_NAME[mid],
                        "method": method,
                        "integrated_nb": round(float(integ), 6),
                        "integrated_vs_treat_all": round(float(integ - int_all), 6),
                        "nb_at_0.10": round(float(curve[nearest_idx(thr, 0.10)]), 6),
                        "nb_at_0.15": round(float(curve[nearest_idx(thr, 0.15)]), 6),
                        "nb_at_0.20": round(float(curve[nearest_idx(thr, 0.20)]), 6),
                    })
                # treat_none reference row (constant 0) for completeness
                mnode["treat_none"] = {
                    "nb_curve": {k: 0.0 for k in thr_keys},
                    "integrated_nb": 0.0,
                    "integrated_vs_treat_all": float(0.0 - int_all),
                }
                flat_rows.append({
                    "endpoint": endpoint,
                    "dataset": DATASET_NAME[did],
                    "model": MODEL_NAME[mid],
                    "method": "treat_none",
                    "integrated_nb": 0.0,
                    "integrated_vs_treat_all": round(float(0.0 - int_all), 6),
                    "nb_at_0.10": 0.0,
                    "nb_at_0.15": 0.0,
                    "nb_at_0.20": 0.0,
                })

                present_models.append(mid)
                per_model_curves[mid] = {
                    "raw": curve_raw, "isotonic": curve_iso, "platt": curve_platt,
                    "treat_all": curve_all, "treat_none": curve_none,
                    "auroc": au, "ece": ec, "n_pos": npos, "prev": float(y_test.mean()),
                    "int_vs_all_raw": int_raw - int_all,
                    "int_vs_all_iso": int_iso - int_all,
                    "int_vs_all_platt": int_platt - int_all,
                }

            if not present_models:
                print(f"  [skip] {endpoint} / {DATASET_NAME[did]}: no model support")
                continue

            # model selection for figures
            best_au = max(present_models, key=lambda m: (per_model_curves[m]["auroc"]
                          if not np.isnan(per_model_curves[m]["auroc"]) else -1))
            best_ece = min(present_models, key=lambda m: per_model_curves[m]["ece"])
            fig_cache[(endpoint, did)] = {
                "thr": thr, "curves": per_model_curves,
                "best_auroc": best_au, "best_ece": best_ece,
                "models": present_models, "cls": cls,
            }
            print(f"  [{endpoint}/{DATASET_NAME[did]}] models={[MODEL_NAME[m] for m in present_models]} "
                  f"best-AUROC={MODEL_NAME[best_au]}({per_model_curves[best_au]['auroc']:.3f}) "
                  f"best-ECE={MODEL_NAME[best_ece]}({per_model_curves[best_ece]['ece']:.3f})")

    # ── write JSON + CSV ────────────────────────────────────────────────────────
    meta = {
        "_meta": {
            "description": "Expanded decision-curve analysis (net benefit) on REAL cached "
                           "linear-probe probabilities. NB(t)=TP/n - FP/n*(t/(1-t)) on TEST "
                           "split; isotonic & Platt recalibrators fit on CAL split. "
                           "integrated_nb = trapezoidal area of NB(t) over the threshold range; "
                           "integrated_vs_treat_all = integrated_nb(method) - integrated_nb(treat_all).",
            "source": "results/probs/{MID}_{DID}.npz",
            "thresholds": {ep: {"lo": RANGES[ep][0], "hi": RANGES[ep][1], "step": 0.01}
                           for ep in PLAN},
            "endpoints": {ep: [{"dataset_id": d, "dataset": DATASET_NAME[d], "class": c}
                               for d, c in dl] for ep, dl in PLAN.items()},
            "excluded_cells": ["M4 x D3 (ECGFM-KED on CODE-15%)"],
        }
    }
    out_json = {**meta, **results}
    (RESULTS / "dca_expanded.json").write_text(json.dumps(out_json, indent=2))

    df = pd.DataFrame(flat_rows, columns=[
        "endpoint", "dataset", "model", "method",
        "integrated_nb", "integrated_vs_treat_all",
        "nb_at_0.10", "nb_at_0.15", "nb_at_0.20"])
    df.to_csv(RESULTS / "dca_expanded.csv", index=False)
    print(f"\nWrote results/dca_expanded.json ({len(results)} endpoints) "
          f"and results/dca_expanded.csv ({len(df)} rows)")

    make_figures(fig_cache)
    return out_json, df, fig_cache


# ── figures ─────────────────────────────────────────────────────────────────--

C_RAW = "#1f4e79"      # raw model
C_ISO = "#e67e22"      # isotonic
C_ALL = "#7f8c8d"      # treat-all
C_NONE = "#000000"     # treat-none


def _plot_panel(ax, info, mid, label_prefix, endpoint):
    thr = info["thr"]
    cd = info["curves"][mid]
    x = thr * 100
    ax.plot(x, cd["raw"], color=C_RAW, lw=2.0, label="Model (raw)")
    ax.plot(x, cd["isotonic"], color=C_ISO, lw=2.0, ls="--", label="Model (isotonic)")
    ax.plot(x, cd["treat_all"], color=C_ALL, lw=1.4, ls="-", alpha=0.9, label="Treat all")
    ax.plot(x, cd["treat_none"], color=C_NONE, lw=1.0, ls=":", alpha=0.7, label="Treat none")
    ax.axhline(0, color="k", lw=0.6, alpha=0.25)
    ax.set_xlabel("Threshold probability (%)", fontsize=10)
    ax.set_ylabel("Net benefit", fontsize=10)
    ax.set_xlim(x.min(), x.max())
    ax.grid(True, alpha=0.25)
    ax.set_title(f"{label_prefix}\n{MODEL_NAME[mid]}  "
                 f"(AUROC={cd['auroc']:.2f}, ECE={cd['ece']:.2f}, prev={cd['prev']:.2f})",
                 fontsize=9.5)


def make_figures(fig_cache):
    # ---- MI figure: best-AUROC and best-ECE on PTB-XL (D1) and MIMIC (D5) ----
    mi_specs = []
    for did in ["D1", "D5"]:
        info = fig_cache.get(("MI", did))
        if info is None:
            continue
        dn = DATASET_NAME[did]
        mi_specs.append((info, info["best_auroc"], f"{dn} - best AUROC"))
        # avoid duplicate panel if best-AUROC == best-ECE
        if info["best_ece"] != info["best_auroc"]:
            mi_specs.append((info, info["best_ece"], f"{dn} - best ECE"))
        else:
            mi_specs.append((info, info["best_ece"], f"{dn} - best ECE (=AUROC)"))

    if mi_specs:
        ncol = 2
        nrow = int(np.ceil(len(mi_specs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.4 * nrow), squeeze=False)
        for ax, (info, mid, lab) in zip(axes.ravel(), mi_specs):
            _plot_panel(ax, info, mid, lab, "MI")
        for ax in axes.ravel()[len(mi_specs):]:
            ax.axis("off")
        axes.ravel()[0].legend(fontsize=8.5, loc="upper right", framealpha=0.9)
        fig.suptitle("Decision-curve analysis - MI endpoint (one-vs-rest)\n"
                     "Net benefit on TEST split: raw vs isotonic-recalibrated, with treat-all / treat-none",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        for ext in ("pdf", "png"):
            fig.savefig(FIGS / f"fig_dca_mi.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote writing/figures/fig_dca_mi.pdf (+png) [{len(mi_specs)} panels]")

    # ---- AF figure: best-AUROC and best-ECE on CODE-15% (D3), CPSC (D4), MIMIC (D5) ----
    af_specs = []
    for did in ["D3", "D4", "D5"]:
        info = fig_cache.get(("AF", did))
        if info is None:
            continue
        dn = DATASET_NAME[did]
        af_specs.append((info, info["best_auroc"], f"{dn} - best AUROC"))
        if info["best_ece"] != info["best_auroc"]:
            af_specs.append((info, info["best_ece"], f"{dn} - best ECE"))
        else:
            af_specs.append((info, info["best_ece"], f"{dn} - best ECE (=AUROC)"))

    if af_specs:
        ncol = 2
        nrow = int(np.ceil(len(af_specs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.4 * nrow), squeeze=False)
        for ax, (info, mid, lab) in zip(axes.ravel(), af_specs):
            _plot_panel(ax, info, mid, lab, "AF")
        for ax in axes.ravel()[len(af_specs):]:
            ax.axis("off")
        axes.ravel()[0].legend(fontsize=8.5, loc="upper right", framealpha=0.9)
        fig.suptitle("Decision-curve analysis - AF endpoint (one-vs-rest)\n"
                     "Net benefit on TEST split: raw vs isotonic-recalibrated, with treat-all / treat-none",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        for ext in ("pdf", "png"):
            fig.savefig(FIGS / f"fig_dca_af.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote writing/figures/fig_dca_af.pdf (+png) [{len(af_specs)} panels]")


# ── printed summary ─────────────────────────────────────────────────────────--

def summarize(out_json, df, fig_cache):
    print("\n" + "=" * 78)
    print("EXPANDED DECISION-CURVE ANALYSIS - SUMMARY (real cached probabilities)")
    print("=" * 78)

    for endpoint in PLAN:
        print(f"\n### {endpoint} endpoint  (threshold range {RANGES[endpoint][0]:.2f}-"
              f"{RANGES[endpoint][1]:.2f}, step 0.01)")
        for did, cls in PLAN[endpoint]:
            info = fig_cache.get((endpoint, did))
            dn = DATASET_NAME[did]
            if info is None:
                print(f"  - {dn}: NO SUPPORT (endpoint absent / insufficient positives) - skipped")
                continue
            sub = df[(df.endpoint == endpoint) & (df.dataset == dn)]
            # rank models by integrated_vs_treat_all (raw)
            raw = sub[sub.method == "raw"].sort_values("integrated_vs_treat_all", ascending=False)
            n_pos = info["curves"][info["models"][0]]["n_pos"]
            prev = info["curves"][info["models"][0]]["prev"]
            print(f"  - {dn} (prev={prev:.3f}, test_pos={n_pos}, "
                  f"{len(info['models'])} models): integrated NB above treat-all "
                  f"[raw | iso | platt]:")
            for _, r in raw.iterrows():
                m = r["model"]
                isov = sub[(sub.model == m) & (sub.method == "isotonic")]["integrated_vs_treat_all"].iloc[0]
                plav = sub[(sub.model == m) & (sub.method == "platt")]["integrated_vs_treat_all"].iloc[0]
                tag = []
                mid = [k for k, v in MODEL_NAME.items() if v == m][0]
                if mid == info["best_auroc"]:
                    tag.append("best-AUROC")
                if mid == info["best_ece"]:
                    tag.append("best-ECE")
                tagstr = (" <- " + ",".join(tag)) if tag else ""
                print(f"        {m:<11s}  raw={r['integrated_vs_treat_all']:+.4f}  "
                      f"iso={isov:+.4f}  platt={plav:+.4f}{tagstr}")

    # calibration -> decisions narrative with real numbers
    print("\n" + "-" * 78)
    print("CALIBRATION -> DECISIONS")
    print("-" * 78)
    rows = df[df.method.isin(["raw", "isotonic"])].copy()
    # aggregate iso - raw deltas in integrated_vs_treat_all
    deltas = []
    for (ep, dn, m), g in rows.groupby(["endpoint", "dataset", "model"]):
        try:
            rraw = g[g.method == "raw"]["integrated_vs_treat_all"].iloc[0]
            riso = g[g.method == "isotonic"]["integrated_vs_treat_all"].iloc[0]
        except IndexError:
            continue
        deltas.append((ep, dn, m, riso - rraw, rraw, riso))
    dd = pd.DataFrame(deltas, columns=["endpoint", "dataset", "model", "iso_minus_raw", "raw", "iso"])
    for ep in PLAN:
        epd = dd[dd.endpoint == ep]
        if epd.empty:
            continue
        mean_delta = epd["iso_minus_raw"].mean()
        n_help = int((epd["iso_minus_raw"] > 1e-4).sum())
        n_hurt = int((epd["iso_minus_raw"] < -1e-4).sum())
        n_tie = len(epd) - n_help - n_hurt
        best = epd.loc[epd["iso_minus_raw"].idxmax()]
        worst = epd.loc[epd["iso_minus_raw"].idxmin()]
        print(f"  {ep}: isotonic vs raw integrated-NB-above-treat-all: "
              f"mean Delta={mean_delta:+.4f} over {len(epd)} model-datasets "
              f"(iso better: {n_help}, worse: {n_hurt}, ~tie: {n_tie})")
        print(f"       largest gain : {best['model']}/{best['dataset']} "
              f"Delta={best['iso_minus_raw']:+.4f} (raw {best['raw']:+.4f} -> iso {best['iso']:+.4f})")
        print(f"       largest loss : {worst['model']}/{worst['dataset']} "
              f"Delta={worst['iso_minus_raw']:+.4f} (raw {worst['raw']:+.4f} -> iso {worst['iso']:+.4f})")

    # where does raw-vs-iso matter most across thresholds (pointwise abs gap)
    print("\n  Thresholds where raw vs isotonic NB differ MOST (|NB_iso - NB_raw|, top cells):")
    gap_rows = []
    for endpoint in PLAN:
        node = out_json.get(endpoint, {})
        for did, cls in PLAN[endpoint]:
            dn = DATASET_NAME[did]
            for mid, mnode in node.get(did, {}).items():
                if "raw" not in mnode or "isotonic" not in mnode:
                    continue
                raw_c = mnode["raw"]["nb_curve"]
                iso_c = mnode["isotonic"]["nb_curve"]
                best_t, best_gap = None, -1
                for tk in raw_c:
                    g = abs(iso_c[tk] - raw_c[tk])
                    if g > best_gap:
                        best_gap, best_t = g, tk
                gap_rows.append((endpoint, dn, MODEL_NAME[mid], best_t,
                                 iso_c[best_t] - raw_c[best_t], best_gap))
    gdf = pd.DataFrame(gap_rows, columns=["endpoint", "dataset", "model", "t", "signed", "absgap"])
    for ep in PLAN:
        g = gdf[gdf.endpoint == ep].sort_values("absgap", ascending=False).head(4)
        for _, r in g.iterrows():
            print(f"       {ep:<3s} {r['model']:<11s}/{r['dataset']:<13s} "
                  f"max gap at t={float(r['t']):.2f}: NB_iso-NB_raw={r['signed']:+.4f}")


if __name__ == "__main__":
    oj, df, fc = build()
    summarize(oj, df, fc)
