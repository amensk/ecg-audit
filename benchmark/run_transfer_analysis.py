#!/usr/bin/env python3
"""
Cross-dataset RECALIBRATION TRANSFER for shared diagnostic endpoints.

For each clinically-shared endpoint, each ordered dataset pair (A_source ->
B_target), and each model present on BOTH datasets (skipping M4 on D3), this:
  1. Extracts the one-vs-rest binary problem for the endpoint on A (cal split)
     and B (test + cal splits).
  2. Fits Platt / isotonic / temperature on A's CALIBRATION split.
  3. Reports on B's TEST split:
       ECE_raw, ECE_indomain (calibrator fit on B's own cal split),
       ECE_transferred (calibrator fit on A, applied to B), per method;
       AUROC (raw + transferred + change); Brier and NLL (raw/indomain/transferred).
  4. Transfer success := ECE_transferred <= ECE_indomain*1.25 AND
                          ECE_transferred < ECE_raw.
     Collapse := ECE_transferred > ECE_raw.

Skips any (endpoint,pair,model) where either A-cal, B-cal, or B-test has
<10 positives or <10 negatives for the endpoint class.

Writes results/transfer.json (nested) and results/transfer.csv (one row per
endpoint x method x model x source x target). Makes writing/figures/fig_transfer.pdf(+.png).
Uses ONLY the frozen probs caches via analysis_common.load_cell; never fabricates.
"""
import sys, json, csv
from pathlib import Path
import numpy as np

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
sys.path.insert(0, str(WS / "code"))
import analysis_common as ac  # noqa: E402

RESULTS = WS / "results"
FIGDIR = WS / "writing" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

MIN_POS, MIN_NEG = 10, 10
MODELS = ["M1", "M2", "M3", "M4", "M5", "M6"]
SKIP_CELLS = {("M4", "D3")}
METHODS = ["platt", "isotonic", "temperature"]

# Endpoint -> {dataset_id: class_name} (clinically shared mapping)
ENDPOINTS = {
    "MI":   {"D1": "MI",   "D5": "MI"},
    "AF":   {"D3": "AF",   "D4": "AF",   "D5": "AF"},
    "NORM": {"D1": "NORM", "D3": "NORM", "D4": "Normal", "D5": "SR"},
    "RBBB": {"D3": "RBBB", "D4": "RBBB", "D5": "RBBB"},
}

DATASET_NAME = {"D1": "PTB-XL", "D3": "CODE-15%", "D4": "CPSC2018", "D5": "MIMIC-IV-ECG"}
MODEL_NAME = ac.MODELS  # {"M1":"MERL", ...}


def col_for(cell, class_name):
    classes = list(cell["classes"])
    return classes.index(class_name) if class_name in classes else None


def support_ok(y):
    npos = int(np.sum(y)); nneg = int(len(y) - npos)
    return (npos >= MIN_POS) and (nneg >= MIN_NEG), npos, nneg


def main():
    recal = ac.get_recalibrators()
    # cache loaded cells per (mid,did)
    cellcache = {}

    def get_cell(mid, did):
        key = (mid, did)
        if key not in cellcache:
            try:
                cellcache[key] = ac.load_cell(mid, did)
            except FileNotFoundError:
                cellcache[key] = None
        return cellcache[key]

    rows = []          # flat rows for CSV
    skips = []         # skip log entries
    nested = {}        # endpoint -> method -> list of records (for aggregation)

    for endpoint, dmap in ENDPOINTS.items():
        dsets = list(dmap.keys())
        nested.setdefault(endpoint, {m: [] for m in METHODS})
        # all ordered pairs A->B, A != B
        for A in dsets:
            for B in dsets:
                if A == B:
                    continue
                clsA, clsB = dmap[A], dmap[B]
                for mid in MODELS:
                    if (mid, A) in SKIP_CELLS or (mid, B) in SKIP_CELLS:
                        skips.append({"endpoint": endpoint, "model": mid,
                                      "source": A, "target": B,
                                      "reason": "model excluded on a dataset (e.g. M4 on D3)"})
                        continue
                    cA = get_cell(mid, A)
                    cB = get_cell(mid, B)
                    if cA is None or cB is None:
                        skips.append({"endpoint": endpoint, "model": mid,
                                      "source": A, "target": B,
                                      "reason": "missing probs cache for source or target"})
                        continue
                    iA = col_for(cA, clsA)
                    iB = col_for(cB, clsB)
                    if iA is None or iB is None:
                        skips.append({"endpoint": endpoint, "model": mid,
                                      "source": A, "target": B,
                                      "reason": "endpoint class absent in source/target class list"})
                        continue

                    # binary vectors
                    yA_cal = cA["y_cal"][:, iA].astype(float)
                    pA_cal = cA["p_cal"][:, iA].astype(float)
                    yB_cal = cB["y_cal"][:, iB].astype(float)
                    pB_cal = cB["p_cal"][:, iB].astype(float)
                    yB_te = cB["y_test"][:, iB].astype(float)
                    pB_te = cB["p_test"][:, iB].astype(float)

                    okA, nposA, nnegA = support_ok(yA_cal)
                    okBc, nposBc, nnegBc = support_ok(yB_cal)
                    okBt, nposBt, nnegBt = support_ok(yB_te)
                    if not (okA and okBc and okBt):
                        skips.append({
                            "endpoint": endpoint, "model": mid, "source": A, "target": B,
                            "reason": "insufficient support (<10 pos or <10 neg)",
                            "A_cal_pos": nposA, "A_cal_neg": nnegA,
                            "B_cal_pos": nposBc, "B_cal_neg": nnegBc,
                            "B_test_pos": nposBt, "B_test_neg": nnegBt})
                        continue

                    ece_raw = ac.ece_equal_width(yB_te, pB_te)
                    aece_raw = ac.adaptive_ece(yB_te, pB_te)
                    brier_raw = ac.brier(yB_te, pB_te)
                    nll_raw = ac.nll(yB_te, pB_te)
                    auroc_raw = ac.auroc(yB_te, pB_te)

                    for method in METHODS:
                        fn = recal[method]
                        # in-domain: fit on B's own cal, apply to B test
                        p_in = fn(pB_cal, yB_cal, pB_te)
                        # transferred: fit on A's cal, apply to B test
                        p_tr = fn(pA_cal, yA_cal, pB_te)

                        ece_in = ac.ece_equal_width(yB_te, p_in)
                        ece_tr = ac.ece_equal_width(yB_te, p_tr)
                        aece_in = ac.adaptive_ece(yB_te, p_in)
                        aece_tr = ac.adaptive_ece(yB_te, p_tr)
                        brier_in = ac.brier(yB_te, p_in)
                        brier_tr = ac.brier(yB_te, p_tr)
                        nll_in = ac.nll(yB_te, p_in)
                        nll_tr = ac.nll(yB_te, p_tr)
                        auroc_tr = ac.auroc(yB_te, p_tr)
                        dauroc = (auroc_tr - auroc_raw) if (not np.isnan(auroc_tr)
                                                            and not np.isnan(auroc_raw)) else float("nan")

                        success = bool((ece_tr <= ece_in * 1.25) and (ece_tr < ece_raw))
                        collapse = bool(ece_tr > ece_raw)

                        rec = {
                            "endpoint": endpoint, "method": method,
                            "model_id": mid, "model": MODEL_NAME[mid],
                            "source_id": A, "source": DATASET_NAME[A], "source_class": clsA,
                            "target_id": B, "target": DATASET_NAME[B], "target_class": clsB,
                            "n_source_cal": int(len(yA_cal)),
                            "n_target_cal": int(len(yB_cal)),
                            "n_target_test": int(len(yB_te)),
                            "src_cal_pos": nposA, "tgt_test_pos": nposBt, "tgt_test_neg": nnegBt,
                            "ece_raw": ece_raw, "ece_indomain": ece_in, "ece_transferred": ece_tr,
                            "aece_raw": aece_raw, "aece_indomain": aece_in, "aece_transferred": aece_tr,
                            "brier_raw": brier_raw, "brier_indomain": brier_in, "brier_transferred": brier_tr,
                            "nll_raw": nll_raw, "nll_indomain": nll_in, "nll_transferred": nll_tr,
                            "auroc_raw": auroc_raw, "auroc_transferred": auroc_tr, "auroc_change": dauroc,
                            "transfer_success": success, "collapse": collapse,
                        }
                        rows.append(rec)
                        nested[endpoint][method].append(rec)

    # ── aggregate: per endpoint x method, success/collapse fractions + mean ECEs ──
    aggregate = {}
    for endpoint, methods in nested.items():
        aggregate[endpoint] = {}
        for method, recs in methods.items():
            if not recs:
                aggregate[endpoint][method] = {"n": 0}
                continue
            n = len(recs)
            ns = sum(r["transfer_success"] for r in recs)
            nc = sum(r["collapse"] for r in recs)
            aggregate[endpoint][method] = {
                "n": n,
                "n_success": int(ns), "frac_success": ns / n,
                "n_collapse": int(nc), "frac_collapse": nc / n,
                "mean_ece_raw": float(np.mean([r["ece_raw"] for r in recs])),
                "mean_ece_indomain": float(np.mean([r["ece_indomain"] for r in recs])),
                "mean_ece_transferred": float(np.mean([r["ece_transferred"] for r in recs])),
                "mean_auroc_change": float(np.nanmean([r["auroc_change"] for r in recs])),
                "mean_brier_raw": float(np.mean([r["brier_raw"] for r in recs])),
                "mean_brier_indomain": float(np.mean([r["brier_indomain"] for r in recs])),
                "mean_brier_transferred": float(np.mean([r["brier_transferred"] for r in recs])),
            }

    # ── write JSON ──
    out_json = {
        "description": "Cross-dataset recalibration transfer for shared diagnostic endpoints.",
        "success_criterion": "ECE_transferred <= ECE_indomain*1.25 AND ECE_transferred < ECE_raw",
        "collapse_criterion": "ECE_transferred > ECE_raw",
        "min_pos": MIN_POS, "min_neg": MIN_NEG,
        "endpoint_class_map": ENDPOINTS,
        "ece_n_bins": 15,
        "n_records": len(rows),
        "n_skipped": len(skips),
        "aggregate": aggregate,
        "records": rows,
        "skipped": skips,
    }
    (RESULTS / "transfer.json").write_text(json.dumps(out_json, indent=2))

    # ── write CSV (one row per endpoint x method x model x source x target) ──
    fields = ["endpoint", "method", "model_id", "model",
              "source_id", "source", "source_class",
              "target_id", "target", "target_class",
              "n_source_cal", "n_target_cal", "n_target_test",
              "src_cal_pos", "tgt_test_pos", "tgt_test_neg",
              "ece_raw", "ece_indomain", "ece_transferred",
              "aece_raw", "aece_indomain", "aece_transferred",
              "brier_raw", "brier_indomain", "brier_transferred",
              "nll_raw", "nll_indomain", "nll_transferred",
              "auroc_raw", "auroc_transferred", "auroc_change",
              "transfer_success", "collapse"]
    with open(RESULTS / "transfer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    make_figure(rows)
    print_summary(rows, aggregate, skips)
    return out_json


# ── figure ──────────────────────────────────────────────────────────────────--
def make_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Panel A: MI PTB-XL<->MIMIC (both directions). Panel B: AF among CODE/CPSC/MIMIC.
    # Grouped bars: per (method) cluster, mean ECE across models for raw/indomain/transferred,
    # broken out by ordered pair.
    def pair_label(r):
        return f"{r['source_id']}→{r['target_id']}"

    def gather(endpoint, allowed_pairs=None):
        # returns dict: (pair,method) -> dict of mean raw/indomain/transferred over models
        agg = {}
        for r in rows:
            if r["endpoint"] != endpoint:
                continue
            pl = pair_label(r)
            if allowed_pairs is not None and pl not in allowed_pairs:
                continue
            key = (pl, r["method"])
            agg.setdefault(key, {"raw": [], "in": [], "tr": []})
            agg[key]["raw"].append(r["ece_raw"])
            agg[key]["in"].append(r["ece_indomain"])
            agg[key]["tr"].append(r["ece_transferred"])
        return agg

    mi_pairs = ["D1→D5", "D5→D1"]
    mi_agg = gather("MI", allowed_pairs=mi_pairs)
    af_agg = gather("AF")  # all pairs among D3/D4/D5

    fig, axes = plt.subplots(2, 1, figsize=(15, 11))

    def plot_panel(ax, agg, title):
        # x groups: blocks of methods within each ordered pair, with a small
        # gap between pairs so direction-of-transfer is visually separable.
        pairs = sorted({k[0] for k in agg.keys()})
        if not pairs:
            ax.set_title(title + "  (no data)")
            ax.axis("off")
            return
        keys, xpos = [], []
        cursor = 0.0
        pair_centers, pair_names = [], []
        for p in pairs:
            pkeys = [(p, m) for m in METHODS if (p, m) in agg]
            start = cursor
            for k in pkeys:
                keys.append(k); xpos.append(cursor); cursor += 1.0
            if pkeys:
                pair_centers.append((start + cursor - 1.0) / 2.0); pair_names.append(p)
            cursor += 0.8  # gap between pairs
        xpos = np.array(xpos); width = 0.27
        raw = np.array([np.mean(agg[k]["raw"]) for k in keys])
        ind = np.array([np.mean(agg[k]["in"]) for k in keys])
        tr = np.array([np.mean(agg[k]["tr"]) for k in keys])
        ax.bar(xpos - width, raw, width, label="ECE raw", color="#9e9e9e")
        ax.bar(xpos, ind, width, label="ECE in-domain recal", color="#2e7d32")
        ax.bar(xpos + width, tr, width, label="ECE transferred recal", color="#c62828")
        ax.set_xticks(xpos)
        ax.set_xticklabels([k[1][:4] for k in keys], fontsize=8)
        # pair labels on a secondary row beneath
        ymin = ax.get_ylim()[0]
        for c, nm in zip(pair_centers, pair_names):
            ax.text(c, -0.16 * (raw.max() if raw.size else 1), nm,
                    ha="center", va="top", fontsize=10, fontweight="bold")
        ax.set_ylabel("ECE (mean over models, 15-bin)")
        ax.set_title(title)
        ax.legend(fontsize=9, loc="upper right", ncol=3)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(xpos.min() - 1.0, xpos.max() + 1.0)
        ax.set_ylim(0, max(raw.max(), tr.max()) * 1.30)
        # annotate collapse: transferred > raw
        for xi, k in zip(xpos, keys):
            if np.mean(agg[k]["tr"]) > np.mean(agg[k]["raw"]):
                ax.text(xi + width, np.mean(agg[k]["tr"]),
                        "collapse", rotation=90, va="bottom", ha="center",
                        fontsize=6.5, color="#c62828")

    plot_panel(axes[0], mi_agg, "MI  (PTB-XL ↔ MIMIC-IV) — bars grouped by direction, labelled by method")
    plot_panel(axes[1], af_agg, "AF  (CODE-15% / CPSC2018 / MIMIC-IV, all 6 directions)")
    fig.suptitle("Cross-dataset recalibration transfer: raw vs in-domain-recal vs transferred-recal ECE\n"
                 "(transferred > raw = calibration collapse under dataset shift)",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGDIR / "fig_transfer.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / "fig_transfer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── summary ─────────────────────────────────────────────────────────────────--
def print_summary(rows, aggregate, skips):
    print("=" * 78)
    print("CROSS-DATASET RECALIBRATION TRANSFER — SUMMARY (real cached probs)")
    print("=" * 78)
    print(f"Records: {len(rows)}   Skipped (endpoint,pair,model): {len(skips)}")
    print(f"Success := ECE_transferred <= 1.25*ECE_indomain AND ECE_transferred < ECE_raw")
    print(f"Collapse := ECE_transferred > ECE_raw\n")

    # Highlighted per-method examples for MI PTB-XL->MIMIC
    print("--- MI  PTB-XL(D1) -> MIMIC(D5), per model & method ---")
    for r in sorted(rows, key=lambda r: (r["model_id"], METHODS.index(r["method"]))):
        if r["endpoint"] == "MI" and r["source_id"] == "D1" and r["target_id"] == "D5":
            tag = "SUCCESS" if r["transfer_success"] else ("COLLAPSE" if r["collapse"] else "partial")
            print(f"  {r['model']:<11} {r['method']:<11} "
                  f"ECE_raw={r['ece_raw']:.4f} indomain={r['ece_indomain']:.4f} "
                  f"transferred={r['ece_transferred']:.4f}  ({tag})  "
                  f"dAUROC={r['auroc_change']:+.4f}")

    print("\n--- MI  MIMIC(D5) -> PTB-XL(D1), per model & method ---")
    for r in sorted(rows, key=lambda r: (r["model_id"], METHODS.index(r["method"]))):
        if r["endpoint"] == "MI" and r["source_id"] == "D5" and r["target_id"] == "D1":
            tag = "SUCCESS" if r["transfer_success"] else ("COLLAPSE" if r["collapse"] else "partial")
            print(f"  {r['model']:<11} {r['method']:<11} "
                  f"ECE_raw={r['ece_raw']:.4f} indomain={r['ece_indomain']:.4f} "
                  f"transferred={r['ece_transferred']:.4f}  ({tag})  "
                  f"dAUROC={r['auroc_change']:+.4f}")

    print("\n--- Aggregate: fraction success / collapse, mean ECE (endpoint x method) ---")
    for endpoint in ENDPOINTS:
        for method in METHODS:
            a = aggregate[endpoint][method]
            if a.get("n", 0) == 0:
                print(f"  {endpoint:<5} {method:<11}  (no supported (model,pair))")
                continue
            print(f"  {endpoint:<5} {method:<11} n={a['n']:<3} "
                  f"success={a['frac_success']*100:5.1f}% collapse={a['frac_collapse']*100:5.1f}%  "
                  f"meanECE raw={a['mean_ece_raw']:.4f} in={a['mean_ece_indomain']:.4f} "
                  f"tr={a['mean_ece_transferred']:.4f}")

    if skips:
        print("\n--- Skipped (insufficient support / missing / class absent) ---")
        # group by reason
        from collections import Counter
        cnt = Counter(s["reason"] for s in skips)
        for reason, n in cnt.items():
            print(f"  [{n}] {reason}")
        # detail the support-related skips (most informative)
        for s in skips:
            if "support" in s["reason"]:
                print(f"     {s['endpoint']} {s.get('model','?')} {s['source']}->{s['target']}: "
                      f"A_cal(+{s.get('A_cal_pos','?')}/-{s.get('A_cal_neg','?')}) "
                      f"B_cal(+{s.get('B_cal_pos','?')}/-{s.get('B_cal_neg','?')}) "
                      f"B_test(+{s.get('B_test_pos','?')}/-{s.get('B_test_neg','?')})")

    print("\nWrote results/transfer.json, results/transfer.csv, "
          "writing/figures/fig_transfer.pdf(+.png)")


if __name__ == "__main__":
    main()
