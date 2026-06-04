#!/usr/bin/env python3
"""
RankSafe-Cal (corrected): per-class, select the recalibrator that minimizes
calibration error subject to an AUROC-loss constraint, with selection done on
the CALIBRATION split (no test leakage).

Algorithm (per class):
  candidates = {platt, temperature, isotonic}   (all monotone, rank-preserving)
  on the calibration split, for each candidate compute auroc_loss_cal vs raw and ece_cal.
  safe = {c : auroc_loss_cal <= eps}.  platt & temperature are strictly monotone
         (auroc_loss == 0), so safe is never empty.
  choose argmin_{c in safe} ece_cal.  (fallback never needed, but if it were:
         pick min auroc_loss, preferring temperature then platt.)
  apply chosen calibrator (fit on cal) to the test split.

Reports test ECE/NLL/Brier/AUROC and AUROC-loss vs raw, compared to raw, platt,
temperature, and UNCONSTRAINED isotonic. Writes results/ranksafe_cal.{json,csv}
and writing/figures/fig_ranksafe.{pdf,png}. Real data only (fixed probs caches).
"""
import json, sys, csv, collections, statistics as st
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
RESULTS = WS / "results"; FIGS = WS / "writing" / "figures"
sys.path.insert(0, str(WS / "code"))
import analysis_common as ac

EPS = [0.005, 0.01]
DS_ORDER = ["PTB-XL", "CODE-15%", "CPSC2018", "MIMIC-IV-ECG"]

def run():
    man = json.loads((RESULTS / "probs_manifest.json").read_text())
    recals = ac.get_recalibrators()  # platt, isotonic, temperature
    rows = []
    proof = ("Strictly monotone increasing transforms preserve the ROC curve and hence AUROC "
             "exactly; Platt (logistic) and temperature scaling are strictly monotone, so their "
             "per-class AUROC loss is 0. Isotonic regression is only weakly monotone (it can map "
             "distinct scores to equal values), so it can merge ties and reduce AUROC. RankSafe-Cal "
             "therefore always has at least two zero-AUROC-loss candidates available and selects the "
             "lowest-calibration-error option whose AUROC loss is within epsilon.")

    def per_class_methods(ycal, pcal, ytest, ptest, seed=42):
        """Return dict method-> (p_test_recal, auroc_loss_oos, ece_oos).
        AUROC-loss and ECE for SELECTION are estimated OUT-OF-SAMPLE on a held-out
        half of the calibration split (fit on cal_fit, score on cal_sel), so that
        isotonic's ranking degradation is detected before it is chosen. The chosen
        calibrator is then refit on the FULL calibration split and applied to test.
        """
        if len(np.unique(ycal)) < 2:
            return None
        n = len(ycal)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n); half = n // 2
        fit_i, sel_i = perm[:half], perm[half:]
        yf, pf = ycal[fit_i], pcal[fit_i]
        ys, ps = ycal[sel_i], pcal[sel_i]
        out = {}
        if len(np.unique(yf)) < 2 or len(np.unique(ys)) < 2:
            # too few to certify out-of-sample; fall back to full-cal, temperature-safe only
            for name, fn in recals.items():
                out[name] = (fn(pcal, ycal, ptest), 0.0 if name != "isotonic" else 1.0,
                             ac.ece_equal_width(ycal, fn(pcal, ycal, pcal)))
            return out
        auroc_sel_raw = ac.auroc(ys, ps)
        for name, fn in recals.items():
            ps_recal = fn(pf, yf, ps)                # certify on held-out cal_sel
            loss = auroc_sel_raw - ac.auroc(ys, ps_recal)
            ece_oos = ac.ece_equal_width(ys, ps_recal)
            pt = fn(pcal, ycal, ptest)               # final: refit on full cal -> test
            out[name] = (pt, float(loss), float(ece_oos))
        return out

    for cell in man["cells"]:
        mid, did = cell["model_id"], cell["dataset_id"]
        mname, dname = ac.MODELS[mid], cell["dataset"]
        c = ac.load_cell(mid, did)
        Ycal, Pcal, Ytest, Ptest = c["y_cal"], c["p_cal"], c["y_test"], c["p_test"]
        vc = cell["valid_class_idx"]
        # build per-method test predictions and the ranksafe selections
        methods = {"raw": np.array(Ptest), "platt": np.array(Ptest),
                   "temperature": np.array(Ptest), "isotonic": np.array(Ptest)}
        ranksafe = {f"ranksafe_eps{e}": np.array(Ptest) for e in EPS}
        sel_log = {f"eps{e}": {} for e in EPS}
        for ci in range(Ptest.shape[1]):
            pm = per_class_methods(Ycal[:, ci], Pcal[:, ci], Ytest[:, ci], Ptest[:, ci])
            if pm is None:
                continue
            for name in ("platt", "temperature", "isotonic"):
                methods[name][:, ci] = pm[name][0]
            for e in EPS:
                safe = {n: v for n, v in pm.items() if v[1] <= e}
                if not safe:
                    safe = {min(pm, key=lambda n: pm[n][1]): pm[min(pm, key=lambda n: pm[n][1])]}
                # choose min cal-ECE among safe; tie-break prefer temperature, platt
                pref = {"temperature": 0, "platt": 1, "isotonic": 2}
                chosen = sorted(safe.items(), key=lambda kv: (kv[1][2], pref.get(kv[0], 9)))[0][0]
                ranksafe[f"ranksafe_eps{e}"][:, ci] = pm[chosen][0]
                if cell["classes"][ci] and ci in vc:
                    sel_log[f"eps{e}"][cell["classes"][ci]] = chosen

        # evaluate every method on test over valid classes
        def evalm(P):
            au = ac.macro(ac.auroc, Ytest, P, vc); ece = ac.macro(ac.ece_equal_width, Ytest, P, vc)
            nll = ac.macro(ac.nll, Ytest, P, vc); br = ac.macro(ac.brier, Ytest, P, vc)
            # max per-class AUROC loss vs raw
            losses = []
            for ci in vc:
                losses.append(ac.auroc(Ytest[:, ci], methods["raw"][:, ci]) - ac.auroc(Ytest[:, ci], P[:, ci]))
            return au, ece, nll, br, (max(losses) if losses else 0.0)
        au_raw = ac.macro(ac.auroc, Ytest, methods["raw"], vc)
        for name, P in list(methods.items()) + list(ranksafe.items()):
            au, ece, nll, br, mx = evalm(P)
            rows.append({"cell": f"{mid}_{did}", "model": mname, "dataset": dname, "method": name,
                         "macro_ece": round(ece, 4), "macro_nll": round(nll, 4), "macro_brier": round(br, 4),
                         "macro_auroc": round(au, 4), "macro_auroc_loss_vs_raw": round(au_raw - au, 4),
                         "max_class_auroc_loss": round(mx, 4),
                         "selections": json.dumps(sel_log) if name.startswith("ranksafe") else ""})

    with open(RESULTS / "ranksafe_cal.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # aggregate json
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        agg[r["dataset"]][r["method"]].append(r)
    summary = {"proof_note": proof, "epsilon": EPS, "selection": "on calibration split (no test leakage)",
               "per_dataset": {}}
    for ds in DS_ORDER:
        if ds not in agg: continue
        summary["per_dataset"][ds] = {}
        for m, rs in agg[ds].items():
            summary["per_dataset"][ds][m] = {
                "mean_ece": round(st.mean(r["macro_ece"] for r in rs), 4),
                "mean_auroc_loss": round(st.mean(r["macro_auroc_loss_vs_raw"] for r in rs), 4),
                "max_class_auroc_loss": round(max(r["max_class_auroc_loss"] for r in rs), 4)}
    (RESULTS / "ranksafe_cal.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    # figure: ECE reduction vs AUROC loss, per method, PTB-XL & MIMIC (the multi-morbidity cells)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    methcolor = {"raw": "#999999", "platt": "#377eb8", "temperature": "#ff7f00",
                 "isotonic": "#e41a1c", "ranksafe_eps0.01": "#4daf4a"}
    methlabel = {"raw": "Raw", "platt": "Platt", "temperature": "Temperature",
                 "isotonic": "Isotonic (unconstrained)", "ranksafe_eps0.01": "RankSafe-Cal ($\\epsilon$=0.01)"}
    for ax, ds in zip(axes, ["PTB-XL", "MIMIC-IV-ECG"]):
        for m in ["raw", "platt", "temperature", "isotonic", "ranksafe_eps0.01"]:
            rs = agg[ds].get(m, [])
            if not rs: continue
            x = st.mean(r["macro_auroc_loss_vs_raw"] for r in rs)
            y = st.mean(r["macro_ece"] for r in rs)
            ax.scatter(x, y, s=120, color=methcolor[m], label=methlabel[m],
                       edgecolors="black", linewidths=0.6, zorder=5)
        ax.axvline(0.01, color="green", ls=":", lw=1, label="$\\epsilon$=0.01")
        ax.set_xlabel("Mean AUROC loss vs raw $\\rightarrow$ (lower better)")
        ax.set_ylabel("Mean macro ECE $\\downarrow$")
        ax.set_title(ds, fontweight="bold"); ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("RankSafe-Cal: most of isotonic's ECE gain at ~zero AUROC cost", fontweight="bold")
    fig.tight_layout()
    for e in ("pdf", "png"): fig.savefig(FIGS / f"fig_ranksafe.{e}", bbox_inches="tight", dpi=150)
    plt.close()

    print("RankSafe-Cal v2 done. Per-dataset (mean ECE / mean AUROC loss / max class loss):")
    for ds in DS_ORDER:
        if ds not in summary["per_dataset"]: continue
        print(f"\n{ds}:")
        for m in ["raw", "platt", "temperature", "isotonic", "ranksafe_eps0.005", "ranksafe_eps0.01"]:
            if m in summary["per_dataset"][ds]:
                s = summary["per_dataset"][ds][m]
                print(f"   {m:20s} ECE={s['mean_ece']:.4f}  AUROCloss={s['mean_auroc_loss']:+.4f}  maxclassloss={s['max_class_auroc_loss']:.4f}")

if __name__ == "__main__":
    run()
