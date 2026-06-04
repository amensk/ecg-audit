#!/usr/bin/env python3
"""Emit LaTeX tables + robustness figure for the extended analyses, from artifacts."""
import json, csv, collections, statistics as st
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
R = WS / "results"; T = WS / "writing" / "tables"; F = WS / "writing" / "figures"
T.mkdir(parents=True, exist_ok=True)
DSO = ["PTB-XL", "CODE-15%", "CPSC2018", "MIMIC-IV-ECG"]
def esc(s): return str(s).replace("%", r"\%")

# ── 1. extended metrics (per dataset means of NLL, adaptive ECE, slope, intercept) ──
def t_extended():
    rows = list(csv.DictReader(open(R / "extended_metrics.csv")))
    by = collections.defaultdict(list)
    for r in rows: by[r["dataset"]].append(r)
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Extended calibration metrics beyond ECE (macro over valid classes, test split), "
         r"averaged per dataset. NLL = negative log-likelihood; calibration slope (ideal $=1$) and "
         r"intercept (ideal $=0$) from Cox recalibration of labels on logits. Slopes far below 1 and "
         r"negative intercepts indicate systematic over-confidence that worsens on PTB-XL.}",
         r"\label{tab:extended-metrics}", r"\small", r"\begin{tabular}{lccccc}", r"\toprule",
         r"Dataset & ECE & adaptive ECE & NLL & cal.\ slope & cal.\ intercept \\", r"\midrule"]
    for ds in DSO:
        rs = by[ds]
        f = lambda k: st.mean(float(r[k]) for r in rs)
        L.append(f"  {esc(ds)} & {f('macro_ece'):.3f} & {f('macro_adaptive_ece'):.3f} & "
                 f"{f('macro_nll'):.3f} & {f('macro_cal_slope'):.3f} & {f('macro_cal_intercept'):+.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (T / "tab_extended_metrics.tex").write_text("\n".join(L)); print("tab_extended_metrics")

# ── 2. RankSafe-Cal ──
def t_ranksafe():
    s = json.load(open(R / "ranksafe_cal.json"))["summary"]["per_dataset"]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{RankSafe-Cal vs.\ baselines (mean over models per dataset). RankSafe-Cal selects, "
         r"per class, the lowest-ECE recalibrator whose \emph{out-of-sample} AUROC loss is within "
         r"$\epsilon{=}0.01$ (certified on a held-out half of the calibration split). It recovers most "
         r"of unconstrained isotonic's ECE reduction at a fraction of the AUROC cost. "
         r"Max class loss = worst per-class test AUROC drop vs.\ raw.}",
         r"\label{tab:ranksafe}", r"\small", r"\begin{tabular}{llccc}", r"\toprule",
         r"Dataset & Method & ECE $\downarrow$ & AUROC loss $\downarrow$ & max class loss $\downarrow$ \\", r"\midrule"]
    meths = [("raw", "Raw"), ("platt", "Platt"), ("temperature", "Temperature"),
             ("isotonic", "Isotonic (unconstr.)"), ("ranksafe_eps0.01", r"\textbf{RankSafe-Cal}")]
    for ds in DSO:
        if ds not in s: continue
        for i, (m, lab) in enumerate(meths):
            if m not in s[ds]: continue
            d = s[ds][m]; dsn = esc(ds) if i == 0 else ""
            L.append(f"  {dsn} & {lab} & {d['mean_ece']:.3f} & {d['mean_auroc_loss']:+.4f} & {d['max_class_auroc_loss']:.3f} \\\\")
        L.append(r"  \midrule")
    L[-1] = r"\bottomrule"; L += [r"\end{tabular}", r"\end{table}"]
    (T / "tab_ranksafe.tex").write_text("\n".join(L)); print("tab_ranksafe")

# ── 3. transfer (MI, AF aggregates) ──
def t_transfer():
    rows = list(csv.DictReader(open(R / "transfer.csv")))
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["method"] != "isotonic": continue
        agg[r["endpoint"]]["raw"].append(float(r["ece_raw"]))
        agg[r["endpoint"]]["indomain"].append(float(r["ece_indomain"]))
        agg[r["endpoint"]]["transferred"].append(float(r["ece_transferred"]))
        agg[r["endpoint"]]["success"].append(1.0 if r["transfer_success"] in ("True","true","1") else 0.0)
        agg[r["endpoint"]]["collapse"].append(1.0 if r["collapse"] in ("True","true","1") else 0.0)
        agg[r["endpoint"]]["aurocchg"].append(float(r["auroc_change"]))
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Cross-dataset calibration \emph{transfer} (isotonic, one-vs-rest). A calibrator fit "
         r"on a source dataset's calibration split is applied to a target dataset's test split. "
         r"ECE values are means over (model, source$\to$target) pairs. Transfer success = transferred "
         r"ECE within 25\% of the in-domain recalibrated ECE and below raw. Calibration transfers "
         r"partially and endpoint-dependently---it does not fully substitute for local recalibration.}",
         r"\label{tab:transfer}", r"\small", r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Endpoint & $n$ pairs & raw ECE & in-dom.\ ECE & transf.\ ECE & success & mean $\Delta$AUROC \\", r"\midrule"]
    names = {"MI": "MI (PTB-XL$\\leftrightarrow$MIMIC)", "AF": "AF (CODE/CPSC/MIMIC)",
             "NORM": "Normal (all)", "RBBB": "RBBB (CODE/CPSC/MIMIC)"}
    for ep in ["MI", "AF", "NORM", "RBBB"]:
        if ep not in agg: continue
        a = agg[ep]; n = len(a["raw"])
        L.append(f"  {names.get(ep,ep)} & {n} & {st.mean(a['raw']):.3f} & {st.mean(a['indomain']):.3f} & "
                 f"{st.mean(a['transferred']):.3f} & {100*st.mean(a['success']):.0f}\\% & {st.mean(a['aurocchg']):+.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (T / "tab_transfer.tex").write_text("\n".join(L)); print("tab_transfer")

# ── 4. conformal (alpha=0.1: coverage & set size by method, per endpoint) ──
def t_conformal():
    rows = list(csv.DictReader(open(R / "conformal.csv")))
    a = [r for r in rows if str(r.get("alpha")) in ("0.1", "0.10")]
    agg = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    def num(r,k):
        for c in (k, k+"_mean", "avg_"+k):
            if c in r:
                try: return float(r[c])
                except: pass
        return None
    # detect column names
    cov_key = next((k for k in rows[0] if "cover" in k.lower()), None)
    size_key = next((k for k in rows[0] if "size" in k.lower()), None)
    ep_key = next((k for k in rows[0] if k.lower() in ("endpoint","target","class")), "endpoint")
    for r in a:
        ep=r.get(ep_key,"?"); m=r.get("method","?")
        try:
            agg[ep][m]["cov"].append(float(r[cov_key])); agg[ep][m]["size"].append(float(r[size_key]))
        except: pass
    L=[r"\begin{table}[t]", r"\centering",
       r"\caption{Split-conformal prediction (one-vs-rest, target coverage $1-\alpha=0.90$), mean over "
       r"models/datasets. Empirical coverage meets the target for all methods (finite-sample guarantee); "
       r"recalibration mainly affects set-size efficiency. Temperature scaling is order-preserving so its "
       r"conformal sets are identical to raw.}",
       r"\label{tab:conformal}", r"\small", r"\begin{tabular}{llcc}", r"\toprule",
       r"Endpoint & Method & coverage & avg.\ set size \\", r"\midrule"]
    for ep in ["MI","AF","NORM"]:
        if ep not in agg: continue
        for i,m in enumerate(["raw","platt","isotonic","temperature"]):
            if m not in agg[ep]: continue
            cov=st.mean(agg[ep][m]["cov"]); sz=st.mean(agg[ep][m]["size"])
            L.append(f"  {ep if i==0 else ''} & {m} & {cov:.3f} & {sz:.3f} \\\\")
        L.append(r"  \midrule")
    L[-1]=r"\bottomrule"; L+=[r"\end{tabular}", r"\end{table}"]
    (T/"tab_conformal.tex").write_text("\n".join(L)); print("tab_conformal")

# ── 5. robustness (PTB-XL & MIMIC, mean±std over 10 seeds) + figure ──
def t_robustness():
    rob = json.load(open(R / "robustness.json"))
    dsets = rob["datasets"]   # D1, D5 -> {dataset, models: {M1..: {model, per_seed:{auroc,ece,...}}}}
    did_for = {"PTB-XL": "D1", "MIMIC-IV-ECG": "D5"}
    L=[r"\begin{table}[t]", r"\centering",
       r"\caption{Multi-seed robustness (10 patient-level resplits, seeds 0--9): mean$\pm$std macro "
       r"AUROC and ECE on the two multi-morbidity datasets. The PTB-XL discrimination--calibration "
       r"tradeoff (S4D best-calibrated) holds in 10/10 seeds; ECG-FM trails MERL/ST-MEM on MIMIC in "
       r"all seeds but is statistically tied with the S4D baseline.}",
       r"\label{tab:robustness}", r"\small", r"\begin{tabular}{llcc}", r"\toprule",
       r"Dataset & Model & AUROC (mean$\pm$std) & ECE (mean$\pm$std) \\", r"\midrule"]
    figdata = {}
    order = ["M1","M2","M3","M4","M5","M6"]
    for ds in ["PTB-XL", "MIMIC-IV-ECG"]:
        block = dsets.get(did_for[ds], {}).get("models", {})
        figdata[ds] = {}
        first=True
        for mid in order:
            md = block.get(mid)
            if not md: continue
            name = md.get("model", mid); ps = md.get("per_seed", {})
            au = ps.get("auroc", []); ec = ps.get("ece", [])
            if not au: continue
            au_m,au_s = float(np.mean(au)), float(np.std(au))
            ec_m,ec_s = float(np.mean(ec)), float(np.std(ec))
            figdata[ds][name]=(au_m,au_s,ec_m,ec_s)
            L.append(f"  {esc(ds) if first else ''} & {name} & {au_m:.3f}$\\pm${au_s:.3f} & {ec_m:.3f}$\\pm${ec_s:.3f} \\\\")
            first=False
        L.append(r"  \midrule")
    L[-1]=r"\bottomrule"; L+=[r"\end{tabular}", r"\end{table}"]
    (T/"tab_robustness.tex").write_text("\n".join(L)); print("tab_robustness")
    # figure
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    models=["MERL","ST-MEM","HuBERT-ECG","ECGFM-KED","ECG-FM","S4D"]
    for ax,ds in zip(axes,["PTB-XL","MIMIC-IV-ECG"]):
        xs=np.arange(len(models))
        au=[figdata[ds].get(m,(np.nan,)*4)[0] for m in models]
        aus=[figdata[ds].get(m,(np.nan,)*4)[1] for m in models]
        ec=[figdata[ds].get(m,(np.nan,)*4)[2] for m in models]
        ecs=[figdata[ds].get(m,(np.nan,)*4)[3] for m in models]
        ax.errorbar(xs-0.0,au,yerr=aus,fmt="o",color="#2c7bb6",label="AUROC",capsize=3)
        ax2=ax.twinx()
        ax2.errorbar(xs,ec,yerr=ecs,fmt="s",color="#d7191c",label="ECE",capsize=3)
        ax.set_xticks(xs); ax.set_xticklabels(models,rotation=20,ha="right",fontsize=8)
        ax.set_ylabel("AUROC",color="#2c7bb6"); ax2.set_ylabel("ECE",color="#d7191c")
        ax.set_title(ds,fontweight="bold")
    fig.suptitle("Multi-seed robustness (mean$\\pm$std over 10 resplits)",fontweight="bold")
    fig.tight_layout()
    for e in ("pdf","png"): fig.savefig(F/f"fig_robustness.{e}",bbox_inches="tight",dpi=150)
    plt.close(); print("fig_robustness")

for fn in (t_extended,t_ranksafe,t_transfer,t_conformal,t_robustness):
    try: fn()
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"FAILED {fn.__name__}: {e}")
print("done")
