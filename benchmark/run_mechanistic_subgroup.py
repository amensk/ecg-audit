#!/usr/bin/env python3
"""
PART A: Mechanistic regression — why do some FMs rank well but calibrate poorly?
  Per (model,dataset,valid-class) row: target = per-class test ECE; predictors =
  per-class AUROC, prevalence, log10(n_pos), cell-level feature_spread (std of L2
  norms, label-free), pretraining-objective family, dataset, multilabel flag.
  OLS via numpy lstsq with standardized continuous predictors + bootstrap 95% CIs + R^2.

PART B: PTB-XL age/sex subgroup calibration across ALL 6 models (+ NLL, + isotonic).
  Age/sex re-derived in the exact load_ptbxl order and asserted aligned to features.

Real data only. Writes results/mechanistic_regression.{json,csv},
results/subgroup_expanded.{json,csv}, and figures fig_mechanistic / fig_subgroup_expanded.
"""
import json, sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
RESULTS = WS / "results"; FIGS = WS / "writing" / "figures"
sys.path.insert(0, str(WS / "code"))
import analysis_common as ac

OBJ = {"M1": "contrastive", "M2": "masked", "M3": "masked",
       "M4": "knowledge", "M5": "hybrid", "M6": "supervised"}
DSname = {"D1": "PTB-XL", "D3": "CODE-15%", "D4": "CPSC2018", "D5": "MIMIC-IV-ECG"}
MULTILABEL = {"D1": 1, "D3": 0, "D4": 0, "D5": 0}

def feat_file(mid, did):
    return RESULTS / (f"features_{mid}_D3exp.npy" if did == "D3" else f"features_{mid}_{did}.npy")

def feature_spread(mid, did):
    f = np.load(feat_file(mid, did))
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.std(np.linalg.norm(f, axis=1)))

# ── PART A ───────────────────────────────────────────────────────────────────
def part_a():
    man = json.loads((RESULTS / "probs_manifest.json").read_text())
    rows = []
    spread_cache = {}
    for cell in man["cells"]:
        mid, did = cell["model_id"], cell["dataset_id"]
        key = (mid, did)
        if key not in spread_cache:
            try: spread_cache[key] = feature_spread(mid, did)
            except Exception: spread_cache[key] = np.nan
        spread = spread_cache[key]
        c = ac.load_cell(mid, did); Y, P = c["y_test"], c["p_test"]
        classes = list(cell["classes"])
        for ci in cell["valid_class_idx"]:
            yt, yp = Y[:, ci], P[:, ci]
            npos = int(yt.sum())
            rows.append({
                "cell": f"{mid}_{did}", "model": ac.MODELS[mid], "dataset": DSname[did],
                "class": classes[ci],
                "ece": ac.ece_equal_width(yt, yp), "auroc": ac.auroc(yt, yp),
                "prevalence": float(yt.mean()), "log_npos": float(np.log10(max(npos, 1))),
                "feature_spread": spread, "objective": OBJ[mid],
                "dataset_cat": did, "multilabel": MULTILABEL[did],
            })
    import csv
    with open(RESULTS / "mechanistic_regression.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # design matrix
    cont = ["auroc", "prevalence", "log_npos", "feature_spread"]
    Xc = np.array([[r[k] for k in cont] for r in rows], float)
    # log-spread (spread spans orders of magnitude due to MERL=34.8)
    Xc[:, 3] = np.log10(Xc[:, 3] + 1e-6)
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
    Xz = (Xc - mu) / sd
    # categoricals (drop-first one-hot)
    objs = ["masked", "knowledge", "hybrid", "supervised"]  # ref=contrastive
    dsets = ["D3", "D4", "D5"]  # ref=D1 (multilabel collinear w/ D1 -> drop multilabel)
    Xcat = []
    names = ["AUROC", "prevalence", "log_npos", "log_feature_spread"]
    for o in objs: names.append(f"obj={o}")
    for d in dsets: names.append(f"dataset={DSname[d]}")
    for r in rows:
        row = [1.0 if r["objective"] == o else 0.0 for o in objs]
        row += [1.0 if r["dataset_cat"] == d else 0.0 for d in dsets]
        Xcat.append(row)
    Xcat = np.array(Xcat, float)
    X = np.hstack([np.ones((len(rows), 1)), Xz, Xcat])
    names = ["intercept"] + names
    y = np.array([r["ece"] for r in rows], float)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    # bootstrap CIs
    rng = np.random.default_rng(42); B = 2000; boot = []
    n = len(rows)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        b, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        boot.append(b)
    boot = np.array(boot)
    lo, hi = np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)
    coefs = {names[i]: {"coef": float(beta[i]), "ci": [float(lo[i]), float(hi[i])],
                        "significant": bool(lo[i] > 0 or hi[i] < 0)} for i in range(len(names))}
    out = {"n_rows": n, "r2": r2, "target": "per-class test ECE",
           "note": "continuous predictors standardized; feature_spread log10; "
                   "ref levels: objective=contrastive, dataset=PTB-XL. "
                   "multilabel collinear with PTB-XL so represented by dataset=PTB-XL ref.",
           "coefficients": coefs}
    (RESULTS / "mechanistic_regression.json").write_text(json.dumps(out, indent=2))

    # figure: coefficient plot (exclude intercept)
    keys = [k for k in names if k != "intercept"]
    vals = [coefs[k]["coef"] for k in keys]
    los = [coefs[k]["coef"] - coefs[k]["ci"][0] for k in keys]
    his = [coefs[k]["ci"][1] - coefs[k]["coef"] for k in keys]
    fig, axx = plt.subplots(figsize=(7.5, 4.5))
    yy = np.arange(len(keys))
    cols = ["#d7191c" if coefs[k]["significant"] else "#999999" for k in keys]
    axx.errorbar(vals, yy, xerr=[los, his], fmt="o", color="black", ecolor="gray", capsize=3)
    for i, k in enumerate(keys):
        axx.plot(vals[i], yy[i], "o", color=cols[i], ms=7)
    axx.axvline(0, color="k", ls="--", lw=0.8)
    axx.set_yticks(yy); axx.set_yticklabels(keys, fontsize=8)
    axx.set_xlabel("Effect on per-class ECE (OLS coef, standardized predictors)")
    axx.set_title(f"Mechanistic regression of ECE  (R$^2$={r2:.2f}, n={n})", fontweight="bold")
    fig.tight_layout()
    for e in ("pdf", "png"): fig.savefig(FIGS / f"fig_mechanistic.{e}", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"PART A done: n={n} R2={r2:.3f}")
    for k in keys:
        c = coefs[k]; star = "*" if c["significant"] else " "
        print(f"   {star} {k:22s} {c['coef']:+.4f}  CI[{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}]")
    return out

# ── PART B ───────────────────────────────────────────────────────────────────
def part_b():
    import pandas as pd, wfdb
    from scipy.signal import resample_poly
    from run_full_benchmark import load_ptbxl, SIGNAL_LEN, SEED
    # ground-truth order/labels/pids from the canonical loader
    sigs, labels, pids, classes = load_ptbxl(max_n=2000)
    n = len(labels)
    # re-derive age/sex in the SAME sampling+iteration order
    root = WS / "raw_data/ptb-xl/1.0.3"
    meta = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    meta.scp_codes = meta.scp_codes.apply(eval)
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0); scp = scp[scp.diagnostic == 1]
    sc_map = dict(zip(scp.index, scp.diagnostic_class)); CLS = ["NORM", "MI", "STTC", "CD", "HYP"]
    def scp2vec(codes):
        v = np.zeros(5, np.float32)
        for cc, conf in codes.items():
            if cc in sc_map and conf >= 50 and sc_map[cc] in CLS: v[CLS.index(sc_map[cc])] = 1
        return v
    rows = meta.sample(min(2000, len(meta)), random_state=SEED)
    age, sex, lab2 = [], [], []
    for idx, row in rows.iterrows():
        try:
            p = root / row.filename_lr
            sig, fields = wfdb.rdsamp(str(p))  # mirror loader's success condition
            age.append(float(row.age)); sex.append(int(row.sex)); lab2.append(scp2vec(row.scp_codes))
        except Exception:
            continue
    age = np.array(age); sex = np.array(sex); lab2 = np.stack(lab2)
    if len(age) != n or not np.array_equal(lab2, labels):
        print(f"PART B ABORT: alignment failed (age n={len(age)} vs {n}, labels_equal={np.array_equal(lab2,labels)})")
        return None
    split = ac._ps(pids, SEED); test = split["test"]; cal = split["cal"]; train = split["train"]
    age_t, sex_t = age[test], sex[test]
    groups = {"age<55": age_t < 55, "age55-70": (age_t >= 55) & (age_t < 70), "age>=70": age_t >= 70,
              "male": sex_t == 0, "female": sex_t == 1}
    recals = ac.get_recalibrators()
    MODELS = {"M1": "MERL", "M2": "ST-MEM", "M3": "HuBERT-ECG", "M4": "ECGFM-KED", "M5": "ECG-FM", "M6": "S4D"}
    out = {}
    import csv
    csv_rows = []
    for mid, mname in MODELS.items():
        feats = np.load(RESULTS / f"features_{mid}_D1.npy")
        if len(feats) != n:
            print(f"   skip {mname}: feature rows {len(feats)} != {n}"); continue
        p_test = ac._probe(feats[train], labels[train], feats[test], SEED)
        p_cal = ac._probe(feats[train], labels[train], feats[cal], SEED)
        # isotonic per class fit on cal
        p_iso = np.array(p_test)
        for ci in range(labels.shape[1]):
            if len(np.unique(labels[cal][:, ci])) > 1:
                p_iso[:, ci] = recals["isotonic"](p_cal[:, ci], labels[cal][:, ci], p_test[:, ci])
        Yte = labels[test]
        out[mname] = {}
        for gname, gmask in groups.items():
            if gmask.sum() < 20:
                continue
            Yg, Pg, Pgi = Yte[gmask], p_test[gmask], p_iso[gmask]
            vc = ac.valid_classes(Yg, min_pos=5, min_neg=5)
            rec = {"n": int(gmask.sum()),
                   "macro_auroc": ac.macro(ac.auroc, Yg, Pg, vc),
                   "macro_ece": ac.macro(ac.ece_equal_width, Yg, Pg, vc),
                   "macro_nll": ac.macro(ac.nll, Yg, Pg, vc),
                   "macro_ece_isotonic": ac.macro(ac.ece_equal_width, Yg, Pgi, vc)}
            out[mname][gname] = rec
            csv_rows.append({"model": mname, "subgroup": gname, **rec})
    (RESULTS / "subgroup_expanded.json").write_text(json.dumps(out, indent=2))
    with open(RESULTS / "subgroup_expanded.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys())); w.writeheader(); w.writerows(csv_rows)
    # figure: ECE by age group across models
    ages = ["age<55", "age55-70", "age>=70"]
    fig, axx = plt.subplots(figsize=(8.5, 4.3)); x = np.arange(len(MODELS)); w = 0.25
    for j, ag in enumerate(ages):
        vals = [out.get(m, {}).get(ag, {}).get("macro_ece", np.nan) for m in MODELS.values()]
        axx.bar(x + (j-1)*w, vals, w, label=ag)
    axx.set_xticks(x); axx.set_xticklabels(list(MODELS.values()), rotation=15, ha="right")
    axx.set_ylabel("Macro ECE (PTB-XL test)"); axx.legend(title="Age group")
    axx.set_title("Subgroup calibration by age across all six encoders", fontweight="bold")
    axx.grid(axis="y", alpha=0.3); fig.tight_layout()
    for e in ("pdf", "png"): fig.savefig(FIGS / f"fig_subgroup_expanded.{e}", bbox_inches="tight", dpi=150)
    plt.close()
    print("PART B done. Age-gradient ECE (model: <55 -> >=70):")
    for m in MODELS.values():
        if m in out and "age<55" in out[m] and "age>=70" in out[m]:
            print(f"   {m:11s} {out[m]['age<55']['macro_ece']:.3f} -> {out[m]['age>=70']['macro_ece']:.3f}")
    return out

if __name__ == "__main__":
    part_a()
    part_b()
    print("mechanistic + subgroup complete")
