#!/usr/bin/env python3
"""
Canonical analysis substrate for the calibration paper's extended analyses.

Precomputes fixed, deterministic probability caches for every evaluated
(model, dataset) cell from the *already-cached* frozen features, so that all
downstream analyses (transfer, conformal, RankSafe-Cal, robustness, DCA,
metrics, mechanistic, subgroup) operate on identical inputs and CANNOT diverge
or invent results.

Outputs (under results/probs/):
  {MID}_{DID}.npz  with arrays: y_cal, p_cal, y_test, p_test, test_pids,
                                cal_pids, train_pids; attrs via companion json
  ../probs_manifest.json  cell list, classes, prevalences, min-support classes

Also exposes metric helpers (NLL, calibration slope/intercept, adaptive/classwise
ECE, Brier) and the ecg_audit recalibrators for reuse by analysis scripts.
"""
import json, sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
RESULTS = WS / "results"
PROBS = RESULTS / "probs"
PROBS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WS / "code"))

SEED = 42
MIN_POS, MIN_NEG = 10, 10

# datasets: id -> (name, loader, feature-cache-suffix)
def _loaders():
    from run_full_benchmark import load_ptbxl, load_cpsc2018
    from mimic_labels_v2 import load_mimic_v2
    from code15_expanded import load_code15_expanded
    return {
        "D1": ("PTB-XL",       lambda: load_ptbxl(max_n=2000),     ""),
        "D3": ("CODE-15%",     lambda: load_code15_expanded(),      "exp"),
        "D4": ("CPSC2018",     lambda: load_cpsc2018(max_n=None),   ""),
        "D5": ("MIMIC-IV-ECG", lambda: load_mimic_v2(max_n=2000),   ""),
    }

MODELS = {"M1": "MERL", "M2": "ST-MEM", "M3": "HuBERT-ECG",
          "M4": "ECGFM-KED", "M5": "ECG-FM", "M6": "S4D"}
# cells to skip (documented exclusion)
SKIP = {("M4", "D3")}

# ── metrics ───────────────────────────────────────────────────────────────────

def ece_equal_width(y, p, n_bins=15):
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (p >= b[i]) & (p < b[i+1] if i < n_bins-1 else p <= b[i+1])
        if m.sum(): e += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(e / len(y))

def adaptive_ece(y, p, n_bins=15):
    y, p = np.asarray(y, float), np.asarray(p, float)
    n = len(y); o = np.argsort(p); yp, yt = p[o], y[o]
    edges = np.linspace(0, n, n_bins + 1).astype(int); e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        if hi > lo: e += (hi-lo) * abs(yt[lo:hi].mean() - yp[lo:hi].mean())
    return float(e / n)

def nll(y, p, eps=1e-7):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), eps, 1-eps)
    return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))

def brier(y, p):
    return float(np.mean((np.asarray(p, float)-np.asarray(y, float))**2))

def sce(y, p):
    return float(np.asarray(p, float).mean() - np.asarray(y, float).mean())

def auroc(y, p):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))

def calibration_slope_intercept(y, p, eps=1e-7):
    """Cox calibration: fit y ~ sigmoid(a*logit(p)+b). slope=a (1=ideal), intercept=b (0=ideal)."""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), eps, 1-eps)
    if len(np.unique(y)) < 2: return float("nan"), float("nan")
    logit = np.log(p/(1-p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit, y.astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])

def valid_classes(Y, min_pos=MIN_POS, min_neg=MIN_NEG):
    out = []
    for c in range(Y.shape[1]):
        npos = int(Y[:, c].sum()); nneg = int(len(Y) - npos)
        if npos >= min_pos and nneg >= min_neg: out.append(c)
    return out

def macro(fn, Y, P, cols):
    vals = []
    for c in cols:
        if len(np.unique(Y[:, c])) < 2: continue
        v = fn(Y[:, c], P[:, c])
        if not np.isnan(v): vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")

# ── recalibrators (reuse the ecg_audit package) ────────────────────────────────

def get_recalibrators():
    from ecg_audit import platt_scale, isotonic_calibrate, temperature_scale
    return {"platt": platt_scale, "isotonic": isotonic_calibrate,
            "temperature": temperature_scale}

# ── probe ───────────────────────────────────────────────────────────────────--

def _probe(Xtr, Ytr, Xev, seed=SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xev = np.nan_to_num(Xev, nan=0.0, posinf=0.0, neginf=0.0)
    sc = StandardScaler().fit(Xtr); Xtr, Xev = sc.transform(Xtr), sc.transform(Xev)
    out = []
    for c in range(Ytr.shape[1]):
        yv = Ytr[:, c]
        if len(np.unique(yv)) < 2:
            out.append(np.full(len(Xev), yv.mean())); continue
        out.append(LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
                   .fit(Xtr, yv).predict_proba(Xev)[:, 1])
    return np.stack(out, 1)

def _patient_split(pids, seed=SEED):
    from run_benchmark_v2 import patient_split
    return patient_split(pids, None, seed=seed) if False else _ps(pids, seed)

def _ps(pids, seed):
    up = np.unique(pids); rng = np.random.default_rng(seed); rng.shuffle(up)
    n = len(up); nt = max(30, int(0.15*n)); nc = max(30, int(0.15*n))
    test, cal = set(up[:nt].tolist()), set(up[nt:nt+nc].tolist())
    idx = {"train": [], "cal": [], "test": []}
    for i, p in enumerate(pids):
        idx["test" if p in test else "cal" if p in cal else "train"].append(i)
    return {k: np.array(v) for k, v in idx.items()}

# ── precompute ──────────────────────────────────────────────────────────────--

def feature_path(mid, did, suffix):
    name = f"features_{mid}_{did}{'exp' if suffix=='exp' else ''}.npy"
    return RESULTS / name

def precompute():
    loaders = _loaders()
    manifest = {"seed": SEED, "min_pos": MIN_POS, "min_neg": MIN_NEG, "cells": []}
    for did, (dname, loader, suf) in loaders.items():
        print(f"[{did}] loading {dname} ...", flush=True)
        sigs, labels, pids, classes = loader()
        split = _ps(pids, SEED)
        for mid, mname in MODELS.items():
            if (mid, did) in SKIP:
                continue
            fp = feature_path(mid, did, suf)
            if not fp.exists():
                print(f"   skip {mid}x{did}: no feature cache {fp.name}")
                continue
            feats = np.load(fp)
            if len(feats) != len(labels):
                print(f"   skip {mid}x{did}: rows {len(feats)} != {len(labels)}")
                continue
            p_test = _probe(feats[split["train"]], labels[split["train"]], feats[split["test"]])
            p_cal = _probe(feats[split["train"]], labels[split["train"]], feats[split["cal"]])
            out = PROBS / f"{mid}_{did}.npz"
            np.savez_compressed(
                out,
                y_cal=labels[split["cal"]], p_cal=p_cal,
                y_test=labels[split["test"]], p_test=p_test,
                test_pids=pids[split["test"]], cal_pids=pids[split["cal"]],
                train_pids=pids[split["train"]],
                classes=np.array(classes, dtype=object))
            vc = valid_classes(labels[split["test"]])
            manifest["cells"].append({
                "model_id": mid, "model": mname, "dataset_id": did, "dataset": dname,
                "classes": classes, "valid_class_idx": vc,
                "valid_classes": [classes[c] for c in vc],
                "n_test": int(len(split["test"])), "n_cal": int(len(split["cal"])),
                "prevalence_test": {classes[c]: float(labels[split["test"]][:, c].mean())
                                    for c in range(len(classes))},
                "probs_file": f"probs/{mid}_{did}.npz"})
            print(f"   {mid}x{did}: n_test={len(split['test'])} valid={len(vc)}")
    (RESULTS / "probs_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest['cells'])} probs caches + probs_manifest.json")
    return manifest

def load_cell(mid, did):
    """Convenience accessor used by analysis scripts."""
    d = np.load(PROBS / f"{mid}_{did}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}

if __name__ == "__main__":
    precompute()
