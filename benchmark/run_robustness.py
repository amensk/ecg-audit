#!/usr/bin/env python3
"""
Multi-seed split-robustness of the headline conclusions.

For each headline dataset (D1 PTB-XL, D5 MIMIC-IV-ECG) and each headline model
(M1 MERL, M2 ST-MEM, M3 HuBERT-ECG, M4 ECGFM-KED, M5 ECG-FM, M6 S4D):
  - Load labels/pids ONCE per dataset (deterministic loaders) and the model's
    FROZEN feature cache (results/features_{MID}_{DID}.npy). No feature recompute.
  - For seeds 0..9: split = ac._ps(pids, seed); fit linear probe on train, predict
    on test and cal (ac._probe). Compute macro (over ac.valid_classes(y_test)):
      AUROC, ECE (equal-width), adaptive ECE, Brier, NLL on test;
      plus isotonic-recalibrated macro ECE (isotonic fit on cal, applied to test)
      and the ECE reduction % = 100*(ece_raw - ece_iso)/ece_raw.
  - Report mean +/- std across seeds, plus min/max.

Conclusion-stability tests across the 10 seeds:
  - PTB-XL: is S4D (M6) macro ECE < EVERY foundation model's macro ECE?
    Report fraction of seeds where it holds + mean ECE gap
    (mean over foundation models of (fm_ece - s4d_ece), averaged over seeds).
  - MIMIC: is ECG-FM (M5) macro AUROC below MERL (M1) AND below S4D (M6)?
    Report fraction of seeds.

Outputs:
  results/robustness.json  (per dataset/model: metric mean/std/min/max over seeds;
                            plus the two stability fractions and details)
  results/robustness.csv   (dataset,model,metric,mean,std,min,max)

Uses ONLY real cached features. No fabrication. Does not touch manuscript files,
probs caches, or benchmark_v2_results.json.
"""
import json, sys, csv
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
sys.path.insert(0, str(WS / "code"))
import analysis_common as ac

RESULTS = WS / "results"

SEEDS = list(range(10))  # 0..9

# Headline cells
DATASETS = {"D1": "PTB-XL", "D5": "MIMIC-IV-ECG"}
MODELS = {"M1": "MERL", "M2": "ST-MEM", "M3": "HuBERT-ECG",
          "M4": "ECGFM-KED", "M5": "ECG-FM", "M6": "S4D"}
FOUNDATION = ["M1", "M2", "M3", "M4", "M5"]   # foundation models (S4D=M6 is the conv. baseline)

# Metrics computed on TEST per seed (macro over valid classes of y_test)
METRIC_FNS = {
    "auroc": ac.auroc,
    "ece": ac.ece_equal_width,
    "adaptive_ece": ac.adaptive_ece,
    "brier": ac.brier,
    "nll": ac.nll,
}
# All metrics tracked (raw test metrics + isotonic-recalibrated ECE + reduction %)
METRIC_NAMES = list(METRIC_FNS.keys()) + ["ece_isotonic", "ece_reduction_pct"]


def load_dataset(did):
    """Load labels/pids ONCE for a dataset id (sigs ignored)."""
    if did == "D1":
        from run_full_benchmark import load_ptbxl
        _, labels, pids, classes = load_ptbxl(max_n=2000)
    elif did == "D5":
        from mimic_labels_v2 import load_mimic_v2
        _, labels, pids, classes = load_mimic_v2(max_n=2000)
    else:
        raise ValueError(f"unsupported dataset {did}")
    return np.asarray(labels), np.asarray(pids), list(classes)


def macro_metrics_for_seed(feats, labels, pids, seed):
    """Fit probe + isotonic on a single seed's split; return dict of macro metrics on test."""
    split = ac._ps(pids, seed)
    tr, ca, te = split["train"], split["cal"], split["test"]

    Ytr, Yca, Yte = labels[tr], labels[ca], labels[te]
    # per-class probabilities on test and cal from a probe trained on train
    p_test = ac._probe(feats[tr], Ytr, feats[te], seed)
    p_cal = ac._probe(feats[tr], Ytr, feats[ca], seed)

    cols = ac.valid_classes(Yte)  # valid classes determined on TEST labels

    res = {}
    # raw test macro metrics
    for name, fn in METRIC_FNS.items():
        res[name] = ac.macro(fn, Yte, p_test, cols)

    # isotonic-recalibrated macro ECE: fit isotonic per-class on cal, apply to test
    iso = ac.get_recalibrators()["isotonic"]
    ece_iso_vals, ece_raw_vals = [], []
    for c in cols:
        if len(np.unique(Yte[:, c])) < 2:
            continue
        # isotonic must be fit where cal has both classes; else fall back to raw probs
        if len(np.unique(Yca[:, c])) < 2:
            p_test_c = p_test[:, c]
        else:
            p_test_c = iso(p_cal[:, c], Yca[:, c], p_test[:, c])
        ece_iso_vals.append(ac.ece_equal_width(Yte[:, c], p_test_c))
        ece_raw_vals.append(ac.ece_equal_width(Yte[:, c], p_test[:, c]))
    res["ece_isotonic"] = float(np.mean(ece_iso_vals)) if ece_iso_vals else float("nan")

    # ECE reduction % computed on the SAME class set used for isotonic (matched comparison)
    macro_raw = float(np.mean(ece_raw_vals)) if ece_raw_vals else float("nan")
    macro_iso = res["ece_isotonic"]
    if np.isnan(macro_raw) or macro_raw == 0 or np.isnan(macro_iso):
        res["ece_reduction_pct"] = float("nan")
    else:
        res["ece_reduction_pct"] = float(100.0 * (macro_raw - macro_iso) / macro_raw)

    return res


def summarize(values):
    """mean/std/min/max over a list of (possibly-nan) floats."""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], float)
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)),
            "min": float(arr.min()), "max": float(arr.max()), "n": int(arr.size)}


def main():
    out = {"seeds": SEEDS, "datasets": {}, "stability": {}}
    csv_rows = []
    # per-dataset cache of per-seed ECE/AUROC for the stability tests
    seed_ece = {did: {mid: [None] * len(SEEDS) for mid in MODELS} for did in DATASETS}
    seed_auroc = {did: {mid: [None] * len(SEEDS) for mid in MODELS} for did in DATASETS}

    for did, dname in DATASETS.items():
        print(f"\n[{did}] loading {dname} ...", flush=True)
        labels, pids, classes = load_dataset(did)
        out["datasets"][did] = {"dataset": dname, "classes": classes, "models": {}}

        for mid, mname in MODELS.items():
            fp = RESULTS / f"features_{mid}_{did}.npy"
            feats = np.load(fp)
            assert len(feats) == len(labels), \
                f"{mid}x{did}: feats {len(feats)} != labels {len(labels)}"

            # collect per-seed metric values
            per_seed = {name: [] for name in METRIC_NAMES}
            for si, seed in enumerate(SEEDS):
                m = macro_metrics_for_seed(feats, labels, pids, seed)
                for name in METRIC_NAMES:
                    per_seed[name].append(m[name])
                seed_ece[did][mid][si] = m["ece"]
                seed_auroc[did][mid][si] = m["auroc"]

            stats = {name: summarize(per_seed[name]) for name in METRIC_NAMES}
            out["datasets"][did]["models"][mid] = {
                "model": mname,
                "per_seed": {name: per_seed[name] for name in METRIC_NAMES},
                "stats": stats,
            }
            for name in METRIC_NAMES:
                s = stats[name]
                csv_rows.append({
                    "dataset": dname, "dataset_id": did,
                    "model": mname, "model_id": mid, "metric": name,
                    "mean": s["mean"], "std": s["std"],
                    "min": s["min"], "max": s["max"],
                })
            a, e = stats["auroc"], stats["ece"]
            print(f"   {mid} {mname:11s}: AUROC {a['mean']:.4f}+/-{a['std']:.4f} "
                  f"ECE {e['mean']:.4f}+/-{e['std']:.4f} "
                  f"ECE_iso {stats['ece_isotonic']['mean']:.4f} "
                  f"red% {stats['ece_reduction_pct']['mean']:.1f}")

    # ── Stability test 1: PTB-XL S4D macro ECE < EVERY foundation model's ECE ──
    s4d_ece = seed_ece["D1"]["M6"]
    holds = []          # per-seed boolean
    gaps_per_seed = []  # per-seed mean over FM of (fm_ece - s4d_ece)
    for si in range(len(SEEDS)):
        s4 = s4d_ece[si]
        fm_eces = [seed_ece["D1"][m][si] for m in FOUNDATION]
        cond = all((s4 is not None) and (f is not None) and (s4 < f) for f in fm_eces)
        holds.append(bool(cond))
        gaps_per_seed.append(float(np.mean([f - s4 for f in fm_eces])))
    frac_d1 = float(np.mean(holds))
    out["stability"]["ptbxl_s4d_best_ece"] = {
        "description": "PTB-XL: S4D (M6) macro ECE < every foundation model's macro ECE",
        "foundation_models": [MODELS[m] for m in FOUNDATION],
        "per_seed_holds": holds,
        "fraction_seeds": frac_d1,
        "n_seeds": len(SEEDS),
        "mean_ece_gap": float(np.mean(gaps_per_seed)),   # mean over seeds of (mean_FM_ece - s4d_ece)
        "per_seed_mean_gap": gaps_per_seed,
        "s4d_ece_per_seed": s4d_ece,
    }

    # ── Stability test 2: MIMIC ECG-FM AUROC below MERL AND below S4D ──
    ecgfm_au = seed_auroc["D5"]["M5"]
    merl_au = seed_auroc["D5"]["M1"]
    s4d_au = seed_auroc["D5"]["M6"]
    holds2 = []
    for si in range(len(SEEDS)):
        e_, m_, s_ = ecgfm_au[si], merl_au[si], s4d_au[si]
        cond = (e_ is not None and m_ is not None and s_ is not None
                and e_ < m_ and e_ < s_)
        holds2.append(bool(cond))
    frac_d5 = float(np.mean(holds2))
    out["stability"]["mimic_ecgfm_below_merl_and_s4d"] = {
        "description": "MIMIC: ECG-FM (M5) macro AUROC below MERL (M1) AND below S4D (M6)",
        "per_seed_holds": holds2,
        "fraction_seeds": frac_d5,
        "n_seeds": len(SEEDS),
        "ecgfm_auroc_per_seed": ecgfm_au,
        "merl_auroc_per_seed": merl_au,
        "s4d_auroc_per_seed": s4d_au,
    }

    # write JSON + CSV
    (RESULTS / "robustness.json").write_text(json.dumps(out, indent=2))
    fields = ["dataset", "dataset_id", "model", "model_id", "metric",
              "mean", "std", "min", "max"]
    with open(RESULTS / "robustness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)

    return out


if __name__ == "__main__":
    main()
