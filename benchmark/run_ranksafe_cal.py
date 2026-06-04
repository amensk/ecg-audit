#!/usr/bin/env python3
"""
TASK 2 - RankSafe-Cal.

Per class, choose from candidate calibrators {platt, temperature, isotonic}
(all from ac.get_recalibrators()) the one that MINIMIZES ECE SUBJECT TO an
AUROC loss <= epsilon vs raw (AUROC_raw - AUROC_method <= epsilon). If none
satisfies, fall back to the rank-preserving method (platt / temperature) with
the smallest AUROC loss, preferring temperature, then platt.

SELECTION PROTOCOL (no test labels used for selection):
  Selection uses CROSS-FITTED (out-of-fold) estimates computed ENTIRELY on the
  calibration set, so test labels are never touched. For each class we run
  stratified K-fold over the calibration set: fit each candidate on the cal-
  train folds and score AUROC + equal-width ECE on the held-out cal fold, then
  pool the held-out predictions across folds to get one out-of-fold AUROC and
  ECE per candidate. We must cross-fit because a calibrator scored on the SAME
  data it was fit on is resubstitution-biased: isotonic in particular drives in-
  sample ECE to ~0 and can even RAISE in-sample AUROC, so an in-sample AUROC-loss
  constraint never binds. Out-of-fold scoring exposes the real (small) AUROC loss
  isotonic incurs from tie-merging, letting the epsilon constraint discriminate.
  The SELECTED calibrator is then RE-FIT on the full (p_cal, y_cal) and applied
  to p_test; ALL reported metrics are on TEST. For classes with too few positives
  to stratify K folds (min class count < N_SPLITS), we fall back to a single
  50/50 stratified cal split for the out-of-fold estimate; if even that is
  impossible the class defaults to the rank-preserving fallback.

Compares RankSafe-Cal vs: raw, unconstrained per-class isotonic, platt,
temperature. Reports macro ECE, NLL, Brier, macro AUROC and AUROC delta vs raw
per cell, for epsilon in {0.005, 0.01}.

Writes results/ranksafe_cal.json, results/ranksafe_cal.csv.
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
MANIFEST = json.loads((RESULTS / "probs_manifest.json").read_text())
RECALS = ac.get_recalibrators()           # platt, isotonic, temperature
EPSILONS = [0.005, 0.01]
# rank-preserving (strictly monotone) fallbacks, preference order
RANK_PRESERVING = ["temperature", "platt"]
N_SPLITS = 5
SEED = 42

PROOF_NOTE = (
    "A strictly monotone increasing transform g of the score s yields, for any "
    "threshold t, the same ordering of examples (s_i<s_j iff g(s_i)<g(s_j)); the "
    "ROC curve is invariant to such reparameterisation and AUROC (= P(score_pos > "
    "score_neg)) is preserved EXACTLY. Platt scaling is a logistic map "
    "sigma(a*p+b) which, with the fitted slope a>0 (positive on calibration "
    "probabilities), is strictly increasing; temperature scaling p->sigma(logit(p)/T) "
    "with T>0 is strictly increasing in p. Hence both preserve per-class AUROC "
    "exactly (up to float rounding). Isotonic regression is non-decreasing but only "
    "WEAKLY monotone: it can map distinct scores to an identical fitted value (flat "
    "pooled-adjacent-violator blocks), merging ties and thus can only reduce, never "
    "increase, AUROC. RankSafe-Cal exploits this by allowing isotonic only when its "
    "measured AUROC loss stays within epsilon, else falling back to a strictly "
    "monotone calibrator (temperature, then platt) that is AUROC-lossless by "
    "construction."
)


def cal_predict(name, p_cal_c, y_cal_c, p_apply_c):
    """Fit calibrator `name` on (p_cal_c,y_cal_c), apply to p_apply_c -> 1d probs."""
    return np.asarray(RECALS[name](p_cal_c, y_cal_c, p_apply_c), float)


def _oof_candidate_scores(p_cal_c, y_cal_c):
    """Cross-fitted (out-of-fold) AUROC/ECE per candidate, all on the cal set.

    Returns (cand_scores, oof_raw_auroc). cand_scores[name] has keys
    incal_auroc, incal_ece, incal_auroc_loss (oof_raw_auroc - oof candidate auroc).
    Test labels are never used.
    """
    from sklearn.model_selection import StratifiedKFold
    y = y_cal_c.astype(int)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    minc = min(n_pos, n_neg)
    if minc < 2:
        return None, float("nan")            # cannot hold out -> caller uses fallback
    k = min(N_SPLITS, minc)
    if k >= 2:
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
        folds = list(splitter.split(p_cal_c, y))
    else:
        folds = []
    if not folds:                            # degenerate: single 50/50 stratified split
        rng = np.random.default_rng(SEED)
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        rng.shuffle(pos); rng.shuffle(neg)
        te = np.concatenate([pos[:max(1, len(pos)//2)], neg[:max(1, len(neg)//2)]])
        tr = np.setdiff1d(np.arange(len(y)), te)
        folds = [(tr, te)]

    # collect pooled out-of-fold predictions per candidate (aligned to cal index)
    oof_pred = {name: np.full(len(y), np.nan) for name in RECALS}
    oof_raw = np.full(len(y), np.nan)
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        oof_raw[te] = p_cal_c[te]            # raw score is identity on held-out fold
        for name in RECALS:
            oof_pred[name][te] = cal_predict(name, p_cal_c[tr], y_cal_c[tr], p_cal_c[te])

    mask = ~np.isnan(oof_raw)
    if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
        return None, float("nan")
    oof_raw_auroc = ac.auroc(y[mask], oof_raw[mask])
    cand = {}
    for name in RECALS:
        pm = oof_pred[name]
        m = mask & ~np.isnan(pm)
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            cand[name] = {"incal_auroc": None, "incal_ece": np.inf,
                          "incal_auroc_loss": np.nan}
            continue
        a = ac.auroc(y[m], pm[m]); e = ac.ece_equal_width(y[m], pm[m])
        loss = (np.nan if np.isnan(oof_raw_auroc) or np.isnan(a) else oof_raw_auroc - a)
        cand[name] = {"incal_auroc": a, "incal_ece": e, "incal_auroc_loss": loss}
    return cand, oof_raw_auroc


def select_calibrator(p_cal_c, y_cal_c, epsilon):
    """Cross-fitted in-calibration selection for one class.

    Selection scores are out-of-fold estimates computed on the CALIBRATION set
    only. Test labels are never touched here. Returns (chosen, reason, cand, oof_raw_auroc).
    """
    cand, auroc_raw = _oof_candidate_scores(p_cal_c, y_cal_c)
    if cand is None:                          # could not cross-fit -> rank-preserving default
        return RANK_PRESERVING[0], "fallback_rank_preserving_no_oof", {}, auroc_raw

    # constraint: out-of-fold AUROC loss <= epsilon (NaN loss treated as not satisfying)
    feasible = {n: d for n, d in cand.items()
                if (d["incal_auroc_loss"] is not None
                    and not np.isnan(d["incal_auroc_loss"])
                    and d["incal_auroc_loss"] <= epsilon)}
    if feasible:
        chosen = min(feasible, key=lambda n: feasible[n]["incal_ece"])
        reason = "constrained_min_ece"
    else:
        # fallback: rank-preserving with smallest AUROC loss; prefer temperature then platt
        def fb_key(n):
            loss = cand[n]["incal_auroc_loss"]
            loss = np.inf if (loss is None or np.isnan(loss)) else loss
            return (loss, RANK_PRESERVING.index(n))
        chosen = min(RANK_PRESERVING, key=fb_key)
        reason = "fallback_rank_preserving"
    return chosen, reason, cand, auroc_raw


def macro_over(metric_fn, Y, P, cols):
    vals = []
    for ci in cols:
        if len(np.unique(Y[:, ci])) < 2:
            continue
        v = metric_fn(Y[:, ci], P[:, ci])
        if not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def build_method_matrix(name, p_cal, y_cal, p_test, cols):
    """Apply a single calibrator to every valid class -> calibrated test matrix."""
    P = np.array(p_test, float)
    for ci in cols:
        P[:, ci] = cal_predict(name, p_cal[:, ci], y_cal[:, ci], p_test[:, ci])
    return P


def metrics_block(Y, P, cols, auroc_raw_per_class):
    """macro ECE/NLL/Brier/AUROC + per-class AUROC delta vs raw."""
    blk = {
        "macro_ece": macro_over(ac.ece_equal_width, Y, P, cols),
        "macro_adaptive_ece": macro_over(ac.adaptive_ece, Y, P, cols),
        "macro_nll": macro_over(ac.nll, Y, P, cols),
        "macro_brier": macro_over(ac.brier, Y, P, cols),
        "macro_auroc": macro_over(ac.auroc, Y, P, cols),
    }
    deltas = []
    for ci in cols:
        a = ac.auroc(Y[:, ci], P[:, ci])
        ar = auroc_raw_per_class[ci]
        if not (np.isnan(a) or np.isnan(ar)):
            deltas.append(ar - a)            # positive = AUROC loss vs raw
    blk["macro_auroc_loss_vs_raw"] = float(np.mean(deltas)) if deltas else float("nan")
    blk["max_auroc_loss_vs_raw"] = float(np.max(deltas)) if deltas else float("nan")
    return blk


def main():
    out = {"protocol": "in_calibration_selection (selection uses calibration labels only; "
                       "metrics reported on test)",
           "epsilons": EPSILONS,
           "candidates": list(RECALS.keys()),
           "rank_preserving_fallback_order": RANK_PRESERVING,
           "proof_note": PROOF_NOTE,
           "cells": {}}
    csv_rows = []

    for cell in MANIFEST["cells"]:
        mid, did = cell["model_id"], cell["dataset_id"]
        key = f"{mid}_{did}"
        c = ac.load_cell(mid, did)
        Y_cal, P_cal = c["y_cal"], c["p_cal"]
        Y, P = c["y_test"], c["p_test"]
        classes = list(c["classes"])
        cols = ac.valid_classes(Y)

        # raw per-class test AUROC (reference for all deltas)
        auroc_raw_pc = {ci: ac.auroc(Y[:, ci], P[:, ci]) for ci in cols}

        cell_rec = {"model": cell["model"], "dataset": cell["dataset"],
                    "valid_classes": [classes[ci] for ci in cols],
                    "n_test": int(len(Y)),
                    "methods": {}, "ranksafe": {}}

        # ---- baseline methods (raw + single-calibrator-applied-to-all-classes)
        method_mats = {"raw": np.array(P, float)}
        for name in RECALS:
            method_mats[name] = build_method_matrix(name, P_cal, Y_cal, P, cols)
        for mname, M in method_mats.items():
            cell_rec["methods"][mname] = metrics_block(Y, M, cols, auroc_raw_pc)

        # ---- RankSafe-Cal for each epsilon
        for eps in EPSILONS:
            chosen_by_class = {}
            sel_debug = {}
            P_rs = np.array(P, float)
            for ci in cols:
                chosen, reason, cand, incal_raw = select_calibrator(
                    P_cal[:, ci], Y_cal[:, ci], eps)
                P_rs[:, ci] = cal_predict(chosen, P_cal[:, ci], Y_cal[:, ci], P[:, ci])
                chosen_by_class[classes[ci]] = chosen
                sel_debug[classes[ci]] = {
                    "chosen": chosen, "reason": reason,
                    "incal_auroc_raw": (None if np.isnan(incal_raw) else float(incal_raw)),
                    "candidates": {n: {k: (None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v))
                                       for k, v in d.items()}
                                   for n, d in cand.items()},
                }
            blk = metrics_block(Y, P_rs, cols, auroc_raw_pc)
            from collections import Counter
            blk["chosen_counts"] = dict(Counter(chosen_by_class.values()))
            blk["chosen_by_class"] = chosen_by_class
            blk["selection"] = sel_debug
            cell_rec["ranksafe"][f"eps_{eps}"] = blk

        out["cells"][key] = cell_rec

        # ---- CSV rows: one row per (cell, method) where method includes ranksafe@eps
        def add_row(method_label, blk):
            csv_rows.append({
                "cell": key, "model": cell["model"], "dataset_id": did,
                "dataset": cell["dataset"], "method": method_label,
                "macro_ece": blk["macro_ece"], "macro_nll": blk["macro_nll"],
                "macro_brier": blk["macro_brier"], "macro_auroc": blk["macro_auroc"],
                "macro_auroc_loss_vs_raw": blk["macro_auroc_loss_vs_raw"],
                "max_auroc_loss_vs_raw": blk["max_auroc_loss_vs_raw"],
            })
        for mname in ["raw", "platt", "temperature", "isotonic"]:
            add_row(mname, cell_rec["methods"][mname])
        for eps in EPSILONS:
            add_row(f"ranksafe_eps{eps}", cell_rec["ranksafe"][f"eps_{eps}"])

    (RESULTS / "ranksafe_cal.json").write_text(json.dumps(out, indent=2))
    fields = ["cell", "model", "dataset_id", "dataset", "method",
              "macro_ece", "macro_nll", "macro_brier", "macro_auroc",
              "macro_auroc_loss_vs_raw", "max_auroc_loss_vs_raw"]
    with open(RESULTS / "ranksafe_cal.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)

    # ---------------- summary print ----------------
    print_summary(out)
    print(f"\nWrote results/ranksafe_cal.json and results/ranksafe_cal.csv "
          f"({len(csv_rows)} rows)")
    return out


def print_summary(out):
    cells = out["cells"]
    print("\n=== TASK 2: RankSafe-Cal vs baselines (per cell, macro on TEST) ===")
    print(f"{'cell':9s} {'method':16s} {'ECE':>7s} {'NLL':>7s} {'Brier':>7s} "
          f"{'AUROC':>7s} {'dAUROC':>8s}")
    for k, v in cells.items():
        for m in ["raw", "isotonic", "platt", "temperature"]:
            b = v["methods"][m]
            print(f"{k:9s} {m:16s} {b['macro_ece']:7.4f} {b['macro_nll']:7.4f} "
                  f"{b['macro_brier']:7.4f} {b['macro_auroc']:7.4f} "
                  f"{b['macro_auroc_loss_vs_raw']:8.4f}")
        for eps in EPSILONS:
            b = v["ranksafe"][f"eps_{eps}"]
            print(f"{k:9s} {('ranksafe@'+str(eps)):16s} {b['macro_ece']:7.4f} "
                  f"{b['macro_nll']:7.4f} {b['macro_brier']:7.4f} "
                  f"{b['macro_auroc']:7.4f} {b['macro_auroc_loss_vs_raw']:8.4f}")
        print()


# dataset display-name -> id (for grouping in figure script too)
DID_OF = {"PTB-XL": "D1", "CODE-15%": "D3", "CPSC2018": "D4", "MIMIC-IV-ECG": "D5"}

if __name__ == "__main__":
    main()
