#!/usr/bin/env python3
"""
Split-conformal prediction as an uncertainty/coverage baseline for the ECG
foundation-model calibration paper.

ONE-VS-REST split-conformal for clinically important endpoints (MI, AF, NORM).

Methodology (binary score-based / APS-style split conformal):
  - Treat each (model,dataset,endpoint) as a binary one-vs-rest problem.
  - Nonconformity score for a candidate label is s = 1 - p_hat(label):
        s(positive) = 1 - p ,   s(negative) = 1 - (1-p) = p
  - On the CALIBRATION split, compute the score of the TRUE label for every
    calibration sample, then take the conformal quantile
        q = the ceil((n+1)(1-alpha)) / n  -th empirical quantile of cal scores
    (finite-sample-valid level; clipped to 1.0 when (n+1)(1-alpha) > n).
  - On TEST, form a prediction SET over {positive, negative}: include a label
    iff its candidate score <= q.
  - Conformal is built on RAW vs Platt- vs isotonic- vs temperature-recalibrated
    probabilities. Recalibrators are fit on (p_cal, y_cal) per endpoint.

Reported per (endpoint, model, dataset, alpha, method):
  empirical marginal coverage, average set size, singleton rate, empty-set
  (forced-abstention) rate, both-labels ("uncertain") rate, the conformal
  quantile, and class-conditional coverage (among true-positives and
  true-negatives separately).

Writes:
  results/conformal.json   (nested: endpoint/model/dataset/alpha/method)
  results/conformal.csv    (flat)
  writing/figures/fig_conformal.pdf (+ .png)

Uses ONLY the frozen probability caches via analysis_common.load_cell.
Does NOT modify probs caches or benchmark_v2_results.json or any .tex.
"""
import json, sys, csv
from pathlib import Path
import numpy as np

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
sys.path.insert(0, str(WS / "code"))
import analysis_common as ac  # noqa: E402

RESULTS = WS / "results"
FIGDIR = WS / "writing" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

MODELS = {"M1": "MERL", "M2": "ST-MEM", "M3": "HuBERT-ECG",
          "M4": "ECGFM-KED", "M5": "ECG-FM", "M6": "S4D"}
DATASETS = {"D1": "PTB-XL", "D3": "CODE-15%", "D4": "CPSC2018", "D5": "MIMIC-IV-ECG"}

# endpoint -> {dataset_id: class-name-in-that-dataset}
ENDPOINTS = {
    "MI":   {"D1": "MI",   "D5": "MI"},
    "AF":   {"D3": "AF",   "D4": "AF",     "D5": "AF"},
    "NORM": {"D1": "NORM", "D3": "NORM",   "D4": "Normal", "D5": "SR"},
}

ALPHAS = [0.1, 0.2]
METHODS = ["raw", "platt", "isotonic", "temperature"]
MIN_POS, MIN_NEG = 10, 10


def conformal_quantile(scores_cal, alpha):
    """Finite-sample split-conformal quantile of calibration nonconformity scores.

    q-level = ceil((n+1)(1-alpha)) / n empirical quantile (== the k-th smallest
    score with k = ceil((n+1)(1-alpha))). If k > n the quantile is +inf, i.e.
    all candidate labels are admitted (set always covers).
    """
    s = np.sort(np.asarray(scores_cal, float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")  # cannot guarantee at this n; admit everything
    if k < 1:
        k = 1
    return float(s[k - 1])


def eval_conformal(p_pos_cal, y_cal, p_pos_test, y_test, alpha):
    """One-vs-rest split conformal for a single binary endpoint at level alpha.

    p_pos_*: P(positive) for cal/test. y_*: binary {0,1} labels.
    Returns a dict of coverage/efficiency metrics.
    """
    p_pos_cal = np.asarray(p_pos_cal, float)
    p_pos_test = np.asarray(p_pos_test, float)
    y_cal = np.asarray(y_cal, int)
    y_test = np.asarray(y_test, int)

    # Calibration nonconformity score of the TRUE label.
    #   positive -> 1 - p_pos ; negative -> 1 - (1-p_pos) = p_pos
    s_cal = np.where(y_cal == 1, 1.0 - p_pos_cal, p_pos_cal)
    q = conformal_quantile(s_cal, alpha)

    # Test candidate scores
    s_pos = 1.0 - p_pos_test       # score for including label "positive"
    s_neg = p_pos_test             # score for including label "negative"
    inc_pos = s_pos <= q
    inc_neg = s_neg <= q

    set_size = inc_pos.astype(int) + inc_neg.astype(int)
    # covered = the TRUE label is in the prediction set
    covered = np.where(y_test == 1, inc_pos, inc_neg)

    n = len(y_test)
    empty = set_size == 0
    singleton = set_size == 1
    both = set_size == 2

    pos_mask = y_test == 1
    neg_mask = y_test == 0
    cov_pos = float(covered[pos_mask].mean()) if pos_mask.any() else float("nan")
    cov_neg = float(covered[neg_mask].mean()) if neg_mask.any() else float("nan")

    return {
        "alpha": float(alpha),
        "target_coverage": float(1.0 - alpha),
        "conformal_quantile": (None if np.isinf(q) else float(q)),
        "quantile_is_inf": bool(np.isinf(q)),
        "n_test": int(n),
        "n_cal": int(len(y_cal)),
        "n_test_pos": int(pos_mask.sum()),
        "n_test_neg": int(neg_mask.sum()),
        "coverage": float(covered.mean()),
        "coverage_tp": cov_pos,   # coverage among true-positives
        "coverage_tn": cov_neg,   # coverage among true-negatives
        "avg_set_size": float(set_size.mean()),
        "singleton_rate": float(singleton.mean()),
        "empty_rate": float(empty.mean()),         # forced abstention
        "both_rate": float(both.mean()),           # "uncertain" / both labels
    }


def main():
    recs = ac.get_recalibrators()  # {platt, isotonic, temperature}
    man = json.loads((RESULTS / "probs_manifest.json").read_text())
    cell_index = {(c["model_id"], c["dataset_id"]): c for c in man["cells"]}

    nested = {}   # endpoint -> model -> dataset -> alpha -> method -> metrics
    flat_rows = []
    skipped = []

    for endpoint, dmap in ENDPOINTS.items():
        nested.setdefault(endpoint, {})
        for mid, mname in MODELS.items():
            for did, cls in dmap.items():
                key = (mid, did)
                if key not in cell_index:
                    # cell does not exist (e.g. documented (M4,D3) exclusion or no feature cache)
                    skipped.append((endpoint, mid, did, "cell-absent"))
                    continue
                cell = cell_index[key]
                classes = list(cell["classes"])
                if cls not in classes:
                    skipped.append((endpoint, mid, did, f"class '{cls}' absent"))
                    continue
                ci = classes.index(cls)

                d = ac.load_cell(mid, did)
                y_cal = d["y_cal"][:, ci].astype(int)
                y_test = d["y_test"][:, ci].astype(int)
                p_cal = d["p_cal"][:, ci].astype(float)
                p_test = d["p_test"][:, ci].astype(float)

                npos = int(y_test.sum()); nneg = int(len(y_test) - npos)
                if npos < MIN_POS or nneg < MIN_NEG:
                    skipped.append((endpoint, mid, did,
                                    f"insufficient support test_pos={npos} test_neg={nneg}"))
                    continue
                # also need both classes present in calibration for recalibrators & scores
                if int(y_cal.sum()) < 1 or int(len(y_cal) - y_cal.sum()) < 1:
                    skipped.append((endpoint, mid, did, "cal single-class"))
                    continue

                nested[endpoint].setdefault(mname, {})
                nested[endpoint][mname].setdefault(DATASETS[did], {})

                # Pre-compute recalibrated probabilities (fit on cal, apply to cal & test)
                prob_variants = {"raw": (p_cal, p_test)}
                for rname, fn in recs.items():
                    p_cal_r = np.asarray(fn(p_cal, y_cal, p_cal), float)
                    p_test_r = np.asarray(fn(p_cal, y_cal, p_test), float)
                    prob_variants[rname] = (p_cal_r, p_test_r)

                for alpha in ALPHAS:
                    akey = f"alpha_{alpha:g}"
                    nested[endpoint][mname][DATASETS[did]].setdefault(akey, {})
                    for method in METHODS:
                        pc, pt = prob_variants[method]
                        m = eval_conformal(pc, y_cal, pt, y_test, alpha)
                        m["model"] = mname
                        m["model_id"] = mid
                        m["dataset"] = DATASETS[did]
                        m["dataset_id"] = did
                        m["endpoint"] = endpoint
                        m["endpoint_class"] = cls
                        m["method"] = method
                        nested[endpoint][mname][DATASETS[did]][akey][method] = m
                        flat_rows.append(m)

    # ---- write JSON (nested) ----
    out_json = {
        "description": "Split-conformal prediction (one-vs-rest) coverage/efficiency "
                       "baseline for ECG foundation-model calibration paper.",
        "method": "binary score-based split conformal; s=1-p_hat(label); "
                  "quantile level ceil((n+1)(1-alpha))/n on calibration true-label scores; "
                  "prediction set over {positive,negative}; label included iff score<=quantile.",
        "alphas": ALPHAS,
        "prob_methods": METHODS,
        "min_pos": MIN_POS, "min_neg": MIN_NEG,
        "skipped": [{"endpoint": e, "model_id": m, "dataset_id": dd, "reason": r}
                    for (e, m, dd, r) in skipped],
        "results": nested,
    }
    (RESULTS / "conformal.json").write_text(json.dumps(out_json, indent=2))

    # ---- write CSV (flat) ----
    cols = ["endpoint", "endpoint_class", "model_id", "model", "dataset_id", "dataset",
            "alpha", "target_coverage", "method",
            "n_test", "n_cal", "n_test_pos", "n_test_neg",
            "coverage", "coverage_tp", "coverage_tn",
            "avg_set_size", "singleton_rate", "empty_rate", "both_rate",
            "conformal_quantile", "quantile_is_inf"]
    with open(RESULTS / "conformal.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in flat_rows:
            w.writerow({k: r.get(k, "") for k in cols})

    print(f"Wrote results/conformal.json and results/conformal.csv "
          f"({len(flat_rows)} rows, {len(skipped)} skipped combos).")
    return out_json, flat_rows, skipped


if __name__ == "__main__":
    main()
