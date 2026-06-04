"""
H4 calibration-size ablation on cached pilot logits.

Reuses the same test-split prediction arrays from the recalibration-gradient
pilot. Varies the calibration set size and runs 5 repetitions per size to
assess whether method rankings change with calibration budget.

H4 claim: MLP underperforms at small n (< 2000) due to overfitting, and
outperforms at large n (> 10000). This pilot covers the small-n regime only
(max ~200 records), but directly tests the small-set MLP-vs-simpler-methods
prediction.
"""

import json
import datetime
import sys
import os
import csv
import traceback

import numpy as np
from scipy.special import softmax
from scipy.special import expit as sigmoid
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

WORKSPACE = "/Users/ameenk/AutoR/runs/20260523_210450/workspace"
RESULTS = os.path.join(WORKSPACE, "results")
NOTES = os.path.join(WORKSPACE, "notes")

# Calibration sizes to test
CAL_SIZES = [25, 50, 75, 100, 150, 200]
N_REPS = 5
RANDOM_SEED = 42


# ── Metric helpers ─────────────────────────────────────────────────────────────

def ece_equal_width(probs, labels, n_bins=15):
    """Equal-width ECE over per-class binary predictions."""
    eces = []
    n_classes = probs.shape[1]
    for c in range(n_classes):
        p = probs[:, c]
        y = labels[:, c]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(p, bins[1:-1])
        ece = 0.0
        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() == 0:
                continue
            acc = y[mask].mean()
            conf = p[mask].mean()
            ece += mask.sum() * abs(acc - conf)
        eces.append(ece / len(y))
    return float(np.mean(eces)) if eces else float("nan")


def brier_score(probs, labels):
    n_classes = probs.shape[1]
    scores = []
    for c in range(n_classes):
        if labels[:, c].sum() == 0 or labels[:, c].sum() == len(labels):
            continue
        scores.append(np.mean((probs[:, c] - labels[:, c]) ** 2))
    return float(np.mean(scores)) if scores else float("nan")


def macro_auroc(probs, labels):
    scores = []
    for c in range(labels.shape[1]):
        if labels[:, c].sum() == 0 or labels[:, c].sum() == len(labels):
            continue
        try:
            scores.append(roc_auc_score(labels[:, c], probs[:, c]))
        except Exception:
            pass
    return float(np.mean(scores)) if scores else float("nan")


def eval_metrics(probs, labels):
    return {
        "auroc": macro_auroc(probs, labels),
        "ece_ew": ece_equal_width(probs, labels),
        "brier": brier_score(probs, labels),
    }


# ── Recalibration methods ──────────────────────────────────────────────────────

def softmax_probs(logits):
    return softmax(logits, axis=1)


def monotone_platt_or_temperature(cal_logits, cal_labels, test_logits):
    """
    Per-class Platt scaling with monotonicity guard.
    Falls back to temperature scaling if any Platt slope is non-positive.
    """
    n_classes = cal_logits.shape[1]
    cal_probs_raw = softmax_probs(cal_logits)
    test_probs_raw = softmax_probs(test_logits)
    test_probs_out = test_probs_raw.copy()

    all_positive = True
    for c in range(n_classes):
        y = cal_labels[:, c].ravel()
        if y.sum() == 0 or y.sum() == len(y):
            continue
        X = cal_logits[:, c].reshape(-1, 1)
        try:
            lr = LogisticRegression(C=1e4, max_iter=1000, solver="lbfgs")
            lr.fit(X, y)
            if lr.coef_[0, 0] <= 0:
                all_positive = False
                break
        except Exception:
            all_positive = False
            break

    if all_positive:
        for c in range(n_classes):
            y = cal_labels[:, c].ravel()
            if y.sum() == 0 or y.sum() == len(y):
                continue
            X_c = cal_logits[:, c].reshape(-1, 1)
            X_t = test_logits[:, c].reshape(-1, 1)
            try:
                lr = LogisticRegression(C=1e4, max_iter=1000, solver="lbfgs")
                lr.fit(X_c, y)
                test_probs_out[:, c] = lr.predict_proba(X_t)[:, 1]
            except Exception:
                pass
        method = "platt"
    else:
        # Temperature scaling: minimize NLL over single scalar T
        from scipy.optimize import minimize_scalar
        def nll(log_T):
            T = np.exp(log_T)
            scaled = cal_logits / T
            p = softmax_probs(scaled)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.mean(cal_labels * np.log(p))

        res = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
        T_opt = float(np.exp(res.x))
        test_probs_out = softmax_probs(test_logits / T_opt)
        method = "temperature"

    return test_probs_out, method


def isotonic_regression_cal(cal_logits, cal_labels, test_logits):
    cal_probs = softmax_probs(cal_logits)
    test_probs = softmax_probs(test_logits).copy()
    n_classes = cal_logits.shape[1]
    for c in range(n_classes):
        y = cal_labels[:, c].ravel()
        if y.sum() == 0 or y.sum() == len(y):
            continue
        try:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(cal_probs[:, c], y)
            test_probs[:, c] = ir.predict(test_probs[:, c])
        except Exception:
            pass
    return test_probs


def learned_mlp_head(cal_logits, cal_labels, test_logits, rng):
    """Small MLP: Linear(C,32)->ReLU->Dropout(0.2)->Linear(32,C), per-class sigmoid outputs."""
    import torch
    import torch.nn as nn

    n_cal, n_classes = cal_logits.shape
    n_val = max(1, int(0.2 * n_cal))
    idx = rng.permutation(n_cal)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    if len(tr_idx) < 2:
        return softmax_probs(test_logits)

    X_tr = torch.tensor(cal_logits[tr_idx], dtype=torch.float32)
    y_tr = torch.tensor(cal_labels[tr_idx], dtype=torch.float32)
    X_val = torch.tensor(cal_logits[val_idx], dtype=torch.float32)
    y_val = torch.tensor(cal_labels[val_idx], dtype=torch.float32)
    X_te = torch.tensor(test_logits, dtype=torch.float32)

    hidden = min(32, max(8, n_cal // 4))
    model = nn.Sequential(
        nn.Linear(n_classes, hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, n_classes),
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None
    patience = 10
    no_improve = 0

    for epoch in range(100):
        model.train()
        opt.zero_grad()
        out = model(X_tr)
        loss = crit(out, y_tr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = crit(val_out, y_val).item()
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_te)).numpy()
    return probs


# ── Main ablation ──────────────────────────────────────────────────────────────

def run_ablation():
    rng = check_random_state(RANDOM_SEED)

    # Load cached arrays
    grad_npz = np.load(os.path.join(RESULTS, "recalibration_gradient_pilot_predictions.npz"), allow_pickle=True)
    s4d_npz = np.load(os.path.join(RESULTS, "s4d_real_pilot_predictions.npz"), allow_pickle=True)

    datasets = [
        {"id": "D1", "name": "PTB-XL"},
        {"id": "D3", "name": "CODE-15%"},
    ]
    models = [
        {"id": "M6_s4d_configured", "name": "Configured S4D"},
        {"id": "M2_stmem_probe", "name": "ST-MEM probe"},
    ]

    all_rows = []
    method_rank_rows = []

    for ds in datasets:
        did = ds["id"]
        cal_labels = s4d_npz[f"{did}_cal_labels"].astype(float)
        max_cal = len(cal_labels)

        for mdl in models:
            mid = mdl["id"]
            cal_logits_full = grad_npz[f"{did}_{mid}_cal_logits"].astype(float)
            test_logits = grad_npz[f"{did}_{mid}_test_logits"].astype(float)
            test_labels = grad_npz[f"{did}_{mid}_test_labels"].astype(float)

            for size in CAL_SIZES:
                if size > max_cal:
                    continue
                for rep in range(N_REPS):
                    rep_rng = check_random_state(RANDOM_SEED + rep * 1000 + size)
                    idx = rep_rng.choice(max_cal, size=size, replace=False)
                    c_log = cal_logits_full[idx]
                    c_lab = cal_labels[idx]

                    # Raw baseline
                    raw_probs = softmax_probs(test_logits)
                    raw_m = eval_metrics(raw_probs, test_labels)

                    # Monotone Platt/Temperature
                    try:
                        mono_probs, mono_type = monotone_platt_or_temperature(c_log, c_lab, test_logits)
                        mono_m = eval_metrics(mono_probs, test_labels)
                    except Exception as e:
                        mono_m = {"auroc": float("nan"), "ece_ew": float("nan"), "brier": float("nan")}
                        mono_type = "error"

                    # Isotonic
                    try:
                        iso_probs = isotonic_regression_cal(c_log, c_lab, test_logits)
                        iso_m = eval_metrics(iso_probs, test_labels)
                    except Exception as e:
                        iso_m = {"auroc": float("nan"), "ece_ew": float("nan"), "brier": float("nan")}

                    # MLP
                    try:
                        mlp_rng = check_random_state(RANDOM_SEED + rep * 1000 + size + 7)
                        mlp_probs = learned_mlp_head(c_log, c_lab, test_logits, mlp_rng)
                        mlp_m = eval_metrics(mlp_probs, test_labels)
                    except Exception as e:
                        mlp_m = {"auroc": float("nan"), "ece_ew": float("nan"), "brier": float("nan")}

                    for method_name, m in [
                        ("raw", raw_m),
                        (f"monotone_{mono_type}", mono_m),
                        ("isotonic", iso_m),
                        ("mlp_head", mlp_m),
                    ]:
                        all_rows.append({
                            "dataset_id": did,
                            "dataset_name": ds["name"],
                            "model_id": mid,
                            "model_name": mdl["name"],
                            "cal_size": size,
                            "rep": rep,
                            "method": method_name,
                            "auroc": round(m["auroc"], 6),
                            "ece_ew": round(m["ece_ew"], 6),
                            "brier": round(m["brier"], 6),
                        })

    # Aggregate across reps: mean and std per (dataset, model, cal_size, method)
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        key = (row["dataset_id"], row["model_id"], row["cal_size"], row["method"])
        agg[key]["auroc"].append(row["auroc"])
        agg[key]["ece_ew"].append(row["ece_ew"])
        agg[key]["brier"].append(row["brier"])

    agg_rows = []
    for (did, mid, size, method), vals in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        ds_name = next(d["name"] for d in datasets if d["id"] == did)
        mdl_name = next(m["name"] for m in models if m["id"] == mid)
        agg_rows.append({
            "dataset_id": did,
            "dataset_name": ds_name,
            "model_id": mid,
            "model_name": mdl_name,
            "cal_size": size,
            "method": method,
            "auroc_mean": round(np.nanmean(vals["auroc"]), 6),
            "auroc_std": round(np.nanstd(vals["auroc"]), 6),
            "ece_ew_mean": round(np.nanmean(vals["ece_ew"]), 6),
            "ece_ew_std": round(np.nanstd(vals["ece_ew"]), 6),
            "brier_mean": round(np.nanmean(vals["brier"]), 6),
            "brier_std": round(np.nanstd(vals["brier"]), 6),
        })

    # Method rankings per (dataset, model, cal_size)
    rank_keys = defaultdict(dict)
    for row in agg_rows:
        if row["method"] == "raw":
            continue
        k = (row["dataset_id"], row["model_id"], row["cal_size"])
        rank_keys[k][row["method"]] = {"ece": row["ece_ew_mean"], "brier": row["brier_mean"], "row": row}

    method_rank_rows = []
    for k, methods in sorted(rank_keys.items()):
        did, mid, size = k
        ds_name = next(d["name"] for d in datasets if d["id"] == did)
        mdl_name = next(m["name"] for m in models if m["id"] == mid)
        sorted_ece = sorted(methods.items(), key=lambda x: x[1]["ece"])
        sorted_brier = sorted(methods.items(), key=lambda x: x[1]["brier"])
        for rank_i, (method, info) in enumerate(sorted_ece):
            brier_rank = next(r for r, (m, _) in enumerate(sorted_brier) if m == method) + 1
            method_rank_rows.append({
                "dataset_id": did,
                "dataset_name": ds_name,
                "model_id": mid,
                "model_name": mdl_name,
                "cal_size": size,
                "method": method,
                "ece_rank": rank_i + 1,
                "brier_rank": brier_rank,
                "ece_ew_mean": round(info["ece"], 6),
                "brier_mean": round(info["brier"], 6),
                "best_by_ece": rank_i == 0,
                "best_by_brier": brier_rank == 1,
            })

    # Count wins per method across (dataset, model, cal_size) cells
    from collections import Counter
    ece_wins = Counter()
    brier_wins = Counter()
    for row in method_rank_rows:
        if row["best_by_ece"]:
            ece_wins[row["method"]] += 1
        if row["best_by_brier"]:
            brier_wins[row["method"]] += 1

    win_summary = {"ece_wins": dict(ece_wins), "brier_wins": dict(brier_wins)}

    # Save outputs
    out_prefix = os.path.join(RESULTS, "h4_calsize_ablation")
    # Raw per-rep table
    with open(out_prefix + "_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # Aggregated table
    with open(out_prefix + "_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)

    # Rankings table
    with open(out_prefix + "_method_ranking.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(method_rank_rows[0].keys()))
        w.writeheader()
        w.writerows(method_rank_rows)

    # JSON summary
    result = {
        "generated_at": datetime.datetime.now().isoformat(),
        "stage": "05_experimentation",
        "experiment": "h4_calsize_ablation_cached_logits",
        "scope": (
            "H4 calibration-size ablation using cached pilot logits. "
            "Tests whether MLP underperforms simpler methods at small calibration sets. "
            "Only covers small-n regime (n<=200 records). "
            "2 datasets x 2 models x 6 sizes x 5 reps x 4 methods."
        ),
        "publication_grade": False,
        "cal_sizes_tested": CAL_SIZES,
        "n_repetitions": N_REPS,
        "datasets": [d["id"] for d in datasets],
        "models": [m["id"] for m in models],
        "methods": ["raw", "monotone_platt_or_temperature", "isotonic", "mlp_head"],
        "win_summary": win_summary,
        "total_recal_cells": len(method_rank_rows),
        "h4_small_n_finding": {
            "description": (
                "MLP-head ECE wins vs total: {}/{} cells. "
                "Monotone Platt/temp ECE wins: {}/{}. "
                "Isotonic ECE wins: {}/{}."
            ).format(
                ece_wins.get("mlp_head", 0), len(method_rank_rows) // 3,
                ece_wins.get("monotone_platt", 0) + ece_wins.get("monotone_temperature", 0)
                    + sum(v for k, v in ece_wins.items() if "monotone" in k),
                len(method_rank_rows) // 3,
                ece_wins.get("isotonic", 0), len(method_rank_rows) // 3,
            )
        },
        "source_artifacts": [
            "results/recalibration_gradient_pilot_predictions.npz",
            "results/s4d_real_pilot_predictions.npz",
        ],
    }

    with open(out_prefix + "_results.json", "w") as f:
        json.dump(result, f, indent=2)

    return all_rows, agg_rows, method_rank_rows, win_summary


if __name__ == "__main__":
    log_path = os.path.join(NOTES, "h4_calsize_ablation.log")
    os.makedirs(NOTES, exist_ok=True)

    class Tee:
        def __init__(self, path):
            self.f = open(path, "w")
            self.orig = sys.stdout
        def write(self, s):
            self.orig.write(s)
            self.f.write(s)
        def flush(self):
            self.orig.flush()
            self.f.flush()

    sys.stdout = Tee(log_path)

    print(f"[{datetime.datetime.now().isoformat()}] Starting H4 calibration-size ablation")
    print(f"Cal sizes: {CAL_SIZES}, N_reps: {N_REPS}")

    try:
        all_rows, agg_rows, rank_rows, win_summary = run_ablation()
        print(f"\nCompleted. Rows: raw={len(all_rows)}, agg={len(agg_rows)}, rank={len(rank_rows)}")
        print(f"ECE wins: {win_summary['ece_wins']}")
        print(f"Brier wins: {win_summary['brier_wins']}")

        # Print per-size MLP ECE rank for quick review
        print("\n--- MLP ECE rank by cal_size (aggregated across datasets+models) ---")
        from collections import defaultdict
        mlp_ece_ranks = defaultdict(list)
        for row in rank_rows:
            if row["method"] == "mlp_head":
                mlp_ece_ranks[row["cal_size"]].append(row["ece_rank"])
        for size in sorted(mlp_ece_ranks):
            ranks = mlp_ece_ranks[size]
            print(f"  n={size:4d}: MLP ECE ranks = {ranks}, mean = {np.mean(ranks):.2f}")

    except Exception:
        traceback.print_exc()
        print("FATAL ERROR in ablation run")

    sys.stdout.f.close()
    sys.stdout = sys.stdout.orig
    print("Done. Log saved to:", log_path)
