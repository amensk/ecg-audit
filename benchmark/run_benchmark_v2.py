#!/usr/bin/env python3
"""
Consolidated benchmark v2: 6 models x 4 datasets, with
  - HuBERT-ECG added via validated local safetensors path
  - improved MIMIC-IV-ECG labels/sampling (natural prevalence, MI added)
  - min-support per-class filtering (min_pos AND min_neg) before macro-averaging
  - bootstrap 95% CIs for macro AUROC, macro ECE, and isotonic ECE reduction

Reuses existing feature caches for M1/M2/M4/M5 on D1/D3/D4 (deterministic
loaders, seed 42); computes HuBERT (M3) everywhere; recomputes all models on
the resampled MIMIC v2 set (D5).

Output: results/benchmark_v2_results.json, results/benchmark_v2_summary.csv
"""
import json, logging, sys, time, csv
from pathlib import Path
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
RESULTS = WS / "results"
sys.path.insert(0, str(WS / "code"))
STMEM_SRC = WS / "checkpoints" / "_sources" / "ST-MEM"
sys.path.insert(0, str(STMEM_SRC))
sys.path.insert(0, str(STMEM_SRC / "models"))
sys.path.insert(0, str(STMEM_SRC / "models" / "encoder"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
MIN_POS = 10
MIN_NEG = 10
N_BOOT = 1000
N_BINS = 15

from run_full_benchmark import (
    load_ptbxl, load_code15, load_cpsc2018, normalize_signals,
    extract_merl, extract_stmem, extract_ecgfmked, extract_ecgfm,
    extract_s4d_features, patient_split,
)
from mimic_labels_v2 import load_mimic_v2
from hubert_extractor import extract_hubert

# ─── feature cache ────────────────────────────────────────────────────────────

def get_features(model_id, ds_id, sigs_raw, sigs_norm, ckpt_path, extractor):
    cache = RESULTS / f"features_{model_id}_{ds_id}.npy"
    if cache.exists():
        f = np.load(cache)
        if len(f) == len(sigs_raw):
            logger.info(f"  [{model_id}_{ds_id}] reuse cache {f.shape}")
            return f
        logger.warning(f"  [{model_id}_{ds_id}] cache rows {len(f)} != {len(sigs_raw)}, recompute")
    t0 = time.time()
    if model_id == "M6":
        f = extract_s4d_features(sigs_norm)
    elif model_id == "M3":
        f = extract_hubert(sigs_raw)            # HuBERT uses RAW signals
    else:
        f = extractor(sigs_norm, ckpt_path)
    logger.info(f"  [{model_id}_{ds_id}] computed {f.shape} in {time.time()-t0:.0f}s")
    np.save(cache, f)
    return f

# ─── probe + metrics ──────────────────────────────────────────────────────────

def linear_probe(X_train, Y_train, X_eval):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_eval = np.nan_to_num(X_eval, nan=0.0, posinf=0.0, neginf=0.0)
    sc = StandardScaler().fit(X_train)
    Xtr, Xev = sc.transform(X_train), sc.transform(X_eval)
    probs = []
    for c in range(Y_train.shape[1]):
        y = Y_train[:, c]
        if len(np.unique(y)) < 2:
            probs.append(np.full(len(Xev), y.mean())); continue
        clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
        clf.fit(Xtr, y)
        probs.append(clf.predict_proba(Xev)[:, 1])
    return np.stack(probs, axis=1)


def _ece(yt, yp, n_bins=N_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (yp >= bins[i]) & (yp < bins[i + 1])
        if m.sum() == 0:
            continue
        e += m.sum() * abs(yt[m].mean() - yp[m].mean())
    return e / len(yt)


def _valid_classes(y_true):
    out = []
    for c in range(y_true.shape[1]):
        npos = int(y_true[:, c].sum()); nneg = int(len(y_true) - npos)
        if npos >= MIN_POS and nneg >= MIN_NEG:
            out.append(c)
    return out


def compute_metrics_v2(y_true, y_prob, class_names, bootstrap=N_BOOT, seed=SEED):
    from sklearn.metrics import roc_auc_score, brier_score_loss
    valid = _valid_classes(y_true)
    per_class = {}
    for c in range(y_true.shape[1]):
        yt, yp = y_true[:, c], y_prob[:, c]
        npos = int(yt.sum()); nneg = int(len(yt) - npos)
        rec = {"n_pos": npos, "n_neg": nneg, "included": c in valid}
        if c in valid:
            rec.update({
                "auroc": float(roc_auc_score(yt, yp)),
                "brier": float(brier_score_loss(yt, yp)),
                "ece_ew": float(_ece(yt, yp)),
                "sce": float(yp.mean() - yt.mean()),
            })
        per_class[class_names[c]] = rec

    def macro(idx_rows, cols):
        aus, ecs, brs, scs = [], [], [], []
        for c in cols:
            yt, yp = y_true[idx_rows, c], y_prob[idx_rows, c]
            if len(np.unique(yt)) < 2:
                continue
            aus.append(roc_auc_score(yt, yp)); ecs.append(_ece(yt, yp))
            brs.append(brier_score_loss(yt, yp)); scs.append(yp.mean() - yt.mean())
        return (np.mean(aus) if aus else np.nan, np.mean(ecs) if ecs else np.nan,
                np.mean(brs) if brs else np.nan, np.mean(scs) if scs else np.nan)

    all_idx = np.arange(len(y_true))
    ma, me, mb, ms = macro(all_idx, valid)
    res = {
        "macro_auroc": float(ma), "macro_ece_ew": float(me),
        "macro_brier": float(mb), "macro_sce": float(ms),
        "n_classes_evaluated": len(valid),
        "included_classes": [class_names[c] for c in valid],
        "per_class": per_class,
    }
    # bootstrap CIs over test rows
    if bootstrap and valid:
        rng = np.random.default_rng(seed)
        ba, be = [], []
        n = len(y_true)
        for _ in range(bootstrap):
            bi = rng.integers(0, n, n)
            a, e, _, _ = macro(bi, valid)
            if not np.isnan(a):
                ba.append(a)
            if not np.isnan(e):
                be.append(e)
        if ba:
            res["macro_auroc_ci"] = [float(np.percentile(ba, 2.5)),
                                     float(np.percentile(ba, 97.5))]
        if be:
            res["macro_ece_ci"] = [float(np.percentile(be, 2.5)),
                                   float(np.percentile(be, 97.5))]
    return res


def recalibrate(y_cal, p_cal, y_test, p_test, class_names):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    out = {}
    recal_preds = {}
    for method in ["platt", "isotonic", "temperature"]:
        pr = np.array(p_test, dtype=float)
        for c in range(p_test.shape[1]):
            ytc, ypc, ypt = y_cal[:, c], p_cal[:, c], p_test[:, c]
            if len(np.unique(ytc)) < 2:
                continue
            try:
                if method == "isotonic":
                    iso = IsotonicRegression(out_of_bounds="clip").fit(ypc, ytc)
                    pr[:, c] = iso.predict(ypt)
                elif method == "platt":
                    lr = LogisticRegression(C=1e6, max_iter=500).fit(ypc.reshape(-1, 1), ytc.astype(int))
                    pr[:, c] = lr.predict_proba(ypt.reshape(-1, 1))[:, 1]
                elif method == "temperature":
                    import torch
                    lg = torch.FloatTensor(np.log(np.clip(ypc, 1e-7, 1 - 1e-7) / np.clip(1 - ypc, 1e-7, 1)))
                    yt = torch.FloatTensor(ytc)
                    T = torch.tensor(1.0, requires_grad=True)
                    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
                    def closure():
                        opt.zero_grad()
                        ps = torch.sigmoid(lg / T)
                        loss = -torch.mean(yt * torch.log(ps + 1e-8) + (1 - yt) * torch.log(1 - ps + 1e-8))
                        loss.backward(); return loss
                    opt.step(closure)
                    Tv = float(T.item())
                    lgt = np.log(np.clip(ypt, 1e-7, 1 - 1e-7) / np.clip(1 - ypt, 1e-7, 1))
                    pr[:, c] = 1 / (1 + np.exp(-lgt / Tv))
            except Exception:
                pass
        out[method] = compute_metrics_v2(y_test, pr, class_names, bootstrap=0)
        out[method]["method"] = method
        recal_preds[method] = pr
    return out, recal_preds


def bootstrap_ece_reduction(y_test, p_raw, p_iso, class_names, B=N_BOOT, seed=SEED):
    """Bootstrap 95% CI for relative macro-ECE reduction (raw->isotonic)."""
    rng = np.random.default_rng(seed)
    valid = _valid_classes(y_test)
    n = len(y_test)
    def macro_ece(rows, P):
        es = []
        for c in valid:
            yt = y_test[rows, c]
            if len(np.unique(yt)) < 2:
                continue
            es.append(_ece(yt, P[rows, c]))
        return np.mean(es) if es else np.nan
    reds = []
    for _ in range(B):
        bi = rng.integers(0, n, n)
        er, ei = macro_ece(bi, p_raw), macro_ece(bi, p_iso)
        if not np.isnan(er) and er > 0 and not np.isnan(ei):
            reds.append(100 * (er - ei) / er)
    if not reds:
        return None
    return [float(np.percentile(reds, 2.5)), float(np.percentile(reds, 97.5))]

# ─── main ─────────────────────────────────────────────────────────────────────

def run():
    t0 = time.time()
    logger.info("=== Benchmark v2 starting ===")

    datasets = {
        "D1": ("PTB-XL",       lambda: load_ptbxl(max_n=2000)),
        "D3": ("CODE-15%",     lambda: load_code15(max_n=2000)),
        "D4": ("CPSC2018",     lambda: load_cpsc2018(max_n=None)),
        "D5": ("MIMIC-IV-ECG", lambda: load_mimic_v2(max_n=2000)),
    }
    models = {
        "M1": ("MERL",       extract_merl,     WS/"checkpoints/merl/merl_checkpoint.pth"),
        "M2": ("ST-MEM",     extract_stmem,    WS/"checkpoints/st_mem/st_mem_vit_base_encoder.pth"),
        "M3": ("HuBERT-ECG", None,             None),
        "M4": ("ECGFM-KED",  extract_ecgfmked, WS/"checkpoints/ecgfm_ked/ecgfm_ked_checkpoint.pth"),
        "M5": ("ECG-FM",     extract_ecgfm,    WS/"checkpoints/ecg_fm/ecg_fm_checkpoint.pth"),
        "M6": ("S4D",        None,             None),
    }

    all_results, skip_log = [], {}

    for ds_id, (ds_name, loader) in datasets.items():
        logger.info(f"\n##### Dataset {ds_id} {ds_name} #####")
        try:
            sigs_raw, labels, pids, classes = loader()
        except Exception as e:
            logger.error(f"{ds_name} load failed: {e}", exc_info=True)
            skip_log[ds_id] = f"load error: {e}"
            continue
        sigs_norm = normalize_signals(sigs_raw)
        split = patient_split(pids, labels)
        logger.info(f"  split train={len(split['train'])} cal={len(split['cal'])} test={len(split['test'])}")

        for model_id, (model_name, extractor, ckpt) in models.items():
            logger.info(f"--- {model_name} x {ds_name} ---")
            try:
                feats = get_features(model_id, ds_id, sigs_raw, sigs_norm, ckpt, extractor)
                Xtr, Ytr = feats[split["train"]], labels[split["train"]]
                Xcal, Ycal = feats[split["cal"]], labels[split["cal"]]
                Xte, Yte = feats[split["test"]], labels[split["test"]]
                p_test = linear_probe(Xtr, Ytr, Xte)
                p_cal = linear_probe(Xtr, Ytr, Xcal)
                phase_a = compute_metrics_v2(Yte, p_test, classes)
                phase_b, recal_preds = recalibrate(Ycal, p_cal, Yte, p_test, classes)
                ece_red_ci = bootstrap_ece_reduction(Yte, p_test, recal_preds["isotonic"], classes)
                logger.info(f"  AUROC={phase_a['macro_auroc']:.3f} "
                            f"CI={phase_a.get('macro_auroc_ci')} "
                            f"ECE={phase_a['macro_ece_ew']:.3f} "
                            f"({phase_a['n_classes_evaluated']} classes)")
                all_results.append({
                    "model_id": model_id, "model_name": model_name,
                    "dataset_id": ds_id, "dataset_name": ds_name,
                    "n_train": len(Xtr), "n_cal": len(Xcal), "n_test": len(Xte),
                    "classes": classes,
                    "phase_a": phase_a, "phase_b": phase_b,
                    "isotonic_ece_reduction_ci": ece_red_ci,
                    "status": "ok",
                    "note": ("ECG-FM pretrained on MIMIC-IV-ECG (in-domain)"
                             if (model_id == "M5" and ds_id == "D5") else ""),
                })
            except Exception as e:
                logger.error(f"  FAILED: {e}", exc_info=True)
                skip_log[f"{model_id}_{ds_id}"] = str(e)[:300]
                all_results.append({
                    "model_id": model_id, "model_name": model_name,
                    "dataset_id": ds_id, "dataset_name": ds_name,
                    "status": "ERROR", "error": str(e)[:300],
                })

    out = {
        "generated_at": datetime.now().isoformat(),
        "execution_time_s": round(time.time() - t0, 1),
        "config": {"seed": SEED, "min_pos": MIN_POS, "min_neg": MIN_NEG,
                   "n_bootstrap": N_BOOT, "n_bins_ece": N_BINS},
        "skip_log": skip_log,
        "results": all_results,
        "n_successful": sum(1 for r in all_results if r.get("status") == "ok"),
        "n_total": len(all_results),
    }
    (RESULTS / "benchmark_v2_results.json").write_text(json.dumps(out, indent=2, default=str))
    logger.info(f"\n=== Done in {out['execution_time_s']}s, "
                f"{out['n_successful']}/{out['n_total']} cells ok ===")

    # summary CSV
    rows = []
    for r in all_results:
        if r.get("status") != "ok":
            continue
        pa = r["phase_a"]
        row = {
            "model": r["model_name"], "dataset": r["dataset_name"],
            "n_test": r["n_test"], "n_classes": pa["n_classes_evaluated"],
            "auroc": round(pa["macro_auroc"], 4),
            "auroc_lo": round(pa.get("macro_auroc_ci", [float('nan'), float('nan')])[0], 4),
            "auroc_hi": round(pa.get("macro_auroc_ci", [float('nan'), float('nan')])[1], 4),
            "ece": round(pa["macro_ece_ew"], 4),
            "ece_lo": round(pa.get("macro_ece_ci", [float('nan'), float('nan')])[0], 4),
            "ece_hi": round(pa.get("macro_ece_ci", [float('nan'), float('nan')])[1], 4),
            "brier": round(pa["macro_brier"], 4),
            "sce": round(pa["macro_sce"], 4),
        }
        for m in ["platt", "isotonic", "temperature"]:
            row[f"{m}_auroc"] = round(r["phase_b"][m]["macro_auroc"], 4)
            row[f"{m}_ece"] = round(r["phase_b"][m]["macro_ece_ew"], 4)
        rows.append(row)
    if rows:
        with open(RESULTS / "benchmark_v2_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        logger.info(f"Summary CSV written ({len(rows)} rows)")
    return out


if __name__ == "__main__":
    run()
