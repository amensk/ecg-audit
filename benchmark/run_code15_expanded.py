#!/usr/bin/env python3
"""
Re-run the CODE-15% (D3) benchmark cells on an expanded multi-part sample
(~8000 records) so rare classes clear min-support. Skips ECGFM-KED (M4), which
is a documented exclusion on CODE-15 (partial architecture reconstruction).

Merges the new D3 cells into results/benchmark_v2_results.json (replacing the
under-powered n=2000 D3 cells) and rewrites benchmark_v2_summary.csv.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from run_benchmark_v2 import (
    normalize_signals, patient_split, linear_probe,
    compute_metrics_v2, recalibrate, bootstrap_ece_reduction,
    extract_merl, extract_stmem, extract_ecgfm, extract_s4d_features,
)
from hubert_extractor import extract_hubert
from code15_expanded import load_code15_expanded

MODELS = {
    "M1": ("MERL",       extract_merl,  WS/"checkpoints/merl/merl_checkpoint.pth"),
    "M2": ("ST-MEM",     extract_stmem, WS/"checkpoints/st_mem/st_mem_vit_base_encoder.pth"),
    "M3": ("HuBERT-ECG", None,          None),
    "M5": ("ECG-FM",     extract_ecgfm, WS/"checkpoints/ecg_fm/ecg_fm_checkpoint.pth"),
    "M6": ("S4D",        None,          None),
}


def get_features(mid, sigs_raw, sigs_norm, extractor, ckpt):
    cache = RESULTS / f"features_{mid}_D3exp.npy"
    if cache.exists():
        f = np.load(cache)
        if len(f) == len(sigs_raw):
            logger.info(f"  [{mid}] reuse {f.shape}"); return f
    t0 = time.time()
    if mid == "M6":
        f = extract_s4d_features(sigs_norm)
    elif mid == "M3":
        f = extract_hubert(sigs_raw)
    else:
        f = extractor(sigs_norm, ckpt)
    np.save(cache, f)
    logger.info(f"  [{mid}] computed {f.shape} in {time.time()-t0:.0f}s")
    return f


def run():
    t0 = time.time()
    logger.info("=== CODE-15 expanded re-run ===")
    sigs_raw, labels, pids, classes = load_code15_expanded()
    sigs_norm = normalize_signals(sigs_raw)
    split = patient_split(pids, labels)
    logger.info(f"split train={len(split['train'])} cal={len(split['cal'])} test={len(split['test'])}")

    new_cells = []
    for mid, (name, extractor, ckpt) in MODELS.items():
        logger.info(f"--- {name} x CODE-15% (expanded) ---")
        feats = get_features(mid, sigs_raw, sigs_norm, extractor, ckpt)
        Xtr, Ytr = feats[split["train"]], labels[split["train"]]
        Xcal, Ycal = feats[split["cal"]], labels[split["cal"]]
        Xte, Yte = feats[split["test"]], labels[split["test"]]
        p_test = linear_probe(Xtr, Ytr, Xte)
        p_cal = linear_probe(Xtr, Ytr, Xcal)
        pa = compute_metrics_v2(Yte, p_test, classes)
        pb, preds = recalibrate(Ycal, p_cal, Yte, p_test, classes)
        ci = bootstrap_ece_reduction(Yte, p_test, preds["isotonic"], classes)
        logger.info(f"  AUROC={pa['macro_auroc']:.3f} CI={pa.get('macro_auroc_ci')} "
                    f"ECE={pa['macro_ece_ew']:.3f} ({pa['n_classes_evaluated']} classes)")
        new_cells.append({
            "model_id": mid, "model_name": name,
            "dataset_id": "D3", "dataset_name": "CODE-15%",
            "n_train": len(Xtr), "n_cal": len(Xcal), "n_test": len(Xte),
            "classes": classes, "phase_a": pa, "phase_b": pb,
            "isotonic_ece_reduction_ci": ci, "status": "ok",
            "note": "expanded multi-part sample (~8000 records, parts 0-4)",
        })

    # merge into benchmark_v2_results.json
    res_path = RESULTS / "benchmark_v2_results.json"
    data = json.loads(res_path.read_text())
    kept = [r for r in data["results"] if r.get("dataset_id") != "D3"]
    # preserve a record that ECGFM-KED x CODE-15 remains excluded
    kept.append({
        "model_id": "M4", "model_name": "ECGFM-KED",
        "dataset_id": "D3", "dataset_name": "CODE-15%",
        "status": "excluded",
        "error": "Partial architecture reconstruction (35 missing keys); "
                 "near-chance AUROC. Excluded from CODE-15 conclusions and Pareto.",
    })
    data["results"] = kept + new_cells
    data["code15_expanded"] = {
        "applied_at": datetime.now().isoformat(),
        "n_records": int(len(sigs_raw)),
        "parts_used": [0, 1, 2, 3, 4],
        "classes": classes,
        "prevalence": {c: float(labels[:, i].mean()) for i, c in enumerate(classes)},
        "rationale": "n=2000 (part0) left only 1 class above min-support; "
                     "expanded to ~8000 across parts so all classes qualify.",
    }
    data["n_successful"] = sum(1 for r in data["results"] if r.get("status") == "ok")
    data["n_total"] = len(data["results"])
    res_path.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"Merged {len(new_cells)} D3 cells into {res_path.name} "
                f"in {time.time()-t0:.0f}s")
    return data


if __name__ == "__main__":
    run()
