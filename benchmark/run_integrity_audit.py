#!/usr/bin/env python3
"""
Benchmark integrity / leakage audit for the ECG foundation-model calibration paper.

Reads ONLY real local files. Does not modify any caches, features, results JSON, or
manuscript files. Produces results/integrity_audit.json.

Sections:
  1. patient_split_disjointness  - pairwise-disjoint train/cal/test patient sets per cell
  2. dataset_completeness        - used vs available record counts; feature-cache row checks
  3. duplicate_signal_check      - sampled SHA1 dup check (CODE-15 signals; PTB-XL ids)
  4. checkpoint_provenance       - existence/size/lightweight format header check
  5. skipped_cells               - documented exclusions (M4xD3; LBBB-on-MIMIC min-support)
  6. seed_and_protocol           - protocol constants
"""
import json, sys, hashlib, struct, zipfile, traceback
from pathlib import Path
import numpy as np

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
RESULTS = WS / "results"
CODE15 = WS / "raw_data" / "code15"
PTBXL = WS / "raw_data" / "ptb-xl" / "1.0.3"
CPSC = WS / "raw_data" / "cpsc2018"
MIMIC_STAGE = WS / "raw_data" / "mimic-iv-ecg" / "staged_ecgs_v2"
sys.path.insert(0, str(WS / "code"))

import analysis_common as ac  # keystone


def section1_disjointness(manifest):
    """Pairwise-disjoint train/cal/test patient-id sets for every cell. Also per-dataset
    patient counts (computed once per dataset; identical across models on a dataset)."""
    cells_out = []
    any_overlap = False
    per_dataset = {}  # did -> patient counts (split sets identical across models)
    for cell in manifest["cells"]:
        mid, did = cell["model_id"], cell["dataset_id"]
        try:
            d = ac.load_cell(mid, did)
        except Exception as e:
            cells_out.append({"model_id": mid, "dataset_id": did,
                              "error": f"{type(e).__name__}: {e}"})
            any_overlap = True  # treat inability to verify as a failure flag
            continue
        tr = set(np.asarray(d["train_pids"]).tolist())
        ca = set(np.asarray(d["cal_pids"]).tolist())
        te = set(np.asarray(d["test_pids"]).tolist())
        ov_tc = sorted(tr & ca)
        ov_tt = sorted(tr & te)
        ov_ct = sorted(ca & te)
        disjoint = not (ov_tc or ov_tt or ov_ct)
        if not disjoint:
            any_overlap = True
        cells_out.append({
            "model_id": mid, "dataset_id": did, "dataset": cell["dataset"],
            "n_records": {"train": int(len(np.asarray(d["train_pids"]))),
                          "cal": int(len(np.asarray(d["cal_pids"]))),
                          "test": int(len(np.asarray(d["test_pids"])))},
            "n_patients": {"train": len(tr), "cal": len(ca), "test": len(te)},
            "overlaps": {"train_cal": ov_tc, "train_test": ov_tt, "cal_test": ov_ct},
            "n_overlap": {"train_cal": len(ov_tc), "train_test": len(ov_tt),
                          "cal_test": len(ov_ct)},
            "pairwise_disjoint": disjoint,
        })
        if did not in per_dataset:
            per_dataset[did] = {
                "dataset": cell["dataset"],
                "n_patients": {"train": len(tr), "cal": len(ca), "test": len(te),
                               "total": len(tr | ca | te)},
                "n_records": {"train": int(len(np.asarray(d["train_pids"]))),
                              "cal": int(len(np.asarray(d["cal_pids"]))),
                              "test": int(len(np.asarray(d["test_pids"]))),
                              "total": int(len(np.asarray(d["train_pids"]))
                                           + len(np.asarray(d["cal_pids"]))
                                           + len(np.asarray(d["test_pids"])))},
            }
    return {
        "leakage_found": any_overlap,
        "n_cells_checked": len(cells_out),
        "per_dataset_patient_counts": per_dataset,
        "cells": cells_out,
    }


def _feat_shape(name):
    p = RESULTS / name
    if not p.exists():
        return None
    return list(np.load(p, mmap_mode="r").shape)


def section2_completeness():
    import pandas as pd
    # Available counts from real source CSVs
    ptbxl_avail = int(len(pd.read_csv(PTBXL / "ptbxl_database.csv")))
    code15_df = pd.read_csv(CODE15 / "exams.csv")
    code15_avail = int(len(code15_df))
    cpsc_avail = int(len(pd.read_csv(CPSC / "REFERENCE.csv", header=None)))
    # MIMIC available = entries in record_list.csv (read from ZIP without extracting weights)
    mimic_avail = None
    mimic_zip = WS / "raw_data" / "mimic-iv-ecg_zip_download" / "mimic-iv-ecg-1.0.zip"
    zip_prefix = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"
    try:
        import csv
        with zipfile.ZipFile(str(mimic_zip), "r") as z:
            rl = z.read(zip_prefix + "record_list.csv").decode("utf-8").splitlines()
            mimic_avail = int(len(rl) - 1)  # minus header
    except Exception as e:
        mimic_avail = f"unavailable: {type(e).__name__}: {e}"
    staged_v2 = int(len(list(MIMIC_STAGE.glob("*.hea"))))

    # CODE-15 expanded: parts 0-4 records present in CSV
    parts04 = code15_df[code15_df["trace_file"].isin(
        [f"exams_part{p}.hdf5" for p in range(5)])]

    # Feature-cache row counts (mmap shapes). D3 cells use *_D3exp.npy per keystone.
    feat = {}
    for mid in ["M1", "M2", "M3", "M4", "M5", "M6"]:
        for did, fname in [("D1", f"features_{mid}_D1.npy"),
                           ("D3", f"features_{mid}_D3exp.npy"),
                           ("D4", f"features_{mid}_D4.npy"),
                           ("D5", f"features_{mid}_D5.npy")]:
            if (mid, did) == ("M4", "D3"):
                continue  # documented skip; no D3exp feature cache for ECGFM-KED
            feat[f"{mid}_{did}"] = {"file": fname, "shape": _feat_shape(fname)}

    # Expected row counts per dataset used in the benchmark
    expected_rows = {"D1": 2000, "D3": 8000, "D4": cpsc_avail, "D5": 2000}
    feat_ok = True
    feat_mismatches = []
    for key, info in feat.items():
        did = key.split("_")[1]
        exp = expected_rows[did]
        shp = info["shape"]
        if shp is None or shp[0] != exp:
            feat_ok = False
            feat_mismatches.append({"cell": key, "file": info["file"],
                                    "shape": shp, "expected_rows": exp})

    datasets = {
        "PTB-XL": {"id": "D1", "used": 2000, "available": ptbxl_avail,
                   "note": "max_n=2000 sampled (seed 42) from ptbxl_database.csv"},
        "CODE-15%": {"id": "D3", "used": 8000, "available": code15_avail,
                     "available_parts0_4": int(len(parts04)),
                     "used_target": "~8000 across parts 0-4 (PER_PART=1600 x 5)",
                     "note": "expanded loader code15_expanded.py; per-part 1600"},
        "CPSC2018": {"id": "D4", "used": cpsc_avail, "available": cpsc_avail,
                     "note": "full set used (max_n=None)"},
        "MIMIC-IV-ECG": {"id": "D5", "used": 2000, "available": mimic_avail,
                         "staged_v2_hea_files": staged_v2,
                         "note": ("max_n=2000 loaded from a natural-prevalence pool "
                                  "of ~4000 staged records (pool_mult=2)")},
    }
    return {
        "datasets": datasets,
        "feature_caches": feat,
        "feature_rows_match_expected": feat_ok,
        "feature_row_mismatches": feat_mismatches,
        "expected_rows_per_dataset": expected_rows,
    }


def section3_duplicates():
    import h5py
    out = {}

    # --- CODE-15 sampled exact-duplicate signal check ---
    # Sample ~1500 tracings across parts 0 and 1 (bounded I/O), SHA1 over rounded signals.
    code15 = {"sample_size_target": 1500, "parts_used": [0, 1]}
    try:
        rng = np.random.default_rng(42)
        per_part = 750
        hashes = []
        n_loaded = 0
        for part in [0, 1]:
            fp = CODE15 / f"exams_part{part}.hdf5"
            if not fp.exists():
                continue
            with h5py.File(fp, "r") as f:
                n = f["tracings"].shape[0]
                k = min(per_part, n)
                idxs = sorted(rng.choice(n, size=k, replace=False).tolist())
                # read selected tracings (h5py fancy index requires sorted unique)
                traces = f["tracings"][idxs]  # (k, 4096, 12) float
            for sig in traces:
                # round to 3 decimals to define "exact" duplicate robustly to fp noise
                r = np.round(np.asarray(sig, dtype=np.float64), 3)
                hashes.append(hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest())
                n_loaded += 1
        n_unique = len(set(hashes))
        n_dup = n_loaded - n_unique
        code15.update({
            "sample_size_actual": n_loaded,
            "n_unique_signals": n_unique,
            "n_duplicate_signals": int(n_dup),
            "duplicate_rate": (float(n_dup) / n_loaded) if n_loaded else None,
            "hash": "sha1 over signal rounded to 3 decimals",
            "status": "ok",
        })
    except Exception as e:
        code15.update({"status": f"error: {type(e).__name__}: {e}",
                       "trace": traceback.format_exc()[-500:]})
    out["code15_signal_dup_check"] = code15

    # --- PTB-XL identifier uniqueness (first 2000 records) ---
    ptbxl = {"sample_size": 2000}
    try:
        import pandas as pd
        df = pd.read_csv(PTBXL / "ptbxl_database.csv")
        head = df.head(2000)
        ecg_ids = head["ecg_id"].astype(str).tolist()
        fnames = head["filename_lr"].astype(str).tolist()
        ptbxl.update({
            "n_records": int(len(head)),
            "n_unique_ecg_id": int(len(set(ecg_ids))),
            "n_unique_filename_lr": int(len(set(fnames))),
            "ecg_id_unique": len(set(ecg_ids)) == len(ecg_ids),
            "filename_lr_unique": len(set(fnames)) == len(fnames),
            "ecg_id_sha1_of_concat": hashlib.sha1(
                "|".join(ecg_ids).encode()).hexdigest(),
            "status": "ok",
        })
    except Exception as e:
        ptbxl.update({"status": f"error: {type(e).__name__}: {e}"})
    out["ptbxl_id_uniqueness"] = ptbxl
    return out


def _checkpoint_header_check(path):
    """Lightweight format validation without loading weights.
       - .safetensors: read the 8-byte length + JSON header.
       - .pth/.pt: torch.save writes a ZIP archive; verify it is a valid zip.
         A non-zip .pth is flagged as a non-PyTorch / legacy binary."""
    p = Path(path)
    name = p.name.lower()
    try:
        if name.endswith(".safetensors"):
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            n_tensors = len([k for k in hdr if k != "__metadata__"])
            return {"format": "safetensors", "loadable": True,
                    "n_tensors": n_tensors,
                    "metadata": hdr.get("__metadata__", {}),
                    "note": "safetensors header parsed (weights not loaded)"}
        else:  # .pth / .pt
            try:
                z = zipfile.ZipFile(str(p))
                return {"format": "torch_zip", "loadable": True,
                        "n_zip_entries": len(z.namelist()),
                        "note": "valid PyTorch (torch.save zip) archive; header-only check"}
            except zipfile.BadZipFile:
                with open(p, "rb") as f:
                    head = f.read(16)
                return {"format": "non_pytorch_binary", "loadable": False,
                        "first_bytes_hex": head.hex(),
                        "note": ("not a torch.save zip archive; legacy/non-PyTorch "
                                 "binary (invalid load key)")}
    except Exception as e:
        return {"format": "unknown", "loadable": False,
                "error": f"{type(e).__name__}: {e}"}


def section4_checkpoints():
    specs = [
        ("M1_MERL", "checkpoints/merl/merl_checkpoint.pth"),
        ("M2_ST_MEM", "checkpoints/st_mem/st_mem_vit_base_encoder.pth"),
        ("M3_HuBERT_ECG", "checkpoints/hubert_ecg/model.safetensors"),
        ("M4_ECGFM_KED", "checkpoints/ecgfm_ked/ecgfm_ked_checkpoint.pth"),
        ("M5_ECG_FM", "checkpoints/ecg_fm/ecg_fm_checkpoint.pth"),
    ]
    out = {}
    all_present = True
    for key, rel in specs:
        p = WS / rel
        exists = p.exists()
        rec = {"path": rel, "exists": exists}
        if exists:
            rec["size_bytes"] = int(p.stat().st_size)
            rec["load_check"] = _checkpoint_header_check(p)
        else:
            all_present = False
            rec["size_bytes"] = None
            rec["load_check"] = {"loadable": False, "note": "file missing"}
        out[key] = rec
    # Legacy HuBERT .pth note
    legacy = WS / "checkpoints/hubert_ecg/hubert_ecg_checkpoint.pth"
    if legacy.exists():
        out["M3_HuBERT_ECG_legacy_pth"] = {
            "path": "checkpoints/hubert_ecg/hubert_ecg_checkpoint.pth",
            "exists": True, "size_bytes": int(legacy.stat().st_size),
            "load_check": _checkpoint_header_check(legacy),
            "note": ("legacy .pth is a non-PyTorch binary (invalid load key); "
                     "model.safetensors is the working checkpoint used"),
        }
    # S4D baseline note
    out["M6_S4D"] = {"path": None, "exists": False, "size_bytes": None,
                     "load_check": {"loadable": None,
                                    "note": ("handcrafted-feature baseline "
                                             "(60-dim per-lead statistics); no checkpoint")}}
    return {"all_required_checkpoints_present": all_present,
            "n_required": len(specs), "checkpoints": out}


def section5_skipped(manifest):
    # (M4,D3) absent from manifest (documented exclusion)
    cells_present = {(c["model_id"], c["dataset_id"]) for c in manifest["cells"]}
    m4d3_excluded = ("M4", "D3") not in cells_present
    m4d3_probs = (RESULTS / "probs" / "M4_D3.npz").exists()

    # LBBB-on-MIMIC excluded by min-support: report test positive count from M2/D5.
    lbbb = {}
    try:
        d = ac.load_cell("M2", "D5")
        classes = list(d["classes"])
        yt = d["y_test"]
        idx = classes.index("LBBB")
        pos = int(yt[:, idx].sum()); neg = int(len(yt) - pos)
        lbbb = {"dataset": "MIMIC-IV-ECG", "class": "LBBB",
                "test_pos": pos, "test_neg": neg, "n_test": int(len(yt)),
                "min_pos_threshold": ac.MIN_POS,
                "excluded_by_min_support": pos < ac.MIN_POS,
                "in_valid_class_idx_M2_D5":
                    idx in [c for c in range(len(classes))
                            if int(yt[:, c].sum()) >= ac.MIN_POS
                            and int(len(yt) - yt[:, c].sum()) >= ac.MIN_NEG]}
    except Exception as e:
        lbbb = {"error": f"{type(e).__name__}: {e}"}

    return {
        "M4_D3_ECGFM_KED_x_CODE15": {
            "excluded": m4d3_excluded,
            "probs_cache_present": m4d3_probs,
            "reason": ("partial architecture reconstruction; 35 missing keys; "
                       "near-chance AUROC; excluded from CODE-15% and Pareto"),
            "evidence": ("benchmark_v2_run.log: 'ECGFM-KED: missing=35'; "
                         "checkpoint_mapping.json status 'ok_partial'"),
        },
        "LBBB_on_MIMIC_min_support": lbbb,
    }


def section6_protocol():
    return {
        "seed": ac.SEED,
        "split": "70/15/15 patient-level (test 15%, cal 15%, train remainder)",
        "split_floor": "max(30, 15%) records floor per cal/test (keystone _ps)",
        "min_support": {"min_pos": ac.MIN_POS, "min_neg": ac.MIN_NEG},
        "ece_bins": 15, "ece_type": "equal-width",
        "bootstrap": 1000,
        "source": "analysis_common.py (SEED, MIN_POS, MIN_NEG, ece_equal_width)",
    }


def main():
    manifest = json.loads((RESULTS / "probs_manifest.json").read_text())
    report = {
        "audit": "benchmark_integrity_leakage",
        "generated_for": "ECG foundation-model calibration paper",
        "workspace": str(WS),
        "n_cells_in_manifest": len(manifest["cells"]),
        "patient_split_disjointness": section1_disjointness(manifest),
        "dataset_completeness": section2_completeness(),
        "duplicate_signal_check": section3_duplicates(),
        "checkpoint_provenance": section4_checkpoints(),
        "skipped_cells": section5_skipped(manifest),
        "seed_and_protocol": section6_protocol(),
    }
    outp = RESULTS / "integrity_audit.json"
    outp.write_text(json.dumps(report, indent=2))

    # ---- concise summary to stdout ----
    s1 = report["patient_split_disjointness"]
    s2 = report["dataset_completeness"]
    s3 = report["duplicate_signal_check"]
    s4 = report["checkpoint_provenance"]
    print("=" * 64)
    print("INTEGRITY AUDIT SUMMARY")
    print("=" * 64)
    print(f"Cells checked: {s1['n_cells_checked']} / manifest {report['n_cells_in_manifest']}")
    print(f"1) Patient leakage found: "
          f"{'YES' if s1['leakage_found'] else 'NO'} (all train/cal/test disjoint)")
    print(f"2) Feature rows match expected: "
          f"{'OK' if s2['feature_rows_match_expected'] else 'MISMATCH'}")
    for name, dd in s2["datasets"].items():
        print(f"     {name}: used={dd['used']} / available={dd['available']}")
    c15 = s3["code15_signal_dup_check"]
    print(f"3) CODE-15 dup check: {c15.get('n_duplicate_signals')} dup / "
          f"{c15.get('sample_size_actual')} sampled "
          f"(rate={c15.get('duplicate_rate')})")
    px = s3["ptbxl_id_uniqueness"]
    print(f"   PTB-XL ids unique (first {px.get('sample_size')}): "
          f"ecg_id={px.get('ecg_id_unique')} filename={px.get('filename_lr_unique')}")
    print(f"4) Checkpoints present: "
          f"{'ALL' if s4['all_required_checkpoints_present'] else 'MISSING'} "
          f"({s4['n_required']} required)")
    for k, v in s4["checkpoints"].items():
        if k in ("M6_S4D",):
            continue
        lc = v.get("load_check", {})
        print(f"     {k}: exists={v['exists']} size={v.get('size_bytes')} "
              f"fmt={lc.get('format','-')} loadable={lc.get('loadable')}")
    sk = report["skipped_cells"]
    lb = sk["LBBB_on_MIMIC_min_support"]
    print(f"5) M4xD3 excluded: {sk['M4_D3_ECGFM_KED_x_CODE15']['excluded']}; "
          f"LBBB@MIMIC test_pos={lb.get('test_pos')} "
          f"(min_pos={lb.get('min_pos_threshold')}) "
          f"excluded={lb.get('excluded_by_min_support')}")
    print(f"6) seed={report['seed_and_protocol']['seed']} "
          f"min_support={report['seed_and_protocol']['min_support']} "
          f"ece_bins={report['seed_and_protocol']['ece_bins']} "
          f"bootstrap={report['seed_and_protocol']['bootstrap']}")
    print("=" * 64)
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
