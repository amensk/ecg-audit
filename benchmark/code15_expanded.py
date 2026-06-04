#!/usr/bin/env python3
"""
Expanded CODE-15% loader: samples across multiple HDF5 parts so that rare
arrhythmia classes clear the min-support threshold (>=10 pos & >=10 neg in the
15% test split). At n=2000 (part0 only) only 1 class qualified; CODE-15 class
prevalences are 1.6-2.8%, requiring ~8000 records for all classes.

Adds a high-prevalence NORM reference class (normal_ecg flag), consistent with
PTB-XL NORM and MIMIC SR reference endpoints.
"""
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
CODE15 = WS / "raw_data" / "code15"
SEED = 42
SIGNAL_LEN = 5000
ORIG_HZ = 400
TARGET_HZ = 500
CODE15_CLASSES = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF", "NORM"]
PARTS = [0, 1, 2, 3, 4]          # which exams_part*.hdf5 to draw from
PER_PART = 1600                   # ~8000 total


def load_code15_expanded(parts=PARTS, per_part=PER_PART, seed=SEED):
    import h5py, pandas as pd
    from scipy.signal import resample_poly

    df = pd.read_csv(CODE15 / "exams.csv")
    df["NORM"] = df["normal_ecg"].astype(int)
    rng = np.random.default_rng(seed)

    sigs, labels, pids = [], [], []
    for p in parts:
        fname = f"exams_part{p}.hdf5"
        sub = df[df["trace_file"] == fname]
        if len(sub) == 0:
            logger.warning(f"  no rows for {fname}, skip")
            continue
        path = CODE15 / fname
        if not path.exists():
            logger.warning(f"  missing {path}, skip")
            continue
        with h5py.File(path, "r") as f:
            hdf5_ids = f["exam_id"][:]
            id2idx = {int(e): i for i, e in enumerate(hdf5_ids)}
            sub = sub[sub["exam_id"].isin(id2idx)]
            take = sub.sample(min(per_part, len(sub)), random_state=seed)
            # sorted row indices for h5py fancy indexing
            rows = sorted((id2idx[int(e)], int(e)) for e in take["exam_id"])
            idxs = [r[0] for r in rows]
            traces = f["tracings"][idxs]            # (k, 4096, 12)
        eid_order = [r[1] for r in rows]
        meta = take.set_index("exam_id")
        for sig, eid in zip(traces, eid_order):
            sig = sig.astype(np.float32)            # (4096, 12) @400Hz
            g = np.gcd(TARGET_HZ, ORIG_HZ)
            sig = resample_poly(sig, TARGET_HZ // g, ORIG_HZ // g, axis=0)
            if len(sig) >= SIGNAL_LEN:
                sig = sig[:SIGNAL_LEN]
            else:
                pad = np.zeros((SIGNAL_LEN, sig.shape[1]), dtype=sig.dtype)
                pad[:len(sig)] = sig; sig = pad
            row = meta.loc[eid]
            sigs.append(sig.T)
            labels.append(np.array([float(row[c]) for c in CODE15_CLASSES], dtype=np.float32))
            pids.append(int(row["patient_id"]))
        logger.info(f"  {fname}: +{len(eid_order)} (total {len(sigs)})")

    sigs = np.stack(sigs)
    labels = np.stack(labels)
    # impute any NaN signals
    if np.isnan(sigs).any():
        sigs = np.nan_to_num(sigs, nan=0.0)
    logger.info(f"CODE-15 expanded: {sigs.shape}, prevalence={labels.mean(0).round(4)}")
    return sigs, labels, np.array(pids), CODE15_CLASSES


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s, y, p, c = load_code15_expanded()
    print("shape:", s.shape, "patients:", len(set(p.tolist())))
    n_test_est = int(0.15 * len(set(p.tolist())))
    for i, cl in enumerate(c):
        print(f"  {cl}: {int(y[:,i].sum())} pos ({100*y[:,i].mean():.2f}%)")
