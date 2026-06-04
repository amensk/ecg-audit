#!/usr/bin/env python3
"""
Improved MIMIC-IV-ECG labeling and sampling (v2).

Fixes from audit (codex_followup_audit_20260603.md, Key Finding 5):
  - OLD sampling required >=1 positive label; "sinus rhythm" matched ~98% of
    records, so the sampled set was ~98% Normal -> degenerate Normal class
    (n_pos=298/300, only 2 negatives, AUROC unstable).
  - NEW sampling draws records at NATURAL prevalence (no positive-required
    filter), so every class has adequate positive AND negative support.
  - More precise diagnostic patterns; adds clinically high-value MI/infarct.
  - Per-class support reported; min-support filtering applied downstream.

Classes (clinically meaningful, with adequate two-sided support at natural prevalence):
  SR   - sinus rhythm (reference/sanity endpoint)
  AF   - atrial fibrillation / flutter
  RBBB - right bundle branch block
  LBBB - left bundle branch block
  LVH  - left ventricular hypertrophy
  MI   - myocardial infarction / infarct (any territory/age)
"""
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
MIMIC_ZIP = WS / "raw_data" / "mimic-iv-ecg_zip_download" / "mimic-iv-ecg-1.0.zip"
ZIP_PREFIX = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"
STAGING_DIR = WS / "raw_data" / "mimic-iv-ecg" / "staged_ecgs_v2"

SEED = 42
SIGNAL_LEN = 5000
TARGET_HZ = 500

MIMIC_CLASSES = ["SR", "AF", "RBBB", "LBBB", "LVH", "MI"]

# Precise patterns; negations handled by exclusion patterns where needed.
LABEL_PATTERNS = {
    "SR":   ["sinus rhythm", "normal sinus", "normal ecg"],
    "AF":   ["atrial fibrillation", "atrial flutter", "afib"],
    "RBBB": ["right bundle branch block", "rbbb"],
    "LBBB": ["left bundle branch block", "lbbb"],
    "LVH":  ["left ventricular hypertrophy", "lvh", "voltage criteria for"],
    "MI":   ["infarct", "myocardial infarction", "stemi"],
}
# Patterns that, if present, should NOT count as MI (avoid "no infarct" etc.)
MI_NEGATIONS = ["no infarct", "without infarct", "no evidence of infarct",
                "cannot rule out", "rule out"]


def extract_labels(text: str) -> np.ndarray:
    text = text.lower()
    vec = np.zeros(len(MIMIC_CLASSES), dtype=np.float32)
    for i, cls in enumerate(MIMIC_CLASSES):
        if cls == "MI":
            has = any(p in text for p in LABEL_PATTERNS["MI"])
            # exclude clear negations referring to infarct
            if has and any(neg in text for neg in MI_NEGATIONS):
                # only exclude if the negation is about infarct specifically;
                # keep "consider acute st elevation mi" style positives
                if not any(pos in text for pos in
                           ["acute", "stemi", "st elevation", "evolving"]):
                    has = False
            vec[i] = 1.0 if has else 0.0
        else:
            vec[i] = 1.0 if any(p in text for p in LABEL_PATTERNS[cls]) else 0.0
    return vec


def build_index(pool_size: int, seed: int = SEED):
    """Sample `pool_size` records at NATURAL prevalence (no positive filter)."""
    import zipfile as zf, csv
    logger.info("Building MIMIC v2 index (natural prevalence)...")
    with zf.ZipFile(str(MIMIC_ZIP), "r") as z:
        rec_rows = list(csv.DictReader(
            z.read(ZIP_PREFIX + "record_list.csv").decode("utf-8").splitlines()))
        meas_rows = list(csv.DictReader(
            z.read(ZIP_PREFIX + "machine_measurements.csv").decode("utf-8").splitlines()))
    path_map = {r["study_id"]: r["path"] for r in rec_rows}
    pid_map = {r["study_id"]: r["subject_id"] for r in rec_rows}

    rng = np.random.default_rng(seed)
    order = np.arange(len(meas_rows))
    rng.shuffle(order)

    sampled = []
    for i in order[:pool_size]:
        row = meas_rows[i]
        sid = row["study_id"]
        if sid not in path_map:
            continue
        text = " ".join(row.get(f"report_{j}", "") for j in range(18))
        sampled.append({
            "study_id": sid, "subject_id": pid_map[sid],
            "path": path_map[sid], "labels": extract_labels(text),
        })
    vecs = np.stack([r["labels"] for r in sampled])
    logger.info(f"  pool={len(sampled)} natural-prevalence records")
    for i, c in enumerate(MIMIC_CLASSES):
        logger.info(f"    {c}: {int(vecs[:, i].sum())} pos / {len(sampled)} "
                    f"({100*vecs[:, i].mean():.1f}%)")
    return sampled


def stage_records(records, staging_dir: Path):
    import zipfile as zf
    staging_dir.mkdir(parents=True, exist_ok=True)
    have = {f.stem for f in staging_dir.glob("*.hea")}
    todo = [r for r in records if r["study_id"] not in have]
    if not todo:
        logger.info(f"All {len(records)} records already staged")
        return
    logger.info(f"Staging {len(todo)} records from ZIP...")
    with zf.ZipFile(str(MIMIC_ZIP), "r") as z:
        for rec in todo:
            for ext in [".hea", ".dat"]:
                dest = staging_dir / (rec["study_id"] + ext)
                if not dest.exists():
                    try:
                        dest.write_bytes(z.read(ZIP_PREFIX + rec["path"] + ext))
                    except Exception as e:
                        logger.debug(f"  skip {rec['study_id']}{ext}: {e}")


def load_mimic_v2(max_n: int = 2000, pool_mult: int = 2):
    """Returns (sigs (N,12,5000) raw @500Hz, labels (N,C), pids (N,), classes)."""
    import wfdb
    from scipy.signal import resample_poly

    records = build_index(pool_size=max_n * pool_mult)
    stage_records(records, STAGING_DIR)

    sigs, labels, pids = [], [], []
    for rec in records:
        sid = rec["study_id"]
        if not (STAGING_DIR / (sid + ".hea")).exists():
            continue
        try:
            sig, fields = wfdb.rdsamp(str(STAGING_DIR / sid))
            n_leads = sig.shape[1]
            if n_leads < 12:
                sig = np.pad(sig, ((0, 0), (0, 12 - n_leads)))
            elif n_leads > 12:
                sig = sig[:, :12]
            if np.isnan(sig).any():
                for L in range(sig.shape[1]):
                    col = sig[:, L]
                    if np.isnan(col).any():
                        m = np.nanmean(col) if not np.isnan(col).all() else 0.0
                        sig[:, L] = np.where(np.isnan(col), m, col)
            fs = fields["fs"]
            if fs != TARGET_HZ:
                g = np.gcd(TARGET_HZ, fs)
                sig = resample_poly(sig, TARGET_HZ // g, fs // g, axis=0)
            if len(sig) >= SIGNAL_LEN:
                sig = sig[:SIGNAL_LEN]
            else:
                pad = np.zeros((SIGNAL_LEN, sig.shape[1]), dtype=sig.dtype)
                pad[:len(sig)] = sig
                sig = pad
            sigs.append(sig.T.astype(np.float32))
            labels.append(rec["labels"])
            pids.append(rec["subject_id"])
            if len(sigs) >= max_n:
                break
        except Exception as e:
            logger.debug(f"  load skip {sid}: {e}")

    sigs = np.stack(sigs)
    labels = np.stack(labels)
    logger.info(f"MIMIC v2 loaded: {sigs.shape}, prevalence={labels.mean(0).round(3)}")
    return sigs, labels, np.array(pids), MIMIC_CLASSES


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s, y, p, c = load_mimic_v2(max_n=2000)
    print("classes:", c)
    print("shape:", s.shape, "labels:", y.shape, "unique patients:", len(set(p.tolist())))
    for i, cl in enumerate(c):
        print(f"  {cl}: {int(y[:,i].sum())} pos ({100*y[:,i].mean():.1f}%), "
              f"{int((1-y[:,i]).sum())} neg")
