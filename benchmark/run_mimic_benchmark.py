#!/usr/bin/env python3
"""
MIMIC-IV-ECG benchmark: stage records from ZIP, extract features, run calibration audit.

Labels from machine_measurements.csv text reports → 5 binary classes:
  Normal, AF, RBBB, LBBB, LVH

ECG-FM was pretrained on MIMIC-IV-ECG → within-pretraining-domain calibration test.
"""
import json, logging, sys, time, tempfile, os, shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")

RUN_DIR = Path("/Users/ameenk/AutoR/runs/20260523_210450")
WS = RUN_DIR / "workspace"
RESULTS_DIR = WS / "results"
RESULTS_DIR.mkdir(exist_ok=True)
STAGING_DIR = WS / "raw_data" / "mimic-iv-ecg"
MIMIC_ZIP = WS / "raw_data" / "mimic-iv-ecg_zip_download" / "mimic-iv-ecg-1.0.zip"
ZIP_PREFIX = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"

STMEM_SRC = WS / "checkpoints" / "_sources" / "ST-MEM"
sys.path.insert(0, str(STMEM_SRC))
sys.path.insert(0, str(STMEM_SRC / "models"))
sys.path.insert(0, str(STMEM_SRC / "models" / "encoder"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
MAX_RECORDS = 2000
SIGNAL_LEN = 5000
N_LEADS = 12
TARGET_HZ = 500

MIMIC_CLASSES = ["Normal", "AF", "RBBB", "LBBB", "LVH"]

LABEL_PATTERNS = {
    "Normal": ["sinus rhythm", "normal sinus", "normal ecg", "normal tracing"],
    "AF":     ["atrial fibrillation"],
    "RBBB":   ["right bundle branch block", "rbbb", "r.b.b.b"],
    "LBBB":   ["left bundle branch block", "lbbb", "l.b.b.b"],
    "LVH":    ["left ventricular hypertrophy", "lvh", "voltage criteria for lvh"],
}

# ─── Label extraction ─────────────────────────────────────────────────────────

def extract_labels_from_reports(row: dict) -> np.ndarray:
    """Parse 18 report columns into binary label vector [Normal, AF, RBBB, LBBB, LVH]."""
    text = " ".join(
        row.get(f"report_{i}", "") for i in range(18)
    ).lower()
    vec = np.zeros(len(MIMIC_CLASSES), dtype=np.float32)
    for i, cls in enumerate(MIMIC_CLASSES):
        if any(pat in text for pat in LABEL_PATTERNS[cls]):
            vec[i] = 1.0
    return vec

# ─── Build label index from ZIP ───────────────────────────────────────────────

def build_mimic_index(n_sample=MAX_RECORDS, seed=SEED):
    """Read CSVs from ZIP, sample n_sample records with >=1 positive label."""
    import zipfile as zf, csv, io
    logger.info("Building MIMIC-IV-ECG label index from ZIP...")

    with zf.ZipFile(str(MIMIC_ZIP), "r") as z:
        # record_list: study_id → path
        rec_data = z.read(ZIP_PREFIX + "record_list.csv").decode("utf-8")
        rec_reader = list(csv.DictReader(rec_data.splitlines()))
        path_map = {r["study_id"]: r["path"] for r in rec_reader}
        pid_map  = {r["study_id"]: r["subject_id"] for r in rec_reader}

        # machine_measurements: study_id → label row
        meas_data = z.read(ZIP_PREFIX + "machine_measurements.csv").decode("utf-8")
        meas_reader = list(csv.DictReader(meas_data.splitlines()))

    logger.info(f"  Records: {len(path_map)}, Measurements: {len(meas_reader)}")

    # Build labeled index
    labeled = []
    rng = np.random.default_rng(seed)
    indices = list(range(len(meas_reader)))
    rng.shuffle(indices)

    for i in indices:
        row = meas_reader[i]
        sid = row["study_id"]
        if sid not in path_map:
            continue
        vec = extract_labels_from_reports(row)
        # Keep records with at least one label; skip unlabeled
        if vec.sum() == 0:
            continue
        labeled.append({
            "study_id": sid,
            "subject_id": pid_map[sid],
            "path": path_map[sid],
            "labels": vec,
        })
        if len(labeled) >= n_sample * 3:  # oversample to allow for load failures
            break

    logger.info(f"  Labeled records found: {len(labeled)}")

    # Prevalence check
    vecs = np.stack([r["labels"] for r in labeled])
    for i, cls in enumerate(MIMIC_CLASSES):
        logger.info(f"    {cls}: {vecs[:, i].sum():.0f} / {len(labeled)} ({100*vecs[:, i].mean():.1f}%)")

    return labeled

# ─── Stage records from ZIP ───────────────────────────────────────────────────

def stage_mimic_records(labeled: list, staging_dir: Path):
    """Extract .hea and .dat files for selected records to staging_dir."""
    import zipfile as zf
    staging_dir.mkdir(parents=True, exist_ok=True)

    already_staged = {f.stem for f in staging_dir.glob("*.hea")}
    to_stage = [r for r in labeled if r["study_id"] not in already_staged]

    if not to_stage:
        logger.info(f"All {len(labeled)} records already staged in {staging_dir}")
        return

    logger.info(f"Staging {len(to_stage)} records from ZIP to {staging_dir}...")
    t0 = time.time()

    with zf.ZipFile(str(MIMIC_ZIP), "r") as z:
        for rec in to_stage:
            path = rec["path"]  # e.g. files/p1000/p10000032/s40689238/40689238
            study_id = rec["study_id"]
            for ext in [".hea", ".dat"]:
                zip_path = ZIP_PREFIX + path + ext
                dest = staging_dir / (study_id + ext)
                if not dest.exists():
                    try:
                        data = z.read(zip_path)
                        dest.write_bytes(data)
                    except Exception as e:
                        logger.debug(f"  Skip {zip_path}: {e}")

    logger.info(f"  Staged in {time.time()-t0:.1f}s")

# ─── Load MIMIC signals ───────────────────────────────────────────────────────

def resample_if_needed(sig, orig_hz, target_hz=TARGET_HZ):
    if orig_hz == target_hz:
        return sig
    from scipy.signal import resample_poly
    g = np.gcd(target_hz, orig_hz)
    return resample_poly(sig, target_hz // g, orig_hz // g, axis=0)

def pad_or_truncate(sig, n):
    if len(sig) >= n:
        return sig[:n]
    out = np.zeros((n, sig.shape[1]), dtype=sig.dtype)
    out[:len(sig)] = sig
    return out

def load_mimic(max_n=MAX_RECORDS):
    """Load MIMIC-IV-ECG records from staging dir."""
    import wfdb

    labeled = build_mimic_index(n_sample=max_n)
    staging_dir = STAGING_DIR / "staged_ecgs"
    stage_mimic_records(labeled[:max_n * 3], staging_dir)

    sigs, labels, pids = [], [], []
    for rec in labeled:
        study_id = rec["study_id"]
        hea_path = staging_dir / (study_id + ".hea")
        dat_path = staging_dir / (study_id + ".dat")

        if not hea_path.exists() or not dat_path.exists():
            continue

        try:
            sig, fields = wfdb.rdsamp(str(staging_dir / study_id))
            orig_hz = fields["fs"]
            n_leads = sig.shape[1]

            # Ensure 12 leads
            if n_leads < 12:
                sig = np.pad(sig, ((0, 0), (0, 12 - n_leads)))
            elif n_leads > 12:
                sig = sig[:, :12]

            # Replace NaN with per-lead mean (some MIMIC leads are missing)
            if np.isnan(sig).any():
                for lead in range(sig.shape[1]):
                    col = sig[:, lead]
                    if np.isnan(col).any():
                        col_mean = np.nanmean(col) if not np.isnan(col).all() else 0.0
                        sig[:, lead] = np.where(np.isnan(col), col_mean, col)

            sig = resample_if_needed(sig, orig_hz)
            sig = pad_or_truncate(sig, SIGNAL_LEN)
            sigs.append(sig.T.astype(np.float32))
            labels.append(rec["labels"])
            pids.append(rec["subject_id"])

            if len(sigs) >= max_n:
                break
        except Exception as e:
            logger.debug(f"  MIMIC load skip {study_id}: {e}")

    logger.info(f"MIMIC-IV-ECG loaded: {len(sigs)} records, {len(MIMIC_CLASSES)} classes")
    sigs_arr = np.stack(sigs)
    labels_arr = np.stack(labels)
    logger.info(f"  Label prevalences: {labels_arr.mean(axis=0).round(3)}")
    return sigs_arr, labels_arr, np.array(pids), MIMIC_CLASSES

# ─── Feature extractors (from run_full_benchmark.py) ─────────────────────────

def normalize_signals(sigs):
    mean = sigs.mean(axis=(0, 2), keepdims=True)
    std  = sigs.std(axis=(0, 2), keepdims=True) + 1e-6
    return (sigs - mean) / std

def extract_stmem(sigs, ckpt_path):
    import torch
    from st_mem_vit import st_mem_vit_base
    model = st_mem_vit_base(num_leads=12, seq_len=2250, patch_size=75)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    from scipy.signal import resample
    sigs_r = resample(sigs, 2250, axis=2).astype(np.float32)
    feats = []
    with torch.no_grad():
        for i in range(0, len(sigs_r), 32):
            batch = torch.FloatTensor(sigs_r[i:i+32])
            feats.append(model(batch).numpy())
    return np.concatenate(feats, axis=0)

def build_merl_ecg_encoder(state_dict):
    import torch, torch.nn as nn
    class BasicBlock1D(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm1d(out_ch)
            self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm1d(out_ch)
            self.relu = nn.ReLU(inplace=True)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_ch != out_ch:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_ch))
        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.relu(out + self.shortcut(x))

    class MERLECGEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(12, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm1d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
            self.layer1 = nn.Sequential(BasicBlock1D(64, 64), BasicBlock1D(64, 64))
            self.layer2 = nn.Sequential(BasicBlock1D(64, 128, 2), BasicBlock1D(128, 128))
            self.layer3 = nn.Sequential(BasicBlock1D(128, 256, 2), BasicBlock1D(256, 256))
            self.layer4 = nn.Sequential(BasicBlock1D(256, 512, 2), BasicBlock1D(512, 512))
            self.avgpool = nn.AdaptiveAvgPool1d(1)
        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            return self.avgpool(x).squeeze(-1)

    enc = MERLECGEncoder()
    enc_sd = {k.replace("ecg_encoder.", ""): v
              for k, v in state_dict.items()
              if k.startswith("ecg_encoder.") and "linear" not in k}
    enc.load_state_dict(enc_sd, strict=False)
    return enc

def extract_merl(sigs, ckpt_path):
    import torch
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc = build_merl_ecg_encoder(state)
    enc.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(sigs), 64):
            feats.append(enc(torch.FloatTensor(sigs[i:i+64])).numpy())
    return np.concatenate(feats, axis=0)

def build_ecgfm_extractor(model_sd):
    import torch, torch.nn as nn
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, k, s):
            super().__init__()
            self.conv = nn.Conv1d(in_ch, out_ch, k, stride=s, bias=False)
            self.norm = nn.GroupNorm(1, out_ch)
            self.act  = nn.GELU()
        def forward(self, x):
            return self.act(self.norm(self.conv(x)))

    class ECGFMExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_layers = nn.Sequential(
                ConvBlock(12, 256, 2, 2), ConvBlock(256, 256, 2, 2),
                ConvBlock(256, 256, 2, 2), ConvBlock(256, 256, 2, 2))
        def forward(self, x):
            return self.conv_layers(x).mean(dim=2)

    ext = ECGFMExtractor()
    conv_sd = {k.replace("feature_extractor.", ""): v
               for k, v in model_sd.items() if k.startswith("feature_extractor.")}
    ext.load_state_dict(conv_sd, strict=False)
    return ext

def extract_ecgfm(sigs, ckpt_path):
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ext = build_ecgfm_extractor(ckpt["model"])
    ext.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(sigs), 64):
            feats.append(ext(torch.FloatTensor(sigs[i:i+64])).numpy())
    return np.concatenate(feats, axis=0)

def build_ecgfmked_extractor(ecg_model_sd):
    import torch, torch.nn as nn
    class ResBlock1D(nn.Module):
        def __init__(self, ch, k=5):
            super().__init__()
            p = k // 2
            self.convpath = nn.Sequential(
                nn.Conv1d(ch, ch, 1, bias=False), nn.BatchNorm1d(ch), nn.ReLU(),
                nn.Conv1d(ch, ch, k, padding=p, bias=False), nn.BatchNorm1d(ch), nn.ReLU(),
                nn.Conv1d(ch, 256, 1, bias=False), nn.BatchNorm1d(256))
            self.idpath = nn.Sequential(nn.Conv1d(ch, 256, 1, bias=False), nn.BatchNorm1d(256))
            self.act = nn.ReLU()
        def forward(self, x):
            return self.act(self.convpath(x) + self.idpath(x))

    class ECGFMKEDEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(12, 32, 5, padding=2, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 32, 5, padding=2, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 64, 5, padding=2, bias=False), nn.BatchNorm1d(64), nn.ReLU())
            self.resblock = ResBlock1D(64)
            self.pool = nn.AdaptiveAvgPool1d(1)
        def forward(self, x):
            return self.pool(self.resblock(self.stem(x))).squeeze(-1)

    enc = ECGFMKEDEncoder()
    sd_map = {}
    for orig_k, v in ecg_model_sd.items():
        parts = orig_k.split(".")
        if parts[0] in ("0", "1", "2"):
            layer_idx = int(parts[0]) * 2
            sd_map[f"stem.{layer_idx}.{'.'.join(parts[1:])}"] = v
        elif parts[0] == "4":
            sd_map[f"resblock.{'.'.join(parts[1:])}"] = v
    enc.load_state_dict(sd_map, strict=False)
    return enc

def extract_ecgfmked(sigs, ckpt_path):
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc = build_ecgfmked_extractor(ckpt["ecg_model"])
    enc.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(sigs), 64):
            feats.append(enc(torch.FloatTensor(sigs[i:i+64])).numpy())
    return np.concatenate(feats, axis=0)

def extract_s4d_features(sigs):
    mean = sigs.mean(axis=2); std = sigs.std(axis=2)
    mx = sigs.max(axis=2); mn = sigs.min(axis=2)
    energy = (sigs**2).mean(axis=2)
    return np.concatenate([mean, std, mx, mn, energy], axis=1)

# ─── Probe + metrics ──────────────────────────────────────────────────────────

def patient_split(patient_ids, seed=SEED):
    unique_pids = np.unique(patient_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_pids)
    n = len(unique_pids)
    n_test = max(30, int(0.15 * n))
    n_cal  = max(30, int(0.15 * n))
    test_pids = set(unique_pids[:n_test].tolist())
    cal_pids  = set(unique_pids[n_test:n_test+n_cal].tolist())
    idx = {"train": [], "cal": [], "test": []}
    for i, pid in enumerate(patient_ids):
        if pid in test_pids:   idx["test"].append(i)
        elif pid in cal_pids:  idx["cal"].append(i)
        else:                  idx["train"].append(i)
    return {k: np.array(v) for k, v in idx.items()}

def linear_probe(X_train, Y_train, X_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    # Impute NaN/Inf (robustness against rare encoder edge cases)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    probs = []
    for c in range(Y_train.shape[1]):
        y = Y_train[:, c]
        if len(np.unique(y)) < 2:
            probs.append(np.full(len(X_te), y.mean()))
            continue
        clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
        clf.fit(X_tr, y)
        probs.append(clf.predict_proba(X_te)[:, 1])
    return np.stack(probs, axis=1)

def compute_metrics(y_true, y_prob, n_bins=15):
    from sklearn.metrics import roc_auc_score, brier_score_loss
    per_class = {}
    aurocs, eces, briers, sces = [], [], [], []
    bins = np.linspace(0, 1, n_bins + 1)
    for c in range(y_true.shape[1]):
        yt, yp = y_true[:, c], y_prob[:, c]
        if len(np.unique(yt)) < 2:
            per_class[c] = {"status": "skipped"}; continue
        try:
            auroc = float(roc_auc_score(yt, yp))
        except Exception:
            auroc = float("nan")
        brier = float(brier_score_loss(yt, yp))
        ece = sum(
            mask.sum() * abs(yt[mask].mean() - yp[mask].mean())
            for i in range(n_bins)
            for mask in [(yp >= bins[i]) & (yp < bins[i+1])]
            if mask.sum() > 0
        ) / len(yt)
        sce = float(yp.mean() - yt.mean())
        per_class[c] = {"auroc": auroc, "brier": brier, "ece_ew": float(ece), "sce": sce,
                        "n_pos": int(yt.sum()), "n_test": int(len(yt))}
        aurocs.append(auroc); eces.append(float(ece))
        briers.append(brier); sces.append(sce)
    return {
        "macro_auroc": float(np.nanmean(aurocs)) if aurocs else float("nan"),
        "macro_ece_ew": float(np.nanmean(eces)) if eces else float("nan"),
        "macro_brier": float(np.nanmean(briers)) if briers else float("nan"),
        "macro_sce": float(np.nanmean(sces)) if sces else float("nan"),
        "per_class": per_class,
        "n_classes_evaluated": len(aurocs),
    }

def recalibrate_and_evaluate(y_cal, p_cal, y_test, p_test, n_classes):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    methods = {}
    for method in ["platt", "isotonic", "temperature"]:
        p_recal = np.zeros_like(p_test)
        for c in range(n_classes):
            yt_cal, yp_cal = y_cal[:, c], p_cal[:, c]
            yp_te = p_test[:, c]
            if len(np.unique(yt_cal)) < 2:
                p_recal[:, c] = yp_te; continue
            try:
                if method == "isotonic":
                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit(yp_cal, yt_cal)
                    p_recal[:, c] = iso.predict(yp_te)
                elif method == "platt":
                    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
                    lr.fit(yp_cal.reshape(-1, 1), yt_cal.astype(int))
                    p_recal[:, c] = lr.predict_proba(yp_te.reshape(-1, 1))[:, 1]
                elif method == "temperature":
                    import torch
                    logits = torch.FloatTensor(np.log(
                        np.clip(yp_cal, 1e-7, 1-1e-7) / np.clip(1-yp_cal, 1e-7, 1)))
                    yt_t = torch.FloatTensor(yt_cal)
                    T = torch.tensor(1.0, requires_grad=True)
                    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
                    def closure():
                        opt.zero_grad()
                        p_s = torch.sigmoid(logits / T)
                        loss = -torch.mean(yt_t*torch.log(p_s+1e-8)+(1-yt_t)*torch.log(1-p_s+1e-8))
                        loss.backward(); return loss
                    opt.step(closure)
                    T_val = float(T.item())
                    logits_te = np.log(np.clip(yp_te, 1e-7, 1-1e-7) / np.clip(1-yp_te, 1e-7, 1))
                    p_recal[:, c] = 1 / (1 + np.exp(-logits_te / T_val))
            except Exception as e:
                p_recal[:, c] = p_test[:, c]
        m = compute_metrics(y_test, p_recal)
        m["method"] = method
        methods[method] = m
    return methods

# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    t0 = time.time()
    logger.info("=== MIMIC-IV-ECG benchmark starting ===")

    # Load dataset
    logger.info("Loading MIMIC-IV-ECG (staging from ZIP)...")
    try:
        sigs, labels, pids, classes = load_mimic(max_n=MAX_RECORDS)
    except Exception as e:
        logger.error(f"MIMIC load FAILED: {e}", exc_info=True)
        result = {"status": "FAILED", "error": str(e), "dataset": "MIMIC-IV-ECG"}
        (RESULTS_DIR / "mimic_benchmark_results.json").write_text(json.dumps(result, indent=2))
        return result

    sigs = normalize_signals(sigs)
    n_classes = len(classes)
    split_idx = patient_split(pids)
    logger.info(f"  Split: train={len(split_idx['train'])}, cal={len(split_idx['cal'])}, test={len(split_idx['test'])}")

    fm_configs = {
        "M1": ("MERL",      extract_merl,     WS/"checkpoints/merl/merl_checkpoint.pth",           512),
        "M2": ("ST-MEM",    extract_stmem,    WS/"checkpoints/st_mem/st_mem_vit_base_encoder.pth", 768),
        "M4": ("ECGFM-KED", extract_ecgfmked, WS/"checkpoints/ecgfm_ked/ecgfm_ked_checkpoint.pth",256),
        "M5": ("ECG-FM",    extract_ecgfm,    WS/"checkpoints/ecg_fm/ecg_fm_checkpoint.pth",       256),
        "M6": ("S4D",       None,             None,                                                  60),
    }

    all_results = []
    skip_log = {}

    for model_id, (model_name, extractor, ckpt_path, emb_dim) in fm_configs.items():
        cell_key = f"{model_id}_D5_MIMIC"
        logger.info(f"\n--- {model_name} × MIMIC-IV-ECG ---")
        try:
            # Feature extraction
            feat_cache_path = RESULTS_DIR / f"features_{model_id}_D5.npy"
            if feat_cache_path.exists():
                feats = np.load(feat_cache_path)
                logger.info(f"  Loaded cached features: {feats.shape}")
            elif extractor is None:
                feats = extract_s4d_features(sigs)
            else:
                logger.info(f"  Extracting {model_name} features...")
                t_ext = time.time()
                feats = extractor(sigs, ckpt_path)
                logger.info(f"  Done in {time.time()-t_ext:.1f}s: {feats.shape}")
                np.save(feat_cache_path, feats)

            X_tr  = feats[split_idx["train"]]
            Y_tr  = labels[split_idx["train"]]
            X_cal = feats[split_idx["cal"]]
            Y_cal = labels[split_idx["cal"]]
            X_te  = feats[split_idx["test"]]
            Y_te  = labels[split_idx["test"]]

            p_test = linear_probe(X_tr, Y_tr, X_te)
            p_cal  = linear_probe(X_tr, Y_tr, X_cal)

            phase_a = compute_metrics(Y_te, p_test)
            phase_b = recalibrate_and_evaluate(Y_cal, p_cal, Y_te, p_test, n_classes)

            logger.info(f"  AUROC={phase_a['macro_auroc']:.3f}  ECE={phase_a['macro_ece_ew']:.3f}  "
                        f"Brier={phase_a['macro_brier']:.3f}  SCE={phase_a['macro_sce']:.3f}")

            all_results.append({
                "model_id": model_id, "model_name": model_name,
                "dataset_id": "D5", "dataset_name": "MIMIC-IV-ECG",
                "n_train": len(X_tr), "n_cal": len(X_cal), "n_test": len(X_te),
                "classes": classes,
                "phase_a": phase_a, "phase_b": phase_b,
                "status": "ok",
                "note": "ECG-FM pretrained on MIMIC-IV-ECG (within-domain)" if model_id == "M5" else "",
            })

        except Exception as e:
            logger.error(f"  FAILED: {e}", exc_info=True)
            skip_log[cell_key] = str(e)[:300]
            all_results.append({
                "model_id": model_id, "model_name": model_name,
                "dataset_id": "D5", "dataset_name": "MIMIC-IV-ECG",
                "status": "ERROR", "error": str(e)[:300],
            })

    out = {
        "generated_at": datetime.now().isoformat(),
        "execution_time_s": round(time.time() - t0, 1),
        "dataset": "MIMIC-IV-ECG",
        "n_records_loaded": len(sigs),
        "classes": classes,
        "skip_log": skip_log,
        "results": all_results,
        "n_successful": sum(1 for r in all_results if r.get("status") == "ok"),
    }

    out_path = RESULTS_DIR / "mimic_benchmark_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    logger.info(f"\n=== Done in {out['execution_time_s']}s === {out_path}")

    # Write CSV
    import csv as csv_mod
    rows = []
    for r in all_results:
        if r.get("status") == "ok":
            pa = r["phase_a"]
            row = {
                "model_id": r["model_id"], "model_name": r["model_name"],
                "dataset_id": "D5", "dataset_name": "MIMIC-IV-ECG",
                "n_test": r["n_test"],
                "macro_auroc": round(pa["macro_auroc"], 4),
                "macro_ece_ew": round(pa["macro_ece_ew"], 4),
                "macro_brier": round(pa["macro_brier"], 4),
                "macro_sce": round(pa["macro_sce"], 4),
            }
            for method, mr in r.get("phase_b", {}).items():
                row[f"{method}_auroc"] = round(mr["macro_auroc"], 4)
                row[f"{method}_ece"]   = round(mr["macro_ece_ew"], 4)
            rows.append(row)

    if rows:
        csv_path = RESULTS_DIR / "mimic_benchmark_summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        logger.info(f"CSV: {csv_path}")

    return out

if __name__ == "__main__":
    run()
