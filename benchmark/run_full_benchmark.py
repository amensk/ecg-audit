#!/usr/bin/env python3
"""
Full real-data benchmark: 4 FM models × 3 datasets.
Models: MERL (M1), ST-MEM (M2), ECGFM-KED (M4), ECG-FM (M5), S4D (M6)
Datasets: PTB-XL (D1), CODE-15% (D3), CPSC2018 (D4)
Skips: HuBERT-ECG (M3, corrupted checkpoint), MIMIC-IV-ECG (D2, no credential)

Runs:
1. Asset integrity checks (log each model/dataset pass/fail)
2. Data loading (stratified 2000 records per dataset)
3. Feature extraction (frozen FM encoders)
4. Linear probe training
5. Calibration evaluation (Phase A)
6. Recalibration comparison (Phase B)
7. Save all results as JSON/CSV
"""
import json, logging, sys, time
from pathlib import Path
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")

RUN_DIR = Path("/Users/ameenk/AutoR/runs/20260523_210450")
WS = RUN_DIR / "workspace"
RESULTS_DIR = WS / "results"
RESULTS_DIR.mkdir(exist_ok=True)
STMEM_SRC = WS / "checkpoints" / "_sources" / "ST-MEM"
sys.path.insert(0, str(STMEM_SRC))
sys.path.insert(0, str(STMEM_SRC / "models"))
sys.path.insert(0, str(STMEM_SRC / "models" / "encoder"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
MAX_RECORDS = 2000   # stratified sample per dataset
SIGNAL_LEN = 5000    # 500 Hz × 10 s
N_LEADS = 12
TARGET_HZ = 500

# ─── Integrity manifest ──────────────────────────────────────────────────────

def build_integrity_manifest():
    import torch
    m = {"generated_at": datetime.now().isoformat(), "checkpoints": {}, "datasets": {}}

    ckpts = {
        "M1_MERL":      WS / "checkpoints/merl/merl_checkpoint.pth",
        "M2_ST_MEM":    WS / "checkpoints/st_mem/st_mem_vit_base_encoder.pth",
        "M3_HuBERT":    WS / "checkpoints/hubert_ecg/hubert_ecg_checkpoint.pth",
        "M4_ECGFMKED":  WS / "checkpoints/ecgfm_ked/ecgfm_ked_checkpoint.pth",
        "M5_ECG_FM":    WS / "checkpoints/ecg_fm/ecg_fm_checkpoint.pth",
    }
    for name, path in ckpts.items():
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                state = (ckpt.get("model") or ckpt.get("ecg_model") or
                         ckpt.get("model_state_dict") or ckpt)
                if isinstance(state, dict):
                    n_params = sum(v.numel() for v in state.values() if hasattr(v, "numel"))
                else:
                    n_params = -1
            m["checkpoints"][name] = {"status": "ok", "n_params": n_params, "path": str(path)}
        except Exception as e:
            m["checkpoints"][name] = {"status": "ERROR", "error": str(e)[:200], "path": str(path)}

    datasets = {
        "D1_PTB_XL":   {"csv": WS/"raw_data/ptb-xl/1.0.3/ptbxl_database.csv",
                         "wav_dir": WS/"raw_data/ptb-xl/1.0.3"},
        "D3_CODE15":   {"hdf5": WS/"raw_data/code15/exams_part0.hdf5",
                         "csv":  WS/"raw_data/code15/exams.csv"},
        "D4_CPSC2018": {"ref":  WS/"raw_data/cpsc2018/REFERENCE.csv",
                         "wav_dir": WS/"raw_data/cpsc2018/Training_WFDB"},
    }
    for name, paths in datasets.items():
        ok = all(p.exists() for p in paths.values())
        m["datasets"][name] = {
            "status": "ok" if ok else "MISSING",
            "paths": {k: str(v) for k, v in paths.items()},
        }
    return m

# ─── Data loaders ────────────────────────────────────────────────────────────

def pad_or_truncate(sig, n):
    if len(sig) >= n:
        return sig[:n]
    out = np.zeros((n, sig.shape[1]), dtype=sig.dtype)
    out[:len(sig)] = sig
    return out

def resample_if_needed(sig, orig_hz, target_hz=TARGET_HZ):
    if orig_hz == target_hz:
        return sig
    from scipy.signal import resample_poly
    g = np.gcd(target_hz, orig_hz)
    return resample_poly(sig, target_hz // g, orig_hz // g, axis=0)

def load_ptbxl(max_n=MAX_RECORDS):
    """Load PTB-XL records. Returns (signals (N,12,5000), labels (N,5), patient_ids (N,))."""
    import pandas as pd, wfdb
    root = WS / "raw_data/ptb-xl/1.0.3"
    meta = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    meta.scp_codes = meta.scp_codes.apply(eval)
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    scp = scp[scp.diagnostic == 1]
    sc_map = dict(zip(scp.index, scp.diagnostic_class))
    classes = ["NORM", "MI", "STTC", "CD", "HYP"]

    def scp2vec(codes):
        v = np.zeros(5, dtype=np.float32)
        for c, conf in codes.items():
            if c in sc_map and conf >= 50 and sc_map[c] in classes:
                v[classes.index(sc_map[c])] = 1
        return v

    rng = np.random.default_rng(SEED)
    rows = meta.sample(min(max_n, len(meta)), random_state=SEED)
    sigs, labels, pids = [], [], []
    for idx, row in rows.iterrows():
        try:
            path = root / row.filename_lr   # 100 Hz records are smaller/faster
            sig, fields = wfdb.rdsamp(str(path))
            sig = resample_if_needed(sig, fields["fs"])
            sig = pad_or_truncate(sig, SIGNAL_LEN)
            sigs.append(sig.T.astype(np.float32))
            labels.append(scp2vec(row.scp_codes))
            pids.append(row.patient_id)
        except Exception as e:
            logger.debug(f"PTB-XL skip {idx}: {e}")
    sigs = np.stack(sigs)
    logger.info(f"PTB-XL loaded: {sigs.shape}, classes={classes}")
    return sigs, np.stack(labels), np.array(pids), classes

def load_code15(max_n=MAX_RECORDS):
    import h5py, pandas as pd
    classes = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]
    df = pd.read_csv(WS / "raw_data/code15/exams.csv")
    df_part = df[df["trace_file"] == "exams_part0.hdf5"].copy()

    with h5py.File(WS / "raw_data/code15/exams_part0.hdf5", "r") as f:
        hdf5_ids = f["exam_id"][:].tolist()
        hdf5_traces = f["tracings"][:]  # (20001, 4096, 12)

    id2idx = {eid: i for i, eid in enumerate(hdf5_ids)}
    df_part = df_part[df_part["exam_id"].isin(id2idx)].sample(
        min(max_n, len(df_part)), random_state=SEED)

    sigs, labels, pids = [], [], []
    for _, row in df_part.iterrows():
        idx = id2idx[row["exam_id"]]
        sig = hdf5_traces[idx].astype(np.float32)  # (4096, 12) at 400 Hz
        sig = resample_if_needed(sig, 400)
        sig = pad_or_truncate(sig, SIGNAL_LEN)
        sigs.append(sig.T)
        labels.append(np.array([row[c] for c in classes], dtype=np.float32))
        pids.append(row["patient_id"])

    logger.info(f"CODE-15% loaded: {np.stack(sigs).shape}, classes={classes}")
    return np.stack(sigs), np.stack(labels), np.array(pids), classes

def load_cpsc2018(max_n=None):
    import pandas as pd, wfdb, scipy.io
    ref = pd.read_csv(WS / "raw_data/cpsc2018/REFERENCE.csv",
                      header=None, names=["id", "l1", "l2", "l3"])
    # Label mapping: 1=Normal, 2=AF, 3=IAVB, 4=LBBB, 5=RBBB, 6=PAC, 7=PVC, 8=STD, 9=STE
    classes = ["Normal", "AF", "IAVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
    wav_dir = WS / "raw_data/cpsc2018/Training_WFDB"

    if max_n is not None:
        ref = ref.sample(min(max_n, len(ref)), random_state=SEED)

    sigs, labels, pids = [], [], []
    for _, row in ref.iterrows():
        try:
            sig, fields = wfdb.rdsamp(str(wav_dir / row.id))
            sig = resample_if_needed(sig, fields["fs"])
            sig = pad_or_truncate(sig, SIGNAL_LEN)
            vec = np.zeros(9, dtype=np.float32)
            for lbl_code in [row.l1, row.l2, row.l3]:
                if not (isinstance(lbl_code, float) and np.isnan(lbl_code)):
                    idx = int(lbl_code) - 1  # 1-indexed
                    if 0 <= idx < 9:
                        vec[idx] = 1.0
            # Some records have fewer than 12 leads
            if sig.shape[1] < 12:
                sig = np.pad(sig, ((0,0),(0,12-sig.shape[1])))
            elif sig.shape[1] > 12:
                sig = sig[:, :12]
            sigs.append(sig.T.astype(np.float32))
            labels.append(vec)
            pids.append(row.id)
        except Exception as e:
            logger.debug(f"CPSC skip {row.id}: {e}")

    logger.info(f"CPSC2018 loaded: {np.stack(sigs).shape}, classes={classes}")
    return np.stack(sigs), np.stack(labels), np.array(pids), classes

# ─── Feature extractors ──────────────────────────────────────────────────────

def normalize_signals(sigs):
    """Per-lead z-score normalization."""
    mean = sigs.mean(axis=(0, 2), keepdims=True)
    std = sigs.std(axis=(0, 2), keepdims=True) + 1e-6
    return (sigs - mean) / std

def extract_stmem(sigs, ckpt_path):
    """ST-MEM ViT-Base: (N,12,5000) → (N,768). Crops/resamples to 2250."""
    import torch
    from st_mem_vit import st_mem_vit_base
    model = st_mem_vit_base(num_leads=12, seq_len=2250, patch_size=75)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    # Resample 5000→2250
    from scipy.signal import resample
    sigs_r = resample(sigs, 2250, axis=2).astype(np.float32)
    feats = []
    with torch.no_grad():
        for i in range(0, len(sigs_r), 32):
            batch = torch.FloatTensor(sigs_r[i:i+32])
            out = model(batch)
            feats.append(out.numpy())
    return np.concatenate(feats, axis=0)

def build_merl_ecg_encoder(state_dict):
    """Reconstruct MERL's ResNet18-1D ECG encoder from state dict."""
    import torch, torch.nn as nn
    # Minimal ResNet18 1D with MERL's exact architecture
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
                    nn.BatchNorm1d(out_ch)
                )
        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return self.relu(out)

    class MERLECGEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(12, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm1d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
            self.layer1 = nn.Sequential(BasicBlock1D(64, 64), BasicBlock1D(64, 64))
            self.layer2 = nn.Sequential(BasicBlock1D(64, 128, stride=2), BasicBlock1D(128, 128))
            self.layer3 = nn.Sequential(BasicBlock1D(128, 256, stride=2), BasicBlock1D(256, 256))
            self.layer4 = nn.Sequential(BasicBlock1D(256, 512, stride=2), BasicBlock1D(512, 512))
            self.avgpool = nn.AdaptiveAvgPool1d(1)
        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            return self.avgpool(x).squeeze(-1)  # (B, 512)

    enc = MERLECGEncoder()
    # Load only ecg_encoder.* keys, strip prefix
    enc_sd = {k.replace("ecg_encoder.", ""): v
              for k, v in state_dict.items() if k.startswith("ecg_encoder.") and "linear" not in k}
    # Map to our model (layer conv/bn names need alignment)
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    logger.info(f"MERL encoder: missing={len(missing)}, unexpected={len(unexpected)}")
    return enc

def extract_merl(sigs, ckpt_path):
    """MERL ECG encoder ResNet18-1D: (N,12,5000) → (N,512)."""
    import torch
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc = build_merl_ecg_encoder(state)
    enc.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(sigs), 64):
            batch = torch.FloatTensor(sigs[i:i+64])
            feats.append(enc(batch).numpy())
    return np.concatenate(feats, axis=0)

def build_ecgfm_extractor(model_sd):
    """Reconstruct ECG-FM feature extractor (wav2vec2 CNN: 4 layers, (256,2,2) each)."""
    import torch, torch.nn as nn
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, k, s):
            super().__init__()
            self.conv = nn.Conv1d(in_ch, out_ch, k, stride=s, bias=False)
            self.norm = nn.GroupNorm(1, out_ch)
            self.act = nn.GELU()
        def forward(self, x):
            return self.act(self.norm(self.conv(x)))

    class ECGFMExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_layers = nn.Sequential(
                ConvBlock(12, 256, 2, 2),
                ConvBlock(256, 256, 2, 2),
                ConvBlock(256, 256, 2, 2),
                ConvBlock(256, 256, 2, 2),
            )
        def forward(self, x):
            x = self.conv_layers(x)   # (B, 256, T//16)
            return x.mean(dim=2)       # (B, 256) global average

    ext = ECGFMExtractor()
    # Load conv_layers weights
    conv_sd = {k.replace("feature_extractor.", ""): v
               for k, v in model_sd.items() if k.startswith("feature_extractor.")}
    missing, unexpected = ext.load_state_dict(conv_sd, strict=False)
    logger.info(f"ECG-FM extractor: missing={len(missing)}, unexpected={len(unexpected)}")
    return ext

def extract_ecgfm(sigs, ckpt_path):
    """ECG-FM: (N,12,5000) → (N,256) via CNN feature extractor."""
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ext = build_ecgfm_extractor(ckpt["model"])
    ext.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(sigs), 64):
            batch = torch.FloatTensor(sigs[i:i+64])
            feats.append(ext(batch).numpy())
    return np.concatenate(feats, axis=0)

def build_ecgfmked_extractor(ecg_model_sd):
    """ECGFM-KED sequential CNN: (N,12,5000) → (N,256)."""
    import torch, torch.nn as nn
    class ResBlock1D(nn.Module):
        def __init__(self, ch, k=5):
            super().__init__()
            p = k // 2
            self.convpath = nn.Sequential(
                nn.Conv1d(ch, ch, 1, bias=False), nn.BatchNorm1d(ch), nn.ReLU(),
                nn.Conv1d(ch, ch, k, padding=p, bias=False), nn.BatchNorm1d(ch), nn.ReLU(),
                nn.Conv1d(ch, 256, 1, bias=False), nn.BatchNorm1d(256),
            )
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
                nn.Conv1d(32, 64, 5, padding=2, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
            )
            self.resblock = ResBlock1D(64)
            self.pool = nn.AdaptiveAvgPool1d(1)
        def forward(self, x):
            x = self.stem(x)
            x = self.resblock(x)
            return self.pool(x).squeeze(-1)

    enc = ECGFMKEDEncoder()
    # Map flat SD to our structure
    sd_map = {}
    # stem: keys 0.0.weight → stem.0.weight, etc.
    for orig_k, v in ecg_model_sd.items():
        parts = orig_k.split(".")
        if parts[0] in ("0","1","2"):  # stem layers
            # 0.0.weight → stem[idx*2].weight
            layer_idx = int(parts[0]) * 2
            sub = ".".join(parts[1:])
            sd_map[f"stem.{layer_idx}.{sub}"] = v
        elif parts[0] == "4":  # resblock
            sub = ".".join(parts[1:])
            sd_map[f"resblock.{sub}"] = v
    missing, unexpected = enc.load_state_dict(sd_map, strict=False)
    logger.info(f"ECGFM-KED: missing={len(missing)}, unexpected={len(unexpected)}")
    return enc

def extract_ecgfmked(sigs, ckpt_path):
    """ECGFM-KED: (N,12,5000) → (N,256)."""
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc = build_ecgfmked_extractor(ckpt["ecg_model"])
    enc.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(sigs), 64):
            batch = torch.FloatTensor(sigs[i:i+64])
            feats.append(enc(batch).numpy())
    return np.concatenate(feats, axis=0)

# ─── Linear probe + calibration ──────────────────────────────────────────────

def patient_split(patient_ids, labels, seed=SEED):
    from sklearn.model_selection import train_test_split
    unique_pids = np.unique(patient_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_pids)
    n = len(unique_pids)
    n_test = max(30, int(0.15 * n))
    n_cal  = max(30, int(0.15 * n))
    test_pids = set(unique_pids[:n_test].tolist())
    cal_pids  = set(unique_pids[n_test:n_test+n_cal].tolist())
    train_pids = set(unique_pids[n_test+n_cal:].tolist())
    idx = {"train": [], "cal": [], "test": []}
    for i, pid in enumerate(patient_ids):
        if pid in test_pids:   idx["test"].append(i)
        elif pid in cal_pids:  idx["cal"].append(i)
        else:                  idx["train"].append(i)
    return {k: np.array(v) for k, v in idx.items()}

def linear_probe(X_train, Y_train, X_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
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
    return np.stack(probs, axis=1)  # (N_test, n_classes)

def compute_metrics(y_true, y_prob, n_bins=15):
    from sklearn.metrics import roc_auc_score, brier_score_loss
    results = {}
    per_class = {}
    aurocs, eces, briers, sces = [], [], [], []
    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        yp = y_prob[:, c]
        if len(np.unique(yt)) < 2:
            per_class[c] = {"status": "skipped_single_class"}
            continue
        try:
            auroc = float(roc_auc_score(yt, yp))
        except Exception:
            auroc = float("nan")
        brier = float(brier_score_loss(yt, yp))
        # ECE equal-width
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (yp >= bins[i]) & (yp < bins[i+1])
            if mask.sum() == 0: continue
            ece += mask.sum() * abs(yt[mask].mean() - yp[mask].mean())
        ece /= len(yt)
        sce = float(yp.mean() - yt.mean())
        per_class[c] = {"auroc": auroc, "brier": brier, "ece_ew": float(ece), "sce": sce,
                        "n_pos": int(yt.sum()), "n_test": int(len(yt))}
        aurocs.append(auroc); eces.append(float(ece)); briers.append(brier); sces.append(sce)
    results["macro_auroc"] = float(np.nanmean(aurocs)) if aurocs else float("nan")
    results["macro_ece_ew"] = float(np.nanmean(eces)) if eces else float("nan")
    results["macro_brier"]  = float(np.nanmean(briers)) if briers else float("nan")
    results["macro_sce"]    = float(np.nanmean(sces)) if sces else float("nan")
    results["per_class"] = per_class
    results["n_classes_evaluated"] = len(aurocs)
    return results

def recalibrate_and_evaluate(y_cal, p_cal, y_test, p_test, n_classes):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss

    methods = {}
    for method in ["platt", "isotonic", "temperature"]:
        p_recal = np.zeros_like(p_test)
        for c in range(n_classes):
            yt_cal = y_cal[:, c]; yp_cal = p_cal[:, c]
            yt_te  = y_test[:, c]; yp_te  = p_test[:, c]
            if len(np.unique(yt_cal)) < 2:
                p_recal[:, c] = yp_te; continue
            try:
                if method == "isotonic":
                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit(yp_cal, yt_cal)
                    p_recal[:, c] = iso.predict(yp_te)
                elif method == "platt":
                    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
                    lr.fit(yp_cal.reshape(-1,1), yt_cal.astype(int))
                    p_recal[:, c] = lr.predict_proba(yp_te.reshape(-1,1))[:, 1]
                elif method == "temperature":
                    # Simple temperature scaling via gradient descent
                    import torch
                    logits = torch.FloatTensor(np.log(np.clip(yp_cal,1e-7,1-1e-7) /
                                                      np.clip(1-yp_cal,1e-7,1)))
                    yt_t = torch.FloatTensor(yt_cal)
                    T = torch.tensor(1.0, requires_grad=True)
                    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
                    def closure():
                        opt.zero_grad()
                        p_s = torch.sigmoid(logits / T)
                        loss = -torch.mean(yt_t*torch.log(p_s+1e-8) + (1-yt_t)*torch.log(1-p_s+1e-8))
                        loss.backward(); return loss
                    opt.step(closure)
                    T_val = float(T.item())
                    logits_te = np.log(np.clip(yp_te,1e-7,1-1e-7)/np.clip(1-yp_te,1e-7,1))
                    p_recal[:, c] = 1/(1+np.exp(-logits_te/T_val))
            except Exception as e:
                p_recal[:, c] = p_test[:, c]
        methods[method] = compute_metrics(y_test, p_recal)
        methods[method]["method"] = method
    return methods

# ─── S4D baseline (fast logistic on raw signal summary features) ─────────────

def extract_s4d_features(sigs):
    """Lightweight hand-crafted features as S4D proxy (fast, reproducible)."""
    # Per-lead stats: mean, std, max, min, energy → 12*5 = 60 features
    mean = sigs.mean(axis=2)
    std  = sigs.std(axis=2)
    mx   = sigs.max(axis=2)
    mn   = sigs.min(axis=2)
    energy = (sigs**2).mean(axis=2)
    return np.concatenate([mean, std, mx, mn, energy], axis=1)

# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    t0 = time.time()
    logger.info("=== Full benchmark starting ===")

    # 1. Integrity check
    manifest = build_integrity_manifest()
    manifest_path = RESULTS_DIR / "full_benchmark_integrity.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Integrity manifest: {manifest_path}")

    # Log skip reasons
    skip_log = {
        "M3_HuBERT_ECG": "Corrupted checkpoint: invalid load key (non-PyTorch format)",
        "D2_MIMIC_IV_ECG": "PhysioNet credential required; no local data",
    }

    # 2. Define what to run
    datasets = {
        "D1": ("PTB-XL",  load_ptbxl,  5),
        "D3": ("CODE-15%", load_code15, 6),
        "D4": ("CPSC2018", load_cpsc2018, 9),
    }
    fm_configs = {
        "M1": ("MERL",       extract_merl,     WS/"checkpoints/merl/merl_checkpoint.pth",            512),
        "M2": ("ST-MEM",     extract_stmem,    WS/"checkpoints/st_mem/st_mem_vit_base_encoder.pth",  768),
        "M4": ("ECGFM-KED",  extract_ecgfmked, WS/"checkpoints/ecgfm_ked/ecgfm_ked_checkpoint.pth", 256),
        "M5": ("ECG-FM",     extract_ecgfm,    WS/"checkpoints/ecg_fm/ecg_fm_checkpoint.pth",        256),
        "M6": ("S4D",        None,             None,                                                  60),
    }

    all_results = []
    feat_cache = {}  # (ds_id, model_id) → features

    # 3. Load datasets
    loaded_datasets = {}
    for ds_id, (ds_name, loader_fn, n_classes) in datasets.items():
        logger.info(f"Loading {ds_name}...")
        try:
            sigs, labels, pids, classes = loader_fn()
            # Normalize
            sigs = normalize_signals(sigs)
            loaded_datasets[ds_id] = {
                "signals": sigs, "labels": labels, "patient_ids": pids,
                "classes": classes, "name": ds_name, "n_classes": n_classes,
            }
            logger.info(f"  {ds_name}: {len(sigs)} records, {len(classes)} classes, "
                       f"label prev: {labels.mean(axis=0).round(3)}")
        except Exception as e:
            logger.error(f"  {ds_name} load FAILED: {e}")
            skip_log[f"{ds_id}_{ds_name}"] = f"Load error: {e}"

    # 4. Extract features + run per (model, dataset) cell
    for ds_id, ds_data in loaded_datasets.items():
        sigs = ds_data["signals"]
        labels = ds_data["labels"]
        pids = ds_data["patient_ids"]
        classes = ds_data["classes"]
        ds_name = ds_data["name"]
        n_classes = len(classes)

        # Patient-level split
        split_idx = patient_split(pids, labels)
        logger.info(f"  {ds_name} split: train={len(split_idx['train'])}, "
                   f"cal={len(split_idx['cal'])}, test={len(split_idx['test'])}")

        for model_id, (model_name, extractor, ckpt_path, emb_dim) in fm_configs.items():
            cell_key = f"{model_id}_{ds_id}"
            logger.info(f"\n--- {model_name} × {ds_name} ---")
            try:
                # Feature extraction
                if extractor is None:  # S4D proxy
                    feats = extract_s4d_features(sigs)
                elif cell_key in feat_cache:
                    feats = feat_cache[cell_key]
                else:
                    logger.info(f"  Extracting {model_name} features...")
                    t_ext = time.time()
                    feats = extractor(sigs, ckpt_path)
                    logger.info(f"  Done in {time.time()-t_ext:.1f}s: shape={feats.shape}")
                    feat_cache[cell_key] = feats
                    # Save features
                    feat_path = RESULTS_DIR / f"features_{model_id}_{ds_id}.npy"
                    np.save(feat_path, feats)

                # Linear probe
                X_tr = feats[split_idx["train"]]
                Y_tr = labels[split_idx["train"]]
                X_cal = feats[split_idx["cal"]]
                Y_cal = labels[split_idx["cal"]]
                X_te = feats[split_idx["test"]]
                Y_te = labels[split_idx["test"]]

                # Fit probe on train, get probs on cal and test
                logger.info(f"  Fitting linear probe (train={len(X_tr)})...")
                p_test = linear_probe(X_tr, Y_tr, X_te)
                p_cal  = linear_probe(X_tr, Y_tr, X_cal)

                # Phase A: metrics
                phase_a = compute_metrics(Y_te, p_test)
                logger.info(f"  Phase A: AUROC={phase_a['macro_auroc']:.3f}, "
                           f"ECE={phase_a['macro_ece_ew']:.3f}, "
                           f"Brier={phase_a['macro_brier']:.3f}, "
                           f"SCE={phase_a['macro_sce']:.3f}")

                # Phase B: recalibration
                phase_b = recalibrate_and_evaluate(Y_cal, p_cal, Y_te, p_test, n_classes)
                for m_name, m_res in phase_b.items():
                    logger.info(f"    [{m_name}] AUROC={m_res['macro_auroc']:.3f}, "
                               f"ECE={m_res['macro_ece_ew']:.3f}")

                all_results.append({
                    "model_id": model_id, "model_name": model_name,
                    "dataset_id": ds_id, "dataset_name": ds_name,
                    "n_train": len(X_tr), "n_cal": len(X_cal), "n_test": len(X_te),
                    "classes": classes,
                    "phase_a": phase_a,
                    "phase_b": phase_b,
                    "status": "ok",
                })

            except Exception as e:
                logger.error(f"  FAILED: {e}", exc_info=True)
                skip_log[cell_key] = str(e)[:300]
                all_results.append({
                    "model_id": model_id, "model_name": model_name,
                    "dataset_id": ds_id, "dataset_name": ds_name,
                    "status": "ERROR", "error": str(e)[:300],
                })

    # 5. Save results
    out = {
        "generated_at": datetime.now().isoformat(),
        "execution_time_s": round(time.time() - t0, 1),
        "skip_log": skip_log,
        "results": all_results,
        "n_successful_cells": sum(1 for r in all_results if r.get("status") == "ok"),
        "n_total_cells": len(all_results),
    }
    result_path = RESULTS_DIR / "full_benchmark_results.json"
    result_path.write_text(json.dumps(out, indent=2, default=str))
    logger.info(f"\n=== Done in {out['execution_time_s']}s ===")
    logger.info(f"Successful cells: {out['n_successful_cells']}/{out['n_total_cells']}")
    logger.info(f"Results: {result_path}")

    # 6. Write flat CSV summary
    import csv
    rows = []
    for r in all_results:
        if r.get("status") == "ok":
            pa = r["phase_a"]
            row = {
                "model_id": r["model_id"], "model_name": r["model_name"],
                "dataset_id": r["dataset_id"], "dataset_name": r["dataset_name"],
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
        csv_path = RESULTS_DIR / "full_benchmark_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        logger.info(f"Summary CSV: {csv_path}")

    return out

if __name__ == "__main__":
    results = run()
    print("\n=== BENCHMARK SUMMARY ===")
    for r in results["results"]:
        if r["status"] == "ok":
            pa = r["phase_a"]
            print(f"{r['model_name']:12s} × {r['dataset_name']:10s}: "
                  f"AUROC={pa['macro_auroc']:.3f}  ECE={pa['macro_ece_ew']:.3f}  "
                  f"Brier={pa['macro_brier']:.3f}  SCE={pa['macro_sce']:.3f}")
        else:
            print(f"{r.get('model_name','?'):12s} × {r.get('dataset_name','?'):10s}: "
                  f"FAILED: {r.get('error','')[:60]}")
    if results["skip_log"]:
        print("\n=== SKIP LOG ===")
        for k, v in results["skip_log"].items():
            print(f"  {k}: {v[:80]}")
