#!/usr/bin/env python3
"""
HuBERT-ECG feature extractor using the validated local safetensors checkpoint.

Canonical preprocessing (matches HuBERT-ECG repo dataset.py + finetune scripts,
all of which use --downsampling_factor=5):
  1. Take 12-lead ECG at 500 Hz, shape (12, T>=2500).
  2. Crop first 5 seconds: (12, 2500).
  3. NaN imputation with non-NaN mean (matches repo).
  4. Flatten leads: reshape(-1) -> (30000,).
  5. Decimate by 5 (anti-aliased): -> (6000,).
  6. Forward through HuBERTECG, mean-pool last_hidden_state over time -> (768,).

The checkpoint loads via AutoModel.from_pretrained(..., trust_remote_code=True)
after importing the local `hubert_ecg` package (registers the custom model).
"""
from pathlib import Path
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
HUBERT_SRC = WS / "checkpoints" / "_sources" / "HuBERT-ECG"
HUBERT_CKPT_DIR = WS / "checkpoints" / "hubert_ecg"

SAMPLES_5S_500HZ = 2500
DOWNSAMPLE_FACTOR = 5
EMB_DIM = 768

_MODEL = None  # lazy singleton


def _load_hubert():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import torch
    if str(HUBERT_SRC) not in sys.path:
        sys.path.insert(0, str(HUBERT_SRC))
    import hubert_ecg  # noqa: F401 (registers HuBERTECG with AutoModel)
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        str(HUBERT_CKPT_DIR), trust_remote_code=True, local_files_only=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    n_params = sum(x.numel() for x in model.parameters())
    logger.info(f"HuBERT-ECG loaded: {n_params:,} params")
    _MODEL = model
    return model


def _preprocess(sigs_raw: np.ndarray) -> np.ndarray:
    """(N,12,>=2500) raw @500Hz -> (N, 6000) canonical HuBERT input."""
    from scipy.signal import decimate
    out = np.empty((len(sigs_raw), 6000), dtype=np.float32)
    for i, sig in enumerate(sigs_raw):
        x = sig[:, :SAMPLES_5S_500HZ]            # (12, 2500)
        nan = np.isnan(x)
        if nan.any():
            x = np.where(nan, x[~nan].mean(), x)
        x = x.reshape(-1)                         # (30000,)
        x = decimate(x, DOWNSAMPLE_FACTOR)        # (6000,)
        out[i] = x.astype(np.float32)
    return out


def extract_hubert(sigs_raw: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """Extract frozen HuBERT-ECG embeddings.

    Args:
        sigs_raw: (N, 12, >=2500) RAW signals at 500 Hz (NOT z-scored).
        batch_size: forward batch size.
    Returns:
        (N, 768) mean-pooled embeddings.
    """
    import torch
    model = _load_hubert()
    X = _preprocess(sigs_raw)
    feats = np.empty((len(X), EMB_DIM), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).float()
            out = model(batch)
            hs = out.last_hidden_state            # (B, T, 768)
            feats[i:i + batch_size] = hs.mean(dim=1).cpu().numpy()
    return feats


if __name__ == "__main__":
    # Sanity check: load + forward on random input
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(0)
    dummy = rng.standard_normal((4, 12, 5000)).astype(np.float32)
    f = extract_hubert(dummy)
    print(f"dummy features: shape={f.shape}, any_nan={np.isnan(f).any()}, "
          f"mean={f.mean():.4f}, std={f.std():.4f}")
