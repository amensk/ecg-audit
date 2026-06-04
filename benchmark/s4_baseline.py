"""S4D supervised baseline training from scratch.

Trains an S4D model end-to-end on each dataset with hyperparameter
search over learning rate and weight decay.
"""

import json
import logging
from pathlib import Path
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from config import S4Hyperparams, RANDOM_SEED, N_LEADS

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class S4DLayer(nn.Module):
    """Diagonal structured state-space layer (S4D).

    Implements the diagonal variant of S4 with DPLR initialization.
    """

    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        log_A_real = torch.log(0.5 * torch.ones(d_model, d_state))
        A_imag = np.pi * torch.arange(d_state).float().repeat(d_model, 1)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D)"""
        residual = x
        x = self.norm(x)

        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        B = self.B.unsqueeze(0)  # (1, D, N)
        C = self.C.unsqueeze(0)

        L = x.shape[1]
        k = torch.arange(L, device=x.device).float()

        vandermonde = torch.exp(A.unsqueeze(-1) * k.unsqueeze(0).unsqueeze(0))
        BC = (C.squeeze(0) * B.squeeze(0)).to(vandermonde.dtype)
        kernel = torch.einsum("dn,dnl->dl", BC, vandermonde).real

        x_fft = torch.fft.rfft(x.transpose(1, 2), n=2 * L)
        k_fft = torch.fft.rfft(kernel, n=2 * L)
        y = torch.fft.irfft(x_fft * k_fft.unsqueeze(0), n=2 * L)[..., :L]
        y = y.transpose(1, 2)

        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        y = self.dropout(self.output_linear(y))
        return y + residual


class S4DModel(nn.Module):
    """S4D model for multi-label ECG classification."""

    def __init__(
        self,
        n_leads: int = 12,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 4,
        n_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_leads, d_model)
        self.layers = nn.ModuleList(
            [S4DLayer(d_model, d_state, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 12, 5000) -> logits (B, n_classes)"""
        x = x.transpose(1, 2)  # (B, 5000, 12)
        x = self.input_proj(x)  # (B, 5000, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pooling
        return self.classifier(x)


def train_s4_single(
    train_signals: np.ndarray,
    train_labels: np.ndarray,
    val_signals: np.ndarray,
    val_labels: np.ndarray,
    n_classes: int,
    lr: float,
    wd: float,
    hp: S4Hyperparams,
) -> dict:
    """Train a single S4D model with given hyperparameters."""
    torch.manual_seed(RANDOM_SEED)

    model = S4DModel(n_leads=N_LEADS, n_classes=n_classes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.max_epochs)

    train_ds = TensorDataset(
        torch.FloatTensor(train_signals),
        torch.FloatTensor(train_labels),
    )
    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True, drop_last=True)

    best_auroc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(hp.max_epochs):
        model.train()
        for sigs, lbls in train_loader:
            sigs, lbls = sigs.to(DEVICE), lbls.to(DEVICE)
            logits = model(sigs)
            loss = criterion(logits, lbls)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_ds = TensorDataset(torch.FloatTensor(val_signals))
            val_loader = DataLoader(val_ds, batch_size=hp.batch_size, shuffle=False)
            val_logits_list = []
            for (batch,) in val_loader:
                val_logits_list.append(model(batch.to(DEVICE)).cpu())
            val_logits = torch.cat(val_logits_list)
            val_probs = torch.sigmoid(val_logits).numpy()

        try:
            auroc = roc_auc_score(val_labels, val_probs, average="macro")
        except ValueError:
            auroc = 0.5

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= hp.patience:
                break

    model.load_state_dict(best_state)
    return {"model": model, "best_auroc": best_auroc, "lr": lr, "wd": wd}


def train_s4_with_search(
    signals: np.ndarray,
    labels: np.ndarray,
    split_indices: dict,
    n_classes: int,
    hp: S4Hyperparams = S4Hyperparams(),
) -> dict:
    """Hyperparameter search over lr and wd, selecting by validation AUROC."""
    train_sigs = signals[split_indices["train"]]
    train_lbls = labels[split_indices["train"]]
    val_sigs = signals[split_indices["cal"]]
    val_lbls = labels[split_indices["cal"]]

    best_result = None
    for lr, wd in product(hp.learning_rates, hp.weight_decays):
        logger.info(f"S4 training: lr={lr}, wd={wd}")
        result = train_s4_single(
            train_sigs, train_lbls, val_sigs, val_lbls, n_classes, lr, wd, hp
        )
        if best_result is None or result["best_auroc"] > best_result["best_auroc"]:
            best_result = result
            logger.info(f"  New best: AUROC={result['best_auroc']:.4f}")

    logger.info(
        f"S4 best: lr={best_result['lr']}, wd={best_result['wd']}, "
        f"AUROC={best_result['best_auroc']:.4f}"
    )
    return best_result


def get_s4_predictions(
    model: S4DModel,
    signals: np.ndarray,
    batch_size: int = 128,
) -> dict:
    """Get logits and probabilities from trained S4 model."""
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(signals))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_logits = []
    all_probs = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            probs = torch.sigmoid(logits)
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return {
        "logits": np.concatenate(all_logits, axis=0),
        "probs": np.concatenate(all_probs, axis=0),
    }


def run_s4_pipeline(
    dataset_id: str,
    signals: np.ndarray,
    labels: np.ndarray,
    split_indices: dict,
    n_classes: int,
    output_dir: Path,
) -> dict:
    """Full S4 pipeline: hyperparameter search, train, predict, save."""
    output_dir.mkdir(parents=True, exist_ok=True)

    result = train_s4_with_search(signals, labels, split_indices, n_classes)
    model = result["model"]

    predictions = {}
    for split_name in ["train", "cal", "test"]:
        sigs = signals[split_indices[split_name]]
        preds = get_s4_predictions(model, sigs)
        predictions[split_name] = preds
        np.savez(
            output_dir / f"{dataset_id}_M6_{split_name}_predictions.npz",
            logits=preds["logits"],
            probs=preds["probs"],
        )

    torch.save(
        model.state_dict(),
        output_dir / f"{dataset_id}_M6_s4_model.pt",
    )

    meta = {
        "model_id": "M6",
        "dataset_id": dataset_id,
        "best_lr": result["lr"],
        "best_wd": result["wd"],
        "best_val_auroc": result["best_auroc"],
        "n_classes": n_classes,
    }
    with open(output_dir / f"{dataset_id}_M6_s4_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {"predictions": predictions, "meta": meta}
