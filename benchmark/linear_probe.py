"""Linear probe training on frozen FM features.

Fits a linear classifier on top of frozen FM embeddings for each
dataset's training split, following the frozen linear probe protocol.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from config import LinearProbeConfig, RANDOM_SEED

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    n_classes: int,
    config: LinearProbeConfig = LinearProbeConfig(),
) -> dict:
    """Train a linear probe with early stopping on validation AUROC."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    input_dim = train_features.shape[1]
    model = LinearProbe(input_dim, n_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(
        torch.FloatTensor(train_features),
        torch.FloatTensor(train_labels),
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True
    )

    best_auroc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        for feats, labels in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            logits = model(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.FloatTensor(val_features).to(DEVICE))
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        try:
            val_auroc = roc_auc_score(val_labels, val_probs, average="macro")
        except ValueError:
            val_auroc = 0.5

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}, best AUROC={best_auroc:.4f}")
                break

    model.load_state_dict(best_state)
    return {"model": model, "best_auroc": best_auroc, "epochs_trained": epoch + 1}


def get_predictions(
    model: LinearProbe,
    features: np.ndarray,
    batch_size: int = 512,
) -> dict:
    """Get logits and probabilities from a trained linear probe."""
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(features))
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


def run_linear_probe_pipeline(
    model_id: str,
    dataset_id: str,
    features: dict,
    labels: np.ndarray,
    split_indices: dict,
    n_classes: int,
    output_dir: Path,
) -> dict:
    """Full linear probe pipeline: train, predict on all splits, save."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_feats = features["train"]
    train_labels = labels[split_indices["train"]]

    cal_feats = features["cal"]
    cal_labels = labels[split_indices["cal"]]
    val_feats = cal_feats
    val_labels = cal_labels

    result = train_linear_probe(
        train_feats, train_labels, val_feats, val_labels, n_classes
    )
    probe = result["model"]

    predictions = {}
    for split_name in ["train", "cal", "test"]:
        preds = get_predictions(probe, features[split_name])
        predictions[split_name] = preds
        np.savez(
            output_dir / f"{dataset_id}_{model_id}_{split_name}_predictions.npz",
            logits=preds["logits"],
            probs=preds["probs"],
        )

    torch.save(
        probe.state_dict(),
        output_dir / f"{dataset_id}_{model_id}_linear_probe.pt",
    )

    meta = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "input_dim": train_feats.shape[1],
        "n_classes": n_classes,
        "best_val_auroc": result["best_auroc"],
        "epochs_trained": result["epochs_trained"],
    }
    with open(output_dir / f"{dataset_id}_{model_id}_probe_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"Linear probe {model_id}/{dataset_id}: "
        f"val_AUROC={result['best_auroc']:.4f}, epochs={result['epochs_trained']}"
    )
    return {"predictions": predictions, "meta": meta}
