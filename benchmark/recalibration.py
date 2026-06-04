"""Post-hoc recalibration methods: temperature, Platt, isotonic, MLP.

All methods are fitted on calibration-set predictions (logits or probs)
and applied to test-set predictions. Parameters are saved for
reproducibility.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from config import MLPRecalConfig, RANDOM_SEED, MLP_CAL_SUBSPLIT

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemperatureScaler:
    """R1: Temperature scaling — single parameter per class, fitted via NLL."""

    def __init__(self):
        self.temperatures = None

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        n_classes = logits.shape[1]
        self.temperatures = np.ones(n_classes)

        for c in range(n_classes):
            def nll(T):
                scaled = logits[:, c] / T[0]
                probs = 1 / (1 + np.exp(-scaled))
                probs = np.clip(probs, 1e-10, 1 - 1e-10)
                return -np.mean(
                    labels[:, c] * np.log(probs) + (1 - labels[:, c]) * np.log(1 - probs)
                )

            result = minimize(nll, x0=[1.0], method="L-BFGS-B", bounds=[(0.01, 100.0)])
            self.temperatures[c] = result.x[0]

        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperatures[np.newaxis, :]
        return 1 / (1 + np.exp(-scaled))

    def get_params(self) -> dict:
        return {"temperatures": self.temperatures.tolist()}


class PlattScaler:
    """R2: Platt scaling — logistic regression per class on logits."""

    def __init__(self):
        self.models = []

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "PlattScaler":
        n_classes = logits.shape[1]
        self.models = []

        for c in range(n_classes):
            lr = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1e10,  # minimal regularization
            )
            lr.fit(logits[:, c:c+1], labels[:, c])
            self.models.append(lr)

        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        n_classes = logits.shape[1]
        probs = np.zeros_like(logits, dtype=np.float64)
        for c in range(n_classes):
            probs[:, c] = self.models[c].predict_proba(logits[:, c:c+1])[:, 1]
        return probs

    def get_params(self) -> dict:
        params = []
        for m in self.models:
            params.append({
                "coef": m.coef_.tolist(),
                "intercept": m.intercept_.tolist(),
            })
        return {"per_class_params": params}


class IsotonicCalibrator:
    """R3: Isotonic regression per class on predicted probabilities."""

    def __init__(self):
        self.models = []

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        n_classes = probs.shape[1]
        self.models = []

        for c in range(n_classes):
            ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            ir.fit(probs[:, c], labels[:, c])
            self.models.append(ir)

        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        n_classes = probs.shape[1]
        calibrated = np.zeros_like(probs)
        for c in range(n_classes):
            calibrated[:, c] = self.models[c].predict(probs[:, c])
        return calibrated

    def get_params(self) -> dict:
        params = []
        for m in self.models:
            params.append({
                "x_thresholds": m.X_thresholds_.tolist() if hasattr(m, 'X_thresholds_') else [],
                "y_thresholds": m.y_thresholds_.tolist() if hasattr(m, 'y_thresholds_') else [],
            })
        return {"per_class_params": params}


class MLPRecalHead(nn.Module):
    """R4: Learned MLP recalibration head.

    Architecture: Linear(C, 64) -> ReLU -> Dropout(0.3) -> Linear(64, C)
    """

    def __init__(self, n_classes: int, config: MLPRecalConfig = MLPRecalConfig()):
        super().__init__()
        self.config = config
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(n_classes, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPCalibrator:
    """Wrapper for MLP recalibration head training and inference."""

    def __init__(self, config: MLPRecalConfig = MLPRecalConfig()):
        self.config = config
        self.model = None

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "MLPCalibrator":
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        n = len(logits)
        n_classes = logits.shape[1]
        n_train = int(n * MLP_CAL_SUBSPLIT["train"])

        perm = np.random.permutation(n)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:]

        train_logits = torch.FloatTensor(logits[train_idx])
        train_labels = torch.FloatTensor(labels[train_idx])
        val_logits = torch.FloatTensor(logits[val_idx])
        val_labels = torch.FloatTensor(labels[val_idx])

        self.model = MLPRecalHead(n_classes, self.config).to(DEVICE)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        criterion = nn.BCEWithLogitsLoss()

        train_ds = TensorDataset(train_logits, train_labels)
        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True
        )

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.config.max_epochs):
            self.model.train()
            for batch_logits, batch_labels in train_loader:
                batch_logits = batch_logits.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                out = self.model(batch_logits)
                loss = criterion(out, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_out = self.model(val_logits.to(DEVICE))
                val_loss = criterion(val_out, val_labels.to(DEVICE)).item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(logits).to(DEVICE))
            return torch.sigmoid(out).cpu().numpy()

    def get_params(self) -> dict:
        return {
            "state_dict_keys": list(self.model.state_dict().keys()),
            "n_parameters": sum(p.numel() for p in self.model.parameters()),
        }


RECALIBRATION_REGISTRY = {
    "R1": ("temperature_scaling", TemperatureScaler, "logits"),
    "R2": ("platt_scaling", PlattScaler, "logits"),
    "R3": ("isotonic_regression", IsotonicCalibrator, "probs"),
    "R4": ("learned_mlp_head", MLPCalibrator, "logits"),
}


def apply_recalibration(
    method_id: str,
    cal_logits: np.ndarray,
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_logits: np.ndarray,
    test_probs: np.ndarray,
) -> dict:
    """Fit a recalibration method on calibration set, apply to test set."""
    name, cls, input_type = RECALIBRATION_REGISTRY[method_id]

    calibrator = cls()
    if input_type == "logits":
        calibrator.fit(cal_logits, cal_labels)
        calibrated_probs = calibrator.transform(test_logits)
    else:
        calibrator.fit(cal_probs, cal_labels)
        calibrated_probs = calibrator.transform(test_probs)

    return {
        "method_id": method_id,
        "method_name": name,
        "calibrated_probs": calibrated_probs,
        "params": calibrator.get_params(),
        "calibrator": calibrator,
    }


def run_all_recalibrations(
    model_id: str,
    dataset_id: str,
    cal_logits: np.ndarray,
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    test_logits: np.ndarray,
    test_probs: np.ndarray,
    output_dir: Path,
) -> dict:
    """Apply all 4 recalibration methods and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for method_id in RECALIBRATION_REGISTRY:
        result = apply_recalibration(
            method_id, cal_logits, cal_probs, cal_labels, test_logits, test_probs
        )
        results[method_id] = result

        np.save(
            output_dir / f"{dataset_id}_{model_id}_{method_id}_calibrated_probs.npy",
            result["calibrated_probs"],
        )

        params_file = output_dir / f"{dataset_id}_{model_id}_{method_id}_params.json"
        with open(params_file, "w") as f:
            json.dump(result["params"], f, indent=2, default=str)

        logger.info(f"Recalibrated {model_id}/{dataset_id} with {result['method_name']}")

    return results
