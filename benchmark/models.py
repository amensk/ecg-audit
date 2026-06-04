"""Model loading and feature extraction for all 5 FMs and S4 baseline.

Each FM is loaded with frozen weights. Feature extraction produces
fixed-size embeddings for downstream linear probing.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import N_LEADS, SIGNAL_LENGTH, FM_NAMES

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FMWrapper(nn.Module):
    """Base wrapper for ECG foundation models.

    Subclasses implement _load_backbone and _extract to handle
    architecture-specific loading and forward passes.
    """

    def __init__(self, model_id: str, checkpoint_path: Path):
        super().__init__()
        self.model_id = model_id
        self.name = FM_NAMES[model_id]
        self.checkpoint_path = checkpoint_path
        self.backbone = None
        self.embedding_dim = None

    def _load_backbone(self):
        raise NotImplementedError

    def load(self):
        self._load_backbone()
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.to(DEVICE)
        logger.info(f"Loaded {self.name} ({self.model_id}), embedding_dim={self.embedding_dim}")

    @torch.no_grad()
    def extract_features(self, signals: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Extract frozen features from signals (N, 12, 5000)."""
        tensor = torch.FloatTensor(signals)
        dataset = TensorDataset(tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        features = []
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            feat = self._extract(batch)
            features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MERLWrapper(FMWrapper):
    """MERL: Contrastive text-signal alignment, ResNet18 backbone."""

    def __init__(self, checkpoint_path: Path):
        super().__init__("M1", checkpoint_path)
        self.embedding_dim = 512

    def _load_backbone(self):
        from torchvision.models import resnet18

        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Identity()

        state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        encoder_keys = {
            k.replace("encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        if encoder_keys:
            self.backbone.load_state_dict(encoder_keys, strict=False)
        else:
            self.backbone.load_state_dict(state_dict, strict=False)

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, 12, 5000) treat as 1-channel 2D
        return self.backbone(x)


class STMEMWrapper(FMWrapper):
    """ST-MEM: Masked spatio-temporal reconstruction, ViT backbone."""

    def __init__(self, checkpoint_path: Path):
        super().__init__("M2", checkpoint_path)
        self.embedding_dim = 768

    def _load_backbone(self):
        state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]

        from st_mem import create_encoder

        self.backbone = create_encoder()
        self.backbone.load_state_dict(state_dict, strict=False)

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class HuBERTECGWrapper(FMWrapper):
    """HuBERT-ECG: Masked hidden-unit prediction, Transformer encoder."""

    def __init__(self, checkpoint_path: Path):
        super().__init__("M3", checkpoint_path)
        self.embedding_dim = 768

    def _load_backbone(self):
        state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]

        from hubert_ecg import HuBERTECG

        self.backbone = HuBERTECG()
        self.backbone.load_state_dict(state_dict, strict=False)

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone.extract_features(x)
        if output.dim() == 3:
            return output.mean(dim=1)
        return output


class ECGFMKEDWrapper(FMWrapper):
    """ECGFM-KED: Knowledge-enhanced multimodal learning."""

    def __init__(self, checkpoint_path: Path):
        super().__init__("M4", checkpoint_path)
        self.embedding_dim = 768

    def _load_backbone(self):
        state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]

        from ecgfm_ked import ECGFMKED

        self.backbone = ECGFMKED()
        self.backbone.load_state_dict(state_dict, strict=False)

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone.encode(x)
        if output.dim() == 3:
            return output.mean(dim=1)
        return output


class ECGFMWrapper(FMWrapper):
    """ECG-FM: Hybrid contrastive+generative pretraining, Transformer."""

    def __init__(self, checkpoint_path: Path):
        super().__init__("M5", checkpoint_path)
        self.embedding_dim = 768

    def _load_backbone(self):
        state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]

        from ecg_fm import ECGFM

        self.backbone = ECGFM()
        self.backbone.load_state_dict(state_dict, strict=False)

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone.extract_features(x)
        if output.dim() == 3:
            return output.mean(dim=1)
        return output


FM_REGISTRY = {
    "M1": MERLWrapper,
    "M2": STMEMWrapper,
    "M3": HuBERTECGWrapper,
    "M4": ECGFMKEDWrapper,
    "M5": ECGFMWrapper,
}


def load_fm(model_id: str, checkpoint_path: Path) -> FMWrapper:
    wrapper_cls = FM_REGISTRY[model_id]
    wrapper = wrapper_cls(checkpoint_path)
    wrapper.load()
    return wrapper


def extract_and_save_features(
    model: FMWrapper,
    signals: np.ndarray,
    split_indices: dict,
    output_dir: Path,
    dataset_id: str,
) -> dict:
    """Extract features for all splits and save to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    features = {}

    for split_name, indices in split_indices.items():
        split_signals = signals[indices]
        feats = model.extract_features(split_signals)
        features[split_name] = feats

        np.save(
            output_dir / f"{dataset_id}_{model.model_id}_{split_name}_features.npy",
            feats,
        )
        logger.info(
            f"Extracted {split_name} features for {model.name} on {dataset_id}: "
            f"shape={feats.shape}"
        )

    return features
