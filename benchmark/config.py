"""Central configuration for the ECG FM calibration study."""

from pathlib import Path
from dataclasses import dataclass, field

RUN_DIR = Path("/Users/ameenk/AutoR/runs/20260523_210450")
WORKSPACE = RUN_DIR / "workspace"
CODE_DIR = WORKSPACE / "code"
DATA_DIR = WORKSPACE / "data"
RESULTS_DIR = WORKSPACE / "results"
FIGURES_DIR = WORKSPACE / "figures"
NOTES_DIR = WORKSPACE / "notes"

RANDOM_SEED = 42
N_BOOTSTRAP = 10_000
ECE_N_BINS = 15
SIGNIFICANCE_LEVEL = 0.05

DATASET_IDS = ["D1", "D2", "D3", "D4"]
DATASET_NAMES = {
    "D1": "PTB-XL",
    "D2": "MIMIC-IV-ECG",
    "D3": "CODE-15%",
    "D4": "CPSC2018",
}

FM_IDS = ["M1", "M2", "M3", "M4", "M5"]
FM_NAMES = {
    "M1": "MERL",
    "M2": "ST-MEM",
    "M3": "HuBERT-ECG",
    "M4": "ECGFM-KED",
    "M5": "ECG-FM",
}
ALL_MODEL_IDS = FM_IDS + ["M6"]
ALL_MODEL_NAMES = {**FM_NAMES, "M6": "S4"}

RECAL_IDS = ["R1", "R2", "R3", "R4"]
RECAL_NAMES = {
    "R1": "temperature_scaling",
    "R2": "platt_scaling",
    "R3": "isotonic_regression",
    "R4": "learned_mlp_head",
}

SPLIT_FRACTIONS = {"train": 0.70, "cal": 0.15, "test": 0.15}
MLP_CAL_SUBSPLIT = {"train": 0.80, "val": 0.20}

H4_ABLATION_SIZES = [500, 1000, 2000, 5000, 10000]
H4_ABLATION_REPS = 5
H4_PRIMARY_DATASETS = ["D1", "D3"]

SAMPLING_RATE_HZ = 500
TARGET_DURATION_S = 10
SIGNAL_LENGTH = SAMPLING_RATE_HZ * TARGET_DURATION_S  # 5000 samples
N_LEADS = 12


@dataclass
class S4Hyperparams:
    learning_rates: list = field(default_factory=lambda: [1e-4, 3e-4, 1e-3])
    weight_decays: list = field(default_factory=lambda: [0, 1e-4, 1e-3])
    max_epochs: int = 100
    patience: int = 15
    batch_size: int = 128


@dataclass
class MLPRecalConfig:
    hidden_dim: int = 64
    dropout: float = 0.3
    lr: float = 1e-3
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 256


@dataclass
class LinearProbeConfig:
    lr: float = 1e-3
    max_epochs: int = 50
    patience: int = 10
    batch_size: int = 256
