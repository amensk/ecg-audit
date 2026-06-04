"""Dataset loading, preprocessing, and patient-stratified splitting.

Handles PTB-XL, MIMIC-IV-ECG, CODE-15%, and CPSC2018.
Implements the 70/15/15 train/cal/test split with patient isolation.
"""

import json
import logging
from pathlib import Path
from typing import Optional
from collections import Counter

import numpy as np
import wfdb
from scipy.signal import resample_poly
from sklearn.model_selection import train_test_split

from config import (
    RANDOM_SEED,
    SAMPLING_RATE_HZ,
    SIGNAL_LENGTH,
    SPLIT_FRACTIONS,
    TARGET_DURATION_S,
    N_LEADS,
)

logger = logging.getLogger(__name__)


def resample_ecg(signal: np.ndarray, orig_hz: int, target_hz: int) -> np.ndarray:
    if orig_hz == target_hz:
        return signal
    return resample_poly(signal, target_hz, orig_hz, axis=0)


def pad_or_truncate(signal: np.ndarray, target_len: int) -> np.ndarray:
    n = signal.shape[0]
    if n >= target_len:
        return signal[:target_len]
    padded = np.zeros((target_len, signal.shape[1]), dtype=signal.dtype)
    padded[:n] = signal
    return padded


def compute_normalization_stats(signals: np.ndarray) -> dict:
    """Compute per-lead mean and std from training signals only."""
    mean = np.mean(signals, axis=(0, 2))  # shape (n_leads,)
    std = np.std(signals, axis=(0, 2)) + 1e-8
    return {"mean": mean, "std": std}


def normalize_signals(signals: np.ndarray, stats: dict) -> np.ndarray:
    mean = stats["mean"][np.newaxis, :, np.newaxis]  # (1, leads, 1)
    std = stats["std"][np.newaxis, :, np.newaxis]
    return (signals - mean) / std


def patient_stratified_split(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    train_frac: float = 0.70,
    cal_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = RANDOM_SEED,
) -> dict:
    """Split patient IDs into train/cal/test with stratification.

    For multi-label data, stratifies on the most frequent label combination
    to approximately preserve label distributions.
    """
    unique_patients = np.unique(patient_ids)

    patient_label_map = {}
    for pid in unique_patients:
        mask = patient_ids == pid
        patient_label_map[pid] = labels[mask][0]

    p_ids = np.array(list(patient_label_map.keys()))
    p_labels = np.array(list(patient_label_map.values()))

    if p_labels.ndim > 1:
        label_keys = ["".join(map(str, row)) for row in p_labels]
    else:
        label_keys = p_labels.tolist()

    def valid_strata(keys, min_count=2):
        counts = Counter(keys)
        return all(count >= min_count for count in counts.values())

    test_size = test_frac / (1.0 - 0.0)
    stratify_first = label_keys if valid_strata(label_keys) else None
    if stratify_first is None:
        logger.warning("Patient split falling back to unstratified test split due to rare strata")
    p_train_cal, p_test = train_test_split(
        p_ids, test_size=test_size, random_state=seed, stratify=stratify_first
    )

    mask_tc = np.isin(p_ids, p_train_cal)
    label_keys_tc = [lk for lk, m in zip(label_keys, mask_tc) if m]

    cal_size = cal_frac / (train_frac + cal_frac)
    stratify_second = label_keys_tc if valid_strata(label_keys_tc) else None
    if stratify_second is None:
        logger.warning("Patient split falling back to unstratified calibration split due to rare strata")
    p_train, p_cal = train_test_split(
        p_train_cal, test_size=cal_size, random_state=seed, stratify=stratify_second
    )

    return {
        "train": set(p_train.tolist()),
        "cal": set(p_cal.tolist()),
        "test": set(p_test.tolist()),
    }


def build_split_indices(patient_ids: np.ndarray, split_patients: dict) -> dict:
    """Map patient-level splits back to ECG-level indices."""
    indices = {"train": [], "cal": [], "test": []}
    for i, pid in enumerate(patient_ids):
        for split_name, pset in split_patients.items():
            if pid in pset:
                indices[split_name].append(i)
                break
    return {k: np.array(v) for k, v in indices.items()}


def verify_no_leakage(split_patients: dict) -> bool:
    train = split_patients["train"]
    cal = split_patients["cal"]
    test = split_patients["test"]
    assert len(train & cal) == 0, "Leakage: train-cal overlap"
    assert len(train & test) == 0, "Leakage: train-test overlap"
    assert len(cal & test) == 0, "Leakage: cal-test overlap"
    logger.info("Leakage check passed: no patient overlap between splits")
    return True


class DatasetLoader:
    """Base class for dataset loading."""

    def __init__(self, dataset_id: str, data_root: Path):
        self.dataset_id = dataset_id
        self.data_root = data_root

    def load(self) -> dict:
        raise NotImplementedError

    def preprocess(self, raw: dict) -> dict:
        raise NotImplementedError


class PTBXLLoader(DatasetLoader):
    """Loader for PTB-XL dataset."""

    def __init__(self, data_root: Path):
        super().__init__("D1", data_root)
        self.ptbxl_path = data_root / "ptb-xl" / "1.0.3"

    def load(self) -> dict:
        import pandas as pd

        metadata = pd.read_csv(self.ptbxl_path / "ptbxl_database.csv", index_col="ecg_id")
        metadata.scp_codes = metadata.scp_codes.apply(lambda x: eval(x))

        superclass_map = self._load_scp_statements()
        signals = []
        patient_ids = []
        labels = []
        ecg_ids = []

        for idx, row in metadata.iterrows():
            signal_path = self.ptbxl_path / row.filename_hr
            signal, _ = wfdb.rdsamp(str(signal_path).replace(".hea", "").replace(".dat", ""))
            signal = pad_or_truncate(signal, SIGNAL_LENGTH)

            label_vec = self._scp_to_superclass(row.scp_codes, superclass_map)

            signals.append(signal)
            patient_ids.append(row.patient_id)
            labels.append(label_vec)
            ecg_ids.append(idx)

        return {
            "signals": np.array(signals),  # (N, 5000, 12)
            "patient_ids": np.array(patient_ids),
            "labels": np.array(labels),  # (N, 5)
            "ecg_ids": np.array(ecg_ids),
            "class_names": ["NORM", "MI", "STTC", "CD", "HYP"],
            "n_classes": 5,
            "task_id": "ptbxl_super",
        }

    def _load_scp_statements(self) -> dict:
        import pandas as pd

        scp = pd.read_csv(self.ptbxl_path / "scp_statements.csv", index_col=0)
        scp = scp[scp.diagnostic == 1]
        return dict(zip(scp.index, scp.diagnostic_class))

    def _scp_to_superclass(self, scp_codes: dict, superclass_map: dict) -> np.ndarray:
        classes = ["NORM", "MI", "STTC", "CD", "HYP"]
        vec = np.zeros(5, dtype=np.float32)
        for code, confidence in scp_codes.items():
            if code in superclass_map and confidence >= 50:
                sc = superclass_map[code]
                if sc in classes:
                    vec[classes.index(sc)] = 1.0
        return vec


class CODE15Loader(DatasetLoader):
    """Loader for CODE-15% dataset."""

    def __init__(self, data_root: Path):
        super().__init__("D3", data_root)
        self.code_path = data_root / "code-15"

    def load(self) -> dict:
        import h5py

        classes = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]
        with h5py.File(self.code_path / "exams.hdf5", "r") as f:
            tracings = np.array(f["tracings"])  # (N, 4096, 12) at 400 Hz

        import pandas as pd

        exam_df = pd.read_csv(self.code_path / "exams.csv")

        signals = []
        for trace in tracings:
            resampled = resample_ecg(trace, orig_hz=400, target_hz=SAMPLING_RATE_HZ)
            resampled = pad_or_truncate(resampled, SIGNAL_LENGTH)
            signals.append(resampled)

        labels = exam_df[classes].values.astype(np.float32)
        patient_ids = exam_df["patient_id"].values

        return {
            "signals": np.array(signals),
            "patient_ids": patient_ids,
            "labels": labels,
            "ecg_ids": exam_df["exam_id"].values,
            "class_names": classes,
            "n_classes": 6,
            "task_id": "code15_ribeiro",
        }


class CPSC2018Loader(DatasetLoader):
    """Loader for CPSC2018 dataset."""

    def __init__(self, data_root: Path):
        super().__init__("D4", data_root)
        self.cpsc_path = data_root / "cpsc2018"

    def load(self) -> dict:
        import pandas as pd

        classes = ["Normal", "AF", "IAVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
        ref = pd.read_csv(self.cpsc_path / "REFERENCE.csv", header=None, names=["id", "label"])

        signals = []
        labels_list = []
        ecg_ids = []

        for _, row in ref.iterrows():
            rec_path = self.cpsc_path / "TrainingSet" / row.id
            signal, fields = wfdb.rdsamp(str(rec_path))
            orig_hz = fields["fs"]

            if orig_hz != SAMPLING_RATE_HZ:
                signal = resample_ecg(signal, orig_hz, SAMPLING_RATE_HZ)
            signal = pad_or_truncate(signal, SIGNAL_LENGTH)

            label_vec = np.zeros(9, dtype=np.float32)
            for lbl in str(row.label).split(","):
                lbl = lbl.strip()
                if lbl in classes:
                    label_vec[classes.index(lbl)] = 1.0

            signals.append(signal)
            labels_list.append(label_vec)
            ecg_ids.append(row.id)

        return {
            "signals": np.array(signals),
            "patient_ids": np.arange(len(signals)),  # 1 ECG per patient
            "labels": np.array(labels_list),
            "ecg_ids": np.array(ecg_ids),
            "class_names": classes,
            "n_classes": 9,
            "task_id": "cpsc_9class",
        }


class MIMICECGLoader(DatasetLoader):
    """Loader for MIMIC-IV-ECG (requires credentialed access)."""

    def __init__(self, data_root: Path):
        super().__init__("D2", data_root)
        self.mimic_path = data_root / "mimic-iv-ecg"

    def load(self) -> dict:
        if not self.mimic_path.exists():
            raise FileNotFoundError(
                f"MIMIC-IV-ECG data not found at {self.mimic_path}. "
                "This dataset requires PhysioNet credentialed access. "
                "Skipping — study proceeds with 3 datasets."
            )
        raise NotImplementedError(
            "MIMIC-IV-ECG loader requires BenchECG task definitions. "
            "Implement once task_ids and class labels are confirmed."
        )


def get_loader(dataset_id: str, data_root: Path) -> DatasetLoader:
    loaders = {
        "D1": PTBXLLoader,
        "D2": MIMICECGLoader,
        "D3": CODE15Loader,
        "D4": CPSC2018Loader,
    }
    return loaders[dataset_id](data_root)


def prepare_dataset(
    dataset_id: str,
    data_root: Path,
    output_dir: Optional[Path] = None,
) -> dict:
    """Full pipeline: load, preprocess, split, normalize, save."""
    loader = get_loader(dataset_id, data_root)
    data = loader.load()

    split_patients = patient_stratified_split(
        data["patient_ids"],
        data["labels"],
        train_frac=SPLIT_FRACTIONS["train"],
        cal_frac=SPLIT_FRACTIONS["cal"],
        test_frac=SPLIT_FRACTIONS["test"],
    )
    verify_no_leakage(split_patients)
    split_indices = build_split_indices(data["patient_ids"], split_patients)

    signals = data["signals"].transpose(0, 2, 1)  # (N, 12, 5000) for models

    norm_stats = compute_normalization_stats(signals[split_indices["train"]])
    signals = normalize_signals(signals, norm_stats)

    result = {
        "dataset_id": dataset_id,
        "signals": signals,
        "labels": data["labels"],
        "patient_ids": data["patient_ids"],
        "ecg_ids": data.get("ecg_ids"),
        "class_names": data["class_names"],
        "n_classes": data["n_classes"],
        "task_id": data["task_id"],
        "split_indices": split_indices,
        "norm_stats": norm_stats,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / f"{dataset_id}_processed.npz",
            signals=signals,
            labels=data["labels"],
            patient_ids=data["patient_ids"],
        )
        with open(output_dir / f"{dataset_id}_split.json", "w") as f:
            json.dump(
                {k: v.tolist() for k, v in split_indices.items()},
                f,
            )
        with open(output_dir / f"{dataset_id}_meta.json", "w") as f:
            json.dump(
                {
                    "class_names": data["class_names"],
                    "n_classes": data["n_classes"],
                    "task_id": data["task_id"],
                    "n_train": len(split_indices["train"]),
                    "n_cal": len(split_indices["cal"]),
                    "n_test": len(split_indices["test"]),
                    "norm_mean": norm_stats["mean"].tolist(),
                    "norm_std": norm_stats["std"].tolist(),
                },
                f,
                indent=2,
            )
        logger.info(
            f"Saved {dataset_id}: train={len(split_indices['train'])}, "
            f"cal={len(split_indices['cal'])}, test={len(split_indices['test'])}"
        )

    return result
