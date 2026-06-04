"""Minimal real-data pilot for Stage 05 refinement.

This script intentionally runs a small, auditable real-data slice before the
full benchmark: PTB-XL + CODE-15%, S4-lite baseline, ST-MEM linear probe,
Phase A metrics, and one Platt recalibration pass.

It is not a publication-scale experiment. It exists to isolate data/checkpoint
readiness and to produce real-data pilot metrics without fabricating FM results.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import evaluate_model_on_dataset  # noqa: E402
from s4_baseline import S4DModel  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "workspace"
RESULTS = WORKSPACE / "results"
NOTES = WORKSPACE / "notes"
RAW = WORKSPACE / "raw_data"
CHECKPOINTS = WORKSPACE / "checkpoints"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PilotDataset:
    dataset_id: str
    dataset_name: str
    signals: np.ndarray  # (N, 12, L)
    labels: np.ndarray
    patient_ids: np.ndarray
    class_names: list[str]
    source_note: str


def _setup_logging() -> logging.Logger:
    NOTES.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("minimal_real_pilot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(NOTES / "minimal_real_pilot.log", mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


STMEM_SIGNAL_LENGTH = 2250


def _pad_or_crop(x: np.ndarray, length: int = STMEM_SIGNAL_LENGTH) -> np.ndarray:
    if x.shape[0] >= length:
        return x[:length]
    out = np.zeros((length, x.shape[1]), dtype=x.dtype)
    out[: x.shape[0]] = x
    return out


def _standardize(signals: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    mean = signals[train_idx].mean(axis=(0, 2), keepdims=True)
    std = signals[train_idx].std(axis=(0, 2), keepdims=True) + 1e-8
    return (signals - mean) / std


def _multilabel_split(patient_ids: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    unique_patients = np.unique(patient_ids)
    patient_labels = []
    for pid in unique_patients:
        first_idx = np.flatnonzero(patient_ids == pid)[0]
        patient_labels.append(labels[first_idx])
    patient_labels = np.asarray(patient_labels)
    label_keys = np.array(["".join(map(str, row.astype(int))) for row in patient_labels])
    unique, counts = np.unique(label_keys, return_counts=True)
    stratify = label_keys if np.all(counts >= 2) and len(unique) > 1 else None
    p_train_cal, p_test = train_test_split(
        unique_patients, test_size=0.2, random_state=SEED, stratify=stratify
    )
    tc_mask = np.isin(unique_patients, p_train_cal)
    label_keys_tc = label_keys[tc_mask]
    unique_tc, counts_tc = np.unique(label_keys_tc, return_counts=True)
    stratify_tc = label_keys_tc if np.all(counts_tc >= 2) and len(unique_tc) > 1 else None
    p_train, p_cal = train_test_split(
        p_train_cal, test_size=0.25, random_state=SEED, stratify=stratify_tc
    )

    train = np.flatnonzero(np.isin(patient_ids, p_train))
    cal = np.flatnonzero(np.isin(patient_ids, p_cal))
    test = np.flatnonzero(np.isin(patient_ids, p_test))
    split_patients = {
        name: set(patient_ids[idx].tolist())
        for name, idx in [("train", train), ("cal", cal), ("test", test)]
    }
    leakage = {
        "train_cal": len(split_patients["train"] & split_patients["cal"]),
        "train_test": len(split_patients["train"] & split_patients["test"]),
        "cal_test": len(split_patients["cal"] & split_patients["test"]),
    }
    return {"train": train, "cal": cal, "test": test, "leakage": leakage}


def _load_ptbxl(max_records: int, logger: logging.Logger) -> PilotDataset:
    meta_path = RAW / "ptb-xl" / "ptbxl_database.csv"
    nested = RAW / "ptb-xl" / "physionet.org" / "files" / "ptb-xl" / "1.0.3"
    scp_path = nested / "scp_statements.csv"
    meta = pd.read_csv(meta_path)
    scp = pd.read_csv(scp_path, index_col=0)
    superclass = scp[scp.diagnostic == 1]["diagnostic_class"].to_dict()
    classes = ["NORM", "MI", "STTC", "CD", "HYP"]

    import wfdb

    signals, labels, patient_ids = [], [], []
    for _, row in meta.iterrows():
        signal_rel = row.filename_hr
        record = nested / signal_rel
        if not (record.with_suffix(".dat").exists() and record.with_suffix(".hea").exists()):
            continue
        sig, _ = wfdb.rdsamp(str(record))
        sig = _pad_or_crop(sig.astype(np.float32), STMEM_SIGNAL_LENGTH)
        codes = eval(row.scp_codes)
        y = np.zeros(len(classes), dtype=np.float32)
        for code, confidence in codes.items():
            if confidence >= 50 and code in superclass and superclass[code] in classes:
                y[classes.index(superclass[code])] = 1.0
        if y.sum() == 0:
            continue
        signals.append(sig.T)
        labels.append(y)
        patient_ids.append(row.patient_id)
        if len(signals) >= max_records:
            break

    logger.info("Loaded PTB-XL pilot records: %d", len(signals))
    return PilotDataset(
        dataset_id="D1",
        dataset_name="PTB-XL",
        signals=np.stack(signals),
        labels=np.stack(labels),
        patient_ids=np.array(patient_ids),
        class_names=classes,
        source_note="PTB-XL high-rate records from nested PhysioNet layout; first available real records.",
    )


def _load_code15(max_records: int, logger: logging.Logger) -> PilotDataset:
    code_root = RAW / "code15"
    meta = pd.read_csv(code_root / "exams.csv")
    classes = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]

    extracted_parts = {p.name for p in code_root.glob("exams_part*.hdf5")}
    if not extracted_parts:
        zip_parts = []
        for zip_path in sorted(code_root.glob("exams_part*.zip")):
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    names = zf.namelist()
                zip_parts.append({
                    "zip": zip_path.name,
                    "members": names[:3],
                    "size_bytes": zip_path.stat().st_size,
                })
            except Exception as exc:
                zip_parts.append({
                    "zip": zip_path.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "size_bytes": zip_path.stat().st_size,
                })
        raise FileNotFoundError(
            "CODE-15% pilot needs at least one extracted exams_part*.hdf5 file. "
            f"Found {len(zip_parts)} zip archives but no extracted HDF5 parts; "
            "the pilot avoids streaming multi-GB HDF5 files from zip archives."
        )

    usable_parts = sorted(extracted_parts)
    meta = meta[meta.trace_file.isin(usable_parts)].copy()
    # Prefer rows with at least one positive label, then add negatives if available.
    meta["n_positive"] = meta[classes].sum(axis=1)
    selected = pd.concat([
        meta[meta.n_positive > 0].head(max_records // 2),
        meta[meta.n_positive == 0].head(max_records // 2),
    ]).head(max_records)

    signals, labels, patient_ids = [], [], []
    grouped = selected.groupby("trace_file")
    for trace_file, group in grouped:
        h5_path = code_root / trace_file
        with h5py.File(h5_path, "r") as h5:
            exam_ids = h5["exam_id"][:]
            id_to_pos = {int(exam_id): i for i, exam_id in enumerate(exam_ids)}
            for _, row in group.iterrows():
                pos = id_to_pos.get(int(row.exam_id))
                if pos is None:
                    continue
                sig = h5["tracings"][pos].astype(np.float32)
                sig = resample_poly(sig, 250, 400, axis=0)
                sig = _pad_or_crop(sig, STMEM_SIGNAL_LENGTH)
                signals.append(sig.T)
                labels.append(row[classes].astype(float).values.astype(np.float32))
                patient_ids.append(row.patient_id)

    logger.info("Loaded CODE-15%% pilot records: %d", len(signals))
    return PilotDataset(
        dataset_id="D3",
        dataset_name="CODE-15%",
        signals=np.stack(signals),
        labels=np.stack(labels),
        patient_ids=np.array(patient_ids),
        class_names=classes,
        source_note="CODE-15% records loaded from extracted HDF5 parts staged under workspace/raw_data/code15.",
    )


def _train_s4_lite(ds: PilotDataset, split: dict[str, np.ndarray], logger: logging.Logger) -> dict:
    torch.manual_seed(SEED)
    n_classes = ds.labels.shape[1]
    model = S4DModel(
        n_leads=12,
        d_model=16,
        d_state=8,
        n_layers=1,
        n_classes=n_classes,
        dropout=0.1,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    x_train = torch.FloatTensor(ds.signals[split["train"]]).to(DEVICE)
    y_train = torch.FloatTensor(ds.labels[split["train"]]).to(DEVICE)
    batch_size = 16
    model.train()
    for epoch in range(3):
        perm = torch.randperm(x_train.shape[0], device=DEVICE)
        losses = []
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            logits = model(x_train[idx])
            loss = loss_fn(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        logger.info("%s S4-lite epoch %d loss %.4f", ds.dataset_id, epoch + 1, float(np.mean(losses)))

    def predict(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        outs = []
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                x = torch.FloatTensor(ds.signals[indices[start : start + batch_size]]).to(DEVICE)
                outs.append(model(x).cpu().numpy())
        logits = np.concatenate(outs)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return logits, probs

    cal_logits, _ = predict(split["cal"])
    test_logits, test_probs = predict(split["test"])
    raw_metrics = evaluate_model_on_dataset(
        "M6_s4_lite", ds.dataset_id, test_probs, ds.labels[split["test"]], ds.class_names
    )
    return {
        "model": model,
        "cal_logits": cal_logits,
        "test_logits": test_logits,
        "test_probs": test_probs,
        "raw_metrics": raw_metrics,
    }


def _fit_safe_platt(cal_logits: np.ndarray, cal_labels: np.ndarray, test_logits: np.ndarray) -> dict:
    calibrated = np.zeros_like(test_logits, dtype=np.float64)
    params = []
    for c in range(cal_logits.shape[1]):
        y = cal_labels[:, c]
        if len(np.unique(y)) < 2:
            p = float(y.mean())
            calibrated[:, c] = p
            params.append({"class_index": c, "status": "constant_calibration_label", "probability": p})
            continue
        lr = LogisticRegression(solver="lbfgs", max_iter=1000, C=1e3)
        lr.fit(cal_logits[:, c : c + 1], y)
        calibrated[:, c] = lr.predict_proba(test_logits[:, c : c + 1])[:, 1]
        params.append({
            "class_index": c,
            "status": "fit",
            "coef": lr.coef_.tolist(),
            "intercept": lr.intercept_.tolist(),
        })
    return {"probs": calibrated, "params": params}


def _load_stmem_model(logger: logging.Logger) -> tuple[dict, nn.Module | None]:
    checkpoint_path = CHECKPOINTS / "st_mem" / "st_mem_vit_base_encoder.pth"
    status = {
        "model_id": "M2",
        "model_name": "ST-MEM",
        "checkpoint_path": str(checkpoint_path.relative_to(WORKSPACE)),
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_verified": False,
        "feature_extraction_executed": False,
        "blocker": None,
    }
    if not checkpoint_path.exists():
        status["blocker"] = "checkpoint_missing"
        return status, None
    try:
        obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = obj.get("model", obj)
        status.update({
            "checkpoint_verified": isinstance(state, dict) and len(state) > 0,
            "state_key_count": len(state) if isinstance(state, dict) else None,
            "checkpoint_top_level_keys": list(obj.keys()) if isinstance(obj, dict) else [],
        })
    except Exception as exc:
        status["blocker"] = f"checkpoint_load_failed:{type(exc).__name__}:{exc}"
        return status, None

    try:
        import einops  # noqa: F401
        status["source_dependency_einops_available"] = True
    except Exception as exc:
        status["source_dependency_einops_available"] = False
        status["blocker"] = f"st_mem_source_dependency_missing:{type(exc).__name__}:einops"
        logger.warning("ST-MEM checkpoint verified but feature extraction blocked: %s", status["blocker"])
        return status, None

    try:
        source_root = CHECKPOINTS / "_sources" / "ST-MEM"
        sys.path.insert(0, str(source_root))
        from models.encoder.st_mem_vit import st_mem_vit_base

        model = st_mem_vit_base(num_leads=12, seq_len=STMEM_SIGNAL_LENGTH, patch_size=75)
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.eval().to(DEVICE)
        with torch.no_grad():
            features = model(torch.zeros(1, 12, STMEM_SIGNAL_LENGTH, device=DEVICE))
        status.update({
            "feature_extraction_executed": True,
            "checkpoint_verified": True,
            "source_root": str(source_root.relative_to(WORKSPACE)),
            "missing_key_count": len(missing),
            "unexpected_key_count": len(unexpected),
            "feature_shape": list(features.shape),
            "blocker": None,
        })
        return status, model
    except Exception as exc:
        status["blocker"] = f"st_mem_feature_extraction_failed:{type(exc).__name__}:{exc}"
        logger.warning("ST-MEM feature extraction failed: %s", status["blocker"])
        return status, None


def _extract_stmem_features(model: nn.Module, signals: np.ndarray, logger: logging.Logger) -> np.ndarray:
    model.eval()
    feats = []
    batch_size = 4
    with torch.no_grad():
        for start in range(0, signals.shape[0], batch_size):
            x = torch.FloatTensor(signals[start:start + batch_size]).to(DEVICE)
            y = model(x).detach().cpu().numpy()
            feats.append(y)
    features = np.concatenate(feats, axis=0)
    logger.info("Extracted ST-MEM features: %s", features.shape)
    return features


def _fit_feature_logistic_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    cal_features: np.ndarray,
    test_features: np.ndarray,
) -> dict:
    cal_logits = np.zeros((cal_features.shape[0], train_labels.shape[1]), dtype=np.float64)
    test_logits = np.zeros((test_features.shape[0], train_labels.shape[1]), dtype=np.float64)
    probe_params = []
    for c in range(train_labels.shape[1]):
        y = train_labels[:, c]
        if len(np.unique(y)) < 2:
            p = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
            logit = float(np.log(p / (1.0 - p)))
            cal_logits[:, c] = logit
            test_logits[:, c] = logit
            probe_params.append({"class_index": c, "status": "constant_train_label", "probability": p})
            continue
        lr = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, class_weight="balanced")
        lr.fit(train_features, y)
        cal_logits[:, c] = lr.decision_function(cal_features)
        test_logits[:, c] = lr.decision_function(test_features)
        probe_params.append({
            "class_index": c,
            "status": "fit",
            "coef_shape": list(lr.coef_.shape),
            "intercept": lr.intercept_.tolist(),
        })
    cal_probs = 1.0 / (1.0 + np.exp(-cal_logits))
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    return {
        "cal_logits": cal_logits,
        "cal_probs": cal_probs,
        "test_logits": test_logits,
        "test_probs": test_probs,
        "probe_params": probe_params,
    }


def _evaluate_stmem_probe(
    stmem_model: nn.Module,
    ds: PilotDataset,
    split: dict[str, np.ndarray],
    logger: logging.Logger,
) -> dict:
    features = _extract_stmem_features(stmem_model, ds.signals, logger)
    probe = _fit_feature_logistic_probe(
        features[split["train"]],
        ds.labels[split["train"]],
        features[split["cal"]],
        features[split["test"]],
    )
    raw_metrics = evaluate_model_on_dataset(
        "M2_stmem_probe",
        ds.dataset_id,
        probe["test_probs"],
        ds.labels[split["test"]],
        ds.class_names,
    )
    return {
        **probe,
        "feature_shape": list(features.shape),
        "raw_metrics": raw_metrics,
    }


def main() -> int:
    logger = _setup_logging()
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    from datetime import datetime

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "05_experimentation",
        "experiment": "minimal_real_data_pilot",
        "scope": "PTB-XL + CODE-15%, S4-lite baseline, ST-MEM linear probe, Platt recalibration",
        "publication_grade": False,
        "limitations": [
            "Small real-data pilot, not full benchmark.",
            "S4-lite uses reduced dimensions and 3 epochs for feasibility.",
            "ST-MEM uses a logistic linear probe on frozen features for feasibility.",
            "Both PTB-XL and CODE-15% are cropped/padded to the ST-MEM checkpoint's 2250-sample input length.",
        ],
        "datasets": {},
        "phase_a_metrics": [],
        "recalibration_metrics": [],
    }
    stmem_status, stmem_model = _load_stmem_model(logger)
    out["fm_checkpoint"] = stmem_status

    for loader in (_load_ptbxl, _load_code15):
        try:
            ds = loader(120, logger)
            split = _multilabel_split(ds.patient_ids, ds.labels)
            ds.signals = _standardize(ds.signals, split["train"])

            out["datasets"][ds.dataset_id] = {
                "dataset_name": ds.dataset_name,
                "n_loaded": int(len(ds.labels)),
                "n_train": int(len(split["train"])),
                "n_cal": int(len(split["cal"])),
                "n_test": int(len(split["test"])),
                "n_classes": int(ds.labels.shape[1]),
                "class_names": ds.class_names,
                "label_prevalence": {
                    name: float(ds.labels[:, i].mean()) for i, name in enumerate(ds.class_names)
                },
                "leakage_overlap_counts": split["leakage"],
                "source_note": ds.source_note,
            }

            model_results = []
            s4_result = _train_s4_lite(ds, split, logger)
            model_results.append({
                "model_id": "M6_s4_lite",
                "model_name": "S4-lite real pilot",
                "result": s4_result,
                "extra": {},
            })

            if stmem_model is not None:
                stmem_result = _evaluate_stmem_probe(stmem_model, ds, split, logger)
                model_results.append({
                    "model_id": "M2_stmem_probe",
                    "model_name": "ST-MEM frozen linear probe real pilot",
                    "result": stmem_result,
                    "extra": {
                        "feature_shape": stmem_result["feature_shape"],
                        "probe_params": stmem_result["probe_params"],
                    },
                })

            for model_result in model_results:
                result = model_result["result"]
                out["phase_a_metrics"].append({
                    "dataset_id": ds.dataset_id,
                    "dataset_name": ds.dataset_name,
                    "model_id": model_result["model_id"],
                    "model_name": model_result["model_name"],
                    "metrics": result["raw_metrics"]["macro"],
                    "per_class": result["raw_metrics"]["per_class"],
                    **model_result["extra"],
                })
                platt = _fit_safe_platt(
                    result["cal_logits"], ds.labels[split["cal"]], result["test_logits"]
                )
                platt_metrics = evaluate_model_on_dataset(
                    f"{model_result['model_id']}_platt",
                    ds.dataset_id,
                    platt["probs"],
                    ds.labels[split["test"]],
                    ds.class_names,
                )
                out["recalibration_metrics"].append({
                    "dataset_id": ds.dataset_id,
                    "dataset_name": ds.dataset_name,
                    "model_id": model_result["model_id"],
                    "model_name": model_result["model_name"],
                    "recal_method": "platt_scaling",
                    "metrics": platt_metrics["macro"],
                    "per_class": platt_metrics["per_class"],
                    "platt_params": platt["params"],
                })
        except Exception as exc:
            logger.exception("Pilot failed for loader %s", loader.__name__)
            out["datasets"][loader.__name__] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    failed_datasets = [
        key for key, value in out["datasets"].items()
        if isinstance(value, dict) and value.get("status") == "failed"
    ]
    if out["fm_checkpoint"].get("blocker") and failed_datasets:
        status = "completed_partial_with_fm_and_dataset_blockers"
    elif out["fm_checkpoint"].get("blocker"):
        status = "completed_partial_with_fm_blocker"
    elif failed_datasets:
        status = "completed_partial_with_dataset_blocker"
    else:
        status = "completed"
    out["status"] = status
    out["failed_datasets"] = failed_datasets
    out_path = RESULTS / "minimal_real_pilot_results.json"
    out_path.write_text(json.dumps(out, indent=2))

    csv_path = RESULTS / "minimal_real_pilot_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "dataset_id",
                "dataset_name",
                "model_id",
                "model_name",
                "recal_method",
                "auroc",
                "auprc",
                "ece_ew",
                "ece_em",
                "brier",
                "sce",
            ],
        )
        writer.writeheader()
        for row in out["phase_a_metrics"]:
            m = row["metrics"]
            writer.writerow({
                "phase": "A_real_pilot",
                "dataset_id": row["dataset_id"],
                "dataset_name": row["dataset_name"],
                "model_id": row["model_id"],
                "model_name": row["model_name"],
                "recal_method": "none",
                "auroc": m.get("auroc"),
                "auprc": m.get("auprc"),
                "ece_ew": m.get("ece_equal_width"),
                "ece_em": m.get("ece_equal_mass"),
                "brier": m.get("brier"),
                "sce": m.get("sce"),
            })
        for row in out["recalibration_metrics"]:
            m = row["metrics"]
            writer.writerow({
                "phase": "B_real_pilot",
                "dataset_id": row["dataset_id"],
                "dataset_name": row["dataset_name"],
                "model_id": row["model_id"],
                "model_name": row["model_name"],
                "recal_method": row["recal_method"],
                "auroc": m.get("auroc"),
                "auprc": m.get("auprc"),
                "ece_ew": m.get("ece_equal_width"),
                "ece_em": m.get("ece_equal_mass"),
                "brier": m.get("brier"),
                "sce": m.get("sce"),
            })

    logger.info("Wrote %s and %s", out_path, csv_path)
    print(json.dumps({
        "status": out["status"],
        "results": str(out_path),
        "summary": str(csv_path),
        "datasets": {
            k: v for k, v in out["datasets"].items()
        },
        "fm_checkpoint": out["fm_checkpoint"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
