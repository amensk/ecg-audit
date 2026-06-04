"""Preflight gate for the real three-open-dataset experiment.

This script implements the Stage 05 refinement decision: try the open
datasets first (PTB-XL, CODE-15%, CPSC2018) and keep MIMIC-IV-ECG out of
scope until credentialed access and task definitions are verified.

It does not download data or fabricate results. It checks whether the
current workspace can launch the real pipeline and writes a machine-readable
blocker report for downstream analysis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


OPEN_DATASETS = {
    "D1": {
        "name": "PTB-XL",
        "candidate_dirs": ["ptb-xl", "ptbxl", "PTB-XL"],
        "required_any": ["ptbxl_database.csv"],
    },
    "D3": {
        "name": "CODE-15%",
        "candidate_dirs": ["code15", "code-15", "CODE-15", "CODE15"],
        "required_any": ["exams.csv", "annotations.csv"],
    },
    "D4": {
        "name": "CPSC2018",
        "candidate_dirs": ["cpsc2018", "CPSC2018", "cpsc"],
        "required_any": ["REFERENCE.csv", "REFERENCE-v3.csv"],
    },
}

CHECKPOINTS = {
    "M1": {
        "name": "MERL",
        "path": Path("merl") / "merl_checkpoint.pth",
        "python_packages": ["torch", "torchvision"],
    },
    "M2": {
        "name": "ST-MEM",
        "path": Path("st_mem") / "st_mem_checkpoint.pth",
        "python_packages": ["torch", "st_mem"],
    },
    "M3": {
        "name": "HuBERT-ECG",
        "path": Path("hubert_ecg") / "hubert_ecg_checkpoint.pth",
        "python_packages": ["torch", "hubert_ecg"],
    },
    "M4": {
        "name": "ECGFM-KED",
        "path": Path("ecgfm_ked") / "ecgfm_ked_checkpoint.pth",
        "python_packages": ["torch", "ecgfm_ked"],
    },
    "M5": {
        "name": "ECG-FM",
        "path": Path("ecg_fm") / "ecg_fm_checkpoint.pth",
        "python_packages": ["torch", "ecg_fm"],
    },
}

REQUIRED_BASE_PACKAGES = ["numpy", "scipy", "sklearn", "pandas", "matplotlib", "wfdb", "h5py"]


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def load_checkpoint_mapping(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Checkpoint mapping must be a JSON object: {path}")
    checkpoints = data.get("checkpoints", data)
    if not isinstance(checkpoints, dict):
        raise ValueError(f"Checkpoint mapping 'checkpoints' must be an object: {path}")
    return checkpoints


def check_dataset(data_root: Path, spec: dict) -> dict:
    candidate_paths = [data_root / d for d in spec["candidate_dirs"]]
    existing_dirs = [p for p in candidate_paths if p.exists()]
    found_required = []
    for directory in existing_dirs:
        for filename in spec["required_any"]:
            matches = list(directory.rglob(filename))
            found_required.extend(str(m) for m in matches[:5])
    return {
        "name": spec["name"],
        "candidate_paths": [str(p) for p in candidate_paths],
        "existing_candidate_paths": [str(p) for p in existing_dirs],
        "required_any": spec["required_any"],
        "found_required_files": found_required,
        "available": bool(existing_dirs and found_required),
    }


def check_checkpoint(checkpoints_root: Path, spec: dict, mapping_entry: dict, mapping_path: Path) -> dict:
    mapped_relative_path = mapping_entry.get("relative_path")
    relative_path = Path(mapped_relative_path) if mapped_relative_path else spec["path"]
    path = checkpoints_root / relative_path
    package_status = {pkg: has_package(pkg) for pkg in spec["python_packages"]}
    return {
        "name": spec["name"],
        "mapping_path": str(mapping_path) if mapping_path.exists() else None,
        "mapping_entry": mapping_entry,
        "relative_path": str(relative_path),
        "expected_path": str(path),
        "checkpoint_exists": path.exists(),
        "python_packages": package_status,
        "available": path.exists() and all(package_status.values()),
    }


def build_launch_command(data_root: Path, checkpoints_root: Path, model_ids: list[str] | None = None) -> list[str]:
    command = [
        sys.executable,
        "run_all.py",
        "--data-root",
        str(data_root),
        "--checkpoints-root",
        str(checkpoints_root),
        "--datasets",
        "D1",
        "D3",
        "D4",
        "--skip-mimic",
        "--phase",
        "all",
    ]
    if model_ids:
        command.extend(["--models", *model_ids])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight real open-dataset ECG experiment")
    parser.add_argument("--data-root", type=Path, default=Path("workspace/raw_data"))
    parser.add_argument("--checkpoints-root", type=Path, default=Path("workspace/checkpoints"))
    parser.add_argument("--checkpoint-mapping", type=Path, default=Path("workspace/code/checkpoint_mapping.json"),
                        help="Optional JSON mapping from model IDs to checkpoint paths under --checkpoints-root")
    parser.add_argument("--output", type=Path, default=Path("workspace/results/open_dataset_real_run_preflight.json"))
    parser.add_argument("--log", type=Path, default=Path("workspace/notes/open_dataset_real_run_preflight.log"))
    parser.add_argument("--attempt-launch", action="store_true",
                        help="Launch run_all.py only if all preflight checks pass")
    parser.add_argument("--min-ready-datasets", type=int, default=1,
                        help="Minimum number of open datasets that must be available to launch")
    parser.add_argument("--min-ready-checkpoints", type=int, default=1,
                        help="Minimum number of FM checkpoints/packages that must be available to launch")
    parser.add_argument("--require-all-datasets", action="store_true",
                        help="Require all configured open datasets to be available")
    parser.add_argument("--require-all-checkpoints", action="store_true",
                        help="Require all configured FM checkpoints/packages to be available")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_root = args.data_root if args.data_root.is_absolute() else repo_root / args.data_root
    checkpoints_root = args.checkpoints_root if args.checkpoints_root.is_absolute() else repo_root / args.checkpoints_root
    checkpoint_mapping_path = (
        args.checkpoint_mapping
        if args.checkpoint_mapping.is_absolute()
        else repo_root / args.checkpoint_mapping
    )
    output = args.output if args.output.is_absolute() else repo_root / args.output
    log_path = args.log if args.log.is_absolute() else repo_root / args.log
    checkpoint_mapping = load_checkpoint_mapping(checkpoint_mapping_path)

    base_packages = {pkg: has_package(pkg) for pkg in REQUIRED_BASE_PACKAGES}
    datasets = {dataset_id: check_dataset(data_root, spec) for dataset_id, spec in OPEN_DATASETS.items()}
    checkpoints = {
        model_id: check_checkpoint(
            checkpoints_root,
            spec,
            checkpoint_mapping.get(model_id, {}),
            checkpoint_mapping_path,
        )
        for model_id, spec in CHECKPOINTS.items()
    }
    torch_available = has_package("torch")

    available_dataset_ids = [k for k, item in datasets.items() if item["available"]]
    available_checkpoint_ids = [k for k, item in checkpoints.items() if item["available"]]
    launch_model_ids = sorted(set(["M6", *available_checkpoint_ids]))
    launch_command = build_launch_command(data_root, checkpoints_root, launch_model_ids)

    datasets_ready = (
        all(item["available"] for item in datasets.values())
        if args.require_all_datasets
        else len(available_dataset_ids) >= max(1, args.min_ready_datasets)
    )
    checkpoints_ready = (
        all(item["available"] for item in checkpoints.values())
        if args.require_all_checkpoints
        else len(available_checkpoint_ids) >= max(1, args.min_ready_checkpoints)
    )

    ready_to_launch = (
        all(base_packages.values())
        and torch_available
        and datasets_ready
        and checkpoints_ready
    )

    launch = {
        "attempt_requested": args.attempt_launch,
        "ready_to_launch": ready_to_launch,
        "command": launch_command,
        "executed": False,
        "return_code": None,
        "stdout_tail": "",
        "stderr_tail": "",
    }

    if args.attempt_launch and ready_to_launch:
        proc = subprocess.run(
            launch_command,
            cwd=repo_root / "workspace" / "code",
            text=True,
            capture_output=True,
            check=False,
        )
        launch.update({
            "executed": True,
            "return_code": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        })

    blockers = []
    non_blocking_gaps = []
    for pkg, ok in base_packages.items():
        if not ok:
            blockers.append(f"missing_base_package:{pkg}")
    if not torch_available:
        blockers.append("missing_base_package:torch")
    missing_datasets = [
        f"missing_open_dataset:{dataset_id}:{status['name']}"
        for dataset_id, status in datasets.items()
        if not status["available"]
    ]
    missing_checkpoints = [
        f"missing_or_unloadable_checkpoint:{model_id}:{status['name']}"
        for model_id, status in checkpoints.items()
        if not status["available"]
    ]

    if args.require_all_datasets:
        blockers.extend(missing_datasets)
    elif len(available_dataset_ids) < max(1, args.min_ready_datasets):
        blockers.append(
            f"insufficient_open_datasets:required={max(1, args.min_ready_datasets)}:available={len(available_dataset_ids)}"
        )
        blockers.extend(missing_datasets)
    else:
        non_blocking_gaps.extend(missing_datasets)

    if args.require_all_checkpoints:
        blockers.extend(missing_checkpoints)
    elif len(available_checkpoint_ids) < max(1, args.min_ready_checkpoints):
        blockers.append(
            f"insufficient_checkpoints:required={max(1, args.min_ready_checkpoints)}:available={len(available_checkpoint_ids)}"
        )
        blockers.extend(missing_checkpoints)
    else:
        non_blocking_gaps.extend(missing_checkpoints)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "05_experimentation",
        "experiment": "real_three_open_dataset_preflight",
        "policy": "Run D1/PTB-XL, D3/CODE-15%, and D4/CPSC2018 first; defer D2/MIMIC-IV-ECG until PhysioNet access and task definitions are verified.",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "repo_root": str(repo_root),
            "data_root": str(data_root),
            "checkpoints_root": str(checkpoints_root),
            "checkpoint_mapping_path": str(checkpoint_mapping_path),
            "checkpoint_mapping_loaded": checkpoint_mapping_path.exists(),
        },
        "base_packages": base_packages,
        "datasets": datasets,
        "available_dataset_ids": available_dataset_ids,
        "available_dataset_count": len(available_dataset_ids),
        "checkpoints": checkpoints,
        "available_checkpoint_ids": available_checkpoint_ids,
        "available_checkpoint_count": len(available_checkpoint_ids),
        "launch_policy": {
            "require_all_datasets": args.require_all_datasets,
            "require_all_checkpoints": args.require_all_checkpoints,
            "min_ready_datasets": max(1, args.min_ready_datasets),
            "min_ready_checkpoints": max(1, args.min_ready_checkpoints),
        },
        "mimic_status": {
            "dataset_id": "D2",
            "name": "MIMIC-IV-ECG",
            "included_in_this_attempt": False,
            "reason": "Deferred by refinement request until credentialed access and BenchECG task definitions are verified.",
        },
        "ready_to_launch": ready_to_launch,
        "blockers": blockers,
        "non_blocking_gaps": non_blocking_gaps,
        "launch": launch,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"generated_at: {report['generated_at']}",
        f"experiment: {report['experiment']}",
        f"ready_to_launch: {ready_to_launch}",
        "blockers:",
        *[f"- {b}" for b in blockers],
        "launch_command:",
        " ".join(launch_command),
    ]
    log_path.write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "output": str(output),
        "log": str(log_path),
        "ready_to_launch": ready_to_launch,
        "blocker_count": len(blockers),
        "blockers": blockers,
    }, indent=2))

    return 0 if ready_to_launch else 2


if __name__ == "__main__":
    raise SystemExit(main())
