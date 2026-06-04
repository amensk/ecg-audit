"""Stage available assets for the real open-dataset ECG experiment.

The script creates the canonical asset layout expected by
preflight_open_dataset_real_run.py and records whether required files are
actually present. It does not download datasets or create placeholder data.
If source assets are supplied under an incoming directory, it copies only the
recognized files into the canonical locations.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


DATASET_TARGETS = {
    "D1": {
        "name": "PTB-XL",
        "target_dir": Path("raw_data") / "ptb-xl",
        "required_any": ["ptbxl_database.csv"],
    },
    "D3": {
        "name": "CODE-15%",
        "target_dir": Path("raw_data") / "code15",
        "required_any": ["exams.csv", "annotations.csv"],
    },
    "D4": {
        "name": "CPSC2018",
        "target_dir": Path("raw_data") / "cpsc2018",
        "required_any": ["REFERENCE.csv", "REFERENCE-v3.csv"],
    },
}

CHECKPOINT_TARGETS = {
    "M1": {"name": "MERL", "target": Path("checkpoints") / "merl" / "merl_checkpoint.pth"},
    "M2": {"name": "ST-MEM", "target": Path("checkpoints") / "st_mem" / "st_mem_checkpoint.pth"},
    "M3": {"name": "HuBERT-ECG", "target": Path("checkpoints") / "hubert_ecg" / "hubert_ecg_checkpoint.pth"},
    "M4": {"name": "ECGFM-KED", "target": Path("checkpoints") / "ecgfm_ked" / "ecgfm_ked_checkpoint.pth"},
    "M5": {"name": "ECG-FM", "target": Path("checkpoints") / "ecg_fm" / "ecg_fm_checkpoint.pth"},
}

CHECKPOINT_HINTS = {
    "M1": ["merl"],
    "M2": ["st_mem", "st-mem", "stmem"],
    "M3": ["hubert_ecg", "hubert-ecg", "hubert"],
    "M4": ["ecgfm_ked", "ecgfm-ked", "ked"],
    "M5": ["ecg_fm", "ecg-fm"],
}


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


def find_first(root: Path, filenames: list[str]) -> Path | None:
    if not root.exists():
        return None
    for filename in filenames:
        matches = list(root.rglob(filename))
        if matches:
            return matches[0]
    return None


def checkpoint_candidates(root: Path, hints: list[str]) -> list[Path]:
    if not root.exists():
        return []
    suffixes = {".pth", ".pt", ".ckpt"}
    candidates = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            lowered = str(path).lower()
            if any(hint in lowered for hint in hints):
                candidates.append(path)
    return sorted(candidates)


def checkpoint_candidates_from_globs(root: Path, patterns: list[str]) -> list[Path]:
    if not root.exists():
        return []
    candidates = []
    for pattern in patterns:
        candidates.extend(p for p in root.glob(pattern) if p.is_file())
    return sorted(set(candidates))


def maybe_copy(src: Path | None, dst: Path, copy_assets: bool) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src is None:
        return {"source": None, "target": str(dst), "copied": False, "target_exists": dst.exists()}
    if copy_assets and not dst.exists():
        shutil.copy2(src, dst)
        copied = True
    else:
        copied = False
    return {"source": str(src), "target": str(dst), "copied": copied, "target_exists": dst.exists()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage real open-dataset experiment assets")
    parser.add_argument("--incoming-data-root", type=Path, default=Path("workspace/incoming_assets/data"))
    parser.add_argument("--incoming-checkpoints-root", type=Path, default=Path("workspace/incoming_assets/checkpoints"))
    parser.add_argument("--workspace-root", type=Path, default=Path("workspace"))
    parser.add_argument("--output", type=Path, default=Path("workspace/results/open_dataset_asset_staging_status.json"))
    parser.add_argument("--checkpoint-mapping", type=Path, default=Path("workspace/code/checkpoint_mapping.json"),
                        help="Optional JSON mapping from model IDs to checkpoint target paths and incoming globs")
    parser.add_argument("--copy-assets", action="store_true",
                        help="Copy recognized incoming files into canonical target paths")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    workspace_root = args.workspace_root if args.workspace_root.is_absolute() else repo_root / args.workspace_root
    incoming_data_root = args.incoming_data_root if args.incoming_data_root.is_absolute() else repo_root / args.incoming_data_root
    incoming_checkpoints_root = (
        args.incoming_checkpoints_root
        if args.incoming_checkpoints_root.is_absolute()
        else repo_root / args.incoming_checkpoints_root
    )
    output = args.output if args.output.is_absolute() else repo_root / args.output
    checkpoint_mapping_path = (
        args.checkpoint_mapping
        if args.checkpoint_mapping.is_absolute()
        else repo_root / args.checkpoint_mapping
    )
    checkpoint_mapping = load_checkpoint_mapping(checkpoint_mapping_path)

    dataset_status = {}
    for dataset_id, spec in DATASET_TARGETS.items():
        target_dir = workspace_root / spec["target_dir"]
        target_dir.mkdir(parents=True, exist_ok=True)
        source = find_first(incoming_data_root, spec["required_any"])
        copy_result = None
        if source is not None:
            copy_result = maybe_copy(source, target_dir / source.name, args.copy_assets)
        found_required = []
        for required in spec["required_any"]:
            found_required.extend(str(p) for p in target_dir.rglob(required))
        dataset_status[dataset_id] = {
            "name": spec["name"],
            "target_dir": str(target_dir),
            "required_any": spec["required_any"],
            "incoming_source": str(source) if source else None,
            "copy_result": copy_result,
            "found_required_files": found_required,
            "staged": bool(found_required),
        }

    checkpoint_status = {}
    for model_id, spec in CHECKPOINT_TARGETS.items():
        mapping_entry = checkpoint_mapping.get(model_id, {})
        mapped_relative_path = mapping_entry.get("relative_path")
        target = workspace_root / (Path(mapped_relative_path) if mapped_relative_path else spec["target"])
        target.parent.mkdir(parents=True, exist_ok=True)
        incoming_globs = mapping_entry.get("incoming_globs") or []
        candidates = (
            checkpoint_candidates_from_globs(incoming_checkpoints_root, incoming_globs)
            if incoming_globs
            else checkpoint_candidates(incoming_checkpoints_root, CHECKPOINT_HINTS[model_id])
        )
        source = candidates[0] if candidates else None
        copy_result = maybe_copy(source, target, args.copy_assets) if source is not None else None
        checkpoint_status[model_id] = {
            "name": spec["name"],
            "target": str(target),
            "mapping_entry": mapping_entry,
            "mapping_path": str(checkpoint_mapping_path) if checkpoint_mapping_path.exists() else None,
            "incoming_candidates": [str(p) for p in candidates[:10]],
            "copy_result": copy_result,
            "staged": target.exists(),
        }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "05_experimentation",
        "experiment": "real_open_dataset_asset_staging",
        "copy_assets": args.copy_assets,
        "incoming_data_root": str(incoming_data_root),
        "incoming_checkpoints_root": str(incoming_checkpoints_root),
        "checkpoint_mapping_path": str(checkpoint_mapping_path),
        "checkpoint_mapping_loaded": checkpoint_mapping_path.exists(),
        "workspace_root": str(workspace_root),
        "datasets": dataset_status,
        "checkpoints": checkpoint_status,
        "all_open_datasets_staged": all(item["staged"] for item in dataset_status.values()),
        "all_checkpoints_staged": all(item["staged"] for item in checkpoint_status.values()),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "output": str(output),
        "all_open_datasets_staged": report["all_open_datasets_staged"],
        "all_checkpoints_staged": report["all_checkpoints_staged"],
    }, indent=2))
    return 0 if report["all_open_datasets_staged"] and report["all_checkpoints_staged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
