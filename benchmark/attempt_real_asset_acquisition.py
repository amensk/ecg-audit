"""Attempt public real-data and checkpoint acquisition for Stage 05.

The script intentionally records exact access failures instead of assuming
availability. It probes official dataset pages and model repositories before
any full download, because the target datasets and checkpoints are large.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


DATASET_SOURCES = {
    "D1": {
        "name": "PTB-XL",
        "url": "https://physionet.org/content/ptb-xl/1.0.3/",
        "credentialed": False,
    },
    "D2": {
        "name": "MIMIC-IV-ECG",
        "url": "https://physionet.org/content/mimic-iv-ecg/",
        "credentialed": True,
    },
    "D3": {
        "name": "CODE-15%",
        "url": "https://zenodo.org/records/4916206",
        "credentialed": False,
    },
    "D4": {
        "name": "CPSC2018",
        "url": "http://2018.icbeb.org/Challenge.html",
        "credentialed": False,
    },
}


MODEL_SOURCES = {
    "M1": {
        "name": "MERL",
        "repo": "https://github.com/cheliu-computation/MERL.git",
    },
    "M2": {
        "name": "ST-MEM",
        "repo": "https://github.com/bakqui/ST-MEM.git",
    },
    "M3": {
        "name": "HuBERT-ECG",
        "repo": "https://github.com/Edoar-do/HuBERT-ECG.git",
    },
    "M4": {
        "name": "ECGFM-KED",
        "repo": "https://github.com/control-spiderman/ECGFM-KED.git",
    },
    "M5": {
        "name": "ECG-FM",
        "repo": "https://github.com/bowang-lab/ECG-FM.git",
    },
}


def probe_url(url: str, timeout: int) -> dict:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "AutoR-Stage05/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {
                "reachable": True,
                "status": response.status,
                "reason": response.reason,
                "final_url": response.geturl(),
                "error_type": None,
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        return {
            "reachable": True,
            "status": exc.code,
            "reason": exc.reason,
            "final_url": url,
            "error_type": "HTTPError",
            "error": str(exc),
        }
    except (urllib.error.URLError, socket.gaierror, TimeoutError) as exc:
        return {
            "reachable": False,
            "status": None,
            "reason": None,
            "final_url": url,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }


def probe_git(repo: str, timeout: int) -> dict:
    try:
        proc = subprocess.run(
            ["git", "ls-remote", "--heads", repo],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "reachable": proc.returncode == 0,
            "return_code": proc.returncode,
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
            "error_type": None,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - preserve exact acquisition failure
        return {
            "reachable": False,
            "return_code": None,
            "stdout_tail": "",
            "stderr_tail": "",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe public ECG datasets and FM checkpoint sources")
    parser.add_argument("--output", type=Path, default=Path("workspace/results/real_asset_acquisition_attempt.json"))
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output = args.output if args.output.is_absolute() else repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    datasets = {}
    for dataset_id, spec in DATASET_SOURCES.items():
        status = probe_url(spec["url"], args.timeout)
        if spec["credentialed"]:
            status["download_attempted"] = False
            status["download_blocked_reason"] = "credentialed_access_required_or_unverified"
        else:
            status["download_attempted"] = False
            status["download_blocked_reason"] = (
                None if status["reachable"] else "source_unreachable_preflight_failed"
            )
        datasets[dataset_id] = {**spec, **status}

    models = {}
    for model_id, spec in MODEL_SOURCES.items():
        status = probe_git(spec["repo"], args.timeout)
        status["checkpoint_download_attempted"] = False
        status["checkpoint_download_blocked_reason"] = (
            None if status["reachable"] else "repository_unreachable_preflight_failed"
        )
        models[model_id] = {**spec, **status}

    result = {
        "stage": "05_experimentation",
        "experiment": "real_asset_acquisition_attempt",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "network_available_for_public_sources": (
            any(item["reachable"] for item in datasets.values())
            or any(item["reachable"] for item in models.values())
        ),
        "datasets": datasets,
        "models": models,
        "summary": {
            "open_datasets_reachable": [
                dataset_id for dataset_id, item in datasets.items()
                if not item["credentialed"] and item["reachable"]
            ],
            "open_datasets_unreachable": [
                dataset_id for dataset_id, item in datasets.items()
                if not item["credentialed"] and not item["reachable"]
            ],
            "credentialed_datasets_deferred": [
                dataset_id for dataset_id, item in datasets.items() if item["credentialed"]
            ],
            "model_repositories_reachable": [
                model_id for model_id, item in models.items() if item["reachable"]
            ],
            "model_repositories_unreachable": [
                model_id for model_id, item in models.items() if not item["reachable"]
            ],
        },
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
