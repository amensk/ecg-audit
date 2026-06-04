"""Leaderboard scaffold: format/aggregate calibration-aware model entries."""
from __future__ import annotations
import json
from pathlib import Path


def composite_score(auroc: float, ece: float, ece_max: float, alpha: float = 0.7) -> float:
    """C(alpha) = alpha*AUROC + (1-alpha)*(1 - ECE/ECE_max)."""
    return float(alpha * auroc + (1 - alpha) * (1 - ece / ece_max))


def make_entry(model: str, dataset: str, auroc: float, ece: float,
               brier: float = None, sce: float = None, **extra) -> dict:
    entry = {"model": model, "dataset": dataset, "auroc": auroc, "ece": ece}
    if brier is not None:
        entry["brier"] = brier
    if sce is not None:
        entry["sce"] = sce
    entry.update(extra)
    return entry


def submit(entry: dict, leaderboard_path: str) -> dict:
    """Append an entry to a JSON leaderboard file (creating it if needed)."""
    path = Path(leaderboard_path)
    board = []
    if path.exists():
        board = json.loads(path.read_text())
    board.append(entry)
    path.write_text(json.dumps(board, indent=2))
    return {"n_entries": len(board), "path": str(path)}


def rank(board: list, alpha: float = 0.7) -> list:
    """Rank entries by composite score C(alpha); ece_max taken across the board."""
    if not board:
        return []
    ece_max = max(e["ece"] for e in board) or 1e-9
    scored = [dict(e, composite=composite_score(e["auroc"], e["ece"], ece_max, alpha))
              for e in board]
    return sorted(scored, key=lambda e: -e["composite"])
