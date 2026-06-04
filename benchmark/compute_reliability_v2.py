#!/usr/bin/env python3
"""
Reliability diagrams (pooled across classes) for each model on PTB-XL and
MIMIC-IV-ECG, recomputed deterministically from cached features + the same
patient split and linear probe used in benchmark v2. Shows raw vs isotonic.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
RESULTS = WS / "results"
FIGS = WS / "writing" / "figures"
sys.path.insert(0, str(WS / "code"))
STMEM = WS / "checkpoints/_sources/ST-MEM"
for p in (STMEM, STMEM/"models", STMEM/"models/encoder"):
    sys.path.insert(0, str(p))

from run_benchmark_v2 import linear_probe, patient_split, _valid_classes
from run_full_benchmark import load_ptbxl, load_cpsc2018
from mimic_labels_v2 import load_mimic_v2

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "figure.dpi": 150})

MODELS = [("M1", "MERL"), ("M2", "ST-MEM"), ("M3", "HuBERT-ECG"),
          ("M4", "ECGFM-KED"), ("M5", "ECG-FM"), ("M6", "S4D")]


def reliability_points(yt, yp, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, ws = [], [], []
    for i in range(n_bins):
        m = (yp >= bins[i]) & (yp < bins[i + 1])
        if m.sum() >= 5:
            xs.append(yp[m].mean()); ys.append(yt[m].mean()); ws.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ws)


def isotonic_recal(y_cal, p_cal, p_test, n_classes):
    from sklearn.isotonic import IsotonicRegression
    pr = np.array(p_test, dtype=float)
    for c in range(n_classes):
        if len(np.unique(y_cal[:, c])) < 2:
            continue
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal[:, c], y_cal[:, c])
        pr[:, c] = iso.predict(p_test[:, c])
    return pr


def make_grid(ds_name, loader, d_id, extra_cache_suffix=""):
    sigs, labels, pids, classes = loader()
    split = patient_split(pids, labels)
    valid = _valid_classes(labels[split["test"]])
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.ravel()
    for ax, (mid, mname) in zip(axes, MODELS):
        cache = RESULTS / f"features_{mid}_{d_id}{extra_cache_suffix}.npy"
        if not cache.exists():
            ax.set_title(f"{mname}\n(no features)", fontsize=9)
            ax.axis("off"); continue
        feats = np.load(cache)
        if len(feats) != len(labels):
            ax.set_title(f"{mname}\n(cache mismatch)", fontsize=9); ax.axis("off"); continue
        p_test = linear_probe(feats[split["train"]], labels[split["train"]], feats[split["test"]])
        p_cal = linear_probe(feats[split["train"]], labels[split["train"]], feats[split["cal"]])
        p_iso = isotonic_recal(labels[split["cal"]], p_cal, p_test, labels.shape[1])
        # pool over valid classes
        yt = np.concatenate([labels[split["test"]][:, c] for c in valid])
        yp_raw = np.concatenate([p_test[:, c] for c in valid])
        yp_iso = np.concatenate([p_iso[:, c] for c in valid])
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, lw=1)
        xr, yr, _ = reliability_points(yt, yp_raw)
        xi, yi, _ = reliability_points(yt, yp_iso)
        ax.plot(xr, yr, "o-", color="#e41a1c", ms=4, lw=1.4, label="Raw")
        ax.plot(xi, yi, "s-", color="#4daf4a", ms=4, lw=1.4, label="Isotonic")
        ax.set_title(mname, fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
        ax.set_xlabel("Predicted prob."); ax.set_ylabel("Empirical freq.")
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Reliability Diagrams (pooled over {len(valid)} classes): {ds_name}",
                 fontweight="bold")
    fig.tight_layout()
    out = FIGS / f"fig_reliability_{d_id}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {out.name} ({len(valid)} classes pooled)")


if __name__ == "__main__":
    make_grid("PTB-XL", lambda: load_ptbxl(max_n=2000), "D1")
    make_grid("MIMIC-IV-ECG", lambda: load_mimic_v2(max_n=2000), "D5")
    print("reliability done")
