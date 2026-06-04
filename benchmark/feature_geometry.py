#!/usr/bin/env python3
"""
Analysis 1: Feature Geometry
- Load PTB-XL features for MERL (M1), ST-MEM (M2), ECG-FM (M5)
- Compute per-class cosine similarity within vs across class
- Compute feature spread (std of L2 norms)
- Compare with known ECE values
- Save: results/feature_geometry.json
- Figure: writing/figures/fig6_feature_geometry.pdf + .png
"""
import sys, json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
RESULTS = WS / "results"
FIGS = WS / "writing" / "figures"
FIGS.mkdir(exist_ok=True)

SEED = 42
# Known ECE/AUROC values from benchmark_v2 artifacts (PTB-XL)
ECE_VALUES = {
    "MERL":       0.0871,
    "ST-MEM":     0.1239,
    "HuBERT-ECG": 0.1365,
    "ECGFM-KED":  0.0792,
    "ECG-FM":     0.0882,
}
AUROC_VALUES = {
    "MERL":       0.828,
    "ST-MEM":     0.838,
    "HuBERT-ECG": 0.787,
    "ECGFM-KED":  0.769,
    "ECG-FM":     0.815,
}
COLORS = {
    "MERL":       "#2c7bb6",
    "ST-MEM":     "#d7191c",
    "HuBERT-ECG": "#8c510a",
    "ECGFM-KED":  "#1a9641",
    "ECG-FM":     "#ff7f00",
}
PRETRAIN_TYPE = {
    "MERL":       "contrastive",
    "ST-MEM":     "masked-reconstruction",
    "HuBERT-ECG": "masked hidden-unit pred.",
    "ECGFM-KED":  "knowledge-enhanced",
    "ECG-FM":     "contrastive+generative",
}

# Reconstruct labels from the same 2000-record PTB-XL sample with seed=42
# This mirrors the load_ptbxl function from run_full_benchmark.py exactly
def reconstruct_ptbxl_labels():
    import pandas as pd
    root = WS / "raw_data/ptb-xl/1.0.3"
    meta = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    meta.scp_codes = meta.scp_codes.apply(eval)
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    scp = scp[scp.diagnostic == 1]
    sc_map = dict(zip(scp.index, scp.diagnostic_class))
    classes = ["NORM", "MI", "STTC", "CD", "HYP"]

    def scp2vec(codes):
        v = np.zeros(5, dtype=np.float32)
        for c, conf in codes.items():
            if c in sc_map and conf >= 50 and sc_map[c] in classes:
                v[classes.index(sc_map[c])] = 1
        return v

    rows = meta.sample(min(2000, len(meta)), random_state=SEED)
    labels = []
    patient_ids = []
    valid_ecg_ids = []
    for idx, row in rows.iterrows():
        # Check if file exists
        path = root / row.filename_lr
        if Path(str(path) + ".hea").exists() or Path(str(path)).with_suffix(".hea").exists():
            labels.append(scp2vec(row.scp_codes))
            patient_ids.append(row.patient_id)
            valid_ecg_ids.append(idx)
    print(f"Reconstructed {len(labels)} labels")
    return np.stack(labels), np.array(patient_ids), valid_ecg_ids, classes

def compute_geometry(features, labels, model_name):
    """Compute feature geometry statistics."""
    n, d = features.shape
    classes = labels.shape[1]

    # L2 normalize features for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms_scalar = np.linalg.norm(features, axis=1)
    feat_spread = float(np.std(norms_scalar))

    features_normed = features / (norms + 1e-8)

    # Compute cosine similarity matrix (subsampled if large)
    max_n = min(n, 500)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n, max_n, replace=False)
    feat_sub = features_normed[idx]
    lab_sub = labels[idx]

    cos_sim = cosine_similarity(feat_sub)  # (max_n, max_n)

    # Per-class within vs across cosine similarity
    within_sims = []
    across_sims = []

    for c in range(classes):
        pos_mask = lab_sub[:, c] == 1
        neg_mask = ~pos_mask
        n_pos = pos_mask.sum()

        if n_pos < 2:
            continue

        # Within-class: upper triangle of pos×pos block
        pos_idx = np.where(pos_mask)[0]
        neg_idx = np.where(neg_mask)[0]

        within_block = cos_sim[np.ix_(pos_idx, pos_idx)]
        iu = np.triu_indices_from(within_block, k=1)
        if len(iu[0]) > 0:
            within_sims.extend(within_block[iu].tolist())

        # Across-class: pos×neg block
        if len(neg_idx) > 0:
            across_block = cos_sim[np.ix_(pos_idx, neg_idx)]
            across_sims.extend(across_block.flatten().tolist())

    within_mean = float(np.mean(within_sims)) if within_sims else float("nan")
    across_mean = float(np.mean(across_sims)) if across_sims else float("nan")

    return {
        "model": model_name,
        "n_samples": n,
        "feature_dim": d,
        "feature_spread_std_l2norm": feat_spread,
        "within_class_cosine_mean": within_mean,
        "across_class_cosine_mean": across_mean,
        "cosine_separation": within_mean - across_mean,
        "ece": ECE_VALUES[model_name],
        "auroc": AUROC_VALUES[model_name],
    }

def main():
    print("Reconstructing PTB-XL labels...")
    labels, pids, ecg_ids, classes = reconstruct_ptbxl_labels()
    n = len(labels)
    print(f"Got {n} label vectors, classes: {classes}")

    models = {
        "MERL":       np.load(RESULTS / "features_M1_D1.npy"),
        "ST-MEM":     np.load(RESULTS / "features_M2_D1.npy"),
        "HuBERT-ECG": np.load(RESULTS / "features_M3_D1.npy"),
        "ECGFM-KED":  np.load(RESULTS / "features_M4_D1.npy"),
        "ECG-FM":     np.load(RESULTS / "features_M5_D1.npy"),
    }

    results = {}
    for model_name, feats in models.items():
        # Align features to available labels
        min_n = min(len(feats), n)
        feat_aligned = feats[:min_n]
        lab_aligned = labels[:min_n]
        print(f"\nComputing geometry for {model_name} ({feat_aligned.shape})...")
        geo = compute_geometry(feat_aligned, lab_aligned, model_name)
        results[model_name] = geo
        print(f"  spread={geo['feature_spread_std_l2norm']:.4f}, "
              f"within_cos={geo['within_class_cosine_mean']:.4f}, "
              f"across_cos={geo['across_class_cosine_mean']:.4f}, "
              f"separation={geo['cosine_separation']:.4f}")

    # Save JSON
    out = {
        "description": "Feature geometry analysis for PTB-XL (5 FMs: MERL, ST-MEM, HuBERT-ECG, ECGFM-KED, ECG-FM)",
        "seed": SEED,
        "classes": classes,
        "models": results
    }
    with open(RESULTS / "feature_geometry.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {RESULTS}/feature_geometry.json")

    # Figure: scatter of feature spread vs ECE
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    model_names = list(results.keys())
    spreads = [results[m]["feature_spread_std_l2norm"] for m in model_names]
    eces = [results[m]["ece"] for m in model_names]
    separations = [results[m]["cosine_separation"] for m in model_names]

    ax = axes[0]
    for m in model_names:
        label_str = f"{m} ({PRETRAIN_TYPE[m]})"
        ax.scatter(results[m]["feature_spread_std_l2norm"], results[m]["ece"],
                   color=COLORS[m], s=120, zorder=5, label=label_str)
        ax.annotate(m, (results[m]["feature_spread_std_l2norm"], results[m]["ece"]),
                    textcoords="offset points", xytext=(8, 3), fontsize=9)
    ax.set_xlabel("Feature Spread (std of L2 norms)", fontsize=11)
    ax.set_ylabel("ECE (PTB-XL, macro)", fontsize=11)
    ax.set_title("Feature Spread vs Calibration Error", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for m in model_names:
        label_str = f"{m} ({PRETRAIN_TYPE[m]})"
        ax2.scatter(results[m]["cosine_separation"], results[m]["ece"],
                    color=COLORS[m], s=120, zorder=5, label=label_str)
        ax2.annotate(m, (results[m]["cosine_separation"], results[m]["ece"]),
                     textcoords="offset points", xytext=(8, 3), fontsize=9)
    ax2.set_xlabel("Cosine Separation (within − across class)", fontsize=11)
    ax2.set_ylabel("ECE (PTB-XL, macro)", fontsize=11)
    ax2.set_title("Class Separation vs Calibration Error", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Feature Geometry vs Calibration — PTB-XL",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        path = FIGS / f"fig6_feature_geometry.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)

    print("\nDone. Key findings:")
    for m in model_names:
        r = results[m]
        print(f"  {m}: spread={r['feature_spread_std_l2norm']:.4f}, "
              f"cos_sep={r['cosine_separation']:.4f}, ECE={r['ece']:.4f}")

if __name__ == "__main__":
    main()
