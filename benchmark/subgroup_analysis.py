#!/usr/bin/env python3
"""
Analysis 4: Subgroup Calibration by Age and Sex
- Load PTB-XL metadata, align to the 2000-record seed=42 sample
- For MERL (M1) and ST-MEM (M2) only (skip partial models)
- Split by age: <55, 55-70, >=70
- Split by sex: 0=male, 1=female
- Compute AUROC, ECE, SCE, Brier per subgroup
- Save: results/subgroup_results.json
- Figure: writing/figures/fig9_subgroup_calibration.pdf + .png
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings("ignore")

WS = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
RESULTS = WS / "results"
FIGS = WS / "writing" / "figures"
FIGS.mkdir(exist_ok=True)

SEED = 42
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
COLORS_MODEL = {
    "MERL":   "#e74c3c",
    "ST-MEM": "#2980b9",
}
PRETRAIN = {
    "MERL":   "contrastive",
    "ST-MEM": "masked-recon.",
}


def compute_ece(probs, labels, n_bins=15):
    """Equal-width ECE."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return ece


def compute_sce(probs, labels, n_bins=15):
    """Signed calibration error: mean(confidence - accuracy)."""
    bins = np.linspace(0, 1, n_bins + 1)
    sce = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        sce += mask.sum() / n * (conf - acc)
    return sce


def macro_metrics(probs_matrix, labels_matrix):
    """
    Compute macro-averaged AUROC, ECE, SCE, Brier.
    probs_matrix: (n, n_classes), labels_matrix: (n, n_classes)
    """
    n, n_classes = labels_matrix.shape
    aurocs, eces, sces, briers = [], [], [], []
    for c in range(n_classes):
        y_true = labels_matrix[:, c]
        y_prob = probs_matrix[:, c]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        try:
            aurocs.append(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
        eces.append(compute_ece(y_prob, y_true))
        sces.append(compute_sce(y_prob, y_true))
        briers.append(brier_score_loss(y_true, y_prob))
    return {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "macro_ece": float(np.mean(eces)) if eces else float("nan"),
        "macro_sce": float(np.mean(sces)) if sces else float("nan"),
        "macro_brier": float(np.mean(briers)) if briers else float("nan"),
        "n_classes_evaluated": len(aurocs),
    }


def reconstruct_labels_and_meta():
    """
    Reconstruct the same 2000-record PTB-XL sample used in the benchmark.
    Returns labels (N, 5), ages (N,), sexes (N,), valid_ecg_ids.
    """
    root = WS / "raw_data/ptb-xl/1.0.3"
    meta = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    meta.scp_codes = meta.scp_codes.apply(eval)
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    scp = scp[scp.diagnostic == 1]
    sc_map = dict(zip(scp.index, scp.diagnostic_class))

    def scp2vec(codes):
        v = np.zeros(5, dtype=np.float32)
        for c, conf in codes.items():
            if c in sc_map and conf >= 50 and sc_map[c] in CLASSES:
                v[CLASSES.index(sc_map[c])] = 1
        return v

    rows = meta.sample(min(2000, len(meta)), random_state=SEED)
    labels, ages, sexes, ecg_ids = [], [], [], []
    for idx, row in rows.iterrows():
        path = root / row.filename_lr
        if Path(str(path) + ".hea").exists() or Path(str(path)).with_suffix(".hea").exists():
            labels.append(scp2vec(row.scp_codes))
            ages.append(float(row.age) if not pd.isna(row.age) else float("nan"))
            sexes.append(int(row.sex) if not pd.isna(row.sex) else -1)
            ecg_ids.append(idx)
    return (np.stack(labels), np.array(ages), np.array(sexes), ecg_ids)


def train_probe_and_predict(features, labels, seed=SEED):
    """
    Train a linear probe on 70% of data, return predictions for all records.
    Uses the same train/cal/test split proportions as the benchmark.
    """
    n = len(features)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    train_n = int(0.7 * n)
    train_idx = idx[:train_n]
    test_idx = idx[train_n:]

    X_train, y_train = features[train_idx], labels[train_idx]
    X_all = features

    n_classes = labels.shape[1]
    probs = np.zeros((n, n_classes), dtype=np.float32)

    for c in range(n_classes):
        if y_train[:, c].sum() < 5:
            probs[:, c] = 0.1
            continue
        lr = LogisticRegression(max_iter=500, C=1.0, random_state=seed)
        lr.fit(X_train, y_train[:, c])
        probs[:, c] = lr.predict_proba(X_all)[:, 1]

    return probs, test_idx


def analyze_subgroups(features, labels, ages, sexes, model_name):
    """Compute subgroup metrics for a single model."""
    min_n = min(len(features), len(labels))
    features = features[:min_n]
    labels = labels[:min_n]
    ages = ages[:min_n]
    sexes = sexes[:min_n]

    print(f"\n  Training probe for {model_name}...")
    probs, test_idx = train_probe_and_predict(features, labels)

    # Work on test split only
    probs_test = probs[test_idx]
    labels_test = labels[test_idx]
    ages_test = ages[test_idx]
    sexes_test = sexes[test_idx]

    age_groups = {
        "age<55": ages_test < 55,
        "age55-70": (ages_test >= 55) & (ages_test < 70),
        "age>=70": ages_test >= 70,
    }
    sex_groups = {
        "male": sexes_test == 0,
        "female": sexes_test == 1,
    }
    all_groups = {"overall": np.ones(len(test_idx), dtype=bool)}
    all_groups.update(age_groups)
    all_groups.update(sex_groups)

    results = {}
    for group_name, mask in all_groups.items():
        n_group = mask.sum()
        if n_group < 20:
            results[group_name] = {"n": int(n_group), "note": "too_small"}
            continue
        metrics = macro_metrics(probs_test[mask], labels_test[mask])
        metrics["n"] = int(n_group)
        results[group_name] = metrics
        print(f"    {group_name}: n={n_group}, AUROC={metrics['macro_auroc']:.3f}, "
              f"ECE={metrics['macro_ece']:.4f}, SCE={metrics['macro_sce']:.4f}")

    return results


def main():
    print("Reconstructing PTB-XL labels and metadata...")
    labels, ages, sexes, ecg_ids = reconstruct_labels_and_meta()
    n = len(labels)
    print(f"Got {n} records")
    print(f"Age range: {np.nanmin(ages):.0f}–{np.nanmax(ages):.0f}, mean={np.nanmean(ages):.1f}")
    print(f"Sex distribution: {(sexes==0).sum()} male, {(sexes==1).sum()} female")

    models = {
        "MERL":   np.load(RESULTS / "features_M1_D1.npy"),
        "ST-MEM": np.load(RESULTS / "features_M2_D1.npy"),
    }

    all_results = {}
    for model_name, feats in models.items():
        print(f"\n=== {model_name} ===")
        subgroup_metrics = analyze_subgroups(feats, labels, ages, sexes, model_name)
        all_results[model_name] = subgroup_metrics

    out = {
        "description": "Subgroup calibration analysis for PTB-XL (MERL and ST-MEM only)",
        "seed": SEED,
        "classes": CLASSES,
        "n_total": n,
        "note": (
            "Linear probe trained on 70% of the 2000-record sample; subgroup metrics "
            "computed on the remaining 30% (test split). Sample sizes per subgroup shown."
        ),
        "models": all_results,
    }
    with open(RESULTS / "subgroup_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {RESULTS}/subgroup_results.json")

    # Figure: grouped bar charts — age groups × ECE for MERL and ST-MEM
    age_groups = ["age<55", "age55-70", "age>=70"]
    age_labels = ["Age < 55", "Age 55–70", "Age ≥ 70"]
    model_names = list(all_results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    metrics_to_plot = [("macro_auroc", "Macro AUROC"), ("macro_ece", "Macro ECE")]
    for ax, (metric_key, metric_label) in zip(axes, metrics_to_plot):
        x = np.arange(len(age_groups))
        width = 0.35
        for i, model in enumerate(model_names):
            vals = []
            ns = []
            for ag in age_groups:
                m = all_results[model].get(ag, {})
                v = m.get(metric_key, float("nan"))
                vals.append(v)
                ns.append(m.get("n", 0))
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=f"{model} ({PRETRAIN[model]})",
                          color=COLORS_MODEL[model], alpha=0.85,
                          edgecolor="black", linewidth=0.5)
            for bar, n_val, v in zip(bars, ns, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                            f"n={n_val}", ha="center", va="bottom", fontsize=7, rotation=45)

        ax.set_xlabel("Age Subgroup", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label + " by Age Subgroup", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(age_labels)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Subgroup Calibration Analysis — PTB-XL (MERL and ST-MEM)\n"
        "Linear probe on frozen features; test split only",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ["pdf", "png"]:
        path = FIGS / f"fig9_subgroup_calibration.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
