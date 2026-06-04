"""Visualization: reliability diagrams, gap-closure charts, heatmaps, H4 curves.

Generates publication-quality figures for all primary results.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import ALL_MODEL_NAMES, FM_NAMES, RECAL_NAMES, DATASET_NAMES, ECE_N_BINS

logger = logging.getLogger(__name__)

COLORS = {
    "M1": "#e41a1c",
    "M2": "#377eb8",
    "M3": "#4daf4a",
    "M4": "#984ea3",
    "M5": "#ff7f00",
    "M6": "#a65628",
}
METHOD_COLORS = {
    "R1": "#1b9e77",
    "R2": "#d95f02",
    "R3": "#7570b3",
    "R4": "#e7298a",
}


def plot_reliability_diagram(
    bins_data: list,
    model_name: str,
    dataset_name: str,
    class_name: str,
    output_path: Path,
    ece: Optional[float] = None,
    sce: Optional[float] = None,
):
    """Single reliability diagram for one model-dataset-class."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), height_ratios=[3, 1],
                                    gridspec_kw={"hspace": 0.05})

    bin_centers = []
    accuracies = []
    confidences = []
    counts = []

    for b in bins_data:
        if b["n_samples"] > 0:
            bin_centers.append(b["bin_center"])
            accuracies.append(b["avg_accuracy"])
            confidences.append(b["avg_confidence"])
            counts.append(b["n_samples"])

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.bar(bin_centers, accuracies, width=1/ECE_N_BINS * 0.8, alpha=0.6,
            edgecolor="black", linewidth=0.5, label="Observed")

    label_parts = []
    if ece is not None:
        label_parts.append(f"ECE={ece:.3f}")
    if sce is not None:
        direction = "under" if sce < 0 else "over"
        label_parts.append(f"SCE={sce:.3f} ({direction}conf.)")
    if label_parts:
        ax1.set_title(f"{model_name} / {dataset_name} / {class_name}\n{', '.join(label_parts)}",
                       fontsize=10)
    else:
        ax1.set_title(f"{model_name} / {dataset_name} / {class_name}", fontsize=10)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8)
    ax1.set_xticklabels([])

    ax2.bar(bin_centers, counts, width=1/ECE_N_BINS * 0.8, color="gray", alpha=0.5)
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_grid(
    phase_a_results: dict,
    output_dir: Path,
    representative_class_idx: int = 0,
):
    """Grid of reliability diagrams: models × datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_id, dataset_results in phase_a_results.items():
        for model_id, model_results in dataset_results.items():
            per_class = model_results.get("per_class", {})
            class_names = list(per_class.keys())
            if not class_names:
                continue

            class_name = class_names[min(representative_class_idx, len(class_names) - 1)]
            class_data = per_class[class_name]

            if "reliability_diagram" not in class_data:
                continue

            plot_reliability_diagram(
                bins_data=class_data["reliability_diagram"]["bins"],
                model_name=ALL_MODEL_NAMES.get(model_id, model_id),
                dataset_name=DATASET_NAMES.get(dataset_id, dataset_id),
                class_name=class_name,
                output_path=output_dir / f"reliability_{model_id}_{dataset_id}.pdf",
                ece=class_data.get("ece_equal_width"),
                sce=class_data.get("sce"),
            )


def plot_gap_closure_bars(h3_results: dict, output_path: Path):
    """Bar chart of gap closure per FM-dataset pair, colored by best method."""
    per_pair = h3_results.get("per_pair", {})
    if not per_pair:
        logger.warning("No gap closure data to plot")
        return

    pairs = sorted(per_pair.keys())
    closures = [per_pair[p]["gap_closure"] for p in pairs]
    methods = [per_pair[p]["best_method"] for p in pairs]
    colors = [METHOD_COLORS.get(m, "gray") for m in methods]

    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 0.6), 5))
    x = np.arange(len(pairs))
    ax.bar(x, closures, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="50% threshold (H3)")
    ax.axhline(y=1.0, color="green", linestyle=":", alpha=0.5, label="100% closure")

    ax.set_xticks(x)
    labels = [p.replace("_", "\n") for p in pairs]
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Gap Closure")
    ax.set_title("H3: Calibration Gap Closure by Best Recalibration Method")
    ax.legend(fontsize=8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, edgecolor="black", label=RECAL_NAMES.get(m, m))
        for m, c in METHOD_COLORS.items()
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_method_ranking_heatmap(phase_b_results: dict, output_path: Path):
    """Heatmap of recalibration method ECE rankings across FM-dataset pairs."""
    methods = list(RECAL_NAMES.keys())
    pairs = []
    rankings_matrix = []

    for dataset_id in sorted(phase_b_results.keys()):
        for fm_id in sorted(phase_b_results[dataset_id].keys()):
            fm_methods = phase_b_results[dataset_id][fm_id]
            eces = {}
            for m_id in methods:
                if m_id in fm_methods:
                    eces[m_id] = fm_methods[m_id]["macro"]["ece_equal_width"]

            if len(eces) == len(methods):
                sorted_methods = sorted(eces.keys(), key=lambda m: eces[m])
                ranks = {m: i + 1 for i, m in enumerate(sorted_methods)}
                rankings_matrix.append([ranks[m] for m in methods])
                pairs.append(f"{FM_NAMES.get(fm_id, fm_id)}\n{DATASET_NAMES.get(dataset_id, dataset_id)}")

    if not rankings_matrix:
        return

    matrix = np.array(rankings_matrix)
    fig, ax = plt.subplots(figsize=(6, max(4, len(pairs) * 0.4)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=4)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([RECAL_NAMES[m] for m in methods], fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=7)
    ax.set_title("Recalibration Method Rankings (1=best ECE)")

    for i in range(len(pairs)):
        for j in range(len(methods)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Rank")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_h4_size_curves(h4_results: dict, output_path: Path):
    """H4: ECE vs calibration set size for each method, aggregated across FMs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ["R1", "R2", "R3", "R4"]
    aggregated = {m: {} for m in methods}

    for pair_key, pair_results in h4_results.items():
        for size_key, size_data in pair_results.items():
            for method_id in methods:
                if method_id in size_data:
                    if size_key not in aggregated[method_id]:
                        aggregated[method_id][size_key] = []
                    aggregated[method_id][size_key].append(size_data[method_id]["mean_ece"])

    for method_id in methods:
        sizes = sorted(
            aggregated[method_id].keys(),
            key=lambda s: int(s) if s != "full" else 999999,
        )
        x_vals = []
        y_means = []
        y_stds = []
        for s in sizes:
            x_vals.append(int(s) if s != "full" else 50000)
            vals = aggregated[method_id][s]
            y_means.append(np.mean(vals))
            y_stds.append(np.std(vals))

        x_vals = np.array(x_vals)
        y_means = np.array(y_means)
        y_stds = np.array(y_stds)

        ax.plot(x_vals, y_means, "o-", color=METHOD_COLORS[method_id],
                label=RECAL_NAMES[method_id], linewidth=2, markersize=6)
        ax.fill_between(x_vals, y_means - y_stds, y_means + y_stds,
                         alpha=0.15, color=METHOD_COLORS[method_id])

    ax.set_xscale("log")
    ax.set_xlabel("Calibration Set Size")
    ax.set_ylabel("ECE (macro-averaged)")
    ax.set_title("H4: Recalibration Performance vs Calibration Set Size")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(
    phase_a_results: dict,
    phase_b_results: dict,
    h3_results: dict,
    h4_results: dict,
    output_dir: Path,
):
    """Generate all primary publication figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_reliability_grid(phase_a_results, output_dir / "reliability_diagrams")
    plot_gap_closure_bars(h3_results, output_dir / "gap_closure_bars.pdf")
    plot_method_ranking_heatmap(phase_b_results, output_dir / "method_ranking_heatmap.pdf")
    plot_h4_size_curves(h4_results, output_dir / "h4_size_curves.pdf")

    logger.info(f"All figures saved to {output_dir}")
