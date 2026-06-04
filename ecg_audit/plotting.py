"""Publication-quality reliability diagrams."""
from __future__ import annotations
import numpy as np


def reliability_curve(y_true, y_prob, n_bins: int = 10):
    """Return (mean_predicted, empirical_frequency, bin_counts) for non-empty bins."""
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys, ws = [], [], []
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if m.sum() > 0:
            xs.append(float(y_prob[m].mean()))
            ys.append(float(y_true[m].mean()))
            ws.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ws)


def reliability_diagram(y_true, y_prob, n_bins: int = 10, title: str = "",
                        out_path: str | None = None, label: str = "model"):
    """Plot a reliability diagram; save to out_path if given. Returns the Axes."""
    import matplotlib
    if out_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs, ys, _ = reliability_curve(y_true, y_prob, n_bins)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.plot(xs, ys, "o-", lw=1.5, label=label)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(loc="upper left")
    if title:
        ax.set_title(title)
    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    return ax
