"""
Nemenyi critical-difference (CD) diagrams for H4 ablation — Demsar (2006) style.

Two-panel figure:
  Left:  n=20,   ECE  — MLP significantly worse than Platt and Temperature
  Right: n=full, Brier — Platt significantly better than all others

Classic layout: left-half methods have labels on the left side of the plot;
right-half methods have labels on the right side. Connector lines link each
label to its rank position on the horizontal axis.

Outputs:
  figures/h4_cd_diagram.png  (150 dpi)
  figures/h4_cd_diagram.pdf  (vector)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE    = Path("/Users/ameenk/AutoR/runs/20260523_210450/workspace")
OUT_PNG = BASE / "figures/h4_cd_diagram.png"
OUT_PDF = BASE / "figures/h4_cd_diagram.pdf"

CD   = 1.049   # Nemenyi CD at α=0.05, k=4, N=20 blocks

# ---------------------------------------------------------------------------
# Panel data: methods sorted best→worst by mean rank
# ---------------------------------------------------------------------------
PANELS = [
    {
        "title": "(a) $n=20$, ECE",
        # (display name, mean rank)
        "methods": [
            ("Platt scaling", 1.80),
            ("Temp. scaling", 2.25),
            ("Isotonic reg.", 2.60),
            ("Learned MLP",   3.35),
        ],
        # Spans of non-significant cliques (pair rank_diff < CD=1.049)
        # Cliques: {Platt,Temp,Iso}→[1.80,2.60], {Iso,MLP}→[2.60,3.35]
        "nonsig_spans": [(1.80, 2.60), (2.60, 3.35)],
    },
    {
        "title": "(b) $n{=}$full, Brier",
        "methods": [
            ("Platt scaling", 1.50),
            ("Isotonic reg.", 2.75),
            ("Learned MLP",   2.75),
            ("Temp. scaling", 3.00),
        ],
        # Clique: {Iso,MLP,Temp}→[2.75,3.00]. Platt isolated.
        "nonsig_spans": [(2.75, 3.00)],
    },
]

# ---------------------------------------------------------------------------
# Demsar-style CD diagram drawing function
# ---------------------------------------------------------------------------

def draw_demsar_panel(ax, panel, cd):
    """
    Classic Demsar (2006) CD diagram.
    - Top-2 ranked methods: labels on LEFT side
    - Bottom-2 ranked methods: labels on RIGHT side
    - Connector: short drop from rank position, then horizontal to label
    - Non-significant groups: thick bars drawn BELOW the axis
    - CD indicator: top-right of axis
    """
    methods   = panel["methods"]      # sorted by rank ascending
    spans     = panel["nonsig_spans"]
    title     = panel["title"]

    n = len(methods)
    ranks = [r for _, r in methods]
    xlim  = (min(ranks) - 0.8, max(ranks) + 0.8)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.70, 0.72)

    # Main horizontal axis at y=0
    ax.axhline(0, color="black", linewidth=1.2, zorder=2)

    # Split: left-side labels (best half) and right-side labels (worst half)
    left_half  = methods[:n//2]     # best-ranked methods → labels on LEFT
    right_half = methods[n//2:]     # worst-ranked methods → labels on RIGHT

    x_left_label  = xlim[0] + 0.05  # left margin for labels (right-aligned)
    x_right_label = xlim[1] - 0.05  # right margin for labels (left-aligned)

    # Drop height for connector lines below axis
    drop_y = -0.22

    def draw_label_left(ax, name, rank, y_label):
        """Draw method dot, connector, and label on the LEFT side."""
        ax.plot(rank, 0, "o", color="black", markersize=7, zorder=5,
                markeredgewidth=1.2)
        # Vertical drop + horizontal line to label
        ax.plot([rank, rank],        [0, drop_y],      color="black", lw=0.9)
        ax.plot([rank, x_left_label+0.02], [drop_y, drop_y], color="black", lw=0.9)
        ax.text(x_left_label, drop_y, name, ha="left", va="center",
                fontsize=8.5, fontfamily="serif",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
        return drop_y   # not used but consistent

    def draw_label_right(ax, name, rank, y_label):
        """Draw method dot, connector, and label on the RIGHT side."""
        ax.plot(rank, 0, "o", color="black", markersize=7, zorder=5,
                markeredgewidth=1.2)
        ax.plot([rank, rank],              [0, drop_y],      color="black", lw=0.9)
        ax.plot([rank, x_right_label-0.02], [drop_y, drop_y], color="black", lw=0.9)
        ax.text(x_right_label, drop_y, name, ha="right", va="center",
                fontsize=8.5, fontfamily="serif",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # Stagger y positions for left-side labels (multiple left methods)
    left_ys  = [drop_y - 0.13 * i for i in range(len(left_half))]
    right_ys = [drop_y - 0.13 * i for i in range(len(right_half))]

    def draw_connector_left(ax, rank, y_label):
        ax.plot(rank, 0, "o", color="black", markersize=7, zorder=5,
                markeredgewidth=1.2)
        ax.plot([rank, rank],              [0, y_label + 0.04],   color="black", lw=0.9)
        ax.plot([rank, x_left_label+0.02], [y_label, y_label],    color="black", lw=0.9)

    def draw_connector_right(ax, rank, y_label):
        ax.plot(rank, 0, "o", color="black", markersize=7, zorder=5,
                markeredgewidth=1.2)
        ax.plot([rank, rank],               [0, y_label + 0.04],  color="black", lw=0.9)
        ax.plot([rank, x_right_label-0.02], [y_label, y_label],   color="black", lw=0.9)

    for i, (name, rank) in enumerate(left_half):
        y_l = left_ys[i]
        draw_connector_left(ax, rank, y_l)
        ax.text(x_left_label, y_l, name, ha="left", va="center",
                fontsize=8.5, fontfamily="serif",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))

    for i, (name, rank) in enumerate(right_half):
        y_r = right_ys[i]
        draw_connector_right(ax, rank, y_r)
        ax.text(x_right_label, y_r, name, ha="right", va="center",
                fontsize=8.5, fontfamily="serif",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # Non-significant group bars — thick lines ABOVE the axis
    bar_y_base = 0.18
    bar_gap    = 0.12
    for j, (r_start, r_end) in enumerate(spans):
        y_bar = bar_y_base + j * bar_gap
        ax.plot([r_start, r_end], [y_bar, y_bar],
                color="black", linewidth=4.5,
                solid_capstyle="round", zorder=4)

    # CD bar indicator — upper right corner
    max_rank = max(ranks)
    cd_x2    = max_rank + 0.6
    cd_x1    = cd_x2 - cd
    cd_y     = 0.55
    ax.annotate("",
                xy=(cd_x1, cd_y), xytext=(cd_x2, cd_y),
                arrowprops=dict(arrowstyle="<->", color="dimgray",
                                lw=1.4, shrinkA=0, shrinkB=0))
    ax.plot([cd_x1, cd_x1], [cd_y-0.03, cd_y+0.03], color="dimgray", lw=1.4)
    ax.plot([cd_x2, cd_x2], [cd_y-0.03, cd_y+0.03], color="dimgray", lw=1.4)
    ax.text((cd_x1+cd_x2)/2, cd_y+0.06, f"CD = {cd:.3f}",
            ha="center", va="bottom", fontsize=8, color="dimgray",
            fontfamily="serif")

    # Axis formatting
    ax.set_title(title, fontsize=10, fontfamily="serif",
                 fontweight="bold", pad=5)
    tick_vals = [t/10 for t in range(
        int(xlim[0]*10), int(xlim[1]*10+1), 5) if xlim[0] <= t/10 <= xlim[1]]
    ax.set_xticks([round(t, 1) for t in tick_vals])
    ax.tick_params(axis="x", labelsize=8, pad=14)
    ax.set_yticks([])
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))
    ax.text((xlim[0]+xlim[1])/2, -0.64, "Mean rank  (1 = best)",
            ha="center", va="bottom", fontsize=8.5, fontfamily="serif")


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
fig.subplots_adjust(wspace=0.32, left=0.02, right=0.98, top=0.89, bottom=0.12)

for ax, panel in zip(axes, PANELS):
    draw_demsar_panel(ax, panel, CD)

fig.text(
    0.5, 0.01,
    "Thick bars connect methods not significantly different "
    "(Nemenyi CD, $\\alpha{=}0.05$, $k{=}4$, $N{=}20$ blocks). "
    " Isolated methods are significantly worse/better than all connected groups.",
    ha="center", va="bottom", fontsize=7.5, color="gray", fontfamily="serif",
)

for fmt, path in [("png", OUT_PNG), ("pdf", OUT_PDF)]:
    plt.savefig(path, dpi=150 if fmt == "png" else None,
                bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}  ({path.stat().st_size:,} bytes)")

plt.close()
print("Done.")
