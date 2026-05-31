"""
Generates the 4 figures used in main.tex:
  figures/interface_framework.pdf
  figures/domain_pattern.pdf
  figures/tradeoff_radar.pdf
  figures/cost_accuracy_pareto.pdf
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

COLORS = {
    "base": "#888888",
    "direct": "#d62728",
    "raw": "#1f77b4",
    "remove": "#2ca02c",
    "egpr": "#9467bd",
}


# -------------------------------------------------------------------------
# Figure 1: interface_framework.pdf
# -------------------------------------------------------------------------

def _box(ax, xy, w, h, text, fc="#f6f6f6", ec="#333", fontsize=9, weight="normal"):
    x, y = xy
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=0.9, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize, weight=weight,
        wrap=True,
    )


def _arrow(ax, src, dst, color="#333", style="-|>", lw=1.0, connectionstyle="arc3,rad=0"):
    a = FancyArrowPatch(
        src, dst,
        arrowstyle=style, mutation_scale=10,
        linewidth=lw, color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(a)


def draw_interface_framework():
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.6)
    ax.set_axis_off()

    # Common input box
    _box(ax, (0.2, 2.55), 1.7, 1.5, "User\nhistory $H_u$",
         fc="#fff3cd", ec="#8a6d3b", weight="bold")

    # ---------------- Direct (top) ----------------
    _box(ax, (3.0, 5.0), 2.6, 1.2, "LLM call\n(candidates in prompt)",
         fc="#fce8e6", ec="#a52a2a")
    _box(ax, (6.3, 5.0), 2.4, 1.2, "Reranked\ncandidates", fc="#fff",
         ec="#a52a2a")
    _box(ax, (9.1, 5.0), 2.6, 1.2, "Transient\nordering",
         fc="#fbd9d3", ec="#a52a2a", weight="bold")
    ax.text(11.95, 6.5, "Direct knowledge state",
            ha="right", va="top", fontsize=10, weight="bold", color="#a52a2a")

    _arrow(ax, (1.9, 3.5), (3.0, 5.5))
    _arrow(ax, (2.1, 5.7), (3.0, 5.7))
    ax.text(2.55, 5.95, "$C_u$", ha="center", fontsize=9, color="#a52a2a")
    _arrow(ax, (5.6, 5.6), (6.3, 5.6))
    _arrow(ax, (8.7, 5.6), (9.1, 5.6))

    # ---------------- Profile (middle) ----------------
    _box(ax, (3.0, 2.9), 2.6, 1.2, "LLM call\n(history only)",
         fc="#e8f1fc", ec="#114a99")
    _box(ax, (6.3, 2.9), 2.4, 1.2,
         "Profile $P_u$\n(claims)", fc="#fff", ec="#114a99")
    _box(ax, (9.1, 2.9), 2.6, 1.2, "Reusable knowledge\nartifact",
         fc="#d3e4fb", ec="#114a99", weight="bold")
    ax.text(11.95, 4.4, "Profile knowledge artifact",
            ha="right", va="top", fontsize=10, weight="bold", color="#114a99")

    _arrow(ax, (1.9, 3.4), (3.0, 3.5))
    _arrow(ax, (5.6, 3.5), (6.3, 3.5))
    _arrow(ax, (8.7, 3.5), (9.1, 3.5))

    # ---------------- Repair (bottom) ----------------
    _box(ax, (3.0, 0.8), 2.6, 1.2, "LLM call\n(history only)",
         fc="#e8f1fc", ec="#114a99")
    _box(ax, (6.3, 0.8), 2.4, 1.2, "Evidence check\n(Remove / EGPR)",
         fc="#fff", ec="#2c7a3a")
    _box(ax, (9.1, 0.8), 2.6, 1.2, "Evidence-governed\nartifact",
         fc="#d9efdc", ec="#2c7a3a", weight="bold")
    ax.text(11.95, 2.3, "Evidence-governed profile",
            ha="right", va="top", fontsize=10, weight="bold", color="#2c7a3a")

    _arrow(ax, (1.9, 3.0), (3.0, 1.4))
    _arrow(ax, (5.6, 1.4), (6.3, 1.4))
    _arrow(ax, (8.7, 1.4), (9.1, 1.4))

    # Side note: candidate set
    ax.text(7.5, 0.25, "Candidate set $C_u$ used only in local reranking " 
            "for profile interfaces; sent into the LLM only for direct.",
            ha="center", va="center", fontsize=8, color="#555", style="italic")

    fig.tight_layout()
    out = OUT_DIR / "interface_framework.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------------------------------------------------------------
# Figure 2: domain_pattern.pdf
# -------------------------------------------------------------------------

def draw_domain_pattern():
    # NDCG@20 from tab:main-utility
    datasets = ["ML-1M (top-50)", "Yelp (top-100)", "Amazon Books (top-100)"]
    methods = ["Base", "Direct", "Profile-Raw", "Profile-Remove", "Profile-EGPR"]
    # Values (NDCG@20). Missing = NaN
    data = np.array([
        # Base    Direct  Raw     Remove  EGPR
        [0.0249, 0.0312, 0.0278,  np.nan, 0.0280],
        [0.0155, 0.0152, 0.0166,  np.nan, 0.0152],
        [0.0043, 0.0039, 0.0049, 0.0050, 0.0048],
    ])
    # Winning method index per dataset (highlight)
    winners = {0: 1, 1: 2, 2: 3}  # ML-1M: Direct, Yelp: Raw, Books: Remove

    method_colors = {
        "Base": COLORS["base"],
        "Direct": COLORS["direct"],
        "Profile-Raw": COLORS["raw"],
        "Profile-Remove": COLORS["remove"],
        "Profile-EGPR": COLORS["egpr"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4))
    bar_w = 0.6

    for ax_i, ax in enumerate(axes):
        vals = data[ax_i]
        xs = np.arange(len(methods))
        bars = []
        for j, (m, v) in enumerate(zip(methods, vals)):
            color = method_colors[m]
            if np.isnan(v):
                continue
            edge = "black" if j == winners[ax_i] else color
            lw = 2.0 if j == winners[ax_i] else 0.8
            b = ax.bar(xs[j], v, width=bar_w,
                       color=color, edgecolor=edge, linewidth=lw)
            bars.extend(b)
            # Value labels
            ax.text(xs[j], v + max(vals[~np.isnan(vals)]) * 0.02,
                    f"{v:.4f}", ha="center", va="bottom",
                    fontsize=7, rotation=0)

        # Base line for reference
        base = vals[0]
        ax.axhline(base, color="#666", linewidth=0.6, linestyle="--", zorder=0)

        ax.set_title(datasets[ax_i], fontsize=10, weight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=7.5)
        ax.set_ylabel("NDCG@20")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        ymax = np.nanmax(vals) * 1.25
        ax.set_ylim(0, ymax)

        # Winner annotation
        winner_method = methods[winners[ax_i]]
        ax.annotate(
            f"Best: {winner_method}",
            xy=(winners[ax_i], vals[winners[ax_i]]),
            xytext=(0.5, 0.95), textcoords="axes fraction",
            ha="center", va="top",
            fontsize=8, weight="bold",
            color=method_colors[winner_method],
        )

    fig.suptitle("The best LLM knowledge state changes by domain",
                 fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "domain_pattern.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------------------------------------------------------------
# Figure 3: tradeoff_radar.pdf
# -------------------------------------------------------------------------

def draw_tradeoff_radar():
    """
    4-axis radar averaging normalized scores across the 3 domains.
    Axes (all higher = better):
      - Accuracy gain (mean (ndcg-base)/base across domains, clipped to [-0.3,0.3] then mapped to 0..1)
      - Cost efficiency (1 - mean cost ratio relative to Direct)
      - Low disruption (mean Overlap@20)
      - Grounding control (1 - mean WUCR; Direct = 1.0 since no profile)
    """
    # Methods to compare
    methods = ["Direct", "Profile-Raw", "Profile-Repair (best)"]
    # Per-domain raw NDCG (Base, Method)
    ndcg = {
        "Direct":         [(0.0249, 0.0312), (0.0155, 0.0152), (0.0043, 0.0039)],
        "Profile-Raw":    [(0.0249, 0.0278), (0.0155, 0.0166), (0.0043, 0.0049)],
        "Profile-Repair (best)": [(0.0249, 0.0280), (0.0155, 0.0152), (0.0043, 0.0050)],
    }
    cost = {
        "Direct": [1.000, 1.000, 1.000],
        "Profile-Raw": [0.446, 0.469, 0.565],
        "Profile-Repair (best)": [0.446, 0.469, 0.565],
    }
    overlap = {
        "Direct": [0.515, 0.605, 0.656],
        "Profile-Raw": [0.699, 0.628, 0.819],
        "Profile-Repair (best)": [0.699, 0.623, 0.910],
    }
    wucr = {
        "Direct": None,  # no profile
        "Profile-Raw": [0.123, 0.089, 0.348],
        "Profile-Repair (best)": [0.009, 0.011, 0.000],
    }

    def mean_gain(pairs):
        gains = [(m - b) / b for b, m in pairs]
        # normalise to [0,1] using clip [-0.3, 0.3]
        m = np.mean(gains)
        return max(0.0, min(1.0, (m + 0.3) / 0.6))

    radar = {}
    for m in methods:
        acc = mean_gain(ndcg[m])
        cost_eff = 1.0 - np.mean(cost[m])
        low_dis = np.mean(overlap[m])
        if wucr[m] is None:
            ground = 1.0  # direct has no profile to be ungrounded
        else:
            ground = 1.0 - np.mean(wucr[m])
        radar[m] = [acc, cost_eff, low_dis, ground]

    labels = ["Accuracy\n(avg gain)", "Cost\nefficiency",
              "Low disruption\n(top-20 overlap)", "Grounding\ncontrol"]
    N = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.0, 5.4), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(180 / N)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7,
                       color="#666")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)

    method_color = {
        "Direct": COLORS["direct"],
        "Profile-Raw": COLORS["raw"],
        "Profile-Repair (best)": COLORS["egpr"],
    }
    for m in methods:
        vals = radar[m] + radar[m][:1]
        ax.plot(angles, vals, linewidth=1.8, label=m, color=method_color[m])
        ax.fill(angles, vals, alpha=0.10, color=method_color[m])

    ax.set_title("Normalized accuracy--reliability--cost trade-off\n"
                 "(averaged across ML-1M, Yelp, Amazon Books)",
                 fontsize=10, weight="bold", pad=22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, frameon=False)

    out = OUT_DIR / "tradeoff_radar.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------------------------------------------------------------
# Figure 4: cost_accuracy_pareto.pdf
# -------------------------------------------------------------------------

def draw_cost_accuracy_pareto():
    """
    Cost-accuracy scatter for each dataset: x = cost ratio vs Direct, y = NDCG@20.
    Pareto-front items are connected by a line.
    """
    datasets = ["ML-1M (top-50)", "Yelp (top-100)", "Amazon Books (top-100)"]
    # (method, cost, ndcg)
    points = {
        "ML-1M (top-50)": [
            ("Base", 0.000, 0.0249),
            ("Direct", 1.000, 0.0312),
            ("Profile-Raw", 0.446, 0.0278),
            ("Profile-EGPR", 0.446, 0.0280),
        ],
        "Yelp (top-100)": [
            ("Base", 0.000, 0.0155),
            ("Direct", 1.000, 0.0152),
            ("Profile-Raw", 0.469, 0.0166),
            ("Profile-EGPR", 0.469, 0.0152),
        ],
        "Amazon Books (top-100)": [
            ("Base", 0.000, 0.0043),
            ("Direct", 1.000, 0.0039),
            ("Profile-Raw", 0.565, 0.0049),
            ("Profile-Remove", 0.565, 0.0050),
            ("Profile-EGPR", 0.565, 0.0048),
        ],
    }
    method_color = {
        "Base": COLORS["base"],
        "Direct": COLORS["direct"],
        "Profile-Raw": COLORS["raw"],
        "Profile-Remove": COLORS["remove"],
        "Profile-EGPR": COLORS["egpr"],
    }
    method_marker = {
        "Base": "s",
        "Direct": "X",
        "Profile-Raw": "o",
        "Profile-Remove": "D",
        "Profile-EGPR": "^",
    }

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    for ax_i, ds in enumerate(datasets):
        ax = axes[ax_i]
        ds_points = points[ds]

        # Pareto front: sort by cost asc, keep monotonically increasing NDCG
        sorted_pts = sorted(ds_points, key=lambda p: (p[1], -p[2]))
        front = []
        best_n = -1
        for name, c, n in sorted_pts:
            if n > best_n:
                front.append((name, c, n))
                best_n = n
        # Draw front line
        if len(front) >= 2:
            fx = [p[1] for p in front]
            fy = [p[2] for p in front]
            ax.plot(fx, fy, color="#444", linestyle="--", linewidth=0.9,
                    alpha=0.7, zorder=1, label="Pareto front")

        # Scatter with anti-overlap label offsets per group
        # Group points by exact (cost, ndcg) proximity for label offsetting
        ys = [p[2] for p in ds_points]
        y_range = max(ys) - min(ys) + 1e-9
        # Pre-compute label offsets when multiple methods share the same cost
        from collections import defaultdict
        cost_groups = defaultdict(list)
        for idx, (name, c, n) in enumerate(ds_points):
            cost_groups[round(c, 3)].append(idx)
        offset_map = {}
        for c_key, idxs in cost_groups.items():
            if len(idxs) <= 1:
                for i in idxs:
                    offset_map[i] = (7, 5)
            else:
                # Sort by ndcg desc and stagger labels above/below
                sorted_idxs = sorted(idxs, key=lambda i: -ds_points[i][2])
                offsets = [(18, 16), (24, -2), (18, -20)]
                for k, i in enumerate(sorted_idxs):
                    offset_map[i] = offsets[k % len(offsets)]
        for idx, (name, c, n) in enumerate(ds_points):
            ax.scatter(c, n,
                       s=100, color=method_color[name],
                       marker=method_marker[name],
                       edgecolor="black", linewidth=0.6, zorder=3,
                       label=name)
            dx, dy = offset_map[idx]
            ax.annotate(name,
                        xy=(c, n),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=7.5, color=method_color[name],
                        arrowprops=dict(arrowstyle="-", color=method_color[name],
                                        lw=0.4, alpha=0.5) if abs(dy) > 8 else None)

        ax.set_title(ds, fontsize=10, weight="bold")
        ax.set_xlabel("Relative API cost (vs Direct)")
        if ax_i == 0:
            ax.set_ylabel("NDCG@20")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(linestyle=":", alpha=0.4)
        ax.set_xlim(-0.08, 1.18)
        # y-axis padding
        ys = [p[2] for p in ds_points]
        pad = (max(ys) - min(ys)) * 0.30 + 1e-4
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    fig.suptitle("Cost--accuracy trade-off across domains",
                 fontsize=11, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT_DIR / "cost_accuracy_pareto.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    draw_interface_framework()
    draw_domain_pattern()
    draw_tradeoff_radar()
    draw_cost_accuracy_pareto()
