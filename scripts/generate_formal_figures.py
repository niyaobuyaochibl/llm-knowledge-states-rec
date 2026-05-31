#!/usr/bin/env python3
"""Generate manuscript-facing figures from formal experiment outputs."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path("/root/temporal_popularity_pilot")
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from temporal_popularity.popularity import assign_buckets, static_popularity  # noqa: E402
from temporal_popularity.reporting import ensure_dirs, markdown_table  # noqa: E402

RESULTS = ROOT / "results"
FORMAL = RESULTS / "formal"
AGG = FORMAL / "aggregate"
FIGURES = FORMAL / "figures"

BUCKET_ORDER = ["Tail", "Mid", "Head"]
TEMP_BUCKET_ORDER = ["Tail", "Mid", "Head", "Dormant"]
METHOD_SHORT = {
    "Base": "Base",
    "StaticPopPenalty@0.1": "StaticPen",
    "StaticPopPenalty@0.01": "StaticPen",
    "TemporalPopPenalty@0.1": "TemporalPen",
    "TemporalPopPenalty@0.03": "TemporalPen",
    "StaticPopCal@1": "StaticCal",
    "TemporalPopCal@1": "TemporalCal",
    "XQuADTail@1": "XQuADTail",
}


@dataclass(frozen=True)
class FigureOutput:
    figure: str
    png: Path
    pdf: Path
    source: str
    note: str


def set_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        rc={
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def save_figure(fig: plt.Figure, stem: str) -> Tuple[Path, Path]:
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def load_transition(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame = frame.reindex(index=BUCKET_ORDER, columns=TEMP_BUCKET_ORDER).fillna(0.0)
    row_sums = frame.sum(axis=1).replace(0, np.nan)
    return frame.div(row_sums, axis=0) * 100.0


def plot_transition_heatmaps() -> FigureOutput:
    specs = [
        ("MovieLens-1M", "RecentPop@180d", RESULTS / "ml1m_e200" / "static_to_recent_transition_counts.csv"),
        ("MovieLens-1M", "DecayPop@180d", RESULTS / "ml1m_e200" / "static_to_decay_transition_counts.csv"),
        ("Yelp original reviews", "RecentPop@180d", RESULTS / "yelp_day1" / "static_to_recent_transition_counts.csv"),
        ("Yelp original reviews", "DecayPop@180d", RESULTS / "yelp_day1" / "static_to_decay_transition_counts.csv"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), sharex=False, sharey=False)
    for ax, (dataset, definition, path) in zip(axes.flat, specs):
        data = load_transition(path)
        sns.heatmap(
            data,
            ax=ax,
            cmap="YlGnBu",
            vmin=0,
            vmax=100,
            annot=True,
            fmt=".1f",
            cbar=ax is axes.flat[-1],
            cbar_kws={"label": "Row percentage"},
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(f"{dataset}\n{definition}")
        ax.set_xlabel("Temporal bucket")
        ax.set_ylabel("Static bucket")
    png, pdf = save_figure(fig, "figure1_bucket_transition_heatmaps")
    return FigureOutput(
        figure="Figure 1",
        png=png,
        pdf=pdf,
        source="ML-1M/Yelp static_to_{recent,decay}_transition_counts.csv",
        note="Row-normalized static-to-temporal bucket transition matrices.",
    )


def plot_gain_scatter() -> FigureOutput:
    table4 = pd.read_csv(AGG / "table4_tod_rfr.csv")
    frame = table4[
        (table4["Method"] != "ALL_METHOD_PAIRS")
        & (table4["TemporalDefinition"] == "Decay")
        & table4["StaticGain_mean"].notna()
        & table4["TemporalGain_mean"].notna()
    ].copy()
    datasets = ["MovieLens-1M", "Yelp original reviews"]
    metrics = ["ARP", "LTR", "PCE"]
    palette = {
        "StaticCal": "#4C78A8",
        "StaticPen": "#F58518",
        "TemporalCal": "#54A24B",
        "TemporalPen": "#B279A2",
        "XQuADTail": "#E45756",
    }
    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=(8.4, 8.0))
    legend_handles: Dict[str, plt.Line2D] = {}
    for row, metric in enumerate(metrics):
        for col, dataset in enumerate(datasets):
            ax = axes[row, col]
            sub = frame[(frame["Dataset"] == dataset) & (frame["Metric"] == metric)].copy()
            sub["MethodShort"] = sub["Method"].map(METHOD_SHORT).fillna(sub["Method"])
            if sub.empty:
                ax.set_axis_off()
                continue
            x = sub["StaticGain_mean"].to_numpy(float)
            y = sub["TemporalGain_mean"].to_numpy(float)
            lo = float(min(x.min(), y.min()))
            hi = float(max(x.max(), y.max()))
            pad = max((hi - lo) * 0.12, 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#666666", linewidth=1.0, linestyle="--")
            for _, point in sub.iterrows():
                label = point["MethodShort"]
                artist = ax.scatter(
                    point["StaticGain_mean"],
                    point["TemporalGain_mean"],
                    s=42,
                    color=palette.get(label, "#333333"),
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=3,
                )
                legend_handles.setdefault(label, artist)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(f"{dataset} - {metric}")
            ax.set_xlabel("Static gain")
            ax.set_ylabel("Decay temporal gain")
            ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
            if metric in {"LTR", "PCE"}:
                ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        loc="lower center",
        ncol=min(5, len(legend_handles)),
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    png = FIGURES / "figure2_static_vs_temporal_gain_scatter.png"
    pdf = FIGURES / "figure2_static_vs_temporal_gain_scatter.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return FigureOutput(
        figure="Figure 2",
        png=png,
        pdf=pdf,
        source="results/formal/aggregate/table4_tod_rfr.csv",
        note="Quality-oriented static gain versus DecayPop temporal gain; dashed line is y=x.",
    )


def plot_group_sensitivity() -> FigureOutput:
    table5 = pd.read_csv(AGG / "table5_group_sensitivity.csv")
    frame = table5[
        table5["Group"].isin(["niche", "mainstream"])
        & table5["PCE_Sensitivity_mean"].notna()
    ].copy()
    frame["MethodShort"] = frame["Method"].map(METHOD_SHORT).fillna(frame["Method"])
    method_order = ["Base", "StaticPen", "TemporalPen", "StaticCal", "TemporalCal", "XQuADTail"]
    datasets = ["MovieLens-1M", "Yelp original reviews"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=False)
    handles = labels = None
    for ax, dataset in zip(axes, datasets):
        sub = frame[frame["Dataset"] == dataset].copy()
        sns.barplot(
            data=sub,
            x="MethodShort",
            y="PCE_Sensitivity_mean",
            hue="Group",
            order=[m for m in method_order if m in set(sub["MethodShort"])],
            hue_order=["niche", "mainstream"],
            palette={"niche": "#4C78A8", "mainstream": "#F58518"},
            ax=ax,
            errorbar=None,
        )
        ax.set_title(dataset)
        ax.set_xlabel("")
        ax.set_ylabel("PCE temporal sensitivity")
        ax.tick_params(axis="x", rotation=30)
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
    if handles and labels:
        fig.legend(handles, labels, loc="upper center", ncol=2, title="", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = FIGURES / "figure3_niche_mainstream_sensitivity.png"
    pdf = FIGURES / "figure3_niche_mainstream_sensitivity.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return FigureOutput(
        figure="Figure 3",
        png=png,
        pdf=pdf,
        source="results/formal/aggregate/table5_group_sensitivity.csv",
        note="Niche versus mainstream PCE temporal sensitivity by method.",
    )


def recent_counts_at(events: pd.DataFrame, n_items: int, timestamp: int, window_days: int = 180) -> np.ndarray:
    start = timestamp - window_days * 24 * 60 * 60
    sub = events[(events["timestamp"] >= start) & (events["timestamp"] < timestamp)]
    return np.bincount(sub["iid"].to_numpy(np.int64), minlength=n_items).astype(np.float32)


def select_trajectory_items(events: pd.DataFrame, train: pd.DataFrame, n_items: int) -> pd.DataFrame:
    static_pop = static_popularity(train, n_items)
    static_bucket = assign_buckets(static_pop, static_pop, dormant_for_zero=False)
    end_time = int(events["timestamp"].max()) + 1
    final_recent = recent_counts_at(events, n_items, end_time)
    items = pd.DataFrame(
        {
            "iid": np.arange(n_items, dtype=np.int64),
            "static_pop": static_pop,
            "static_bucket": static_bucket,
            "final_recent": final_recent,
        }
    )
    active = items[items["static_pop"] > 0].copy()

    head_dormant = active[(active["static_bucket"] == 2) & (active["final_recent"] == 0)]
    if head_dormant.empty:
        head_dormant = active[active["static_bucket"] == 2].sort_values(["final_recent", "static_pop"], ascending=[True, False])
    else:
        head_dormant = head_dormant.sort_values("static_pop", ascending=False)

    tail_rising = active[active["static_bucket"] == 0].sort_values(["final_recent", "static_pop"], ascending=[False, True])
    stable_head = active[active["static_bucket"] == 2].sort_values(["final_recent", "static_pop"], ascending=[False, False])

    rows = [
        ("Static head, later dormant/low", head_dormant.iloc[0]),
        ("Static tail, later active", tail_rising.iloc[0]),
        ("Stable temporal head", stable_head.iloc[0]),
    ]
    selected = []
    used = set()
    for label, row in rows:
        iid = int(row["iid"])
        if iid in used:
            continue
        used.add(iid)
        selected.append(
            {
                "label": label,
                "iid": iid,
                "static_pop": float(row["static_pop"]),
                "final_recent": float(row["final_recent"]),
            }
        )
    return pd.DataFrame(selected)


def trajectory_series(events: pd.DataFrame, items: Sequence[int], snapshots: Sequence[int]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    n_items = int(events["iid"].max()) + 1
    for ts in snapshots:
        counts = recent_counts_at(events, n_items, int(ts))
        date = pd.to_datetime(int(ts), unit="s")
        for iid in items:
            rows.append({"timestamp": int(ts), "date": date, "iid": int(iid), "RecentPop180d": float(counts[int(iid)])})
    return pd.DataFrame(rows)


def plot_item_trajectories() -> FigureOutput:
    events_path = ROOT / "data" / "yelp_day1" / "all_events_log_observable.csv"
    train_path = ROOT / "data" / "yelp_day1" / "train.csv"
    events = pd.read_csv(events_path, usecols=["timestamp", "iid"])
    train = pd.read_csv(train_path, usecols=["iid"])
    n_items = int(max(events["iid"].max(), train["iid"].max())) + 1
    selected = select_trajectory_items(events, train, n_items)
    start = int(events["timestamp"].min())
    end = int(events["timestamp"].max()) + 1
    snapshots = np.linspace(start, end, 72, dtype=np.int64)
    series = trajectory_series(events, selected["iid"].tolist(), snapshots)
    series = series.merge(selected[["iid", "label"]], on="iid", how="left")

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    palette = {
        "Static head, later dormant/low": "#4C78A8",
        "Static tail, later active": "#E45756",
        "Stable temporal head": "#54A24B",
    }
    for label, sub in series.groupby("label", sort=False):
        iid = int(sub["iid"].iloc[0])
        meta = selected[selected["iid"] == iid].iloc[0]
        display = f"{label} (item {iid}, static={meta['static_pop']:.0f}, final recent={meta['final_recent']:.0f})"
        ax.plot(sub["date"], sub["RecentPop180d"], label=display, color=palette.get(label), linewidth=1.8)
    ax.set_title("Yelp item popularity trajectories")
    ax.set_xlabel("Time")
    ax.set_ylabel("RecentPop@180d")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=True, loc="upper left")
    png, pdf = save_figure(fig, "figure4_yelp_item_popularity_trajectories")

    selected.to_csv(FIGURES / "figure4_selected_items.csv", index=False)
    return FigureOutput(
        figure="Figure 4",
        png=png,
        pdf=pdf,
        source="data/yelp_day1/all_events_log_observable.csv and train.csv",
        note="Example item RecentPop@180d trajectories selected from static/temporal bucket disagreement patterns.",
    )


def write_manifest(outputs: Iterable[FigureOutput]) -> None:
    rows = [
        {
            "Figure": out.figure,
            "PNG": str(out.png.relative_to(ROOT)),
            "PDF": str(out.pdf.relative_to(ROOT)),
            "Source": out.source,
            "Note": out.note,
        }
        for out in outputs
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(FIGURES / "figure_manifest.csv", index=False)
    lines = [
        "# Formal Figure Manifest",
        "",
        markdown_table(frame),
        "",
        "All figures are generated by `scripts/generate_formal_figures.py`.",
    ]
    (FIGURES / "figure_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs(FIGURES)
    set_style()
    outputs = [
        plot_transition_heatmaps(),
        plot_gain_scatter(),
        plot_group_sensitivity(),
        plot_item_trajectories(),
    ]
    write_manifest(outputs)
    for out in outputs:
        print(f"Wrote {out.figure}: {out.png}", flush=True)
    print(f"Wrote figure manifest under {FIGURES}", flush=True)


if __name__ == "__main__":
    main()
