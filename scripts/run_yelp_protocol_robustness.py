#!/usr/bin/env python3
"""Yelp temporal-popularity protocol robustness checks.

This is a formal-prep script, not a full experiment launch. It reuses the
Yelp Day-1 split and checks whether the drift signal is robust to:

1. Recent/decay windows: 90, 180, 365 days.
2. Weighted snapshot counts: 50, 100, 200, 400.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_ml1m_minipilot import SECONDS_PER_DAY, assign_buckets, static_popularity  # noqa: E402
from run_yelp_day1_drift import compute_weighted_transition, markdown_table  # noqa: E402
from run_yelp_day2_base import read_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default=Path("/root/temporal_popularity_pilot/data/yelp_day1"))
    parser.add_argument(
        "--outdir", type=Path, default=Path("/root/temporal_popularity_pilot/results/yelp_protocol_robustness")
    )
    parser.add_argument("--windows", type=int, nargs="+", default=[90, 180, 365])
    parser.add_argument("--snapshot-counts", type=int, nargs="+", default=[50, 100, 200, 400])
    return parser.parse_args()


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def build_snapshots(test: pd.DataFrame, snapshot_count: int) -> tuple[np.ndarray, np.ndarray]:
    times = np.sort(test["timestamp"].to_numpy(np.int64), kind="mergesort")
    n = len(times)
    n_snapshots = min(snapshot_count, n)
    edges = np.linspace(0, n, n_snapshots + 1, dtype=np.int64)
    snapshot_times: List[int] = []
    weights: List[int] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        segment = times[left:right]
        snapshot_times.append(int(np.median(segment)))
        weights.append(int(right - left))
    return np.asarray(snapshot_times, dtype=np.int64), np.asarray(weights, dtype=np.int64)


def add_metadata(stats: Dict[str, float], kind: str, days: int, snapshot_count: int) -> Dict[str, float]:
    out = dict(stats)
    out["Kind"] = kind
    out["Days"] = days
    out["SnapshotCount"] = snapshot_count
    return out


def summarize_against_main(df: pd.DataFrame, main_snapshot_count: int = 200, main_days: int = 180) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["Kind", "Days", "SnapshotCount"], keep="first").copy()
    main = df[(df["SnapshotCount"] == main_snapshot_count) & (df["Days"] == main_days)].copy()
    rows: List[Dict[str, object]] = []
    for _, main_row in main.iterrows():
        kind = main_row["Kind"]
        comparable = df[df["Kind"] == kind]
        for _, row in comparable.iterrows():
            rows.append(
                {
                    "Kind": kind,
                    "Days": int(row["Days"]),
                    "SnapshotCount": int(row["SnapshotCount"]),
                    "BDR": float(row["BDR"]),
                    "TER": float(row["TER"]),
                    "HDR": float(row["HDR"]),
                    "ZRPR": float(row["ZRPR"]),
                    "BDR_delta_vs_180d_200snap": float(row["BDR"] - main_row["BDR"]),
                    "TER_delta_vs_180d_200snap": float(row["TER"] - main_row["TER"]),
                }
            )
    return pd.DataFrame(rows)


def write_report(outdir: Path, window_df: pd.DataFrame, snapshot_df: pd.DataFrame, combined_summary: pd.DataFrame) -> None:
    robust_decay = bool((window_df[(window_df["Kind"] == "DecayPop") & (window_df["BDR"] >= 0.20)]).shape[0] == 3)
    robust_recent = bool((window_df[(window_df["Kind"] == "RecentPop") & (window_df["BDR"] >= 0.20)]).shape[0] == 3)
    snapshot_stable = bool(
        snapshot_df.groupby("Kind")["BDR"].agg(lambda s: float(s.max() - s.min())).max() < 0.05
    )
    lines = [
        "# Yelp Protocol Robustness",
        "",
        "This check reuses the Yelp Day-1 split and does not train recommenders.",
        "",
        "## Window Robustness",
        "",
        markdown_table(window_df),
        "",
        "## Snapshot-Count Robustness",
        "",
        markdown_table(snapshot_df),
        "",
        "## Deltas Versus Main Setting",
        "",
        markdown_table(combined_summary),
        "",
        "## Interpretation",
        "",
        f"- RecentPop BDR >= 20% for all 90/180/365 windows: {robust_recent}",
        f"- DecayPop BDR >= 20% for all 90/180/365 taus: {robust_decay}",
        f"- Snapshot-count BDR range < 0.05 for every kind: {snapshot_stable}",
    ]
    outdir.joinpath("protocol_robustness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs(args.outdir)
    train, _, test, all_events = read_split(args.datadir)
    n_items = int(train["iid"].max() + 1)
    static_pop = static_popularity(train, n_items)
    static_buckets = assign_buckets(static_pop, static_pop, dormant_for_zero=False)

    print("Running 90/180/365 window robustness with 200 snapshots...", flush=True)
    main_snapshot_times, main_weights = build_snapshots(test, 200)
    window_rows: List[Dict[str, float]] = []
    for days in args.windows:
        print(f"window robustness: RecentPop@{days}d", flush=True)
        _, recent_stats = compute_weighted_transition(
            all_events,
            main_snapshot_times,
            main_weights,
            n_items,
            static_pop,
            static_buckets,
            days * SECONDS_PER_DAY,
            days * SECONDS_PER_DAY,
            "recent",
        )
        window_rows.append(add_metadata(recent_stats, "RecentPop", days, 200))

        print(f"window robustness: DecayPop@{days}d", flush=True)
        _, decay_stats = compute_weighted_transition(
            all_events,
            main_snapshot_times,
            main_weights,
            n_items,
            static_pop,
            static_buckets,
            days * SECONDS_PER_DAY,
            days * SECONDS_PER_DAY,
            "decay",
        )
        window_rows.append(add_metadata(decay_stats, "DecayPop", days, 200))
    window_df = pd.DataFrame(window_rows)
    window_df.to_csv(args.outdir / "window_robustness.csv", index=False)

    print("Running 50/100/200/400 snapshot-count robustness at 180d...", flush=True)
    snapshot_rows: List[Dict[str, float]] = []
    for count in args.snapshot_counts:
        snapshot_times, weights = build_snapshots(test, count)
        print(f"snapshot robustness: RecentPop@180d snapshots={count}", flush=True)
        _, recent_stats = compute_weighted_transition(
            all_events,
            snapshot_times,
            weights,
            n_items,
            static_pop,
            static_buckets,
            180 * SECONDS_PER_DAY,
            180 * SECONDS_PER_DAY,
            "recent",
        )
        snapshot_rows.append(add_metadata(recent_stats, "RecentPop", 180, count))

        print(f"snapshot robustness: DecayPop@180d snapshots={count}", flush=True)
        _, decay_stats = compute_weighted_transition(
            all_events,
            snapshot_times,
            weights,
            n_items,
            static_pop,
            static_buckets,
            180 * SECONDS_PER_DAY,
            180 * SECONDS_PER_DAY,
            "decay",
        )
        snapshot_rows.append(add_metadata(decay_stats, "DecayPop", 180, count))
    snapshot_df = pd.DataFrame(snapshot_rows)
    snapshot_df.to_csv(args.outdir / "snapshot_count_robustness.csv", index=False)

    combined = pd.concat([window_df, snapshot_df], ignore_index=True, sort=False).drop_duplicates(
        subset=["Kind", "Days", "SnapshotCount"], keep="first"
    )
    combined_summary = summarize_against_main(combined)
    combined_summary.to_csv(args.outdir / "robustness_deltas_vs_main.csv", index=False)
    write_report(args.outdir, window_df, snapshot_df, combined_summary)
    print(f"Done. Report: {args.outdir / 'protocol_robustness_report.md'}", flush=True)


if __name__ == "__main__":
    main()
