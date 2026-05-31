#!/usr/bin/env python3
"""Exact-vs-snapshot temporal drift check on a Yelp test subset.

The main Yelp protocol uses weighted test-time snapshots for scalability. This
script estimates the approximation error by computing exact temporal buckets at
the true test timestamp for a stratified subset of users, then comparing that to
the same users evaluated via the 200-snapshot protocol.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_ml1m_minipilot import SECONDS_PER_DAY, assign_buckets, static_popularity  # noqa: E402
from run_yelp_day1_drift import compute_weighted_transition, markdown_table  # noqa: E402
from run_yelp_day2_base import build_snapshots, read_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default=Path("/root/temporal_popularity_pilot/data/yelp_day1"))
    parser.add_argument(
        "--outdir", type=Path, default=Path("/root/temporal_popularity_pilot/results/yelp_exact_subset_check")
    )
    parser.add_argument("--subset-size", type=int, default=5000)
    parser.add_argument("--strata", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot-count", type=int, default=200)
    parser.add_argument("--window-days", type=int, default=180)
    parser.add_argument("--tau-days", type=int, default=180)
    return parser.parse_args()


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def stratified_time_subset(test: pd.DataFrame, subset_size: int, strata: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ordered = test.sort_values(["timestamp", "uid"], kind="mergesort").reset_index(drop=True)
    edges = np.linspace(0, len(ordered), strata + 1, dtype=np.int64)
    per_stratum = max(1, subset_size // strata)
    selected_indices: List[np.ndarray] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        candidates = np.arange(left, right, dtype=np.int64)
        take = min(per_stratum, len(candidates))
        selected_indices.append(rng.choice(candidates, size=take, replace=False))

    selected = np.concatenate(selected_indices)
    if len(selected) < subset_size:
        remaining = np.setdiff1d(np.arange(len(ordered), dtype=np.int64), selected, assume_unique=False)
        top_up = rng.choice(remaining, size=min(subset_size - len(selected), len(remaining)), replace=False)
        selected = np.concatenate([selected, top_up])
    elif len(selected) > subset_size:
        selected = rng.choice(selected, size=subset_size, replace=False)

    subset = ordered.iloc[np.sort(selected)].copy().reset_index(drop=True)
    return subset


def subset_snapshot_weights(
    full_test: pd.DataFrame,
    subset: pd.DataFrame,
    n_users: int,
    snapshot_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    snapshot_times, _, user_snapshot = build_snapshots(full_test, n_users, snapshot_count)
    weights = np.bincount(user_snapshot[subset["uid"].to_numpy(np.int64)], minlength=len(snapshot_times))
    active = weights > 0
    return snapshot_times[active], weights[active].astype(np.int64)


def stats_row(stats: Dict[str, float], mode: str, kind: str, subset_size: int) -> Dict[str, object]:
    return {
        "Mode": mode,
        "Kind": kind,
        "SubsetUsers": subset_size,
        "BDR": float(stats["BDR"]),
        "TER": float(stats["TER"]),
        "HDR": float(stats["HDR"]),
        "ZRPR": float(stats["ZRPR"]),
        "DormantPct": float(stats["DormantPct"]),
        "WeightedPairs": int(stats["WeightedPairs"]),
        "Snapshots": int(stats["Snapshots"]),
    }


def write_report(outdir: Path, comparison: pd.DataFrame, deltas: pd.DataFrame) -> None:
    max_bdr_error = float(deltas["BDR_abs_error"].max())
    max_ter_error = float(deltas["TER_abs_error"].max())
    lines = [
        "# Yelp Exact Subset Check",
        "",
        "This check compares exact temporal buckets at each selected user's true test timestamp against the 200-snapshot approximation for the same users.",
        "",
        "## Exact Versus Snapshot",
        "",
        markdown_table(comparison),
        "",
        "## Absolute Errors",
        "",
        markdown_table(deltas),
        "",
        "## Interpretation",
        "",
        f"- Max BDR absolute error: {max_bdr_error:.6f}",
        f"- Max TER absolute error: {max_ter_error:.6f}",
        f"- Snapshot approximation acceptable at <0.01 BDR error: {max_bdr_error < 0.01}",
    ]
    outdir.joinpath("exact_subset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs(args.outdir)
    train, _, test, all_events = read_split(args.datadir)
    n_users = int(max(train["uid"].max(), test["uid"].max()) + 1)
    n_items = int(train["iid"].max() + 1)
    static_pop = static_popularity(train, n_items)
    static_buckets = assign_buckets(static_pop, static_pop, dormant_for_zero=False)

    subset = stratified_time_subset(test, args.subset_size, args.strata, args.seed)
    subset.to_csv(args.outdir / "subset_users.csv", index=False)
    exact_times = subset["timestamp"].to_numpy(np.int64)
    exact_weights = np.ones(len(subset), dtype=np.int64)
    snapshot_times, snapshot_weights = subset_snapshot_weights(test, subset, n_users, args.snapshot_count)
    pd.DataFrame({"timestamp": snapshot_times, "weight": snapshot_weights}).to_csv(
        args.outdir / "subset_snapshot_weights.csv", index=False
    )

    rows: List[Dict[str, object]] = []
    for mode, times, weights in [
        ("exact", exact_times, exact_weights),
        ("snapshot", snapshot_times, snapshot_weights),
    ]:
        print(f"{mode}: RecentPop@{args.window_days}d subset={len(subset)}", flush=True)
        _, recent_stats = compute_weighted_transition(
            all_events,
            times,
            weights,
            n_items,
            static_pop,
            static_buckets,
            args.window_days * SECONDS_PER_DAY,
            args.tau_days * SECONDS_PER_DAY,
            "recent",
        )
        rows.append(stats_row(recent_stats, mode, "RecentPop", len(subset)))

        print(f"{mode}: DecayPop@{args.tau_days}d subset={len(subset)}", flush=True)
        _, decay_stats = compute_weighted_transition(
            all_events,
            times,
            weights,
            n_items,
            static_pop,
            static_buckets,
            args.window_days * SECONDS_PER_DAY,
            args.tau_days * SECONDS_PER_DAY,
            "decay",
        )
        rows.append(stats_row(decay_stats, mode, "DecayPop", len(subset)))

    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.outdir / "exact_vs_snapshot_subset.csv", index=False)

    delta_rows: List[Dict[str, object]] = []
    for kind in ["RecentPop", "DecayPop"]:
        exact = comparison[(comparison["Kind"] == kind) & (comparison["Mode"] == "exact")].iloc[0]
        snapshot = comparison[(comparison["Kind"] == kind) & (comparison["Mode"] == "snapshot")].iloc[0]
        delta_rows.append(
            {
                "Kind": kind,
                "BDR_exact": float(exact["BDR"]),
                "BDR_snapshot": float(snapshot["BDR"]),
                "BDR_abs_error": abs(float(exact["BDR"]) - float(snapshot["BDR"])),
                "TER_exact": float(exact["TER"]),
                "TER_snapshot": float(snapshot["TER"]),
                "TER_abs_error": abs(float(exact["TER"]) - float(snapshot["TER"])),
                "HDR_exact": float(exact["HDR"]),
                "HDR_snapshot": float(snapshot["HDR"]),
                "HDR_abs_error": abs(float(exact["HDR"]) - float(snapshot["HDR"])),
                "ZRPR_exact": float(exact["ZRPR"]),
                "ZRPR_snapshot": float(snapshot["ZRPR"]),
                "ZRPR_abs_error": abs(float(exact["ZRPR"]) - float(snapshot["ZRPR"])),
            }
        )
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(args.outdir / "exact_vs_snapshot_errors.csv", index=False)
    write_report(args.outdir, comparison, deltas)
    print(f"Done. Report: {args.outdir / 'exact_subset_report.md'}", flush=True)


if __name__ == "__main__":
    main()
