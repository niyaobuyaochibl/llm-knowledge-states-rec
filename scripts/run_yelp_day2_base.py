#!/usr/bin/env python3
"""Yelp Day-2 Base-only LightGCN pilot with static vs temporal metrics."""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_ml1m_minipilot import (  # noqa: E402
    SECONDS_PER_DAY,
    assign_buckets,
    build_norm_adj,
    build_user_histories,
    popularity_percentiles,
    set_seed,
    static_popularity,
    topk_indices,
    train_lightgcn,
    user_groups,
)


@dataclass
class SplitForTraining:
    train: pd.DataFrame
    n_users: int
    n_items: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default=Path("/root/temporal_popularity_pilot/data/yelp_day1"))
    parser.add_argument("--outdir", type=Path, default=Path("/root/temporal_popularity_pilot/results/yelp_day2_base"))
    parser.add_argument("--figdir", type=Path, default=Path("/root/temporal_popularity_pilot/figures/yelp_day2_base"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--samples-per-epoch", type=int, default=400_000)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--snapshot-count", type=int, default=200)
    parser.add_argument("--window-days", type=int, default=180)
    parser.add_argument("--tau-days", type=int, default=180)
    return parser.parse_args()


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def read_split(datadir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usecols = ["uid", "iid", "timestamp"]
    dtypes = {"uid": np.int32, "iid": np.int32, "timestamp": np.int64}
    train = pd.read_csv(datadir / "train.csv", usecols=usecols, dtype=dtypes)
    val = pd.read_csv(datadir / "val.csv", usecols=usecols, dtype=dtypes)
    test = pd.read_csv(datadir / "test.csv", usecols=usecols, dtype=dtypes)
    all_events = pd.read_csv(datadir / "all_events_log_observable.csv", usecols=usecols, dtype=dtypes)
    return train, val, test, all_events


def build_exclude_lists(train_histories: Sequence[np.ndarray], val: pd.DataFrame, n_users: int) -> List[np.ndarray]:
    val_items: List[List[int]] = [[] for _ in range(n_users)]
    for uid, iid in val[["uid", "iid"]].itertuples(index=False):
        val_items[int(uid)].append(int(iid))
    out: List[np.ndarray] = []
    for uid in range(n_users):
        items = train_histories[uid].tolist()
        if val_items[uid]:
            items.extend(val_items[uid])
        out.append(np.asarray(sorted(set(items)), dtype=np.int64))
    return out


def build_snapshots(test: pd.DataFrame, n_users: int, snapshot_count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = test[["uid", "timestamp"]].sort_values(["timestamp", "uid"], kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    n_snapshots = min(snapshot_count, n)
    edges = np.linspace(0, n, n_snapshots + 1, dtype=np.int64)
    snapshot_times: List[int] = []
    weights: List[int] = []
    user_snapshot = np.zeros(n_users, dtype=np.int16)
    for snap_idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if right <= left:
            continue
        segment = ordered.iloc[left:right]
        snapshot_times.append(int(np.median(segment["timestamp"].to_numpy(np.int64))))
        weights.append(int(right - left))
        user_snapshot[segment["uid"].to_numpy(np.int64)] = snap_idx
    return np.asarray(snapshot_times, dtype=np.int64), np.asarray(weights, dtype=np.int64), user_snapshot


def build_temporal_snapshot_features(
    all_events: pd.DataFrame,
    snapshot_times: np.ndarray,
    n_items: int,
    static_pop: np.ndarray,
    window_seconds: int,
    tau_seconds: int,
) -> Dict[str, np.ndarray]:
    events = all_events.sort_values(["timestamp", "iid"], kind="mergesort").reset_index(drop=True)
    event_times = events["timestamp"].to_numpy(np.int64)
    event_items = events["iid"].to_numpy(np.int64)
    order = np.argsort(snapshot_times, kind="mergesort")
    sorted_times = snapshot_times[order]

    recent_counts = np.zeros(n_items, dtype=np.float32)
    decay_counts = np.zeros(n_items, dtype=np.float32)
    add_ptr = 0
    recent_left_ptr = 0
    decay_current_time = None

    recent_pop_sorted = np.zeros((len(snapshot_times), n_items), dtype=np.float32)
    decay_pop_sorted = np.zeros((len(snapshot_times), n_items), dtype=np.float32)
    recent_bucket_sorted = np.zeros((len(snapshot_times), n_items), dtype=np.int8)
    decay_bucket_sorted = np.zeros((len(snapshot_times), n_items), dtype=np.int8)
    recent_pct_sorted = np.zeros((len(snapshot_times), n_items), dtype=np.float32)
    decay_pct_sorted = np.zeros((len(snapshot_times), n_items), dtype=np.float32)

    for row, t in enumerate(sorted_times):
        t_int = int(t)
        if decay_current_time is None:
            decay_current_time = t_int
        elif t_int > decay_current_time:
            decay_counts *= math.exp(-(t_int - decay_current_time) / tau_seconds)
            decay_current_time = t_int

        add_start = add_ptr
        while add_ptr < len(event_times) and event_times[add_ptr] < t_int:
            item = event_items[add_ptr]
            recent_counts[item] += 1.0
            add_ptr += 1
        if add_ptr > add_start:
            new_items = event_items[add_start:add_ptr]
            new_times = event_times[add_start:add_ptr]
            contrib = np.exp(-(t_int - new_times) / tau_seconds).astype(np.float32)
            np.add.at(decay_counts, new_items, contrib)

        window_start = t_int - window_seconds
        while recent_left_ptr < add_ptr and event_times[recent_left_ptr] < window_start:
            recent_counts[event_items[recent_left_ptr]] -= 1.0
            recent_left_ptr += 1

        recent_pop_sorted[row] = recent_counts
        decay_pop_sorted[row] = decay_counts
        recent_bucket_sorted[row] = assign_buckets(recent_counts, static_pop, dormant_for_zero=True)
        decay_bucket_sorted[row] = assign_buckets(decay_counts, static_pop, dormant_for_zero=True)
        recent_pct_sorted[row] = popularity_percentiles(recent_counts, static_pop)
        decay_pct_sorted[row] = popularity_percentiles(decay_counts, static_pop)

        if row == 0 or (row + 1) % 25 == 0 or row + 1 == len(sorted_times):
            print(f"temporal snapshot features={row + 1}/{len(sorted_times)}", flush=True)

    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return {
        "recent_pop": recent_pop_sorted[inverse],
        "decay_pop": decay_pop_sorted[inverse],
        "recent_bucket": recent_bucket_sorted[inverse],
        "decay_bucket": decay_bucket_sorted[inverse],
        "recent_pct": recent_pct_sorted[inverse],
        "decay_pct": decay_pct_sorted[inverse],
    }


def median_or_zero(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.median(values))


def evaluate_base_full_ranking(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    test: pd.DataFrame,
    train_histories: Sequence[np.ndarray],
    exclude_lists: Sequence[np.ndarray],
    static_pop: np.ndarray,
    static_bucket: np.ndarray,
    static_pct: np.ndarray,
    temporal: Dict[str, np.ndarray],
    user_snapshot: np.ndarray,
    groups: Dict[int, str],
    topk: int,
    batch_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    test_sorted = test.sort_values("uid", kind="mergesort").reset_index(drop=True)
    eval_users = test_sorted["uid"].to_numpy(np.int64)
    targets = test_sorted["iid"].to_numpy(np.int64)
    n_items = item_emb.shape[0]
    item_emb_t = item_emb.T.astype(np.float32)
    recs = np.zeros((len(eval_users), topk), dtype=np.int64)
    rows: List[Dict[str, object]] = []

    static_hist_median = np.asarray(
        [median_or_zero(static_pct[train_histories[int(uid)]]) for uid in range(len(train_histories))],
        dtype=np.float32,
    )

    for start in range(0, len(eval_users), batch_size):
        end = min(start + batch_size, len(eval_users))
        batch_users = eval_users[start:end]
        scores = user_emb[batch_users].astype(np.float32) @ item_emb_t
        for local_row, uid in enumerate(batch_users):
            global_row = start + local_row
            row_scores = scores[local_row].copy()
            row_scores[exclude_lists[int(uid)]] = -np.inf
            rec = topk_indices(row_scores, topk)
            recs[global_row] = rec

            target = int(targets[global_row])
            hit_positions = np.flatnonzero(rec == target)
            hit = len(hit_positions) > 0
            ndcg = 1.0 / math.log2(int(hit_positions[0]) + 2) if hit else 0.0
            recall = 1.0 if hit else 0.0
            snap = int(user_snapshot[int(uid)])
            hist = train_histories[int(uid)]
            recent_pct = temporal["recent_pct"][snap]
            decay_pct = temporal["decay_pct"][snap]

            rows.append(
                {
                    "Method": "Base",
                    "uid": int(uid),
                    "Group": groups[int(uid)],
                    "NDCG@20": ndcg,
                    "Recall@20": recall,
                    "HitRate@20": recall,
                    "Static_ARP@20": float(np.mean(static_pop[rec])),
                    "Recent_ARP@20": float(np.mean(temporal["recent_pop"][snap, rec])),
                    "Decay_ARP@20": float(np.mean(temporal["decay_pop"][snap, rec])),
                    "Static_LTR@20": float(np.mean(static_bucket[rec] == 0)),
                    "Recent_LTR@20": float(np.mean(temporal["recent_bucket"][snap, rec] == 0)),
                    "Decay_LTR@20": float(np.mean(temporal["decay_bucket"][snap, rec] == 0)),
                    "Static_HeadRatio@20": float(np.mean(static_bucket[rec] == 2)),
                    "Recent_HeadRatio@20": float(np.mean(temporal["recent_bucket"][snap, rec] == 2)),
                    "Decay_HeadRatio@20": float(np.mean(temporal["decay_bucket"][snap, rec] == 2)),
                    "Static_PCE@20": float(abs(np.median(static_pct[rec]) - static_hist_median[int(uid)])),
                    "Recent_PCE@20": float(abs(np.median(recent_pct[rec]) - median_or_zero(recent_pct[hist]))),
                    "Decay_PCE@20": float(abs(np.median(decay_pct[rec]) - median_or_zero(decay_pct[hist]))),
                    "Static_SPS@20": float(np.median(static_pct[rec]) - static_hist_median[int(uid)]),
                    "Recent_SPS@20": float(np.median(recent_pct[rec]) - median_or_zero(recent_pct[hist])),
                    "Decay_SPS@20": float(np.median(decay_pct[rec]) - median_or_zero(decay_pct[hist])),
                }
            )
        if end == len(eval_users) or end % (batch_size * 50) == 0:
            print(f"evaluated users={end}/{len(eval_users)}", flush=True)

    user_df = pd.DataFrame(rows)
    numeric_cols = [col for col in user_df.columns if col not in {"Method", "uid", "Group"}]
    summary = user_df[numeric_cols].mean(numeric_only=True).to_frame().T
    summary.insert(0, "Method", "Base")
    summary["Users"] = len(user_df)
    return summary, user_df, recs


def group_summary(user_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for group in ["niche", "mainstream", "balanced"]:
        sub = user_df[user_df["Group"] == group]
        if sub.empty:
            continue
        rows.append(
            {
                "Method": "Base",
                "Group": group,
                "Users": int(sub["uid"].nunique()),
                "Static_PCE": float(sub["Static_PCE@20"].mean()),
                "Recent_PCE": float(sub["Recent_PCE@20"].mean()),
                "Decay_PCE": float(sub["Decay_PCE@20"].mean()),
                "Recent_PCE_Change": float((sub["Recent_PCE@20"] - sub["Static_PCE@20"]).mean()),
                "Decay_PCE_Change": float((sub["Decay_PCE@20"] - sub["Static_PCE@20"]).mean()),
                "Recent_PCE_Sensitivity": float((sub["Recent_PCE@20"] - sub["Static_PCE@20"]).abs().mean()),
                "Decay_PCE_Sensitivity": float((sub["Decay_PCE@20"] - sub["Static_PCE@20"]).abs().mean()),
                "Static_LTR": float(sub["Static_LTR@20"].mean()),
                "Recent_LTR": float(sub["Recent_LTR@20"].mean()),
                "Decay_LTR": float(sub["Decay_LTR@20"].mean()),
                "Recent_LTR_Shrinkage": float(sub["Static_LTR@20"].mean() - sub["Recent_LTR@20"].mean()),
                "Decay_LTR_Shrinkage": float(sub["Static_LTR@20"].mean() - sub["Decay_LTR@20"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    indexed = out.set_index("Group")
    if {"niche", "mainstream"}.issubset(indexed.index):
        gap = {
            "Method": "Base",
            "Group": "GTSG_niche_minus_mainstream",
            "Recent_PCE_Sensitivity": float(
                indexed.loc["niche", "Recent_PCE_Sensitivity"]
                - indexed.loc["mainstream", "Recent_PCE_Sensitivity"]
            ),
            "Decay_PCE_Sensitivity": float(
                indexed.loc["niche", "Decay_PCE_Sensitivity"]
                - indexed.loc["mainstream", "Decay_PCE_Sensitivity"]
            ),
            "Recent_LTR_Shrinkage": float(
                indexed.loc["niche", "Recent_LTR_Shrinkage"] - indexed.loc["mainstream", "Recent_LTR_Shrinkage"]
            ),
            "Decay_LTR_Shrinkage": float(
                indexed.loc["niche", "Decay_LTR_Shrinkage"] - indexed.loc["mainstream", "Decay_LTR_Shrinkage"]
            ),
        }
        out = pd.concat([out, pd.DataFrame([gap])], ignore_index=True, sort=False)
    return out


def markdown_table(df: pd.DataFrame, float_digits: int = 6) -> str:
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.{float_digits}g}")
        else:
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else str(x))
    lines = [
        "| " + " | ".join(str(col) for col in formatted.columns) + " |",
        "| " + " | ".join(["---"] * len(formatted.columns)) + " |",
    ]
    for row in formatted.values.tolist():
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def plot_group_sensitivity(group_df: pd.DataFrame, path: Path) -> None:
    plot_df = group_df[group_df["Group"].isin(["niche", "mainstream"])]
    labels = ["Recent_PCE_Sensitivity", "Decay_PCE_Sensitivity"]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    for offset, group in [(-width / 2, "niche"), (width / 2, "mainstream")]:
        sub = plot_df[plot_df["Group"] == group].iloc[0]
        ax.bar(x + offset, [sub[label] for label in labels], width, label=group)
    ax.set_xticks(x, ["Recent", "Decay"])
    ax.set_ylabel("Temporal PCE sensitivity")
    ax.set_title("Yelp Base group temporal sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_report(outdir: Path, summary: pd.DataFrame, group_df: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Yelp Day-2 Base-Only LightGCN",
        "",
        "This run trains only the Base LightGCN recommender and evaluates the same full-ranking recommendations under static and snapshot-weighted temporal popularity metrics.",
        "",
        f"- Seed: {args.seed}",
        f"- Epochs: {args.epochs}",
        f"- Samples per epoch: {args.samples_per_epoch}",
        f"- Snapshots: {args.snapshot_count}",
        "",
        "## Metrics",
        "",
        markdown_table(summary),
        "",
        "## Group Sensitivity",
        "",
        markdown_table(group_df),
    ]
    outdir.joinpath("day2_base_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs(args.outdir, args.figdir)

    print("Loading Yelp Day-1 split...", flush=True)
    train, val, test, all_events = read_split(args.datadir)
    n_users = int(max(train["uid"].max(), val["uid"].max(), test["uid"].max()) + 1)
    n_items = int(train["iid"].max() + 1)
    split = SplitForTraining(train=train, n_users=n_users, n_items=n_items)
    print(f"users={n_users:,} items={n_items:,} train={len(train):,}", flush=True)

    static_pop = static_popularity(train, n_items)
    static_bucket = assign_buckets(static_pop, static_pop, dormant_for_zero=False)
    static_pct = popularity_percentiles(static_pop, static_pop)
    train_histories = build_user_histories(train, n_users)
    exclude_lists = build_exclude_lists(train_histories, val, n_users)
    groups = user_groups(train_histories, static_pct)

    snapshot_times, weights, user_snapshot = build_snapshots(test, n_users, args.snapshot_count)
    pd.DataFrame({"timestamp": snapshot_times, "weight": weights}).to_csv(args.outdir / "snapshot_times.csv", index=False)
    print("Building temporal snapshot metric features...", flush=True)
    temporal = build_temporal_snapshot_features(
        all_events,
        snapshot_times,
        n_items,
        static_pop,
        args.window_days * SECONDS_PER_DAY,
        args.tau_days * SECONDS_PER_DAY,
    )

    print("Training Base LightGCN...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_adj = build_norm_adj(train, n_users, n_items, device)
    model = train_lightgcn(split, norm_adj, args, device)
    model.eval()
    with torch.no_grad():
        user_emb_t, item_emb_t = model.propagate(norm_adj)
    user_emb = user_emb_t.detach().cpu().numpy().astype(np.float32)
    item_emb = item_emb_t.detach().cpu().numpy().astype(np.float32)
    torch.save(model.state_dict(), args.outdir / "lightgcn_base_seed42.pt")

    print("Evaluating full-ranking Base recommendations...", flush=True)
    summary, user_df, recs = evaluate_base_full_ranking(
        user_emb,
        item_emb,
        test,
        train_histories,
        exclude_lists,
        static_pop,
        static_bucket,
        static_pct,
        temporal,
        user_snapshot,
        groups,
        args.topk,
        args.eval_batch_size,
    )
    group_df = group_summary(user_df)
    summary.to_csv(args.outdir / "table3_base_static_vs_temporal_evaluation.csv", index=False)
    user_df.to_csv(args.outdir / "test_user_level_base_metrics.csv", index=False)
    group_df.to_csv(args.outdir / "table5_base_group_temporal_sensitivity.csv", index=False)
    np.save(args.outdir / "recs_base.npy", recs)
    plot_group_sensitivity(group_df, args.figdir / "figure3_base_group_temporal_sensitivity.png")
    write_report(args.outdir, summary, group_df, args)
    print(f"Done. Report: {args.outdir / 'day2_base_report.md'}", flush=True)


if __name__ == "__main__":
    main()
