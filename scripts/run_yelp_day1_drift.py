#!/usr/bin/env python3
"""Day-1 temporal popularity drift audit for raw Yelp reviews.

This script reprocesses the original Yelp review JSON with timestamps:
1. Extracts user, business, and timestamp fields from raw JSONL.
2. Applies iterative 10-core filtering.
3. Builds leave-one-out train/validation/test splits by timestamp.
4. Computes static-vs-temporal bucket drift using weighted test-time snapshots.

The snapshot weighting is a scalability compromise for large Yelp: test timestamps
are split into equal-frequency bins, one temporal popularity state is computed at
the median timestamp of each bin, and transition counts are weighted by the number
of test users in that bin.
"""

import argparse
import calendar
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SECONDS_PER_DAY = 24 * 60 * 60
STATIC_BUCKET_NAMES = ["Tail", "Mid", "Head"]
TEMPORAL_BUCKET_NAMES = ["Tail", "Mid", "Head", "Dormant"]

USER_RE = re.compile(br'"user_id":"([^"]+)"')
ITEM_RE = re.compile(br'"business_id":"([^"]+)"')
DATE_RE = re.compile(br'"date":"([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("/root/autodl-tmp/yelp-raw/yelp_academic_dataset_review.json"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("/root/temporal_popularity_pilot/results/yelp_day1"))
    parser.add_argument("--figdir", type=Path, default=Path("/root/temporal_popularity_pilot/figures/yelp_day1"))
    parser.add_argument("--datadir", type=Path, default=Path("/root/temporal_popularity_pilot/data/yelp_day1"))
    parser.add_argument("--k-core", type=int, default=10)
    parser.add_argument("--window-days", type=int, default=180)
    parser.add_argument("--tau-days", type=int, default=180)
    parser.add_argument("--snapshot-count", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    return parser.parse_args()


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def parse_timestamp(date_bytes: bytes) -> int:
    year = int(date_bytes[0:4])
    month = int(date_bytes[5:7])
    day = int(date_bytes[8:10])
    hour = int(date_bytes[11:13])
    minute = int(date_bytes[14:16])
    second = int(date_bytes[17:19])
    return int(calendar.timegm((year, month, day, hour, minute, second)))


def extract_fields(line: bytes) -> Tuple[bytes, bytes, int]:
    user_match = USER_RE.search(line)
    item_match = ITEM_RE.search(line)
    date_match = DATE_RE.search(line)
    if user_match and item_match and date_match:
        return user_match.group(1), item_match.group(1), parse_timestamp(date_match.group(1))

    # Fallback for unexpected JSON key order or escaping.
    row = json.loads(line)
    timestamp = int(pd.Timestamp(row["date"], tz="UTC").timestamp())
    return row["user_id"].encode("utf-8"), row["business_id"].encode("utf-8"), timestamp


def get_or_add(mapping: Dict[bytes, int], key: bytes) -> int:
    value = mapping.get(key)
    if value is None:
        value = len(mapping)
        mapping[key] = value
    return value


def load_yelp_reviews(path: Path, chunk_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[bytes, int], Dict[bytes, int]]:
    user2id: Dict[bytes, int] = {}
    item2id: Dict[bytes, int] = {}
    user_parts: List[np.ndarray] = []
    item_parts: List[np.ndarray] = []
    time_parts: List[np.ndarray] = []
    users: List[int] = []
    items: List[int] = []
    times: List[int] = []
    total = 0

    with path.open("rb") as f:
        for line in f:
            user_raw, item_raw, timestamp = extract_fields(line)
            users.append(get_or_add(user2id, user_raw))
            items.append(get_or_add(item2id, item_raw))
            times.append(timestamp)
            total += 1
            if len(users) >= chunk_size:
                user_parts.append(np.asarray(users, dtype=np.int32))
                item_parts.append(np.asarray(items, dtype=np.int32))
                time_parts.append(np.asarray(times, dtype=np.int64))
                print(
                    f"extracted={total:,} users={len(user2id):,} items={len(item2id):,}",
                    flush=True,
                )
                users.clear()
                items.clear()
                times.clear()

    if users:
        user_parts.append(np.asarray(users, dtype=np.int32))
        item_parts.append(np.asarray(items, dtype=np.int32))
        time_parts.append(np.asarray(times, dtype=np.int64))
        print(f"extracted={total:,} users={len(user2id):,} items={len(item2id):,}", flush=True)

    return (
        np.concatenate(user_parts),
        np.concatenate(item_parts),
        np.concatenate(time_parts),
        user2id,
        item2id,
    )


def iterative_k_core(
    users: np.ndarray,
    items: np.ndarray,
    n_users_raw: int,
    n_items_raw: int,
    k: int,
) -> np.ndarray:
    mask = np.ones(len(users), dtype=bool)
    iteration = 0
    while True:
        iteration += 1
        active_users = users[mask]
        active_items = items[mask]
        user_counts = np.bincount(active_users, minlength=n_users_raw)
        item_counts = np.bincount(active_items, minlength=n_items_raw)
        new_mask = mask & (user_counts[users] >= k) & (item_counts[items] >= k)
        removed = int(mask.sum() - new_mask.sum())
        print(
            f"k-core iter={iteration} interactions={int(new_mask.sum()):,} removed={removed:,}",
            flush=True,
        )
        if removed == 0:
            return new_mask
        mask = new_mask


def make_dataframe(users: np.ndarray, items: np.ndarray, times: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uid_old": users[mask].astype(np.int32, copy=False),
            "iid_old": items[mask].astype(np.int32, copy=False),
            "timestamp": times[mask].astype(np.int64, copy=False),
        }
    ).sort_values(["uid_old", "timestamp", "iid_old"], kind="mergesort").reset_index(drop=True)


def split_leave_one_out(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    sizes = df.groupby("uid_old", sort=False)["iid_old"].transform("size").to_numpy(np.int32)
    ranks = df.groupby("uid_old", sort=False).cumcount().to_numpy(np.int32)
    train = df[ranks < sizes - 2].copy()
    val = df[ranks == sizes - 2].copy()
    test = df[ranks == sizes - 1].copy()

    train_items = set(train["iid_old"].unique().tolist())
    valid_val_users = set(val[val["iid_old"].isin(train_items)]["uid_old"].unique().tolist())
    valid_test_users = set(test[test["iid_old"].isin(train_items)]["uid_old"].unique().tolist())
    valid_users = valid_val_users.intersection(valid_test_users)

    train = train[train["uid_old"].isin(valid_users)].copy()
    val = val[val["uid_old"].isin(valid_users)].copy()
    test = test[test["uid_old"].isin(valid_users)].copy()

    train_items_arr = np.sort(train["iid_old"].unique())
    users_arr = np.sort(train["uid_old"].unique())
    user_map = {int(old): idx for idx, old in enumerate(users_arr)}
    item_map = {int(old): idx for idx, old in enumerate(train_items_arr)}

    def remap(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[frame["iid_old"].isin(item_map)].copy()
        out["uid"] = out["uid_old"].map(user_map).astype(np.int32)
        out["iid"] = out["iid_old"].map(item_map).astype(np.int32)
        return out.sort_values(["uid", "timestamp", "iid"], kind="mergesort").reset_index(drop=True)

    train = remap(train)
    val = remap(val)
    test = remap(test)
    all_events = remap(df[df["uid_old"].isin(valid_users)].copy())

    stats = {
        "users": int(len(user_map)),
        "items": int(len(item_map)),
        "train": int(len(train)),
        "val": int(len(val)),
        "test": int(len(test)),
        "all_events": int(len(all_events)),
    }
    return train, val, test, all_events, stats


def static_popularity(train: pd.DataFrame, n_items: int) -> np.ndarray:
    return np.bincount(train["iid"].to_numpy(np.int64), minlength=n_items).astype(np.float32)


def assign_buckets(pop: np.ndarray, static_pop: np.ndarray, dormant_for_zero: bool) -> np.ndarray:
    n_items = len(pop)
    buckets = np.full(n_items, 1, dtype=np.int8)
    if dormant_for_zero:
        active = np.flatnonzero(pop > 0)
        buckets[pop <= 0] = 3
    else:
        active = np.arange(n_items, dtype=np.int64)
    if len(active) == 0:
        return buckets

    order_local = np.lexsort((active, static_pop[active], pop[active]))
    ordered = active[order_local]
    n_active = len(ordered)
    n_tail = max(1, int(math.ceil(0.2 * n_active)))
    n_head = max(1, int(math.ceil(0.2 * n_active)))
    buckets[ordered[:n_tail]] = 0
    buckets[ordered[n_tail : max(n_tail, n_active - n_head)]] = 1
    buckets[ordered[max(n_tail, n_active - n_head) :]] = 2
    return buckets


def build_snapshots(test: pd.DataFrame, snapshot_count: int) -> Tuple[np.ndarray, np.ndarray]:
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


def compute_weighted_transition(
    all_events: pd.DataFrame,
    snapshot_times: np.ndarray,
    weights: np.ndarray,
    n_items: int,
    static_pop: np.ndarray,
    static_buckets: np.ndarray,
    window_seconds: int,
    tau_seconds: int,
    definition: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    events = all_events.sort_values(["timestamp", "iid"], kind="mergesort").reset_index(drop=True)
    event_times = events["timestamp"].to_numpy(np.int64)
    event_items = events["iid"].to_numpy(np.int64)

    order = np.argsort(snapshot_times, kind="mergesort")
    sorted_times = snapshot_times[order]
    sorted_weights = weights[order]

    recent_counts = np.zeros(n_items, dtype=np.float32)
    cumulative_counts = np.zeros(n_items, dtype=np.float32)
    decay_counts = np.zeros(n_items, dtype=np.float32)
    add_ptr = 0
    recent_left_ptr = 0
    decay_current_time: Optional[int] = None

    counts = np.zeros((3, 4), dtype=np.int64)
    bdr_num = 0
    total_pairs = 0
    tail_total = 0
    tail_exit = 0
    head_total = 0
    head_decay = 0
    zero_count = 0
    dormant_count = 0
    tail_mask = static_buckets == 0
    head_mask = static_buckets == 2

    for idx, (t, weight) in enumerate(zip(sorted_times, sorted_weights), start=1):
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
            cumulative_counts[item] += 1.0
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

        if definition == "recent":
            pop = recent_counts
        elif definition == "decay":
            pop = decay_counts
        elif definition == "cumulative":
            pop = cumulative_counts
        else:
            raise ValueError(definition)

        temporal_buckets = assign_buckets(pop, static_pop, dormant_for_zero=True)
        for static_bucket in range(3):
            mask = static_buckets == static_bucket
            row_counts = np.bincount(temporal_buckets[mask], minlength=4)
            counts[static_bucket, :] += row_counts[:4].astype(np.int64) * int(weight)

        bdr_num += int((temporal_buckets != static_buckets).sum()) * int(weight)
        zero_count += int((pop <= 0).sum()) * int(weight)
        dormant_count += int((temporal_buckets == 3).sum()) * int(weight)
        total_pairs += n_items * int(weight)
        tail_total += int(tail_mask.sum()) * int(weight)
        head_total += int(head_mask.sum()) * int(weight)
        tail_exit += int((temporal_buckets[tail_mask] != 0).sum()) * int(weight)
        head_decay += int((temporal_buckets[head_mask] != 2).sum()) * int(weight)

        if idx == 1 or idx % 25 == 0 or idx == len(sorted_times):
            print(
                f"{definition} snapshots={idx}/{len(sorted_times)} "
                f"time={pd.to_datetime(t_int, unit='s')} weight={int(weight)}",
                flush=True,
            )

    matrix = pd.DataFrame(counts, index=STATIC_BUCKET_NAMES, columns=TEMPORAL_BUCKET_NAMES)
    label = {
        "recent": "RecentPop",
        "decay": "DecayPop",
        "cumulative": "CumulativePop",
    }[definition]
    stats = {
        "Definition": label,
        "BDR": bdr_num / total_pairs,
        "TER": tail_exit / tail_total,
        "HDR": head_decay / head_total,
        "ZRPR": zero_count / total_pairs,
        "DormantPct": dormant_count / total_pairs,
        "WeightedPairs": int(total_pairs),
        "Snapshots": int(len(sorted_times)),
    }
    return matrix, stats


def plot_transition(matrix: pd.DataFrame, title: str, path: Path) -> None:
    row_norm = matrix.div(matrix.sum(axis=1), axis=0).fillna(0.0)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.imshow(row_norm.to_numpy(), cmap="Blues", vmin=0, vmax=max(0.01, row_norm.to_numpy().max()))
    ax.set_xticks(np.arange(len(row_norm.columns)), row_norm.columns)
    ax.set_yticks(np.arange(len(row_norm.index)), row_norm.index)
    ax.set_xlabel("Temporal bucket")
    ax.set_ylabel("Static bucket")
    ax.set_title(title)
    for i in range(row_norm.shape[0]):
        for j in range(row_norm.shape[1]):
            ax.text(j, i, f"{row_norm.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, float_digits: int = 6) -> str:
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.{float_digits}g}")
        else:
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(col) for col in formatted.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in formatted.values.tolist():
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_report(
    outdir: Path,
    stats: Dict[str, int],
    time_min: int,
    time_max: int,
    retained_ratio: float,
    drift_stats: pd.DataFrame,
    snapshot_count: int,
) -> None:
    criteria = {
        "BDR>=20%": bool((drift_stats["BDR"] >= 0.20).any()),
        "TailExitRate>=30%": bool((drift_stats["TER"] >= 0.30).any()),
        "NoGo_BDR<10%": bool((drift_stats["BDR"] < 0.10).all()),
        "RecentOnlyMechanicalZerosRisk": bool(
            (drift_stats.loc[drift_stats["Definition"].str.contains("Recent"), "ZRPR"].max() > 0.50)
            and (drift_stats.loc[drift_stats["Definition"].str.contains("Decay"), "BDR"].max() < 0.10)
        ),
    }
    (outdir / "day1_go_no_go_partial.json").write_text(json.dumps(criteria, indent=2), encoding="utf-8")

    lines = [
        "# Yelp Original Review Day-1 Drift Audit",
        "",
        "This run uses raw Yelp review JSON timestamps and rebuilds 10-core filtering. Drift is computed with weighted test-time snapshots for scalability.",
        "",
        "## Dataset",
        "",
        f"- Users: {stats['users']}",
        f"- Items: {stats['items']}",
        f"- Train / Val / Test interactions: {stats['train']} / {stats['val']} / {stats['test']}",
        f"- All retained log-observable events: {stats['all_events']}",
        f"- Time span: {pd.to_datetime(time_min, unit='s')} to {pd.to_datetime(time_max, unit='s')}",
        f"- Retained ratio after 10-core and split cleanup: {retained_ratio:.4f}",
        f"- Weighted snapshots: {snapshot_count}",
        "",
        "## Drift",
        "",
        markdown_table(drift_stats),
        "",
        "## Partial Day-1 Go/No-Go",
        "",
    ]
    for key, value in criteria.items():
        lines.append(f"- {key}: {value}")
    (outdir / "day1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs(args.outdir, args.figdir, args.datadir)

    print("Extracting raw Yelp review interactions...", flush=True)
    users, items, times, user2id, item2id = load_yelp_reviews(args.reviews, args.chunk_size)
    raw_interactions = len(users)
    raw_users = len(user2id)
    raw_items = len(item2id)
    print(
        f"raw interactions={raw_interactions:,} users={raw_users:,} items={raw_items:,}",
        flush=True,
    )

    print("Applying iterative 10-core filtering...", flush=True)
    core_mask = iterative_k_core(users, items, raw_users, raw_items, args.k_core)
    df = make_dataframe(users, items, times, core_mask)

    print("Building leave-one-out split...", flush=True)
    train, val, test, all_events, stats = split_leave_one_out(df)
    retained_ratio = (len(train) + len(val) + len(test)) / raw_interactions
    n_items = stats["items"]

    train.to_csv(args.datadir / "train.csv", index=False)
    val.to_csv(args.datadir / "val.csv", index=False)
    test.to_csv(args.datadir / "test.csv", index=False)
    all_events.to_csv(args.datadir / "all_events_log_observable.csv", index=False)
    pd.DataFrame([stats]).to_csv(args.outdir / "split_stats.csv", index=False)

    dataset_table = pd.DataFrame(
        [
            {
                "Dataset": "Yelp original reviews",
                "Users": stats["users"],
                "Items": stats["items"],
                "Interactions": len(train) + len(val) + len(test),
                "TimeSpan": f"{pd.to_datetime(all_events['timestamp'].min(), unit='s').date()} to {pd.to_datetime(all_events['timestamp'].max(), unit='s').date()}",
                "TimestampSource": "review JSON date field",
                "10CoreRetainedRatio": retained_ratio,
                "SnapshotWeighted": True,
                "Snapshots": args.snapshot_count,
            }
        ]
    )
    dataset_table.to_csv(args.outdir / "table1_dataset_timestamp_availability.csv", index=False)

    print("Computing weighted temporal drift snapshots...", flush=True)
    static_pop = static_popularity(train, n_items)
    static_buckets = assign_buckets(static_pop, static_pop, dormant_for_zero=False)
    snapshot_times, weights = build_snapshots(test, args.snapshot_count)
    pd.DataFrame({"timestamp": snapshot_times, "weight": weights}).to_csv(args.outdir / "snapshot_times.csv", index=False)

    recent_matrix, recent_stats = compute_weighted_transition(
        all_events,
        snapshot_times,
        weights,
        n_items,
        static_pop,
        static_buckets,
        args.window_days * SECONDS_PER_DAY,
        args.tau_days * SECONDS_PER_DAY,
        "recent",
    )
    recent_stats["Definition"] = f"RecentPop@{args.window_days}d"
    decay_matrix, decay_stats = compute_weighted_transition(
        all_events,
        snapshot_times,
        weights,
        n_items,
        static_pop,
        static_buckets,
        args.window_days * SECONDS_PER_DAY,
        args.tau_days * SECONDS_PER_DAY,
        "decay",
    )
    decay_stats["Definition"] = f"DecayPop@{args.tau_days}d"

    recent_matrix.to_csv(args.outdir / "static_to_recent_transition_counts.csv")
    decay_matrix.to_csv(args.outdir / "static_to_decay_transition_counts.csv")
    drift_stats = pd.DataFrame([recent_stats, decay_stats])
    drift_stats.to_csv(args.outdir / "table2_temporal_drift_stats.csv", index=False)
    plot_transition(
        recent_matrix,
        f"Yelp static to RecentPop@{args.window_days}d transition",
        args.figdir / "figure1_static_to_recent_transition.png",
    )
    plot_transition(
        decay_matrix,
        f"Yelp static to DecayPop@{args.tau_days}d transition",
        args.figdir / "figure1_static_to_decay_transition.png",
    )

    write_report(
        args.outdir,
        stats,
        int(all_events["timestamp"].min()),
        int(all_events["timestamp"].max()),
        retained_ratio,
        drift_stats,
        len(snapshot_times),
    )
    print(f"Done. Report: {args.outdir / 'day1_report.md'}", flush=True)


if __name__ == "__main__":
    main()
