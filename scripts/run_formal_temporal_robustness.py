#!/usr/bin/env python3
"""Robustness appendix for train-only and cumulative temporal popularity."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/root/temporal_popularity_pilot")
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from temporal_popularity.data import (  # noqa: E402
    activity_controlled_user_groups,
    build_user_histories,
    infer_shape,
    read_interaction_split,
)
from temporal_popularity.popularity import (  # noqa: E402
    SECONDS_PER_DAY,
    assign_buckets,
    popularity_percentiles,
    static_popularity,
)
from temporal_popularity.reporting import ensure_dirs, markdown_table  # noqa: E402
from temporal_popularity.snapshots import user_snapshot_map  # noqa: E402
from temporal_popularity.temporal import build_temporal_snapshot_features  # noqa: E402

FORMAL = ROOT / "results" / "formal"
OUT = FORMAL / "robustness"

DATASET_DIRS = [FORMAL / "ml1m", FORMAL / "yelp"]
DATASET_LABELS = {
    "ml1m": "MovieLens-1M",
    "yelp_original_reviews": "Yelp original reviews",
}
TEMPORAL_DEFINITIONS = ["Recent", "Decay", "Cumulative"]
METHOD_METRICS = ["ARP", "LTR", "PCE"]


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def formal_seed_dirs() -> Iterable[Path]:
    for dataset_dir in DATASET_DIRS:
        if not dataset_dir.exists():
            continue
        for seed_dir in sorted(dataset_dir.glob("seed*")):
            if seed_dir.is_dir():
                yield seed_dir


def dataset_label(config: Mapping[str, object], seed_dir: Path) -> str:
    dataset = str(config.get("dataset") or seed_dir.parent.name)
    return DATASET_LABELS.get(dataset, dataset)


def seed_value(config: Mapping[str, object], seed_dir: Path) -> int:
    if "seed" in config:
        return int(config["seed"])
    return int(seed_dir.name.replace("seed", ""))


def build_eval_snapshots(
    eval_frame: pd.DataFrame,
    n_users: int,
    config: Mapping[str, object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    evaluation = config["evaluation"]
    protocol = evaluation["temporal_protocol"]
    if protocol == "exact_per_user_timestamp":
        ordered = eval_frame.sort_values("uid", kind="mergesort").reset_index(drop=True)
        snapshot_times = ordered["timestamp"].to_numpy(np.int64)
        weights = np.ones(len(snapshot_times), dtype=np.int64)
        user_snapshot = np.zeros(n_users, dtype=np.int32)
        user_snapshot[ordered["uid"].to_numpy(np.int64)] = np.arange(len(ordered), dtype=np.int32)
    elif protocol == "weighted_test_time_snapshots":
        snapshot_times, weights, user_snapshot = user_snapshot_map(
            eval_frame, n_users, int(evaluation["snapshot_count"])
        )
    else:
        raise ValueError(f"Unsupported temporal protocol: {protocol}")
    snapshot_table = pd.DataFrame({"timestamp": snapshot_times, "weight": weights})
    return snapshot_times, weights, user_snapshot, snapshot_table


def build_cumulative_snapshot_features(
    events: pd.DataFrame,
    snapshot_times: np.ndarray,
    n_items: int,
    static_pop: np.ndarray,
    progress_every: int = 25,
) -> Dict[str, np.ndarray]:
    ordered_events = events.sort_values(["timestamp", "iid"], kind="mergesort").reset_index(drop=True)
    event_times = ordered_events["timestamp"].to_numpy(np.int64)
    event_items = ordered_events["iid"].to_numpy(np.int64)
    order = np.argsort(snapshot_times, kind="mergesort")
    sorted_times = snapshot_times[order]

    counts = np.zeros(n_items, dtype=np.float32)
    pop_sorted = np.zeros((len(snapshot_times), n_items), dtype=np.float32)
    bucket_sorted = np.zeros((len(snapshot_times), n_items), dtype=np.int8)
    pct_sorted = np.zeros((len(snapshot_times), n_items), dtype=np.float32)
    add_ptr = 0
    for row, t in enumerate(sorted_times):
        t_int = int(t)
        add_start = add_ptr
        while add_ptr < len(event_times) and event_times[add_ptr] < t_int:
            add_ptr += 1
        if add_ptr > add_start:
            np.add.at(counts, event_items[add_start:add_ptr], 1.0)
        pop_sorted[row] = counts
        bucket_sorted[row] = assign_buckets(counts, static_pop, dormant_for_zero=True)
        pct_sorted[row] = popularity_percentiles(counts, static_pop)
        if row == 0 or (row + 1) % progress_every == 0 or row + 1 == len(sorted_times):
            print(f"cumulative snapshot features={row + 1}/{len(sorted_times)}", flush=True)

    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return {
        "pop": pop_sorted[inverse],
        "bucket": bucket_sorted[inverse],
        "pct": pct_sorted[inverse],
    }


def rename_temporal_features(features: Mapping[str, np.ndarray], definition: str) -> Dict[str, np.ndarray]:
    if definition == "Recent":
        return {
            "pop": features["recent_pop"],
            "bucket": features["recent_bucket"],
            "pct": features["recent_pct"],
        }
    if definition == "Decay":
        return {
            "pop": features["decay_pop"],
            "bucket": features["decay_bucket"],
            "pct": features["decay_pct"],
        }
    if definition == "Cumulative":
        return {
            "pop": features["pop"],
            "bucket": features["bucket"],
            "pct": features["pct"],
        }
    raise ValueError(definition)


def weighted_drift_rows(
    dataset: str,
    protocol: str,
    definition: str,
    static_bucket: np.ndarray,
    temporal: Mapping[str, np.ndarray],
    weights: np.ndarray,
) -> Dict[str, object]:
    temporal_bucket = temporal["bucket"]
    temporal_pop = temporal["pop"]
    item_count = len(static_bucket)
    weight_float = weights.astype(np.float64)
    denom = float(item_count * weight_float.sum())
    static = static_bucket[None, :]
    tail = static_bucket == 0
    head = static_bucket == 2
    tail_denom = float(tail.sum() * weight_float.sum())
    head_denom = float(head.sum() * weight_float.sum())
    return {
        "Dataset": dataset,
        "TemporalProtocol": protocol,
        "Definition": f"{definition}Pop@180d" if definition in {"Recent", "Decay"} else "CumulativePop@t",
        "BDR": float(((temporal_bucket != static) * weight_float[:, None]).sum() / denom),
        "TER": float(((temporal_bucket[:, tail] != 0) * weight_float[:, None]).sum() / tail_denom),
        "HDR": float(((temporal_bucket[:, head] != 2) * weight_float[:, None]).sum() / head_denom),
        "ZRPR": float(((temporal_pop <= 0) * weight_float[:, None]).sum() / denom),
        "DormantPct": float(((temporal_bucket == 3) * weight_float[:, None]).sum() / denom),
        "WeightedPairs": denom,
        "Snapshots": int(len(weights)),
    }


def rec_file_for_method(seed_dir: Path, method: str) -> Path:
    safe = method.replace("@", "_").replace(".", "p")
    return seed_dir / f"recs_{safe}.npy"


def median_or_zero(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else 0.0


def compute_method_rows(
    dataset: str,
    seed: int,
    seed_dir: Path,
    methods: Sequence[str],
    test: pd.DataFrame,
    train_histories: Sequence[np.ndarray],
    groups: Mapping[int, str],
    user_snapshot: np.ndarray,
    static_pop: np.ndarray,
    static_bucket: np.ndarray,
    static_pct: np.ndarray,
    temporal: Mapping[str, np.ndarray],
    protocol: str,
    definition: str,
    existing_summary: pd.DataFrame,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    eval_sorted = test.sort_values("uid", kind="mergesort").reset_index(drop=True)
    eval_users = eval_sorted["uid"].to_numpy(np.int64)
    static_hist_median = np.asarray(
        [
            median_or_zero(static_pct[train_histories[int(uid)]])
            for uid in range(len(train_histories))
        ],
        dtype=np.float32,
    )

    method_rows: List[Dict[str, object]] = []
    group_rows: List[Dict[str, object]] = []
    summary_indexed = existing_summary.set_index("Method")

    for method in methods:
        rec_path = rec_file_for_method(seed_dir, method)
        if not rec_path.exists():
            print(f"Skipping missing recs: {rec_path}", flush=True)
            continue
        recs = np.load(rec_path)
        totals = defaultdict(float)
        group_totals: Dict[str, Dict[str, float]] = {
            group: defaultdict(float) for group in ["niche", "mainstream", "balanced"]
        }
        group_counts = defaultdict(int)

        for row_idx, uid_np in enumerate(eval_users):
            uid = int(uid_np)
            rec = recs[row_idx]
            hist = train_histories[uid]
            snap = int(user_snapshot[uid])
            pop = temporal["pop"][snap]
            bucket = temporal["bucket"][snap]
            pct = temporal["pct"][snap]

            static_arp = float(np.mean(static_pop[rec]))
            temporal_arp = float(np.mean(pop[rec]))
            static_ltr = float(np.mean(static_bucket[rec] == 0))
            temporal_ltr = float(np.mean(bucket[rec] == 0))
            static_head = float(np.mean(static_bucket[rec] == 2))
            temporal_head = float(np.mean(bucket[rec] == 2))
            static_pce = float(abs(np.median(static_pct[rec]) - static_hist_median[uid]))
            temporal_pce = float(abs(np.median(pct[rec]) - median_or_zero(pct[hist])))

            values = {
                "Static_ARP@20": static_arp,
                "Temporal_ARP@20": temporal_arp,
                "Static_LTR@20": static_ltr,
                "Temporal_LTR@20": temporal_ltr,
                "Static_HeadRatio@20": static_head,
                "Temporal_HeadRatio@20": temporal_head,
                "Static_PCE@20": static_pce,
                "Temporal_PCE@20": temporal_pce,
            }
            for key, value in values.items():
                totals[key] += value

            group = groups[uid]
            group_counts[group] += 1
            group_totals[group]["Static_PCE"] += static_pce
            group_totals[group]["Temporal_PCE"] += temporal_pce
            group_totals[group]["PCE_Sensitivity"] += abs(temporal_pce - static_pce)
            group_totals[group]["Static_LTR"] += static_ltr
            group_totals[group]["Temporal_LTR"] += temporal_ltr

        users = len(eval_users)
        existing = summary_indexed.loc[method] if method in summary_indexed.index else pd.Series(dtype=float)
        row = {
            "Dataset": dataset,
            "Seed": seed,
            "TemporalProtocol": protocol,
            "TemporalDefinition": definition,
            "Method": method,
            "Users": users,
            "NDCG@20": float(existing.get("NDCG@20", np.nan)),
            "Recall@20": float(existing.get("Recall@20", np.nan)),
        }
        for key, total in totals.items():
            row[key] = total / users
        method_rows.append(row)

        for group in ["niche", "mainstream", "balanced"]:
            count = group_counts[group]
            if count == 0:
                continue
            group_row = {
                "Dataset": dataset,
                "Seed": seed,
                "TemporalProtocol": protocol,
                "TemporalDefinition": definition,
                "Method": method,
                "Group": group,
                "Users": count,
                "Static_PCE": group_totals[group]["Static_PCE"] / count,
                "Temporal_PCE": group_totals[group]["Temporal_PCE"] / count,
                "Temporal_PCE_Change": (
                    group_totals[group]["Temporal_PCE"] - group_totals[group]["Static_PCE"]
                )
                / count,
                "PCE_Sensitivity": group_totals[group]["PCE_Sensitivity"] / count,
                "Static_LTR": group_totals[group]["Static_LTR"] / count,
                "Temporal_LTR": group_totals[group]["Temporal_LTR"] / count,
                "LTR_Shrinkage": (
                    group_totals[group]["Static_LTR"] - group_totals[group]["Temporal_LTR"]
                )
                / count,
            }
            group_rows.append(group_row)

        sub = pd.DataFrame([row for row in group_rows if row["Method"] == method]).set_index("Group")
        if {"niche", "mainstream"}.issubset(sub.index):
            group_rows.append(
                {
                    "Dataset": dataset,
                    "Seed": seed,
                    "TemporalProtocol": protocol,
                    "TemporalDefinition": definition,
                    "Method": method,
                    "Group": "GTSG_niche_minus_mainstream",
                    "PCE_Sensitivity": float(
                        sub.loc["niche", "PCE_Sensitivity"] - sub.loc["mainstream", "PCE_Sensitivity"]
                    ),
                    "LTR_Shrinkage": float(sub.loc["niche", "LTR_Shrinkage"] - sub.loc["mainstream", "LTR_Shrinkage"]),
                }
            )

    return method_rows, group_rows


def quality(row: pd.Series, metric: str, definition: str) -> float:
    prefix = "Static" if definition == "Static" else "Temporal"
    if metric == "ARP":
        return -float(row[f"{prefix}_ARP@20"])
    if metric == "LTR":
        return float(row[f"{prefix}_LTR@20"])
    if metric == "PCE":
        return -float(row[f"{prefix}_PCE@20"])
    raise ValueError(metric)


def audit_rows(
    summary: pd.DataFrame,
    dataset: str,
    seed: int,
    protocol: str,
    definition: str,
    methods: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    indexed = summary.set_index("Method")
    if "Base" not in indexed.index:
        return rows
    base = indexed.loc["Base"]
    present_methods = [method for method in methods if method in indexed.index]
    for method in present_methods:
        if method == "Base":
            continue
        method_row = indexed.loc[method]
        for metric in METHOD_METRICS:
            static_gain = quality(method_row, metric, "Static") - quality(base, metric, "Static")
            temporal_gain = quality(method_row, metric, "Temporal") - quality(base, metric, "Temporal")
            rows.append(
                {
                    "Dataset": dataset,
                    "Seed": seed,
                    "TemporalProtocol": protocol,
                    "TemporalDefinition": definition,
                    "Method": method,
                    "Metric": metric,
                    "StaticGain": static_gain,
                    "TemporalGain": temporal_gain,
                    "TOD": static_gain - temporal_gain,
                }
            )

    for metric in METHOD_METRICS:
        flips = 0
        pairs = 0
        for i, left in enumerate(present_methods):
            for right in present_methods[i + 1 :]:
                static_diff = quality(indexed.loc[left], metric, "Static") - quality(indexed.loc[right], metric, "Static")
                temporal_diff = quality(indexed.loc[left], metric, "Temporal") - quality(
                    indexed.loc[right], metric, "Temporal"
                )
                if np.sign(static_diff) != np.sign(temporal_diff):
                    flips += 1
                pairs += 1
        rows.append(
            {
                "Dataset": dataset,
                "Seed": seed,
                "TemporalProtocol": protocol,
                "TemporalDefinition": definition,
                "Method": "ALL_METHOD_PAIRS",
                "Metric": metric,
                "RFR": flips / pairs if pairs else np.nan,
                "FlipPairs": flips,
                "Pairs": pairs,
            }
        )
    return rows


def summarize(frame: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    numeric_cols = [col for col in frame.select_dtypes(include=[np.number]).columns if col not in set(keys) and col != "Seed"]
    grouped = frame.groupby(list(keys), dropna=False)
    counts = grouped["Seed"].nunique().rename("SeedCount").reset_index()
    mean = grouped[numeric_cols].mean().add_suffix("_mean").reset_index()
    std = grouped[numeric_cols].std(ddof=1).add_suffix("_std").reset_index()
    return counts.merge(mean, on=list(keys), how="left").merge(std, on=list(keys), how="left")


def process_dataset(
    first_seed_dir: Path,
    seed_dirs: Sequence[Path],
    drift_rows: List[Dict[str, object]],
    metric_rows: List[Dict[str, object]],
    tod_rows: List[Dict[str, object]],
    group_rows: List[Dict[str, object]],
) -> None:
    config = read_json(first_seed_dir / "config.json")
    dataset = dataset_label(config, first_seed_dir)
    datadir = Path(config["prepared_data"]["datadir"])
    train, val, test, all_events = read_interaction_split(datadir)
    n_users, n_items = infer_shape(train, val, test)
    static_pop = static_popularity(train, n_items)
    static_bucket = assign_buckets(static_pop, static_pop, dormant_for_zero=False)
    static_pct = popularity_percentiles(static_pop, static_pop)
    train_histories = build_user_histories(train, n_users)
    groups = activity_controlled_user_groups(train_histories, static_pct)
    snapshot_times, weights, user_snapshot, snapshot_table = build_eval_snapshots(test, n_users, config)
    snapshot_table.to_csv(OUT / f"{first_seed_dir.parent.name}_robustness_snapshot_times.csv", index=False)

    popularity_cfg = config["popularity"]
    window_seconds = int(popularity_cfg["main_recent_window_days"]) * SECONDS_PER_DAY
    tau_seconds = int(popularity_cfg["main_decay_tau_days"]) * SECONDS_PER_DAY

    protocols = [
        ("TrainOnly", train, ["Recent", "Decay", "Cumulative"]),
        ("LogObservable", all_events, ["Cumulative"]),
    ]
    for protocol, events, definitions in protocols:
        print(f"Building robustness features: {dataset} {protocol}", flush=True)
        recent_decay = None
        if any(definition in {"Recent", "Decay"} for definition in definitions):
            recent_decay = build_temporal_snapshot_features(
                events,
                snapshot_times,
                n_items,
                static_pop,
                window_seconds,
                tau_seconds,
            )
        cumulative = None
        if "Cumulative" in definitions:
            cumulative = build_cumulative_snapshot_features(events, snapshot_times, n_items, static_pop)

        for definition in definitions:
            if definition in {"Recent", "Decay"}:
                assert recent_decay is not None
                temporal = rename_temporal_features(recent_decay, definition)
            else:
                assert cumulative is not None
                temporal = rename_temporal_features(cumulative, definition)
            drift_rows.append(
                weighted_drift_rows(dataset, protocol, definition, static_bucket, temporal, weights)
            )

            for seed_dir in seed_dirs:
                seed_config = read_json(seed_dir / "config.json")
                seed = seed_value(seed_config, seed_dir)
                existing_summary = pd.read_csv(seed_dir / "test_method_metrics_full.csv")
                methods = list(existing_summary["Method"].astype(str))
                print(f"Evaluating robustness metrics: {dataset} seed={seed} {protocol} {definition}", flush=True)
                seed_metric_rows, seed_group_rows = compute_method_rows(
                    dataset=dataset,
                    seed=seed,
                    seed_dir=seed_dir,
                    methods=methods,
                    test=test,
                    train_histories=train_histories,
                    groups=groups,
                    user_snapshot=user_snapshot,
                    static_pop=static_pop,
                    static_bucket=static_bucket,
                    static_pct=static_pct,
                    temporal=temporal,
                    protocol=protocol,
                    definition=definition,
                    existing_summary=existing_summary,
                )
                metric_rows.extend(seed_metric_rows)
                group_rows.extend(seed_group_rows)
                tod_rows.extend(
                    audit_rows(
                        pd.DataFrame(seed_metric_rows),
                        dataset,
                        seed,
                        protocol,
                        definition,
                        methods,
                    )
                )


def write_report(
    drift: pd.DataFrame,
    metrics: pd.DataFrame,
    tod: pd.DataFrame,
    groups: pd.DataFrame,
) -> None:
    yelp_drift = drift[drift["Dataset"] == "Yelp original reviews"].copy()
    yelp_tod = tod[
        (tod["Dataset"] == "Yelp original reviews")
        & (tod["Method"] == "ALL_METHOD_PAIRS")
        & (tod["Metric"].isin(["LTR", "PCE"]))
    ].copy()
    lines = [
        "# Formal Temporal Robustness Appendix",
        "",
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Robustness Drift Preview",
        "",
        markdown_table(yelp_drift) if not yelp_drift.empty else "No Yelp drift rows.",
        "",
        "## Robustness RFR Preview",
        "",
        markdown_table(yelp_tod[["Dataset", "TemporalProtocol", "TemporalDefinition", "Metric", "RFR_mean", "FlipPairs_mean", "Pairs_mean"]])
        if not yelp_tod.empty
        else "No Yelp RFR rows.",
        "",
        "## Output Files",
        "",
        "- `robustness_drift.csv`",
        "- `robustness_method_metrics.csv`",
        "- `robustness_tod_rfr.csv`",
        "- `robustness_group_sensitivity.csv`",
        "- by-seed versions of each method/TOD/group table",
    ]
    (OUT / "temporal_robustness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs(OUT)
    drift_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    tod_rows: List[Dict[str, object]] = []
    group_rows: List[Dict[str, object]] = []

    by_dataset: Dict[Path, List[Path]] = defaultdict(list)
    for seed_dir in formal_seed_dirs():
        by_dataset[seed_dir.parent].append(seed_dir)

    for dataset_dir, seed_dirs in sorted(by_dataset.items()):
        process_dataset(seed_dirs[0], seed_dirs, drift_rows, metric_rows, tod_rows, group_rows)

    drift = pd.DataFrame(drift_rows)
    metrics_by_seed = pd.DataFrame(metric_rows)
    tod_by_seed = pd.DataFrame(tod_rows)
    groups_by_seed = pd.DataFrame(group_rows)
    metrics = summarize(metrics_by_seed, ["Dataset", "TemporalProtocol", "TemporalDefinition", "Method"])
    tod = summarize(tod_by_seed, ["Dataset", "TemporalProtocol", "TemporalDefinition", "Method", "Metric"])
    groups = summarize(groups_by_seed, ["Dataset", "TemporalProtocol", "TemporalDefinition", "Method", "Group"])

    drift.to_csv(OUT / "robustness_drift.csv", index=False)
    metrics_by_seed.to_csv(OUT / "robustness_method_metrics_by_seed.csv", index=False)
    metrics.to_csv(OUT / "robustness_method_metrics.csv", index=False)
    tod_by_seed.to_csv(OUT / "robustness_tod_rfr_by_seed.csv", index=False)
    tod.to_csv(OUT / "robustness_tod_rfr.csv", index=False)
    groups_by_seed.to_csv(OUT / "robustness_group_sensitivity_by_seed.csv", index=False)
    groups.to_csv(OUT / "robustness_group_sensitivity.csv", index=False)
    write_report(drift, metrics, tod, groups)
    print(f"Wrote temporal robustness outputs under {OUT}", flush=True)


if __name__ == "__main__":
    main()
