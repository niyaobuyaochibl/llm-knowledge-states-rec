#!/usr/bin/env python3
"""Yelp Day-3 Static/Temporal PopPenalty pilot."""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_ml1m_minipilot import (  # noqa: E402
    LAMBDA_GRID,
    LightGCN,
    SECONDS_PER_DAY,
    assign_buckets,
    build_norm_adj,
    build_user_histories,
    compute_tod_rfr,
    group_sensitivity,
    popularity_percentiles,
    select_lambda,
    set_seed,
    static_popularity,
    topk_indices,
    user_groups,
)
from run_yelp_day2_base import (  # noqa: E402
    build_exclude_lists,
    build_snapshots,
    build_temporal_snapshot_features,
    markdown_table,
    read_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default=Path("/root/temporal_popularity_pilot/data/yelp_day1"))
    parser.add_argument("--base-outdir", type=Path, default=Path("/root/temporal_popularity_pilot/results/yelp_day2_base"))
    parser.add_argument("--outdir", type=Path, default=Path("/root/temporal_popularity_pilot/results/yelp_day3_poppenalty"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--snapshot-count", type=int, default=200)
    parser.add_argument("--window-days", type=int, default=180)
    parser.add_argument("--tau-days", type=int, default=180)
    return parser.parse_args()


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def zscore(values: np.ndarray) -> np.ndarray:
    std = values.std()
    if std < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - values.mean()) / std).astype(np.float32)


def median_or_zero(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.median(values))


def make_method_specs(lambdas: Sequence[float]) -> Dict[str, Tuple[str, Optional[float]]]:
    specs: Dict[str, Tuple[str, Optional[float]]] = {"Base": ("base", None)}
    for lam in lambdas:
        specs[f"StaticPopPenalty@{lam:g}"] = ("static", float(lam))
    for lam in lambdas:
        specs[f"TemporalPopPenalty@{lam:g}"] = ("temporal", float(lam))
    return specs


def compute_metric_row(
    method: str,
    uid: int,
    group: str,
    rec: np.ndarray,
    target: int,
    hist: np.ndarray,
    snap: int,
    static_pop: np.ndarray,
    static_bucket: np.ndarray,
    static_pct: np.ndarray,
    temporal: Mapping[str, np.ndarray],
    static_hist_median: np.ndarray,
) -> Dict[str, object]:
    hit_positions = np.flatnonzero(rec == target)
    hit = len(hit_positions) > 0
    ndcg = 1.0 / math.log2(int(hit_positions[0]) + 2) if hit else 0.0
    recall = 1.0 if hit else 0.0
    recent_pct = temporal["recent_pct"][snap]
    decay_pct = temporal["decay_pct"][snap]
    return {
        "Method": method,
        "uid": int(uid),
        "Group": group,
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
        "Static_PCE@20": float(abs(np.median(static_pct[rec]) - static_hist_median[uid])),
        "Recent_PCE@20": float(abs(np.median(recent_pct[rec]) - median_or_zero(recent_pct[hist]))),
        "Decay_PCE@20": float(abs(np.median(decay_pct[rec]) - median_or_zero(decay_pct[hist]))),
        "Static_SPS@20": float(np.median(static_pct[rec]) - static_hist_median[uid]),
        "Recent_SPS@20": float(np.median(recent_pct[rec]) - median_or_zero(recent_pct[hist])),
        "Decay_SPS@20": float(np.median(decay_pct[rec]) - median_or_zero(decay_pct[hist])),
    }


def evaluate_poppenalty(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    eval_frame: pd.DataFrame,
    train_histories: Sequence[np.ndarray],
    exclude_lists: Sequence[np.ndarray],
    static_pop: np.ndarray,
    static_bucket: np.ndarray,
    static_pct: np.ndarray,
    temporal: Mapping[str, np.ndarray],
    user_snapshot: np.ndarray,
    groups: Dict[int, str],
    method_specs: Mapping[str, Tuple[str, Optional[float]]],
    topk: int,
    batch_size: int,
    collect_user_level: bool,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, np.ndarray]]:
    eval_sorted = eval_frame.sort_values("uid", kind="mergesort").reset_index(drop=True)
    eval_users = eval_sorted["uid"].to_numpy(np.int64)
    targets = eval_sorted["iid"].to_numpy(np.int64)
    item_emb_t = item_emb.T.astype(np.float32)
    method_names = list(method_specs.keys())
    metric_sums: Dict[str, Dict[str, float]] = {name: {} for name in method_names}
    metric_counts: Dict[str, int] = {name: 0 for name in method_names}
    user_rows: List[Dict[str, object]] = []
    recs: Dict[str, np.ndarray] = {
        name: np.zeros((len(eval_users), topk), dtype=np.int64) for name in method_names if collect_user_level
    }
    static_hist_median = np.asarray(
        [median_or_zero(static_pct[train_histories[int(uid)]]) for uid in range(len(train_histories))],
        dtype=np.float32,
    )

    for start in range(0, len(eval_users), batch_size):
        end = min(start + batch_size, len(eval_users))
        batch_users = eval_users[start:end]
        scores = user_emb[batch_users].astype(np.float32) @ item_emb_t
        for local_row, uid_np in enumerate(batch_users):
            uid = int(uid_np)
            global_row = start + local_row
            row_scores = scores[local_row]
            excluded = exclude_lists[uid]
            candidate_mask = np.ones(item_emb.shape[0], dtype=bool)
            candidate_mask[excluded] = False
            candidate_items = np.flatnonzero(candidate_mask)
            candidate_scores = row_scores[candidate_items]
            snap = int(user_snapshot[uid])
            score_z = zscore(candidate_scores)
            static_z = zscore(static_pop[candidate_items].astype(np.float32))
            decay_z = zscore(temporal["decay_pop"][snap, candidate_items].astype(np.float32))
            hist = train_histories[uid]
            target = int(targets[global_row])

            for method, (kind, lam) in method_specs.items():
                if kind == "base":
                    rank_scores = candidate_scores
                elif kind == "static":
                    rank_scores = score_z - float(lam) * static_z
                elif kind == "temporal":
                    rank_scores = score_z - float(lam) * decay_z
                else:
                    raise ValueError(kind)
                rec = candidate_items[topk_indices(rank_scores, topk)]
                if collect_user_level:
                    recs[method][global_row] = rec
                row = compute_metric_row(
                    method,
                    uid,
                    groups[uid],
                    rec,
                    target,
                    hist,
                    snap,
                    static_pop,
                    static_bucket,
                    static_pct,
                    temporal,
                    static_hist_median,
                )
                if collect_user_level:
                    user_rows.append(row)
                for key, value in row.items():
                    if key in {"Method", "uid", "Group"}:
                        continue
                    metric_sums[method][key] = metric_sums[method].get(key, 0.0) + float(value)
                metric_counts[method] += 1
        if end == len(eval_users) or end % (batch_size * 50) == 0:
            print(f"evaluated users={end}/{len(eval_users)} methods={len(method_names)}", flush=True)

    summary_rows: List[Dict[str, object]] = []
    for method in method_names:
        count = metric_counts[method]
        row = {"Method": method, "Users": count}
        for key, total in metric_sums[method].items():
            row[key] = total / count
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    user_df = pd.DataFrame(user_rows) if collect_user_level else None
    return summary, user_df, recs


def write_report(
    outdir: Path,
    val_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    tod: pd.DataFrame,
    group_df: pd.DataFrame,
    selected_static_lambda: float,
    selected_temporal_lambda: float,
) -> None:
    selected = test_summary[
        test_summary["Method"].isin(
            ["Base", f"StaticPopPenalty@{selected_static_lambda:g}", f"TemporalPopPenalty@{selected_temporal_lambda:g}"]
        )
    ]
    lines = [
        "# Yelp Day-3 PopPenalty",
        "",
        "This run reuses the Yelp Day-2 Base LightGCN checkpoint and applies post-hoc Static/Temporal PopPenalty reranking.",
        "",
        "## Selected Lambdas",
        "",
        f"- Static PopPenalty: lambda={selected_static_lambda:g}",
        f"- Temporal PopPenalty: lambda={selected_temporal_lambda:g}",
        "",
        "## Validation Summary",
        "",
        markdown_table(val_summary),
        "",
        "## Test Metrics",
        "",
        markdown_table(selected),
        "",
        "## Temporal Overclaim / Ranking Flip",
        "",
        markdown_table(tod),
        "",
        "## Group Sensitivity",
        "",
        markdown_table(group_df),
    ]
    outdir.joinpath("day3_poppenalty_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs(args.outdir)

    print("Loading Yelp split and Base checkpoint...", flush=True)
    train, val, test, all_events = read_split(args.datadir)
    n_users = int(max(train["uid"].max(), val["uid"].max(), test["uid"].max()) + 1)
    n_items = int(train["iid"].max() + 1)
    static_pop = static_popularity(train, n_items)
    static_bucket = assign_buckets(static_pop, static_pop, dormant_for_zero=False)
    static_pct = popularity_percentiles(static_pop, static_pop)
    train_histories = build_user_histories(train, n_users)
    groups = user_groups(train_histories, static_pct)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_adj = build_norm_adj(train, n_users, n_items, device)
    model = LightGCN(n_users, n_items, args.embedding_dim, args.layers).to(device)
    state_path = args.base_outdir / "lightgcn_base_seed42.pt"
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()
    with torch.no_grad():
        user_emb_t, item_emb_t = model.propagate(norm_adj)
    user_emb = user_emb_t.detach().cpu().numpy().astype(np.float32)
    item_emb = item_emb_t.detach().cpu().numpy().astype(np.float32)

    print("Validation: building temporal snapshots...", flush=True)
    val_snapshot_times, _, val_user_snapshot = build_snapshots(val, n_users, args.snapshot_count)
    val_temporal = build_temporal_snapshot_features(
        all_events,
        val_snapshot_times,
        n_items,
        static_pop,
        args.window_days * SECONDS_PER_DAY,
        args.tau_days * SECONDS_PER_DAY,
    )
    val_exclude = train_histories
    method_specs_all = make_method_specs(LAMBDA_GRID)
    print("Validation: full-ranking all lambda candidates...", flush=True)
    val_summary, _, _ = evaluate_poppenalty(
        user_emb,
        item_emb,
        val,
        train_histories,
        val_exclude,
        static_pop,
        static_bucket,
        static_pct,
        val_temporal,
        val_user_snapshot,
        groups,
        method_specs_all,
        args.topk,
        args.eval_batch_size,
        collect_user_level=False,
    )
    val_summary.to_csv(args.outdir / "validation_all_lambda_metrics.csv", index=False)
    selected_static_lambda, static_lambda_table = select_lambda(
        val_summary, "Base", "StaticPopPenalty@", "Static_LTR@20"
    )
    selected_temporal_lambda, temporal_lambda_table = select_lambda(
        val_summary, "Base", "TemporalPopPenalty@", "Decay_LTR@20"
    )
    static_lambda_table.to_csv(args.outdir / "validation_static_lambda_table.csv", index=False)
    temporal_lambda_table.to_csv(args.outdir / "validation_temporal_lambda_table.csv", index=False)

    print("Test: building temporal snapshots...", flush=True)
    test_snapshot_times, _, test_user_snapshot = build_snapshots(test, n_users, args.snapshot_count)
    test_temporal = build_temporal_snapshot_features(
        all_events,
        test_snapshot_times,
        n_items,
        static_pop,
        args.window_days * SECONDS_PER_DAY,
        args.tau_days * SECONDS_PER_DAY,
    )
    test_exclude = build_exclude_lists(train_histories, val, n_users)
    selected_specs = {
        "Base": ("base", None),
        f"StaticPopPenalty@{selected_static_lambda:g}": ("static", selected_static_lambda),
        f"TemporalPopPenalty@{selected_temporal_lambda:g}": ("temporal", selected_temporal_lambda),
    }
    print("Test: full-ranking selected methods...", flush=True)
    test_summary, test_user_level, recs = evaluate_poppenalty(
        user_emb,
        item_emb,
        test,
        train_histories,
        test_exclude,
        static_pop,
        static_bucket,
        static_pct,
        test_temporal,
        test_user_snapshot,
        groups,
        selected_specs,
        args.topk,
        args.eval_batch_size,
        collect_user_level=True,
    )
    method_names = list(selected_specs.keys())
    tod = compute_tod_rfr(test_summary, method_names, temporal_definition="Decay")
    group_df = group_sensitivity(test_user_level, method_names)

    test_summary.to_csv(args.outdir / "table3_static_vs_temporal_evaluation.csv", index=False)
    test_user_level.to_csv(args.outdir / "test_user_level_metrics.csv", index=False)
    tod.to_csv(args.outdir / "table4_temporal_overclaim_ranking_flip.csv", index=False)
    group_df.to_csv(args.outdir / "table5_group_temporal_sensitivity.csv", index=False)
    for method, matrix in recs.items():
        np.save(args.outdir / f"recs_{method.replace('@', '_').replace('.', 'p')}.npy", matrix)

    write_report(
        args.outdir,
        val_summary,
        test_summary,
        tod,
        group_df,
        selected_static_lambda,
        selected_temporal_lambda,
    )
    print(f"Done. Report: {args.outdir / 'day3_poppenalty_report.md'}", flush=True)


if __name__ == "__main__":
    main()
