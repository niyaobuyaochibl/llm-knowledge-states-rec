#!/usr/bin/env python3
"""Evaluate Static/Temporal PopCal for an already trained formal seed.

This script does not retrain LightGCN. It loads the checkpoint produced by
`run_formal_seed.py`, evaluates PopCal lambdas on validation, evaluates the
selected PopCal variants on test, and writes extended per-seed tables that
combine Base, PopPenalty, and PopCal methods.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path("/root/temporal_popularity_pilot")
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

from run_formal_seed import build_eval_temporal, evaluate_methods, update_manifest  # noqa: E402
from temporal_popularity.audit import group_temporal_sensitivity, temporal_overclaim_and_rfr  # noqa: E402
from temporal_popularity.data import (  # noqa: E402
    activity_controlled_user_groups,
    build_exclude_lists,
    build_user_histories,
    infer_shape,
    read_interaction_split,
    set_seed,
)
from temporal_popularity.model import LightGCN, build_norm_adj  # noqa: E402
from temporal_popularity.popularity import assign_buckets, popularity_percentiles, static_popularity  # noqa: E402
from temporal_popularity.reporting import markdown_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def device_from_arg(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def popcal_specs(lambdas: object) -> Dict[str, Tuple[str, Optional[float]]]:
    specs: Dict[str, Tuple[str, Optional[float]]] = {"Base": ("base", None)}
    for lam in lambdas:
        specs[f"StaticPopCal@{float(lam):g}"] = ("static_cal", float(lam))
    for lam in lambdas:
        specs[f"TemporalPopCal@{float(lam):g}"] = ("temporal_cal", float(lam))
    return specs


def select_lambda_min(
    val_summary: pd.DataFrame,
    base_method: str,
    prefix: str,
    metric: str,
    ndcg_col: str = "NDCG@20",
) -> Tuple[float, pd.DataFrame]:
    """Select lambda by minimizing a calibration metric under 5% NDCG drop."""
    base_ndcg = float(val_summary.loc[val_summary["Method"] == base_method, ndcg_col].iloc[0])
    candidates = val_summary[val_summary["Method"].str.startswith(prefix)].copy()
    candidates["lambda"] = candidates["Method"].str.split("@").str[-1].astype(float)
    candidates["ndcg_drop"] = base_ndcg - candidates[ndcg_col]
    candidates["eligible_ndcg"] = candidates[ndcg_col] >= 0.95 * base_ndcg
    eligible = candidates[candidates["eligible_ndcg"]]
    if eligible.empty:
        eligible = candidates.sort_values(["ndcg_drop", "lambda"], ascending=[True, True]).head(1)
    selected = eligible.sort_values([metric, ndcg_col, "lambda"], ascending=[True, False, True]).iloc[0]
    return float(selected["lambda"]), candidates


def write_popcal_report(
    run_dir: Path,
    popcal_summary: pd.DataFrame,
    extended_summary: pd.DataFrame,
    tod: pd.DataFrame,
    group_df: pd.DataFrame,
) -> None:
    lines = [
        "# Formal PopCal Extension Report",
        "",
        "## PopCal Test Metrics",
        "",
        markdown_table(popcal_summary),
        "",
        "## Extended Test Method Metrics",
        "",
        markdown_table(extended_summary),
        "",
        "## Extended Temporal Overclaim / Ranking Flip",
        "",
        markdown_table(tod),
        "",
        "## Extended Group Temporal Sensitivity",
        "",
        markdown_table(group_df),
    ]
    (run_dir / "popcal_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = args.config.parent
    if (run_dir / "popcal.completed.ok").exists() and not args.force:
        raise SystemExit(f"PopCal already completed: {run_dir}")
    checkpoint = run_dir / "lightgcn.pt"
    if not checkpoint.exists():
        raise SystemExit(f"Missing trained checkpoint: {checkpoint}")

    seed = int(config["seed"])
    set_seed(seed)
    datadir = Path(config["prepared_data"]["datadir"])
    train, val, test, all_events = read_interaction_split(datadir)
    n_users, n_items = infer_shape(train, val, test)
    static_pop = static_popularity(train, n_items)
    static_bucket = assign_buckets(static_pop, static_pop, dormant_for_zero=False)
    static_pct = popularity_percentiles(static_pop, static_pop)
    train_histories = build_user_histories(train, n_users)
    groups = activity_controlled_user_groups(train_histories, static_pct)
    topk = int(config["evaluation"]["topk"])
    batch_size = int(args.eval_batch_size or 128)

    device = device_from_arg(args.device)
    model_cfg = config["backbone"]
    model = LightGCN(n_users, n_items, int(model_cfg["embedding_dim"]), int(model_cfg["layers"])).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    norm_adj = build_norm_adj(train, n_users, n_items, device)
    model.eval()
    with torch.no_grad():
        user_emb_t, item_emb_t = model.propagate(norm_adj)
    user_emb = user_emb_t.detach().cpu().numpy().astype(np.float32)
    item_emb = item_emb_t.detach().cpu().numpy().astype(np.float32)

    print("Building validation temporal states...", flush=True)
    val_temporal, val_user_snapshot, val_snapshot_table = build_eval_temporal(
        val, all_events, n_users, n_items, static_pop, config
    )
    val_snapshot_table.to_csv(run_dir / "validation_popcal_snapshot_times.csv", index=False)
    val_summary, _, _ = evaluate_methods(
        user_emb,
        item_emb,
        val,
        train_histories,
        build_exclude_lists(train_histories, None, n_users),
        static_pop,
        static_bucket,
        static_pct,
        val_temporal,
        val_user_snapshot,
        groups,
        popcal_specs(config["lambda_grid"]),
        topk,
        batch_size,
        collect_user_level=False,
    )
    val_summary.to_csv(run_dir / "validation_popcal_all_lambda_metrics.csv", index=False)

    static_lambda, static_lambda_table = select_lambda_min(
        val_summary,
        "Base",
        "StaticPopCal@",
        "Static_PCE@20",
    )
    temporal_lambda, temporal_lambda_table = select_lambda_min(
        val_summary,
        "Base",
        "TemporalPopCal@",
        "Decay_PCE@20",
    )
    static_lambda_table.to_csv(run_dir / "validation_static_popcal_lambda_table.csv", index=False)
    temporal_lambda_table.to_csv(run_dir / "validation_temporal_popcal_lambda_table.csv", index=False)

    print("Building test temporal states...", flush=True)
    test_temporal, test_user_snapshot, test_snapshot_table = build_eval_temporal(
        test, all_events, n_users, n_items, static_pop, config
    )
    test_snapshot_table.to_csv(run_dir / "test_popcal_snapshot_times.csv", index=False)
    selected_specs = {
        "Base": ("base", None),
        f"StaticPopCal@{static_lambda:g}": ("static_cal", static_lambda),
        f"TemporalPopCal@{temporal_lambda:g}": ("temporal_cal", temporal_lambda),
    }
    popcal_summary, popcal_user_level, recs = evaluate_methods(
        user_emb,
        item_emb,
        test,
        train_histories,
        build_exclude_lists(train_histories, val, n_users),
        static_pop,
        static_bucket,
        static_pct,
        test_temporal,
        test_user_snapshot,
        groups,
        selected_specs,
        topk,
        batch_size,
        collect_user_level=True,
    )
    popcal_summary.to_csv(run_dir / "test_popcal_method_metrics.csv", index=False)
    popcal_user_level.to_csv(run_dir / "test_popcal_user_level_metrics.csv", index=False)
    for method, matrix in recs.items():
        if method == "Base":
            continue
        np.save(run_dir / f"recs_{method.replace('@', '_').replace('.', 'p')}.npy", matrix)

    penalty_summary = pd.read_csv(run_dir / "test_method_metrics.csv")
    extended_summary = pd.concat(
        [penalty_summary, popcal_summary[popcal_summary["Method"] != "Base"]],
        ignore_index=True,
    )
    penalty_user_level = pd.read_csv(run_dir / "test_user_level_metrics.csv")
    extended_user_level = pd.concat(
        [penalty_user_level, popcal_user_level[popcal_user_level["Method"] != "Base"]],
        ignore_index=True,
    )
    method_names = extended_summary["Method"].tolist()
    tod = temporal_overclaim_and_rfr(extended_summary, method_names, temporal_definition="Decay")
    group_df = group_temporal_sensitivity(extended_user_level, method_names)
    extended_summary.to_csv(run_dir / "test_method_metrics_extended.csv", index=False)
    extended_user_level.to_csv(run_dir / "test_user_level_metrics_extended.csv", index=False)
    tod.to_csv(run_dir / "tod_rfr_extended.csv", index=False)
    group_df.to_csv(run_dir / "group_sensitivity_extended.csv", index=False)
    write_popcal_report(run_dir, popcal_summary, extended_summary, tod, group_df)

    (run_dir / "popcal.completed.ok").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    update_manifest(
        run_dir,
        {
            "popcal_status": "completed",
            "selected_static_popcal_lambda": static_lambda,
            "selected_temporal_popcal_lambda": temporal_lambda,
            "popcal_report": str(run_dir / "popcal_report.md"),
        },
    )
    print(f"Done. PopCal report: {run_dir / 'popcal_report.md'}", flush=True)


if __name__ == "__main__":
    main()
