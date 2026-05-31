#!/usr/bin/env python3
"""Evaluate an xQuAD-style long-tail post-processing baseline.

This is the representative existing-baseline slot for the formal experiment.
It reuses trained LightGCN checkpoints and appends a static-tail post-processing
baseline to the already extended Base/PopPenalty/PopCal tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

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
from temporal_popularity.eval import select_lambda  # noqa: E402
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


def xquad_specs(lambdas: object) -> Dict[str, Tuple[str, Optional[float]]]:
    specs: Dict[str, Tuple[str, Optional[float]]] = {"Base": ("base", None)}
    for lam in lambdas:
        specs[f"XQuADTail@{float(lam):g}"] = ("xquad_tail", float(lam))
    return specs


def write_xquad_report(
    run_dir: Path,
    xquad_summary: pd.DataFrame,
    full_summary: pd.DataFrame,
    tod: pd.DataFrame,
    group_df: pd.DataFrame,
) -> None:
    lines = [
        "# Formal xQuAD-Style Baseline Report",
        "",
        "## xQuAD-Style Test Metrics",
        "",
        markdown_table(xquad_summary),
        "",
        "## Full Test Method Metrics",
        "",
        markdown_table(full_summary),
        "",
        "## Full Temporal Overclaim / Ranking Flip",
        "",
        markdown_table(tod),
        "",
        "## Full Group Temporal Sensitivity",
        "",
        markdown_table(group_df),
    ]
    (run_dir / "xquad_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = args.config.parent
    if (run_dir / "xquad.completed.ok").exists() and not args.force:
        raise SystemExit(f"xQuAD-style baseline already completed: {run_dir}")
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
    val_snapshot_table.to_csv(run_dir / "validation_xquad_snapshot_times.csv", index=False)
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
        xquad_specs(config["lambda_grid"]),
        topk,
        batch_size,
        collect_user_level=False,
    )
    val_summary.to_csv(run_dir / "validation_xquad_all_lambda_metrics.csv", index=False)
    xquad_lambda, xquad_lambda_table = select_lambda(
        val_summary,
        "Base",
        "XQuADTail@",
        "Static_LTR@20",
    )
    xquad_lambda_table.to_csv(run_dir / "validation_xquad_lambda_table.csv", index=False)

    print("Building test temporal states...", flush=True)
    test_temporal, test_user_snapshot, test_snapshot_table = build_eval_temporal(
        test, all_events, n_users, n_items, static_pop, config
    )
    test_snapshot_table.to_csv(run_dir / "test_xquad_snapshot_times.csv", index=False)
    selected_specs = {
        "Base": ("base", None),
        f"XQuADTail@{xquad_lambda:g}": ("xquad_tail", xquad_lambda),
    }
    xquad_summary, xquad_user_level, recs = evaluate_methods(
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
    xquad_summary.to_csv(run_dir / "test_xquad_method_metrics.csv", index=False)
    xquad_user_level.to_csv(run_dir / "test_xquad_user_level_metrics.csv", index=False)
    for method, matrix in recs.items():
        if method == "Base":
            continue
        np.save(run_dir / f"recs_{method.replace('@', '_').replace('.', 'p')}.npy", matrix)

    base_summary_path = run_dir / "test_method_metrics_extended.csv"
    base_user_path = run_dir / "test_user_level_metrics_extended.csv"
    if not base_summary_path.exists():
        base_summary_path = run_dir / "test_method_metrics.csv"
    if not base_user_path.exists():
        base_user_path = run_dir / "test_user_level_metrics.csv"
    base_summary = pd.read_csv(base_summary_path)
    base_user_level = pd.read_csv(base_user_path)
    full_summary = pd.concat(
        [base_summary, xquad_summary[xquad_summary["Method"] != "Base"]],
        ignore_index=True,
    )
    full_user_level = pd.concat(
        [base_user_level, xquad_user_level[xquad_user_level["Method"] != "Base"]],
        ignore_index=True,
    )
    method_names = full_summary["Method"].tolist()
    tod = temporal_overclaim_and_rfr(full_summary, method_names, temporal_definition="Decay")
    group_df = group_temporal_sensitivity(full_user_level, method_names)
    full_summary.to_csv(run_dir / "test_method_metrics_full.csv", index=False)
    full_user_level.to_csv(run_dir / "test_user_level_metrics_full.csv", index=False)
    tod.to_csv(run_dir / "tod_rfr_full.csv", index=False)
    group_df.to_csv(run_dir / "group_sensitivity_full.csv", index=False)
    write_xquad_report(run_dir, xquad_summary, full_summary, tod, group_df)

    (run_dir / "xquad.completed.ok").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    update_manifest(
        run_dir,
        {
            "xquad_status": "completed",
            "selected_xquad_lambda": xquad_lambda,
            "xquad_report": str(run_dir / "xquad_report.md"),
        },
    )
    print(f"Done. xQuAD-style report: {run_dir / 'xquad_report.md'}", flush=True)


if __name__ == "__main__":
    main()
