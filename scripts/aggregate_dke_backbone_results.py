#!/usr/bin/env python3
"""Aggregate DKE backbone robustness outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path("/root/temporal_popularity_pilot")
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from temporal_popularity.reporting import ensure_dirs, markdown_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/formal/backbone_robustness")
    parser.add_argument("--dataset", default="ml1m")
    parser.add_argument("--backbone", default="sasrec")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/formal/backbone_robustness/aggregate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    return parser.parse_args()


def read_seed_outputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: List[pd.DataFrame] = []
    tod_rows: List[pd.DataFrame] = []
    group_rows: List[pd.DataFrame] = []
    for seed in args.seeds:
        run_dir = args.root / args.dataset / f"seed{seed}" / args.backbone
        metrics_path = run_dir / "test_method_metrics.csv"
        tod_path = run_dir / "tod_rfr.csv"
        group_path = run_dir / "group_sensitivity.csv"
        missing = [str(path) for path in [metrics_path, tod_path, group_path] if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing outputs for seed {seed}: {missing}")
        metrics = pd.read_csv(metrics_path)
        tod = pd.read_csv(tod_path)
        group = pd.read_csv(group_path)
        for frame in [metrics, tod, group]:
            frame.insert(0, "Seed", seed)
            frame.insert(0, "Backbone", "SASRec")
            frame.insert(0, "Dataset", args.dataset)
        metric_rows.append(metrics)
        tod_rows.append(tod)
        group_rows.append(group)
    return pd.concat(metric_rows, ignore_index=True), pd.concat(tod_rows, ignore_index=True), pd.concat(group_rows, ignore_index=True)


def mean_std(frame: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    agg = frame.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "std"]).reset_index()
    agg.columns = [
        "_".join([part for part in col if part]) if isinstance(col, tuple) else col
        for col in agg.columns.to_flat_index()
    ]
    return agg


def compact_table(metrics: pd.DataFrame, tod: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    metric_agg = mean_std(
        metrics,
        ["Dataset", "Backbone", "Method"],
        ["NDCG@20", "Static_LTR@20", "Decay_LTR@20", "Static_PCE@20", "Decay_PCE@20"],
    )
    rfr = tod[tod["Method"] == "ALL_METHOD_PAIRS"].copy()
    rfr_agg = mean_std(rfr, ["Dataset", "Backbone", "Metric"], ["RFR", "FlipPairs", "Pairs"])
    gtsg = group[group["Group"] == "GTSG_niche_minus_mainstream"].copy()
    gtsg_agg = mean_std(gtsg, ["Dataset", "Backbone", "Method"], ["PCE_Sensitivity", "LTR_Shrinkage"])

    rows: List[Dict[str, object]] = []
    base = metric_agg[metric_agg["Method"] == "Base"].iloc[0]
    rows.append(
        {
            "Panel": "Base metrics",
            "Item": "NDCG@20",
            "Mean": base["NDCG@20_mean"],
            "Std": base["NDCG@20_std"],
        }
    )
    rows.append(
        {
            "Panel": "Base metrics",
            "Item": "Static LTR@20",
            "Mean": base["Static_LTR@20_mean"],
            "Std": base["Static_LTR@20_std"],
        }
    )
    rows.append(
        {
            "Panel": "Base metrics",
            "Item": "Decay LTR@20",
            "Mean": base["Decay_LTR@20_mean"],
            "Std": base["Decay_LTR@20_std"],
        }
    )
    rows.append(
        {
            "Panel": "Base metrics",
            "Item": "Static PCE@20",
            "Mean": base["Static_PCE@20_mean"],
            "Std": base["Static_PCE@20_std"],
        }
    )
    rows.append(
        {
            "Panel": "Base metrics",
            "Item": "Decay PCE@20",
            "Mean": base["Decay_PCE@20_mean"],
            "Std": base["Decay_PCE@20_std"],
        }
    )
    for _, row in rfr_agg.iterrows():
        rows.append(
            {
                "Panel": "Ranking flips",
                "Item": f"{row['Metric']} RFR",
                "Mean": row["RFR_mean"],
                "Std": row["RFR_std"],
            }
        )
    base_gtsg = gtsg_agg[gtsg_agg["Method"] == "Base"]
    if not base_gtsg.empty:
        row = base_gtsg.iloc[0]
        rows.append(
            {
                "Panel": "Group sensitivity",
                "Item": "Base GTSG PCE sensitivity",
                "Mean": row["PCE_Sensitivity_mean"],
                "Std": row["PCE_Sensitivity_std"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    ensure_dirs(args.outdir)
    metrics, tod, group = read_seed_outputs(args)
    metrics.to_csv(args.outdir / f"{args.dataset}_{args.backbone}_method_metrics_by_seed.csv", index=False)
    tod.to_csv(args.outdir / f"{args.dataset}_{args.backbone}_tod_rfr_by_seed.csv", index=False)
    group.to_csv(args.outdir / f"{args.dataset}_{args.backbone}_group_sensitivity_by_seed.csv", index=False)

    metric_agg = mean_std(
        metrics,
        ["Dataset", "Backbone", "Method"],
        ["NDCG@20", "Recall@20", "Static_LTR@20", "Decay_LTR@20", "Static_PCE@20", "Decay_PCE@20"],
    )
    tod_agg = mean_std(
        tod,
        ["Dataset", "Backbone", "Method", "Metric"],
        ["TOD", "RFR", "FlipPairs", "Pairs"],
    )
    group_agg = mean_std(
        group,
        ["Dataset", "Backbone", "Method", "Group"],
        ["PCE_Sensitivity", "LTR_Shrinkage"],
    )
    compact = compact_table(metrics, tod, group)
    metric_agg.to_csv(args.outdir / f"{args.dataset}_{args.backbone}_method_metrics_aggregate.csv", index=False)
    tod_agg.to_csv(args.outdir / f"{args.dataset}_{args.backbone}_tod_rfr_aggregate.csv", index=False)
    group_agg.to_csv(args.outdir / f"{args.dataset}_{args.backbone}_group_sensitivity_aggregate.csv", index=False)
    compact.to_csv(args.outdir / f"{args.dataset}_{args.backbone}_compact_backbone_table.csv", index=False)

    report = [
        "# DKE Backbone Robustness Aggregate",
        "",
        "This report aggregates SASRec backbone robustness outputs across formal seeds.",
        "",
        "## Compact Table",
        "",
        markdown_table(compact),
        "",
        "## Method Metrics",
        "",
        markdown_table(metric_agg),
        "",
        "## TOD / RFR",
        "",
        markdown_table(tod_agg),
        "",
        "## Group Sensitivity",
        "",
        markdown_table(group_agg),
    ]
    (args.outdir / f"{args.dataset}_{args.backbone}_backbone_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.outdir / f'{args.dataset}_{args.backbone}_backbone_report.md'}")


if __name__ == "__main__":
    main()
