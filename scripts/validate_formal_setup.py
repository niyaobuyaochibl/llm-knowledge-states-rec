#!/usr/bin/env python3
"""Validate formal experiment configs and available prepared data.

This script performs no training and no recommendation evaluation. It checks
whether formal configs are internally coherent and whether the currently prepared
mini-pilot splits/results can support the next formal implementation step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path("/root/temporal_popularity_pilot")
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from temporal_popularity.data import infer_shape, read_interaction_split  # noqa: E402
from temporal_popularity.reporting import ensure_dirs, markdown_table  # noqa: E402


DATASET_DATA_DIRS = {
    "ml1m": ROOT / "data/ml1m_e200",
    "yelp_original_reviews": ROOT / "data/yelp_day1",
}

DATASET_REQUIRED_RESULTS = {
    "ml1m": [
        ROOT / "results/ml1m_e200/pilot_report.md",
        ROOT / "results/ml1m_e200/table3_static_vs_temporal_evaluation.csv",
        ROOT / "results/ml1m_e200/table4_temporal_overclaim_ranking_flip.csv",
    ],
    "yelp_original_reviews": [
        ROOT / "results/yelp_day1/day1_report.md",
        ROOT / "results/yelp_day2_base/day2_base_report.md",
        ROOT / "results/yelp_day3_poppenalty/day3_poppenalty_report.md",
        ROOT / "results/yelp_protocol_robustness/protocol_robustness_report.md",
        ROOT / "results/yelp_exact_subset_check/exact_subset_report.md",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        default=[
            ROOT / "configs/formal/ml1m_formal_template.json",
            ROOT / "configs/formal/yelp_formal_template.json",
        ],
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/formal/aggregate")
    return parser.parse_args()


def check_config(config_path: Path) -> Dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    datadir = DATASET_DATA_DIRS[dataset]
    train, val, test, all_events = read_interaction_split(datadir)
    n_users, n_items = infer_shape(train, val, test)
    expected_seed_count = len(config.get("seeds", []))
    missing_results = [str(path) for path in DATASET_REQUIRED_RESULTS[dataset] if not path.exists()]

    errors: List[str] = []
    if not config["evaluation"].get("full_ranking"):
        errors.append("evaluation.full_ranking must be true")
    if config["evaluation"].get("topk") != 20:
        errors.append("evaluation.topk must be 20")
    if config["split"].get("k_core") != 10:
        errors.append("split.k_core must be 10")
    if expected_seed_count != 3:
        errors.append("formal config must contain exactly 3 seeds")
    if missing_results:
        errors.append("missing required mini-pilot/protocol result files")

    return {
        "Config": str(config_path),
        "Dataset": dataset,
        "Users": n_users,
        "Items": n_items,
        "Train": len(train),
        "Val": len(val),
        "Test": len(test),
        "AllEvents": len(all_events),
        "Seeds": ",".join(str(seed) for seed in config.get("seeds", [])),
        "TemporalProtocol": config["evaluation"].get("temporal_protocol"),
        "Ready": not errors,
        "Errors": "; ".join(errors),
        "MissingResults": "; ".join(missing_results),
    }


def main() -> None:
    args = parse_args()
    ensure_dirs(args.outdir)
    rows = [check_config(path) for path in args.configs]
    summary = pd.DataFrame(rows)
    summary.to_csv(args.outdir / "formal_setup_validation.csv", index=False)
    report = [
        "# Formal Setup Validation",
        "",
        "This is a dry-run validation. It performs no training and no recommendation evaluation.",
        "",
        markdown_table(summary),
    ]
    (args.outdir / "formal_setup_validation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not bool(summary["Ready"].all()):
        raise SystemExit(1)
    print(f"Done. Report: {args.outdir / 'formal_setup_validation.md'}", flush=True)


if __name__ == "__main__":
    main()
