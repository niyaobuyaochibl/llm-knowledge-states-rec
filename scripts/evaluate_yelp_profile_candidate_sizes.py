#!/usr/bin/env python3
"""Evaluate Yelp profile reranking under matched candidate-set sizes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path("/root/temporal_popularity_pilot")
SRC = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPT_DIR))

from temporal_popularity.data import infer_shape, read_interaction_split  # noqa: E402
from run_yelp_profile_rerank_pilot import (  # noqa: E402
    CandidateBatch,
    ClaimRecord,
    build_ordered_histories,
    build_yelp_metadata_for_iids,
    evaluate_methods,
    needed_iids_for_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default=ROOT / "data/yelp_day1")
    parser.add_argument("--profile-run-dir", type=Path, default=ROOT / "results/egpr_profile_repair/yelp_seed42_deepseek_300_expressive5")
    parser.add_argument("--reviews-path", type=Path, default=Path("/root/autodl-tmp/yelp-raw/yelp_academic_dataset_review.json"))
    parser.add_argument("--business-path", type=Path, default=Path("/root/autodl-tmp/yelp-raw/yelp_academic_dataset_business.json"))
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/egpr_profile_repair/yelp_direct_vs_profile/profile_candidate_sizes")
    parser.add_argument("--candidate-sizes", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--history-limit", type=int, default=20)
    parser.add_argument("--lambda-grid", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    parser.add_argument("--metadata-progress-every", type=int, default=500000)
    return parser.parse_args()


def load_batch(path: Path, split_name: str, candidate_size: int) -> CandidateBatch:
    data = np.load(path / f"candidates_lightgcn_{split_name}.npz")
    return CandidateBatch(
        users=data["users"].astype(np.int64),
        targets=data["targets"].astype(np.int64),
        candidates=data["candidates"][:, :candidate_size].astype(np.int64),
        scores=data["scores"][:, :candidate_size].astype(np.float32),
        split_name=split_name,
    )


def load_claim_records(path: Path) -> Dict[int, List[ClaimRecord]]:
    records: Dict[int, List[ClaimRecord]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rec = ClaimRecord(
                uid=int(row["uid"]),
                claim_id=int(row["claim_id"]),
                claim=str(row["claim"]),
                claim_type=str(row["claim_type"]),
                confidence=float(row.get("confidence", 0.0) or 0.0),
                support_count=int(row["support_count"]),
                support_score=float(row["support_score"]),
                support_weight=float(row["support_weight"]),
                status=str(row["status"]),
                supporting_items=[int(x) for x in row.get("supporting_items", [])],
            )
            records.setdefault(rec.uid, []).append(rec)
    return records


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "-" if pd.isna(x) else f"{x:.6f}")
        else:
            display[col] = display[col].astype(str)
    lines = ["| " + " | ".join(display.columns) + " |", "| " + " | ".join(["---"] * len(display.columns)) + " |"]
    for row in display.values.tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    train, val, test, _ = read_interaction_split(args.datadir)
    n_users, n_items = infer_shape(train, val, test)
    ordered_histories = build_ordered_histories(train, n_users)
    records = load_claim_records(args.profile_run_dir / "claim_support.jsonl")

    max_size = max(args.candidate_sizes)
    val_max = load_batch(args.profile_run_dir, "val", max_size)
    test_max = load_batch(args.profile_run_dir, "test", max_size)
    profile_users = sorted(set(val_max.users.astype(int).tolist() + test_max.users.astype(int).tolist()))
    needed = needed_iids_for_metadata(profile_users, ordered_histories, val_max, test_max, args.history_limit)
    print(f"building Yelp profile candidate-size metadata for {len(needed):,} needed items", flush=True)
    meta = build_yelp_metadata_for_iids(args.datadir, args.reviews_path, args.business_path, needed, n_items, args.metadata_progress_every)

    perf_tables = []
    rel_tables = []
    lambda_tables = []
    for size in args.candidate_sizes:
        print(f"evaluating profile rerank candidate_size={size}", flush=True)
        val_batch = load_batch(args.profile_run_dir, "val", size)
        test_batch = load_batch(args.profile_run_dir, "test", size)
        performance, reliability, lambda_table, _ = evaluate_methods(val_batch, test_batch, records, meta, args.lambda_grid, args.topk)
        performance.insert(0, "CandidateSet", size)
        reliability.insert(0, "CandidateSet", size)
        lambda_table.insert(0, "CandidateSet", size)
        perf_tables.append(performance)
        rel_tables.append(reliability)
        lambda_tables.append(lambda_table)

    perf = pd.concat(perf_tables, ignore_index=True)
    rel = pd.concat(rel_tables, ignore_index=True)
    lambdas = pd.concat(lambda_tables, ignore_index=True)
    perf.to_csv(args.outdir / "profile_candidate_performance.csv", index=False)
    rel.to_csv(args.outdir / "profile_candidate_reliability.csv", index=False)
    lambdas.to_csv(args.outdir / "profile_candidate_lambda_validation.csv", index=False)

    lines = [
        "# Yelp Profile Rerank Candidate-Size Evaluation",
        "",
        f"Profile run: `{args.profile_run_dir}`.",
        f"Candidate sizes: {args.candidate_sizes}. Output top-{args.topk}.",
        "",
        "## Performance",
        "",
        markdown_table(perf),
        "",
        "## Reliability",
        "",
        markdown_table(rel),
        "",
    ]
    (args.outdir / "profile_candidate_size_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Done. Report: {args.outdir / 'profile_candidate_size_report.md'}", flush=True)


if __name__ == "__main__":
    main()
