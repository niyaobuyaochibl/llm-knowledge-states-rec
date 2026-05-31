#!/usr/bin/env python3
"""Affected-user and rank-disruption analysis for profile-before-ranking pilots."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/root/temporal_popularity_pilot")
SRC = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPT_DIR))

from temporal_popularity.data import infer_shape, read_interaction_split  # noqa: E402
from run_llm_selective_invocation_pilot import read_item_metadata  # noqa: E402
from run_egpr_profile_repair_pilot import (  # noqa: E402
    CandidateBatch as MlCandidateBatch,
    ClaimRecord as MlClaimRecord,
    profile_scores_for_batch as ml_profile_scores_for_batch,
    ranked_from_profile_scores as ml_ranked_from_profile_scores,
)
from run_yelp_profile_rerank_pilot import (  # noqa: E402
    CandidateBatch as YelpCandidateBatch,
    ClaimRecord as YelpClaimRecord,
    build_ordered_histories as yelp_build_ordered_histories,
    build_yelp_metadata_for_iids,
    needed_iids_for_metadata,
    profile_scores_for_batch as yelp_profile_scores_for_batch,
    ranked_from_profile_scores as yelp_ranked_from_profile_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/egpr_profile_repair/affected_user_analysis")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--history-limit", type=int, default=20)
    parser.add_argument("--ml1m-datadir", type=Path, default=ROOT / "data/ml1m")
    parser.add_argument("--ml1m-movies-path", type=Path, default=Path("/root/autodl-tmp/projects_archive/MGPrompt-CDR/data/raw/ml-1m/movies.dat"))
    parser.add_argument("--ml1m-direct-run", type=Path, default=ROOT / "results/llm_selective/ml1m_seed42_deepseek_500")
    parser.add_argument("--ml1m-profile-run", type=Path, default=ROOT / "results/egpr_profile_repair/ml1m_seed42_deepseek_500_expressive5")
    parser.add_argument("--yelp-datadir", type=Path, default=ROOT / "data/yelp_day1")
    parser.add_argument("--yelp-profile-run", type=Path, default=ROOT / "results/egpr_profile_repair/yelp_seed42_deepseek_300_expressive5")
    parser.add_argument("--yelp-direct-top50", type=Path, default=ROOT / "results/egpr_profile_repair/yelp_direct_vs_profile/direct_rerank_300")
    parser.add_argument("--yelp-direct-top100", type=Path, default=ROOT / "results/egpr_profile_repair/yelp_direct_vs_profile/direct_rerank_300_top100")
    parser.add_argument("--yelp-reviews-path", type=Path, default=Path("/root/autodl-tmp/yelp-raw/yelp_academic_dataset_review.json"))
    parser.add_argument("--yelp-business-path", type=Path, default=Path("/root/autodl-tmp/yelp-raw/yelp_academic_dataset_business.json"))
    parser.add_argument("--metadata-progress-every", type=int, default=500000)
    return parser.parse_args()


def ndcg_recall(ranked: np.ndarray, targets: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    ndcg = np.zeros(len(targets), dtype=np.float64)
    recall = np.zeros(len(targets), dtype=np.float64)
    for row, target_np in enumerate(targets):
        pos = np.flatnonzero(ranked[row, :topk] == int(target_np))
        if len(pos):
            recall[row] = 1.0
            ndcg[row] = 1.0 / math.log2(int(pos[0]) + 2)
    return ndcg, recall


def load_ml_claim_records(path: Path) -> Dict[int, List[MlClaimRecord]]:
    records: Dict[int, List[MlClaimRecord]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rec = MlClaimRecord(
                uid=int(row["uid"]), claim_id=int(row["claim_id"]), claim=str(row["claim"]),
                claim_type=str(row["claim_type"]), confidence=float(row.get("confidence", 0.0) or 0.0),
                support_count=int(row["support_count"]), support_score=float(row["support_score"]),
                support_weight=float(row["support_weight"]), status=str(row["status"]),
                supporting_items=[int(x) for x in row.get("supporting_items", [])],
            )
            records.setdefault(rec.uid, []).append(rec)
    return records


def load_yelp_claim_records(path: Path) -> Dict[int, List[YelpClaimRecord]]:
    records: Dict[int, List[YelpClaimRecord]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rec = YelpClaimRecord(
                uid=int(row["uid"]), claim_id=int(row["claim_id"]), claim=str(row["claim"]),
                claim_type=str(row["claim_type"]), confidence=float(row.get("confidence", 0.0) or 0.0),
                support_count=int(row["support_count"]), support_score=float(row["support_score"]),
                support_weight=float(row["support_weight"]), status=str(row["status"]),
                supporting_items=[int(x) for x in row.get("supporting_items", [])],
            )
            records.setdefault(rec.uid, []).append(rec)
    return records


def topk_jaccard(a: np.ndarray, b: np.ndarray, topk: int) -> np.ndarray:
    out = np.zeros(len(a), dtype=np.float64)
    for i in range(len(a)):
        left = set(int(x) for x in a[i, :topk])
        right = set(int(x) for x in b[i, :topk])
        out[i] = len(left.intersection(right)) / len(left.union(right)) if left or right else 1.0
    return out


def overlap_ratio(a: np.ndarray, b: np.ndarray, topk: int) -> np.ndarray:
    out = np.zeros(len(a), dtype=np.float64)
    for i in range(len(a)):
        left = set(int(x) for x in a[i, :topk])
        right = set(int(x) for x in b[i, :topk])
        out[i] = len(left.intersection(right)) / float(topk)
    return out


def mean_abs_rank_shift(a: np.ndarray, b: np.ndarray, topk: int) -> np.ndarray:
    out = np.zeros(len(a), dtype=np.float64)
    for i in range(len(a)):
        pos_a = {int(item): pos for pos, item in enumerate(a[i, :topk])}
        pos_b = {int(item): pos for pos, item in enumerate(b[i, :topk])}
        common = set(pos_a).intersection(pos_b)
        if not common:
            out[i] = float(topk)
        else:
            out[i] = float(np.mean([abs(pos_a[item] - pos_b[item]) for item in common]))
    return out


def target_rank(ranked: np.ndarray, targets: np.ndarray, topk: int) -> np.ndarray:
    ranks = np.full(len(targets), topk + 1, dtype=np.int32)
    for i, target_np in enumerate(targets):
        pos = np.flatnonzero(ranked[i, :topk] == int(target_np))
        if len(pos):
            ranks[i] = int(pos[0]) + 1
    return ranks


def method_rows(dataset: str, candidate_setting: str, users: np.ndarray, targets: np.ndarray, methods: Mapping[str, np.ndarray], topk: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_ranked = methods["LightGCN"]
    base_ndcg, base_recall = ndcg_recall(base_ranked, targets, topk)
    method_rows_out = []
    per_user_rows = []
    method_metrics: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, ranked in methods.items():
        ndcg, recall = ndcg_recall(ranked, targets, topk)
        method_metrics[name] = (ndcg, recall)
        delta = ndcg - base_ndcg
        jac = topk_jaccard(ranked, base_ranked, topk) if name != "LightGCN" else np.ones(len(users))
        overlap = overlap_ratio(ranked, base_ranked, topk) if name != "LightGCN" else np.ones(len(users))
        shift = mean_abs_rank_shift(ranked, base_ranked, topk) if name != "LightGCN" else np.zeros(len(users))
        ranks = target_rank(ranked, targets, topk)
        changed = jac < 0.999999
        method_rows_out.append({
            "Dataset": dataset,
            "CandidateSetting": candidate_setting,
            "Method": name,
            "Users": int(len(users)),
            "NDCG@20": float(ndcg.mean()),
            "Recall@20": float(recall.mean()),
            "MeanDeltaVsBase": float(delta.mean()),
            "WinsVsBase": int(np.sum(delta > 1e-12)),
            "TiesVsBase": int(np.sum(np.abs(delta) <= 1e-12)),
            "LossesVsBase": int(np.sum(delta < -1e-12)),
            "ChangedRankingRateVsBase": float(changed.mean()),
            "Top20JaccardVsBase": float(jac.mean()),
            "Top20OverlapVsBase": float(overlap.mean()),
            "MeanAbsRankShiftVsBase": float(shift.mean()),
            "HitUsers": int(recall.sum()),
        })
        for idx, uid in enumerate(users):
            per_user_rows.append({
                "Dataset": dataset,
                "CandidateSetting": candidate_setting,
                "uid": int(uid),
                "Method": name,
                "Target": int(targets[idx]),
                "NDCG@20": float(ndcg[idx]),
                "Recall@20": float(recall[idx]),
                "DeltaVsBase": float(delta[idx]),
                "TargetRank@20PlusOne": int(ranks[idx]),
                "Top20JaccardVsBase": float(jac[idx]),
                "Top20OverlapVsBase": float(overlap[idx]),
                "MeanAbsRankShiftVsBase": float(shift[idx]),
            })

    any_hit = np.zeros(len(users), dtype=bool)
    any_delta = np.zeros(len(users), dtype=bool)
    any_ranking_changed = np.zeros(len(users), dtype=bool)
    for name, ranked in methods.items():
        ndcg, recall = method_metrics[name]
        any_hit |= recall > 0.0
        any_delta |= np.abs(ndcg - base_ndcg) > 1e-12
        if name != "LightGCN":
            any_ranking_changed |= topk_jaccard(ranked, base_ranked, topk) < 0.999999
    subset_rows = []
    masks = {
        "all_users": np.ones(len(users), dtype=bool),
        "any_method_hit": any_hit,
        "any_ndcg_delta_vs_base": any_delta,
        "any_ranking_changed_vs_base": any_ranking_changed,
    }
    for subset, mask in masks.items():
        if not mask.any():
            continue
        for name, (ndcg, recall) in method_metrics.items():
            delta = ndcg - base_ndcg
            subset_rows.append({
                "Dataset": dataset,
                "CandidateSetting": candidate_setting,
                "Subset": subset,
                "Method": name,
                "Users": int(mask.sum()),
                "NDCG@20": float(ndcg[mask].mean()),
                "Recall@20": float(recall[mask].mean()),
                "MeanDeltaVsBase": float(delta[mask].mean()),
                "WinsVsBase": int(np.sum(delta[mask] > 1e-12)),
                "TiesVsBase": int(np.sum(np.abs(delta[mask]) <= 1e-12)),
                "LossesVsBase": int(np.sum(delta[mask] < -1e-12)),
            })
    return pd.DataFrame(method_rows_out), pd.DataFrame(per_user_rows), pd.DataFrame(subset_rows)


def load_ml1m_methods(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    direct_npz = np.load(args.ml1m_direct_run / "candidates_lightgcn_test.npz")
    profile_npz = np.load(args.ml1m_profile_run / "candidates_lightgcn_test.npz")
    users = direct_npz["users"].astype(np.int64)
    targets = direct_npz["targets"].astype(np.int64)
    train, val, test, _ = read_interaction_split(args.ml1m_datadir)
    _, n_items = infer_shape(train, val, test)
    meta = read_item_metadata(args.ml1m_movies_path, args.ml1m_datadir / "mappings.json", n_items)
    batch = MlCandidateBatch(
        users=profile_npz["users"].astype(np.int64),
        targets=profile_npz["targets"].astype(np.int64),
        candidates=profile_npz["candidates"].astype(np.int64),
        scores=profile_npz["scores"].astype(np.float32),
        split_name="test",
    )
    records = load_ml_claim_records(args.ml1m_profile_run / "claim_support.jsonl")
    profile_scores = ml_profile_scores_for_batch(batch, records, meta, "weighted")
    perf = pd.read_csv(args.ml1m_profile_run / "table2_recommendation_performance.csv")
    lam = float(perf.loc[perf["Method"] == "LightGCN + Evidence-Weighted Repair", "SelectedLambda"].iloc[0])
    methods = {
        "LightGCN": direct_npz["candidates"][:, : args.topk].astype(np.int64),
        "DeepSeek Direct Rerank": np.load(args.ml1m_direct_run / "reranked_lightgcn_original_top20.npy").astype(np.int64),
        "Profile Rerank EGPR": ml_ranked_from_profile_scores(batch, profile_scores, lam, args.topk),
    }
    return users, targets, methods


def load_yelp_methods(args: argparse.Namespace, direct_dir: Path, candidate_size: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    train, val, test, _ = read_interaction_split(args.yelp_datadir)
    n_users, n_items = infer_shape(train, val, test)
    ordered_histories = yelp_build_ordered_histories(train, n_users)
    records = load_yelp_claim_records(args.yelp_profile_run / "claim_support.jsonl")
    val_data = np.load(args.yelp_profile_run / "candidates_lightgcn_val.npz")
    test_data = np.load(args.yelp_profile_run / "candidates_lightgcn_test.npz")
    users = test_data["users"].astype(np.int64)
    targets = test_data["targets"].astype(np.int64)
    val_batch = YelpCandidateBatch(
        users=val_data["users"].astype(np.int64),
        targets=val_data["targets"].astype(np.int64),
        candidates=val_data["candidates"][:, :candidate_size].astype(np.int64),
        scores=val_data["scores"][:, :candidate_size].astype(np.float32),
        split_name="val",
    )
    test_batch = YelpCandidateBatch(
        users=users,
        targets=targets,
        candidates=test_data["candidates"][:, :candidate_size].astype(np.int64),
        scores=test_data["scores"][:, :candidate_size].astype(np.float32),
        split_name="test",
    )
    profile_users = sorted(set(val_batch.users.astype(int).tolist() + test_batch.users.astype(int).tolist()))
    max_batch = YelpCandidateBatch(
        users=test_data["users"].astype(np.int64),
        targets=test_data["targets"].astype(np.int64),
        candidates=test_data["candidates"].astype(np.int64),
        scores=test_data["scores"].astype(np.float32),
        split_name="test",
    )
    needed = needed_iids_for_metadata(profile_users, ordered_histories, max_batch, max_batch, args.history_limit)
    print(f"building Yelp metadata for affected-user analysis candidate_size={candidate_size}: {len(needed):,} items", flush=True)
    meta = build_yelp_metadata_for_iids(args.yelp_datadir, args.yelp_reviews_path, args.yelp_business_path, needed, n_items, args.metadata_progress_every)
    # Selected lambda is 0.3 for both top-50 and top-100 in the candidate-size report; read defensively.
    lambda_table = pd.read_csv(ROOT / "results/egpr_profile_repair/yelp_direct_vs_profile/profile_candidate_sizes/profile_candidate_performance.csv")
    lam = float(lambda_table[(lambda_table["CandidateSet"] == candidate_size) & (lambda_table["Method"] == "LightGCN + Evidence-Weighted Repair")]["SelectedLambda"].iloc[0])
    profile_scores = yelp_profile_scores_for_batch(test_batch, records, meta, "weighted")
    profile_ranked = yelp_ranked_from_profile_scores(test_batch, profile_scores, lam, args.topk)
    methods = {
        "LightGCN": test_batch.candidates[:, : args.topk].astype(np.int64),
        "DeepSeek Direct Rerank": np.load(direct_dir / "direct_rerank_top20.npy").astype(np.int64),
        "Profile Rerank EGPR": profile_ranked,
    }
    return users, targets, methods


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
    method_tables = []
    per_user_tables = []
    subset_tables = []

    print("analyzing ML-1M", flush=True)
    users, targets, methods = load_ml1m_methods(args)
    method_df, per_user_df, subset_df = method_rows("ML-1M", "direct top-50 / profile top-100", users, targets, methods, args.topk)
    method_tables.append(method_df); per_user_tables.append(per_user_df); subset_tables.append(subset_df)

    print("analyzing Yelp top-50", flush=True)
    users, targets, methods = load_yelp_methods(args, args.yelp_direct_top50, 50)
    method_df, per_user_df, subset_df = method_rows("Yelp", "matched top-50", users, targets, methods, args.topk)
    method_tables.append(method_df); per_user_tables.append(per_user_df); subset_tables.append(subset_df)

    print("analyzing Yelp top-100", flush=True)
    users, targets, methods = load_yelp_methods(args, args.yelp_direct_top100, 100)
    method_df, per_user_df, subset_df = method_rows("Yelp", "matched top-100", users, targets, methods, args.topk)
    method_tables.append(method_df); per_user_tables.append(per_user_df); subset_tables.append(subset_df)

    methods_all = pd.concat(method_tables, ignore_index=True)
    per_user_all = pd.concat(per_user_tables, ignore_index=True)
    subsets_all = pd.concat(subset_tables, ignore_index=True)
    methods_all.to_csv(args.outdir / "affected_user_method_summary.csv", index=False)
    per_user_all.to_csv(args.outdir / "affected_user_per_user_metrics.csv", index=False)
    subsets_all.to_csv(args.outdir / "affected_user_subset_summary.csv", index=False)

    focus_methods = methods_all[methods_all["Method"].isin(["DeepSeek Direct Rerank", "Profile Rerank EGPR"])].copy()
    focus_subsets = subsets_all[(subsets_all["Subset"].isin(["any_method_hit", "any_ndcg_delta_vs_base"])) & (subsets_all["Method"].isin(["LightGCN", "DeepSeek Direct Rerank", "Profile Rerank EGPR"]))].copy()
    lines = [
        "# Affected-User and Rank-Disruption Analysis",
        "",
        "Affected-user diagnostics explain why all-user paired significance is weak: most users are unchanged at NDCG@20, while method differences concentrate on a small subset of users whose hit status or target rank changes.",
        "",
        "## Method-Level Rank Disruption",
        "",
        markdown_table(focus_methods),
        "",
        "## Affected-User Subsets",
        "",
        markdown_table(focus_subsets),
        "",
        "## Interpretation",
        "",
        "- Top20JaccardVsBase and Top20OverlapVsBase measure how much a method disrupts the base recommendation list; lower values mean more disruption.",
        "- Any-method-hit and any-delta subsets should be treated as diagnostics only, not replacements for all-user metrics.",
        "- Large tie counts in all-user tests are expected because most sampled users have no hit under any method.",
        "",
        "## Artifacts",
        "",
        "- `affected_user_method_summary.csv`",
        "- `affected_user_subset_summary.csv`",
        "- `affected_user_per_user_metrics.csv`",
    ]
    (args.outdir / "affected_user_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done. Report: {args.outdir / 'affected_user_analysis_report.md'}", flush=True)


if __name__ == "__main__":
    main()
