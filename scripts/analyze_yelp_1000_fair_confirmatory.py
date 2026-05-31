#!/usr/bin/env python3
"""Yelp 1000 fair direct-vs-profile confirmatory analysis."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
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
from run_yelp_profile_rerank_pilot import (  # noqa: E402
    CandidateBatch,
    ClaimRecord,
    build_ordered_histories,
    build_yelp_metadata_for_iids,
    needed_iids_for_metadata,
    profile_scores_for_batch,
    ranked_from_profile_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default=ROOT / "data/yelp_day1")
    parser.add_argument("--profile-run-dir", type=Path, default=ROOT / "results/egpr_profile_repair/yelp_seed42_deepseek_1000_expressive5")
    parser.add_argument("--profile-candidate-dir", type=Path, default=ROOT / "results/egpr_profile_repair/yelp_direct_vs_profile/profile_candidate_sizes_1000")
    parser.add_argument("--direct-top50-dir", type=Path, default=ROOT / "results/egpr_profile_repair/yelp_direct_vs_profile/direct_rerank_1000_top50")
    parser.add_argument("--direct-top100-dir", type=Path, default=ROOT / "results/egpr_profile_repair/yelp_direct_vs_profile/direct_rerank_1000_top100")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/egpr_profile_repair/yelp_direct_vs_profile/yelp_1000_fair_confirmatory")
    parser.add_argument("--reviews-path", type=Path, default=Path("/root/autodl-tmp/yelp-raw/yelp_academic_dataset_review.json"))
    parser.add_argument("--business-path", type=Path, default=Path("/root/autodl-tmp/yelp-raw/yelp_academic_dataset_business.json"))
    parser.add_argument("--candidate-sizes", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--history-limit", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-progress-every", type=int, default=500000)
    parser.add_argument("--input-price-per-1m", type=float, default=0.14)
    parser.add_argument("--output-price-per-1m", type=float, default=0.28)
    return parser.parse_args()


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


def load_batch(path: Path, split_name: str, candidate_size: int) -> CandidateBatch:
    data = np.load(path / f"candidates_lightgcn_{split_name}.npz")
    return CandidateBatch(
        users=data["users"].astype(np.int64),
        targets=data["targets"].astype(np.int64),
        candidates=data["candidates"][:, :candidate_size].astype(np.int64),
        scores=data["scores"][:, :candidate_size].astype(np.float32),
        split_name=split_name,
    )


def metrics_from_ranked(ranked: np.ndarray, targets: np.ndarray, topk: int) -> Dict[str, np.ndarray]:
    ndcg = np.zeros(len(targets), dtype=np.float64)
    recall = np.zeros(len(targets), dtype=np.float64)
    for row, target_np in enumerate(targets):
        pos = np.flatnonzero(ranked[row, :topk] == int(target_np))
        if len(pos):
            recall[row] = 1.0
            ndcg[row] = 1.0 / math.log2(int(pos[0]) + 2)
    return {"NDCG@20": ndcg, "Recall@20": recall, "HitRate@20": recall.copy()}


def reliability(base_ndcg: np.ndarray, method_ndcg: np.ndarray) -> Dict[str, float]:
    delta = method_ndcg - base_ndcg
    pos = delta[delta > 0.0]
    neg = delta[delta < 0.0]
    return {
        "HarmRate": float(np.mean(delta < 0.0)),
        "PositiveGainRate": float(np.mean(delta > 0.0)),
        "MeanDeltaNDCG@20": float(delta.mean()),
        "PositiveGainSum": float(pos.sum()) if len(pos) else 0.0,
        "NegativeGainSum": float(neg.sum()) if len(neg) else 0.0,
        "GainHarmRatio": float(pos.sum() / abs(neg.sum())) if len(neg) and abs(float(neg.sum())) > 0.0 else np.inf,
    }


def bootstrap_ci(delta: np.ndarray, rng: np.random.Generator, samples: int) -> Tuple[float, float]:
    means = np.empty(samples, dtype=np.float64)
    n = len(delta)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(delta[idx].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def sign_flip_p(delta: np.ndarray, rng: np.random.Generator, samples: int) -> float:
    observed = abs(float(delta.mean()))
    nonzero = delta[np.abs(delta) > 1e-12]
    if observed <= 0.0 or len(nonzero) == 0:
        return 1.0
    count = 0
    for _ in range(samples):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(nonzero))
        stat = abs(float((nonzero * signs).sum() / len(delta)))
        if stat >= observed - 1e-15:
            count += 1
    return float((count + 1) / (samples + 1))


def topk_jaccard(a: np.ndarray, b: np.ndarray, topk: int) -> np.ndarray:
    out = np.zeros(len(a), dtype=np.float64)
    for row in range(len(a)):
        left = set(int(x) for x in a[row, :topk])
        right = set(int(x) for x in b[row, :topk])
        out[row] = len(left & right) / len(left | right) if left or right else 1.0
    return out


def topk_overlap(a: np.ndarray, b: np.ndarray, topk: int) -> np.ndarray:
    out = np.zeros(len(a), dtype=np.float64)
    for row in range(len(a)):
        left = set(int(x) for x in a[row, :topk])
        right = set(int(x) for x in b[row, :topk])
        out[row] = len(left & right) / float(topk)
    return out


def mean_abs_rank_shift(a: np.ndarray, b: np.ndarray, topk: int) -> np.ndarray:
    out = np.zeros(len(a), dtype=np.float64)
    for row in range(len(a)):
        left = {int(item): pos for pos, item in enumerate(a[row, :topk])}
        right = {int(item): pos for pos, item in enumerate(b[row, :topk])}
        common = set(left) & set(right)
        out[row] = float(np.mean([abs(left[item] - right[item]) for item in common])) if common else float(topk)
    return out


def profile_generation_cost(args: argparse.Namespace) -> float:
    trace_path = args.profile_run_dir / "profile_cost_trace.csv"
    if not trace_path.exists():
        return float("nan")
    trace = pd.read_csv(trace_path)
    return float((trace["input_tokens"].sum() / 1_000_000.0) * args.input_price_per_1m + (trace["output_tokens"].sum() / 1_000_000.0) * args.output_price_per_1m)


def direct_cost(path: Path) -> float:
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    return float(manifest["estimated_cost_usd"])


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "-" if pd.isna(x) else ("inf" if np.isinf(x) else f"{x:.6f}"))
        else:
            display[col] = display[col].astype(str)
    lines = ["| " + " | ".join(display.columns) + " |", "| " + " | ".join(["---"] * len(display.columns)) + " |"]
    for row in display.values.tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train, val, test, _ = read_interaction_split(args.datadir)
    n_users, n_items = infer_shape(train, val, test)
    histories = build_ordered_histories(train, n_users)
    records = load_claim_records(args.profile_run_dir / "claim_support.jsonl")

    max_size = max(args.candidate_sizes)
    val_max = load_batch(args.profile_run_dir, "val", max_size)
    test_max = load_batch(args.profile_run_dir, "test", max_size)
    profile_users = sorted(set(val_max.users.astype(int).tolist() + test_max.users.astype(int).tolist()))
    needed = needed_iids_for_metadata(profile_users, histories, val_max, test_max, args.history_limit)
    print(f"building Yelp metadata for fair confirmatory analysis: {len(needed):,} items", flush=True)
    meta = build_yelp_metadata_for_iids(args.datadir, args.reviews_path, args.business_path, needed, n_items, args.metadata_progress_every)

    perf_report = pd.read_csv(args.profile_candidate_dir / "profile_candidate_performance.csv")
    direct_dirs = {50: args.direct_top50_dir, 100: args.direct_top100_dir}
    profile_cost = profile_generation_cost(args)

    summary_rows: List[Dict[str, object]] = []
    stats_rows: List[Dict[str, object]] = []
    rank_rows: List[Dict[str, object]] = []
    subset_rows: List[Dict[str, object]] = []

    for size in args.candidate_sizes:
        print(f"analyzing candidate_size={size}", flush=True)
        test_batch = load_batch(args.profile_run_dir, "test", size)
        base_ranked = test_batch.candidates[:, : args.topk].astype(np.int64)
        direct_ranked = np.load(direct_dirs[size] / "direct_rerank_top20.npy").astype(np.int64)
        methods_ranked: Dict[str, np.ndarray] = {
            "LightGCN": base_ranked,
            "DeepSeek Direct Rerank": direct_ranked,
        }
        for method_key, method_name in [("raw", "Profile Rerank Raw"), ("remove", "Profile Rerank Remove"), ("weighted", "Profile Rerank EGPR")]:
            row = perf_report[(perf_report["CandidateSet"] == size) & (perf_report["Method"] == {
                "raw": "LightGCN + Raw Profile",
                "remove": "LightGCN + Remove Repair",
                "weighted": "LightGCN + Evidence-Weighted Repair",
            }[method_key])]
            lam = float(row["SelectedLambda"].iloc[0])
            scores = profile_scores_for_batch(test_batch, records, meta, method_key)
            methods_ranked[method_name] = ranked_from_profile_scores(test_batch, scores, lam, args.topk)

        method_metrics = {name: metrics_from_ranked(ranked, test_batch.targets, args.topk) for name, ranked in methods_ranked.items()}
        base_ndcg = method_metrics["LightGCN"]["NDCG@20"]
        direct_usd = direct_cost(direct_dirs[size])
        for name, metrics in method_metrics.items():
            cost = 0.0
            cost_ratio = 0.0
            if name == "DeepSeek Direct Rerank":
                cost = direct_usd
                cost_ratio = 1.0
            elif name.startswith("Profile"):
                cost = profile_cost
                cost_ratio = profile_cost / direct_usd if direct_usd > 0.0 else np.nan
            rel = reliability(base_ndcg, metrics["NDCG@20"]) if name != "LightGCN" else {
                "HarmRate": 0.0,
                "PositiveGainRate": 0.0,
                "MeanDeltaNDCG@20": 0.0,
                "PositiveGainSum": 0.0,
                "NegativeGainSum": 0.0,
                "GainHarmRatio": np.nan,
            }
            summary_rows.append({
                "CandidateSet": size,
                "Method": name,
                "NDCG@20": float(metrics["NDCG@20"].mean()),
                "Recall@20": float(metrics["Recall@20"].mean()),
                "DeltaVsBase": float(metrics["NDCG@20"].mean() - base_ndcg.mean()),
                "HarmRate": rel["HarmRate"],
                "GHR": rel["GainHarmRatio"],
                "TestCostUSD": cost,
                "CostVsDirect": cost_ratio,
            })

        comparisons = [
            ("Direct vs Base", "DeepSeek Direct Rerank", "LightGCN"),
            ("Profile Raw vs Base", "Profile Rerank Raw", "LightGCN"),
            ("Profile EGPR vs Base", "Profile Rerank EGPR", "LightGCN"),
            ("Profile Raw vs Direct", "Profile Rerank Raw", "DeepSeek Direct Rerank"),
            ("Profile EGPR vs Direct", "Profile Rerank EGPR", "DeepSeek Direct Rerank"),
            ("Profile EGPR vs Raw", "Profile Rerank EGPR", "Profile Rerank Raw"),
        ]
        for label, left_name, right_name in comparisons:
            for metric in ["NDCG@20", "Recall@20"]:
                left = method_metrics[left_name][metric]
                right = method_metrics[right_name][metric]
                delta = left - right
                lo, hi = bootstrap_ci(delta, rng, args.bootstrap_samples)
                stats_rows.append({
                    "CandidateSet": size,
                    "Comparison": label,
                    "Metric": metric,
                    "Users": int(len(delta)),
                    "MeanDelta": float(delta.mean()),
                    "BootstrapCI95Low": lo,
                    "BootstrapCI95High": hi,
                    "SignFlipPValueTwoSided": sign_flip_p(delta, rng, args.permutation_samples),
                    "Wins": int(np.sum(delta > 1e-12)),
                    "Ties": int(np.sum(np.abs(delta) <= 1e-12)),
                    "Losses": int(np.sum(delta < -1e-12)),
                })

        any_hit = np.zeros(len(test_batch.users), dtype=bool)
        for metrics in method_metrics.values():
            any_hit |= metrics["Recall@20"] > 0.0
        for name, ranked in methods_ranked.items():
            metrics = method_metrics[name]
            delta = metrics["NDCG@20"] - base_ndcg
            if name == "LightGCN":
                jac = np.ones(len(test_batch.users), dtype=np.float64)
                overlap = np.ones(len(test_batch.users), dtype=np.float64)
                shift = np.zeros(len(test_batch.users), dtype=np.float64)
            else:
                jac = topk_jaccard(ranked, base_ranked, args.topk)
                overlap = topk_overlap(ranked, base_ranked, args.topk)
                shift = mean_abs_rank_shift(ranked, base_ranked, args.topk)
            rank_rows.append({
                "CandidateSet": size,
                "Method": name,
                "Users": int(len(test_batch.users)),
                "NDCG@20": float(metrics["NDCG@20"].mean()),
                "Recall@20": float(metrics["Recall@20"].mean()),
                "WinsVsBase": int(np.sum(delta > 1e-12)),
                "TiesVsBase": int(np.sum(np.abs(delta) <= 1e-12)),
                "LossesVsBase": int(np.sum(delta < -1e-12)),
                "Top20OverlapVsBase": float(overlap.mean()),
                "Top20JaccardVsBase": float(jac.mean()),
                "MeanAbsRankShiftVsBase": float(shift.mean()),
                "HitUsers": int(metrics["Recall@20"].sum()),
            })
            if any_hit.any():
                subset_rows.append({
                    "CandidateSet": size,
                    "Subset": "any_method_hit",
                    "Method": name,
                    "Users": int(any_hit.sum()),
                    "NDCG@20": float(metrics["NDCG@20"][any_hit].mean()),
                    "Recall@20": float(metrics["Recall@20"][any_hit].mean()),
                    "MeanDeltaVsBase": float(delta[any_hit].mean()),
                    "WinsVsBase": int(np.sum(delta[any_hit] > 1e-12)),
                    "TiesVsBase": int(np.sum(np.abs(delta[any_hit]) <= 1e-12)),
                    "LossesVsBase": int(np.sum(delta[any_hit] < -1e-12)),
                })

    summary = pd.DataFrame(summary_rows)
    stats = pd.DataFrame(stats_rows)
    rank = pd.DataFrame(rank_rows)
    subsets = pd.DataFrame(subset_rows)
    summary.to_csv(args.outdir / "yelp_1000_fair_summary.csv", index=False)
    stats.to_csv(args.outdir / "yelp_1000_paired_stats.csv", index=False)
    rank.to_csv(args.outdir / "yelp_1000_rank_disruption.csv", index=False)
    subsets.to_csv(args.outdir / "yelp_1000_affected_subset.csv", index=False)

    ndcg_stats = stats[stats["Metric"] == "NDCG@20"].copy()
    lines = [
        "# Yelp 1000 Fair Confirmatory Analysis",
        "",
        f"Profile run: `{args.profile_run_dir}`.",
        "Direct rerank and profile rerank are evaluated on matched LightGCN top-50 and top-100 candidate sets.",
        f"Profile generation cost is reusable and fixed across candidate sizes: `{profile_cost:.6f}` USD.",
        "",
        "## Fair Accuracy / Reliability / Cost",
        "",
        markdown_table(summary),
        "",
        "## Paired NDCG Tests",
        "",
        markdown_table(ndcg_stats),
        "",
        "## Rank Disruption",
        "",
        markdown_table(rank),
        "",
        "## Affected-Hit Subset",
        "",
        markdown_table(subsets),
        "",
        "## Interpretation",
        "",
        "- On 1000 Yelp users, Direct DeepSeek reranking remains below LightGCN under both matched top-50 and matched top-100 candidate sets.",
        "- Raw expressive profile reranking is the strongest profile setting on Yelp 1000; Evidence-Weighted EGPR reduces unsupported-claim weight but harms utility relative to Raw.",
        "- Profile reranking remains cheaper than Direct, especially under top-100 candidates, but its rank-disruption advantage is not uniform on Yelp 1000; the main stable advantage is lower cost with better Raw-profile utility than Direct.",
        "",
        "## Artifacts",
        "",
        "- `yelp_1000_fair_summary.csv`",
        "- `yelp_1000_paired_stats.csv`",
        "- `yelp_1000_rank_disruption.csv`",
        "- `yelp_1000_affected_subset.csv`",
        "- `run_manifest.json`",
        "",
    ]
    (args.outdir / "yelp_1000_fair_confirmatory_report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "status": "completed",
        "profile_run_dir": str(args.profile_run_dir),
        "direct_top50_dir": str(args.direct_top50_dir),
        "direct_top100_dir": str(args.direct_top100_dir),
        "profile_candidate_dir": str(args.profile_candidate_dir),
        "candidate_sizes": args.candidate_sizes,
        "users": 1000,
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "profile_generation_cost_usd": profile_cost,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Report: {args.outdir / 'yelp_1000_fair_confirmatory_report.md'}", flush=True)


if __name__ == "__main__":
    main()
