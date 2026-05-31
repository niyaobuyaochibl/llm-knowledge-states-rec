#!/usr/bin/env python3
"""Amazon Books fair direct-vs-profile confirmatory analysis."""

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

from run_amazon_books_profile_rerank_pilot import (  # noqa: E402
    CandidateBatch,
    ClaimRecord,
    load_batch,
    load_metadata,
    profile_scores,
    ranked_from_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-run-dir", type=Path, default=ROOT / "results/egpr_profile_repair/amazon_books_seed42_lightgcn_1000")
    parser.add_argument("--profile-run-dir", type=Path, default=ROOT / "results/egpr_profile_repair/amazon_books_seed42_deepseek_1000_expressive5")
    parser.add_argument("--direct-top50-dir", type=Path, default=ROOT / "results/egpr_profile_repair/amazon_books_direct_vs_profile/direct_rerank_1000_top50")
    parser.add_argument("--direct-top100-dir", type=Path, default=ROOT / "results/egpr_profile_repair/amazon_books_direct_vs_profile/direct_rerank_1000_top100")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/egpr_profile_repair/amazon_books_direct_vs_profile/amazon_books_1000_fair_confirmatory")
    parser.add_argument("--datadir", type=Path, default=ROOT / "data/amazon_books_subset")
    parser.add_argument("--candidate-sizes", nargs="+", type=int, default=[50, 100])
    parser.add_argument("--lambda-grid", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
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


def truncate_batch(batch: CandidateBatch, candidate_size: int) -> CandidateBatch:
    return CandidateBatch(
        users=batch.users,
        targets=batch.targets,
        candidates=batch.candidates[:, :candidate_size].astype(np.int64),
        scores=batch.scores[:, :candidate_size].astype(np.float32),
        split_name=batch.split_name,
    )


def metrics_from_ranked_array(ranked: np.ndarray, targets: np.ndarray, topk: int) -> Dict[str, np.ndarray]:
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


def topk_overlap(a: np.ndarray, b: np.ndarray, topk: int) -> np.ndarray:
    out = np.zeros(len(a), dtype=np.float64)
    for row in range(len(a)):
        left = set(int(x) for x in a[row, :topk])
        right = set(int(x) for x in b[row, :topk])
        out[row] = len(left & right) / float(topk)
    return out


def topk_jaccard(a: np.ndarray, b: np.ndarray, topk: int) -> np.ndarray:
    out = np.zeros(len(a), dtype=np.float64)
    for row in range(len(a)):
        left = set(int(x) for x in a[row, :topk])
        right = set(int(x) for x in b[row, :topk])
        out[row] = len(left & right) / len(left | right) if left or right else 1.0
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


def select_lambda(val_batch: CandidateBatch, records: Mapping[int, List[ClaimRecord]], meta, method_key: str, grid: List[float], topk: int) -> Tuple[float, float]:
    pscores = profile_scores(val_batch, records, meta, method_key)
    best_lam = grid[0]
    best_ndcg = -1.0
    for lam in grid:
        ranked = ranked_from_scores(val_batch, pscores, lam, topk)
        ndcg = float(metrics_from_ranked_array(ranked, val_batch.targets, topk)["NDCG@20"].mean())
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_lam = lam
    return float(best_lam), float(best_ndcg)


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

    meta = load_metadata(args.datadir)
    records = load_claim_records(args.profile_run_dir / "claim_support.jsonl")
    val_full = load_batch(args.candidate_run_dir, "val")
    test_full = load_batch(args.candidate_run_dir, "test")
    direct_dirs = {50: args.direct_top50_dir, 100: args.direct_top100_dir}
    profile_cost = profile_generation_cost(args)

    summary_rows: List[Dict[str, object]] = []
    stats_rows: List[Dict[str, object]] = []
    rank_rows: List[Dict[str, object]] = []
    subset_rows: List[Dict[str, object]] = []
    lambda_rows: List[Dict[str, object]] = []

    for size in args.candidate_sizes:
        print(f"analyzing Amazon Books candidate_size={size}", flush=True)
        val_batch = truncate_batch(val_full, size)
        test_batch = truncate_batch(test_full, size)
        base_ranked = test_batch.candidates[:, : args.topk].astype(np.int64)
        direct_ranked = np.load(direct_dirs[size] / "direct_rerank_top20.npy").astype(np.int64)
        methods_ranked: Dict[str, np.ndarray] = {
            "LightGCN": base_ranked,
            "DeepSeek Direct Rerank": direct_ranked,
        }
        for method_key, method_name in [
            ("raw", "Profile Rerank Raw"),
            ("remove", "Profile Rerank Remove"),
            ("weighted", "Profile Rerank EGPR"),
        ]:
            lam, val_ndcg = select_lambda(val_batch, records, meta, method_key, args.lambda_grid, args.topk)
            lambda_rows.append({"CandidateSet": size, "Method": method_name, "SelectedLambda": lam, "ValidationNDCG@20": val_ndcg})
            scores = profile_scores(test_batch, records, meta, method_key)
            methods_ranked[method_name] = ranked_from_scores(test_batch, scores, lam, args.topk)

        method_metrics = {name: metrics_from_ranked_array(ranked, test_batch.targets, args.topk) for name, ranked in methods_ranked.items()}
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
            ("Profile Remove vs Base", "Profile Rerank Remove", "LightGCN"),
            ("Profile EGPR vs Base", "Profile Rerank EGPR", "LightGCN"),
            ("Profile Raw vs Direct", "Profile Rerank Raw", "DeepSeek Direct Rerank"),
            ("Profile Remove vs Direct", "Profile Rerank Remove", "DeepSeek Direct Rerank"),
            ("Profile EGPR vs Direct", "Profile Rerank EGPR", "DeepSeek Direct Rerank"),
            ("Profile EGPR vs Raw", "Profile Rerank EGPR", "Profile Rerank Raw"),
            ("Profile Remove vs Raw", "Profile Rerank Remove", "Profile Rerank Raw"),
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
    lambdas = pd.DataFrame(lambda_rows)
    summary.to_csv(args.outdir / "amazon_books_1000_fair_summary.csv", index=False)
    stats.to_csv(args.outdir / "amazon_books_1000_paired_stats.csv", index=False)
    rank.to_csv(args.outdir / "amazon_books_1000_rank_disruption.csv", index=False)
    subsets.to_csv(args.outdir / "amazon_books_1000_affected_subset.csv", index=False)
    lambdas.to_csv(args.outdir / "amazon_books_1000_profile_lambdas.csv", index=False)

    ndcg_stats = stats[stats["Metric"] == "NDCG@20"].copy()
    lines = [
        "# Amazon Books 1000 Fair Confirmatory Analysis",
        "",
        f"Profile run: `{args.profile_run_dir}`.",
        "Direct rerank and profile rerank are evaluated on matched LightGCN top-50 and top-100 candidate sets.",
        f"Profile generation cost is reusable and fixed across candidate sizes: `{profile_cost:.6f}` USD.",
        "",
        "## Fair Accuracy / Reliability / Cost",
        "",
        markdown_table(summary),
        "",
        "## Profile Lambda Selection",
        "",
        markdown_table(lambdas),
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
        "## Artifacts",
        "",
        "- `amazon_books_1000_fair_summary.csv`",
        "- `amazon_books_1000_paired_stats.csv`",
        "- `amazon_books_1000_rank_disruption.csv`",
        "- `amazon_books_1000_affected_subset.csv`",
        "- `amazon_books_1000_profile_lambdas.csv`",
        "- `run_manifest.json`",
        "",
    ]
    (args.outdir / "amazon_books_1000_fair_confirmatory_report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "status": "completed",
        "candidate_run_dir": str(args.candidate_run_dir),
        "profile_run_dir": str(args.profile_run_dir),
        "direct_top50_dir": str(args.direct_top50_dir),
        "direct_top100_dir": str(args.direct_top100_dir),
        "candidate_sizes": args.candidate_sizes,
        "users": int(len(test_full.users)),
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "profile_generation_cost_usd": profile_cost,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Report: {args.outdir / 'amazon_books_1000_fair_confirmatory_report.md'}", flush=True)


if __name__ == "__main__":
    main()
