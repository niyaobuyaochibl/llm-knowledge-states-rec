#!/usr/bin/env python3
"""Bootstrap and paired sign-flip tests for profile-before-ranking pilots."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/egpr_profile_repair/statistical_tests")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--ml1m-datadir", type=Path, default=ROOT / "data/ml1m")
    parser.add_argument("--ml1m-movies-path", type=Path, default=Path("/root/autodl-tmp/projects_archive/MGPrompt-CDR/data/raw/ml-1m/movies.dat"))
    parser.add_argument("--ml1m-direct-run", type=Path, default=ROOT / "results/llm_selective/ml1m_seed42_deepseek_500")
    parser.add_argument("--ml1m-profile-run", type=Path, default=ROOT / "results/egpr_profile_repair/ml1m_seed42_deepseek_500_expressive5")
    parser.add_argument("--yelp-profile-run", type=Path, default=ROOT / "results/egpr_profile_repair/yelp_seed42_deepseek_300_expressive5")
    parser.add_argument("--yelp-direct-top50", type=Path, default=ROOT / "results/egpr_profile_repair/yelp_direct_vs_profile/direct_rerank_300")
    parser.add_argument("--yelp-direct-top100", type=Path, default=ROOT / "results/egpr_profile_repair/yelp_direct_vs_profile/direct_rerank_300_top100")
    return parser.parse_args()


def ndcg_recall_from_ranked(ranked: np.ndarray, targets: np.ndarray, topk: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    ndcg = np.zeros(len(targets), dtype=np.float64)
    recall = np.zeros(len(targets), dtype=np.float64)
    for row, target_np in enumerate(targets):
        target = int(target_np)
        positions = np.flatnonzero(ranked[row, :topk] == target)
        if len(positions):
            recall[row] = 1.0
            ndcg[row] = 1.0 / math.log2(int(positions[0]) + 2)
    return ndcg, recall


def load_ml_claim_records(path: Path) -> Dict[int, List[MlClaimRecord]]:
    records: Dict[int, List[MlClaimRecord]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rec = MlClaimRecord(
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


def load_ml1m_arrays(args: argparse.Namespace) -> Dict[str, Dict[str, np.ndarray]]:
    direct_npz = np.load(args.ml1m_direct_run / "candidates_lightgcn_test.npz")
    profile_npz = np.load(args.ml1m_profile_run / "candidates_lightgcn_test.npz")
    users = direct_npz["users"].astype(np.int64)
    targets = direct_npz["targets"].astype(np.int64)
    if not np.array_equal(users, profile_npz["users"].astype(np.int64)):
        raise RuntimeError("ML-1M direct/profile users differ")
    if not np.array_equal(targets, profile_npz["targets"].astype(np.int64)):
        raise RuntimeError("ML-1M direct/profile targets differ")

    base_ranked = direct_npz["candidates"][:, : args.topk].astype(np.int64)
    direct_ranked = np.load(args.ml1m_direct_run / "reranked_lightgcn_original_top20.npy").astype(np.int64)

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
    profile_ranked = ml_ranked_from_profile_scores(batch, profile_scores, lam, args.topk)

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for method, ranked in [
        ("LightGCN", base_ranked),
        ("DeepSeek Direct Rerank", direct_ranked),
        ("Profile Rerank EGPR", profile_ranked),
    ]:
        ndcg, recall = ndcg_recall_from_ranked(ranked, targets, args.topk)
        out[method] = {"NDCG@20": ndcg, "Recall@20": recall}
    return out


def load_yelp_arrays(args: argparse.Namespace, direct_dir: Path, profile_candidate_size: int) -> Dict[str, Dict[str, np.ndarray]]:
    profile_npz = np.load(args.yelp_profile_run / "candidates_lightgcn_test.npz")
    targets = profile_npz["targets"].astype(np.int64)
    base_ranked = profile_npz["candidates"][:, : args.topk].astype(np.int64)
    direct_ranked = np.load(direct_dir / "direct_rerank_top20.npy").astype(np.int64)
    # Candidate-size profile EGPR result is identical for top-50 and top-100 in the current run,
    # but compute it from saved per-user metrics for exact consistency with the report.
    profile_per_user = pd.read_csv(args.yelp_profile_run / "per_user_lightgcn_evidence_weighted_repair.csv")
    lightgcn_per_user = pd.read_csv(args.yelp_profile_run / "per_user_lightgcn.csv")

    base_ndcg, base_recall = ndcg_recall_from_ranked(base_ranked, targets, args.topk)
    direct_ndcg, direct_recall = ndcg_recall_from_ranked(direct_ranked, targets, args.topk)
    # Sanity check against saved per-user LightGCN metrics.
    saved_base = lightgcn_per_user["NDCG@20"].to_numpy(np.float64)
    if len(saved_base) == len(base_ndcg) and not np.allclose(saved_base, base_ndcg):
        raise RuntimeError("Yelp saved LightGCN per-user metrics differ from candidate reconstruction")
    out = {
        "LightGCN": {"NDCG@20": base_ndcg, "Recall@20": base_recall},
        "DeepSeek Direct Rerank": {"NDCG@20": direct_ndcg, "Recall@20": direct_recall},
        "Profile Rerank EGPR": {
            "NDCG@20": profile_per_user["NDCG@20"].to_numpy(np.float64),
            "Recall@20": profile_per_user["Recall@20"].to_numpy(np.float64),
        },
    }
    return out


def bootstrap_ci(delta: np.ndarray, rng: np.random.Generator, samples: int) -> Tuple[float, float]:
    n = len(delta)
    means = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        means[i] = float(delta[idx].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def sign_flip_pvalue(delta: np.ndarray, rng: np.random.Generator, samples: int) -> float:
    observed = abs(float(delta.mean()))
    if observed <= 0.0:
        return 1.0
    nonzero = delta[np.abs(delta) > 1e-12]
    if len(nonzero) == 0:
        return 1.0
    count = 0
    for _ in range(samples):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(nonzero))
        stat = abs(float((nonzero * signs).sum() / len(delta)))
        if stat >= observed - 1e-15:
            count += 1
    return float((count + 1) / (samples + 1))


def paired_rows(dataset: str, candidate_set: str, arrays: Mapping[str, Mapping[str, np.ndarray]], rng: np.random.Generator, args: argparse.Namespace) -> List[Dict[str, object]]:
    comparisons = [
        ("Direct vs Base", "DeepSeek Direct Rerank", "LightGCN"),
        ("Profile EGPR vs Base", "Profile Rerank EGPR", "LightGCN"),
        ("Profile EGPR vs Direct", "Profile Rerank EGPR", "DeepSeek Direct Rerank"),
    ]
    rows: List[Dict[str, object]] = []
    for label, method, baseline in comparisons:
        for metric in ["NDCG@20", "Recall@20"]:
            left = arrays[method][metric]
            right = arrays[baseline][metric]
            delta = left - right
            ci_low, ci_high = bootstrap_ci(delta, rng, args.bootstrap_samples)
            rows.append(
                {
                    "Dataset": dataset,
                    "CandidateSet": candidate_set,
                    "Comparison": label,
                    "Metric": metric,
                    "Users": int(len(delta)),
                    "BaselineMean": float(right.mean()),
                    "MethodMean": float(left.mean()),
                    "MeanDelta": float(delta.mean()),
                    "BootstrapCI95Low": ci_low,
                    "BootstrapCI95High": ci_high,
                    "SignFlipPValueTwoSided": sign_flip_pvalue(delta, rng, args.permutation_samples),
                    "Wins": int(np.sum(delta > 1e-12)),
                    "Ties": int(np.sum(np.abs(delta) <= 1e-12)),
                    "Losses": int(np.sum(delta < -1e-12)),
                }
            )
    return rows


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
    rng = np.random.default_rng(args.seed)

    rows: List[Dict[str, object]] = []
    print("loading ML-1M arrays", flush=True)
    ml_arrays = load_ml1m_arrays(args)
    rows.extend(paired_rows("ML-1M", "Direct top-50 / Profile top-100", ml_arrays, rng, args))

    print("loading Yelp top-50 arrays", flush=True)
    yelp50 = load_yelp_arrays(args, args.yelp_direct_top50, 50)
    rows.extend(paired_rows("Yelp", "matched top-50", yelp50, rng, args))

    print("loading Yelp top-100 arrays", flush=True)
    yelp100 = load_yelp_arrays(args, args.yelp_direct_top100, 100)
    rows.extend(paired_rows("Yelp", "matched top-100", yelp100, rng, args))

    result = pd.DataFrame(rows)
    result.to_csv(args.outdir / "paired_bootstrap_significance.csv", index=False)

    focus = result[(result["Metric"] == "NDCG@20") & (result["Comparison"].isin(["Profile EGPR vs Direct", "Direct vs Base", "Profile EGPR vs Base"]))].copy()
    focus = focus[
        [
            "Dataset", "CandidateSet", "Comparison", "Users", "BaselineMean", "MethodMean", "MeanDelta",
            "BootstrapCI95Low", "BootstrapCI95High", "SignFlipPValueTwoSided", "Wins", "Ties", "Losses",
        ]
    ]
    focus.to_csv(args.outdir / "paired_bootstrap_significance_ndcg_focus.csv", index=False)

    lines = [
        "# Paired Significance Tests",
        "",
        f"Bootstrap samples: {args.bootstrap_samples}. Sign-flip permutation samples: {args.permutation_samples}. Seed: {args.seed}.",
        "",
        "The sign-flip test is a paired non-parametric randomization test on per-user deltas. It is used here because SciPy is not installed in the current environment.",
        "",
        "## NDCG@20 Focus",
        "",
        markdown_table(focus),
        "",
        "## Interpretation",
        "",
        "- Positive MeanDelta means the method named first in the comparison improved over the baseline named second.",
        "- Confidence intervals crossing zero should be treated as pilot evidence rather than a formal significance claim.",
        "- These tests are paired on the exact same users and targets within each dataset/candidate setting.",
        "",
        "## Artifacts",
        "",
        "- `paired_bootstrap_significance.csv`",
        "- `paired_bootstrap_significance_ndcg_focus.csv`",
        "",
        f"Generated at {datetime.now(timezone.utc).isoformat()}.",
    ]
    (args.outdir / "paired_bootstrap_significance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done. Report: {args.outdir / 'paired_bootstrap_significance_report.md'}", flush=True)


if __name__ == "__main__":
    main()
