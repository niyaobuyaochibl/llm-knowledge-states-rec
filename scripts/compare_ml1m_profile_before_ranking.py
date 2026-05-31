#!/usr/bin/env python3
"""Generic ML-1M direct-rerank vs profile-before-ranking comparison.

This script is intended for confirmatory runs where only the main expressive
profile setting is generated. It makes no API calls.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path("/root/temporal_popularity_pilot")
INPUT_PRICE_PER_1M = 0.14
OUTPUT_PRICE_PER_1M = 0.28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-run", type=Path, required=True)
    parser.add_argument("--profile-run", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--direct-label", default="DeepSeek Direct Rerank")
    parser.add_argument("--profile-label", default="Expressive")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ndcg_recall_from_ranked(ranked: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    rows = []
    for row, target_np in enumerate(targets):
        target = int(target_np)
        positions = np.flatnonzero(ranked[row] == target)
        hit = len(positions) > 0
        ndcg = 1.0 / math.log2(int(positions[0]) + 2) if hit else 0.0
        rows.append({"NDCG@20": ndcg, "Recall@20": float(hit), "HitRate@20": float(hit)})
    return pd.DataFrame(rows)


def summary(metrics: pd.DataFrame) -> Dict[str, float]:
    return {col: float(metrics[col].mean()) for col in ["NDCG@20", "Recall@20", "HitRate@20"]}


def reliability(base: pd.DataFrame, method: pd.DataFrame) -> Dict[str, float]:
    delta = method["NDCG@20"].to_numpy(np.float64) - base["NDCG@20"].to_numpy(np.float64)
    pos = delta[delta > 0.0]
    neg = delta[delta < 0.0]
    return {
        "HarmRate": float(np.mean(delta < 0.0)),
        "PositiveGainRate": float(np.mean(delta > 0.0)),
        "MeanDeltaNDCG@20": float(np.mean(delta)),
        "PositiveGainSum": float(pos.sum()) if len(pos) else 0.0,
        "NegativeGainSum": float(neg.sum()) if len(neg) else 0.0,
        "GainHarmRatio": float(pos.sum() / abs(neg.sum())) if len(neg) and abs(neg.sum()) > 1e-12 else np.inf,
    }


def token_cost(input_tokens: float, output_tokens: float) -> float:
    return input_tokens / 1_000_000 * INPUT_PRICE_PER_1M + output_tokens / 1_000_000 * OUTPUT_PRICE_PER_1M


def direct_cost(direct_run: Path) -> Dict[str, float]:
    cost = pd.read_csv(direct_run / "cost_trace_lightgcn_original.csv")
    input_tokens = float(cost["input_tokens"].sum())
    output_tokens = float(cost["output_tokens"].sum())
    return {
        "APICalls_TestUsers": float(len(cost)),
        "InputTokens_TestUsers": input_tokens,
        "OutputTokens_TestUsers": output_tokens,
        "TotalTokens_TestUsers": input_tokens + output_tokens,
        "EstimatedCostUSD_TestUsers": token_cost(input_tokens, output_tokens),
        "LatencySeconds_TestUsers": float(cost["latency_seconds"].sum()),
        "LatencyPerTestUser": float(cost["latency_seconds"].mean()),
    }


def profile_cost(profile_run: Path, test_users: np.ndarray) -> Dict[str, float]:
    cost = pd.read_csv(profile_run / "profile_cost_trace.csv")
    test_set = set(int(uid) for uid in test_users)
    test_cost = cost[cost["uid"].astype(int).isin(test_set)]
    all_input = float(cost["input_tokens"].sum())
    all_output = float(cost["output_tokens"].sum())
    test_input = float(test_cost["input_tokens"].sum())
    test_output = float(test_cost["output_tokens"].sum())
    return {
        "APICalls_TestUsers": float(len(test_cost)),
        "InputTokens_TestUsers": test_input,
        "OutputTokens_TestUsers": test_output,
        "TotalTokens_TestUsers": test_input + test_output,
        "EstimatedCostUSD_TestUsers": token_cost(test_input, test_output),
        "LatencySeconds_TestUsers": float(test_cost["latency_seconds"].sum()),
        "LatencyPerTestUser": float(test_cost["latency_seconds"].mean()) if len(test_cost) else 0.0,
        "APICalls_ValTestUnique": float(len(cost)),
        "InputTokens_ValTestUnique": all_input,
        "OutputTokens_ValTestUnique": all_output,
        "TotalTokens_ValTestUnique": all_input + all_output,
        "EstimatedCostUSD_ValTestUnique": token_cost(all_input, all_output),
        "LatencySeconds_ValTestUnique": float(cost["latency_seconds"].sum()),
    }


def first_row(df: pd.DataFrame, method: str) -> pd.Series:
    rows = df.loc[df["Method"] == method]
    if rows.empty:
        raise RuntimeError(f"Missing method row: {method}")
    return rows.iloc[0]


def optional_per_user(profile_run: Path, safe_name: str) -> Optional[pd.DataFrame]:
    path = profile_run / f"per_user_{safe_name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def win_tie_loss(a: np.ndarray, b: np.ndarray) -> Dict[str, int]:
    delta = a.astype(np.float64) - b.astype(np.float64)
    return {
        "Wins": int(np.sum(delta > 0.0)),
        "Ties": int(np.sum(delta == 0.0)),
        "Losses": int(np.sum(delta < 0.0)),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: "inf" if np.isinf(v) else ("" if pd.isna(v) else f"{v:.6f}"))
        else:
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else str(v))
    headers = [str(c) for c in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in display.values.tolist():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)

    direct_npz = np.load(args.direct_run / "candidates_lightgcn_test.npz")
    direct_users = direct_npz["users"].astype(np.int64)
    targets = direct_npz["targets"].astype(np.int64)
    direct_candidates = direct_npz["candidates"].astype(np.int64)
    base_ranked = direct_candidates[:, : args.topk]
    direct_ranked = np.load(args.direct_run / f"reranked_lightgcn_original_top{args.topk}.npy").astype(np.int64)

    profile_npz = np.load(args.profile_run / "candidates_lightgcn_test.npz")
    profile_users = profile_npz["users"].astype(np.int64)
    profile_targets = profile_npz["targets"].astype(np.int64)
    profile_candidates = profile_npz["candidates"].astype(np.int64)
    if not np.array_equal(direct_users, profile_users):
        raise RuntimeError("Direct and profile test users differ.")
    if not np.array_equal(targets, profile_targets):
        raise RuntimeError("Direct and profile targets differ.")
    prefix_width = direct_candidates.shape[1]
    if not np.array_equal(direct_candidates, profile_candidates[:, :prefix_width]):
        raise RuntimeError("Direct candidates do not match profile candidate prefix.")

    perf = pd.read_csv(args.profile_run / "table2_recommendation_performance.csv")
    rel = pd.read_csv(args.profile_run / "table3_reliability.csv")
    faith = pd.read_csv(args.profile_run / "table1_profile_faithfulness.csv")
    manifest = json.loads((args.profile_run / "run_manifest.json").read_text(encoding="utf-8"))

    base_metrics = ndcg_recall_from_ranked(base_ranked, targets)
    direct_metrics = ndcg_recall_from_ranked(direct_ranked, targets)
    direct_costs = direct_cost(args.direct_run)
    profile_costs = profile_cost(args.profile_run, direct_users)

    rows: List[Dict[str, object]] = []
    base_summary = summary(base_metrics)
    rows.append({
        "Method": "LightGCN",
        "Prompt": "-",
        "UCR": np.nan,
        "WeightedUCR": np.nan,
        "ProfileDriftScore": np.nan,
        **base_summary,
        **reliability(base_metrics, base_metrics),
        "EstimatedCostUSD_TestUsers": 0.0,
        "CostRatioVsDirect_TestUsers": 0.0,
        "EstimatedCostUSD_ValTestUnique": 0.0,
        "APICalls_TestUsers": 0.0,
    })
    rows.append({
        "Method": args.direct_label,
        "Prompt": "direct",
        "UCR": np.nan,
        "WeightedUCR": np.nan,
        "ProfileDriftScore": np.nan,
        **summary(direct_metrics),
        **reliability(base_metrics, direct_metrics),
        **direct_costs,
        "EstimatedCostUSD_ValTestUnique": np.nan,
    })

    for label, perf_method, faith_method, safe_name in [
        (f"Profile Rerank {args.profile_label} Raw", "LightGCN + Raw Profile", "Raw Profile", "lightgcn_raw_profile"),
        (f"Profile Rerank {args.profile_label} EGPR", "LightGCN + Evidence-Weighted Repair", "Evidence-Weighted Repair", "lightgcn_evidence_weighted_repair"),
    ]:
        perf_row = first_row(perf, perf_method)
        rel_row = first_row(rel, perf_method)
        faith_row = first_row(faith, faith_method)
        row = {
            "Method": label,
            "Prompt": str(manifest.get("prompt_variant", args.profile_label)),
            "ClaimsPerUser": int(manifest.get("claims_per_user", 0)),
            "UCR": float(faith_row["UCR"]),
            "WeightedUCR": float(faith_row["WeightedUCR"]),
            "ProfileDriftScore": float(faith_row["ProfileDriftScore"]),
            "NDCG@20": float(perf_row["NDCG@20"]),
            "Recall@20": float(perf_row["Recall@20"]),
            "HitRate@20": float(perf_row["HitRate@20"]),
            "HarmRate": float(rel_row["HarmRate"]),
            "PositiveGainRate": float(rel_row["PositiveGainRate"]),
            "MeanDeltaNDCG@20": float(rel_row["MeanDeltaNDCG@20"]),
            "PositiveGainSum": float(rel_row["PositiveGainSum"]),
            "NegativeGainSum": float(rel_row["NegativeGainSum"]),
            "GainHarmRatio": float(rel_row["GainHarmRatio"]),
            **profile_costs,
        }
        per_user = optional_per_user(args.profile_run, safe_name)
        if per_user is not None:
            wtld = win_tie_loss(per_user["NDCG@20"].to_numpy(), direct_metrics["NDCG@20"].to_numpy())
            row.update({f"VsDirect{key}": value for key, value in wtld.items()})
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison["NDCGGainVsBase"] = comparison["NDCG@20"] - float(base_summary["NDCG@20"])
    direct_cost_value = float(direct_costs["EstimatedCostUSD_TestUsers"])
    comparison["CostRatioVsDirect_TestUsers"] = comparison["EstimatedCostUSD_TestUsers"] / direct_cost_value
    comparison.to_csv(args.outdir / "ml1m_profile_before_ranking_comparison.csv", index=False)

    summary_cols = [
        "Method", "Prompt", "NDCG@20", "Recall@20", "NDCGGainVsBase", "HarmRate", "GainHarmRatio",
        "UCR", "WeightedUCR", "ProfileDriftScore", "EstimatedCostUSD_TestUsers",
        "CostRatioVsDirect_TestUsers", "EstimatedCostUSD_ValTestUnique", "APICalls_TestUsers",
    ]
    optional_cols = ["VsDirectWins", "VsDirectTies", "VsDirectLosses"]
    summary_cols += [c for c in optional_cols if c in comparison.columns]
    slim = comparison[summary_cols].copy()
    slim.to_csv(args.outdir / "ml1m_profile_before_ranking_summary.csv", index=False)

    decision = {
        "dataset": "ML-1M",
        "users": int(len(direct_users)),
        "same_test_users": True,
        "same_targets": True,
        "same_candidate_prefix": True,
        "direct_ndcg": float(comparison.loc[comparison["Method"] == args.direct_label, "NDCG@20"].iloc[0]),
        "profile_raw_ndcg": float(comparison.loc[comparison["Method"].str.endswith(" Raw"), "NDCG@20"].iloc[0]),
        "profile_egpr_ndcg": float(comparison.loc[comparison["Method"].str.endswith(" EGPR"), "NDCG@20"].iloc[0]),
        "direct_test_cost_usd": direct_cost_value,
        "profile_test_cost_usd": float(profile_costs["EstimatedCostUSD_TestUsers"]),
        "profile_full_unique_cost_usd": float(profile_costs["EstimatedCostUSD_ValTestUnique"]),
        "profile_egpr_beats_direct": bool(comparison.loc[comparison["Method"].str.endswith(" EGPR"), "NDCG@20"].iloc[0] > comparison.loc[comparison["Method"] == args.direct_label, "NDCG@20"].iloc[0]),
        "profile_cost_below_direct": bool(profile_costs["EstimatedCostUSD_TestUsers"] < direct_cost_value),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.outdir / "ml1m_profile_before_ranking_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# ML-1M Profile-Before-Ranking Confirmatory Comparison",
        "",
        "Same test users, targets, and LightGCN candidate prefix are used for direct reranking and profile reranking.",
        "",
        "## Summary",
        "",
        markdown_table(slim),
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(decision, indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "- This report is generated without API calls from completed direct/profile run artifacts.",
        "- Use paired bootstrap/sign-flip tests on per-user metrics before making significance claims.",
        "",
        "## Artifacts",
        "",
        "- `ml1m_profile_before_ranking_summary.csv`",
        "- `ml1m_profile_before_ranking_comparison.csv`",
        "- `ml1m_profile_before_ranking_decision.json`",
        "",
    ]
    (args.outdir / "ml1m_profile_before_ranking_decision.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.outdir / "ml1m_profile_before_ranking_decision.md")


if __name__ == "__main__":
    main()
