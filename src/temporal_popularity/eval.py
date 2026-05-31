"""Full-ranking evaluation and reranking helpers."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .popularity import zscore


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return top-k item indices for a score vector."""
    if len(scores) <= k:
        return np.argsort(-scores, kind="mergesort")
    partial = np.argpartition(-scores, kth=k - 1)[:k]
    return partial[np.argsort(-scores[partial], kind="mergesort")]


def median_or_zero(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else 0.0


def method_specs(lambdas: Sequence[float]) -> Dict[str, Tuple[str, Optional[float]]]:
    """Build Base/StaticPopPenalty/TemporalPopPenalty method specs."""
    specs: Dict[str, Tuple[str, Optional[float]]] = {"Base": ("base", None)}
    for lam in lambdas:
        specs[f"StaticPopPenalty@{lam:g}"] = ("static", float(lam))
    for lam in lambdas:
        specs[f"TemporalPopPenalty@{lam:g}"] = ("temporal", float(lam))
    return specs


def select_lambda(
    val_summary: pd.DataFrame,
    base_method: str,
    prefix: str,
    metric: str,
    ndcg_col: str = "NDCG@20",
) -> Tuple[float, pd.DataFrame]:
    """Select lambda under the 5% NDCG-drop rule."""
    base_ndcg = float(val_summary.loc[val_summary["Method"] == base_method, ndcg_col].iloc[0])
    candidates = val_summary[val_summary["Method"].str.startswith(prefix)].copy()
    candidates["lambda"] = candidates["Method"].str.split("@").str[-1].astype(float)
    candidates["ndcg_drop"] = base_ndcg - candidates[ndcg_col]
    eligible = candidates[candidates[ndcg_col] >= 0.95 * base_ndcg]
    if eligible.empty:
        eligible = candidates.sort_values(["ndcg_drop", "lambda"], ascending=[True, True]).head(1)
    selected = eligible.sort_values([metric, ndcg_col, "lambda"], ascending=[False, False, True]).iloc[0]
    return float(selected["lambda"]), candidates


def metric_row(
    method: str,
    uid: int,
    group: str,
    rec: np.ndarray,
    target: int,
    hist: np.ndarray,
    snap: int,
    static_pop: np.ndarray,
    static_bucket: np.ndarray,
    static_pct: np.ndarray,
    temporal: Mapping[str, np.ndarray],
    static_hist_median: np.ndarray,
) -> Dict[str, object]:
    """Compute accuracy and popularity metrics for one user recommendation list."""
    hit_positions = np.flatnonzero(rec == target)
    hit = len(hit_positions) > 0
    ndcg = 1.0 / math.log2(int(hit_positions[0]) + 2) if hit else 0.0
    recall = 1.0 if hit else 0.0
    recent_pct = temporal["recent_pct"][snap]
    decay_pct = temporal["decay_pct"][snap]
    return {
        "Method": method,
        "uid": int(uid),
        "Group": group,
        "NDCG@20": ndcg,
        "Recall@20": recall,
        "HitRate@20": recall,
        "Static_ARP@20": float(np.mean(static_pop[rec])),
        "Recent_ARP@20": float(np.mean(temporal["recent_pop"][snap, rec])),
        "Decay_ARP@20": float(np.mean(temporal["decay_pop"][snap, rec])),
        "Static_LTR@20": float(np.mean(static_bucket[rec] == 0)),
        "Recent_LTR@20": float(np.mean(temporal["recent_bucket"][snap, rec] == 0)),
        "Decay_LTR@20": float(np.mean(temporal["decay_bucket"][snap, rec] == 0)),
        "Static_HeadRatio@20": float(np.mean(static_bucket[rec] == 2)),
        "Recent_HeadRatio@20": float(np.mean(temporal["recent_bucket"][snap, rec] == 2)),
        "Decay_HeadRatio@20": float(np.mean(temporal["decay_bucket"][snap, rec] == 2)),
        "Static_PCE@20": float(abs(np.median(static_pct[rec]) - static_hist_median[uid])),
        "Recent_PCE@20": float(abs(np.median(recent_pct[rec]) - median_or_zero(recent_pct[hist]))),
        "Decay_PCE@20": float(abs(np.median(decay_pct[rec]) - median_or_zero(decay_pct[hist]))),
        "Static_SPS@20": float(np.median(static_pct[rec]) - static_hist_median[uid]),
        "Recent_SPS@20": float(np.median(recent_pct[rec]) - median_or_zero(recent_pct[hist])),
        "Decay_SPS@20": float(np.median(decay_pct[rec]) - median_or_zero(decay_pct[hist])),
    }


def summarize_user_metrics(user_rows: List[Dict[str, object]], method_names: Sequence[str]) -> pd.DataFrame:
    """Average user-level rows into method-level summary rows."""
    user_df = pd.DataFrame(user_rows)
    summary_rows: List[Dict[str, object]] = []
    for method in method_names:
        sub = user_df[user_df["Method"] == method]
        numeric_cols = [col for col in sub.columns if col not in {"Method", "uid", "Group"}]
        row = sub[numeric_cols].mean(numeric_only=True).to_dict()
        row["Method"] = method
        row["Users"] = int(sub["uid"].nunique())
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def rank_scores_for_method(
    kind: str,
    lam: Optional[float],
    candidate_scores: np.ndarray,
    candidate_items: np.ndarray,
    static_pop: np.ndarray,
    temporal_decay_pop: np.ndarray,
    static_pct: Optional[np.ndarray] = None,
    temporal_decay_pct: Optional[np.ndarray] = None,
    static_target: Optional[float] = None,
    temporal_target: Optional[float] = None,
) -> np.ndarray:
    """Apply Base, PopPenalty, or PopCal scoring."""
    if kind == "base":
        return candidate_scores
    score_z = zscore(candidate_scores)
    if kind == "static":
        return score_z - float(lam) * zscore(static_pop[candidate_items].astype(np.float32))
    if kind == "temporal":
        return score_z - float(lam) * zscore(temporal_decay_pop[candidate_items].astype(np.float32))
    if kind == "static_cal":
        if static_pct is None or static_target is None:
            raise ValueError("static_cal requires static_pct and static_target")
        cal_gap = np.abs(static_pct[candidate_items].astype(np.float32) - float(static_target))
        return score_z - float(lam) * cal_gap
    if kind == "temporal_cal":
        if temporal_decay_pct is None or temporal_target is None:
            raise ValueError("temporal_cal requires temporal_decay_pct and temporal_target")
        cal_gap = np.abs(temporal_decay_pct[candidate_items].astype(np.float32) - float(temporal_target))
        return score_z - float(lam) * cal_gap
    if kind == "xquad_tail":
        if static_pct is None:
            raise ValueError("xquad_tail requires static_pct")
        tail_bonus = 1.0 - static_pct[candidate_items].astype(np.float32)
        return score_z + float(lam) * zscore(tail_bonus)
    raise ValueError(kind)
