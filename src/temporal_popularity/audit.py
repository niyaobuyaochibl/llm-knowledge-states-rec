"""Conclusion-stability audit metrics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


def quality_value(row: pd.Series, metric: str, definition: str) -> float:
    if metric == "LTR":
        return float(row[f"{definition}_LTR@20"])
    if metric == "ARP":
        return -float(row[f"{definition}_ARP@20"])
    if metric == "PCE":
        return -float(row[f"{definition}_PCE@20"])
    if metric == "NDCG":
        return float(row["NDCG@20"])
    raise ValueError(metric)


def temporal_overclaim_and_rfr(
    summary: pd.DataFrame,
    methods: Sequence[str],
    temporal_definition: str = "Decay",
    metrics: Iterable[str] = ("ARP", "LTR", "PCE"),
) -> pd.DataFrame:
    """Compute TOD for each method and RFR across all method pairs."""
    rows: List[Dict[str, object]] = []
    indexed = summary.set_index("Method")
    base = indexed.loc["Base"]
    metrics = list(metrics)
    for method in methods:
        if method == "Base":
            continue
        method_row = indexed.loc[method]
        for metric in metrics:
            static_gain = quality_value(method_row, metric, "Static") - quality_value(base, metric, "Static")
            temporal_gain = quality_value(method_row, metric, temporal_definition) - quality_value(
                base, metric, temporal_definition
            )
            rows.append(
                {
                    "Method": method,
                    "Metric": metric,
                    "TemporalDefinition": temporal_definition,
                    "StaticGain": static_gain,
                    "TemporalGain": temporal_gain,
                    "TOD": static_gain - temporal_gain,
                }
            )

    for metric in metrics:
        pair_flips = 0
        pairs = 0
        for i, left in enumerate(methods):
            for right in methods[i + 1 :]:
                static_diff = quality_value(indexed.loc[left], metric, "Static") - quality_value(
                    indexed.loc[right], metric, "Static"
                )
                temporal_diff = quality_value(indexed.loc[left], metric, temporal_definition) - quality_value(
                    indexed.loc[right], metric, temporal_definition
                )
                if np.sign(static_diff) != np.sign(temporal_diff):
                    pair_flips += 1
                pairs += 1
        rows.append(
            {
                "Method": "ALL_METHOD_PAIRS",
                "Metric": metric,
                "TemporalDefinition": temporal_definition,
                "StaticGain": np.nan,
                "TemporalGain": np.nan,
                "TOD": np.nan,
                "RFR": pair_flips / pairs if pairs else np.nan,
                "FlipPairs": pair_flips,
                "Pairs": pairs,
            }
        )
    return pd.DataFrame(rows)


def group_temporal_sensitivity(user_level: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    """Compute group-wise static-vs-temporal PCE/LTR sensitivity."""
    rows: List[Dict[str, object]] = []
    for method in methods:
        method_df = user_level[user_level["Method"] == method].copy()
        for group in ["niche", "mainstream", "balanced"]:
            group_df = method_df[method_df["Group"] == group]
            if group_df.empty:
                continue
            rows.append(
                {
                    "Method": method,
                    "Group": group,
                    "Users": int(group_df["uid"].nunique()),
                    "Static_PCE": float(group_df["Static_PCE@20"].mean()),
                    "Decay_PCE": float(group_df["Decay_PCE@20"].mean()),
                    "Temporal_PCE_Change": float((group_df["Decay_PCE@20"] - group_df["Static_PCE@20"]).mean()),
                    "PCE_Sensitivity": float((group_df["Decay_PCE@20"] - group_df["Static_PCE@20"]).abs().mean()),
                    "Static_LTR": float(group_df["Static_LTR@20"].mean()),
                    "Decay_LTR": float(group_df["Decay_LTR@20"].mean()),
                    "LTR_Shrinkage": float(group_df["Static_LTR@20"].mean() - group_df["Decay_LTR@20"].mean()),
                }
            )
    output = pd.DataFrame(rows)
    gap_rows: List[Dict[str, object]] = []
    for method in methods:
        sub = output[output["Method"] == method].set_index("Group")
        if {"niche", "mainstream"}.issubset(sub.index):
            gap_rows.append(
                {
                    "Method": method,
                    "Group": "GTSG_niche_minus_mainstream",
                    "PCE_Sensitivity": float(
                        sub.loc["niche", "PCE_Sensitivity"] - sub.loc["mainstream", "PCE_Sensitivity"]
                    ),
                    "LTR_Shrinkage": float(sub.loc["niche", "LTR_Shrinkage"] - sub.loc["mainstream", "LTR_Shrinkage"]),
                }
            )
    if gap_rows:
        output = pd.concat([output, pd.DataFrame(gap_rows)], ignore_index=True, sort=False)
    return output
