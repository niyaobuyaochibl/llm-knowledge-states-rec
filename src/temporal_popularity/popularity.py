"""Popularity counts, percentiles, and bucket assignment."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


SECONDS_PER_DAY = 24 * 60 * 60
STATIC_BUCKET_NAMES = ["Tail", "Mid", "Head"]
TEMPORAL_BUCKET_NAMES = ["Tail", "Mid", "Head", "Dormant"]


def static_popularity(train: pd.DataFrame, n_items: int, item_col: str = "iid") -> np.ndarray:
    """Count item frequency in train interactions."""
    return np.bincount(train[item_col].to_numpy(np.int64), minlength=n_items).astype(np.float32)


def assign_buckets(pop: np.ndarray, static_pop: np.ndarray, dormant_for_zero: bool) -> np.ndarray:
    """Assign Tail/Mid/Head buckets, optionally using Dormant for zero temporal popularity.

    Tie order is deterministic: popularity, StaticPop, item id.
    """
    n_items = len(pop)
    buckets = np.full(n_items, 1, dtype=np.int8)
    if dormant_for_zero:
        active = np.flatnonzero(pop > 0)
        buckets[pop <= 0] = 3
    else:
        active = np.arange(n_items, dtype=np.int64)
    if len(active) == 0:
        return buckets

    order_local = np.lexsort((active, static_pop[active], pop[active]))
    ordered = active[order_local]
    n_active = len(ordered)
    n_tail = max(1, int(math.ceil(0.2 * n_active)))
    n_head = max(1, int(math.ceil(0.2 * n_active)))
    buckets[ordered[:n_tail]] = 0
    buckets[ordered[n_tail : max(n_tail, n_active - n_head)]] = 1
    buckets[ordered[max(n_tail, n_active - n_head) :]] = 2
    return buckets


def popularity_percentiles(pop: np.ndarray, static_pop: np.ndarray) -> np.ndarray:
    """Return deterministic item popularity percentiles in [0, 1]."""
    items = np.arange(len(pop), dtype=np.int64)
    ordered = np.lexsort((items, static_pop, pop))
    pct = np.empty(len(pop), dtype=np.float32)
    if len(pop) == 1:
        pct[ordered] = 1.0
    else:
        pct[ordered] = np.linspace(0.0, 1.0, len(pop), dtype=np.float32)
    return pct


def zscore(values: np.ndarray) -> np.ndarray:
    """Stable z-score used by PopPenalty reranking."""
    std = values.std()
    if std < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - values.mean()) / std).astype(np.float32)
