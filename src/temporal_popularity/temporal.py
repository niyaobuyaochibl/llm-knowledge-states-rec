"""Temporal popularity state builders."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .popularity import assign_buckets, popularity_percentiles


def build_temporal_snapshot_features(
    all_events: pd.DataFrame,
    snapshot_times: np.ndarray,
    n_items: int,
    static_pop: np.ndarray,
    window_seconds: int,
    tau_seconds: int,
    progress_every: int = 25,
) -> Dict[str, np.ndarray]:
    """Build recent/decay popularity, buckets, and percentiles at snapshot times."""
    events = all_events.sort_values(["timestamp", "iid"], kind="mergesort").reset_index(drop=True)
    event_times = events["timestamp"].to_numpy(np.int64)
    event_items = events["iid"].to_numpy(np.int64)
    order = np.argsort(snapshot_times, kind="mergesort")
    sorted_times = snapshot_times[order]

    recent_counts = np.zeros(n_items, dtype=np.float32)
    decay_counts = np.zeros(n_items, dtype=np.float32)
    add_ptr = 0
    recent_left_ptr = 0
    decay_current_time: Optional[int] = None

    recent_pop_sorted = np.zeros((len(snapshot_times), n_items), dtype=np.float32)
    decay_pop_sorted = np.zeros((len(snapshot_times), n_items), dtype=np.float32)
    recent_bucket_sorted = np.zeros((len(snapshot_times), n_items), dtype=np.int8)
    decay_bucket_sorted = np.zeros((len(snapshot_times), n_items), dtype=np.int8)
    recent_pct_sorted = np.zeros((len(snapshot_times), n_items), dtype=np.float32)
    decay_pct_sorted = np.zeros((len(snapshot_times), n_items), dtype=np.float32)

    for row, t in enumerate(sorted_times):
        t_int = int(t)
        if decay_current_time is None:
            decay_current_time = t_int
        elif t_int > decay_current_time:
            decay_counts *= math.exp(-(t_int - decay_current_time) / tau_seconds)
            decay_current_time = t_int

        add_start = add_ptr
        while add_ptr < len(event_times) and event_times[add_ptr] < t_int:
            item = event_items[add_ptr]
            recent_counts[item] += 1.0
            add_ptr += 1
        if add_ptr > add_start:
            new_items = event_items[add_start:add_ptr]
            new_times = event_times[add_start:add_ptr]
            contrib = np.exp(-(t_int - new_times) / tau_seconds).astype(np.float32)
            np.add.at(decay_counts, new_items, contrib)

        window_start = t_int - window_seconds
        while recent_left_ptr < add_ptr and event_times[recent_left_ptr] < window_start:
            recent_counts[event_items[recent_left_ptr]] -= 1.0
            recent_left_ptr += 1

        recent_pop_sorted[row] = recent_counts
        decay_pop_sorted[row] = decay_counts
        recent_bucket_sorted[row] = assign_buckets(recent_counts, static_pop, dormant_for_zero=True)
        decay_bucket_sorted[row] = assign_buckets(decay_counts, static_pop, dormant_for_zero=True)
        recent_pct_sorted[row] = popularity_percentiles(recent_counts, static_pop)
        decay_pct_sorted[row] = popularity_percentiles(decay_counts, static_pop)

        if row == 0 or (row + 1) % progress_every == 0 or row + 1 == len(sorted_times):
            print(f"temporal snapshot features={row + 1}/{len(sorted_times)}", flush=True)

    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return {
        "recent_pop": recent_pop_sorted[inverse],
        "decay_pop": decay_pop_sorted[inverse],
        "recent_bucket": recent_bucket_sorted[inverse],
        "decay_bucket": decay_bucket_sorted[inverse],
        "recent_pct": recent_pct_sorted[inverse],
        "decay_pct": decay_pct_sorted[inverse],
    }
