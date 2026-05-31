"""Snapshot helpers for scalable temporal popularity evaluation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def weighted_time_snapshots(test: pd.DataFrame, snapshot_count: int, time_col: str = "timestamp") -> Tuple[np.ndarray, np.ndarray]:
    """Split test timestamps into equal-frequency bins and return median timestamps plus weights."""
    times = np.sort(test[time_col].to_numpy(np.int64), kind="mergesort")
    n = len(times)
    n_snapshots = min(snapshot_count, n)
    edges = np.linspace(0, n, n_snapshots + 1, dtype=np.int64)
    snapshot_times: List[int] = []
    weights: List[int] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        segment = times[left:right]
        snapshot_times.append(int(np.median(segment)))
        weights.append(int(right - left))
    return np.asarray(snapshot_times, dtype=np.int64), np.asarray(weights, dtype=np.int64)


def user_snapshot_map(
    test: pd.DataFrame,
    n_users: int,
    snapshot_count: int,
    user_col: str = "uid",
    time_col: str = "timestamp",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return snapshot times, bin weights, and each user's snapshot id."""
    ordered = test[[user_col, time_col]].sort_values([time_col, user_col], kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    n_snapshots = min(snapshot_count, n)
    edges = np.linspace(0, n, n_snapshots + 1, dtype=np.int64)
    snapshot_times: List[int] = []
    weights: List[int] = []
    user_snapshot = np.zeros(n_users, dtype=np.int32)
    for snap_idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if right <= left:
            continue
        segment = ordered.iloc[left:right]
        snapshot_times.append(int(np.median(segment[time_col].to_numpy(np.int64))))
        weights.append(int(right - left))
        user_snapshot[segment[user_col].to_numpy(np.int64)] = snap_idx
    return np.asarray(snapshot_times, dtype=np.int64), np.asarray(weights, dtype=np.int64), user_snapshot
