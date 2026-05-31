"""Dataset split and grouping helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_interaction_split(datadir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read train/validation/test/all_events CSVs with standard columns."""
    usecols = ["uid", "iid", "timestamp"]
    dtypes = {"uid": np.int32, "iid": np.int32, "timestamp": np.int64}
    train = pd.read_csv(datadir / "train.csv", usecols=usecols, dtype=dtypes)
    val = pd.read_csv(datadir / "val.csv", usecols=usecols, dtype=dtypes)
    test = pd.read_csv(datadir / "test.csv", usecols=usecols, dtype=dtypes)
    all_events = pd.read_csv(datadir / "all_events_log_observable.csv", usecols=usecols, dtype=dtypes)
    return train, val, test, all_events


def infer_shape(*frames: pd.DataFrame) -> Tuple[int, int]:
    """Infer number of users/items from remapped split frames."""
    max_uid = max(int(frame["uid"].max()) for frame in frames if not frame.empty)
    max_iid = max(int(frame["iid"].max()) for frame in frames if not frame.empty)
    return max_uid + 1, max_iid + 1


def build_user_histories(frame: pd.DataFrame, n_users: int) -> List[np.ndarray]:
    """Build sorted unique item histories for each user."""
    histories: List[List[int]] = [[] for _ in range(n_users)]
    for uid, iid in frame[["uid", "iid"]].itertuples(index=False):
        histories[int(uid)].append(int(iid))
    return [np.array(sorted(set(items)), dtype=np.int64) for items in histories]


def build_exclude_lists(train_histories: Sequence[np.ndarray], val: pd.DataFrame | None, n_users: int) -> List[np.ndarray]:
    """Return per-user items excluded during test ranking."""
    output = [items.copy() for items in train_histories]
    if val is None:
        return output
    val_items: List[List[int]] = [[] for _ in range(n_users)]
    for uid, iid in val[["uid", "iid"]].itertuples(index=False):
        val_items[int(uid)].append(int(iid))
    for uid in range(n_users):
        if val_items[uid]:
            output[uid] = np.asarray(sorted(set(output[uid].tolist() + val_items[uid])), dtype=np.int64)
    return output


def activity_controlled_user_groups(train_histories: Sequence[np.ndarray], static_pct: np.ndarray) -> Dict[int, str]:
    """Group users into niche/mainstream/balanced within activity tertiles."""
    n_users = len(train_histories)
    lengths = np.array([len(items) for items in train_histories])
    length_order = np.argsort(lengths, kind="mergesort")
    activity = np.empty(n_users, dtype=np.int8)
    for bucket, idx in enumerate(np.array_split(length_order, 3)):
        activity[idx] = bucket

    user_pref = np.array(
        [float(np.median(static_pct[items])) if len(items) else 0.0 for items in train_histories],
        dtype=np.float32,
    )
    labels: Dict[int, str] = {}
    for bucket in range(3):
        members = np.flatnonzero(activity == bucket)
        ordered = members[np.argsort(user_pref[members], kind="mergesort")]
        n = len(ordered)
        n_edge = int(np.floor(0.3 * n))
        niche = set(ordered[:n_edge].tolist())
        mainstream = set(ordered[n - n_edge :].tolist())
        for uid in members:
            if int(uid) in niche:
                labels[int(uid)] = "niche"
            elif int(uid) in mainstream:
                labels[int(uid)] = "mainstream"
            else:
                labels[int(uid)] = "balanced"
    return labels
