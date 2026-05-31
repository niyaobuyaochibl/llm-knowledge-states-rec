#!/usr/bin/env python3
"""Mini-pilot for temporal popularity definition sensitivity on MovieLens-1M.

This script intentionally stays scoped to the 3-day pilot:
1. Rebuild ML-1M with 10-core filtering and leave-one-out timestamp split.
2. Audit static-vs-temporal popularity bucket drift.
3. Train a seed-42 LightGCN base model.
4. Evaluate Base, Static PopPenalty, and Temporal PopPenalty by full ranking.

All ratings are treated as implicit positive interactions for this pilot.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


SECONDS_PER_DAY = 24 * 60 * 60
BUCKET_NAMES = ["Tail", "Mid", "Head", "Dormant"]
STATIC_BUCKET_NAMES = ["Tail", "Mid", "Head"]
LAMBDA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    all_events: pd.DataFrame
    n_users: int
    n_items: int
    user2id: Dict[int, int]
    item2id: Dict[int, int]
    retained_ratio: float
    raw_counts: Dict[str, int]


@dataclass
class TemporalContext:
    users: np.ndarray
    times: np.ndarray
    recent: np.ndarray
    decay: np.ndarray
    cumulative: np.ndarray


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.n_layers = n_layers
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def propagate(self, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        layers = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            layers.append(all_emb)
        out = torch.stack(layers, dim=0).mean(dim=0)
        return out[: self.n_users], out[self.n_users :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ratings",
        type=Path,
        default=Path("/root/autodl-tmp/projects_archive/MGPrompt-CDR/data/raw/ml-1m/ratings.dat"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("/root/temporal_popularity_pilot/results/ml1m"))
    parser.add_argument("--figdir", type=Path, default=Path("/root/temporal_popularity_pilot/figures/ml1m"))
    parser.add_argument("--datadir", type=Path, default=Path("/root/temporal_popularity_pilot/data/ml1m"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-core", type=int, default=10)
    parser.add_argument("--window-days", type=int, default=180)
    parser.add_argument("--tau-days", type=int, default=180)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--samples-per-epoch", type=int, default=200_000)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def load_ml1m(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_raw", "item_raw", "rating", "timestamp"],
        dtype={"user_raw": np.int64, "item_raw": np.int64, "rating": np.float32, "timestamp": np.int64},
    )
    return df.sort_values(["user_raw", "timestamp", "item_raw"]).reset_index(drop=True)


def iterative_k_core(df: pd.DataFrame, k: int) -> pd.DataFrame:
    current = df.copy()
    while True:
        before = len(current)
        user_counts = current["user_raw"].value_counts()
        item_counts = current["item_raw"].value_counts()
        current = current[
            current["user_raw"].isin(user_counts[user_counts >= k].index)
            & current["item_raw"].isin(item_counts[item_counts >= k].index)
        ].copy()
        if len(current) == before:
            return current.sort_values(["user_raw", "timestamp", "item_raw"]).reset_index(drop=True)


def split_leave_one_out(df: pd.DataFrame, k_core_ratio_base: int) -> SplitData:
    grouped = df.sort_values(["user_raw", "timestamp", "item_raw"]).groupby("user_raw", sort=True)
    pieces_train: List[pd.DataFrame] = []
    pieces_val: List[pd.DataFrame] = []
    pieces_test: List[pd.DataFrame] = []
    for _, group in grouped:
        if len(group) < 3:
            continue
        pieces_train.append(group.iloc[:-2])
        pieces_val.append(group.iloc[[-2]])
        pieces_test.append(group.iloc[[-1]])

    train = pd.concat(pieces_train, ignore_index=True)
    val = pd.concat(pieces_val, ignore_index=True)
    test = pd.concat(pieces_test, ignore_index=True)

    train_items = set(train["item_raw"].unique())
    valid_users = set(val[val["item_raw"].isin(train_items)]["user_raw"]).intersection(
        set(test[test["item_raw"].isin(train_items)]["user_raw"])
    )
    train = train[train["user_raw"].isin(valid_users)].copy()
    val = val[val["user_raw"].isin(valid_users)].copy()
    test = test[test["user_raw"].isin(valid_users)].copy()

    train_items = sorted(train["item_raw"].unique())
    users = sorted(train["user_raw"].unique())
    user2id = {u: idx for idx, u in enumerate(users)}
    item2id = {i: idx for idx, i in enumerate(train_items)}

    def map_frame(frame: pd.DataFrame) -> pd.DataFrame:
        mapped = frame[frame["item_raw"].isin(item2id)].copy()
        mapped["uid"] = mapped["user_raw"].map(user2id).astype(np.int64)
        mapped["iid"] = mapped["item_raw"].map(item2id).astype(np.int64)
        return mapped.sort_values(["uid", "timestamp", "iid"]).reset_index(drop=True)

    train = map_frame(train)
    val = map_frame(val)
    test = map_frame(test)
    retained_users = set(user2id.keys())
    all_events = df[df["user_raw"].isin(retained_users) & df["item_raw"].isin(item2id)].copy()
    all_events = map_frame(all_events)

    raw_counts = {
        "users_raw": int(df["user_raw"].nunique()),
        "items_raw": int(df["item_raw"].nunique()),
        "interactions_raw": int(k_core_ratio_base),
        "users_after_split": int(len(user2id)),
        "items_after_split": int(len(item2id)),
        "interactions_after_split": int(len(train) + len(val) + len(test)),
    }

    return SplitData(
        train=train,
        val=val,
        test=test,
        all_events=all_events,
        n_users=len(user2id),
        n_items=len(item2id),
        user2id=user2id,
        item2id=item2id,
        retained_ratio=(len(train) + len(val) + len(test)) / max(1, k_core_ratio_base),
        raw_counts=raw_counts,
    )


def save_split(split: SplitData, datadir: Path) -> None:
    split.train.to_csv(datadir / "train.csv", index=False)
    split.val.to_csv(datadir / "val.csv", index=False)
    split.test.to_csv(datadir / "test.csv", index=False)
    split.all_events.to_csv(datadir / "all_events_log_observable.csv", index=False)
    with (datadir / "mappings.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "user2id": {str(k): int(v) for k, v in split.user2id.items()},
                "item2id": {str(k): int(v) for k, v in split.item2id.items()},
            },
            f,
        )


def static_popularity(train: pd.DataFrame, n_items: int) -> np.ndarray:
    counts = np.bincount(train["iid"].to_numpy(np.int64), minlength=n_items).astype(np.float32)
    return counts


def assign_buckets(pop: np.ndarray, static_pop: np.ndarray, dormant_for_zero: bool) -> np.ndarray:
    n_items = len(pop)
    buckets = np.full(n_items, 1, dtype=np.int8)
    if dormant_for_zero:
        active = np.flatnonzero(pop > 0)
        buckets[pop <= 0] = 3
    else:
        active = np.arange(n_items, dtype=np.int64)
    if len(active) == 0:
        return buckets

    # Ascending order: low popularity is tail, high popularity is head.
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
    items = np.arange(len(pop), dtype=np.int64)
    ordered = np.lexsort((items, static_pop, pop))
    pct = np.empty(len(pop), dtype=np.float32)
    if len(pop) == 1:
        pct[ordered] = 1.0
    else:
        pct[ordered] = np.linspace(0.0, 1.0, len(pop), dtype=np.float32)
    return pct


def build_temporal_context(
    users: np.ndarray,
    times: np.ndarray,
    all_events: pd.DataFrame,
    n_items: int,
    window_seconds: int,
    tau_seconds: int,
) -> TemporalContext:
    events = all_events.sort_values(["timestamp", "iid"]).reset_index(drop=True)
    event_times = events["timestamp"].to_numpy(np.int64)
    event_items = events["iid"].to_numpy(np.int64)

    order = np.argsort(times, kind="mergesort")
    sorted_times = times[order]
    recent_sorted = np.zeros((len(times), n_items), dtype=np.float32)
    decay_sorted = np.zeros((len(times), n_items), dtype=np.float32)
    cumulative_sorted = np.zeros((len(times), n_items), dtype=np.float32)

    recent_counts = np.zeros(n_items, dtype=np.float32)
    cumulative_counts = np.zeros(n_items, dtype=np.float32)
    decay_counts = np.zeros(n_items, dtype=np.float32)
    add_ptr = 0
    recent_left_ptr = 0
    decay_current_time: Optional[int] = None

    for sorted_row, t in enumerate(sorted_times):
        t_int = int(t)
        if decay_current_time is None:
            decay_current_time = t_int
        elif t_int > decay_current_time:
            decay_counts *= math.exp(-(t_int - decay_current_time) / tau_seconds)
            decay_current_time = t_int

        add_start = add_ptr
        while add_ptr < len(event_times) and event_times[add_ptr] < t_int:
            item = event_items[add_ptr]
            cumulative_counts[item] += 1.0
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

        recent_sorted[sorted_row] = recent_counts
        decay_sorted[sorted_row] = decay_counts
        cumulative_sorted[sorted_row] = cumulative_counts

    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order))
    return TemporalContext(
        users=users,
        times=times,
        recent=recent_sorted[inverse_order],
        decay=decay_sorted[inverse_order],
        cumulative=cumulative_sorted[inverse_order],
    )


def compute_transition_stats(
    context: TemporalContext,
    static_buckets: np.ndarray,
    static_pop: np.ndarray,
    definition: str,
    pop_matrix: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    counts = np.zeros((3, 4), dtype=np.int64)
    bdr_num = 0
    total = 0
    tail_total = 0
    tail_exit = 0
    head_total = 0
    head_decay = 0
    zero_count = 0
    dormant_count = 0

    for row in range(pop_matrix.shape[0]):
        pop = pop_matrix[row]
        temp_buckets = assign_buckets(pop, static_pop, dormant_for_zero=True)
        zero_mask = pop <= 0
        zero_count += int(zero_mask.sum())
        dormant_count += int((temp_buckets == 3).sum())
        for sb in range(3):
            mask = static_buckets == sb
            row_counts = np.bincount(temp_buckets[mask], minlength=4)
            counts[sb, :] += row_counts[:4]
        bdr_num += int((temp_buckets != static_buckets).sum())
        total += len(static_buckets)
        tail_mask = static_buckets == 0
        head_mask = static_buckets == 2
        tail_total += int(tail_mask.sum())
        head_total += int(head_mask.sum())
        tail_exit += int((temp_buckets[tail_mask] != 0).sum())
        head_decay += int((temp_buckets[head_mask] != 2).sum())

    matrix = pd.DataFrame(counts, index=STATIC_BUCKET_NAMES, columns=BUCKET_NAMES)
    stats = {
        "Definition": definition,
        "BDR": bdr_num / total,
        "TER": tail_exit / tail_total,
        "HDR": head_decay / head_total,
        "ZRPR": zero_count / total,
        "DormantPct": dormant_count / total,
        "Pairs": int(total),
    }
    return matrix, stats


def build_norm_adj(train: pd.DataFrame, n_users: int, n_items: int, device: torch.device) -> torch.Tensor:
    users = train["uid"].to_numpy(np.int64)
    items = train["iid"].to_numpy(np.int64) + n_users
    src = np.concatenate([users, items])
    dst = np.concatenate([items, users])
    edge_index_np = np.vstack([src, dst])
    n_nodes = n_users + n_items
    deg = np.bincount(edge_index_np[0], minlength=n_nodes).astype(np.float32)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0.0
    weights = deg_inv_sqrt[edge_index_np[0]] * deg_inv_sqrt[edge_index_np[1]]
    edge_index = torch.from_numpy(edge_index_np).long().to(device)
    edge_weight = torch.from_numpy(weights).float().to(device)
    return torch.sparse_coo_tensor(edge_index, edge_weight, (n_nodes, n_nodes), device=device).coalesce()


def build_user_histories(frame: pd.DataFrame, n_users: int) -> List[np.ndarray]:
    histories: List[List[int]] = [[] for _ in range(n_users)]
    for uid, iid in frame[["uid", "iid"]].itertuples(index=False):
        histories[int(uid)].append(int(iid))
    return [np.array(sorted(set(items)), dtype=np.int64) for items in histories]


def sample_negatives(
    rng: np.random.Generator,
    users: np.ndarray,
    user_pos_sets: Sequence[set],
    n_items: int,
) -> np.ndarray:
    neg = rng.integers(0, n_items, size=len(users), dtype=np.int64)
    unresolved = np.array([item in user_pos_sets[int(user)] for user, item in zip(users, neg)], dtype=bool)
    while unresolved.any():
        idx = np.flatnonzero(unresolved)
        neg[idx] = rng.integers(0, n_items, size=len(idx), dtype=np.int64)
        unresolved[idx] = np.array(
            [item in user_pos_sets[int(user)] for user, item in zip(users[idx], neg[idx])], dtype=bool
        )
    return neg


def train_lightgcn(
    split: SplitData,
    norm_adj: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> LightGCN:
    model = LightGCN(split.n_users, split.n_items, args.embedding_dim, args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    train_pairs = split.train[["uid", "iid"]].to_numpy(np.int64)
    train_histories = build_user_histories(split.train, split.n_users)
    user_pos_sets = [set(items.tolist()) for items in train_histories]
    n_samples = min(args.samples_per_epoch, len(train_pairs))

    for epoch in range(1, args.epochs + 1):
        sample_idx = rng.integers(0, len(train_pairs), size=n_samples, dtype=np.int64)
        users_np = train_pairs[sample_idx, 0]
        pos_np = train_pairs[sample_idx, 1]
        neg_np = sample_negatives(rng, users_np, user_pos_sets, split.n_items)

        users = torch.from_numpy(users_np).long().to(device)
        pos = torch.from_numpy(pos_np).long().to(device)
        neg = torch.from_numpy(neg_np).long().to(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        user_emb, item_emb = model.propagate(norm_adj)
        u = user_emb[users]
        p = item_emb[pos]
        n = item_emb[neg]
        pos_scores = (u * p).sum(dim=1)
        neg_scores = (u * n).sum(dim=1)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        reg_loss = (u.pow(2).sum(dim=1) + p.pow(2).sum(dim=1) + n.pow(2).sum(dim=1)).mean()
        loss = bpr_loss + args.reg * reg_loss
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} loss={loss.item():.5f} "
                f"bpr={bpr_loss.item():.5f} reg={reg_loss.item():.5f}",
                flush=True,
            )
    return model


def make_exclude_lists(histories: Sequence[np.ndarray], extra: Optional[pd.DataFrame], n_users: int) -> List[np.ndarray]:
    output = [items.copy() for items in histories]
    if extra is not None:
        extra_items = [[] for _ in range(n_users)]
        for uid, iid in extra[["uid", "iid"]].itertuples(index=False):
            extra_items[int(uid)].append(int(iid))
        for uid in range(n_users):
            if extra_items[uid]:
                output[uid] = np.array(sorted(set(output[uid].tolist() + extra_items[uid])), dtype=np.int64)
    return output


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if len(scores) <= k:
        order = np.argsort(-scores, kind="mergesort")
        return order
    partial = np.argpartition(-scores, kth=k - 1)[:k]
    return partial[np.argsort(-scores[partial], kind="mergesort")]


def zscore(values: np.ndarray) -> np.ndarray:
    std = values.std()
    if std < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - values.mean()) / std).astype(np.float32)


def generate_recommendations(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    eval_users: np.ndarray,
    exclude_lists: Sequence[np.ndarray],
    static_pop: np.ndarray,
    context: TemporalContext,
    lambdas: Sequence[float],
    topk: int,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    method_names = ["Base"]
    method_names.extend([f"StaticPopPenalty@{lam:g}" for lam in lambdas])
    method_names.extend([f"TemporalPopPenalty@{lam:g}" for lam in lambdas])
    recs = {name: np.zeros((len(eval_users), topk), dtype=np.int64) for name in method_names}

    all_items = np.arange(item_emb.shape[0], dtype=np.int64)
    item_emb_t = item_emb.T.astype(np.float32)
    for start in range(0, len(eval_users), batch_size):
        end = min(start + batch_size, len(eval_users))
        batch_users = eval_users[start:end]
        score_matrix = user_emb[batch_users].astype(np.float32) @ item_emb_t
        for local_row, uid in enumerate(batch_users):
            global_row = start + local_row
            scores = score_matrix[local_row]
            candidate = np.ones(item_emb.shape[0], dtype=bool)
            candidate[exclude_lists[int(uid)]] = False
            candidate_items = all_items[candidate]
            candidate_scores = scores[candidate_items]
            base_order = topk_indices(candidate_scores, topk)
            recs["Base"][global_row] = candidate_items[base_order]

            score_z = zscore(candidate_scores)
            static_z = zscore(static_pop[candidate_items].astype(np.float32))
            decay_z = zscore(context.decay[global_row, candidate_items].astype(np.float32))
            for lam in lambdas:
                static_scores = score_z - lam * static_z
                static_order = topk_indices(static_scores, topk)
                recs[f"StaticPopPenalty@{lam:g}"][global_row] = candidate_items[static_order]

                temporal_scores = score_z - lam * decay_z
                temporal_order = topk_indices(temporal_scores, topk)
                recs[f"TemporalPopPenalty@{lam:g}"][global_row] = candidate_items[temporal_order]
    return recs


def user_groups(train_histories: Sequence[np.ndarray], static_pct: np.ndarray) -> Dict[int, str]:
    n_users = len(train_histories)
    lengths = np.array([len(items) for items in train_histories])
    length_order = np.argsort(lengths, kind="mergesort")
    activity = np.empty(n_users, dtype=np.int8)
    for bucket, idx in enumerate(np.array_split(length_order, 3)):
        activity[idx] = bucket

    labels: Dict[int, str] = {}
    user_pref = np.array(
        [float(np.median(static_pct[items])) if len(items) else 0.0 for items in train_histories], dtype=np.float32
    )
    for bucket in range(3):
        members = np.flatnonzero(activity == bucket)
        ordered = members[np.argsort(user_pref[members], kind="mergesort")]
        n = len(ordered)
        n_edge = int(math.floor(0.3 * n))
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


def evaluate_recommendations(
    recs: Mapping[str, np.ndarray],
    eval_users: np.ndarray,
    targets: np.ndarray,
    train_histories: Sequence[np.ndarray],
    static_pop: np.ndarray,
    static_buckets: np.ndarray,
    static_pct: np.ndarray,
    context: TemporalContext,
    groups: Mapping[int, str],
    topk: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, object]] = []
    user_rows: List[Dict[str, object]] = []

    n_eval = len(eval_users)
    n_items = len(static_pop)
    recent_buckets_mat = np.empty((n_eval, n_items), dtype=np.int8)
    decay_buckets_mat = np.empty((n_eval, n_items), dtype=np.int8)
    recent_pct_mat = np.empty((n_eval, n_items), dtype=np.float32)
    decay_pct_mat = np.empty((n_eval, n_items), dtype=np.float32)
    static_hist_median = np.empty(n_eval, dtype=np.float32)
    recent_hist_median = np.empty(n_eval, dtype=np.float32)
    decay_hist_median = np.empty(n_eval, dtype=np.float32)

    for row, uid in enumerate(eval_users):
        hist = train_histories[int(uid)]
        recent_pop = context.recent[row]
        decay_pop = context.decay[row]
        recent_buckets = assign_buckets(recent_pop, static_pop, dormant_for_zero=True)
        decay_buckets = assign_buckets(decay_pop, static_pop, dormant_for_zero=True)
        recent_pct = popularity_percentiles(recent_pop, static_pop)
        decay_pct = popularity_percentiles(decay_pop, static_pop)
        recent_buckets_mat[row] = recent_buckets
        decay_buckets_mat[row] = decay_buckets
        recent_pct_mat[row] = recent_pct
        decay_pct_mat[row] = decay_pct
        static_hist_median[row] = float(np.median(static_pct[hist]))
        recent_hist_median[row] = float(np.median(recent_pct[hist]))
        decay_hist_median[row] = float(np.median(decay_pct[hist]))

    for method, rec_matrix in recs.items():
        per_user: List[Dict[str, object]] = []
        for row, uid in enumerate(eval_users):
            rec = rec_matrix[row]
            target = int(targets[row])
            hit_positions = np.flatnonzero(rec == target)
            hit = len(hit_positions) > 0
            ndcg = 1.0 / math.log2(int(hit_positions[0]) + 2) if hit else 0.0
            recall = 1.0 if hit else 0.0

            static_rec_pct = static_pct[rec]
            recent_pop = context.recent[row]
            decay_pop = context.decay[row]
            recent_buckets = recent_buckets_mat[row]
            decay_buckets = decay_buckets_mat[row]
            recent_pct = recent_pct_mat[row]
            decay_pct = decay_pct_mat[row]

            row_metrics = {
                "Method": method,
                "uid": int(uid),
                "Group": groups[int(uid)],
                "NDCG@20": ndcg,
                "Recall@20": recall,
                "HitRate@20": recall,
                "Static_ARP@20": float(np.mean(static_pop[rec])),
                "Recent_ARP@20": float(np.mean(recent_pop[rec])),
                "Decay_ARP@20": float(np.mean(decay_pop[rec])),
                "Static_LTR@20": float(np.mean(static_buckets[rec] == 0)),
                "Recent_LTR@20": float(np.mean(recent_buckets[rec] == 0)),
                "Decay_LTR@20": float(np.mean(decay_buckets[rec] == 0)),
                "Static_HeadRatio@20": float(np.mean(static_buckets[rec] == 2)),
                "Recent_HeadRatio@20": float(np.mean(recent_buckets[rec] == 2)),
                "Decay_HeadRatio@20": float(np.mean(decay_buckets[rec] == 2)),
                "Static_PCE@20": float(abs(np.median(static_rec_pct) - static_hist_median[row])),
                "Recent_PCE@20": float(abs(np.median(recent_pct[rec]) - recent_hist_median[row])),
                "Decay_PCE@20": float(abs(np.median(decay_pct[rec]) - decay_hist_median[row])),
                "Static_SPS@20": float(np.median(static_rec_pct) - static_hist_median[row]),
                "Recent_SPS@20": float(np.median(recent_pct[rec]) - recent_hist_median[row]),
                "Decay_SPS@20": float(np.median(decay_pct[rec]) - decay_hist_median[row]),
                "StaticTail_TemporalTail_Recent_Validity": float(
                    np.mean(recent_buckets[rec[static_buckets[rec] == 0]] == 0)
                )
                if np.any(static_buckets[rec] == 0)
                else np.nan,
                "StaticTail_TemporalTail_Decay_Validity": float(
                    np.mean(decay_buckets[rec[static_buckets[rec] == 0]] == 0)
                )
                if np.any(static_buckets[rec] == 0)
                else np.nan,
            }
            per_user.append(row_metrics)

        user_df = pd.DataFrame(per_user)
        user_rows.extend(per_user)
        numeric_cols = [col for col in user_df.columns if col not in {"Method", "uid", "Group"}]
        means = user_df[numeric_cols].mean(numeric_only=True).to_dict()
        means["Method"] = method
        means["Users"] = len(user_df)
        summary_rows.append(means)

    summary = pd.DataFrame(summary_rows)
    user_level = pd.DataFrame(user_rows)
    return summary, user_level


def select_lambda(
    val_summary: pd.DataFrame,
    base_method: str,
    prefix: str,
    metric: str,
    ndcg_col: str = "NDCG@20",
) -> Tuple[float, pd.DataFrame]:
    base_ndcg = float(val_summary.loc[val_summary["Method"] == base_method, ndcg_col].iloc[0])
    candidates = val_summary[val_summary["Method"].str.startswith(prefix)].copy()
    candidates["lambda"] = candidates["Method"].str.split("@").str[-1].astype(float)
    candidates["ndcg_drop"] = base_ndcg - candidates[ndcg_col]
    eligible = candidates[candidates[ndcg_col] >= 0.95 * base_ndcg]
    if eligible.empty:
        eligible = candidates.sort_values(["ndcg_drop", "lambda"], ascending=[True, True]).head(1)
    selected = eligible.sort_values([metric, ndcg_col, "lambda"], ascending=[False, False, True]).iloc[0]
    return float(selected["lambda"]), candidates


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


def compute_tod_rfr(summary: pd.DataFrame, methods: Sequence[str], temporal_definition: str = "Decay") -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    indexed = summary.set_index("Method")
    base = indexed.loc["Base"]
    metrics = ["ARP", "LTR", "PCE"]
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


def group_sensitivity(user_level: pd.DataFrame, selected_methods: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for method in selected_methods:
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
    group_df = pd.DataFrame(rows)
    gap_rows = []
    for method in selected_methods:
        sub = group_df[group_df["Method"] == method].set_index("Group")
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
        group_df = pd.concat([group_df, pd.DataFrame(gap_rows)], ignore_index=True, sort=False)
    return group_df


def plot_transition(matrix: pd.DataFrame, title: str, path: Path) -> None:
    row_norm = matrix.div(matrix.sum(axis=1), axis=0).fillna(0.0)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.imshow(row_norm.to_numpy(), cmap="Blues", vmin=0, vmax=max(0.01, row_norm.to_numpy().max()))
    ax.set_xticks(np.arange(len(row_norm.columns)), row_norm.columns)
    ax.set_yticks(np.arange(len(row_norm.index)), row_norm.index)
    ax.set_xlabel("Temporal bucket")
    ax.set_ylabel("Static bucket")
    ax.set_title(title)
    for i in range(row_norm.shape[0]):
        for j in range(row_norm.shape[1]):
            ax.text(j, i, f"{row_norm.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_gain_scatter(tod: pd.DataFrame, path: Path) -> None:
    plot_df = tod.dropna(subset=["StaticGain", "TemporalGain"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for metric, sub in plot_df.groupby("Metric"):
        ax.scatter(sub["StaticGain"], sub["TemporalGain"], label=metric)
        for _, row in sub.iterrows():
            label = row["Method"].replace("PopPenalty", "PP")
            ax.annotate(label, (row["StaticGain"], row["TemporalGain"]), fontsize=7, alpha=0.8)
    lo = min(plot_df["StaticGain"].min(), plot_df["TemporalGain"].min(), 0.0)
    hi = max(plot_df["StaticGain"].max(), plot_df["TemporalGain"].max(), 0.0)
    ax.plot([lo, hi], [lo, hi], color="gray", linewidth=1)
    ax.set_xlabel("Static gain")
    ax.set_ylabel("Temporal gain")
    ax.set_title("Static vs temporal debiasing gain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_group_sensitivity(group_df: pd.DataFrame, path: Path) -> None:
    plot_df = group_df[group_df["Group"].isin(["niche", "mainstream"])].copy()
    methods = plot_df["Method"].unique().tolist()
    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for offset, group in [(-width / 2, "niche"), (width / 2, "mainstream")]:
        values = [
            float(plot_df[(plot_df["Method"] == method) & (plot_df["Group"] == group)]["PCE_Sensitivity"].iloc[0])
            if not plot_df[(plot_df["Method"] == method) & (plot_df["Group"] == group)].empty
            else 0.0
            for method in methods
        ]
        ax.bar(x + offset, values, width, label=group)
    ax.set_xticks(x, [m.replace("PopPenalty", "PP") for m in methods], rotation=20, ha="right")
    ax.set_ylabel("|Decay PCE - Static PCE|")
    ax.set_title("Group temporal sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, index: bool = False, float_digits: int = 6) -> str:
    table = df.copy()
    if index:
        table = table.reset_index()
    formatted = table.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(
                lambda x: "" if pd.isna(x) else f"{float(x):.{float_digits}g}"
            )
        else:
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(col) for col in formatted.columns]
    rows = formatted.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_markdown_report(
    outdir: Path,
    split: SplitData,
    drift_stats: pd.DataFrame,
    selected_static_lambda: float,
    selected_temporal_lambda: float,
    test_summary: pd.DataFrame,
    tod: pd.DataFrame,
    group_df: pd.DataFrame,
    criteria: Dict[str, object],
) -> None:
    selected = test_summary[
        test_summary["Method"].isin(
            ["Base", f"StaticPopPenalty@{selected_static_lambda:g}", f"TemporalPopPenalty@{selected_temporal_lambda:g}"]
        )
    ].copy()

    lines = [
        "# ML-1M Temporal Popularity Mini-Pilot",
        "",
        "All ratings are treated as implicit positive interactions. Temporal popularity uses log-observable global interactions before each user's evaluation timestamp.",
        "",
        "## Dataset",
        "",
        f"- Users: {split.n_users}",
        f"- Items: {split.n_items}",
        f"- Train / Val / Test interactions: {len(split.train)} / {len(split.val)} / {len(split.test)}",
        f"- Time span: {pd.to_datetime(split.all_events['timestamp'].min(), unit='s')} to {pd.to_datetime(split.all_events['timestamp'].max(), unit='s')}",
        f"- 10-core retained ratio after split cleanup: {split.retained_ratio:.4f}",
        "",
        "## Day 1 Drift",
        "",
        markdown_table(drift_stats, index=False),
        "",
        "## Selected Lambdas",
        "",
        f"- Static PopPenalty: lambda={selected_static_lambda:g}",
        f"- Temporal PopPenalty: lambda={selected_temporal_lambda:g}",
        "",
        "## Day 2-3 Main Test Metrics",
        "",
        selected[
            [
                "Method",
                "NDCG@20",
                "Recall@20",
                "Static_ARP@20",
                "Recent_ARP@20",
                "Decay_ARP@20",
                "Static_LTR@20",
                "Recent_LTR@20",
                "Decay_LTR@20",
                "Static_PCE@20",
                "Recent_PCE@20",
                "Decay_PCE@20",
            ]
        ].pipe(markdown_table, index=False),
        "",
        "## Temporal Overclaim / Ranking Flip",
        "",
        markdown_table(tod, index=False),
        "",
        "## Group Sensitivity",
        "",
        markdown_table(group_df, index=False),
        "",
        "## Go / No-Go Criteria",
        "",
    ]
    for key, value in criteria.items():
        lines.append(f"- {key}: {value}")
    (outdir / "pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs(args.outdir, args.figdir, args.datadir)

    print("Loading and splitting ML-1M...", flush=True)
    raw = load_ml1m(args.ratings)
    raw_interactions = len(raw)
    filtered = iterative_k_core(raw, args.k_core)
    split = split_leave_one_out(filtered, raw_interactions)
    save_split(split, args.datadir)

    static_pop = static_popularity(split.train, split.n_items)
    static_buckets = assign_buckets(static_pop, static_pop, dormant_for_zero=False)
    static_pct = popularity_percentiles(static_pop, static_pop)
    train_histories = build_user_histories(split.train, split.n_users)
    groups = user_groups(train_histories, static_pct)

    dataset_availability = pd.DataFrame(
        [
            {
                "Dataset": "MovieLens-1M",
                "Users": split.n_users,
                "Items": split.n_items,
                "Interactions": len(split.train) + len(split.val) + len(split.test),
                "TimeSpan": f"{pd.to_datetime(split.all_events['timestamp'].min(), unit='s').date()} to {pd.to_datetime(split.all_events['timestamp'].max(), unit='s').date()}",
                "TimestampSource": "ratings.dat timestamp",
                "10CoreRetainedRatio": split.retained_ratio,
                "PilotStatus": "run",
            },
            {
                "Dataset": "Yelp original reviews",
                "Users": np.nan,
                "Items": np.nan,
                "Interactions": np.nan,
                "TimeSpan": "not processed in ML-1M mini-pilot",
                "TimestampSource": "review JSON date field verified",
                "10CoreRetainedRatio": np.nan,
                "PilotStatus": "timestamp available; large raw file",
            },
            {
                "Dataset": "Amazon Books 5-core raw",
                "Users": np.nan,
                "Items": np.nan,
                "Interactions": np.nan,
                "TimeSpan": "not processed in ML-1M mini-pilot",
                "TimestampSource": "unixReviewTime/reviewTime verified",
                "10CoreRetainedRatio": np.nan,
                "PilotStatus": "timestamp available; large gzip file",
            },
        ]
    )
    dataset_availability.to_csv(args.outdir / "table1_dataset_timestamp_availability.csv", index=False)

    print("Computing temporal popularity contexts...", flush=True)
    window_seconds = args.window_days * SECONDS_PER_DAY
    tau_seconds = args.tau_days * SECONDS_PER_DAY
    val_users = split.val.sort_values("uid")["uid"].to_numpy(np.int64)
    val_times = split.val.sort_values("uid")["timestamp"].to_numpy(np.int64)
    val_targets = split.val.sort_values("uid")["iid"].to_numpy(np.int64)
    test_users = split.test.sort_values("uid")["uid"].to_numpy(np.int64)
    test_times = split.test.sort_values("uid")["timestamp"].to_numpy(np.int64)
    test_targets = split.test.sort_values("uid")["iid"].to_numpy(np.int64)

    val_context = build_temporal_context(val_users, val_times, split.all_events, split.n_items, window_seconds, tau_seconds)
    test_context = build_temporal_context(
        test_users, test_times, split.all_events, split.n_items, window_seconds, tau_seconds
    )

    recent_matrix, recent_stats = compute_transition_stats(
        test_context, static_buckets, static_pop, f"RecentPop@{args.window_days}d", test_context.recent
    )
    decay_matrix, decay_stats = compute_transition_stats(
        test_context, static_buckets, static_pop, f"DecayPop@{args.tau_days}d", test_context.decay
    )
    recent_matrix.to_csv(args.outdir / "static_to_recent_transition_counts.csv")
    decay_matrix.to_csv(args.outdir / "static_to_decay_transition_counts.csv")
    drift_stats = pd.DataFrame([recent_stats, decay_stats])
    drift_stats.to_csv(args.outdir / "table2_temporal_drift_stats.csv", index=False)
    plot_transition(
        recent_matrix,
        f"Static to RecentPop@{args.window_days}d bucket transition",
        args.figdir / "figure1_static_to_recent_transition.png",
    )
    plot_transition(
        decay_matrix,
        f"Static to DecayPop@{args.tau_days}d bucket transition",
        args.figdir / "figure1_static_to_decay_transition.png",
    )

    print("Training LightGCN base model...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm_adj = build_norm_adj(split.train, split.n_users, split.n_items, device)
    model = train_lightgcn(split, norm_adj, args, device)
    model.eval()
    with torch.no_grad():
        user_emb_t, item_emb_t = model.propagate(norm_adj)
    user_emb = user_emb_t.detach().cpu().numpy().astype(np.float32)
    item_emb = item_emb_t.detach().cpu().numpy().astype(np.float32)
    torch.save(model.state_dict(), args.outdir / "lightgcn_seed42.pt")

    print("Generating validation recommendations for lambda selection...", flush=True)
    val_exclude = make_exclude_lists(train_histories, None, split.n_users)
    val_recs = generate_recommendations(
        user_emb,
        item_emb,
        val_users,
        val_exclude,
        static_pop,
        val_context,
        LAMBDA_GRID,
        args.topk,
        args.eval_batch_size,
    )
    val_summary, val_user_level = evaluate_recommendations(
        val_recs,
        val_users,
        val_targets,
        train_histories,
        static_pop,
        static_buckets,
        static_pct,
        val_context,
        groups,
        args.topk,
    )
    val_summary.to_csv(args.outdir / "validation_all_lambda_metrics.csv", index=False)
    val_user_level.to_csv(args.outdir / "validation_user_level_metrics.csv", index=False)
    selected_static_lambda, static_lambda_table = select_lambda(
        val_summary, "Base", "StaticPopPenalty@", "Static_LTR@20"
    )
    selected_temporal_lambda, temporal_lambda_table = select_lambda(
        val_summary, "Base", "TemporalPopPenalty@", "Decay_LTR@20"
    )
    static_lambda_table.to_csv(args.outdir / "validation_static_lambda_table.csv", index=False)
    temporal_lambda_table.to_csv(args.outdir / "validation_temporal_lambda_table.csv", index=False)

    print("Generating test recommendations...", flush=True)
    test_exclude = make_exclude_lists(train_histories, split.val, split.n_users)
    test_recs_all = generate_recommendations(
        user_emb,
        item_emb,
        test_users,
        test_exclude,
        static_pop,
        test_context,
        LAMBDA_GRID,
        args.topk,
        args.eval_batch_size,
    )
    selected_method_names = [
        "Base",
        f"StaticPopPenalty@{selected_static_lambda:g}",
        f"TemporalPopPenalty@{selected_temporal_lambda:g}",
    ]
    test_recs = {name: test_recs_all[name] for name in selected_method_names}
    test_summary, test_user_level = evaluate_recommendations(
        test_recs,
        test_users,
        test_targets,
        train_histories,
        static_pop,
        static_buckets,
        static_pct,
        test_context,
        groups,
        args.topk,
    )
    test_summary.to_csv(args.outdir / "table3_static_vs_temporal_evaluation.csv", index=False)
    test_user_level.to_csv(args.outdir / "test_user_level_metrics.csv", index=False)
    for method, rec_matrix in test_recs.items():
        np.save(args.outdir / f"recs_{method.replace('@', '_').replace('.', 'p')}.npy", rec_matrix)

    tod = compute_tod_rfr(test_summary, selected_method_names, temporal_definition="Decay")
    tod.to_csv(args.outdir / "table4_temporal_overclaim_ranking_flip.csv", index=False)
    group_df = group_sensitivity(test_user_level, selected_method_names)
    group_df.to_csv(args.outdir / "table5_group_temporal_sensitivity.csv", index=False)
    plot_gain_scatter(tod, args.figdir / "figure2_static_vs_temporal_gain_scatter.png")
    plot_group_sensitivity(group_df, args.figdir / "figure3_group_temporal_sensitivity.png")

    base_row = test_summary.set_index("Method").loc["Base"]
    static_row = test_summary.set_index("Method").loc[f"StaticPopPenalty@{selected_static_lambda:g}"]
    temporal_row = test_summary.set_index("Method").loc[f"TemporalPopPenalty@{selected_temporal_lambda:g}"]
    static_ltr_gain = static_row["Static_LTR@20"] - base_row["Static_LTR@20"]
    temporal_ltr_gain = static_row["Decay_LTR@20"] - base_row["Decay_LTR@20"]
    shrinkage = 1.0 - (temporal_ltr_gain / static_ltr_gain) if abs(static_ltr_gain) > 1e-12 else np.nan
    rfr_max = tod["RFR"].max(skipna=True)
    group_gap_rows = group_df[group_df["Group"] == "GTSG_niche_minus_mainstream"]
    gtsg_pce = (
        float(group_gap_rows[group_gap_rows["Method"] == "Base"]["PCE_Sensitivity"].iloc[0])
        if not group_gap_rows[group_gap_rows["Method"] == "Base"].empty
        else np.nan
    )
    base_group = group_df[(group_df["Method"] == "Base") & (group_df["Group"].isin(["niche", "mainstream"]))]
    sensitivity_ratio = np.nan
    if set(base_group["Group"]) == {"niche", "mainstream"}:
        niche_s = float(base_group[base_group["Group"] == "niche"]["PCE_Sensitivity"].iloc[0])
        main_s = float(base_group[base_group["Group"] == "mainstream"]["PCE_Sensitivity"].iloc[0])
        sensitivity_ratio = niche_s / main_s if main_s > 0 else np.inf

    criteria = {
        "BDR>=20%": bool((drift_stats["BDR"] >= 0.20).any()),
        "TailExitRate>=30%": bool((drift_stats["TER"] >= 0.30).any()),
        "StaticPopPenaltyStaticLTRGainShrinkage>=30%": bool(shrinkage >= 0.30) if not np.isnan(shrinkage) else False,
        "AnyRankingFlip": bool(rfr_max > 0) if not np.isnan(rfr_max) else False,
        "NicheSensitivity>=1.25xMainstream": bool(sensitivity_ratio >= 1.25)
        if not np.isnan(sensitivity_ratio)
        else False,
        "StaticTailTemporalValidityRecentBase": float(base_row["StaticTail_TemporalTail_Recent_Validity"]),
        "StaticTailTemporalValidityDecayBase": float(base_row["StaticTail_TemporalTail_Decay_Validity"]),
        "StaticLTRGain": float(static_ltr_gain),
        "DecayLTRGainForStaticPopPenalty": float(temporal_ltr_gain),
        "LTRGainShrinkage": float(shrinkage) if not np.isnan(shrinkage) else None,
        "MaxRFR": float(rfr_max) if not np.isnan(rfr_max) else None,
        "BaseNicheMainstreamPCESensitivityRatio": float(sensitivity_ratio)
        if not np.isnan(sensitivity_ratio)
        else None,
        "BaseGTSG_PCE": gtsg_pce,
    }
    (args.outdir / "go_no_go_criteria.json").write_text(json.dumps(criteria, indent=2), encoding="utf-8")

    write_markdown_report(
        args.outdir,
        split,
        drift_stats,
        selected_static_lambda,
        selected_temporal_lambda,
        test_summary,
        tod,
        group_df,
        criteria,
    )

    print(f"Done. Report: {args.outdir / 'pilot_report.md'}", flush=True)


if __name__ == "__main__":
    main()
