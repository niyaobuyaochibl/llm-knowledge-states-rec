"""LightGCN model and BPR training utilities."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    """Minimal LightGCN backbone for implicit recommendation."""

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


def build_norm_adj(train: pd.DataFrame, n_users: int, n_items: int, device: torch.device) -> torch.Tensor:
    """Build LightGCN symmetric normalized adjacency from train interactions."""
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


def sample_negatives(
    rng: np.random.Generator,
    users: np.ndarray,
    user_pos_sets: Sequence[set],
    n_items: int,
) -> np.ndarray:
    """Sample one negative item per user."""
    neg = rng.integers(0, n_items, size=len(users), dtype=np.int64)
    unresolved = np.array([item in user_pos_sets[int(user)] for user, item in zip(users, neg)], dtype=bool)
    while unresolved.any():
        idx = np.flatnonzero(unresolved)
        neg[idx] = rng.integers(0, n_items, size=len(idx), dtype=np.int64)
        unresolved[idx] = np.array(
            [item in user_pos_sets[int(user)] for user, item in zip(users[idx], neg[idx])],
            dtype=bool,
        )
    return neg


def train_lightgcn(
    model: LightGCN,
    train: pd.DataFrame,
    train_histories: Sequence[np.ndarray],
    norm_adj: torch.Tensor,
    n_items: int,
    seed: int,
    epochs: int,
    samples_per_epoch: int,
    lr: float,
    reg: float,
    device: torch.device,
) -> LightGCN:
    """Train LightGCN with sampled BPR loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    train_pairs = train[["uid", "iid"]].to_numpy(np.int64)
    user_pos_sets = [set(items.tolist()) for items in train_histories]
    n_samples = min(samples_per_epoch, len(train_pairs))

    for epoch in range(1, epochs + 1):
        sample_idx = rng.integers(0, len(train_pairs), size=n_samples, dtype=np.int64)
        users_np = train_pairs[sample_idx, 0]
        pos_np = train_pairs[sample_idx, 1]
        neg_np = sample_negatives(rng, users_np, user_pos_sets, n_items)
        users = torch.from_numpy(users_np).long().to(device)
        pos = torch.from_numpy(pos_np).long().to(device)
        neg = torch.from_numpy(neg_np).long().to(device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        user_emb, item_emb = model.propagate(norm_adj)
        u = user_emb[users]
        p = item_emb[pos]
        n = item_emb[neg]
        bpr_loss = -F.logsigmoid((u * p).sum(dim=1) - (u * n).sum(dim=1)).mean()
        reg_loss = (u.pow(2).sum(dim=1) + p.pow(2).sum(dim=1) + n.pow(2).sum(dim=1)).mean()
        loss = bpr_loss + reg * reg_loss
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch={epoch:03d} loss={loss.item():.5f} bpr={bpr_loss.item():.5f}", flush=True)
    return model
