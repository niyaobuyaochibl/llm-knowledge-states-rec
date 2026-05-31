"""Minimal sequential recommendation utilities for DKE backbone checks."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


class SASRec(nn.Module):
    """Small self-attention sequential recommender.

    Item ids passed to the model are one-based; zero is reserved for padding.
    Scores returned by :meth:`score_all` use the original zero-based item order.
    """

    def __init__(
        self,
        n_items: int,
        max_len: int = 50,
        embedding_dim: int = 64,
        layers: int = 2,
        heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_items = int(n_items)
        self.max_len = int(max_len)
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def encode(self, seqs: torch.Tensor) -> torch.Tensor:
        """Encode padded item sequences and return one vector per user."""
        batch_size, seq_len = seqs.shape
        positions = torch.arange(seq_len, device=seqs.device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_embedding(seqs) + self.position_embedding(positions)
        x = self.dropout(x)
        padding_mask = seqs.eq(0)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=seqs.device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.layer_norm(x)
        lengths = seqs.ne(0).sum(dim=1).clamp(min=1)
        last_pos = lengths - 1
        return x[torch.arange(batch_size, device=seqs.device), last_pos]

    def score_all(self, seqs: torch.Tensor) -> torch.Tensor:
        """Return scores for all zero-based items."""
        rep = self.encode(seqs)
        return rep @ self.item_embedding.weight[1:].T


def build_ordered_user_sequences(frame: pd.DataFrame, n_users: int) -> List[np.ndarray]:
    """Build timestamp-ordered item sequences for each user."""
    ordered = frame.sort_values(["uid", "timestamp", "iid"], kind="mergesort")
    seqs: List[List[int]] = [[] for _ in range(n_users)]
    for uid, iid in ordered[["uid", "iid"]].itertuples(index=False):
        seqs[int(uid)].append(int(iid))
    return [np.asarray(items, dtype=np.int64) for items in seqs]


def left_padded_sequence(items: Sequence[int], max_len: int) -> np.ndarray:
    """Return one-based, left-padded sequence tensor row."""
    row = np.zeros(max_len, dtype=np.int64)
    if len(items) == 0:
        return row
    tail = np.asarray(items[-max_len:], dtype=np.int64) + 1
    row[-len(tail) :] = tail
    return row


def sequence_rows_for_users(sequences: Sequence[np.ndarray], users: Sequence[int], max_len: int) -> np.ndarray:
    """Build a batch of padded sequence rows for evaluation."""
    rows = np.zeros((len(users), max_len), dtype=np.int64)
    for row, uid in enumerate(users):
        rows[row] = left_padded_sequence(sequences[int(uid)], max_len)
    return rows


def sample_training_batch(
    rng: np.random.Generator,
    sequences: Sequence[np.ndarray],
    train_users: np.ndarray,
    user_item_sets: Sequence[set[int]],
    n_items: int,
    batch_size: int,
    max_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample SASRec one-step prediction triples."""
    users = rng.choice(train_users, size=batch_size, replace=True)
    seq_rows = np.zeros((batch_size, max_len), dtype=np.int64)
    pos_items = np.zeros(batch_size, dtype=np.int64)
    neg_items = np.zeros(batch_size, dtype=np.int64)
    for row, uid_np in enumerate(users):
        uid = int(uid_np)
        seq = sequences[uid]
        target_pos = int(rng.integers(1, len(seq)))
        history = seq[:target_pos]
        pos = int(seq[target_pos])
        neg = int(rng.integers(0, n_items))
        seen = user_item_sets[uid]
        while neg in seen:
            neg = int(rng.integers(0, n_items))
        seq_rows[row] = left_padded_sequence(history, max_len)
        pos_items[row] = pos + 1
        neg_items[row] = neg + 1
    return seq_rows, pos_items, neg_items


def train_sasrec(
    model: SASRec,
    sequences: Sequence[np.ndarray],
    n_items: int,
    seed: int,
    epochs: int,
    samples_per_epoch: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> SASRec:
    """Train SASRec with sampled BPR loss."""
    rng = np.random.default_rng(seed)
    train_users = np.asarray([uid for uid, seq in enumerate(sequences) if len(seq) >= 2], dtype=np.int64)
    if len(train_users) == 0:
        raise ValueError("SASRec training requires users with at least two train interactions.")
    user_item_sets = [set(map(int, seq.tolist())) for seq in sequences]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = max(1, int(math.ceil(samples_per_epoch / batch_size)))

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            seq_rows, pos_items, neg_items = sample_training_batch(
                rng, sequences, train_users, user_item_sets, n_items, batch_size, model.max_len
            )
            seq_t = torch.as_tensor(seq_rows, dtype=torch.long, device=device)
            pos_t = torch.as_tensor(pos_items, dtype=torch.long, device=device)
            neg_t = torch.as_tensor(neg_items, dtype=torch.long, device=device)
            rep = model.encode(seq_t)
            pos_score = torch.sum(rep * model.item_embedding(pos_t), dim=1)
            neg_score = torch.sum(rep * model.item_embedding(neg_t), dim=1)
            loss = -F.logsigmoid(pos_score - neg_score).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        print(f"sasrec epoch={epoch}/{epochs} loss={total_loss / steps_per_epoch:.4f}", flush=True)
    return model
