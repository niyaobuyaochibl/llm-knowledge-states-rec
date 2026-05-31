#!/usr/bin/env python3
"""Train a lightweight LightGCN and generate Amazon Books candidate sets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path("/root/temporal_popularity_pilot")
SRC = ROOT / "src"
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPT_DIR))

from temporal_popularity.data import build_exclude_lists, build_user_histories, infer_shape, read_interaction_split, set_seed  # noqa: E402
from temporal_popularity.eval import topk_indices  # noqa: E402
from temporal_popularity.model import LightGCN, build_norm_adj, train_lightgcn  # noqa: E402
from run_llm_selective_invocation_pilot import device_from_arg, ensure_dirs, sample_eval_frame  # noqa: E402
from run_egpr_profile_repair_pilot import build_ordered_histories  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default=ROOT / "data/amazon_books_subset")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results/egpr_profile_repair/amazon_books_seed42_lightgcn_1000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-users", type=int, default=1000)
    parser.add_argument("--top-candidates", type=int, default=100)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--history-limit", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples-per-epoch", type=int, default=200000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--reg", type=float, default=0.0001)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def metrics_from_ranked(ranked: np.ndarray, targets: np.ndarray, topk: int) -> pd.DataFrame:
    rows = []
    for row, target_np in enumerate(targets):
        pos = np.flatnonzero(ranked[row, :topk] == int(target_np))
        hit = len(pos) > 0
        rows.append({
            "NDCG@20": 1.0 / math.log2(int(pos[0]) + 2) if hit else 0.0,
            "Recall@20": float(hit),
            "HitRate@20": float(hit),
        })
    return pd.DataFrame(rows)


def generate_candidates(eval_frame: pd.DataFrame, exclude_lists: Sequence[np.ndarray], user_emb: np.ndarray, item_emb: np.ndarray, topn: int, batch_size: int, split_name: str) -> Dict[str, np.ndarray]:
    users = eval_frame["uid"].to_numpy(np.int64)
    targets = eval_frame["iid"].to_numpy(np.int64)
    candidates = np.zeros((len(users), topn), dtype=np.int64)
    scores = np.zeros((len(users), topn), dtype=np.float32)
    item_emb_t = item_emb.T.astype(np.float32)
    for start in range(0, len(users), batch_size):
        end = min(start + batch_size, len(users))
        batch_users = users[start:end]
        score_batch = user_emb[batch_users].astype(np.float32) @ item_emb_t
        for local, uid_np in enumerate(batch_users):
            uid = int(uid_np)
            row_scores = score_batch[local].copy()
            excluded = exclude_lists[uid]
            if len(excluded):
                row_scores[excluded] = -np.inf
            idx = topk_indices(row_scores, topn)
            chosen = row_scores[idx].astype(np.float32)
            finite = np.isfinite(chosen)
            if not finite.all():
                fill = float(chosen[finite].min() - 1.0) if finite.any() else 0.0
                chosen = np.where(finite, chosen, fill).astype(np.float32)
            candidates[start + local] = idx.astype(np.int64)
            scores[start + local] = chosen
        if end == len(users) or end % (batch_size * 20) == 0:
            print(f"generated {split_name} candidates {end}/{len(users)}", flush=True)
    return {"users": users, "targets": targets, "candidates": candidates, "scores": scores}


def load_metadata(datadir: Path) -> Dict[int, Dict[str, object]]:
    rows: Dict[int, Dict[str, object]] = {}
    with (datadir / "item_metadata.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[int(row["iid"])] = row
    return rows


def write_histories(outdir: Path, users: Sequence[int], histories: Sequence[np.ndarray], metadata: Dict[int, Dict[str, object]], history_limit: int) -> None:
    with (outdir / "user_history.jsonl").open("w", encoding="utf-8") as handle:
        for uid in users:
            items = []
            for iid_np in histories[int(uid)][-history_limit:]:
                iid = int(iid_np)
                meta = metadata.get(iid, {})
                items.append({
                    "iid": iid,
                    "title": meta.get("title", "Unknown Book"),
                    "categories": meta.get("categories", []),
                })
            handle.write(json.dumps({"uid": int(uid), "history": items}, ensure_ascii=False) + "\n")


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{x:.6f}")
        else:
            display[col] = display[col].astype(str)
    lines = ["| " + " | ".join(display.columns) + " |", "| " + " | ".join(["---"] * len(display.columns)) + " |"]
    for row in display.values.tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_dirs(args.outdir)
    set_seed(args.seed)
    train, val, test, _ = read_interaction_split(args.datadir)
    n_users, n_items = infer_shape(train, val, test)
    train_histories = build_user_histories(train, n_users)
    ordered_histories = build_ordered_histories(train, n_users)
    val_eval = sample_eval_frame(val, args.max_users, args.seed + 1)
    test_eval = sample_eval_frame(test, args.max_users, args.seed + 2)
    profile_users = sorted(set(val_eval["uid"].astype(int).tolist() + test_eval["uid"].astype(int).tolist()))
    metadata = load_metadata(args.datadir)
    write_histories(args.outdir, profile_users, ordered_histories, metadata, args.history_limit)

    device = device_from_arg(args.device)
    ckpt = args.outdir / "lightgcn.pt"
    norm_adj = build_norm_adj(train, n_users, n_items, device)
    model = LightGCN(n_users, n_items, args.embedding_dim, args.layers).to(device)
    if ckpt.exists() and not args.force_train:
        print(f"loading existing checkpoint {ckpt}", flush=True)
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        print("training Amazon Books LightGCN", flush=True)
        model = train_lightgcn(model, train, train_histories, norm_adj, n_items, args.seed, args.epochs, args.samples_per_epoch, args.learning_rate, args.reg, device)
        torch.save(model.state_dict(), ckpt)
    model.eval()
    with torch.no_grad():
        user_emb_t, item_emb_t = model.propagate(norm_adj)
    user_emb = user_emb_t.detach().cpu().numpy().astype(np.float32)
    item_emb = item_emb_t.detach().cpu().numpy().astype(np.float32)
    np.save(args.outdir / "user_emb.npy", user_emb)
    np.save(args.outdir / "item_emb.npy", item_emb)

    val_exclude = build_exclude_lists(train_histories, None, n_users)
    test_exclude = build_exclude_lists(train_histories, val, n_users)
    val_batch = generate_candidates(val_eval, val_exclude, user_emb, item_emb, args.top_candidates, args.eval_batch_size, "val")
    test_batch = generate_candidates(test_eval, test_exclude, user_emb, item_emb, args.top_candidates, args.eval_batch_size, "test")
    for split, batch in [("val", val_batch), ("test", test_batch)]:
        np.savez_compressed(args.outdir / f"candidates_lightgcn_{split}.npz", **batch)
    base_ranked = test_batch["candidates"][:, :args.topk]
    base_metrics = metrics_from_ranked(base_ranked, test_batch["targets"], args.topk)
    summary = pd.DataFrame([{"Method": "LightGCN", **{c: float(base_metrics[c].mean()) for c in base_metrics.columns}}])
    summary.to_csv(args.outdir / "base_method_metrics.csv", index=False)
    manifest = {
        "status": "completed",
        "dataset": "amazon_books_subset",
        "seed": args.seed,
        "users": int(n_users),
        "items": int(n_items),
        "train": int(len(train)),
        "val_eval": int(len(val_eval)),
        "test_eval": int(len(test_eval)),
        "profile_users": int(len(profile_users)),
        "top_candidates": args.top_candidates,
        "topk": args.topk,
        "embedding_dim": args.embedding_dim,
        "layers": args.layers,
        "epochs": args.epochs,
        "samples_per_epoch": args.samples_per_epoch,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Amazon Books LightGCN Candidate Preparation",
        "",
        f"Dataset: Amazon Books subset. Users: {n_users}. Items: {n_items}. Train interactions: {len(train)}.",
        f"Validation/test sampled users: {len(val_eval)} / {len(test_eval)}. Candidate set: top-{args.top_candidates}.",
        "",
        "## Base Performance",
        "",
        markdown_table(summary),
        "",
        "## Artifacts",
        "",
        "- `lightgcn.pt`",
        "- `user_emb.npy`",
        "- `item_emb.npy`",
        "- `candidates_lightgcn_val.npz`",
        "- `candidates_lightgcn_test.npz`",
        "- `user_history.jsonl`",
        "- `base_method_metrics.csv`",
    ]
    (args.outdir / "amazon_books_lightgcn_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done. Report: {args.outdir / 'amazon_books_lightgcn_report.md'}", flush=True)


if __name__ == "__main__":
    main()
